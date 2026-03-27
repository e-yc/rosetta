#!/usr/bin/env python3
"""
The Bridge Experiment

Train a linear projection (nn.Linear 4096->4096) end-to-end through frozen Llama
layers against next-token prediction loss. Initialize from Procrustes (S2).

Data: GLM layer 5 activations from the Phase 2 memmap, reconstructed as
per-input sequences at aligned positions. Targets: Llama token IDs at
the corresponding aligned positions.

Trains Procrustes-init and random-init baselines sequentially.
"""

import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

GRAFT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECTIONS_DIR = os.path.join(GRAFT_DIR, "projections")
BRIDGE_DIR = os.path.join(GRAFT_DIR, "bridge")
os.makedirs(BRIDGE_DIR, exist_ok=True)

# Splice point: S2 (GLM layer 5 -> Llama layer 15)
GLM_LAYER = 5
LLAMA_SPLICE_LAYER = 15

# Training config
BATCH_SIZE = 4
LR = 1e-4
LR_MIN = 1e-6
EPOCHS = 3
LOG_INTERVAL = 100
EVAL_INTERVAL = 500
MAX_SEQ_LEN = 128  # pad/truncate aligned sequences to this


class AlignedSequenceDataset(Dataset):
    """
    Reconstructs per-input sequences of GLM layer 5 activations at aligned positions.
    Returns (glm_hidden_states, position_ids, attention_mask, target_token_ids).
    """

    def __init__(self, corpus, index, glm_store, llama_tokenizer, glm_layer, max_seq_len):
        self.max_seq_len = max_seq_len
        self.samples = []

        for entry in corpus:
            input_id = entry["input_id"]
            text = entry["raw_text"]
            positions_a = entry[config.CORPUS_POSITIONS_KEY_A]  # GLM positions
            positions_b = entry[config.CORPUS_POSITIONS_KEY_B]  # Llama positions

            # Get Llama token IDs for targets
            llama_enc = llama_tokenizer(text, add_special_tokens=False)
            llama_token_ids = llama_enc["input_ids"]

            # Gather aligned rows from memmap
            rows = []
            llama_positions = []
            targets = []
            for pair_idx, (pos_a, pos_b) in enumerate(zip(positions_a, positions_b)):
                key = f"{input_id}_{pair_idx}"
                if key not in index:
                    continue
                if pos_b >= len(llama_token_ids):
                    continue
                rows.append(index[key])
                llama_positions.append(pos_b)
                targets.append(llama_token_ids[pos_b])

            if len(rows) < 4:  # skip very short sequences
                continue

            self.samples.append({
                "rows": np.array(rows),
                "positions": np.array(llama_positions),
                "targets": np.array(targets),
            })

        self.glm_store = glm_store
        self.glm_layer = glm_layer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        seq_len = min(len(s["rows"]), self.max_seq_len)

        # Load GLM activations from memmap
        glm_acts = self.glm_store[s["rows"][:seq_len], self.glm_layer, :].astype(np.float32)

        # Pad to max_seq_len
        padded_acts = np.zeros((self.max_seq_len, glm_acts.shape[1]), dtype=np.float32)
        padded_acts[:seq_len] = glm_acts

        positions = np.zeros(self.max_seq_len, dtype=np.int64)
        positions[:seq_len] = s["positions"][:seq_len]

        targets = np.full(self.max_seq_len, -100, dtype=np.int64)  # -100 = ignore in CE
        targets[:seq_len] = s["targets"][:seq_len]

        # Attention mask: 1 for real tokens, 0 for padding
        attn_mask = np.zeros(self.max_seq_len, dtype=np.float32)
        attn_mask[:seq_len] = 1.0

        return {
            "hidden_states": torch.from_numpy(padded_acts),
            "position_ids": torch.from_numpy(positions),
            "attention_mask": torch.from_numpy(attn_mask),
            "targets": torch.from_numpy(targets),
            "seq_len": seq_len,
        }


def load_llama():
    token = os.environ.get("HF_TOKEN") or True
    for mid in [config.MODEL_B_ID] + config.MODEL_B_FALLBACKS:
        try:
            print(f"  Loading from {mid} ...", flush=True)
            cfg = AutoConfig.from_pretrained(mid, trust_remote_code=True, token=token)
            if not hasattr(cfg, "max_length"):
                cfg.max_length = getattr(cfg, "seq_length", 8192)
            model = AutoModelForCausalLM.from_pretrained(
                mid, config=cfg, torch_dtype=torch.float16, device_map="cuda",
                trust_remote_code=True, token=token,
            )
            tokenizer = AutoTokenizer.from_pretrained(mid, trust_remote_code=True, token=token)
            return model, tokenizer
        except Exception as e:
            print(f"  FAILED ({str(e)[:80]})")
    sys.exit(1)


def build_partial_forward(llama_model, splice_layer):
    """
    Returns a function that takes projected hidden states and runs them
    through Llama layers splice_layer..end, returning logits.
    """
    llama = llama_model.model
    lm_head = llama_model.lm_head

    def forward_fn(hidden, attention_mask, position_ids):
        batch_size, seq_len, _ = hidden.shape
        device = hidden.device

        cache_position = torch.arange(seq_len, device=device)
        causal_mask = llama._update_causal_mask(
            attention_mask, hidden, cache_position,
            past_key_values=None, output_attentions=False
        )

        position_embeddings = None
        if hasattr(llama, "rotary_emb"):
            position_embeddings = llama.rotary_emb(hidden, position_ids)

        for layer in llama.layers[splice_layer:]:
            kwargs = {"attention_mask": causal_mask, "position_ids": position_ids}
            if position_embeddings is not None:
                kwargs["position_embeddings"] = position_embeddings
            layer_out = layer(hidden, **kwargs)
            hidden = layer_out[0]

        hidden = llama.norm(hidden)
        logits = lm_head(hidden)
        return logits

    return forward_fn


@torch.no_grad()
def evaluate(projection, forward_fn, dataloader, device, max_batches=None):
    """Compute validation loss, KL vs native (if available), top-1, top-5."""
    projection.eval()
    total_loss = 0.0
    total_tokens = 0
    top1_correct = 0
    top5_correct = 0

    for i, batch in enumerate(dataloader):
        if max_batches and i >= max_batches:
            break

        hs = batch["hidden_states"].to(device)
        pos = batch["position_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        targets = batch["targets"].to(device)

        projected = projection(hs.float()).half()
        logits = forward_fn(projected, mask, pos)

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_targets = targets[:, 1:].contiguous()

        # Mask out padding
        valid = shift_targets != -100
        if valid.sum() == 0:
            continue

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_targets.view(-1),
            ignore_index=-100,
            reduction="sum"
        )

        # Top-1 and top-5
        valid_logits = shift_logits[valid]
        valid_targets = shift_targets[valid]
        top1_correct += (valid_logits.argmax(dim=-1) == valid_targets).sum().item()
        top5 = valid_logits.topk(5, dim=-1).indices
        top5_correct += (top5 == valid_targets.unsqueeze(-1)).any(dim=-1).sum().item()

        n = valid.sum().item()
        total_loss += loss.item()
        total_tokens += n

    projection.train()
    if total_tokens == 0:
        return {"loss": float("inf"), "top1": 0.0, "top5": 0.0, "tokens": 0}

    return {
        "loss": total_loss / total_tokens,
        "perplexity": math.exp(min(total_loss / total_tokens, 20)),
        "top1": top1_correct / total_tokens,
        "top5": top5_correct / total_tokens,
        "tokens": total_tokens,
    }


def train_projection(name, projection, forward_fn, train_loader, val_loader,
                     device, epochs, lr, lr_min):
    """Train a projection and log the learning curve."""
    print(f"\n{'='*70}")
    print(f"  TRAINING: {name}")
    print(f"{'='*70}")

    optimizer = torch.optim.AdamW(projection.parameters(), lr=lr, weight_decay=0.01)
    total_steps = epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=lr_min
    )

    log = []
    global_step = 0
    best_val_loss = float("inf")

    # Pre-training eval
    val_metrics = evaluate(projection, forward_fn, val_loader, device, max_batches=50)
    print(f"  Pre-train val: loss={val_metrics['loss']:.4f}, ppl={val_metrics['perplexity']:.1f}, "
          f"top1={val_metrics['top1']:.4f}, top5={val_metrics['top5']:.4f}", flush=True)
    log.append({"step": 0, "phase": "val", **val_metrics})

    for epoch in range(epochs):
        print(f"\n  Epoch {epoch + 1}/{epochs}", flush=True)
        epoch_loss = 0.0
        epoch_tokens = 0
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            hs = batch["hidden_states"].to(device)
            pos = batch["position_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            targets = batch["targets"].to(device)

            # Forward: projection (trainable) -> frozen Llama layers
            projected = projection(hs.float()).half()
            logits = forward_fn(projected, mask, pos)

            # Next-token loss
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = targets[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_targets.view(-1),
                ignore_index=-100,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(projection.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            n_tokens = (shift_targets != -100).sum().item()
            epoch_loss += loss.item() * n_tokens
            epoch_tokens += n_tokens
            global_step += 1

            if global_step % LOG_INTERVAL == 0:
                avg_loss = epoch_loss / max(epoch_tokens, 1)
                lr_now = scheduler.get_last_lr()[0]
                print(f"    step {global_step}: train_loss={avg_loss:.4f}, "
                      f"lr={lr_now:.2e}", flush=True)
                log.append({"step": global_step, "phase": "train",
                            "loss": avg_loss, "lr": lr_now})

            if global_step % EVAL_INTERVAL == 0:
                val_metrics = evaluate(projection, forward_fn, val_loader, device, max_batches=50)
                print(f"    step {global_step} val: loss={val_metrics['loss']:.4f}, "
                      f"ppl={val_metrics['perplexity']:.1f}, "
                      f"top1={val_metrics['top1']:.4f}, top5={val_metrics['top5']:.4f}", flush=True)
                log.append({"step": global_step, "phase": "val", **val_metrics})

                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    torch.save(projection.state_dict(),
                               os.path.join(BRIDGE_DIR, f"{name}_best.pt"))

        # End of epoch eval
        elapsed = time.time() - t0
        val_metrics = evaluate(projection, forward_fn, val_loader, device, max_batches=50)
        print(f"  Epoch {epoch+1} done in {elapsed/60:.1f} min. "
              f"Val: loss={val_metrics['loss']:.4f}, ppl={val_metrics['perplexity']:.1f}, "
              f"top1={val_metrics['top1']:.4f}, top5={val_metrics['top5']:.4f}", flush=True)
        log.append({"step": global_step, "phase": "epoch_end", "epoch": epoch + 1, **val_metrics})

        torch.save(projection.state_dict(),
                   os.path.join(BRIDGE_DIR, f"{name}_epoch{epoch+1}.pt"))

    # Final position-dependent eval
    print(f"\n  Position-dependent KL analysis...", flush=True)
    pos_kl = evaluate_by_position(projection, forward_fn, val_loader, device)
    log.append({"step": global_step, "phase": "position_kl", **pos_kl})
    print(f"    KL by position: {pos_kl}", flush=True)

    return log


@torch.no_grad()
def evaluate_by_position(projection, forward_fn, dataloader, device, max_batches=50):
    """Compute loss bucketed by sequence position."""
    projection.eval()
    buckets = {"0-16": [], "16-32": [], "32-64": [], "64-128": []}

    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        hs = batch["hidden_states"].to(device)
        pos = batch["position_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        targets = batch["targets"].to(device)

        projected = projection(hs.float()).half()
        logits = forward_fn(projected, mask, pos)

        # Per-position loss
        shift_logits = logits[:, :-1, :]
        shift_targets = targets[:, 1:]

        for b in range(shift_logits.shape[0]):
            for t in range(shift_logits.shape[1]):
                if shift_targets[b, t] == -100:
                    continue
                token_loss = F.cross_entropy(
                    shift_logits[b, t].unsqueeze(0),
                    shift_targets[b, t].unsqueeze(0)
                ).item()
                p = t  # position in sequence
                if p < 16:
                    buckets["0-16"].append(token_loss)
                elif p < 32:
                    buckets["16-32"].append(token_loss)
                elif p < 64:
                    buckets["32-64"].append(token_loss)
                else:
                    buckets["64-128"].append(token_loss)

    projection.train()
    return {k: float(np.mean(v)) if v else None for k, v in buckets.items()}


def main():
    print("=" * 70)
    print("THE BRIDGE EXPERIMENT")
    print("=" * 70)

    # Load corpus and index
    print("\nLoading corpus...", flush=True)
    corpus = []
    with open(config.CORPUS_PATH) as f:
        for line in f:
            corpus.append(json.loads(line))
    with open(config.ACTIVATION_INDEX_PATH) as f:
        index_meta = json.load(f)
    index = index_meta["index"]
    total_rows = index_meta["total_rows"]
    hidden_dim = config.HIDDEN_DIM

    # Load GLM memmap
    pass1_meta_path = os.path.join(config.DATA_DIR, "pass1_metadata.json")
    with open(pass1_meta_path) as f:
        pass1_meta = json.load(f)
    num_layers_glm = pass1_meta["stored_layers"]
    glm_store = np.memmap(
        config.MODEL_A_MMAP_PATH, dtype=np.float16, mode="r",
        shape=(total_rows, num_layers_glm, hidden_dim),
    )

    # Load Llama
    print("\nLoading Llama...", flush=True)
    llama_model, llama_tokenizer = load_llama()

    # Freeze Llama
    for param in llama_model.parameters():
        param.requires_grad = False
    print(f"  Llama frozen ({sum(p.numel() for p in llama_model.parameters())/1e9:.1f}B params, all frozen)")

    # Build datasets
    print("\nBuilding datasets...", flush=True)
    n_train = int(len(corpus) * 0.9)
    train_corpus = corpus[:n_train]
    val_corpus = corpus[n_train:]

    train_dataset = AlignedSequenceDataset(
        train_corpus, index, glm_store, llama_tokenizer, GLM_LAYER, MAX_SEQ_LEN
    )
    val_dataset = AlignedSequenceDataset(
        val_corpus, index, glm_store, llama_tokenizer, GLM_LAYER, MAX_SEQ_LEN
    )
    print(f"  Train: {len(train_dataset)} sequences")
    print(f"  Val: {len(val_dataset)} sequences")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)

    # Build partial forward function
    forward_fn = build_partial_forward(llama_model, LLAMA_SPLICE_LAYER)

    # Verify gradients flow correctly
    print("\nVerifying gradient flow...", flush=True)
    test_proj = nn.Linear(hidden_dim, hidden_dim, bias=True).cuda()
    test_batch = next(iter(train_loader))
    hs = test_batch["hidden_states"].cuda()
    pos = test_batch["position_ids"].cuda()
    mask = test_batch["attention_mask"].cuda()
    tgt = test_batch["targets"].cuda()

    out = forward_fn(test_proj(hs.float()).half(), mask, pos)
    loss = F.cross_entropy(out[:, :-1].reshape(-1, out.size(-1)), tgt[:, 1:].reshape(-1), ignore_index=-100)
    loss.backward()

    assert test_proj.weight.grad is not None, "No gradient on projection weight!"
    assert test_proj.weight.grad.abs().sum() > 0, "Zero gradient on projection!"
    llama_moved = any(p.grad is not None for p in llama_model.parameters())
    assert not llama_moved, "Llama parameters have gradients — they should be frozen!"
    print(f"  Gradient flow verified. Projection grad norm: {test_proj.weight.grad.norm():.4f}")
    del test_proj
    torch.cuda.empty_cache()

    # --- Training run 1: Procrustes-initialized ---
    print("\n  Initializing from Procrustes (S2)...", flush=True)
    proj_procrustes = nn.Linear(hidden_dim, hidden_dim, bias=True).cuda()
    sp_dir = os.path.join(PROJECTIONS_DIR, "S2")
    W_proc = np.load(os.path.join(sp_dir, "P4_W.npy"))
    mean_glm = np.load(os.path.join(sp_dir, "P4_mean_glm.npy"))
    mean_llama = np.load(os.path.join(sp_dir, "P4_mean_llama.npy"))
    # Affine: y = (x - mean_glm) @ W + mean_llama = x @ W + (mean_llama - mean_glm @ W)
    bias_init = mean_llama - mean_glm @ W_proc
    proj_procrustes.weight.data = torch.from_numpy(W_proc.T.copy()).float().cuda()
    proj_procrustes.bias.data = torch.from_numpy(bias_init.copy()).float().cuda()

    log_procrustes = train_projection(
        "procrustes_init", proj_procrustes, forward_fn,
        train_loader, val_loader, "cuda", EPOCHS, LR, LR_MIN
    )

    # Save learning curve
    with open(os.path.join(BRIDGE_DIR, "log_procrustes.json"), "w") as f:
        json.dump(log_procrustes, f, indent=2)

    # --- Training run 2: Random orthogonal initialized ---
    print("\n  Initializing from random orthogonal...", flush=True)
    proj_random = nn.Linear(hidden_dim, hidden_dim, bias=True).cuda()
    rng = np.random.RandomState(123)
    Q, _ = np.linalg.qr(rng.randn(hidden_dim, hidden_dim))
    proj_random.weight.data = torch.from_numpy(Q.T.copy()).float().cuda()
    nn.init.zeros_(proj_random.bias)

    log_random = train_projection(
        "random_init", proj_random, forward_fn,
        train_loader, val_loader, "cuda", EPOCHS, LR, LR_MIN
    )

    with open(os.path.join(BRIDGE_DIR, "log_random.json"), "w") as f:
        json.dump(log_random, f, indent=2)

    # --- Baselines (no training) ---
    print("\n--- BASELINES ---")

    # Native Llama baseline
    print("  Computing native Llama baseline...", flush=True)
    native_metrics = evaluate_native_llama(llama_model, llama_tokenizer, val_corpus)
    print(f"  Native Llama: loss={native_metrics['loss']:.4f}, "
          f"ppl={native_metrics['perplexity']:.1f}, "
          f"top1={native_metrics['top1']:.4f}")

    # Identity baseline (P0)
    proj_identity = nn.Linear(hidden_dim, hidden_dim, bias=True).cuda()
    nn.init.eye_(proj_identity.weight)
    nn.init.zeros_(proj_identity.bias)
    identity_metrics = evaluate(proj_identity, forward_fn, val_loader, "cuda", max_batches=50)
    print(f"  Identity (P0): loss={identity_metrics['loss']:.4f}, "
          f"ppl={identity_metrics['perplexity']:.1f}, "
          f"top1={identity_metrics['top1']:.4f}")

    # Untrained Procrustes baseline (P4)
    proj_p4 = nn.Linear(hidden_dim, hidden_dim, bias=True).cuda()
    proj_p4.weight.data = torch.from_numpy(W_proc.T.copy()).float().cuda()
    proj_p4.bias.data = torch.from_numpy(bias_init.copy()).float().cuda()
    p4_metrics = evaluate(proj_p4, forward_fn, val_loader, "cuda", max_batches=50)
    print(f"  Procrustes (P4, untrained): loss={p4_metrics['loss']:.4f}, "
          f"ppl={p4_metrics['perplexity']:.1f}, "
          f"top1={p4_metrics['top1']:.4f}")

    # --- Summary ---
    summary = {
        "native_llama": native_metrics,
        "identity_baseline": identity_metrics,
        "procrustes_untrained": p4_metrics,
        "procrustes_trained_final": log_procrustes[-2] if len(log_procrustes) > 1 else {},
        "random_trained_final": log_random[-2] if len(log_random) > 1 else {},
    }
    with open(os.path.join(BRIDGE_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  BRIDGE EXPERIMENT COMPLETE")
    print(f"  Results saved to {BRIDGE_DIR}/")
    print(f"{'='*70}")

    del llama_model, glm_store
    torch.cuda.empty_cache()


@torch.no_grad()
def evaluate_native_llama(model, tokenizer, val_corpus, max_inputs=200):
    """Run native Llama on validation corpus, compute CE loss."""
    total_loss = 0.0
    total_tokens = 0
    top1_correct = 0

    for entry in val_corpus[:max_inputs]:
        text = entry["raw_text"]
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False,
                           max_length=MAX_SEQ_LEN, truncation=True)
        input_ids = inputs["input_ids"].to("cuda")

        out = model(input_ids=input_ids)
        logits = out.logits

        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]

        loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction="sum"
        )

        n = shift_labels.numel()
        total_loss += loss.item()
        total_tokens += n
        top1_correct += (shift_logits.argmax(dim=-1) == shift_labels).sum().item()

    return {
        "loss": total_loss / max(total_tokens, 1),
        "perplexity": math.exp(min(total_loss / max(total_tokens, 1), 20)),
        "top1": top1_correct / max(total_tokens, 1),
        "top5": 0.0,  # skip for native
        "tokens": total_tokens,
    }


if __name__ == "__main__":
    main()
