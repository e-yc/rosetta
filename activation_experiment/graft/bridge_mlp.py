#!/usr/bin/env python3
"""
MLP Ablation for the Bridge Experiment

Same setup as bridge.py but replaces the linear projection with:
  1. Two-layer MLP: 4096 -> 4096 -> 4096 with GELU (16.8M + 16.8M = 33.6M params)
  2. Bottleneck MLP: 4096 -> 1024 -> 4096 with GELU (4.2M + 4.2M = 8.4M params)

Compares against the linear bridge result to determine whether the
loss=2.58 ceiling is a limitation of linear transforms or an
informational barrier at the splice point.

Initializes the first layer from Procrustes, second layer from identity.
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
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Reuse dataset and helpers from bridge.py
from bridge import (
    AlignedSequenceDataset, load_llama, build_partial_forward,
    evaluate, evaluate_by_position, evaluate_native_llama,
    GLM_LAYER, LLAMA_SPLICE_LAYER, BATCH_SIZE, MAX_SEQ_LEN,
    LOG_INTERVAL, EVAL_INTERVAL, GRAFT_DIR, PROJECTIONS_DIR,
)

BRIDGE_DIR = os.path.join(GRAFT_DIR, "bridge_mlp")
os.makedirs(BRIDGE_DIR, exist_ok=True)

LR = 1e-4
LR_MIN = 1e-6
EPOCHS = 3


def make_mlp_2layer(hidden_dim, init_from_procrustes=True):
    """4096 -> 4096 -> 4096 with GELU. ~33.6M params."""
    mlp = nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim, bias=True),
        nn.GELU(),
        nn.Linear(hidden_dim, hidden_dim, bias=True),
    )

    if init_from_procrustes:
        sp_dir = os.path.join(PROJECTIONS_DIR, "S2")
        W_proc = np.load(os.path.join(sp_dir, "P4_W.npy"))
        mean_glm = np.load(os.path.join(sp_dir, "P4_mean_glm.npy"))
        mean_llama = np.load(os.path.join(sp_dir, "P4_mean_llama.npy"))

        # First layer: Procrustes rotation + bias
        bias_init = -mean_glm @ W_proc  # shift input before rotation
        mlp[0].weight.data = torch.from_numpy(W_proc.T.copy()).float()
        mlp[0].bias.data = torch.from_numpy(bias_init.copy()).float()

        # Second layer: identity + mean_llama bias (so full transform ≈ Procrustes at init)
        nn.init.eye_(mlp[2].weight)
        mlp[2].bias.data = torch.from_numpy(mean_llama.copy()).float()
    else:
        # Xavier init
        nn.init.xavier_uniform_(mlp[0].weight)
        nn.init.zeros_(mlp[0].bias)
        nn.init.xavier_uniform_(mlp[2].weight)
        nn.init.zeros_(mlp[2].bias)

    return mlp


def make_mlp_bottleneck(hidden_dim, bottleneck_dim=1024, init_from_procrustes=False):
    """4096 -> 1024 -> 4096 with GELU. ~8.4M params."""
    mlp = nn.Sequential(
        nn.Linear(hidden_dim, bottleneck_dim, bias=True),
        nn.GELU(),
        nn.Linear(bottleneck_dim, hidden_dim, bias=True),
    )
    nn.init.xavier_uniform_(mlp[0].weight)
    nn.init.zeros_(mlp[0].bias)
    nn.init.xavier_uniform_(mlp[2].weight)
    nn.init.zeros_(mlp[2].bias)
    return mlp


def train_model(name, model, forward_fn, train_loader, val_loader,
                device, epochs, lr, lr_min):
    """Train and log learning curve. Same as bridge.py but for any nn.Module."""
    print(f"\n{'='*70}")
    print(f"  TRAINING: {name}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_params:,}")
    print(f"{'='*70}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=lr_min
    )

    log = []
    global_step = 0
    best_val_loss = float("inf")

    # Pre-training eval
    val_metrics = evaluate(model, forward_fn, val_loader, device, max_batches=50)
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

            projected = model(hs.float()).half()
            logits = forward_fn(projected, mask, pos)

            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = targets[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_targets.view(-1),
                ignore_index=-100,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
                val_metrics = evaluate(model, forward_fn, val_loader, device, max_batches=50)
                print(f"    step {global_step} val: loss={val_metrics['loss']:.4f}, "
                      f"ppl={val_metrics['perplexity']:.1f}, "
                      f"top1={val_metrics['top1']:.4f}, top5={val_metrics['top5']:.4f}", flush=True)
                log.append({"step": global_step, "phase": "val", **val_metrics})

                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    torch.save(model.state_dict(),
                               os.path.join(BRIDGE_DIR, f"{name}_best.pt"))

        elapsed = time.time() - t0
        val_metrics = evaluate(model, forward_fn, val_loader, device, max_batches=50)
        print(f"  Epoch {epoch+1} done in {elapsed/60:.1f} min. "
              f"Val: loss={val_metrics['loss']:.4f}, ppl={val_metrics['perplexity']:.1f}, "
              f"top1={val_metrics['top1']:.4f}, top5={val_metrics['top5']:.4f}", flush=True)
        log.append({"step": global_step, "phase": "epoch_end", "epoch": epoch + 1, **val_metrics})

        torch.save(model.state_dict(),
                   os.path.join(BRIDGE_DIR, f"{name}_epoch{epoch+1}.pt"))

    # Position-dependent eval
    print(f"\n  Position-dependent analysis...", flush=True)
    pos_kl = evaluate_by_position(model, forward_fn, val_loader, device)
    log.append({"step": global_step, "phase": "position_kl", **pos_kl})
    print(f"    KL by position: {pos_kl}", flush=True)

    return log


def main():
    print("=" * 70)
    print("MLP ABLATION — BRIDGE EXPERIMENT")
    print("=" * 70)

    hidden_dim = config.HIDDEN_DIM

    # Load corpus and memmap
    print("\nLoading corpus...", flush=True)
    corpus = []
    with open(config.CORPUS_PATH) as f:
        for line in f:
            corpus.append(json.loads(line))
    with open(config.ACTIVATION_INDEX_PATH) as f:
        index_meta = json.load(f)
    index = index_meta["index"]
    total_rows = index_meta["total_rows"]

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
    for param in llama_model.parameters():
        param.requires_grad = False

    # Build datasets
    print("\nBuilding datasets...", flush=True)
    n_train = int(len(corpus) * 0.9)
    train_dataset = AlignedSequenceDataset(
        corpus[:n_train], index, glm_store, llama_tokenizer, GLM_LAYER, MAX_SEQ_LEN
    )
    val_dataset = AlignedSequenceDataset(
        corpus[n_train:], index, glm_store, llama_tokenizer, GLM_LAYER, MAX_SEQ_LEN
    )
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)

    forward_fn = build_partial_forward(llama_model, LLAMA_SPLICE_LAYER)

    # --- Run 1: Two-layer MLP, Procrustes-init ---
    print("\n  Building 2-layer MLP (Procrustes-init)...", flush=True)
    mlp_2layer = make_mlp_2layer(hidden_dim, init_from_procrustes=True).cuda()
    log_mlp2 = train_model(
        "mlp_2layer_procrustes", mlp_2layer, forward_fn,
        train_loader, val_loader, "cuda", EPOCHS, LR, LR_MIN
    )
    with open(os.path.join(BRIDGE_DIR, "log_mlp2_procrustes.json"), "w") as f:
        json.dump(log_mlp2, f, indent=2)
    del mlp_2layer
    torch.cuda.empty_cache()

    # --- Run 2: Two-layer MLP, random init ---
    print("\n  Building 2-layer MLP (random-init)...", flush=True)
    mlp_2layer_rand = make_mlp_2layer(hidden_dim, init_from_procrustes=False).cuda()
    log_mlp2_rand = train_model(
        "mlp_2layer_random", mlp_2layer_rand, forward_fn,
        train_loader, val_loader, "cuda", EPOCHS, LR, LR_MIN
    )
    with open(os.path.join(BRIDGE_DIR, "log_mlp2_random.json"), "w") as f:
        json.dump(log_mlp2_rand, f, indent=2)
    del mlp_2layer_rand
    torch.cuda.empty_cache()

    # --- Run 3: Bottleneck MLP (4096->1024->4096) ---
    print("\n  Building bottleneck MLP (4096->1024->4096)...", flush=True)
    mlp_bottle = make_mlp_bottleneck(hidden_dim, bottleneck_dim=1024).cuda()
    log_bottle = train_model(
        "mlp_bottleneck_1024", mlp_bottle, forward_fn,
        train_loader, val_loader, "cuda", EPOCHS, LR, LR_MIN
    )
    with open(os.path.join(BRIDGE_DIR, "log_bottleneck.json"), "w") as f:
        json.dump(log_bottle, f, indent=2)
    del mlp_bottle
    torch.cuda.empty_cache()

    # --- Summary ---
    # Load linear bridge results for comparison
    linear_log_path = os.path.join(GRAFT_DIR, "bridge", "log_procrustes.json")
    linear_final = {}
    if os.path.exists(linear_log_path):
        with open(linear_log_path) as f:
            linear_log = json.load(f)
        for entry in reversed(linear_log):
            if entry.get("phase") == "epoch_end":
                linear_final = entry
                break

    summary = {
        "linear_bridge (from bridge.py)": linear_final,
        "mlp_2layer_procrustes": log_mlp2[-2] if len(log_mlp2) > 1 else {},
        "mlp_2layer_random": log_mlp2_rand[-2] if len(log_mlp2_rand) > 1 else {},
        "mlp_bottleneck_1024": log_bottle[-2] if len(log_bottle) > 1 else {},
    }

    with open(os.path.join(BRIDGE_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  MLP ABLATION COMPLETE")
    print(f"{'='*70}")
    print(f"\n  Summary:")
    for name, metrics in summary.items():
        loss = metrics.get("loss", "?")
        ppl = metrics.get("perplexity", "?")
        top1 = metrics.get("top1", "?")
        top5 = metrics.get("top5", "?")
        print(f"    {name:<35} loss={loss}  ppl={ppl}  top1={top1}  top5={top5}")
    print(f"\n  Results saved to {BRIDGE_DIR}/")

    del llama_model, glm_store
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
