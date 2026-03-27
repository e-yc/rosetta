#!/usr/bin/env python3
"""
Category Attribution Experiment

Evaluates the trained bridge, untrained Procrustes, trained MLP, and native Llama
on a per-category held-out eval corpus. Determines whether Phase 2 geometric
analysis (rank profiles) predicts where the bridge succeeds and fails.
"""

import json
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from bridge import (
    AlignedSequenceDataset, load_llama, build_partial_forward,
    GLM_LAYER, LLAMA_SPLICE_LAYER, MAX_SEQ_LEN,
    GRAFT_DIR, PROJECTIONS_DIR,
)

RESULTS_DIR = os.path.join(GRAFT_DIR, "category_attribution")
os.makedirs(RESULTS_DIR, exist_ok=True)

EVAL_PER_CATEGORY = 100
HIDDEN_DIM = config.HIDDEN_DIM


def load_glm():
    token = os.environ.get("HF_TOKEN") or True
    for mid in [config.MODEL_A_ID] + config.MODEL_A_FALLBACKS:
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


def build_eval_corpus():
    """Stratified holdout: last EVAL_PER_CATEGORY inputs from each category."""
    print("\nBuilding evaluation corpus (stratified)...", flush=True)
    corpus = []
    with open(config.CORPUS_PATH) as f:
        for line in f:
            corpus.append(json.loads(line))

    # Group all inputs by category
    all_by_cat = defaultdict(list)
    for entry in corpus:
        all_by_cat[entry["category"]].append(entry)

    # Take the last EVAL_PER_CATEGORY from each category as holdout
    by_category = {}
    for cat, entries in sorted(all_by_cat.items()):
        holdout = entries[-EVAL_PER_CATEGORY:]
        by_category[cat] = holdout
        print(f"  {cat}: {len(holdout)} eval inputs (from ids {holdout[0]['input_id']}-{holdout[-1]['input_id']})")

    total = sum(len(v) for v in by_category.values())
    print(f"  Total: {total} inputs")

    return by_category


def evaluate_native_llama_by_category(model, tokenizer, eval_by_cat):
    """Run native Llama on eval corpus, collect per-category metrics."""
    print("\n--- NATIVE LLAMA ---", flush=True)
    results = {}

    for cat, entries in sorted(eval_by_cat.items()):
        total_loss = 0.0
        total_tokens = 0
        top1_correct = 0
        top5_correct = 0

        for entry in entries:
            text = entry["raw_text"]
            inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False,
                               max_length=MAX_SEQ_LEN, truncation=True)
            input_ids = inputs["input_ids"].to("cuda")

            with torch.no_grad():
                out = model(input_ids=input_ids)
            logits = out.logits

            shift_logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]
            n = shift_labels.numel()

            loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1), reduction="sum"
            ).item()

            top1_correct += (shift_logits.argmax(dim=-1) == shift_labels).sum().item()
            top5 = shift_logits.topk(5, dim=-1).indices
            top5_correct += (top5 == shift_labels.unsqueeze(-1)).any(dim=-1).sum().item()

            total_loss += loss
            total_tokens += n

        results[cat] = {
            "loss": total_loss / max(total_tokens, 1),
            "perplexity": math.exp(min(total_loss / max(total_tokens, 1), 20)),
            "top1": top1_correct / max(total_tokens, 1),
            "top5": top5_correct / max(total_tokens, 1),
            "tokens": total_tokens,
        }
        print(f"  {cat}: loss={results[cat]['loss']:.3f}, top1={results[cat]['top1']:.3f}", flush=True)

    return results


def evaluate_graft_by_category(projection, forward_fn, eval_by_cat, index,
                                glm_store, llama_tokenizer, glm_model, glm_tokenizer,
                                label):
    """Run a graft config on eval corpus, collect per-category metrics + position KL."""
    print(f"\n--- {label} ---", flush=True)
    results = {}

    for cat, entries in sorted(eval_by_cat.items()):
        total_loss = 0.0
        total_tokens = 0
        top1_correct = 0
        top5_correct = 0
        kl_by_pos = defaultdict(list)

        for entry in entries:
            input_id = entry["input_id"]
            text = entry["raw_text"]
            positions_a = entry[config.CORPUS_POSITIONS_KEY_A]
            positions_b = entry[config.CORPUS_POSITIONS_KEY_B]

            # Get Llama token IDs for targets
            llama_enc = llama_tokenizer(text, add_special_tokens=False)
            llama_token_ids = llama_enc["input_ids"]

            # Get GLM hidden states at splice layer
            glm_inputs = glm_tokenizer(text, return_tensors="pt", add_special_tokens=False)
            glm_input_ids = glm_inputs["input_ids"].to("cuda")
            with torch.no_grad():
                glm_out = glm_model(input_ids=glm_input_ids, output_hidden_states=True)
            glm_hs = glm_out.hidden_states[GLM_LAYER]  # (1, glm_seq_len, dim)

            # Build aligned sequence from GLM hidden states
            rows = []
            llama_positions = []
            targets = []
            for pair_idx, (pos_a, pos_b) in enumerate(zip(positions_a, positions_b)):
                if pos_a >= glm_hs.shape[1] or pos_b >= len(llama_token_ids):
                    continue
                rows.append(pos_a)  # GLM position in the hidden state sequence
                llama_positions.append(pos_b)
                targets.append(llama_token_ids[pos_b])

            if len(rows) < 4:
                del glm_out
                continue

            seq_len = min(len(rows), MAX_SEQ_LEN)
            pos_a_arr = torch.tensor(rows[:seq_len], dtype=torch.long)
            glm_vecs = glm_hs[0, pos_a_arr, :].float().unsqueeze(0)  # (1, seq, dim)

            # Apply projection
            with torch.no_grad():
                projected = projection(glm_vecs).half()

                # Build position_ids and mask
                pos_ids = torch.tensor(llama_positions[:seq_len], dtype=torch.long,
                                       device="cuda").unsqueeze(0)
                mask = torch.ones(1, seq_len, device="cuda")

                # Partial forward through Llama
                logits = forward_fn(projected, mask, pos_ids)

            # Targets
            tgt = torch.tensor(targets[:seq_len], dtype=torch.long, device="cuda")

            # Next-token loss
            shift_logits = logits[0, :-1, :]
            shift_targets = tgt[1:]
            n = shift_targets.numel()

            if n == 0:
                del glm_out
                continue

            loss = F.cross_entropy(
                shift_logits, shift_targets, reduction="sum"
            ).item()

            top1_correct += (shift_logits.argmax(dim=-1) == shift_targets).sum().item()
            top5 = shift_logits.topk(5, dim=-1).indices
            top5_correct += (top5 == shift_targets.unsqueeze(-1)).any(dim=-1).sum().item()

            # Per-position KL (using loss as proxy)
            for t in range(n):
                token_loss = F.cross_entropy(
                    shift_logits[t].unsqueeze(0), shift_targets[t].unsqueeze(0)
                ).item()
                pos = t
                if pos < 16:
                    kl_by_pos["0-16"].append(token_loss)
                elif pos < 32:
                    kl_by_pos["16-32"].append(token_loss)
                elif pos < 64:
                    kl_by_pos["32-64"].append(token_loss)
                else:
                    kl_by_pos["64-128"].append(token_loss)

            total_loss += loss
            total_tokens += n
            del glm_out

        results[cat] = {
            "loss": total_loss / max(total_tokens, 1),
            "perplexity": math.exp(min(total_loss / max(total_tokens, 1), 20)),
            "top1": top1_correct / max(total_tokens, 1),
            "top5": top5_correct / max(total_tokens, 1),
            "tokens": total_tokens,
            "loss_by_position": {k: float(np.mean(v)) if v else None for k, v in kl_by_pos.items()},
        }
        print(f"  {cat}: loss={results[cat]['loss']:.3f}, top1={results[cat]['top1']:.3f}, "
              f"pos_loss={results[cat]['loss_by_position']}", flush=True)

    return results


def main():
    print("=" * 70)
    print("CATEGORY ATTRIBUTION EXPERIMENT")
    print("=" * 70)

    eval_by_cat = build_eval_corpus()
    categories = sorted(eval_by_cat.keys())

    # Load index
    with open(config.ACTIVATION_INDEX_PATH) as f:
        index_meta = json.load(f)
    index = index_meta["index"]
    total_rows = index_meta["total_rows"]

    # Load GLM memmap
    pass1_meta_path = os.path.join(config.DATA_DIR, "pass1_metadata.json")
    with open(pass1_meta_path) as f:
        pass1_meta = json.load(f)
    glm_store = np.memmap(
        config.MODEL_A_MMAP_PATH, dtype=np.float16, mode="r",
        shape=(total_rows, pass1_meta["stored_layers"], HIDDEN_DIM),
    )

    # --- Step 1: GLM extraction for eval inputs ---
    print("\nLoading GLM...", flush=True)
    glm_model, glm_tokenizer = load_glm()

    # --- Step 2: Load Llama ---
    # We need both models briefly. GLM ~18GB + Llama ~16GB = 34GB > 32GB.
    # Solution: run GLM extraction in the graft eval function itself,
    # loading both models would OOM. Instead, extract GLM hidden states first,
    # save to dict, unload GLM, then load Llama.

    # Pre-extract GLM hidden states for all eval inputs
    print("\nPre-extracting GLM layer 5 hidden states for eval...", flush=True)
    glm_hidden_cache = {}  # input_id -> (1, seq_len, dim) tensor on CPU
    for cat, entries in eval_by_cat.items():
        for entry in entries:
            text = entry["raw_text"]
            inputs = glm_tokenizer(text, return_tensors="pt", add_special_tokens=False)
            input_ids = inputs["input_ids"].to("cuda")
            with torch.no_grad():
                out = glm_model(input_ids=input_ids, output_hidden_states=True)
            glm_hidden_cache[entry["input_id"]] = out.hidden_states[GLM_LAYER][0].float().cpu()
            del out
    print(f"  Cached {len(glm_hidden_cache)} inputs")

    del glm_model, glm_tokenizer
    torch.cuda.empty_cache()

    # Load Llama
    print("\nLoading Llama...", flush=True)
    llama_model, llama_tokenizer = load_llama()
    for param in llama_model.parameters():
        param.requires_grad = False

    forward_fn = build_partial_forward(llama_model, LLAMA_SPLICE_LAYER)

    # --- Native Llama baseline ---
    native_results = evaluate_native_llama_by_category(llama_model, llama_tokenizer, eval_by_cat)

    # --- Build projection variants ---
    sp_dir = os.path.join(PROJECTIONS_DIR, "S2")

    # Trained linear bridge
    bridge_linear = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=True).cuda()
    bridge_path = os.path.join(GRAFT_DIR, "bridge", "procrustes_init_best.pt")
    if os.path.exists(bridge_path):
        bridge_linear.load_state_dict(torch.load(bridge_path, map_location="cuda"))
    else:
        # Try epoch checkpoint
        for ep in [3, 2, 1]:
            p = os.path.join(GRAFT_DIR, "bridge", f"procrustes_init_epoch{ep}.pt")
            if os.path.exists(p):
                bridge_linear.load_state_dict(torch.load(p, map_location="cuda"))
                break
    bridge_linear.eval()

    # Untrained Procrustes
    procrustes = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=True).cuda()
    W_proc = np.load(os.path.join(sp_dir, "P4_W.npy"))
    mean_glm = np.load(os.path.join(sp_dir, "P4_mean_glm.npy"))
    mean_llama = np.load(os.path.join(sp_dir, "P4_mean_llama.npy"))
    bias_init = mean_llama - mean_glm @ W_proc
    procrustes.weight.data = torch.from_numpy(W_proc.T.copy()).float().cuda()
    procrustes.bias.data = torch.from_numpy(bias_init.copy()).float().cuda()
    procrustes.eval()

    # Trained MLP
    mlp_path = os.path.join(GRAFT_DIR, "bridge_mlp", "mlp_2layer_procrustes_best.pt")
    mlp_bridge = None
    if os.path.exists(mlp_path):
        mlp_bridge = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=True),
            nn.GELU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=True),
        ).cuda()
        mlp_bridge.load_state_dict(torch.load(mlp_path, map_location="cuda"))
        mlp_bridge.eval()

    # --- Evaluate each graft variant ---
    # Custom eval that uses pre-cached GLM hidden states
    def eval_with_cache(projection, label):
        print(f"\n--- {label} ---", flush=True)
        results = {}
        for cat, entries in sorted(eval_by_cat.items()):
            total_loss = 0.0
            total_tokens = 0
            top1_correct = 0
            top5_correct = 0
            loss_by_pos = defaultdict(list)

            for entry in entries:
                input_id = entry["input_id"]
                if input_id not in glm_hidden_cache:
                    continue
                text = entry["raw_text"]
                positions_a = entry[config.CORPUS_POSITIONS_KEY_A]
                positions_b = entry[config.CORPUS_POSITIONS_KEY_B]

                llama_enc = llama_tokenizer(text, add_special_tokens=False)
                llama_token_ids = llama_enc["input_ids"]

                glm_hs = glm_hidden_cache[input_id]  # (glm_seq_len, dim) CPU

                rows = []
                llama_positions = []
                targets = []
                for pos_a, pos_b in zip(positions_a, positions_b):
                    if pos_a >= glm_hs.shape[0] or pos_b >= len(llama_token_ids):
                        continue
                    rows.append(pos_a)
                    llama_positions.append(pos_b)
                    targets.append(llama_token_ids[pos_b])

                if len(rows) < 4:
                    continue

                seq_len = min(len(rows), MAX_SEQ_LEN)
                pos_a_arr = torch.tensor(rows[:seq_len], dtype=torch.long)
                glm_vecs = glm_hs[pos_a_arr].unsqueeze(0).cuda()  # (1, seq, dim)

                with torch.no_grad():
                    projected = projection(glm_vecs).half()
                    pos_ids = torch.tensor(llama_positions[:seq_len], dtype=torch.long,
                                           device="cuda").unsqueeze(0)
                    mask = torch.ones(1, seq_len, device="cuda")
                    logits = forward_fn(projected, mask, pos_ids)

                tgt = torch.tensor(targets[:seq_len], dtype=torch.long, device="cuda")
                shift_logits = logits[0, :-1, :]
                shift_targets = tgt[1:]
                n = shift_targets.numel()
                if n == 0:
                    continue

                loss = F.cross_entropy(shift_logits, shift_targets, reduction="sum").item()
                top1_correct += (shift_logits.argmax(dim=-1) == shift_targets).sum().item()
                top5 = shift_logits.topk(5, dim=-1).indices
                top5_correct += (top5 == shift_targets.unsqueeze(-1)).any(dim=-1).sum().item()

                for t in range(n):
                    tl = F.cross_entropy(shift_logits[t:t+1], shift_targets[t:t+1]).item()
                    p = t
                    bucket = "0-16" if p < 16 else "16-32" if p < 32 else "32-64" if p < 64 else "64-128"
                    loss_by_pos[bucket].append(tl)

                total_loss += loss
                total_tokens += n

            results[cat] = {
                "loss": total_loss / max(total_tokens, 1),
                "perplexity": math.exp(min(total_loss / max(total_tokens, 1), 20)),
                "top1": top1_correct / max(total_tokens, 1),
                "top5": top5_correct / max(total_tokens, 1),
                "tokens": total_tokens,
                "loss_by_position": {k: float(np.mean(v)) if v else None for k, v in loss_by_pos.items()},
            }
            print(f"  {cat}: loss={results[cat]['loss']:.3f}, top1={results[cat]['top1']:.3f}", flush=True)
        return results

    bridge_results = eval_with_cache(bridge_linear, "TRAINED LINEAR BRIDGE")
    procrustes_results = eval_with_cache(procrustes, "UNTRAINED PROCRUSTES")
    mlp_results = eval_with_cache(mlp_bridge, "TRAINED MLP") if mlp_bridge else {}

    # --- Load Phase 2 rank data ---
    rank_data = {}
    rank_path = os.path.join(config.RESULTS_DIR, "category_analysis.json")
    if os.path.exists(rank_path):
        with open(rank_path) as f:
            cat_analysis = json.load(f)
        # Average rank_95 across layers for each category
        for cat, layers in cat_analysis.get("category_rank_profiles", {}).items():
            ranks = [v["rank_95"] for v in layers.values() if isinstance(v, dict)]
            rank_data[cat] = int(np.mean(ranks)) if ranks else None

    # --- Compile results ---
    print(f"\n{'='*70}")
    print(f"  CATEGORY ATTRIBUTION RESULTS")
    print(f"{'='*70}")

    print(f"\n  {'Category':<16} {'Native':>7} {'Bridge':>7} {'Gap':>6} {'Top-1':>6} "
          f"{'Procr':>7} {'MLP':>7} {'Rank':>6}")
    print(f"  {'-'*75}")

    all_results = {}
    for cat in categories:
        native_loss = native_results.get(cat, {}).get("loss", 0)
        bridge_loss = bridge_results.get(cat, {}).get("loss", 0)
        bridge_top1 = bridge_results.get(cat, {}).get("top1", 0)
        procr_loss = procrustes_results.get(cat, {}).get("loss", 0)
        mlp_loss = mlp_results.get(cat, {}).get("loss", 0) if mlp_results else 0
        rank = rank_data.get(cat, "?")
        gap = bridge_loss - native_loss

        print(f"  {cat:<16} {native_loss:>7.3f} {bridge_loss:>7.3f} {gap:>+6.3f} "
              f"{bridge_top1:>5.1%} {procr_loss:>7.3f} {mlp_loss:>7.3f} {rank:>6}")

        all_results[cat] = {
            "native_loss": native_loss,
            "bridge_loss": bridge_loss,
            "gap": gap,
            "bridge_top1": bridge_top1,
            "bridge_top5": bridge_results.get(cat, {}).get("top5", 0),
            "procrustes_loss": procr_loss,
            "mlp_loss": mlp_loss,
            "training_gain": procr_loss - bridge_loss,
            "rank_95": rank,
            "bridge_pos_kl": bridge_results.get(cat, {}).get("loss_by_position", {}),
            "procrustes_pos_kl": procrustes_results.get(cat, {}).get("loss_by_position", {}),
        }

    # Training gain table
    print(f"\n  Training gain (untrained Procrustes -> trained bridge):")
    print(f"  {'Category':<16} {'Untrained':>10} {'Trained':>10} {'Gain':>8} {'Rank':>6}")
    print(f"  {'-'*55}")
    for cat in sorted(categories, key=lambda c: all_results[c]["training_gain"], reverse=True):
        r = all_results[cat]
        print(f"  {cat:<16} {r['procrustes_loss']:>10.3f} {r['bridge_loss']:>10.3f} "
              f"{r['training_gain']:>+8.3f} {r['rank_95']:>6}")

    # Multilingual position breakdown
    print(f"\n  Multilingual per-position loss:")
    print(f"  {'Position':<12} {'Untrained':>10} {'Trained':>10}")
    if "multilingual" in all_results:
        ml = all_results["multilingual"]
        for pos in ["0-16", "16-32", "32-64", "64-128"]:
            u = ml["procrustes_pos_kl"].get(pos, "?")
            t = ml["bridge_pos_kl"].get(pos, "?")
            u_str = f"{u:.3f}" if isinstance(u, float) else str(u)
            t_str = f"{t:.3f}" if isinstance(t, float) else str(t)
            print(f"  {pos:<12} {u_str:>10} {t_str:>10}")

    # Save everything
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # Scatter plot data: rank vs gap
    scatter_data = []
    for cat in categories:
        r = all_results[cat]
        if isinstance(r["rank_95"], (int, float)):
            scatter_data.append({"category": cat, "rank": r["rank_95"], "gap": r["gap"],
                                  "bridge_loss": r["bridge_loss"], "top1": r["bridge_top1"]})
    with open(os.path.join(RESULTS_DIR, "scatter_data.json"), "w") as f:
        json.dump(scatter_data, f, indent=2)

    if len(scatter_data) >= 3:
        ranks = np.array([d["rank"] for d in scatter_data])
        gaps = np.array([d["gap"] for d in scatter_data])
        log_ranks = np.log10(np.maximum(ranks, 1))
        if log_ranks.std() > 0:
            correlation = np.corrcoef(log_ranks, gaps)[0, 1]
            r_squared = correlation ** 2
            print(f"\n  Scatter: log(rank) vs gap correlation = {correlation:.3f}, R² = {r_squared:.3f}")

    print(f"\n  Results saved to {RESULTS_DIR}/")

    del llama_model
    torch.cuda.empty_cache()
    print(f"\n{'='*70}")
    print(f"  CATEGORY ATTRIBUTION COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
