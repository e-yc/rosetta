#!/usr/bin/env python3
"""
Graft Experiment — Phase 2: Functional Diagnostics

Runs projected GLM activations through Llama's late layers via partial forward pass.
Measures KL divergence, top-1 agreement, and error propagation vs native Llama.

Tests top configs from Phase 1 on a held-out eval subset.
"""

import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

GRAFT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECTIONS_DIR = os.path.join(GRAFT_DIR, "projections")

# Configs to test: (name, glm_layer, llama_layer, projection_id)
# Focus on best performers from Phase 1 + key comparisons
CONFIGS = [
    # Best overall: S3 × P4
    ("S3_P4", 15, 15, "P4"),
    ("S3_P1", 15, 15, "P1"),
    ("S3_P0", 15, 15, "P0"),  # identity baseline
    # Atlas layer
    ("S2_P4", 5, 15, "P4"),
    ("S2_P1", 5, 15, "P1"),
    # Highest CKA
    ("S1_P4", 4, 23, "P4"),
    # Late layer
    ("S4_P4", 35, 31, "P4"),
    # Negative controls
    ("S0_P4", 0, 0, "P4"),
    ("S5_P4", 39, 32, "P4"),
    ("S3_P3", 15, 15, "P3"),  # random orthogonal
]

EVAL_SIZE = 200  # inputs from holdout portion of corpus


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


def load_projection(sp_name, proj_id, hidden_dim):
    """Load a fitted projection. Returns a function that transforms activations."""
    sp_dir = os.path.join(PROJECTIONS_DIR, sp_name)

    if proj_id == "P0":
        return lambda x: x  # identity

    elif proj_id == "P1":
        W = torch.from_numpy(np.load(os.path.join(sp_dir, "P1_W.npy"))).cuda()
        b = torch.from_numpy(np.load(os.path.join(sp_dir, "P1_b.npy"))).cuda()
        return lambda x: x @ W.T + b

    elif proj_id == "P3":
        rng = np.random.RandomState(42)
        Q, _ = np.linalg.qr(rng.randn(hidden_dim, hidden_dim))
        Q_t = torch.from_numpy(Q.T.astype(np.float32)).cuda()
        return lambda x: x @ Q_t

    elif proj_id == "P4":
        W = torch.from_numpy(np.load(os.path.join(sp_dir, "P4_W.npy"))).cuda()
        mean_glm = torch.from_numpy(np.load(os.path.join(sp_dir, "P4_mean_glm.npy"))).cuda()
        mean_llama = torch.from_numpy(np.load(os.path.join(sp_dir, "P4_mean_llama.npy"))).cuda()
        return lambda x: (x - mean_glm) @ W + mean_llama

    else:
        raise ValueError(f"Unknown projection {proj_id}")


def split_forward_with_intermediates(model, hidden, splice_layer, input_ids):
    """Run Llama from splice_layer to end, return logits and per-layer hidden states."""
    llama = model.model
    device = input_ids.device
    seq_len = input_ids.shape[1]

    intermediates = [hidden.clone()]

    with torch.no_grad():
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        cache_position = torch.arange(seq_len, device=device)
        causal_mask = llama._update_causal_mask(
            None, hidden, cache_position, past_key_values=None, output_attentions=False
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
            intermediates.append(hidden.clone())

        hidden = llama.norm(hidden)
        logits = model.lm_head(hidden)

    return logits, intermediates


def compute_metrics(logits_native, logits_graft):
    """Compute KL divergence, top-1 agreement, top-5 overlap per position."""
    # (seq_len, vocab_size)
    p_native = F.log_softmax(logits_native.float(), dim=-1)
    p_graft = F.log_softmax(logits_graft.float(), dim=-1)
    q_native = F.softmax(logits_native.float(), dim=-1)

    # KL(native || graft) per position
    kl = F.kl_div(p_graft, q_native, reduction="none", log_target=False).sum(dim=-1)

    # Top-1 agreement
    top1_native = logits_native.argmax(dim=-1)
    top1_graft = logits_graft.argmax(dim=-1)
    top1_agree = (top1_native == top1_graft).float()

    # Top-5 overlap
    top5_native = logits_native.topk(5, dim=-1).indices
    top5_graft = logits_graft.topk(5, dim=-1).indices
    overlap = torch.zeros(logits_native.shape[0], device=logits_native.device)
    for i in range(5):
        overlap += (top5_graft == top5_native[:, i:i+1]).any(dim=-1).float()
    overlap /= 5.0

    return kl.cpu().numpy(), top1_agree.cpu().numpy(), overlap.cpu().numpy()


def main():
    print("=" * 70)
    print("GRAFT PHASE 2: FUNCTIONAL DIAGNOSTICS")
    print("=" * 70)

    # Load eval corpus (last portion, held out from projection fitting)
    print("\nLoading corpus...", flush=True)
    corpus = []
    with open(config.CORPUS_PATH) as f:
        for line in f:
            corpus.append(json.loads(line))
    n_inputs = len(corpus)
    holdout_start = int(n_inputs * 0.8)
    eval_corpus = corpus[holdout_start:holdout_start + EVAL_SIZE]
    print(f"  Eval corpus: {len(eval_corpus)} inputs (ids {eval_corpus[0]['input_id']}-{eval_corpus[-1]['input_id']})")

    # --- Step 1: Extract GLM intermediates ---
    print("\nLoading GLM for intermediate extraction...", flush=True)
    glm_model, glm_tokenizer = load_glm()

    # Collect GLM hidden states at needed layers
    glm_layers_needed = sorted(set(gl for _, gl, _, _ in CONFIGS))
    print(f"  Extracting GLM layers: {glm_layers_needed}")

    glm_intermediates = {}  # {input_id: {glm_layer: tensor (seq_len, dim)}}

    for entry in eval_corpus:
        input_id = entry["input_id"]
        text = entry["raw_text"]
        try:
            inputs = glm_tokenizer(text, return_tensors="pt", add_special_tokens=False)
            input_ids = inputs["input_ids"].to("cuda")
            with torch.no_grad():
                out = glm_model(input_ids=input_ids, output_hidden_states=True)
            hs = out.hidden_states
            glm_intermediates[input_id] = {}
            for gl in glm_layers_needed:
                if gl < len(hs):
                    glm_intermediates[input_id][gl] = hs[gl][0].float().cpu()  # (seq_len, dim)
            del out, hs
        except Exception as e:
            print(f"  Skip input {input_id}: {e}")
            continue

    print(f"  Extracted {len(glm_intermediates)} inputs")

    del glm_model, glm_tokenizer
    torch.cuda.empty_cache()

    # --- Step 2: Load Llama, run diagnostics ---
    print("\nLoading Llama for diagnostics...", flush=True)
    llama_model, llama_tokenizer = load_llama()
    hidden_dim = config.HIDDEN_DIM

    # First: get native Llama logits and hidden states for comparison
    print("\n--- NATIVE LLAMA BASELINE ---")
    native_data = {}  # {input_id: {"logits": tensor, "hidden_states": list}}

    for entry in eval_corpus:
        input_id = entry["input_id"]
        text = entry["raw_text"]
        try:
            inputs = llama_tokenizer(text, return_tensors="pt", add_special_tokens=False)
            input_ids = inputs["input_ids"].to("cuda")
            with torch.no_grad():
                out = llama_model(input_ids=input_ids, output_hidden_states=True)
            native_data[input_id] = {
                "logits": out.logits[0].cpu(),  # (seq_len, vocab)
                "input_ids": input_ids.cpu(),
                "seq_len": input_ids.shape[1],
            }
            # Save hidden states at splice layers for error propagation
            splice_layers = sorted(set(ll for _, _, ll, _ in CONFIGS))
            native_data[input_id]["hidden_at_splice"] = {
                ll: out.hidden_states[ll][0].cpu() for ll in splice_layers if ll < len(out.hidden_states)
            }
            del out
        except Exception:
            continue

    print(f"  Baseline computed for {len(native_data)} inputs")

    # --- Step 3: Run each graft config ---
    print("\n--- GRAFT DIAGNOSTICS ---")
    all_results = {}

    for cfg_name, gl, ll, proj_id in CONFIGS:
        sp_name = cfg_name.split("_")[0]
        print(f"\n  Config: {cfg_name} (GLM {gl} -> proj {proj_id} -> Llama {ll}+)", flush=True)

        try:
            proj_fn = load_projection(sp_name, proj_id, hidden_dim)
        except Exception as e:
            print(f"    SKIP: {e}")
            continue

        kl_all = []
        top1_all = []
        top5_all = []
        positions_all = []
        n_tokens = 0

        for entry in eval_corpus:
            input_id = entry["input_id"]
            if input_id not in glm_intermediates or input_id not in native_data:
                continue
            if gl not in glm_intermediates[input_id]:
                continue

            native = native_data[input_id]
            glm_hs = glm_intermediates[input_id][gl]  # (glm_seq_len, dim) CPU

            # Project GLM activations
            glm_gpu = glm_hs.unsqueeze(0).cuda()  # (1, seq_len, dim)
            projected = proj_fn(glm_gpu)  # (1, seq_len, dim)

            # Note: GLM and Llama may have different seq lengths for the same text.
            # Truncate to the shorter one for comparison.
            llama_seq_len = native["seq_len"]
            glm_seq_len = projected.shape[1]
            min_len = min(llama_seq_len, glm_seq_len)

            if min_len < 2:
                continue

            projected_trunc = projected[:, :min_len, :]

            # Run through Llama's late layers
            logits_graft, _ = split_forward_with_intermediates(
                llama_model, projected_trunc.half(), ll, native["input_ids"][:, :min_len].cuda()
            )

            # Compare to native Llama logits (truncated to same length)
            logits_native = native["logits"][:min_len, :].cuda()
            logits_graft = logits_graft[0, :min_len, :]

            kl, top1, top5 = compute_metrics(logits_native, logits_graft)
            positions = np.arange(min_len)

            kl_all.append(kl)
            top1_all.append(top1)
            top5_all.append(top5)
            positions_all.append(positions)
            n_tokens += min_len

        if not kl_all:
            print(f"    No valid inputs")
            continue

        kl_all = np.concatenate(kl_all)
        top1_all = np.concatenate(top1_all)
        top5_all = np.concatenate(top5_all)
        positions_all = np.concatenate(positions_all)

        # Position-bucketed KL
        kl_by_pos = {}
        for lo, hi in [(0, 16), (16, 32), (32, 64), (64, 128)]:
            mask = (positions_all >= lo) & (positions_all < hi)
            if mask.sum() > 0:
                kl_by_pos[f"{lo}-{hi}"] = float(np.median(kl_all[mask]))

        result = {
            "kl_median": float(np.median(kl_all)),
            "kl_mean": float(np.mean(kl_all)),
            "kl_p90": float(np.percentile(kl_all, 90)),
            "top1_agreement": float(top1_all.mean()),
            "top5_overlap": float(top5_all.mean()),
            "kl_by_position": kl_by_pos,
            "n_tokens": n_tokens,
            "n_inputs": len(kl_all) // max(1, min_len),
        }
        all_results[cfg_name] = result

        print(f"    KL median={result['kl_median']:.2f}, mean={result['kl_mean']:.2f}, "
              f"top1={result['top1_agreement']:.3f}, top5={result['top5_overlap']:.3f}, "
              f"tokens={n_tokens}", flush=True)
        print(f"    KL by position: {kl_by_pos}", flush=True)

    # --- Save results ---
    out_path = os.path.join(GRAFT_DIR, "phase2_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    # --- Print summary table ---
    print(f"\n{'='*70}")
    print(f"  PHASE 2 SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Config':<15} {'KL med':>8} {'KL mean':>8} {'Top-1':>7} {'Top-5':>7}")
    print(f"  {'-'*50}")
    for cfg_name in sorted(all_results.keys()):
        r = all_results[cfg_name]
        print(f"  {cfg_name:<15} {r['kl_median']:>8.2f} {r['kl_mean']:>8.2f} "
              f"{r['top1_agreement']:>7.3f} {r['top5_overlap']:>7.3f}")

    del llama_model
    torch.cuda.empty_cache()
    print("\nPhase 2 complete.")


if __name__ == "__main__":
    main()
