#!/usr/bin/env python3
"""
Graft Experiment — Phase 1: Fit Projections

1. Load Llama, stream through corpus, extract activations at splice layers
2. Pair with GLM activations from memmap
3. Fit 6 projection variants at each of 6 splice points
4. Run diagnostics: holdout residual, sample efficiency, RoPE positional check
"""

import json
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

GRAFT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECTIONS_DIR = os.path.join(GRAFT_DIR, "projections")
os.makedirs(PROJECTIONS_DIR, exist_ok=True)

# Splice points: (name, glm_layer, llama_layer)
SPLICE_POINTS = [
    ("S0", 0, 0),
    ("S1", 4, 23),
    ("S2", 5, 15),
    ("S3", 15, 15),
    ("S4", 35, 31),
    ("S5", 39, 32),
]

# We need Llama hidden states at these layers (0-indexed in hidden_states tuple)
LLAMA_LAYERS_NEEDED = sorted(set(ll for _, _, ll in SPLICE_POINTS))  # [0, 15, 23, 31, 32]

HOLDOUT_FRACTION = 0.2
HOLDOUT_SAMPLE_PER_SPLICE = 20000  # save this many holdout pairs per splice for evaluation


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


def load_corpus_and_index():
    corpus = []
    with open(config.CORPUS_PATH) as f:
        for line in f:
            corpus.append(json.loads(line))
    with open(config.ACTIVATION_INDEX_PATH) as f:
        index_meta = json.load(f)
    return corpus, index_meta


def extract_llama_activations(model, tokenizer, corpus, index_meta):
    """
    Run Llama on the corpus, extract activations at needed layers.
    Accumulate least-squares statistics for projection fitting and
    save a holdout subset for evaluation.
    """
    print("\n--- EXTRACTING LLAMA ACTIVATIONS & ACCUMULATING STATS ---")

    index = index_meta["index"]
    total_rows = index_meta["total_rows"]
    hidden_dim = config.HIDDEN_DIM

    # Load GLM memmap
    pass1_meta_path = os.path.join(config.DATA_DIR, "pass1_metadata.json")
    with open(pass1_meta_path) as f:
        pass1_meta = json.load(f)
    num_layers_glm = pass1_meta["stored_layers"]

    store_glm = np.memmap(
        config.MODEL_A_MMAP_PATH, dtype=np.float16, mode="r",
        shape=(total_rows, num_layers_glm, hidden_dim),
    )

    # For each splice point, accumulate X^T X and X^T Y for least squares
    # X = GLM activations (with bias column), Y = Llama activations
    # (X^T X) is (4097, 4097), (X^T Y) is (4097, 4096) — ~128MB each in float64
    accumulators = {}
    holdout_pairs = {}  # {splice_name: {"glm": [...], "llama": [...], "positions": [...]}}
    pair_counts = {}
    sample_efficiency = {}  # track residual at 10K, 50K, 100K, all

    for name, gl, ll in SPLICE_POINTS:
        d = hidden_dim + 1  # +1 for bias
        accumulators[name] = {
            "XtX": np.zeros((d, d), dtype=np.float64),
            "XtY": np.zeros((d, hidden_dim), dtype=np.float64),
            "sum_glm": np.zeros(hidden_dim, dtype=np.float64),
            "sum_llama": np.zeros(hidden_dim, dtype=np.float64),
            "sum_sq_norm_glm": 0.0,
            "sum_sq_norm_llama": 0.0,
        }
        holdout_pairs[name] = {"glm": [], "llama": [], "positions": []}
        pair_counts[name] = 0
        sample_efficiency[name] = []

    # Determine train/holdout split
    n_inputs = len(corpus)
    holdout_start = int(n_inputs * (1 - HOLDOUT_FRACTION))
    print(f"  Train inputs: 0-{holdout_start-1}, Holdout inputs: {holdout_start}-{n_inputs-1}")

    start_time = time.time()
    processed = 0

    for entry in corpus:
        input_id = entry["input_id"]
        text = entry["raw_text"]
        positions_b = entry[config.CORPUS_POSITIONS_KEY_B]  # Llama positions
        positions_a = entry[config.CORPUS_POSITIONS_KEY_A]  # GLM positions (for memmap row lookup)
        is_holdout = (input_id >= holdout_start)

        # Run Llama inference
        try:
            inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
            input_ids = inputs["input_ids"].to("cuda")
            with torch.no_grad():
                out = model(input_ids=input_ids, output_hidden_states=True)
            hidden_states = out.hidden_states
        except Exception:
            continue

        # Collect valid rows and positions for this input
        valid_rows = []
        valid_pos_b_list = []
        for pair_idx, (pos_a, pos_b) in enumerate(zip(positions_a, positions_b)):
            key = f"{input_id}_{pair_idx}"
            if key not in index:
                continue
            valid_rows.append(index[key])
            valid_pos_b_list.append(pos_b)

        if not valid_rows:
            del out, hidden_states
            processed += 1
            continue

        rows_arr = np.array(valid_rows)
        pos_b_arr = np.array(valid_pos_b_list)

        # For each splice point, batch-accumulate on GPU
        for sp_name, gl, ll in SPLICE_POINTS:
            if ll >= len(hidden_states):
                continue
            hs = hidden_states[ll]
            seq_len = hs.shape[1]

            # Filter valid positions
            mask = pos_b_arr < seq_len
            if not mask.any():
                continue
            cur_rows = rows_arr[mask] if not mask.all() else rows_arr
            cur_pos = pos_b_arr[mask] if not mask.all() else pos_b_arr

            # Load GLM from memmap, Llama from GPU
            glm_np = store_glm[cur_rows, gl, :].astype(np.float32)  # (n, dim)
            t_glm = torch.from_numpy(glm_np).cuda()
            t_llama = hs[0, cur_pos, :].float()  # (n, dim) on GPU

            # NaN filter
            nan_mask = torch.isnan(t_glm).any(dim=1) | torch.isnan(t_llama).any(dim=1)
            if nan_mask.any():
                keep = ~nan_mask
                t_glm = t_glm[keep]
                t_llama = t_llama[keep]
                cur_pos = cur_pos[keep.cpu().numpy()]

            n = t_glm.shape[0]
            if n == 0:
                continue

            if is_holdout:
                if len(holdout_pairs[sp_name]["glm"]) < HOLDOUT_SAMPLE_PER_SPLICE:
                    remaining = HOLDOUT_SAMPLE_PER_SPLICE - len(holdout_pairs[sp_name]["glm"])
                    take = min(n, remaining)
                    holdout_pairs[sp_name]["glm"].append(t_glm[:take].cpu().numpy())
                    holdout_pairs[sp_name]["llama"].append(t_llama[:take].cpu().numpy())
                    holdout_pairs[sp_name]["positions"].append(cur_pos[:take])
            else:
                # Batch GPU accumulation: X^T X and X^T Y
                # Augment GLM with bias column
                ones = torch.ones(n, 1, device="cuda", dtype=torch.float32)
                X = torch.cat([t_glm, ones], dim=1)  # (n, 4097)
                Y = t_llama  # (n, 4096)

                # Accumulate on GPU, transfer to CPU
                XtX_batch = (X.T @ X).double().cpu().numpy()
                XtY_batch = (X.T @ Y).double().cpu().numpy()

                acc = accumulators[sp_name]
                acc["XtX"] += XtX_batch
                acc["XtY"] += XtY_batch
                acc["sum_glm"] += t_glm.double().sum(dim=0).cpu().numpy()
                acc["sum_llama"] += t_llama.double().sum(dim=0).cpu().numpy()

            pair_counts[sp_name] += n

        del out, hidden_states
        if processed % 50 == 0:
            torch.cuda.empty_cache()

        processed += 1
        if processed % 500 == 0:
            elapsed = time.time() - start_time
            rate = processed / elapsed
            remaining = (n_inputs - processed) / rate / 60
            print(f"  [{processed}/{n_inputs}] {rate:.1f} inputs/sec, "
                  f"~{remaining:.1f} min remaining", flush=True)

    elapsed = time.time() - start_time
    print(f"\n  Extraction complete in {elapsed/60:.1f} min")
    for sp_name, _, _ in SPLICE_POINTS:
        n_train = pair_counts[sp_name] - len(holdout_pairs[sp_name]["glm"])
        n_hold = len(holdout_pairs[sp_name]["glm"])
        print(f"  {sp_name}: {n_train} train pairs, {n_hold} holdout pairs")

    # Convert holdout to arrays (lists of arrays -> single array)
    for sp_name in holdout_pairs:
        if holdout_pairs[sp_name]["glm"]:
            holdout_pairs[sp_name]["glm"] = np.concatenate(holdout_pairs[sp_name]["glm"])
            holdout_pairs[sp_name]["llama"] = np.concatenate(holdout_pairs[sp_name]["llama"])
            holdout_pairs[sp_name]["positions"] = np.concatenate(holdout_pairs[sp_name]["positions"])

    del store_glm
    return accumulators, holdout_pairs, pair_counts


def fit_projections(accumulators, holdout_pairs, pair_counts):
    """Fit all 6 projection variants at each splice point."""
    print("\n--- FITTING PROJECTIONS ---")

    results = {}

    for sp_name, gl, ll in SPLICE_POINTS:
        print(f"\n  {sp_name} (GLM {gl} -> Llama {ll}):")
        acc = accumulators[sp_name]
        hold_glm = holdout_pairs[sp_name]["glm"]
        hold_llama = holdout_pairs[sp_name]["llama"]
        hold_pos = holdout_pairs[sp_name]["positions"]

        if not isinstance(hold_glm, np.ndarray) or len(hold_glm) == 0:
            print(f"    No holdout data, skipping")
            continue

        n_train = pair_counts[sp_name] - len(hold_glm)
        hidden_dim = hold_glm.shape[1]
        results[sp_name] = {}

        # --- P0: Identity ---
        projected = hold_glm.copy()
        res = evaluate_projection(projected, hold_llama, hold_pos, "P0:Identity")
        results[sp_name]["P0"] = res

        # --- P1: Learned affine (W*x + b) ---
        print(f"    Fitting P1 (affine, {hidden_dim+1}x{hidden_dim} params)...", end=" ", flush=True)
        try:
            # Solve (X^T X) * [W; b]^T = X^T Y
            Wb, residuals, rank, sv = np.linalg.lstsq(acc["XtX"], acc["XtY"], rcond=None)
            W_affine = Wb[:hidden_dim, :].T  # (4096, 4096)
            b_affine = Wb[hidden_dim, :]      # (4096,)
            projected = hold_glm @ W_affine.T + b_affine
            print("done")
            res = evaluate_projection(projected, hold_llama, hold_pos, "P1:Affine")
            results[sp_name]["P1"] = res

            # Save
            sp_dir = os.path.join(PROJECTIONS_DIR, sp_name)
            os.makedirs(sp_dir, exist_ok=True)
            np.save(os.path.join(sp_dir, "P1_W.npy"), W_affine.astype(np.float32))
            np.save(os.path.join(sp_dir, "P1_b.npy"), b_affine.astype(np.float32))
        except Exception as e:
            print(f"FAILED: {e}")
            results[sp_name]["P1"] = {"error": str(e)}

        # --- P2: Mean-shift only (x + b) ---
        mean_glm = acc["sum_glm"] / n_train
        mean_llama = acc["sum_llama"] / n_train
        b_shift = (mean_llama - mean_glm).astype(np.float32)
        projected = hold_glm + b_shift
        res = evaluate_projection(projected, hold_llama, hold_pos, "P2:MeanShift")
        results[sp_name]["P2"] = res
        np.save(os.path.join(sp_dir, "P2_b.npy"), b_shift)

        # --- P3: Random orthogonal ---
        rng = np.random.RandomState(42)
        Q, _ = np.linalg.qr(rng.randn(hidden_dim, hidden_dim))
        projected = hold_glm @ Q.T.astype(np.float32)
        res = evaluate_projection(projected, hold_llama, hold_pos, "P3:RandOrth")
        results[sp_name]["P3"] = res

        # --- P4: Procrustes (orthogonal W) ---
        print(f"    Fitting P4 (Procrustes)...", end=" ", flush=True)
        # Center the data
        glm_c = hold_glm - hold_glm.mean(axis=0)
        llama_c = hold_llama - hold_llama.mean(axis=0)
        # SVD of cross-covariance
        M = glm_c.T @ llama_c  # (dim, dim)
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        W_proc = (U @ Vt).astype(np.float32)  # orthogonal
        # Apply with mean correction
        projected = (hold_glm - hold_glm.mean(axis=0)) @ W_proc + hold_llama.mean(axis=0)
        print("done")
        res = evaluate_projection(projected, hold_llama, hold_pos, "P4:Procrustes")
        results[sp_name]["P4"] = res
        np.save(os.path.join(sp_dir, "P4_W.npy"), W_proc)
        np.save(os.path.join(sp_dir, "P4_mean_glm.npy"), hold_glm.mean(axis=0).astype(np.float32))
        np.save(os.path.join(sp_dir, "P4_mean_llama.npy"), hold_llama.mean(axis=0).astype(np.float32))

        # --- P5: Rank-1 correction (I + alpha * v * v^T + b) ---
        print(f"    Fitting P5 (rank-1)...", end=" ", flush=True)
        eigen_dir = os.path.join(config.RESULTS_DIR, "eigen")
        evec_path = os.path.join(eigen_dir, f"layer_A{gl}_B{ll}_eigenvectors.npy")
        if os.path.exists(evec_path):
            evecs = np.load(evec_path)
            v = evecs[:, 0].astype(np.float64)  # PC0 direction
            # Project holdout GLM onto v
            glm_proj_v = hold_glm.astype(np.float64) @ v  # (N,)
            # Residual after identity: llama - glm
            residual = hold_llama.astype(np.float64) - hold_glm.astype(np.float64)
            # Fit alpha: minimize ||residual - alpha * (glm @ v) * v||^2
            # Plus bias b
            # residual_i = alpha * (x_i . v) * v + b
            # This is a linear regression with features [(x.v)*v, 1]
            # But simpler: project residual onto v direction for alpha, rest for b
            res_along_v = residual @ v  # (N,)
            # alpha * (x . v) = res_along_v => alpha = sum(res_along_v * glm_proj_v) / sum(glm_proj_v^2)
            alpha = float(np.dot(res_along_v, glm_proj_v) / (np.dot(glm_proj_v, glm_proj_v) + 1e-10))
            # bias = mean(residual - alpha * outer_product_correction)
            correction = alpha * np.outer(glm_proj_v, v)  # (N, dim)
            b_rank1 = (residual - correction).mean(axis=0).astype(np.float32)

            projected = hold_glm + alpha * np.outer(hold_glm.astype(np.float64) @ v, v).astype(np.float32) + b_rank1
            print(f"alpha={alpha:.4f}")
            res = evaluate_projection(projected.astype(np.float32), hold_llama, hold_pos, "P5:Rank1")
            results[sp_name]["P5"] = res
            results[sp_name]["P5"]["alpha"] = alpha
            np.save(os.path.join(sp_dir, "P5_v.npy"), v.astype(np.float32))
            np.save(os.path.join(sp_dir, "P5_alpha.npy"), np.array([alpha], dtype=np.float32))
            np.save(os.path.join(sp_dir, "P5_b.npy"), b_rank1)
        else:
            print(f"no eigenvectors at {evec_path}")
            results[sp_name]["P5"] = {"error": "no eigenvectors"}

    return results


def evaluate_projection(projected, target, positions, label):
    """Evaluate projection quality on holdout set."""
    # L2 residual
    diff = projected - target
    l2_per = np.linalg.norm(diff, axis=1)
    target_norms = np.linalg.norm(target, axis=1)
    relative_l2 = l2_per / (target_norms + 1e-10)

    # Cosine similarity
    cos = np.sum(projected * target, axis=1) / (
        np.linalg.norm(projected, axis=1) * np.linalg.norm(target, axis=1) + 1e-10
    )

    # Per-position cosine similarity (RoPE diagnostic)
    pos_buckets = [(0, 16), (16, 32), (32, 64), (64, 128)]
    cos_by_pos = {}
    for lo, hi in pos_buckets:
        mask = (positions >= lo) & (positions < hi)
        if mask.sum() > 0:
            cos_by_pos[f"{lo}-{hi}"] = float(cos[mask].mean())

    res = {
        "l2_mean": float(l2_per.mean()),
        "l2_std": float(l2_per.std()),
        "relative_l2_mean": float(relative_l2.mean()),
        "cos_mean": float(cos.mean()),
        "cos_std": float(cos.std()),
        "cos_min": float(cos.min()),
        "cos_by_position": cos_by_pos,
        "n_holdout": len(target),
    }

    print(f"    {label}: cos={cos.mean():.4f} (min={cos.min():.4f}), "
          f"rel_L2={relative_l2.mean():.4f}, "
          f"pos_cos={cos_by_pos}")

    return res


def main():
    print("=" * 70)
    print("GRAFT PHASE 1: FIT PROJECTIONS")
    print("=" * 70)

    # Load Llama
    print("\nLoading Llama...", flush=True)
    model, tokenizer = load_llama()

    # Load corpus
    print("\nLoading corpus...", flush=True)
    corpus, index_meta = load_corpus_and_index()
    print(f"  {len(corpus)} inputs, {index_meta['total_rows']} aligned pairs")

    # Extract activations and accumulate stats
    accumulators, holdout_pairs, pair_counts = extract_llama_activations(
        model, tokenizer, corpus, index_meta
    )

    # Unload Llama (projection fitting is CPU-only)
    print("\nUnloading Llama...")
    del model
    torch.cuda.empty_cache()

    # Fit projections
    results = fit_projections(accumulators, holdout_pairs, pair_counts)

    # Save results
    out_path = os.path.join(GRAFT_DIR, "phase1_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    print("\nPhase 1 complete.")


if __name__ == "__main__":
    main()
