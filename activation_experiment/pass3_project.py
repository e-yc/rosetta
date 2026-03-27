#!/usr/bin/env python3
"""
Activation Differential Experiment — Pass 3: Projection

1. Eigendecompose covariance matrices from Pass 2
2. Identify the "atlas layer" (lowest rank / most structured disagreement)
3. Reload models one at a time, project activations onto top PCs
4. Extract extreme tokens for each PC direction

Requires: NVIDIA GPU with >= 20GB VRAM (for projection)
Can also run in CPU-only mode (eigendecomposition + atlas from saved projections).
"""

import json
import os
import sys
import time

import numpy as np

import config


# ---------------------------------------------------------------------------
# Eigendecomposition
# ---------------------------------------------------------------------------

def eigendecompose_all():
    """Eigendecompose all overall covariance matrices, compute rank profiles."""
    print("\n--- EIGENDECOMPOSITION ---")

    # Load metadata
    meta_path = os.path.join(config.RESULTS_DIR, "pass2_metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)

    matched_pairs = [(d["a_layer"], d["b_layer"]) for d in meta["matched_pairs"]]
    hidden_dim = meta["hidden_dim"]

    rank_profile = {}
    eigenvalue_spectra = {}

    for a_l, b_l in matched_pairs:
        prefix = f"layer_A{a_l}_B{b_l}"
        cov_path = os.path.join(config.OVERALL_STATS_DIR, f"{prefix}_cov_diff.npy")

        if not os.path.exists(cov_path):
            print(f"  Skipping {prefix}: covariance not found")
            continue

        cov = np.load(cov_path)

        # Eigendecompose (eigh for symmetric matrices)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort descending
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Clip tiny negatives (numerical noise)
        eigenvalues = np.maximum(eigenvalues, 0)

        # Cumulative explained variance
        total_var = eigenvalues.sum()
        if total_var > 0:
            cumvar = np.cumsum(eigenvalues) / total_var
            rank_95 = int(np.searchsorted(cumvar, config.RANK_THRESHOLD) + 1)
        else:
            cumvar = np.zeros_like(eigenvalues)
            rank_95 = 0

        rank_profile[prefix] = {
            "a_layer": a_l,
            "b_layer": b_l,
            "rank_95": rank_95,
            "total_variance": float(total_var),
            "top_10_eigenvalues": eigenvalues[:10].tolist(),
        }
        eigenvalue_spectra[prefix] = eigenvalues[:100].tolist()  # save top 100

        # Save eigenvectors for the atlas
        eigen_dir = os.path.join(config.RESULTS_DIR, "eigen")
        os.makedirs(eigen_dir, exist_ok=True)
        np.save(os.path.join(eigen_dir, f"{prefix}_eigenvalues.npy"), eigenvalues.astype(np.float32))
        np.save(os.path.join(eigen_dir, f"{prefix}_eigenvectors.npy"),
                eigenvectors[:, :config.TOP_K_PCS].astype(np.float32))

        print(f"  {prefix}: rank_95={rank_95}, total_var={total_var:.2e}, "
              f"top eigenvalue={eigenvalues[0]:.2e}")

    # Save rank profile
    with open(os.path.join(config.RESULTS_DIR, "rank_profile.json"), "w") as f:
        json.dump(rank_profile, f, indent=2)
    with open(os.path.join(config.RESULTS_DIR, "eigenvalue_spectra.json"), "w") as f:
        json.dump(eigenvalue_spectra, f, indent=2)

    # Find atlas layer (lowest rank_95)
    if rank_profile:
        atlas_layer = min(rank_profile.items(), key=lambda x: x[1]["rank_95"])
        atlas_key = atlas_layer[0]
        atlas_info = atlas_layer[1]
        print(f"\n  Atlas layer: {atlas_key} (rank_95={atlas_info['rank_95']})")
    else:
        atlas_key = None
        atlas_info = None
        print("\n  WARNING: No rank profile computed")

    atlas_meta = {
        "atlas_layer_key": atlas_key,
        "atlas_a_layer": atlas_info["a_layer"] if atlas_info else None,
        "atlas_b_layer": atlas_info["b_layer"] if atlas_info else None,
        "atlas_rank_95": atlas_info["rank_95"] if atlas_info else None,
    }
    with open(os.path.join(config.RESULTS_DIR, "atlas_meta.json"), "w") as f:
        json.dump(atlas_meta, f, indent=2)

    return atlas_key, rank_profile


def eigendecompose_categories():
    """Eigendecompose per-category covariance matrices."""
    print("\n--- CATEGORY EIGENDECOMPOSITION ---")

    meta_path = os.path.join(config.RESULTS_DIR, "pass2_metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)

    category_layers = [(d["a_layer"], d["b_layer"]) for d in meta["category_layers"]]
    categories = meta["categories"]

    category_results = {}
    for cat in categories:
        category_results[cat] = {}
        cat_dir = os.path.join(config.CATEGORY_STATS_DIR, cat)

        for a_l, b_l in category_layers:
            prefix = f"layer_A{a_l}_B{b_l}"
            cov_path = os.path.join(cat_dir, f"{prefix}_cov_diff.npy")

            if not os.path.exists(cov_path):
                continue

            cov = np.load(cov_path)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            idx = eigenvalues.argsort()[::-1]
            eigenvalues = np.maximum(eigenvalues[idx], 0)
            eigenvectors = eigenvectors[:, idx]

            total_var = eigenvalues.sum()
            if total_var > 0:
                cumvar = np.cumsum(eigenvalues) / total_var
                rank_95 = int(np.searchsorted(cumvar, config.RANK_THRESHOLD) + 1)
            else:
                rank_95 = 0

            category_results[cat][prefix] = {
                "rank_95": rank_95,
                "total_variance": float(total_var),
            }

            # Save top eigenvectors for subspace overlap computation
            eigen_dir = os.path.join(config.RESULTS_DIR, "eigen", "categories", cat)
            os.makedirs(eigen_dir, exist_ok=True)
            np.save(os.path.join(eigen_dir, f"{prefix}_eigenvectors_top10.npy"),
                    eigenvectors[:, :10].astype(np.float32))

    # Compute subspace overlap between categories
    overlap_results = {}
    for a_l, b_l in category_layers:
        prefix = f"layer_A{a_l}_B{b_l}"
        overlap_matrix = np.zeros((len(categories), len(categories)))

        for i, cat1 in enumerate(categories):
            for j, cat2 in enumerate(categories):
                v1_path = os.path.join(config.RESULTS_DIR, "eigen", "categories",
                                        cat1, f"{prefix}_eigenvectors_top10.npy")
                v2_path = os.path.join(config.RESULTS_DIR, "eigen", "categories",
                                        cat2, f"{prefix}_eigenvectors_top10.npy")
                if os.path.exists(v1_path) and os.path.exists(v2_path):
                    V1 = np.load(v1_path)
                    V2 = np.load(v2_path)
                    k = min(10, V1.shape[1], V2.shape[1])
                    overlap = V1[:, :k].T @ V2[:, :k]
                    overlap_matrix[i, j] = float(np.sum(overlap ** 2) / k)
                else:
                    overlap_matrix[i, j] = 0.0

        overlap_results[prefix] = {
            "categories": categories,
            "overlap_matrix": overlap_matrix.tolist(),
        }

    # Save category analysis
    with open(os.path.join(config.RESULTS_DIR, "category_analysis.json"), "w") as f:
        json.dump({
            "category_rank_profiles": category_results,
            "subspace_overlap": overlap_results,
        }, f, indent=2)

    print(f"  Category analysis saved.")
    return category_results


# ---------------------------------------------------------------------------
# Atlas Projection (requires GPU)
# ---------------------------------------------------------------------------

def project_atlas(atlas_key):
    """Project activations onto top PCs of the atlas layer."""
    print(f"\n--- ATLAS PROJECTION ({atlas_key}) ---")

    # Load atlas eigenvectors
    eigen_dir = os.path.join(config.RESULTS_DIR, "eigen")
    eigvecs = np.load(os.path.join(eigen_dir, f"{atlas_key}_eigenvectors.npy"))  # (hidden_dim, top_k)
    eigvals = np.load(os.path.join(eigen_dir, f"{atlas_key}_eigenvalues.npy"))
    mean_diff = np.load(os.path.join(config.OVERALL_STATS_DIR, f"{atlas_key}_mean_diff.npy"))

    # Load metadata
    with open(os.path.join(config.RESULTS_DIR, "atlas_meta.json")) as f:
        atlas_meta = json.load(f)
    a_layer = atlas_meta["atlas_a_layer"]
    b_layer = atlas_meta["atlas_b_layer"]

    # Load corpus and index
    corpus = []
    with open(config.CORPUS_PATH) as f:
        for line in f:
            corpus.append(json.loads(line))
    with open(config.ACTIVATION_INDEX_PATH) as f:
        index_meta = json.load(f)
    index = index_meta["index"]

    total_rows = index_meta["total_rows"]
    hidden_dim = index_meta["hidden_dim"]

    # Read actual stored layer count from pass1 metadata
    pass1_meta_path = os.path.join(config.DATA_DIR, "pass1_metadata.json")
    with open(pass1_meta_path) as f:
        pass1_meta = json.load(f)
    num_layers_a = pass1_meta["stored_layers"]

    # Load Model A activations
    store_a = np.memmap(
        config.MODEL_A_MMAP_PATH, dtype=np.float16, mode="r",
        shape=(total_rows, num_layers_a, hidden_dim),
    )

    # We need Model B activations at the atlas layer
    # Two approaches:
    # (A) Re-run Model B inference (GPU required)
    # (B) Compute differential from stored Model A + re-run Model B
    # Going with (A): reload Model B and re-run

    has_gpu = False
    try:
        import torch
        has_gpu = torch.cuda.is_available()
    except ImportError:
        pass

    if has_gpu:
        print("  GPU available — running projection with live Model B inference")
        _project_with_gpu(corpus, index, store_a, a_layer, b_layer,
                          eigvecs, mean_diff, hidden_dim)
    else:
        print("  No GPU — projecting Model A activations only (differential = act_a - mean_a)")
        _project_cpu_only(corpus, index, store_a, a_layer, eigvecs, mean_diff, hidden_dim)

    # Save variance explained
    total_var = eigvals.sum()
    variance_explained = []
    for k in range(min(config.TOP_K_PCS, len(eigvals))):
        variance_explained.append({
            "pc": k,
            "eigenvalue": float(eigvals[k]),
            "variance_pct": float(eigvals[k] / total_var * 100) if total_var > 0 else 0,
            "cumulative_pct": float(eigvals[:k + 1].sum() / total_var * 100) if total_var > 0 else 0,
        })

    with open(os.path.join(config.ATLAS_DIR, "variance_explained.json"), "w") as f:
        json.dump(variance_explained, f, indent=2)

    del store_a


def _project_with_gpu(corpus, index, store_a, a_layer, b_layer,
                       eigvecs, mean_diff, hidden_dim):
    """Project differentials onto PCs using GPU for Model B inference."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load Model B
    from transformers import AutoConfig
    token = os.environ.get("HF_TOKEN") or True
    for mid in [config.MODEL_B_ID] + config.MODEL_B_FALLBACKS:
        try:
            model_config = AutoConfig.from_pretrained(mid, trust_remote_code=True, token=token)
            if not hasattr(model_config, "max_length"):
                model_config.max_length = getattr(model_config, "seq_length", 8192)
            model_b = AutoModelForCausalLM.from_pretrained(
                mid, config=model_config, torch_dtype=torch.float16, device_map="cuda",
                trust_remote_code=True, token=token,
            )
            tokenizer_b = AutoTokenizer.from_pretrained(mid, trust_remote_code=True, token=token)
            break
        except Exception:
            continue
    else:
        print("  Could not load Model B — falling back to CPU-only projection")
        _project_cpu_only(corpus, index, store_a, a_layer, eigvecs, mean_diff, hidden_dim)
        return

    top_k = eigvecs.shape[1]
    # Collect all projections and token contexts
    projections = []  # list of (projection_vector, input_id, pair_idx, token_text, context)
    token_meta = []

    for entry in corpus:
        input_id = entry["input_id"]
        text = entry["raw_text"]
        positions_a = entry[config.CORPUS_POSITIONS_KEY_A]
        positions_b = entry[config.CORPUS_POSITIONS_KEY_B]

        try:
            inp = tokenizer_b(text, return_tensors="pt", add_special_tokens=False)
            inp = {k: v.to(model_b.device) for k, v in inp.items()}
            with torch.no_grad():
                out = model_b(**inp, output_hidden_states=True)
            hs_b = out.hidden_states
        except Exception:
            continue

        for pair_idx, (pos_a, pos_b) in enumerate(zip(positions_a, positions_b)):
            key = f"{input_id}_{pair_idx}"
            if key not in index:
                continue
            row = index[key]

            vec_a = store_a[row, a_layer, :].astype(np.float32)
            if b_layer < len(hs_b) and pos_b < hs_b[b_layer].shape[1]:
                vec_b = hs_b[b_layer][0, pos_b, :].float().cpu().numpy()
            else:
                continue

            diff = vec_a - vec_b
            proj = (diff - mean_diff) @ eigvecs  # (top_k,)

            projections.append(proj)

            # Get token context
            tokens_b = tokenizer_b.convert_ids_to_tokens(
                inp["input_ids"][0].cpu().tolist()
            )
            token_str = tokens_b[pos_b] if pos_b < len(tokens_b) else "???"
            # Context: ±3 tokens
            ctx_start = max(0, pos_b - 3)
            ctx_end = min(len(tokens_b), pos_b + 4)
            context = " ".join(tokens_b[ctx_start:ctx_end])

            token_meta.append({
                "input_id": input_id,
                "pair_idx": pair_idx,
                "token": token_str,
                "context": context,
                "category": entry["category"],
                "text_snippet": text[:100],
            })

        del out, hs_b
        torch.cuda.empty_cache()

        if len(projections) % 5000 == 0 and len(projections) > 0:
            print(f"    Projected {len(projections)} tokens...")

    del model_b
    torch.cuda.empty_cache()

    _save_atlas_extremes(projections, token_meta)


def _project_cpu_only(corpus, index, store_a, a_layer, eigvecs, mean_diff, hidden_dim):
    """CPU-only fallback: project Model A activations (no Model B available)."""
    mean_a_path = os.path.join(config.OVERALL_STATS_DIR,
                                f"layer_A{a_layer}_B0_mean_a.npy")
    if os.path.exists(mean_a_path):
        mean_a = np.load(mean_a_path)
    else:
        mean_a = np.zeros(hidden_dim, dtype=np.float32)

    projections = []
    token_meta = []

    for entry in corpus:
        input_id = entry["input_id"]
        positions_a = entry[config.CORPUS_POSITIONS_KEY_A]

        for pair_idx, pos_a in enumerate(positions_a):
            key = f"{input_id}_{pair_idx}"
            if key not in index:
                continue
            row = index[key]

            vec_a = store_a[row, a_layer, :].astype(np.float32)
            # Without Model B, use (vec_a - mean_a) as an approximation
            proj = (vec_a - mean_a) @ eigvecs

            projections.append(proj)
            token_meta.append({
                "input_id": input_id,
                "pair_idx": pair_idx,
                "token": f"pos_{pos_a}",
                "context": entry["raw_text"][:80],
                "category": entry["category"],
                "text_snippet": entry["raw_text"][:100],
            })

    _save_atlas_extremes(projections, token_meta)


def _save_atlas_extremes(projections, token_meta):
    """Sort projections per PC and save extreme tokens."""
    if not projections:
        print("  WARNING: No projections to save")
        return

    projections = np.array(projections)  # (N, top_k)
    top_k = projections.shape[1]
    n_extremes = config.ATLAS_EXTREMES

    print(f"\n  Saving atlas extremes for {top_k} PCs ({len(projections)} total tokens)...")

    for k in range(top_k):
        pc_values = projections[:, k]
        sorted_idx = np.argsort(pc_values)

        # Bottom extremes (most negative)
        bottom_idx = sorted_idx[:n_extremes]
        # Top extremes (most positive)
        top_idx = sorted_idx[-n_extremes:][::-1]

        extremes = {
            "pc_index": k,
            "total_tokens": len(pc_values),
            "value_range": [float(pc_values.min()), float(pc_values.max())],
            "std": float(pc_values.std()),
            "top_positive": [
                {
                    "value": float(pc_values[i]),
                    **token_meta[i],
                }
                for i in top_idx
            ],
            "top_negative": [
                {
                    "value": float(pc_values[i]),
                    **token_meta[i],
                }
                for i in bottom_idx
            ],
        }

        path = os.path.join(config.ATLAS_DIR, f"pc_{k}_extremes.json")
        with open(path, "w") as f:
            json.dump(extremes, f, indent=2)

    # Save projections array for further analysis
    np.save(os.path.join(config.ATLAS_DIR, "projections.npy"), projections.astype(np.float32))
    with open(os.path.join(config.ATLAS_DIR, "token_meta.json"), "w") as f:
        json.dump(token_meta, f)

    # Generate atlas_summary.md
    summary_lines = ["# Disagreement Atlas Summary\n"]
    summary_lines.append(f"Total tokens projected: {len(projections)}\n")
    summary_lines.append(f"Number of PCs: {top_k}\n\n")

    for k in range(top_k):
        pc_path = os.path.join(config.ATLAS_DIR, f"pc_{k}_extremes.json")
        with open(pc_path) as f:
            ext = json.load(f)

        summary_lines.append(f"## PC {k}\n")
        summary_lines.append(f"- Value range: [{ext['value_range'][0]:.3f}, {ext['value_range'][1]:.3f}]\n")
        summary_lines.append(f"- Std: {ext['std']:.3f}\n\n")

        summary_lines.append("**Top positive tokens:**\n")
        for item in ext["top_positive"][:5]:
            summary_lines.append(f"- `{item['token']}` (val={item['value']:.3f}, "
                                  f"cat={item['category']}) — \"{item.get('context', '')}\"\n")

        summary_lines.append("\n**Top negative tokens:**\n")
        for item in ext["top_negative"][:5]:
            summary_lines.append(f"- `{item['token']}` (val={item['value']:.3f}, "
                                  f"cat={item['category']}) — \"{item.get('context', '')}\"\n")
        summary_lines.append("\n---\n\n")

    with open(os.path.join(config.ATLAS_DIR, "atlas_summary.md"), "w", encoding="utf-8") as f:
        f.writelines(summary_lines)

    print(f"  Atlas saved to {config.ATLAS_DIR}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("PASS 3: EIGENDECOMPOSITION & PROJECTION")
    print("=" * 70)

    # Check Pass 2 output exists
    meta_path = os.path.join(config.RESULTS_DIR, "pass2_metadata.json")
    if not os.path.exists(meta_path):
        print(f"ERROR: Pass 2 metadata not found at {meta_path}")
        print("Run pass2_differential.py first.")
        sys.exit(1)

    # Step 1: Eigendecompose all layers
    atlas_key, rank_profile = eigendecompose_all()

    # Step 2: Eigendecompose per-category
    eigendecompose_categories()

    # Step 3: Project onto atlas PCs
    if atlas_key:
        project_atlas(atlas_key)
    else:
        print("\nSkipping atlas projection — no atlas layer found.")

    print("\nPass 3 complete.")


if __name__ == "__main__":
    main()
