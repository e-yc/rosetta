#!/usr/bin/env python3
"""
Activation Differential Experiment — Visualization (Part 4)

Generates all plots from Pass 2/3 outputs. CPU-only.
"""

import json
import os
import sys

import numpy as np

import config

# Use non-interactive backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def load_metadata():
    path = os.path.join(config.RESULTS_DIR, "pass2_metadata.json")
    if not os.path.exists(path):
        print(f"ERROR: {path} not found.")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 4.1 Layer Correspondence Heatmap
# ---------------------------------------------------------------------------

def plot_layer_correspondence():
    """Plot the CKA layer correspondence heatmap."""
    cka_path = config.LAYER_CORRESPONDENCE_PATH
    mapping_path = config.LAYER_MAPPING_PATH

    if not os.path.exists(cka_path):
        print("  Skipping layer_correspondence_heatmap.png (no data)")
        return

    cka = np.load(cka_path)
    with open(mapping_path) as f:
        mapping = json.load(f)

    fig, ax = plt.subplots(figsize=(14, 10))
    im = ax.imshow(cka, cmap="viridis", aspect="auto", vmin=0, vmax=1)

    # Highlight optimal correspondence
    for a_str, info in mapping.items():
        a = int(a_str)
        b = info["model_b_layer"]
        ax.plot(b, a, "rx", markersize=6, markeredgewidth=1.5)

    ax.set_xlabel(f"Model B ({config.MODEL_B_NAME}) Layer", fontsize=12)
    ax.set_ylabel(f"Model A ({config.MODEL_A_NAME}) Layer", fontsize=12)
    ax.set_title("CKA Layer Correspondence\n(red × = optimal mapping)", fontsize=14)

    # Tick labels every 5 layers
    a_ticks = list(range(0, cka.shape[0], 5))
    b_ticks = list(range(0, cka.shape[1], 5))
    ax.set_xticks(b_ticks)
    ax.set_yticks(a_ticks)

    plt.colorbar(im, label="CKA Similarity", shrink=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, "layer_correspondence_heatmap.png"), dpi=150)
    plt.close()
    print("  Saved layer_correspondence_heatmap.png")


# ---------------------------------------------------------------------------
# 4.2 Rank Profile
# ---------------------------------------------------------------------------

def plot_rank_profile():
    """Plot disagreement rank vs layer depth."""
    rank_path = os.path.join(config.RESULTS_DIR, "rank_profile.json")
    if not os.path.exists(rank_path):
        print("  Skipping rank_profile.png (no data)")
        return

    with open(rank_path) as f:
        rank_profile = json.load(f)

    layers = []
    ranks = []
    variances = []
    for key in sorted(rank_profile.keys()):
        info = rank_profile[key]
        layers.append(info["a_layer"])
        ranks.append(info["rank_95"])
        variances.append(info["total_variance"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Rank profile
    ax1.plot(layers, ranks, "b-o", markersize=5, linewidth=2)
    min_idx = np.argmin(ranks)
    ax1.axvline(layers[min_idx], color="r", linestyle="--", alpha=0.5,
                label=f"Min rank at layer {layers[min_idx]}")
    ax1.set_ylabel("Rank at 95% Variance", fontsize=12)
    ax1.set_title("Disagreement Rank Profile", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Total variance
    ax2.plot(layers, variances, "g-o", markersize=5, linewidth=2)
    ax2.set_ylabel("Total Variance", fontsize=12)
    ax2.set_xlabel(f"Model A ({config.MODEL_A_NAME}) Layer", fontsize=12)
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, "rank_profile.png"), dpi=150)
    plt.close()
    print("  Saved rank_profile.png")


# ---------------------------------------------------------------------------
# 4.3 Eigenvalue Spectra
# ---------------------------------------------------------------------------

def plot_eigenvalue_spectra():
    """Plot eigenvalue spectra across layers."""
    spectra_path = os.path.join(config.RESULTS_DIR, "eigenvalue_spectra.json")
    if not os.path.exists(spectra_path):
        print("  Skipping eigenvalue_spectra.png (no data)")
        return

    with open(spectra_path) as f:
        spectra = json.load(f)

    fig, ax = plt.subplots(figsize=(14, 8))

    cmap = plt.cm.viridis(np.linspace(0, 1, len(spectra)))
    for i, (key, eigenvalues) in enumerate(sorted(spectra.items())):
        if eigenvalues:
            ax.semilogy(range(len(eigenvalues)), eigenvalues, color=cmap[i],
                        alpha=0.6, linewidth=1, label=key if i % 4 == 0 else None)

    ax.set_xlabel("Eigenvalue Index", fontsize=12)
    ax.set_ylabel("Eigenvalue (log scale)", fontsize=12)
    ax.set_title("Eigenvalue Spectra Across Layers", fontsize=14)
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, "eigenvalue_spectra.png"), dpi=150)
    plt.close()
    print("  Saved eigenvalue_spectra.png")


# ---------------------------------------------------------------------------
# 4.4 Variance Explained (Atlas)
# ---------------------------------------------------------------------------

def plot_variance_explained():
    """Bar chart of variance explained by top PCs at atlas layer."""
    var_path = os.path.join(config.ATLAS_DIR, "variance_explained.json")
    if not os.path.exists(var_path):
        print("  Skipping variance_explained.png (no data)")
        return

    with open(var_path) as f:
        var_data = json.load(f)

    pcs = [item["pc"] for item in var_data]
    pcts = [item["variance_pct"] for item in var_data]
    cum_pcts = [item["cumulative_pct"] for item in var_data]

    fig, ax1 = plt.subplots(figsize=(12, 6))
    bars = ax1.bar(pcs, pcts, color="steelblue", alpha=0.8, label="Individual")
    ax1.set_xlabel("Principal Component", fontsize=12)
    ax1.set_ylabel("Variance Explained (%)", fontsize=12, color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")

    ax2 = ax1.twinx()
    ax2.plot(pcs, cum_pcts, "r-o", markersize=4, linewidth=2, label="Cumulative")
    ax2.set_ylabel("Cumulative Variance (%)", fontsize=12, color="red")
    ax2.tick_params(axis="y", labelcolor="red")
    ax2.axhline(95, color="red", linestyle="--", alpha=0.3, label="95% threshold")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    ax1.set_title("Variance Explained by Top Principal Components (Atlas Layer)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, "variance_explained.png"), dpi=150)
    plt.close()
    print("  Saved variance_explained.png")


# ---------------------------------------------------------------------------
# 4.5 Category Overlap Heatmaps
# ---------------------------------------------------------------------------

def plot_category_overlap():
    """Plot subspace overlap heatmaps per layer."""
    cat_path = os.path.join(config.RESULTS_DIR, "category_analysis.json")
    if not os.path.exists(cat_path):
        print("  Skipping category_overlap_heatmap.png (no data)")
        return

    with open(cat_path) as f:
        cat_data = json.load(f)

    overlap_data = cat_data.get("subspace_overlap", {})
    if not overlap_data:
        print("  Skipping category_overlap_heatmap.png (no overlap data)")
        return

    n_layers = len(overlap_data)
    cols = min(4, n_layers)
    rows = (n_layers + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if n_layers == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)

    for idx, (layer_key, layer_data) in enumerate(sorted(overlap_data.items())):
        r, c = divmod(idx, cols)
        ax = axes[r, c]

        categories = layer_data["categories"]
        matrix = np.array(layer_data["overlap_matrix"])

        im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(categories)))
        ax.set_yticks(range(len(categories)))
        ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=6)
        ax.set_yticklabels(categories, fontsize=6)
        ax.set_title(layer_key, fontsize=8)

        for i in range(len(categories)):
            for j in range(len(categories)):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=5)

    # Hide empty subplots
    for idx in range(n_layers, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis("off")

    plt.suptitle("Cross-Category Subspace Overlap (top-10 PCs)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, "category_overlap_heatmap.png"), dpi=150)
    plt.close()
    print("  Saved category_overlap_heatmap.png")


# ---------------------------------------------------------------------------
# 4.6 Category Rank Profiles
# ---------------------------------------------------------------------------

def plot_category_rank_profiles():
    """Overlay rank profiles for each category."""
    cat_path = os.path.join(config.RESULTS_DIR, "category_analysis.json")
    if not os.path.exists(cat_path):
        print("  Skipping category_rank_profiles.png (no data)")
        return

    with open(cat_path) as f:
        cat_data = json.load(f)

    rank_profiles = cat_data.get("category_rank_profiles", {})
    if not rank_profiles:
        print("  Skipping category_rank_profiles.png (no rank data)")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(rank_profiles)))

    for idx, (cat, layers_data) in enumerate(sorted(rank_profiles.items())):
        if not layers_data:
            continue
        # Extract layer indices and ranks
        layer_ranks = []
        for key in sorted(layers_data.keys()):
            # Parse layer number from key like "layer_A0_B0"
            parts = key.split("_")
            a_layer = int(parts[1][1:])
            rank = layers_data[key]["rank_95"]
            layer_ranks.append((a_layer, rank))

        layer_ranks.sort()
        xs = [lr[0] for lr in layer_ranks]
        ys = [lr[1] for lr in layer_ranks]
        ax.plot(xs, ys, "-o", color=colors[idx], markersize=5, linewidth=2,
                label=cat, alpha=0.8)

    ax.set_xlabel(f"Model A ({config.MODEL_A_NAME}) Layer", fontsize=12)
    ax.set_ylabel("Rank at 95% Variance", fontsize=12)
    ax.set_title("Disagreement Rank by Input Category", fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, "category_rank_profiles.png"), dpi=150)
    plt.close()
    print("  Saved category_rank_profiles.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("VISUALIZATION — Generating Plots")
    print("=" * 70)

    plot_layer_correspondence()
    plot_rank_profile()
    plot_eigenvalue_spectra()
    plot_variance_explained()
    plot_category_overlap()
    plot_category_rank_profiles()

    print("\n" + "=" * 70)
    print(f"All plots saved to {config.RESULTS_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
