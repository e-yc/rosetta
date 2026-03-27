#!/usr/bin/env python3
"""
Activation Differential Experiment — Analysis (Part 4)

Runs post-hoc analysis on Pass 2/3 outputs. CPU-only.
Computes rank profiles, category comparisons, and generates
summary statistics. Use visualize.py for plots.
"""

import json
import os
import sys

import numpy as np

import config


def load_metadata():
    meta_path = os.path.join(config.RESULTS_DIR, "pass2_metadata.json")
    if not os.path.exists(meta_path):
        print(f"ERROR: {meta_path} not found. Run pass2_differential.py first.")
        sys.exit(1)
    with open(meta_path) as f:
        return json.load(f)


def analyze_rank_profile():
    """Analyze the disagreement rank profile across layers."""
    print("\n--- RANK PROFILE ANALYSIS ---")

    rank_path = os.path.join(config.RESULTS_DIR, "rank_profile.json")
    if not os.path.exists(rank_path):
        print("  No rank profile found. Run pass3_project.py first.")
        return

    with open(rank_path) as f:
        rank_profile = json.load(f)

    print(f"\n  {'Layer Pair':>25} {'Rank95':>8} {'TotalVar':>12}")
    print("  " + "-" * 50)

    layers = []
    ranks = []
    variances = []
    for key in sorted(rank_profile.keys()):
        info = rank_profile[key]
        layers.append(info["a_layer"])
        ranks.append(info["rank_95"])
        variances.append(info["total_variance"])
        print(f"  {key:>25} {info['rank_95']:>8} {info['total_variance']:>12.2e}")

    if ranks:
        min_rank = min(ranks)
        max_rank = max(ranks)
        min_idx = ranks.index(min_rank)
        max_idx = ranks.index(max_rank)
        print(f"\n  Most structured (lowest rank): layer {layers[min_idx]} → rank {min_rank}")
        print(f"  Least structured (highest rank): layer {layers[max_idx]} → rank {max_rank}")
        print(f"  Rank range: {min_rank} — {max_rank}")

        # Check for U-shape
        mid = len(ranks) // 2
        early = np.mean(ranks[:mid // 2]) if mid > 0 else 0
        middle = np.mean(ranks[mid // 2:mid + mid // 2]) if mid > 0 else 0
        late = np.mean(ranks[mid + mid // 2:]) if mid > 0 else 0
        if early > middle and late > middle:
            print(f"  Shape: U-shaped (early={early:.0f}, middle={middle:.0f}, late={late:.0f})")
        elif early < middle < late:
            print(f"  Shape: Monotonically increasing")
        elif early > middle > late:
            print(f"  Shape: Monotonically decreasing")
        else:
            print(f"  Shape: No clear pattern (early={early:.0f}, middle={middle:.0f}, late={late:.0f})")


def analyze_categories():
    """Analyze per-category disagreement patterns."""
    print("\n--- CATEGORY ANALYSIS ---")

    cat_path = os.path.join(config.RESULTS_DIR, "category_analysis.json")
    if not os.path.exists(cat_path):
        print("  No category analysis found. Run pass3_project.py first.")
        return

    with open(cat_path) as f:
        cat_data = json.load(f)

    # Category rank profiles
    rank_profiles = cat_data.get("category_rank_profiles", {})
    if rank_profiles:
        print("\n  Per-category rank profiles:")
        categories = sorted(rank_profiles.keys())
        for cat in categories:
            layers_data = rank_profiles[cat]
            if layers_data:
                ranks = [v["rank_95"] for v in layers_data.values()]
                avg_rank = np.mean(ranks)
                print(f"    {cat:>20}: avg_rank_95 = {avg_rank:.0f} "
                      f"(range {min(ranks)}–{max(ranks)})")

    # Subspace overlap
    overlap_data = cat_data.get("subspace_overlap", {})
    if overlap_data:
        print("\n  Subspace overlap between categories (averaged across layers):")
        # Average across layers
        all_cats = None
        sum_matrix = None
        count = 0
        for layer_key, layer_data in overlap_data.items():
            cats = layer_data["categories"]
            matrix = np.array(layer_data["overlap_matrix"])
            if all_cats is None:
                all_cats = cats
                sum_matrix = np.zeros_like(matrix)
            sum_matrix += matrix
            count += 1

        if count > 0 and all_cats is not None:
            avg_matrix = sum_matrix / count
            # Print as table
            header = f"  {'':>18}" + "".join(f"{c:>14}" for c in all_cats)
            print(header)
            for i, cat in enumerate(all_cats):
                row = f"  {cat:>18}" + "".join(f"{avg_matrix[i, j]:>13.3f}" for j in range(len(all_cats)))
                print(row)


def analyze_atlas():
    """Summarize atlas findings."""
    print("\n--- ATLAS SUMMARY ---")

    atlas_meta_path = os.path.join(config.RESULTS_DIR, "atlas_meta.json")
    if not os.path.exists(atlas_meta_path):
        print("  No atlas metadata found. Run pass3_project.py first.")
        return

    with open(atlas_meta_path) as f:
        atlas_meta = json.load(f)

    print(f"  Atlas layer: {atlas_meta['atlas_layer_key']}")
    print(f"  Rank at 95%: {atlas_meta['atlas_rank_95']}")

    # Check variance explained
    var_path = os.path.join(config.ATLAS_DIR, "variance_explained.json")
    if os.path.exists(var_path):
        with open(var_path) as f:
            var_data = json.load(f)
        print(f"\n  Top PCs variance explained:")
        for item in var_data[:10]:
            print(f"    PC {item['pc']:>2}: {item['variance_pct']:.1f}% "
                  f"(cumulative: {item['cumulative_pct']:.1f}%)")

    # Check extremes
    pc0_path = os.path.join(config.ATLAS_DIR, "pc_0_extremes.json")
    if os.path.exists(pc0_path):
        with open(pc0_path) as f:
            pc0 = json.load(f)
        print(f"\n  PC 0 extremes preview:")
        print(f"    Value range: [{pc0['value_range'][0]:.3f}, {pc0['value_range'][1]:.3f}]")
        print(f"    Top positive:")
        for item in pc0["top_positive"][:3]:
            print(f"      {item['token']:>20} (val={item['value']:.3f}, cat={item['category']})")
        print(f"    Top negative:")
        for item in pc0["top_negative"][:3]:
            print(f"      {item['token']:>20} (val={item['value']:.3f}, cat={item['category']})")


def analyze_layer_correspondence():
    """Analyze the CKA layer correspondence."""
    print("\n--- LAYER CORRESPONDENCE ANALYSIS ---")

    cka_path = config.LAYER_CORRESPONDENCE_PATH
    mapping_path = config.LAYER_MAPPING_PATH

    if not os.path.exists(cka_path):
        print("  No layer correspondence found. Run pass2_differential.py first.")
        return

    cka = np.load(cka_path)
    with open(mapping_path) as f:
        mapping = json.load(f)

    print(f"  CKA matrix shape: {cka.shape}")
    print(f"  Value range: [{cka.min():.3f}, {cka.max():.3f}]")
    print(f"  Mean CKA: {cka.mean():.3f}")

    # Check if mapping is roughly linear
    a_layers = []
    b_layers = []
    for a_str, info in sorted(mapping.items(), key=lambda x: int(x[0])):
        a_layers.append(int(a_str))
        b_layers.append(info["model_b_layer"])

    if len(a_layers) > 2:
        correlation = np.corrcoef(a_layers, b_layers)[0, 1]
        print(f"\n  Layer correspondence correlation: {correlation:.3f}")
        if correlation > 0.95:
            print("  → Strongly linear mapping (layers correspond proportionally)")
        elif correlation > 0.8:
            print("  → Mostly linear with some deviations")
        else:
            print("  → Non-linear mapping (complex correspondence)")

        # Check for ideal linear mapping
        ideal_ratio = cka.shape[1] / cka.shape[0]  # GLM layers / Llama layers
        print(f"\n  Ideal linear ratio: {ideal_ratio:.2f}")
        print(f"  Actual mapping (Model A → Model B):")
        for a, b in zip(a_layers, b_layers):
            ideal_b = a * ideal_ratio
            deviation = b - ideal_b
            marker = " ***" if abs(deviation) > 3 else ""
            print(f"    Layer {a:>2} → {b:>2} (ideal: {ideal_b:.1f}, dev: {deviation:+.1f}){marker}")

    # Write correspondence_analysis.md
    lines = ["# Layer Correspondence Analysis\n\n"]
    lines.append(f"## CKA Matrix\n")
    lines.append(f"- Shape: {cka.shape[0]} (Model A) × {cka.shape[1]} (Model B)\n")
    lines.append(f"- Value range: [{cka.min():.3f}, {cka.max():.3f}]\n")
    lines.append(f"- Mean CKA: {cka.mean():.3f}\n\n")

    if len(a_layers) > 2:
        lines.append(f"## Correspondence Pattern\n")
        lines.append(f"- Linear correlation: {correlation:.3f}\n")
        lines.append(f"- The mapping is {'roughly linear' if correlation > 0.9 else 'non-linear'}\n\n")

    lines.append(f"## Layer Mapping Table\n\n")
    lines.append(f"| Model A Layer | Model B Layer | CKA Score |\n")
    lines.append(f"|:---:|:---:|:---:|\n")
    for a_str, info in sorted(mapping.items(), key=lambda x: int(x[0])):
        lines.append(f"| {a_str} | {info['model_b_layer']} | {info['cka_score']:.3f} |\n")

    with open(os.path.join(config.RESULTS_DIR, "correspondence_analysis.md"), "w") as f:
        f.writelines(lines)
    print(f"\n  Saved correspondence_analysis.md")


def main():
    print("=" * 70)
    print("ANALYSIS — Post-hoc Statistics and Summaries")
    print("=" * 70)

    analyze_layer_correspondence()
    analyze_rank_profile()
    analyze_categories()
    analyze_atlas()

    print("\n" + "=" * 70)
    print("Analysis complete. Run visualize.py for plots.")
    print("=" * 70)


if __name__ == "__main__":
    main()
