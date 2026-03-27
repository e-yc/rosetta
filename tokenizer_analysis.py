#!/usr/bin/env python3
"""
Tokenizer Translation Table — Main Analysis Script

Compares tokenizers across major open-source LLMs:
  Part 1: Vocabulary analysis
  Part 2: Segmentation divergence
  Part 3: Token alignment / translation table
  Part 4: Visualization

Outputs go to results/ directory.
"""

import json
import os
import sys
import shutil
import unicodedata
from itertools import combinations
from collections import defaultdict

import numpy as np

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY = [
    ("llama4_scout", "meta-llama/Llama-4-Scout-17B-16E-Instruct",
     ["meta-llama/Llama-4-Scout-17B-16E", "meta-llama/Llama-4-Maverick-17B-128E-Instruct"]),
    ("llama3_8b", "meta-llama/Meta-Llama-3.1-8B-Instruct",
     ["meta-llama/Meta-Llama-3.1-8B", "meta-llama/Llama-3.2-3B-Instruct"]),
    ("deepseek_v3", "deepseek-ai/DeepSeek-V3",
     ["deepseek-ai/DeepSeek-V2.5", "deepseek-ai/deepseek-llm-7b-base"]),
    ("qwen3_32b", "Qwen/Qwen3-32B",
     ["Qwen/Qwen3-8B", "Qwen/Qwen2.5-32B-Instruct", "Qwen/Qwen2.5-7B-Instruct"]),
    ("glm4_9b", "THUDM/glm-4-9b",
     ["THUDM/glm-4-9b-chat", "THUDM/chatglm3-6b"]),
    ("kimi_k2", "moonshotai/Kimi-K2-Instruct",
     ["moonshotai/Kimi-K2-Base", "moonshotai/Kimi-K1.5"]),
    ("mistral_large", "mistralai/Mistral-Large-Instruct-2411",
     ["mistralai/Mistral-Large-Instruct-2407", "mistralai/Mistral-7B-Instruct-v0.3"]),
    ("gemma3_27b", "google/gemma-3-27b-it",
     ["google/gemma-3-12b-it", "google/gemma-2-9b-it", "google/gemma-2-2b-it"]),
    ("gpt_oss_20b", "openai/gpt-oss-20b",
     ["openai/gpt-oss-20b-preview"]),
]

# ---------------------------------------------------------------------------
# Tokenizer loading
# ---------------------------------------------------------------------------

def load_tokenizers():
    """Load tokenizers for all models, using fallbacks as needed."""
    from transformers import AutoTokenizer

    loaded = {}
    for short_name, primary_id, fallbacks in MODEL_REGISTRY:
        candidates = [primary_id] + fallbacks
        for model_id in candidates:
            try:
                print(f"  Loading {short_name} from {model_id} ...", end=" ", flush=True)
                tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
                loaded[short_name] = tok
                print("OK")
                break
            except Exception as e:
                err = str(e)
                if len(err) > 120:
                    err = err[:120] + "..."
                print(f"FAILED ({err})")
        else:
            print(f"  *** Skipping {short_name}: all candidates failed ***")

    return loaded


# ---------------------------------------------------------------------------
# Part 1 — Vocabulary Analysis
# ---------------------------------------------------------------------------

def _get_vocab_set(tokenizer) -> set[str]:
    """Return set of all token strings in the tokenizer's vocabulary."""
    vocab = tokenizer.get_vocab()
    result = set()
    for k in vocab.keys():
        if isinstance(k, bytes):
            try:
                result.add(k.decode("utf-8", errors="replace"))
            except Exception:
                result.add(str(k))
        else:
            result.add(str(k))
    return result


def _categorize_token(tok) -> str:
    """Classify a token string into a category."""
    if isinstance(tok, bytes):
        try:
            tok = tok.decode("utf-8", errors="replace")
        except Exception:
            tok = str(tok)
    tok = str(tok)

    # Strip common BPE prefixes for analysis
    stripped = tok.lstrip("Ġ▁Ã").lstrip("##").strip()

    if not stripped:
        return "subword_fragments"

    if len(stripped) == 1 and stripped.isascii():
        return "single_chars"

    if stripped.isdigit() or (stripped and all(c.isdigit() or c in ".,+-eExX" for c in stripped)):
        return "numeric"

    # Check non-Latin
    has_non_latin = False
    for c in stripped:
        if c.isalpha():
            try:
                script_name = unicodedata.name(c, "")
            except (ValueError, TypeError):
                continue
            if any(kw in script_name for kw in ["CJK", "HIRAGANA", "KATAKANA", "HANGUL",
                                                  "CYRILLIC", "ARABIC", "DEVANAGARI",
                                                  "THAI", "BENGALI", "TAMIL", "HEBREW",
                                                  "GREEK"]):
                has_non_latin = True
                break
    if has_non_latin:
        return "non_latin"

    # Subword fragments: starts with BPE marker or not a complete word
    if tok.startswith(("Ġ", "▁", "##", "Ã")) or (not stripped[0].isalpha() if stripped else True):
        return "subword_fragments"

    # Check if it looks like a common English word (all alpha, reasonable length)
    if stripped.isalpha() and stripped.isascii() and 2 <= len(stripped) <= 20:
        return "common_english"

    return "subword_fragments"


def vocab_analysis(tokenizers: dict) -> dict:
    """Part 1: Vocabulary overlap analysis."""
    print("\n" + "=" * 70)
    print("PART 1: VOCABULARY ANALYSIS")
    print("=" * 70)

    vocabs = {}
    models_info = {}
    for name, tok in tokenizers.items():
        v = _get_vocab_set(tok)
        vocabs[name] = v
        tok_type = getattr(tok, "__class__", type(tok)).__name__
        models_info[name] = {
            "vocab_size": len(v),
            "tokenizer_type": tok_type,
        }
        print(f"  {name}: {len(v):,} tokens ({tok_type})")

    # Pairwise analysis
    names = sorted(tokenizers.keys())
    pairwise = {}

    # Print header for overlap matrix
    print(f"\n{'Pairwise Vocabulary Overlap (%)':^70}")
    header = f"{'':>16}" + "".join(f"{n:>14}" for n in names)
    print(header)
    print("-" * len(header))

    overlap_matrix = {}
    for a in names:
        row_vals = {}
        for b in names:
            if a == b:
                row_vals[b] = 100.0
                continue
            key = f"{a}_vs_{b}"
            rkey = f"{b}_vs_{a}"
            if rkey in pairwise:
                row_vals[b] = pairwise[rkey]["overlap_pct"] * 100
                continue

            shared = vocabs[a] & vocabs[b]
            union = vocabs[a] | vocabs[b]
            unique_a = vocabs[a] - vocabs[b]
            unique_b = vocabs[b] - vocabs[a]
            overlap_pct = len(shared) / len(union) if union else 0

            # Category breakdown
            cat_stats = {}
            for cat_name in ["single_chars", "common_english", "subword_fragments",
                             "numeric", "non_latin"]:
                cat_a = {t for t in vocabs[a] if _categorize_token(t) == cat_name}
                cat_b = {t for t in vocabs[b] if _categorize_token(t) == cat_name}
                cat_shared = cat_a & cat_b
                cat_union = cat_a | cat_b
                cat_stats[cat_name] = {
                    "shared": len(cat_shared),
                    "total_union": len(cat_union),
                    "pct": round(len(cat_shared) / len(cat_union), 4) if cat_union else 0,
                }

            pairwise[key] = {
                "shared_tokens": len(shared),
                "unique_to_" + a: len(unique_a),
                "unique_to_" + b: len(unique_b),
                "overlap_pct": round(overlap_pct, 4),
                "overlap_by_category": cat_stats,
            }
            row_vals[b] = overlap_pct * 100

        overlap_matrix[a] = row_vals
        row_str = f"{a:>16}" + "".join(f"{row_vals[b]:>13.1f}%" for b in names)
        print(row_str)

    result = {"models": models_info, "pairwise": pairwise}
    return result, overlap_matrix


# ---------------------------------------------------------------------------
# Part 2 — Segmentation Divergence
# ---------------------------------------------------------------------------

def _char_to_token_map(tokenizer, text: str) -> tuple[list[str], list[int]]:
    """Return (token_strings, char_to_token) where char_to_token[i] = token index for char i."""
    try:
        enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        offsets = enc["offset_mapping"]
        ids = enc["input_ids"]
    except Exception:
        # Fallback for slow tokenizers
        from tokenizer_translation import _compute_offsets_fallback
        ids, tok_strs, spans = _compute_offsets_fallback(tokenizer, text)
        tokens = []
        valid_offsets = []
        for t, (s, e) in zip(tok_strs, spans):
            if s == e:
                continue
            tokens.append(t)
            valid_offsets.append((s, e))
        char_map = [-1] * len(text)
        for idx, (s, e) in enumerate(valid_offsets):
            for ci in range(s, min(e, len(text))):
                char_map[ci] = idx
        return tokens, char_map

    tokens = []
    valid_offsets = []
    for tid, (s, e) in zip(ids, offsets):
        if s == e:
            continue
        tokens.append(tokenizer.convert_ids_to_tokens([tid])[0])
        valid_offsets.append((s, e))

    char_map = [-1] * len(text)
    for idx, (s, e) in enumerate(valid_offsets):
        for ci in range(s, min(e, len(text))):
            char_map[ci] = idx

    return tokens, char_map


def _token_boundaries(char_map: list[int]) -> set[int]:
    """Return set of character positions where token boundaries occur."""
    boundaries = set()
    prev = -2
    for i, t in enumerate(char_map):
        if t != prev and t >= 0:
            boundaries.add(i)
        prev = t
    # Add end boundaries
    for i in range(len(char_map) - 1):
        if char_map[i] >= 0 and (char_map[i + 1] != char_map[i]):
            boundaries.add(i + 1)
    if char_map and char_map[-1] >= 0:
        boundaries.add(len(char_map))
    return boundaries


def segmentation_analysis(tokenizers: dict, corpus: list[tuple[str, str]]) -> dict:
    """Part 2: Segmentation divergence analysis."""
    print("\n" + "=" * 70)
    print("PART 2: SEGMENTATION DIVERGENCE")
    print("=" * 70)

    names = sorted(tokenizers.keys())
    categories = sorted(set(cat for _, cat in corpus))
    n_by_cat = defaultdict(int)
    for _, cat in corpus:
        n_by_cat[cat] += 1

    # Precompute tokenizations
    print("  Tokenizing corpus across all models...")
    tok_data = {}  # (model, sent_idx) -> (tokens, char_map)
    for name, tok in tokenizers.items():
        print(f"    {name}...", flush=True)
        for i, (sent, _) in enumerate(corpus):
            try:
                tokens, char_map = _char_to_token_map(tok, sent)
                tok_data[(name, i)] = (tokens, char_map)
            except Exception:
                tok_data[(name, i)] = ([], [-1] * len(sent))

    # Pairwise analysis
    pairwise = {}
    for a, b in combinations(names, 2):
        key = f"{a}_vs_{b}"
        by_cat = defaultdict(lambda: {
            "ratios": [], "word_agree": [], "word_diff1": [], "word_diff2p": [],
            "boundary_scores": [],
        })

        for i, (sent, cat) in enumerate(corpus):
            toks_a, cmap_a = tok_data[(a, i)]
            toks_b, cmap_b = tok_data[(b, i)]

            na = len(toks_a)
            nb = len(toks_b)
            if na == 0 or nb == 0:
                continue

            # Token count ratio
            ratio = max(na, nb) / min(na, nb)
            by_cat[cat]["ratios"].append(ratio)

            # Word-level divergence
            words = sent.split()
            agree = diff1 = diff2p = 0
            word_start = 0
            for w in words:
                idx = sent.find(w, word_start)
                if idx == -1:
                    continue
                wend = idx + len(w)
                word_start = wend

                # Count tokens covering this word span
                toks_a_for_word = set()
                toks_b_for_word = set()
                for ci in range(idx, min(wend, len(cmap_a))):
                    if cmap_a[ci] >= 0:
                        toks_a_for_word.add(cmap_a[ci])
                for ci in range(idx, min(wend, len(cmap_b))):
                    if cmap_b[ci] >= 0:
                        toks_b_for_word.add(cmap_b[ci])

                na_w = len(toks_a_for_word)
                nb_w = len(toks_b_for_word)
                diff = abs(na_w - nb_w)
                if diff == 0:
                    agree += 1
                elif diff == 1:
                    diff1 += 1
                else:
                    diff2p += 1

            total_words = agree + diff1 + diff2p
            if total_words > 0:
                by_cat[cat]["word_agree"].append(agree / total_words)
                by_cat[cat]["word_diff1"].append(diff1 / total_words)
                by_cat[cat]["word_diff2p"].append(diff2p / total_words)

            # Boundary alignment
            bounds_a = _token_boundaries(cmap_a)
            bounds_b = _token_boundaries(cmap_b)
            if bounds_a and bounds_b:
                shared_bounds = bounds_a & bounds_b
                union_bounds = bounds_a | bounds_b
                score = len(shared_bounds) / len(union_bounds) if union_bounds else 1.0
                by_cat[cat]["boundary_scores"].append(score)

        # Aggregate
        cat_results = {}
        all_ratios = []
        for cat in categories:
            d = by_cat[cat]
            all_ratios.extend(d["ratios"])
            cat_results[cat] = {
                "avg_token_count_ratio": round(float(np.mean(d["ratios"])), 3) if d["ratios"] else None,
                "word_agreement_pct": round(float(np.mean(d["word_agree"])), 3) if d["word_agree"] else None,
                "word_diff_1_pct": round(float(np.mean(d["word_diff1"])), 3) if d["word_diff1"] else None,
                "word_diff_2plus_pct": round(float(np.mean(d["word_diff2p"])), 3) if d["word_diff2p"] else None,
                "boundary_alignment_score": round(float(np.mean(d["boundary_scores"])), 3) if d["boundary_scores"] else None,
            }

        pairwise[key] = {
            "avg_token_count_ratio": round(float(np.mean(all_ratios)), 3) if all_ratios else None,
            "by_category": cat_results,
        }

    # Print summary
    print(f"\n{'Boundary Alignment Scores':^70}")
    pairs = list(pairwise.keys())
    print(f"{'Pair':>30}", end="")
    for cat in categories:
        print(f"{cat:>14}", end="")
    print()
    print("-" * (30 + 14 * len(categories)))
    for pair_key in pairs:
        cats = pairwise[pair_key]["by_category"]
        print(f"{pair_key:>30}", end="")
        for cat in categories:
            val = cats.get(cat, {}).get("boundary_alignment_score")
            print(f"{val:>13.3f}" if val is not None else f"{'N/A':>14}", end="")
        print()

    result = {
        "corpus_stats": {
            "total_sentences": len(corpus),
            "categories": dict(n_by_cat),
        },
        "pairwise": pairwise,
    }
    return result


# ---------------------------------------------------------------------------
# Part 3 — Token Alignment / Translation Table
# ---------------------------------------------------------------------------

def alignment_analysis(tokenizers: dict, corpus: list[tuple[str, str]]) -> dict:
    """Part 3: Alignment / translation table analysis."""
    from tokenizer_translation import align_tokens, classify_alignment, get_activation_pairs

    print("\n" + "=" * 70)
    print("PART 3: TOKEN ALIGNMENT / TRANSLATION TABLE")
    print("=" * 70)

    names = sorted(tokenizers.keys())
    categories = sorted(set(cat for _, cat in corpus))

    pairwise = {}
    for a, b in combinations(names, 2):
        key = f"{a}_vs_{b}"
        print(f"  Aligning {a} vs {b}...")

        by_cat = defaultdict(lambda: {
            "bucket_1": [], "bucket_2": [], "bucket_3": [],
            "clean_points": [], "unit_sizes_a": [], "unit_sizes_b": [],
        })

        for sent, cat in corpus:
            try:
                alignment = align_tokens(
                    sent, tokenizers[a], tokenizers[b], name_a=a, name_b=b
                )
                buckets = classify_alignment(alignment)
                by_cat[cat]["bucket_1"].append(buckets["bucket_1_pct"])
                by_cat[cat]["bucket_2"].append(buckets["bucket_2_pct"])
                by_cat[cat]["bucket_3"].append(buckets["bucket_3_pct"])
                by_cat[cat]["clean_points"].append(buckets["bucket_1_count"])

                for link in alignment.alignment:
                    by_cat[cat]["unit_sizes_a"].append(len(link.model_a_indices))
                    by_cat[cat]["unit_sizes_b"].append(len(link.model_b_indices))
            except Exception as e:
                pass  # Skip problematic sentences

        # Aggregate
        cat_results = {}
        all_b1, all_b2, all_b3, all_clean = [], [], [], []
        for cat in categories:
            d = by_cat[cat]
            all_b1.extend(d["bucket_1"])
            all_b2.extend(d["bucket_2"])
            all_b3.extend(d["bucket_3"])
            all_clean.extend(d["clean_points"])
            cat_results[cat] = {
                "bucket_1_pct": round(float(np.mean(d["bucket_1"])), 4) if d["bucket_1"] else None,
                "bucket_2_pct": round(float(np.mean(d["bucket_2"])), 4) if d["bucket_2"] else None,
                "bucket_3_pct": round(float(np.mean(d["bucket_3"])), 4) if d["bucket_3"] else None,
                "avg_clean_points_per_sentence": round(float(np.mean(d["clean_points"])), 2) if d["clean_points"] else None,
                "avg_unit_size_model_a": round(float(np.mean(d["unit_sizes_a"])), 3) if d["unit_sizes_a"] else None,
                "avg_unit_size_model_b": round(float(np.mean(d["unit_sizes_b"])), 3) if d["unit_sizes_b"] else None,
            }

        pairwise[key] = {
            "overall": {
                "bucket_1_pct": round(float(np.mean(all_b1)), 4) if all_b1 else None,
                "bucket_2_pct": round(float(np.mean(all_b2)), 4) if all_b2 else None,
                "bucket_3_pct": round(float(np.mean(all_b3)), 4) if all_b3 else None,
                "avg_clean_points_per_sentence": round(float(np.mean(all_clean)), 2) if all_clean else None,
            },
            "by_category": cat_results,
        }

    # Print summary
    print(f"\n{'Bucket Distribution (Overall)':^70}")
    print(f"{'Pair':>30} {'Bucket1%':>10} {'Bucket2%':>10} {'Bucket3%':>10} {'CleanPts':>10}")
    print("-" * 70)
    for key, data in pairwise.items():
        o = data["overall"]
        b1 = f"{o['bucket_1_pct']*100:.1f}" if o["bucket_1_pct"] is not None else "N/A"
        b2 = f"{o['bucket_2_pct']*100:.1f}" if o["bucket_2_pct"] is not None else "N/A"
        b3 = f"{o['bucket_3_pct']*100:.1f}" if o["bucket_3_pct"] is not None else "N/A"
        cp = f"{o['avg_clean_points_per_sentence']:.1f}" if o["avg_clean_points_per_sentence"] is not None else "N/A"
        print(f"{key:>30} {b1:>10} {b2:>10} {b3:>10} {cp:>10}")

    return pairwise


# ---------------------------------------------------------------------------
# Part 4 — Visualization
# ---------------------------------------------------------------------------

def visualize(tokenizers, overlap_matrix, seg_data, align_data, corpus, results_dir):
    """Part 4: Generate all visualizations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    print("\n" + "=" * 70)
    print("PART 4: VISUALIZATION")
    print("=" * 70)

    names = sorted(tokenizers.keys())
    categories = sorted(set(cat for _, cat in corpus))

    # 4.1 Vocabulary Overlap Heatmap
    print("  Generating vocab_overlap_heatmap.png ...")
    fig, ax = plt.subplots(figsize=(max(10, len(names)), max(8, len(names) * 0.8)))
    matrix = np.zeros((len(names), len(names)))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            matrix[i, j] = overlap_matrix.get(a, {}).get(b, 0)

    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=100)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", fontsize=7,
                    color="white" if matrix[i, j] > 50 else "black")
    plt.colorbar(im, label="Overlap %")
    ax.set_title("Pairwise Vocabulary Overlap (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "vocab_overlap_heatmap.png"), dpi=150)
    plt.close()

    # 4.2 Boundary Alignment Heatmap
    print("  Generating boundary_alignment_heatmap.png ...")
    n_cats = len(categories)
    n_pairs = len(list(combinations(names, 2)))
    pair_labels = [f"{a} vs {b}" for a, b in combinations(names, 2)]

    if n_pairs > 0 and n_cats > 0:
        fig, ax = plt.subplots(figsize=(max(12, n_cats * 1.5), max(6, n_pairs * 0.4)))
        ba_matrix = np.zeros((n_pairs, n_cats))
        for pi, (a, b) in enumerate(combinations(names, 2)):
            key = f"{a}_vs_{b}"
            if key in seg_data.get("pairwise", {}):
                cats = seg_data["pairwise"][key].get("by_category", {})
                for ci, cat in enumerate(categories):
                    val = cats.get(cat, {}).get("boundary_alignment_score")
                    ba_matrix[pi, ci] = val if val is not None else 0

        im = ax.imshow(ba_matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(n_cats))
        ax.set_yticks(range(n_pairs))
        ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(pair_labels, fontsize=6)
        for i in range(n_pairs):
            for j in range(n_cats):
                ax.text(j, i, f"{ba_matrix[i, j]:.2f}", ha="center", va="center", fontsize=5)
        plt.colorbar(im, label="Boundary Alignment Score")
        ax.set_title("Boundary Alignment Scores by Category")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "boundary_alignment_heatmap.png"), dpi=150)
        plt.close()

    # 4.3 Bucket Distribution Bar Chart
    print("  Generating bucket_distribution.png ...")
    if align_data and n_pairs > 0:
        fig, axes = plt.subplots(1, n_cats, figsize=(max(16, n_cats * 2.5), max(6, n_pairs * 0.3)),
                                  sharey=True)
        if n_cats == 1:
            axes = [axes]

        for ci, cat in enumerate(categories):
            ax = axes[ci]
            b1_vals, b2_vals, b3_vals = [], [], []
            labels = []
            for a, b_model in combinations(names, 2):
                key = f"{a}_vs_{b_model}"
                if key in align_data:
                    cat_data = align_data[key].get("by_category", {}).get(cat, {})
                    b1 = (cat_data.get("bucket_1_pct") or 0) * 100
                    b2 = (cat_data.get("bucket_2_pct") or 0) * 100
                    b3 = (cat_data.get("bucket_3_pct") or 0) * 100
                else:
                    b1 = b2 = b3 = 0
                b1_vals.append(b1)
                b2_vals.append(b2)
                b3_vals.append(b3)
                labels.append(f"{a}\nvs\n{b_model}")

            y = np.arange(len(labels))
            ax.barh(y, b1_vals, color="#2ecc71", label="Bucket 1 (exact)")
            ax.barh(y, b2_vals, left=b1_vals, color="#f39c12", label="Bucket 2 (minor)")
            lefts = [a + b for a, b in zip(b1_vals, b2_vals)]
            ax.barh(y, b3_vals, left=lefts, color="#e74c3c", label="Bucket 3 (major)")
            ax.set_yticks(y)
            ax.set_yticklabels(labels, fontsize=5)
            ax.set_title(cat, fontsize=8)
            ax.set_xlim(0, 100)
            if ci == 0:
                ax.legend(fontsize=6, loc="lower right")

        plt.suptitle("Bucket Distribution by Category and Model Pair", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "bucket_distribution.png"), dpi=150)
        plt.close()

    # 4.4 Example Alignment Visualization
    print("  Generating alignment_examples.png ...")
    from test_corpus import get_example_sentences
    from tokenizer_translation import align_tokens

    examples = get_example_sentences()
    # Pick first two models for visualization
    if len(names) >= 2:
        ma, mb = names[0], names[1]
        fig, axes = plt.subplots(len(examples), 1, figsize=(16, 4 * len(examples)))
        if len(examples) == 1:
            axes = [axes]

        colors = plt.cm.Set3(np.linspace(0, 1, 20))

        for idx, (sent, cat) in enumerate(examples):
            ax = axes[idx]
            ax.set_xlim(0, max(len(sent), 1))
            ax.set_ylim(-0.5, 3.5)
            ax.set_title(f"Category: {cat}", fontsize=10, fontweight="bold")

            try:
                alignment = align_tokens(sent, tokenizers[ma], tokenizers[mb],
                                          name_a=ma, name_b=mb)

                # Draw original text
                ax.text(0, 3, sent[:80] + ("..." if len(sent) > 80 else ""),
                        fontsize=7, fontfamily="monospace", va="center")

                # Draw model A tokens
                x_pos = 0
                for ti, (tok, (s, e)) in enumerate(zip(alignment.model_a_tokens,
                                                        alignment.model_a_char_spans)):
                    width = e - s
                    color = colors[ti % len(colors)]
                    ax.barh(2, width, left=s, height=0.6, color=color, edgecolor="gray",
                            linewidth=0.5, alpha=0.7)
                    if width > 1:
                        display = tok[:int(width)] if len(tok) > width else tok
                        ax.text(s + width / 2, 2, display, ha="center", va="center",
                                fontsize=5, clip_on=True)

                # Draw model B tokens
                for ti, (tok, (s, e)) in enumerate(zip(alignment.model_b_tokens,
                                                        alignment.model_b_char_spans)):
                    width = e - s
                    color = colors[ti % len(colors)]
                    ax.barh(1, width, left=s, height=0.6, color=color, edgecolor="gray",
                            linewidth=0.5, alpha=0.7)
                    if width > 1:
                        display = tok[:int(width)] if len(tok) > width else tok
                        ax.text(s + width / 2, 1, display, ha="center", va="center",
                                fontsize=5, clip_on=True)

                # Draw alignment lines
                for link in alignment.alignment:
                    for ai in link.model_a_indices:
                        for bi in link.model_b_indices:
                            if ai < len(alignment.model_a_char_spans) and \
                               bi < len(alignment.model_b_char_spans):
                                a_mid = sum(alignment.model_a_char_spans[ai]) / 2
                                b_mid = sum(alignment.model_b_char_spans[bi]) / 2
                                ax.plot([a_mid, b_mid], [1.7, 1.3], "k-",
                                        alpha=0.2, linewidth=0.5)

                ax.text(-0.5, 2, ma, fontsize=7, ha="right", va="center", fontweight="bold")
                ax.text(-0.5, 1, mb, fontsize=7, ha="right", va="center", fontweight="bold")

            except Exception as e:
                ax.text(0.5, 1.5, f"Error: {e}", transform=ax.transAxes, ha="center")

            ax.set_yticks([])
            ax.set_xticks([])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["left"].set_visible(False)

        plt.suptitle(f"Token Alignment Examples: {ma} vs {mb}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "alignment_examples.png"), dpi=150)
        plt.close()

    print("  All visualizations saved.")


# ---------------------------------------------------------------------------
# Final Summary
# ---------------------------------------------------------------------------

def print_summary(tokenizers, align_data, seg_data):
    """Print key findings."""
    print("\n" + "=" * 70)
    print("SUMMARY OF KEY FINDINGS")
    print("=" * 70)

    names = sorted(tokenizers.keys())

    # 1. Highest/lowest tokenizer compatibility
    if align_data:
        best_pair = None
        best_score = -1
        worst_pair = None
        worst_score = 2
        for key, data in align_data.items():
            b1 = data.get("overall", {}).get("bucket_1_pct")
            if b1 is not None:
                if b1 > best_score:
                    best_score = b1
                    best_pair = key
                if b1 < worst_score:
                    worst_score = b1
                    worst_pair = key

        if best_pair:
            print(f"\n1. HIGHEST TOKENIZER COMPATIBILITY:")
            print(f"   {best_pair}: {best_score*100:.1f}% exact-match (bucket 1) alignment units")
        if worst_pair:
            print(f"\n   LOWEST TOKENIZER COMPATIBILITY:")
            print(f"   {worst_pair}: {worst_score*100:.1f}% exact-match (bucket 1) alignment units")

    # 2. Most divergent corpus categories
    if seg_data and "pairwise" in seg_data:
        cat_scores = defaultdict(list)
        for key, data in seg_data["pairwise"].items():
            for cat, stats in data.get("by_category", {}).items():
                score = stats.get("boundary_alignment_score")
                if score is not None:
                    cat_scores[cat].append(score)

        if cat_scores:
            print(f"\n2. CORPUS CATEGORY DIVERGENCE (avg boundary alignment across all pairs):")
            sorted_cats = sorted(cat_scores.items(), key=lambda x: np.mean(x[1]))
            for cat, scores in sorted_cats:
                print(f"   {cat:>20}: {np.mean(scores):.3f}")
            print(f"   Most divergent: {sorted_cats[0][0]}")
            print(f"   Most aligned:   {sorted_cats[-1][0]}")

    # 3. Clean alignment percentages
    if align_data:
        print(f"\n3. BUCKET 1 (CLEAN) ALIGNMENT PERCENTAGES:")
        for key, data in sorted(align_data.items()):
            b1 = data.get("overall", {}).get("bucket_1_pct")
            if b1 is not None:
                print(f"   {key:>30}: {b1*100:.1f}%")

    # 4. Recommendation
    if align_data:
        print(f"\n4. RECOMMENDATION FOR ACTIVATION DIFFERENTIAL ANALYSIS:")
        ranked = sorted(
            [(k, v["overall"]["bucket_1_pct"])
             for k, v in align_data.items()
             if v.get("overall", {}).get("bucket_1_pct") is not None],
            key=lambda x: -x[1]
        )
        if ranked:
            print(f"   Best pairs for cross-model activation comparison (by bucket 1 %):")
            for i, (pair, score) in enumerate(ranked[:5], 1):
                print(f"   {i}. {pair}: {score*100:.1f}% clean alignment points")
            print(f"\n   These pairs maximize the number of directly comparable token positions,")
            print(f"   minimizing the need for pooling/averaging across misaligned token groups.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 70)
    print("TOKENIZER TRANSLATION TABLE — CROSS-MODEL ANALYSIS")
    print("=" * 70)

    # Load tokenizers
    print("\nLoading tokenizers...")
    tokenizers = load_tokenizers()

    if len(tokenizers) < 2:
        print("\nERROR: Need at least 2 tokenizers to compare. Exiting.")
        sys.exit(1)

    print(f"\nSuccessfully loaded {len(tokenizers)} tokenizers: {', '.join(sorted(tokenizers.keys()))}")

    # Load corpus
    from test_corpus import get_corpus
    corpus = get_corpus()
    print(f"Corpus: {len(corpus)} sentences")

    # Part 1: Vocabulary Analysis
    vocab_result, overlap_matrix = vocab_analysis(tokenizers)
    with open(os.path.join(results_dir, "vocab_analysis.json"), "w") as f:
        json.dump(vocab_result, f, indent=2)
    print(f"\n  Saved vocab_analysis.json")

    # Part 2: Segmentation Divergence
    seg_result = segmentation_analysis(tokenizers, corpus)
    with open(os.path.join(results_dir, "segmentation_analysis.json"), "w") as f:
        json.dump(seg_result, f, indent=2)
    print(f"\n  Saved segmentation_analysis.json")

    # Part 3: Alignment Analysis
    align_result = alignment_analysis(tokenizers, corpus)
    with open(os.path.join(results_dir, "alignment_stats.json"), "w") as f:
        json.dump(align_result, f, indent=2)
    print(f"\n  Saved alignment_stats.json")

    # Copy the translation module to results
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tokenizer_translation.py")
    dst = os.path.join(results_dir, "tokenizer_translation.py")
    shutil.copy2(src, dst)
    print(f"  Copied tokenizer_translation.py to results/")

    # Part 4: Visualization
    visualize(tokenizers, overlap_matrix, seg_result, align_result, corpus, results_dir)

    # Final summary
    print_summary(tokenizers, align_result, seg_result)

    print(f"\n{'=' * 70}")
    print(f"All outputs saved to: {results_dir}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
