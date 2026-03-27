# rosetta
Tokenizer and activation behavior research on OSS models

All experiments run locally on: Gigabyte RTX 5090 (32GB VRAM)

---

## Setup

```bash
pip install torch transformers accelerate numpy scipy matplotlib seaborn scikit-learn datasets tiktoken sentencepiece --break-system-packages
```

HuggingFace auth required for gated models (Llama, Gemma):
```bash
huggingface-cli login
```

---

## Phase 1: Tokenizer Translation Table (complete)

Compares tokenizers across 9 open-source LLMs. Produces vocabulary overlap analysis, segmentation divergence metrics, and a character-span alignment module that maps token positions between any two models.

```bash
cd rosetta
python tokenizer_analysis.py
```

**Outputs** in `results/`:
- `vocab_analysis.json` — pairwise vocabulary overlap stats
- `segmentation_analysis.json` — boundary alignment scores per corpus category
- `alignment_stats.json` — bucket distribution (exact match / minor split / major divergence)
- `tokenizer_translation.py` — reusable alignment module
- 4 PNG visualizations (heatmaps, bar charts, alignment examples)

**Key finding:** DeepSeek-V3 vs GLM-4 have 100% bucket-1 alignment. Multilingual text is the most divergent category; simple English the most aligned.

---

## Phase 2 Results: Activation Differential Experiment (complete)

Compared hidden-state activations of GLM-4 9B and Llama 3.1 8B across 920,060 aligned token pairs from 10,000 inputs spanning six categories. Full report with visualizations: [`activation_experiment/results/report.md`](activation_experiment/results/report.md).

### Layer correspondence: Llama layer 15 is a universal attractor

CKA analysis (41x33 matrix) reveals that **21 of GLM's 41 layers** (layers 5-25) all map to Llama layer 15, with CKA scores of 0.943--0.951. Peak CKA is 0.973 (GLM-4 vs Llama-23). GLM distributes across 21 layers what Llama compresses into one.

| GLM layers | Best Llama match | CKA | Pattern |
|:---:|:---:|:---:|:---|
| 0 | 0 | 0.879 | Embeddings correspond well |
| 2-4 | 23-30 | 0.79-0.97 | Non-monotonic early mapping |
| **5-25** | **15** | **0.943-0.951** | **21-layer plateau** |
| 26-38 | 19-31 | 0.79-0.95 | Monotonic late mapping |
| 39-40 | 32 | 0.66-0.72 | Final layers diverge |

### Disagreement is one-dimensional

At the atlas layer (GLM-5 vs Llama-15), rank_95 = **1**. PC0 captures **97.1%** of all variance. Across GLM layers 5-22, rank is consistently 1. The models are nearly identical at this depth, differing along a single direction in 4096D space.

### PC0 is a language/position detector (97.1%)

| Direction | Extreme tokens | Pattern |
|:---|:---|:---|
| Most negative (biggest disagreement) | `Dep` (French), `S` (Spanish), `Make` | Non-English and sentence-initial tokens |
| Most positive (smallest disagreement) | `us`, `ius`, `Code`, `ire` | English morphemes in mid-sentence context |

### PC1 separates year tokens from non-English morphemes (1.6%)

| Direction | Extreme tokens | Pattern |
|:---|:---|:---|
| Positive | `193`, `197`, `191` | English year prefixes (Wikipedia dates) |
| Negative | `obre`, `uis`, `utsch` | Non-English subword fragments |

### Later PCs capture domain structure

- **PC2**: code indentation (positive) vs conversational nouns like `transportation`, `restaurant` (negative)
- **PC3**: Python type annotations `value: Optional[T]` (positive) vs English narrative (negative)
- **PC4**: concrete nouns `cotton`, `grass` (positive) vs code block delimiters `{` (negative)
- **PC6**: math results `240`, `75` (positive) vs conversational conjunctions (negative)

---

## Phase 3 Results: The Graft & Bridge Experiments (complete)

Tested whether the geometric convergence from Phase 2 translates to functional interchangeability — can GLM's early layers feed into Llama's late layers and produce coherent output? Full experiment design: [`activation_experiment/GRAFT_EXPERIMENT.md`](activation_experiment/GRAFT_EXPERIMENT.md).

### The Graft: geometric similarity does not predict functional compatibility

Fitted 6 projection types (identity, learned affine, mean-shift, random orthogonal, Procrustes, rank-1 correction) at 6 splice points and measured cosine similarity on holdout data, then ran the top 10 configurations through Llama's actual layers and measured next-token KL divergence.

**Phase 1 — projection fitting (cosine similarity on holdout):**

|  | P0:Identity | P1:Affine | P2:MeanShift | P3:RandOrth | P4:Procrustes | P5:Rank1 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| S0 (embed, r=2953) | 0.001 | 0.606 | 0.184 | -0.001 | 0.667 | 0.218 |
| S1 (CKA=.973, r=482) | 0.001 | 0.465 | 0.334 | -0.006 | 0.412 | 0.252 |
| S2 (atlas, r=1) | -0.003 | 0.564 | 0.197 | -0.004 | 0.630 | 0.430 |
| S3 (mid, r=1) | -0.005 | 0.613 | 0.167 | -0.004 | **0.796** | 0.308 |
| S4 (late) | 0.000 | 0.573 | 0.107 | 0.001 | 0.764 | 0.087 |
| S5 (control) | -0.001 | 0.577 | 0.060 | 0.011 | 0.770 | 0.087 |

Key findings:
- **Identity (P0) is near-zero everywhere.** Despite CKA > 0.94, raw activations are completely incompatible. CKA does not predict functional interchangeability.
- **Random orthogonal (P3) is near-zero everywhere.** The alignment captures real structure, not just downstream robustness.
- **Procrustes (P4) consistently wins.** The models are rotationally misaligned but geometrically similar. Orthogonal rotation captures more than unconstrained affine.
- **Rank-1 correction (P5) is insufficient.** P1 >> P5. The rank-1 eigendecomposition misses most of the needed correction.
- **Highest CKA (S1) is actually the worst splice point** for most projections. CKA does not predict best splice point.

**Phase 2 — functional diagnostics (KL divergence through Llama's actual layers):**

| Config | KL median | Top-1 agree | Top-5 overlap |
|:---|:---:|:---:|:---:|
| S2_P4 (atlas+Procrustes) | **6.92** | 3.1% | 5.7% |
| S1_P4 (highest CKA) | 7.33 | 0.4% | 2.6% |
| S3_P4 (mid-plateau) | 8.69 | **12.4%** | **14.6%** |
| S3_P0 (identity) | 10.63 | 0.6% | 2.0% |
| S0_P4 (embed, control) | 16.76 | 0.2% | 1.0% |

Without training, even the best projection produces only 12.4% top-1 agreement. The geometric bridge alone is not enough.

### The Bridge: gradient descent closes the gap

Trained a single `nn.Linear(4096, 4096, bias=True)` (16.8M parameters) end-to-end through frozen Llama layers 15-32 against next-token cross-entropy loss. Initialized from the Procrustes matrix at the atlas layer (S2). Also trained a random-init baseline.

| Configuration | Loss | Perplexity | Top-1 | Top-5 |
|:---|:---:|:---:|:---:|:---:|
| Native Llama (ground truth) | 1.95 | 7.0 | 61.0% | — |
| **Procrustes-init, trained** | **2.58** | **13.2** | **51.7%** | **71.7%** |
| Random-init, trained | 5.38 | 217.4 | 22.5% | 38.8% |
| Procrustes, untrained | 7.52 | 1,840 | 8.6% | 17.1% |
| Identity (raw splice) | 11.67 | 116,574 | 2.0% | — |

**The bridge works.** A trained linear projection achieves 51.7% top-1 agreement with native Llama and perplexity of 13.2 (vs Llama's native 7.0). The graft predicts the same next token as Llama more than half the time.

**Procrustes initialization is critical.** At step 500, Procrustes-init had 49.4% top-1; random-init at step 2000 had only 18.5%. After full training, Procrustes-init reaches 51.7% vs random-init's 22.5%. The geometric analysis from Phase 2 isn't just a measurement — it's a usable initialization that gradient descent builds on.

**Position-dependent KL after training:**

| Position | Procrustes trained | Random trained |
|:---:|:---:|:---:|
| 0-16 | 4.10 | 6.62 |
| 16-32 | 1.93 | 5.13 |
| 32-64 | 2.39 | 5.00 |
| 64-128 | 1.95 | 5.01 |

The trained projection learned to compensate for RoPE incompatibility at longer positions. The remaining high KL at positions 0-16 reflects the sentence-initial/multilingual disagreement (PC0 from Phase 2) that a global linear transform cannot fully resolve.

**The linear bridge has a ceiling at ~2.5 nats** (vs native 1.95). This 0.6-nat gap is NOT a limitation of linear transforms — it's an informational barrier at the splice point.

### MLP ablation: the ceiling is informational, not expressional

Replaced the linear projection with nonlinear alternatives to test whether the 2.58 loss ceiling could be broken:

| Configuration | Params | Loss | PPL | Top-1 | Top-5 |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Linear (Procrustes-init)** | **16.8M** | **2.58** | **13.2** | **51.7%** | **71.7%** |
| MLP 2-layer (Procrustes-init) | 33.6M | 3.01 | 20.2 | 44.9% | 66.2% |
| MLP 2-layer (random-init) | 33.6M | 4.47 | 87.6 | 30.5% | 48.3% |
| MLP bottleneck (4096→1024→4096) | 8.4M | 5.21 | 183.7 | 23.1% | 40.1% |

**The linear bridge is the best bridge.** Adding nonlinearity (GELU, double the parameters) made things worse — the MLP overfits (train loss 1.77 vs val loss 3.01). The ceiling is not about the expressiveness of the transform. It's about what information GLM's first 5 layers compute vs what Llama's layers 15-32 need. Whatever Llama layers 0-14 provide that GLM layers 0-5 don't cannot be recovered by any transform on the splice point activations, because it was never computed.

### Category attribution: code grafts perfectly, geometry doesn't predict function

Evaluated the trained bridge per-category on a stratified holdout (100 inputs per category):

| Category | Native loss | Bridge loss | Gap | Top-1 | Phase 2 rank |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Code** | **0.650** | **0.007** | **-0.643** | **99.8%** | 283 |
| Math reasoning | 1.643 | 1.252 | -0.391 | 68.6% | 872 |
| Conversational | 2.087 | 1.496 | -0.591 | 62.2% | 1098 |
| Multilingual | 2.230 | 2.174 | -0.056 | 53.8% | 1204 |
| English web | 2.412 | 2.334 | -0.077 | 50.3% | 1333 |
| Mixed edge | 2.067 | 2.586 | +0.519 | 51.0% | 204 |

**Code grafts losslessly.** The bridge achieves 99.8% top-1 agreement with native Llama on code — essentially perfect. The trained projection learned to translate GLM's code representations into Llama's with near-zero information loss.

**Geometry does not predict function (R² = 0.098).** The Phase 2 rank profile does not forecast per-category bridge quality. Mixed edge cases have low rank (204, structured disagreement) but the worst bridge gap (+0.519). Conversational has high rank (1098, diffuse) but a strong bridge (-0.591). The geometric analysis describes the representational relationship accurately but does not predict functional composability.

### What this means

Two independently trained models (GLM-4 9B and Llama 3.1 8B) can be composed into a functional chimera — GLM's early layers producing representations that Llama's late layers decode into coherent next-token predictions — using only a trained 4096x4096 linear projection. The cost of this composition is a 2x perplexity increase, bounded by an informational barrier rather than a transform expressiveness barrier. The geometric similarity discovered in Phase 2 (CKA 0.95, rank-1 disagreement) provides the initialization that makes this bridge trainable in minutes rather than hours.

---

## Phase 2: Activation Differential Experiment

Extracts hidden-state activations from GLM-4 9B and Llama 3.1 8B at aligned token positions, computes per-layer activation differentials, and analyzes the geometry of representational disagreement.

### Step 1: Build corpus (no GPU needed, already done)

```bash
cd activation_experiment
python corpus_builder.py
```

Produces `data/corpus.jsonl` (10K inputs, ~940K aligned token pairs) from Wikipedia, Alpaca, GSM8K, code snippets, and multilingual text.

### Step 2: Extract Model A activations (GPU required)

```bash
python pass1_extract.py
```

- Loads GLM-4 9B in fp16 (~18GB VRAM)
- Extracts hidden states at all 41 layers (embedding + 40 transformer) for each aligned token
- Writes to `activations/model_a_activations.mmap` (~295GB)
- Checkpoints every 500 inputs — resumable if interrupted

### Step 3: Compute differentials (GPU required)

```bash
python pass2_differential.py
```

- Loads Llama 3.1 8B in fp16 (~16GB VRAM)
- Computes CKA layer correspondence (41×33 matrix) — discovers which GLM layers map to which Llama layers
- Runs GPU-accelerated streaming Welford covariance on activation differentials (vec_a - vec_b) at each matched layer pair
- Per-category covariance at 8 evenly-spaced layers
- Checkpoints every 500 inputs — resumable if interrupted
- Outputs to `results/overall_stats/` and `results/category_stats/`

### Step 4: Eigendecompose and project (GPU for projection, CPU for eigen)

```bash
python pass3_project.py
```

- Eigendecomposes all covariance matrices
- Identifies the "atlas layer" (lowest rank = most structured disagreement)
- Projects all ~940K differential vectors onto top 20 PCs
- Extracts top/bottom 50 extreme tokens per PC with context
- Outputs to `results/atlas/`

### Step 5: Analysis and visualization (CPU only)

```bash
python analysis.py
python visualize.py
```

Produces rank profiles, eigenvalue spectra, category overlap heatmaps, variance explained charts, layer correspondence heatmap, and the disagreement atlas.

### Config

All hyperparameters in `activation_experiment/config.py`:
- `BATCH_SIZE = 16` — adjust down if OOM
- `MAX_TOKENS = 128` — input truncation length
- `RANK_THRESHOLD = 0.95` — cumulative variance threshold for rank computation
- `TOP_K_PCS = 20` — number of PCs for the atlas
- `CKA_SUBSET_SIZE = 500` — inputs used for layer correspondence

Models are configured in `config.py`. Currently A=GLM-4 9B (extracted first), B=Llama 3.1 8B (run live). Llama weights sourced from `NousResearch/Meta-Llama-3.1-8B-Instruct` (ungated mirror).

### Expected outputs after full run

```
results/
  layer_correspondence.npy          # 41×33 CKA matrix
  layer_correspondence_heatmap.png
  layer_mapping.json                # Optimal GLM→Llama layer map
  correspondence_analysis.md
  rank_profile.json
  rank_profile.png
  eigenvalue_spectra.png
  variance_explained.png
  category_overlap_heatmap.png
  category_rank_profiles.png
  category_analysis.json
  pass2_metadata.json
  overall_stats/layer_*_mean_diff.npy
  overall_stats/layer_*_cov_diff.npy
  atlas/
    pc_0_extremes.json ... pc_19_extremes.json
    atlas_summary.md
    variance_explained.json
    projections.npy
```
