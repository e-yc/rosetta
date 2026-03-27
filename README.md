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
