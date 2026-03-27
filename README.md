# rosetta

Cross-model activation analysis and chimeric composition of open-source LLMs.

All experiments run locally on an NVIDIA RTX 5090 (32GB VRAM). Full report: [`activation_experiment/results/report.md`](activation_experiment/results/report.md).

---

## Results

### Phase 1: Tokenizer Translation Table

Compared tokenizers across 9 open-source LLMs. Produced vocabulary overlap analysis, segmentation divergence metrics, and a character-span alignment module used by all subsequent experiments.

**Key finding:** DeepSeek-V3 vs GLM-4 have 100% bucket-1 alignment. Multilingual text is the most divergent category; simple English the most aligned.

### Phase 2: Activation Differentials (GLM-4 9B vs Llama 3.1 8B)

Compared hidden-state activations across 920,060 aligned token pairs from 10,000 inputs spanning six categories.

**Layer correspondence:** 21 of GLM's 41 layers (5-25) all map to a single Llama layer (15) with CKA 0.943-0.951. Peak CKA is 0.973. GLM distributes across 21 layers what Llama compresses into one.

**Disagreement is one-dimensional.** At the atlas layer (GLM-5 vs Llama-15), rank_95 = 1. PC0 captures 97.1% of all variance and separates non-English/sentence-initial tokens (biggest disagreement) from mid-sentence English (smallest).

### Phase 3: The Graft & Bridge

Tested whether geometric convergence enables functional composition.

**Geometric alignment alone fails.** Despite CKA > 0.94, identity-projected activations produce near-zero top-1 agreement. Procrustes rotation achieves cos=0.796 but only 12.4% top-1 through Llama's actual layers.

**A trained linear bridge closes the gap:**

| Configuration | Loss | PPL | Top-1 | Top-5 |
|:---|:---:|:---:|:---:|:---:|
| Native Llama | 1.95 | 7.0 | 61.0% | -- |
| **Trained bridge (Procrustes-init)** | **2.58** | **13.2** | **51.7%** | **71.7%** |
| Trained bridge (random-init) | 5.38 | 217 | 22.5% | 38.8% |

Procrustes initialization from the geometric analysis enables convergence in 500 steps vs thousands for random init.

**The ceiling is informational, not expressional.** An MLP with double the parameters performs worse (loss 3.01) due to overfitting. The 0.6-nat gap is information that GLM's first 5 layers never compute.

**Code grafts losslessly (99.8% top-1).** But geometry doesn't predict per-category function (R² = 0.098):

| Category | Native loss | Bridge loss | Top-1 |
|:---|:---:|:---:|:---:|
| Code | 0.650 | 0.007 | 99.8% |
| Math | 1.643 | 1.252 | 68.6% |
| Conversational | 2.087 | 1.496 | 62.2% |
| Multilingual | 2.230 | 2.174 | 53.8% |
| English web | 2.412 | 2.334 | 50.3% |

**Generation works but degrades.** The chimera produces fluent English and reasonable answers, but writes logically broken code, wrong math, and French with morphological stuttering (`siècleclele`, `Allemagneagne`). Per-token accuracy doesn't survive autoregressive error compounding.

---

## Setup

```bash
pip install torch transformers accelerate bitsandbytes numpy scipy matplotlib seaborn scikit-learn datasets tiktoken sentencepiece
```

Llama weights from `NousResearch/Meta-Llama-3.1-8B-Instruct` (ungated). GLM from `THUDM/glm-4-9b-chat`.

---

## Reproduction

### Phase 2: Activation analysis

```bash
cd activation_experiment
python run_all.py          # corpus build + extraction + differentials + analysis + viz (~2.5 hours)
```

Or step by step:

```bash
python corpus_builder.py   # Build 10K-input corpus with token alignment
python pass1_extract.py    # Extract GLM-4 activations to memmap (~70 min, ~295GB)
python pass2_differential.py  # CKA + streaming covariance with Llama (~45 min)
python pass3_project.py    # Eigendecomposition + PCA projection (~30 min)
python analysis.py         # Post-hoc statistics
python visualize.py        # Generate plots
```

### Phase 3: Graft experiments

```bash
cd graft
python validate_split.py       # Validate partial forward pass infrastructure
python fit_projections.py      # Fit 36 projection configs, ~45 min
python diagnostics.py          # Logit-level functional diagnostics, ~30 min
python bridge.py               # Train linear bridge end-to-end, ~20 min
python bridge_mlp.py           # MLP ablation, ~1.5 hours
python category_attribution.py # Per-category evaluation, ~30 min
python generate_validation.py  # Autoregressive generation test, ~15 min
```

### Config

All hyperparameters in `activation_experiment/config.py`. Key settings:
- `MODEL_A_ID` / `MODEL_B_ID` — model pair (currently GLM-4 / Llama 3.1)
- `BATCH_SIZE = 16`, `MAX_TOKENS = 128`, `RANK_THRESHOLD = 0.95`
- `CKA_SUBSET_SIZE = 500`, `TOP_K_PCS = 20`

### Output structure

```
activation_experiment/
  results/
    report.md                        # Full report with all findings
    layer_correspondence_heatmap.png # CKA matrix visualization
    rank_profile.png                 # Per-layer rank at 95% variance
    eigenvalue_spectra.png           # Eigenvalue decay curves
    variance_explained.png           # Cumulative variance by PC
    category_overlap_heatmap.png     # Subspace overlap between categories
    atlas/                           # Disagreement atlas (top PCs + extreme tokens)
  graft/
    phase1_results.json              # 36-cell projection ablation matrix
    phase2_results.json              # Functional diagnostics (KL, top-k)
    bridge/                          # Trained bridge checkpoints + learning curves
    bridge_mlp/                      # MLP ablation checkpoints
    category_attribution/            # Per-category bridge evaluation
    generation_validation.json       # Autoregressive generation samples
  GRAFT_EXPERIMENT.md                # Experiment design document
```
