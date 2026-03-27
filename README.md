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

### Phase 4: Additive Injection — capability transfer without replacement

Instead of replacing Llama's residual stream, inject a scaled delta from GLM: `h_injected = h_llama + alpha * (bridge(h_glm) - h_llama)`. Llama keeps its full computation; GLM contributes what it uniquely knows.

**Phase 0 — baselines (GLM wins on English, math, French; Llama wins on Chinese, code):**

| Category | GLM loss | Llama loss | Winner |
|:---|:---:|:---:|:---|
| Chinese | 4.509 | 3.424 | Llama |
| English | 3.275 | 4.578 | GLM |
| Code | 2.769 | 2.172 | Llama |
| Math | 1.683 | 1.991 | GLM |
| French | 3.041 | 3.876 | GLM |

**Phase 1 — alpha sweep (trained bridge delta):**

| | Chinese | English | Code | Math | French |
|:---|:---:|:---:|:---:|:---:|:---:|
| Native GLM | 4.509 | 3.275 | 2.769 | 1.683 | 3.041 |
| Native Llama (a=0.0) | 4.144 | 4.578 | 2.172 | 1.991 | 3.921 |
| a=0.1 | 4.040 | 4.129 | 2.100 | 1.871 | 3.755 |
| a=0.2 | 3.975 | 3.682 | 1.998 | 1.748 | 3.587 |
| **a=0.5** | **4.097** | **2.842** | **1.883** | **1.546** | **3.286** |
| a=1.0 (full replace) | 17.766 | 3.255 | 3.362 | 1.991 | 4.449 |

**Additive injection transfers capabilities.** At alpha=0.5, the injected model beats native Llama on every category and beats native GLM on math and code:
- English: 4.578 -> 2.842 (-37.9%)
- Math: 1.991 -> 1.546 (-22.4%)
- French: 3.921 -> 3.286 (-16.2%)
- Code: 2.172 -> 1.883 (-13.3%)
- Chinese: 4.144 -> 4.097 (-1.1%)

**The untrained Procrustes delta shows no signal** — loss barely moves at low alpha and degrades at high alpha. The trained bridge carries functional information that raw geometric alignment does not. Gradient descent in the bridge experiment captured something real that static Procrustes rotation misses.

**Alpha=1.0 (full replacement) is catastrophic** — loss 15-18, confirming that Llama's intact residual stream is essential. The additive formulation preserves it while the graft destroys it.

**Generation with injection fixes every graft artifact.** The graft (full replacement) produced broken code, wrong math, and French with morphological stuttering. The injection (alpha=0.5) preserves Llama's residual stream, preventing error compounding:

| Category | Graft generation | Injection generation |
|:---|:---|:---|
| Code (fibonacci) | Returns `n` — broken | Correct recursive implementation |
| Code (merge sort) | Broken logic | Correct divide-and-conquer |
| Math ($2 apples + $3 oranges) | Wrong answer ($30) | Correct answer ($19) |
| French | Stuttering: `siècleclele`, `Allemagneagne` | Clean: `milieu du XVIIIe siècle` |
| Conversational (stack vs queue) | Vague, no LIFO/FIFO | Uses LIFO/FIFO terminology correctly |

The injection produces generation quality comparable to or better than native Llama across all categories. Speed: 1-4 tok/s (2x Llama forward passes per token).

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
