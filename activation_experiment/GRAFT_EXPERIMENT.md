# The Graft Experiment (v3)

## Core question

Does geometric similarity (CKA, rank) between two models' representations predict functional compatibility — i.e., can one model's early layers feed into the other's late layers and produce meaningful computation?

The Phase 2 results show GLM-4 and Llama 3.1 converge to near-identical representational geometry at their midlayers (CKA 0.95, rank-1 disagreement). This experiment tests whether that geometric convergence extends to functional interchangeability, and precisely characterizes where it breaks down.

## Key architectural asymmetries

These are not minor details — they shape the experiment design and constrain which failure modes are possible.

| Property | Llama 3.1 8B | GLM-4 9B |
|:---|:---|:---|
| KV heads (GQA) | 8 (4:1 ratio) | 2 (16:1 ratio) |
| RoPE config | theta=500K, llama3 scaling (factor=8) | original_rope, ratio=500 |
| QKV bias | No | Yes |
| FFN hidden dim | 14,336 | 13,696 |
| LayerNorm epsilon | ~1e-5 | 1.5625e-7 |
| Total layers | 32 | 40 |

The GQA mismatch (8 vs 2 KV heads) means even geometrically identical residual streams will produce different attention patterns — Llama's 8 KV heads partition the representation space differently than GLM's 2. This is the most likely failure mode for the forward graft. For the reverse graft (Llama→GLM), the mismatch is structurally different: GLM's more aggressive grouping (16:1) may be *more* robust to input perturbation since each KV head covers a wider subspace, or *less* robust since there's less redundancy. The reverse direction result should not be over-interpreted as evidence about "early-layer portability" without accounting for this asymmetry.

## RoPE compatibility: the dedicated diagnostic

RoPE is the most likely failure mode that no projection variant can fix. The projections (P0-P5) all operate on the residual stream globally, but RoPE is applied inside each attention layer to Q and K specifically. GLM's layers 0-L bake GLM's positional encoding into the residual stream through attention patterns computed with GLM's RoPE. When Llama's layer L' receives this residual stream, it applies Llama's RoPE to its own Q/K projections, creating a hybrid of two incompatible positional schemes.

**Dedicated RoPE diagnostic (Phase 1):** Compute cosine similarity between projected GLM activations and native Llama activations **as a function of sequence position**. If RoPE incompatibility is a problem, cosine similarity will degrade at longer positions as the phase mismatch between the two RoPE schemes accumulates. If cosine similarity is flat across positions, RoPE is not the dominant failure mode.

Also compute per-position KL divergence in Phase 2a. If KL is low at position 0-10 but degrades at position 50+, that's a RoPE signature. If KL is uniformly bad at all positions, the problem is representational, not positional.

## Splice point candidates

Five points spanning the full spectrum:

| ID | GLM layer | Llama layer | CKA | Rank_95 | Rationale |
|:---|:---:|:---:|:---:|:---:|:---|
| S0 | 0 | 0 | 0.879 | 2953 | Embedding — negative control (diffuse, unstructured) |
| S1 | 4 | 23 | 0.973 | 482 | Highest CKA — most similar geometry but high-rank disagreement |
| S2 | 5 | 15 | 0.943 | 1 | Atlas layer — most structured disagreement (rank-1) |
| S3 | 15 | 15 | 0.951 | 1 | Mid-plateau — symmetric layer index, rank-1 |
| S4 | 35 | 31 | 0.903 | high | Late layers — degraded correspondence |
| S5 | 39 | 32 | 0.661 | ~3000 | Near-final — negative control |

S0 and S5 are both negative controls but at opposite ends: S0 tests splicing at a point of diffuse, unstructured disagreement (high rank, decent CKA), S5 at a point of poor geometric match. If both fail while S2/S3 succeed, the rank metric is validated across the full range.

Note: S1's rank of 482 is from the corrected (Llama-tokenizer) run. Verify by checking `rank_profile.json` for `layer_A4_B23`. The tension between S1 (highest CKA, high rank) and S2 (lower CKA, rank-1) is the experiment's most interesting axis. CKA and rank are measuring different things — this ablation determines which one predicts functional compatibility.

## Projection variants (fitted at each splice point)

| ID | Projection | Purpose |
|:---|:---|:---|
| P0 | Identity (no transform) | Baseline: are the models close enough raw? |
| P1 | Learned affine (W*x + b) | Best linear correction (full 4096x4096) |
| P2 | Mean-shift only (x + b) | Is geometry already aligned, just offset? |
| P3 | Random orthogonal | Control: is downstream just robust to rotations? |
| P4 | Procrustes (orthogonal W) | Geometry-preserving alignment (best rotation/reflection) |
| P5 | Rank-1 correction (I + alpha * v * v^T + b) | Uses only the PC0 direction from eigendecomposition |

P5 is the cleanest possible correction. The disagreement is rank-1, so the correction should be rank-1. If P5 matches P1, then the entire cross-model transform reduces to one scalar (alpha) and one direction (v) — the strongest possible evidence that rank-1 disagreement translates to a rank-1 fix. If P1 >> P5, the full affine is capturing structure beyond what the eigendecomposition reveals, and the rank-1 metric is misleading about the true correction needed.

---

## Phase 1: Fit projections and RoPE diagnostic (CPU, ~30 min)

Use the 920K paired activation vectors from Phase 2. For each splice point, extract the relevant layer pair from the stored data.

### 1a: Fit projections

**P1 (affine):** Solve W, b = argmin ||W * GLM_L + b - Llama_L'||^2 across 80% of pairs. Use `numpy.linalg.lstsq` with bias folded into an augmented matrix. Evaluate on 20% holdout.

**Sample efficiency check:** Also fit P1 on 10K, 50K, and 100K subsets and evaluate all on the same holdout set. If holdout residual saturates early (10-50K), the projection captures genuine structure and will transfer to the eval corpus. If residual keeps improving through 920K, the projection may be memorizing corpus-specific patterns. This takes minutes of extra CPU time and is essential for trusting the projection's generalization.

**P4 (Procrustes):** SVD of cross-covariance: U, S, Vt = svd(Llama.T @ GLM); W = U @ Vt.

**P5 (Rank-1):** Load PC0 eigenvector v from `results/eigen/layer_A{L}_B{L'}_eigenvectors.npy`. Fit scalar alpha and bias b to minimize ||x + alpha * (v^T x) * v + b - Llama_L'||^2 over training set. This is a 4097-parameter optimization (alpha + bias vector).

### 1b: Projection diagnostics

For each splice point × projection on the holdout set:
- Holdout L2 residual (absolute and relative to activation norms)
- Per-dimension correlation between projected GLM and native Llama
- Cosine similarity distribution per token
- Activation norm distribution: projected vs native (histograms)

### 1c: RoPE positional diagnostic

Using the holdout set, compute cosine similarity between projected GLM and native Llama activations **binned by sequence position** (bins: 0-15, 16-31, 32-63, 64-127). Plot as a function of position. If cosine similarity degrades with position, RoPE incompatibility is active and no amount of projection fitting will fix it — the experiment should document this as the dominant failure mode and explore position-dependent corrections as future work.

### 1d: Calibration baseline

Compute the KL noise floor: run Llama on the same inputs with two different causal attention masks (one with and one without a dummy padding token at position 0, then strip it). The KL divergence between these two runs is the noise floor below which graft KL is indistinguishable from Llama self-noise. All Phase 2 KL values should be compared against this floor, not against zero.

---

## Phase 2: Functional diagnostics (GPU, ~3 hours)

The core of the experiment. Do NOT generate text yet.

### Step 2a: Next-token logit comparison

For a 500-input evaluation corpus (held out from training, diverse across all 6 categories):

1. Load GLM. For each of the 6 splice points, run GLM layers 0-L on all inputs, save intermediate activations. Unload GLM. (~15 min)

2. Load Llama. Run full native Llama on all inputs → ground truth logits at every position. Then for each of 36 configs (6 splice × 6 projections), apply projection to saved GLM activations, run partial Llama forward pass starting at layer L', collect logits. (~2.5 hours)

Per-position metrics for each config:
- **KL divergence**: D_KL(Llama_logits || Graft_logits)
- **Top-1 agreement**: does the graft predict the same top token as Llama?
- **Top-5 overlap**: how many of graft's top-5 appear in Llama's top-5?
- **Rank of correct token**: where does the graft rank the actual next token?
- **KL by position bucket**: break KL down by sequence position (0-15, 16-31, 32-63, 64-127) to detect RoPE degradation

**Engineering the partial forward pass:**

```python
layers = model.model.layers  # list of LlamaDecoderLayer

hidden = projected_activations  # (batch, seq_len, 4096)

# position_ids: standard 0..seq_len-1 range
# This gives Llama's layers their expected positional encoding.
# The residual stream from GLM carries GLM's positional info from
# layers 0-L, and Llama's RoPE at layer L' will overlay its own.
# The diagnostic in 1c tells us whether this hybrid is workable.
position_ids = torch.arange(seq_len).unsqueeze(0).to(device)

# Causal attention mask
attention_mask = torch.ones(1, 1, seq_len, seq_len, device=device)
attention_mask = attention_mask.tril()

for layer in layers[splice_llama_layer:]:
    hidden = layer(hidden, attention_mask=mask, position_ids=position_ids)[0]

hidden = model.model.norm(hidden)
logits = model.lm_head(hidden)
```

Note: the exact attention_mask format depends on the transformers version and model implementation. Debug this carefully — incorrect masking will produce silently wrong results.

**Partial forward pass validation (do this first, before anything else):** Run Llama's full native forward pass and compare logits against a manually-split pass (layers 0-14 then 15-31 using the same code path as the graft). They will NOT be bitwise identical — floating point accumulation order in attention changes with different code paths, and some fused operations behave differently layer-by-layer. Set a tolerance: cosine similarity > 0.9999 and max absolute logit difference < 0.01. If the self-split can't hit that, the partial forward pass infrastructure is broken and nothing downstream is trustworthy. Build and validate this before writing any other graft code.

### Step 2b: Attention pattern divergence

Instrument Llama at the splice layer **and the next 3 layers** (L' through L'+3). At each layer, for each attention head, compare attention weight matrices between:
- Native Llama input (ground truth)
- Projected GLM input (graft)

Compute per-head Jensen-Shannon divergence of attention distributions. Report:
- Which heads diverge most at the splice layer?
- Does divergence amplify or dampen across layers L' through L'+3?
- Is the disruption uniform across heads or concentrated in a few? With Llama's 8 KV heads, if disruption concentrates in 1-2 heads, those may correspond to functions that GLM distributes differently via its 2-head GQA, and a targeted per-head correction might be feasible.

### Step 2c: Error propagation

For the best-performing splice config from 2a:

1. Run the graft through Llama layers L' to 32, save the residual stream at every layer
2. Run native Llama, save the residual stream at every layer from L' to 32
3. At each layer: cosine similarity, L2 distance (absolute and relative to norm), and KL divergence of the logit distribution if you project through the LM head at that layer

Plot all three as curves. The cosine similarity curve shows whether errors amplify or dampen. The intermediate-logit KL curve shows whether functional divergence precedes or follows representational divergence. If cosine stays high but KL degrades, the function is more sensitive than the geometry — evidence that CKA overstates compatibility.

---

## Phase 3: Ablation matrix

The 36-cell grid (6 splice points × 6 projections) is the main result:

```
                     P0:Id  P1:Affine  P2:Shift  P3:RandOrth  P4:Procr  P5:Rank1
S0 (CKA=.879,r=2953) KL=?   KL=?       KL=?      KL=?         KL=?      KL=?
S1 (CKA=.973,r=482)  KL=?   KL=?       KL=?      KL=?         KL=?      KL=?
S2 (CKA=.943,r=1)    KL=?   KL=?       KL=?      KL=?         KL=?      KL=?
S3 (CKA=.951,r=1)    KL=?   KL=?       KL=?      KL=?         KL=?      KL=?
S4 (CKA=.903,r=hi)   KL=?   KL=?       KL=?      KL=?         KL=?      KL=?
S5 (CKA=.661,r=3K)   KL=?   KL=?       KL=?      KL=?         KL=?      KL=?

(Also report top-1 agreement and calibration-adjusted KL for each cell)
```

### Interpretation guide

| Pattern | Meaning |
|:---|:---|
| P0 works at S2 but not S4/S5 | Identity sufficient at rank-1 layers; geometry alone determines compatibility |
| P1 << P0 everywhere | Learned correction essential; raw geometry isn't enough |
| P1 ≈ P2 | Models rotationally aligned, just offset (strongest convergence claim) |
| P1 ≈ P5 | Full 4096x4096 affine is overkill; correction is literally one scalar × one direction |
| P5 << P1 | Rank-1 metric is misleading; correction needs more than the dominant eigenvector |
| P3 ≈ P1 | Downstream robust to arbitrary rotations (projection not capturing specific structure) |
| P4 < P1 | Non-orthogonal (scaling) corrections matter; models not just rotated versions |
| S1 beats S2 | CKA beats rank as splice predictor |
| S2 beats S1 | Rank beats CKA as splice predictor |
| KL degrades with sequence position | RoPE incompatibility is the dominant failure mode |
| KL uniform across positions | Failure is representational, not positional |
| All cells bad vs calibration floor | Geometric similarity doesn't predict functional compatibility |

### Reverse direction

Run Llama 0-L' → projection → GLM L-40 for the best forward splice point. Note: the GQA asymmetry (Llama 8 KV heads → GLM 2 KV heads) makes this a structurally different challenge than the forward direction. GLM's more aggressive grouping (16:1) may be more or less robust to input perturbation — either way, the result reflects GQA compatibility as much as early-layer portability. Document this caveat explicitly; do not interpret asymmetric results as evidence about which model's "early computation is more portable" without controlling for the GQA confound.

---

## Phase 4: Generation (conditional)

Proceed only if Phase 2 shows at least one configuration with:
- Mean KL < calibration floor × 5 (within an order of magnitude of Llama's self-noise)
- Top-1 agreement > 40% (above what random logit perturbation would give)

If neither threshold is met, the experiment is complete. The contribution is the ablation matrix and the precise characterization of the failure mode (RoPE vs attention vs representational). This is still a publishable result.

If thresholds are met:

### Evaluation corpus
200 inputs, not from the training corpus:
- 50 English continuations (Wikipedia first sentences)
- 50 instruction-following (Alpaca-style)
- 50 code completion (function signatures)
- 25 multilingual (French/Spanish starters)
- 25 math (GSM8K-style)

### Generation
Greedy decoding, 128 tokens max. Collect for each input:
- Llama alone
- GLM alone
- Best graft config
- Identity graft (P0 at same splice point)

### Metrics

**Quantitative:**
- Perplexity evaluated by Llama and by GLM separately
- Self-BLEU between graft and each donor (which does it resemble?)
- Distinct-n (n=1,2,3) for degeneration detection

**The chimera test (qualitative):**
For 20 representative outputs across categories, annotate:
- Does graft resemble Llama, GLM, neither, or a blend?
- On multilingual: GLM's multilingual strength or Llama's English bias?
- On code: whose style dominates?

The chimera thesis is only validated if the graft shows characteristics of BOTH donors that neither alone exhibits. If it produces Llama-like text, the early layers are fungible and the late layers dominate — interesting but not a chimera. Document the distinction clearly.

---

## Implementation

### Scripts

1. `graft_fit_projections.py` — Phase 1. Fits all projections, runs RoPE diagnostic, computes calibration baseline, saves matrices and diagnostic plots. CPU only.

2. `graft_diagnostics.py` — Phase 2. Sequential model loading, partial forward passes, logit KL, attention divergence, error propagation. GPU required.

3. `graft_generate.py` — Phase 4 (conditional). Generation pipeline. GPU required.

4. `graft_report.py` — Compiles ablation matrix, diagnostic plots, and interpretation into report.

### Compute budget

| Step | Time estimate | GPU |
|:---|:---|:---|
| Phase 1: Fit projections + diagnostics | ~30 min | No |
| Phase 2a: Logit comparison (36 configs × 500 inputs) | ~3 hours | Yes |
| Phase 2b: Attention patterns (top 5 configs × 4 layers) | ~45 min | Yes |
| Phase 2c: Error propagation (best config) | ~15 min | Yes |
| Phase 3: Reverse direction (best config) | ~30 min | Yes |
| Phase 4: Generation (if warranted) | ~1 hour | Yes |
| **Total** | **~6-7 hours** | |

Budget an additional 3-4 hours for debugging the partial forward pass. The layer indexing, attention mask format, position_ids dtype, and KV cache handling are all potential sources of silent errors. Validate the partial pass by comparing native Llama logits against a manually-split forward pass (layers 0-14 then 15-31) — cosine similarity > 0.9999 and max absolute logit difference < 0.01. If the self-split can't hit that, the infrastructure is broken.

### Model loading strategy

1. Load GLM (~18GB). Run layers 0-L for all eval inputs at all 6 splice points. Save activations to disk (~50MB). Unload GLM.

2. Load Llama (~16GB). Run native forward pass (ground truth). Then for each of 36 splice configs, apply projection and run partial forward pass. Llama stays loaded for the entire diagnostic phase.

3. If Phase 4: unload Llama, load GLM for GLM-alone generation. Unload, reload Llama for Llama-alone and graft generation.

---

## What to expect

**Most likely outcome:** P1 at S2/S3 produces KL of 5-15 nats — recognizably related but functionally degraded. P0 performs surprisingly close to P1 at rank-1 splice points, confirming genuine proximity. P5 (rank-1 correction) matches P1 at S2 but not at S1, confirming that rank-1 disagreement translates to rank-1 correction where it applies. P3 (random orthogonal) produces garbage, confirming alignment is real. KL degrades with sequence position, implicating RoPE. Late splice points fail across all projections.

**Best case:** P5 ≈ P1 at S2, KL < calibration × 5, proceeds to generation with coherent chimeric output that blends characteristics of both donors.

**Most informative outcome:** S1 and S2 produce dramatically different results, definitively answering whether CKA or rank is the better predictor of functional compatibility. This, combined with the RoPE positional analysis and the P5-vs-P1 comparison, yields a precise characterization of what geometric similarity metrics actually measure and where they stop predicting function.

**The contribution regardless of outcome:** A 36-cell ablation matrix mapping (CKA, rank, projection type) → functional compatibility (KL divergence), with diagnostic attribution of failure modes (RoPE vs attention vs representational). This is the first quantitative measurement of the gap between representational similarity and functional interchangeability.
