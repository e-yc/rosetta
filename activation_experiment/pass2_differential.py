#!/usr/bin/env python3
"""
Activation Differential Experiment -- Pass 2: Differentials & Streaming Covariance

1. Compute CKA layer correspondence (stored model layers <-> live model layers)
2. Load Model B, run inference, compute differentials against Model A activations
3. Maintain streaming Welford covariance matrices per matched layer pair

Supports checkpointing -- resumable if interrupted.

Requires: NVIDIA GPU with >= 20GB VRAM
"""

import json
import os
import pickle
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config


PASS2_CHECKPOINT_DIR = os.path.join(config.DATA_DIR, "pass2_checkpoint")
PASS2_CHECKPOINT_META = os.path.join(PASS2_CHECKPOINT_DIR, "meta.json")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_corpus():
    corpus = []
    with open(config.CORPUS_PATH) as f:
        for line in f:
            corpus.append(json.loads(line))
    with open(config.ACTIVATION_INDEX_PATH) as f:
        index_meta = json.load(f)
    return corpus, index_meta


def load_model(model_id, fallbacks, name):
    from transformers import AutoConfig
    token = os.environ.get("HF_TOKEN") or True
    for mid in [model_id] + fallbacks:
        try:
            print(f"  Loading {name} from {mid} ...", flush=True)
            model_config = AutoConfig.from_pretrained(mid, trust_remote_code=True, token=token)
            if not hasattr(model_config, "max_length"):
                model_config.max_length = getattr(model_config, "seq_length", 8192)
            model = AutoModelForCausalLM.from_pretrained(
                mid, config=model_config, torch_dtype=torch.float16, device_map="cuda",
                trust_remote_code=True, token=token,
            )
            tokenizer = AutoTokenizer.from_pretrained(mid, trust_remote_code=True, token=token)
            print(f"  Loaded: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B params")
            return model, tokenizer
        except Exception as e:
            print(f"  FAILED ({str(e)[:100]})")
    print(f"  *** Could not load {name} ***")
    sys.exit(1)


def linear_cka(X, Y):
    """Centered Kernel Alignment between two activation matrices."""
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    hsic_xy = np.linalg.norm(X.T @ Y, "fro") ** 2
    hsic_xx = np.linalg.norm(X.T @ X, "fro") ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, "fro") ** 2
    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-12:
        return 0.0
    return float(hsic_xy / denom)


class WelfordCovariance:
    """Streaming covariance with Chan's parallel merge algorithm."""

    def __init__(self, dim):
        self.dim = dim
        self.count = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.M2 = np.zeros((dim, dim), dtype=np.float64)

    def merge_batch(self, n_b, mean_b, M2_b):
        """Merge pre-computed batch stats using Chan's parallel algorithm.

        Args:
            n_b: number of samples in the batch
            mean_b: (dim,) float64 mean of the batch
            M2_b: (dim, dim) float64 scatter matrix of the batch
        """
        if n_b == 0:
            return
        n_a = self.count
        n_ab = n_a + n_b
        if n_a == 0:
            self.count = n_b
            self.mean = mean_b.copy()
            self.M2 = M2_b.copy()
            return
        delta = mean_b - self.mean
        self.M2 += M2_b + np.outer(delta, delta) * (n_a * n_b / n_ab)
        self.mean = (n_a * self.mean + n_b * mean_b) / n_ab
        self.count = n_ab

    @property
    def covariance(self):
        if self.count < 2:
            return np.zeros((self.dim, self.dim), dtype=np.float64)
        return self.M2 / (self.count - 1)

    def save(self, mean_path, cov_path):
        np.save(mean_path, self.mean.astype(np.float32))
        np.save(cov_path, self.covariance.astype(np.float32))

    def get_state(self):
        return {"count": self.count, "mean": self.mean, "M2": self.M2}

    def set_state(self, state):
        self.count = state["count"]
        self.mean = state["mean"]
        self.M2 = state["M2"]


def gpu_scatter_matrices(batch_tensor):
    """Compute mean and scatter matrix on GPU using torch.

    Args:
        batch_tensor: (n, dim) float32 tensor on CUDA

    Returns:
        n, mean (dim,) float64 numpy, M2 (dim, dim) float64 numpy
    """
    n = batch_tensor.shape[0]
    if n == 0:
        return 0, None, None
    mean = batch_tensor.mean(dim=0)  # (dim,)
    centered = batch_tensor - mean   # (n, dim)
    # Scatter matrix: centered.T @ centered — done entirely on GPU
    M2 = centered.T @ centered       # (dim, dim)
    return n, mean.double().cpu().numpy(), M2.double().cpu().numpy()


def save_welford_dict(wf_dict, path):
    """Save a dict of (key -> WelfordCovariance) to a file."""
    data = {}
    for k, wf in wf_dict.items():
        sk = str(k)
        data[sk] = wf.get_state()
    np.savez_compressed(path, **{
        f"{sk}__{field}": state[field]
        for sk, state in data.items()
        for field in ["mean", "M2"]
    })
    # Save counts separately as JSON (numpy can't store ints in npz easily)
    counts_path = path + ".counts.json"
    counts = {sk: state["count"] for sk, state in data.items()}
    with open(counts_path, "w") as f:
        json.dump(counts, f)


def load_welford_dict(wf_dict, path):
    """Load saved state into an existing dict of WelfordCovariance objects."""
    if not os.path.exists(path + ".npz" if not path.endswith(".npz") else path):
        actual_path = path if path.endswith(".npz") else path + ".npz"
        if not os.path.exists(actual_path):
            return False
    npz = np.load(path if path.endswith(".npz") else path + ".npz")
    counts_path = (path if path.endswith(".npz") else path + ".npz") + ".counts.json"
    if not os.path.exists(counts_path):
        # Try without .npz suffix for counts
        counts_path = path + ".counts.json"
    with open(counts_path) as f:
        counts = json.load(f)

    for k, wf in wf_dict.items():
        sk = str(k)
        if sk in counts:
            wf.set_state({
                "count": counts[sk],
                "mean": npz[f"{sk}__mean"],
                "M2": npz[f"{sk}__M2"],
            })
    return True


def save_running_mean_dict(rm_dict, path):
    """Save a dict of (key -> RunningMean) to npz + json."""
    arrays = {}
    counts = {}
    for k, rm in rm_dict.items():
        sk = str(k)
        arrays[f"{sk}__sum"] = rm.sum
        counts[sk] = rm.count
    np.savez_compressed(path, **arrays)
    with open(path + ".counts.json", "w") as f:
        json.dump(counts, f)


def load_running_mean_dict(rm_dict, path):
    """Load saved state into RunningMean objects."""
    npz_path = path if path.endswith(".npz") else path + ".npz"
    if not os.path.exists(npz_path):
        return False
    npz = np.load(npz_path)
    counts_path = path + ".counts.json"
    with open(counts_path) as f:
        counts = json.load(f)
    for k, rm in rm_dict.items():
        sk = str(k)
        if sk in counts:
            rm.count = counts[sk]
            rm.sum = npz[f"{sk}__sum"]
    return True


def save_checkpoint(last_input_id, nan_count, processed,
                    overall_welford, mean_a_accum, mean_b_accum,
                    category_welford, categories):
    """Save full checkpoint of streaming covariance state."""
    os.makedirs(PASS2_CHECKPOINT_DIR, exist_ok=True)

    print(f"  Saving checkpoint at input {last_input_id}...", flush=True)
    t0 = time.time()

    save_welford_dict(overall_welford, os.path.join(PASS2_CHECKPOINT_DIR, "overall.npz"))
    save_running_mean_dict(mean_a_accum, os.path.join(PASS2_CHECKPOINT_DIR, "mean_a.npz"))
    save_running_mean_dict(mean_b_accum, os.path.join(PASS2_CHECKPOINT_DIR, "mean_b.npz"))

    for cat in categories:
        save_welford_dict(
            category_welford[cat],
            os.path.join(PASS2_CHECKPOINT_DIR, f"cat_{cat}.npz"),
        )

    meta = {
        "last_input_id": last_input_id,
        "nan_count": nan_count,
        "processed": processed,
    }
    with open(PASS2_CHECKPOINT_META, "w") as f:
        json.dump(meta, f)

    print(f"  Checkpoint saved in {time.time() - t0:.1f}s", flush=True)


def load_checkpoint(overall_welford, mean_a_accum, mean_b_accum,
                    category_welford, categories):
    """Load checkpoint if it exists."""
    if not os.path.exists(PASS2_CHECKPOINT_META):
        return None

    print("  Loading checkpoint...", flush=True)
    t0 = time.time()

    with open(PASS2_CHECKPOINT_META) as f:
        meta = json.load(f)

    load_welford_dict(overall_welford, os.path.join(PASS2_CHECKPOINT_DIR, "overall.npz"))
    load_running_mean_dict(mean_a_accum, os.path.join(PASS2_CHECKPOINT_DIR, "mean_a.npz"))
    load_running_mean_dict(mean_b_accum, os.path.join(PASS2_CHECKPOINT_DIR, "mean_b.npz"))

    for cat in categories:
        load_welford_dict(
            category_welford[cat],
            os.path.join(PASS2_CHECKPOINT_DIR, f"cat_{cat}.npz"),
        )

    print(f"  Checkpoint loaded in {time.time() - t0:.1f}s "
          f"(input {meta['last_input_id']}, {meta['processed']} processed)",
          flush=True)
    return meta


# ---------------------------------------------------------------------------
# Phase 1: CKA Layer Correspondence
# ---------------------------------------------------------------------------

def compute_layer_correspondence(model_b, tokenizer_b, corpus, index_meta):
    """Compute CKA between stored model layers and live model layers."""

    # Skip if already computed
    if (os.path.exists(config.LAYER_CORRESPONDENCE_PATH) and
            os.path.exists(config.LAYER_MAPPING_PATH)):
        print("\n--- CKA LAYER CORRESPONDENCE (cached) ---")
        cka_matrix = np.load(config.LAYER_CORRESPONDENCE_PATH)
        with open(config.LAYER_MAPPING_PATH) as f:
            mapping = json.load(f)
        # Detect num_layers_b from the model
        sample_text = corpus[0]["raw_text"]
        inp = tokenizer_b(sample_text, return_tensors="pt", add_special_tokens=False)
        inp = {k: v.to(model_b.device) for k, v in inp.items()}
        with torch.no_grad():
            out = model_b(**inp, output_hidden_states=True)
        num_layers_b = len(out.hidden_states)
        del out
        torch.cuda.empty_cache()
        print(f"  Loaded from disk: {cka_matrix.shape[0]}x{cka_matrix.shape[1]} CKA matrix")
        print(f"  {len(mapping)} layer mappings")
        return mapping, num_layers_b

    print("\n--- CKA LAYER CORRESPONDENCE ---")

    total_rows = index_meta["total_rows"]
    hidden_dim = index_meta["hidden_dim"]
    pass1_meta_path = os.path.join(config.DATA_DIR, "pass1_metadata.json")
    with open(pass1_meta_path) as f:
        pass1_meta = json.load(f)
    num_layers_a = pass1_meta["stored_layers"]

    store_a = np.memmap(
        config.MODEL_A_MMAP_PATH, dtype=np.float16, mode="r",
        shape=(total_rows, num_layers_a, hidden_dim),
    )

    subset_inputs = corpus[:config.CKA_SUBSET_SIZE]
    index = index_meta["index"]

    # Detect num layers for model B
    sample_text = subset_inputs[0]["raw_text"]
    inp = tokenizer_b(sample_text, return_tensors="pt", add_special_tokens=False)
    inp = {k: v.to(model_b.device) for k, v in inp.items()}
    with torch.no_grad():
        out = model_b(**inp, output_hidden_states=True)
    num_layers_b = len(out.hidden_states)
    print(f"  Model B has {num_layers_b} hidden states (embedding + {num_layers_b - 1} layers)")
    del out
    torch.cuda.empty_cache()

    print(f"  Collecting activations from {len(subset_inputs)} inputs...")
    max_vectors = 5000
    vecs_a = {l: [] for l in range(num_layers_a)}
    vecs_b = {l: [] for l in range(num_layers_b)}
    total_collected = 0

    for entry in subset_inputs:
        if total_collected >= max_vectors:
            break

        input_id = entry["input_id"]
        text = entry["raw_text"]
        positions_a = entry[config.CORPUS_POSITIONS_KEY_A]
        positions_b = entry[config.CORPUS_POSITIONS_KEY_B]

        try:
            inp = tokenizer_b(text, return_tensors="pt", add_special_tokens=False)
            inp = {k: v.to(model_b.device) for k, v in inp.items()}
            with torch.no_grad():
                out = model_b(**inp, output_hidden_states=True)
            hidden_states_b = out.hidden_states
        except Exception:
            continue

        for pair_idx, (pos_a, pos_b) in enumerate(zip(positions_a, positions_b)):
            if total_collected >= max_vectors:
                break
            key = f"{input_id}_{pair_idx}"
            if key not in index:
                continue
            row = index[key]

            for l in range(num_layers_a):
                vecs_a[l].append(store_a[row, l, :].astype(np.float32))
            for l in range(min(num_layers_b, len(hidden_states_b))):
                if pos_b < hidden_states_b[l].shape[1]:
                    vecs_b[l].append(hidden_states_b[l][0, pos_b, :].cpu().numpy().astype(np.float32))
                else:
                    vecs_b[l].append(np.zeros(hidden_dim, dtype=np.float32))
            total_collected += 1

        del out, hidden_states_b
        torch.cuda.empty_cache()

    print(f"  Collected {total_collected} activation vectors")

    for l in range(num_layers_a):
        vecs_a[l] = np.stack(vecs_a[l])
    for l in range(num_layers_b):
        vecs_b[l] = np.stack(vecs_b[l])

    print(f"  Computing {num_layers_a}x{num_layers_b} CKA matrix...")
    cka_matrix = np.zeros((num_layers_a, num_layers_b), dtype=np.float32)
    for i in range(num_layers_a):
        for j in range(num_layers_b):
            cka_matrix[i, j] = linear_cka(vecs_a[i], vecs_b[j])
        if (i + 1) % 5 == 0:
            print(f"    Layer {i + 1}/{num_layers_a} done")

    np.save(config.LAYER_CORRESPONDENCE_PATH, cka_matrix)
    print(f"  Saved CKA matrix to {config.LAYER_CORRESPONDENCE_PATH}")

    mapping = {}
    for i in range(num_layers_a):
        best_j = int(np.argmax(cka_matrix[i]))
        mapping[str(i)] = {
            "model_b_layer": best_j,
            "cka_score": float(cka_matrix[i, best_j]),
        }
        print(f"    A layer {i:>2} -> B layer {best_j:>2} (CKA={cka_matrix[i, best_j]:.3f})")

    with open(config.LAYER_MAPPING_PATH, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"  Saved layer mapping to {config.LAYER_MAPPING_PATH}")

    del store_a, vecs_a, vecs_b
    return mapping, num_layers_b


# ---------------------------------------------------------------------------
# Phase 2: Streaming Covariance (with checkpointing)
# ---------------------------------------------------------------------------

class RunningMean:
    """Lightweight accumulator that only tracks the mean (no covariance)."""
    def __init__(self, dim):
        self.dim = dim
        self.count = 0
        self.sum = np.zeros(dim, dtype=np.float64)

    @property
    def mean(self):
        if self.count == 0:
            return np.zeros(self.dim, dtype=np.float64)
        return self.sum / self.count

    def add_batch(self, vecs_f32):
        """vecs_f32: (n, dim) float32 numpy array."""
        self.count += len(vecs_f32)
        self.sum += vecs_f32.astype(np.float64).sum(axis=0)

    def get_state(self):
        return {"count": self.count, "sum": self.sum}

    def set_state(self, state):
        self.count = state["count"]
        self.sum = state["sum"]


def compute_differentials(model_b, tokenizer_b, corpus, index_meta, mapping, num_layers_b):
    """Compute activation differentials and streaming covariance.

    All scatter matrix math stays on GPU. CPU only receives final results
    at checkpoint/save time.
    """
    print("\n--- STREAMING COVARIANCE ---")

    total_rows = index_meta["total_rows"]
    hidden_dim = index_meta["hidden_dim"]
    index = index_meta["index"]

    pass1_meta_path = os.path.join(config.DATA_DIR, "pass1_metadata.json")
    with open(pass1_meta_path) as f:
        pass1_meta = json.load(f)
    num_layers_a = pass1_meta["stored_layers"]

    store_a = np.memmap(
        config.MODEL_A_MMAP_PATH, dtype=np.float16, mode="r",
        shape=(total_rows, num_layers_a, hidden_dim),
    )

    # Determine matched layer pairs
    matched_pairs = []
    for a_layer_str, info in mapping.items():
        matched_pairs.append((int(a_layer_str), info["model_b_layer"]))
    matched_pairs.sort()
    n_matched = len(matched_pairs)
    a_layers = [a for a, b in matched_pairs]
    b_layers = [b for a, b in matched_pairs]
    print(f"  {n_matched} matched layer pairs")

    # Category layer selection
    category_layer_indices = np.linspace(0, n_matched - 1, config.CATEGORY_LAYERS, dtype=int).tolist()
    category_layers = [matched_pairs[i] for i in category_layer_indices]
    # Map (a_l, b_l) -> index in matched_pairs for category layers
    cat_layer_mp_indices = set(category_layer_indices)
    print(f"  Per-category covariance at {config.CATEGORY_LAYERS} layers: "
          f"{[f'A{a}->B{b}' for a, b in category_layers]}")

    categories = sorted(set(e["category"] for e in corpus))
    cat_to_idx = {cat: i for i, cat in enumerate(categories)}

    # --- GPU-resident accumulators ---
    # Overall diff: count (n_matched,), sum (n_matched, dim), M2 (n_matched, dim, dim)
    gpu_count = torch.zeros(n_matched, dtype=torch.long, device="cuda")
    gpu_sum = torch.zeros(n_matched, hidden_dim, dtype=torch.float64, device="cuda")
    gpu_M2 = torch.zeros(n_matched, hidden_dim, hidden_dim, dtype=torch.float32, device="cuda")

    # Mean A/B: just sums
    gpu_sum_a = torch.zeros(n_matched, hidden_dim, dtype=torch.float64, device="cuda")
    gpu_sum_b = torch.zeros(n_matched, hidden_dim, dtype=torch.float64, device="cuda")

    # Per-category diff: only at selected layers
    n_cats = len(categories)
    n_cat_layers = len(category_layer_indices)
    gpu_cat_count = torch.zeros(n_cats, n_cat_layers, dtype=torch.long, device="cuda")
    gpu_cat_sum = torch.zeros(n_cats, n_cat_layers, hidden_dim, dtype=torch.float64, device="cuda")
    gpu_cat_M2 = torch.zeros(n_cats, n_cat_layers, hidden_dim, hidden_dim, dtype=torch.float32, device="cuda")

    print(f"  GPU accumulator VRAM: ~{(gpu_M2.nelement() * 4 + gpu_cat_M2.nelement() * 4) / 1e9:.1f} GB")

    # Try to resume from checkpoint
    last_processed_id = -1
    processed = 0
    nan_count = 0

    if os.path.exists(PASS2_CHECKPOINT_META):
        print("  Loading checkpoint...", flush=True)
        with open(PASS2_CHECKPOINT_META) as f:
            ckpt_meta = json.load(f)
        last_processed_id = ckpt_meta["last_input_id"]
        processed = ckpt_meta["processed"]
        nan_count = ckpt_meta["nan_count"]
        # Load GPU tensors from npz
        ckpt_data = np.load(os.path.join(PASS2_CHECKPOINT_DIR, "gpu_state.npz"))
        gpu_count.copy_(torch.from_numpy(ckpt_data["count"]))
        gpu_sum.copy_(torch.from_numpy(ckpt_data["sum"]))
        gpu_M2.copy_(torch.from_numpy(ckpt_data["M2"]))
        gpu_sum_a.copy_(torch.from_numpy(ckpt_data["sum_a"]))
        gpu_sum_b.copy_(torch.from_numpy(ckpt_data["sum_b"]))
        gpu_cat_count.copy_(torch.from_numpy(ckpt_data["cat_count"]))
        gpu_cat_sum.copy_(torch.from_numpy(ckpt_data["cat_sum"]))
        gpu_cat_M2.copy_(torch.from_numpy(ckpt_data["cat_M2"]))
        print(f"  Resumed from input {last_processed_id} ({processed} processed)", flush=True)

    def _save_checkpoint(input_id):
        os.makedirs(PASS2_CHECKPOINT_DIR, exist_ok=True)
        t0 = time.time()
        np.savez(os.path.join(PASS2_CHECKPOINT_DIR, "gpu_state.npz"),
                 count=gpu_count.cpu().numpy(),
                 sum=gpu_sum.cpu().numpy(),
                 M2=gpu_M2.cpu().numpy(),
                 sum_a=gpu_sum_a.cpu().numpy(),
                 sum_b=gpu_sum_b.cpu().numpy(),
                 cat_count=gpu_cat_count.cpu().numpy(),
                 cat_sum=gpu_cat_sum.cpu().numpy(),
                 cat_M2=gpu_cat_M2.cpu().numpy())
        with open(PASS2_CHECKPOINT_META, "w") as f:
            json.dump({"last_input_id": input_id, "nan_count": nan_count,
                        "processed": processed}, f)
        print(f"  Checkpoint at input {input_id} in {time.time() - t0:.1f}s", flush=True)

    # Process corpus
    start_time = time.time()
    new_processed = 0

    for entry in corpus:
        input_id = entry["input_id"]
        if input_id <= last_processed_id:
            continue

        text = entry["raw_text"]
        positions_a = entry[config.CORPUS_POSITIONS_KEY_A]
        positions_b = entry[config.CORPUS_POSITIONS_KEY_B]
        category = entry["category"]
        cat_idx = cat_to_idx[category]

        # Model B inference
        try:
            inp = tokenizer_b(text, return_tensors="pt", add_special_tokens=False)
            inp = {k: v.to(model_b.device) for k, v in inp.items()}
            with torch.no_grad():
                out = model_b(**inp, output_hidden_states=True)
            hidden_states_b = out.hidden_states
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                continue
            raise
        except Exception:
            continue

        # Collect valid rows/positions
        valid_rows = []
        valid_pos_b = []
        for pair_idx, (pos_a, pos_b) in enumerate(zip(positions_a, positions_b)):
            key = f"{input_id}_{pair_idx}"
            if key not in index:
                continue
            valid_rows.append(index[key])
            valid_pos_b.append(pos_b)

        if not valid_rows:
            del out, hidden_states_b
            processed += 1
            new_processed += 1
            continue

        rows_arr = np.array(valid_rows)
        pos_b_arr = np.array(valid_pos_b)

        # Process all matched layer pairs
        for mp_idx, (a_l, b_l) in enumerate(matched_pairs):
            if b_l >= len(hidden_states_b):
                continue
            hs_b = hidden_states_b[b_l]
            seq_len_b = hs_b.shape[1]

            mask = pos_b_arr < seq_len_b
            if not mask.any():
                continue
            cur_pos = pos_b_arr[mask] if not mask.all() else pos_b_arr
            cur_rows = rows_arr[mask] if not mask.all() else rows_arr

            # Load vecs to GPU
            vecs_a_np = store_a[cur_rows, a_l, :]
            t_a = torch.from_numpy(vecs_a_np.astype(np.float32)).cuda()
            t_b = hs_b[0, cur_pos, :].float()

            # NaN filter
            nan_mask = torch.isnan(t_a).any(dim=1) | torch.isnan(t_b).any(dim=1)
            if nan_mask.any():
                nan_count += nan_mask.sum().item()
                keep = ~nan_mask
                t_a = t_a[keep]
                t_b = t_b[keep]

            n = t_a.shape[0]
            if n == 0:
                continue

            t_diff = t_a - t_b

            # Chan's parallel merge — entirely on GPU
            n_a = gpu_count[mp_idx].item()
            n_b = n
            n_ab = n_a + n_b

            batch_mean = t_diff.mean(dim=0)  # (dim,) float32
            centered = t_diff - batch_mean
            batch_M2 = centered.T @ centered  # (dim, dim) float32 on GPU

            if n_a == 0:
                gpu_sum[mp_idx] = batch_mean.double() * n_b
                gpu_M2[mp_idx] = batch_M2
            else:
                old_mean = gpu_sum[mp_idx].float() / n_a
                delta = batch_mean - old_mean  # (dim,) float32
                correction = torch.outer(delta, delta) * (n_a * n_b / n_ab)
                gpu_M2[mp_idx] += batch_M2 + correction
                gpu_sum[mp_idx] += batch_mean.double() * n_b
            gpu_count[mp_idx] = n_ab

            # Running means for A/B
            gpu_sum_a[mp_idx] += t_a.double().sum(dim=0)
            gpu_sum_b[mp_idx] += t_b.double().sum(dim=0)

            # Per-category
            if mp_idx in cat_layer_mp_indices:
                cl_idx = category_layer_indices.index(mp_idx)
                n_ca = gpu_cat_count[cat_idx, cl_idx].item()
                n_cab = n_ca + n_b
                if n_ca == 0:
                    gpu_cat_sum[cat_idx, cl_idx] = batch_mean.double() * n_b
                    gpu_cat_M2[cat_idx, cl_idx] = batch_M2
                else:
                    old_cat_mean = gpu_cat_sum[cat_idx, cl_idx].float() / n_ca
                    delta_c = batch_mean - old_cat_mean
                    gpu_cat_M2[cat_idx, cl_idx] += batch_M2 + torch.outer(delta_c, delta_c) * (n_ca * n_b / n_cab)
                    gpu_cat_sum[cat_idx, cl_idx] += batch_mean.double() * n_b
                gpu_cat_count[cat_idx, cl_idx] = n_cab

        del out, hidden_states_b
        if new_processed % 100 == 0:
            torch.cuda.empty_cache()

        processed += 1
        new_processed += 1

        if new_processed % config.PROGRESS_INTERVAL == 0:
            elapsed = time.time() - start_time
            rate = new_processed / elapsed
            remaining_inputs = len(corpus) - (last_processed_id + 1 + new_processed)
            remaining = remaining_inputs / rate / 60 if rate > 0 else 0
            print(f"  [{processed}/{len(corpus)}] {rate:.1f} inputs/sec, "
                  f"~{remaining:.1f} min remaining, NaN: {nan_count}", flush=True)

        if new_processed % config.CHECKPOINT_INTERVAL == 0:
            _save_checkpoint(input_id)

    elapsed = time.time() - start_time
    print(f"\n  Differential computation complete.", flush=True)
    print(f"  Processed: {processed} total ({new_processed} this run) in {elapsed / 60:.1f} minutes")
    print(f"  NaN/Inf skipped: {nan_count}")

    # Save final results — transfer GPU tensors to CPU and save as numpy
    print("\n  Saving overall statistics...")
    counts_np = gpu_count.cpu().numpy()
    sums_np = gpu_sum.cpu().numpy()
    M2s_np = gpu_M2.cpu().numpy()
    sums_a_np = gpu_sum_a.cpu().numpy()
    sums_b_np = gpu_sum_b.cpu().numpy()

    for mp_idx, (a_l, b_l) in enumerate(matched_pairs):
        prefix = f"layer_A{a_l}_B{b_l}"
        cnt = counts_np[mp_idx]
        mean_diff = (sums_np[mp_idx] / cnt).astype(np.float32) if cnt > 0 else np.zeros(hidden_dim, dtype=np.float32)
        cov_diff = (M2s_np[mp_idx] / (cnt - 1)).astype(np.float32) if cnt > 1 else np.zeros((hidden_dim, hidden_dim), dtype=np.float32)
        np.save(os.path.join(config.OVERALL_STATS_DIR, f"{prefix}_mean_diff.npy"), mean_diff)
        np.save(os.path.join(config.OVERALL_STATS_DIR, f"{prefix}_cov_diff.npy"), cov_diff)
        mean_a = (sums_a_np[mp_idx] / cnt).astype(np.float32) if cnt > 0 else np.zeros(hidden_dim, dtype=np.float32)
        mean_b = (sums_b_np[mp_idx] / cnt).astype(np.float32) if cnt > 0 else np.zeros(hidden_dim, dtype=np.float32)
        np.save(os.path.join(config.OVERALL_STATS_DIR, f"{prefix}_mean_a.npy"), mean_a)
        np.save(os.path.join(config.OVERALL_STATS_DIR, f"{prefix}_mean_b.npy"), mean_b)
    print(f"  Saved to {config.OVERALL_STATS_DIR}/")

    print("  Saving per-category statistics...")
    cat_counts_np = gpu_cat_count.cpu().numpy()
    cat_sums_np = gpu_cat_sum.cpu().numpy()
    cat_M2s_np = gpu_cat_M2.cpu().numpy()

    for ci, cat in enumerate(categories):
        for cli, cl_mp_idx in enumerate(category_layer_indices):
            a_l, b_l = matched_pairs[cl_mp_idx]
            prefix = f"layer_A{a_l}_B{b_l}"
            cat_dir = os.path.join(config.CATEGORY_STATS_DIR, cat)
            cnt = cat_counts_np[ci, cli]
            mean_diff = (cat_sums_np[ci, cli] / cnt).astype(np.float32) if cnt > 0 else np.zeros(hidden_dim, dtype=np.float32)
            cov_diff = (cat_M2s_np[ci, cli] / (cnt - 1)).astype(np.float32) if cnt > 1 else np.zeros((hidden_dim, hidden_dim), dtype=np.float32)
            np.save(os.path.join(cat_dir, f"{prefix}_mean_diff.npy"), mean_diff)
            np.save(os.path.join(cat_dir, f"{prefix}_cov_diff.npy"), cov_diff)
    print(f"  Saved to {config.CATEGORY_STATS_DIR}/")

    # Save metadata
    meta = {
        "matched_pairs": [{"a_layer": a, "b_layer": b} for a, b in matched_pairs],
        "category_layers": [{"a_layer": a, "b_layer": b} for a, b in category_layers],
        "category_layer_indices": category_layer_indices,
        "total_processed": processed,
        "nan_count": nan_count,
        "num_layers_a": num_layers_a,
        "num_layers_b": num_layers_b,
        "hidden_dim": hidden_dim,
        "categories": categories,
        "overall_sample_counts": {
            f"A{a}_B{b}": int(counts_np[i])
            for i, (a, b) in enumerate(matched_pairs)
        },
    }
    with open(os.path.join(config.RESULTS_DIR, "pass2_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Clean up checkpoint after successful completion
    if os.path.exists(PASS2_CHECKPOINT_DIR):
        import shutil
        shutil.rmtree(PASS2_CHECKPOINT_DIR)
        print("  Checkpoint cleaned up.")

    del store_a


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("PASS 2: ACTIVATION DIFFERENTIALS & STREAMING COVARIANCE")
    print("=" * 70)

    # Load corpus
    print("\nLoading corpus...")
    corpus, index_meta = load_corpus()
    print(f"  {len(corpus)} inputs, {index_meta['total_rows']} aligned pairs")

    # Verify Pass 1 output exists
    if not os.path.exists(config.MODEL_A_MMAP_PATH):
        print(f"  ERROR: Model A activations not found at {config.MODEL_A_MMAP_PATH}")
        print(f"  Run pass1_extract.py first.")
        sys.exit(1)

    # Load Model B
    print("\nLoading Model B...")
    model_b, tokenizer_b = load_model(config.MODEL_B_ID, config.MODEL_B_FALLBACKS, config.MODEL_B_NAME)

    # Phase 1: CKA layer correspondence (skips if cached)
    mapping, num_layers_b = compute_layer_correspondence(
        model_b, tokenizer_b, corpus, index_meta
    )

    # Phase 2: Streaming covariance (resumable)
    compute_differentials(model_b, tokenizer_b, corpus, index_meta, mapping, num_layers_b)

    # Cleanup
    print("\nCleaning up GPU memory...")
    del model_b
    torch.cuda.empty_cache()
    print("Pass 2 complete.")


if __name__ == "__main__":
    main()
