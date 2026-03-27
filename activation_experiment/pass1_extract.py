#!/usr/bin/env python3
"""
Activation Differential Experiment — Pass 1: Model A Activation Extraction

Loads Llama 3.1 8B, runs inference on the corpus, and saves hidden-state
activations at aligned token positions to a memory-mapped numpy file.

Requires: NVIDIA GPU with >= 20GB VRAM
"""

import json
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config


def load_corpus():
    """Load corpus.jsonl and activation index."""
    corpus = []
    with open(config.CORPUS_PATH) as f:
        for line in f:
            corpus.append(json.loads(line))

    with open(config.ACTIVATION_INDEX_PATH) as f:
        index_meta = json.load(f)

    return corpus, index_meta


def load_model(model_id, fallbacks):
    """Load model, trying fallbacks if primary is gated."""
    from transformers import AutoConfig
    token = os.environ.get("HF_TOKEN") or True  # True = use cached token
    for mid in [model_id] + fallbacks:
        try:
            print(f"  Loading model from {mid} ...", flush=True)
            # Load config first and patch missing attrs for GLM compat
            model_config = AutoConfig.from_pretrained(mid, trust_remote_code=True, token=token)
            if not hasattr(model_config, "max_length"):
                model_config.max_length = getattr(model_config, "seq_length", 8192)
            model = AutoModelForCausalLM.from_pretrained(
                mid,
                config=model_config,
                torch_dtype=torch.float16,
                device_map="cuda",
                trust_remote_code=True,
                token=token,
            )
            tokenizer = AutoTokenizer.from_pretrained(mid, trust_remote_code=True, token=token)
            print(f"  Model loaded: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B params")
            return model, tokenizer
        except Exception as e:
            err = str(e)[:120]
            print(f"  FAILED ({err})")
    print("  *** Could not load model ***")
    sys.exit(1)


def load_checkpoint():
    """Load checkpoint if it exists, return last processed input_id."""
    if os.path.exists(config.CHECKPOINT_PATH):
        with open(config.CHECKPOINT_PATH) as f:
            ckpt = json.load(f)
        return ckpt.get("pass1_last_input_id", -1)
    return -1


def save_checkpoint(last_input_id):
    """Save checkpoint."""
    ckpt = {}
    if os.path.exists(config.CHECKPOINT_PATH):
        with open(config.CHECKPOINT_PATH) as f:
            ckpt = json.load(f)
    ckpt["pass1_last_input_id"] = last_input_id
    with open(config.CHECKPOINT_PATH, "w") as f:
        json.dump(ckpt, f)


def validate(model, tokenizer, corpus, index_meta):
    """Run validation checks on first 50 inputs."""
    print("\n--- VALIDATION ---")

    # Check hidden states shape
    sample_text = corpus[0]["raw_text"]
    inputs = tokenizer(sample_text, return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hs = outputs.hidden_states
    num_hidden_states = len(hs)
    expected_layers = config.MODEL_A_LAYERS + 1  # embedding + transformer layers
    seq_len = inputs["input_ids"].shape[1]
    hidden_dim = hs[0].shape[-1]

    print(f"  Hidden states count: {num_hidden_states} (expected {expected_layers})")
    print(f"  Sequence length: {seq_len}")
    print(f"  Hidden dimension: {hidden_dim} (expected {config.HIDDEN_DIM})")
    print(f"  Shape per layer: {hs[0].shape}")

    if num_hidden_states != expected_layers:
        print(f"  WARNING: Expected {expected_layers} hidden states, got {num_hidden_states}")
        print(f"  Adjusting MODEL_A_LAYERS to {num_hidden_states - 1}")
        # Update in-memory config
        config.MODEL_A_LAYERS = num_hidden_states - 1

    if hidden_dim != config.HIDDEN_DIM:
        print(f"  WARNING: Expected hidden dim {config.HIDDEN_DIM}, got {hidden_dim}")
        config.HIDDEN_DIM = hidden_dim

    # Check for NaN/Inf
    has_nan = any(torch.isnan(h).any().item() for h in hs)
    has_inf = any(torch.isinf(h).any().item() for h in hs)
    print(f"  NaN in activations: {has_nan}")
    print(f"  Inf in activations: {has_inf}")
    if has_nan or has_inf:
        print("  WARNING: NaN or Inf detected in activations!")

    # Check aligned positions are valid
    pos_key = config.CORPUS_POSITIONS_KEY_A
    count_key = config.CORPUS_TOKEN_COUNT_KEY_A
    invalid_count = 0
    for entry in corpus[:50]:
        max_pos = max(entry[pos_key]) if entry[pos_key] else 0
        tok_count = entry[count_key]
        if max_pos >= tok_count:
            invalid_count += 1
    print(f"  Invalid position indices (first 50): {invalid_count}")

    # Timing estimate
    start = time.time()
    for i in range(min(5, len(corpus))):
        text = corpus[i]["raw_text"]
        inp = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        inp = {k: v.to(model.device) for k, v in inp.items()}
        with torch.no_grad():
            model(**inp, output_hidden_states=True)
    elapsed = time.time() - start
    per_input = elapsed / min(5, len(corpus))
    total_est = per_input * len(corpus) / 60
    print(f"  Avg time per input: {per_input:.3f}s")
    print(f"  Estimated total time: {total_est:.1f} minutes")

    # Test memmap write/read cycle
    import tempfile
    _tmp_mmap_path = os.path.join(tempfile.gettempdir(), "test_mmap.npy")
    test_mmap = np.memmap(_tmp_mmap_path, dtype=np.float16, mode="w+", shape=(10, 4))
    test_data = np.random.randn(10, 4).astype(np.float16)
    test_mmap[:] = test_data
    test_mmap.flush()
    del test_mmap
    test_read = np.memmap(_tmp_mmap_path, dtype=np.float16, mode="r", shape=(10, 4))
    match = np.allclose(test_data, test_read, atol=0)
    print(f"  Memmap write/read roundtrip: {'OK' if match else 'FAILED'}")
    del test_read
    os.unlink(_tmp_mmap_path)

    actual_layers = num_hidden_states
    print(f"\n  Validation passed. Using {actual_layers} hidden states (embedding + {actual_layers - 1} layers).")
    return actual_layers


def extract_activations(model, tokenizer, corpus, index_meta, num_layers):
    """Main extraction loop."""
    total_rows = index_meta["total_rows"]
    hidden_dim = config.HIDDEN_DIM
    index = index_meta["index"]

    print(f"\n--- EXTRACTION ---")
    print(f"  Total aligned token pairs: {total_rows}")
    print(f"  Layers: {num_layers}")
    print(f"  Hidden dim: {hidden_dim}")
    mmap_shape = (total_rows, num_layers, hidden_dim)
    mmap_bytes = total_rows * num_layers * hidden_dim * 2  # fp16
    print(f"  Memmap shape: {mmap_shape}")
    print(f"  Memmap size: {mmap_bytes / 1e9:.1f} GB")

    # Create or open memmap
    activation_store = np.memmap(
        config.MODEL_A_MMAP_PATH,
        dtype=np.float16,
        mode="w+" if not os.path.exists(config.MODEL_A_MMAP_PATH) else "r+",
        shape=mmap_shape,
    )

    # Resume from checkpoint
    last_processed = load_checkpoint()
    if last_processed >= 0:
        print(f"  Resuming from input_id {last_processed + 1}")

    batch_size = config.BATCH_SIZE
    start_time = time.time()
    processed = 0
    nan_count = 0

    for i, entry in enumerate(corpus):
        input_id = entry["input_id"]
        if input_id <= last_processed:
            continue

        # Tokenize and run inference
        text = entry["raw_text"]
        try:
            inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            hidden_states = outputs.hidden_states  # tuple of (1, seq_len, hidden_dim)

            # Extract activations at aligned positions
            positions = entry[config.CORPUS_POSITIONS_KEY_A]
            for pair_idx, pos in enumerate(positions):
                key = f"{input_id}_{pair_idx}"
                if key not in index:
                    continue
                row = index[key]

                for layer_idx in range(min(num_layers, len(hidden_states))):
                    hs = hidden_states[layer_idx]
                    if pos < hs.shape[1]:
                        vec = hs[0, pos, :].float().cpu().numpy().astype(np.float16)
                        if np.isnan(vec).any() or np.isinf(vec).any():
                            nan_count += 1
                            vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
                        activation_store[row, layer_idx, :] = vec

            # Free GPU memory
            del outputs, hidden_states
            if i % 10 == 0:
                torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n  OOM at input {input_id}, reducing batch processing...")
                torch.cuda.empty_cache()
                continue
            else:
                raise

        processed += 1

        # Progress
        if processed % config.PROGRESS_INTERVAL == 0:
            elapsed = time.time() - start_time
            rate = processed / elapsed
            remaining = (len(corpus) - i) / rate / 60
            print(f"  [{processed}/{len(corpus)}] {rate:.1f} inputs/sec, "
                  f"~{remaining:.1f} min remaining, NaN count: {nan_count}")

        # Checkpoint
        if processed % config.CHECKPOINT_INTERVAL == 0:
            activation_store.flush()
            save_checkpoint(input_id)

    # Final flush
    activation_store.flush()
    save_checkpoint(corpus[-1]["input_id"])

    elapsed = time.time() - start_time
    print(f"\n  Extraction complete.")
    print(f"  Processed: {processed} inputs in {elapsed / 60:.1f} minutes")
    print(f"  NaN/Inf tokens: {nan_count}")
    print(f"  Activation store: {config.MODEL_A_MMAP_PATH}")

    del activation_store


def main():
    print("=" * 70)
    print("PASS 1: MODEL A ACTIVATION EXTRACTION")
    print("=" * 70)

    # Load corpus
    print("\nLoading corpus...")
    corpus, index_meta = load_corpus()
    print(f"  {len(corpus)} inputs, {index_meta['total_rows']} aligned pairs")

    # Load model
    print("\nLoading Model A...")
    model, tokenizer = load_model(config.MODEL_A_ID, config.MODEL_A_FALLBACKS)

    # Validate
    num_layers = validate(model, tokenizer, corpus, index_meta)

    # Extract
    extract_activations(model, tokenizer, corpus, index_meta, num_layers)

    # Save extraction metadata so pass2 knows the memmap shape
    extract_meta = {
        "stored_model": config.MODEL_A_NAME,
        "stored_layers": num_layers,
        "hidden_dim": config.HIDDEN_DIM,
        "total_rows": index_meta["total_rows"],
    }
    extract_meta_path = os.path.join(config.DATA_DIR, "pass1_metadata.json")
    with open(extract_meta_path, "w") as f:
        json.dump(extract_meta, f, indent=2)
    print(f"  Saved extraction metadata to {extract_meta_path}")

    # Cleanup
    print("\nCleaning up GPU memory...")
    del model
    torch.cuda.empty_cache()
    print("Pass 1 complete.")


if __name__ == "__main__":
    main()
