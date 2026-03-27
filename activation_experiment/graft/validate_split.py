#!/usr/bin/env python3
"""
Graft Experiment — Step 0: Validate Partial Forward Pass

Two approaches tested:
1. Hook-based: capture intermediate hidden states from full forward pass,
   then feed them through remaining layers manually.
2. Direct: call model.model.forward() with a modified layer range.

Tolerance: cosine similarity > 0.9999, max absolute logit diff < 0.01
"""

import json
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def load_llama():
    token = os.environ.get("HF_TOKEN") or True
    for mid in [config.MODEL_B_ID] + config.MODEL_B_FALLBACKS:
        try:
            print(f"  Loading from {mid} ...", flush=True)
            cfg = AutoConfig.from_pretrained(mid, trust_remote_code=True, token=token)
            if not hasattr(cfg, "max_length"):
                cfg.max_length = getattr(cfg, "seq_length", 8192)
            model = AutoModelForCausalLM.from_pretrained(
                mid, config=cfg, torch_dtype=torch.float16, device_map="cuda",
                trust_remote_code=True, token=token,
            )
            tokenizer = AutoTokenizer.from_pretrained(mid, trust_remote_code=True, token=token)
            print(f"  Loaded: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B params")
            return model, tokenizer
        except Exception as e:
            print(f"  FAILED ({str(e)[:80]})")
    print("  *** Could not load Llama ***")
    sys.exit(1)


def full_forward(model, input_ids):
    """Run full forward pass, return logits and all hidden states."""
    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True)
    return out.logits, out.hidden_states


def split_forward(model, hidden_state_at_split, split_after_layer, input_ids):
    """
    Take a hidden state captured after `split_after_layer` from the full pass,
    and run it through the remaining layers + norm + lm_head.

    Uses the model's own layer modules but calls them via the internal
    _update_causal_mask for correct masking.
    """
    llama = model.model
    device = input_ids.device
    batch_size, seq_len = input_ids.shape

    with torch.no_grad():
        hidden = hidden_state_at_split

        # Prepare position_ids and cache_position
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        cache_position = torch.arange(seq_len, device=device)

        # Use model's own causal mask preparation
        causal_mask = llama._update_causal_mask(
            None, hidden, cache_position, past_key_values=None, output_attentions=False
        )

        # Compute position embeddings (for models that use them)
        position_embeddings = None
        if hasattr(llama, "rotary_emb"):
            position_embeddings = llama.rotary_emb(hidden, position_ids)

        # Run remaining layers
        for layer in llama.layers[split_after_layer + 1:]:
            kwargs = {
                "attention_mask": causal_mask,
                "position_ids": position_ids,
            }
            if position_embeddings is not None:
                kwargs["position_embeddings"] = position_embeddings
            layer_out = layer(hidden, **kwargs)
            hidden = layer_out[0]

        # Final norm + LM head
        hidden = llama.norm(hidden)
        logits = model.lm_head(hidden)

    return logits


def compare_logits(logits_full, logits_split, label):
    """Compare two logit tensors and report metrics."""
    l1 = logits_full.float().reshape(-1, logits_full.shape[-1])
    l2 = logits_split.float().reshape(-1, logits_split.shape[-1])

    cos = torch.nn.functional.cosine_similarity(l1, l2, dim=-1)
    cos_mean = cos.mean().item()
    cos_min = cos.min().item()

    max_abs_diff = (l1 - l2).abs().max().item()
    mean_abs_diff = (l1 - l2).abs().mean().item()

    top1_full = l1.argmax(dim=-1)
    top1_split = l2.argmax(dim=-1)
    top1_agree = (top1_full == top1_split).float().mean().item()

    passed = cos_min > 0.9999 and max_abs_diff < 0.01

    print(f"\n  {label}:")
    print(f"    Cosine sim:    mean={cos_mean:.6f}  min={cos_min:.6f}  {'PASS' if cos_min > 0.9999 else 'FAIL'}")
    print(f"    Max abs diff:  {max_abs_diff:.6f}  {'PASS' if max_abs_diff < 0.01 else 'FAIL'}")
    print(f"    Mean abs diff: {mean_abs_diff:.6f}")
    print(f"    Top-1 agree:   {top1_agree:.4f}")
    print(f"    {'PASS' if passed else '*** FAIL ***'}")

    return passed, {
        "cos_mean": cos_mean, "cos_min": cos_min,
        "max_abs_diff": max_abs_diff, "mean_abs_diff": mean_abs_diff,
        "top1_agreement": top1_agree, "passed": passed,
    }


def main():
    print("=" * 70)
    print("GRAFT STEP 0: VALIDATE PARTIAL FORWARD PASS")
    print("=" * 70)

    # Llama layers to validate splitting after
    # These correspond to the splice points: S0(0), S2/S3(15-1=14), S1(23-1=22), S4(31-1=30), S5(32-1=31)
    # split_after_layer means: layers 0..L run first, layers L+1..end run second
    # For splice at Llama layer L', we split after L'-1 (the hidden state output of layer L'-1
    # is the input to layer L')
    # But hidden_states[L] is the output of transformer layer L-1 (hidden_states[0] = embedding)
    # So hidden_states[L'] is what we'd feed into layer L'
    split_after_indices = {
        "S0: embed->layer0": 0,     # hidden_states[0] = embedding, feed into layer 0
        "S2/S3: ->layer15": 15,     # hidden_states[15] = output of layer 14, feed into layer 15
        "S1: ->layer23": 23,        # hidden_states[23] = output of layer 22, feed into layer 23
        "S4: ->layer31": 31,        # hidden_states[31] = output of layer 30, feed into layer 31
    }

    print("\nLoading Llama...", flush=True)
    model, tokenizer = load_llama()

    num_layers = len(model.model.layers)
    print(f"  Model has {num_layers} transformer layers")
    print(f"  output_hidden_states will give {num_layers + 1} tensors (embedding + {num_layers} layers)")

    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "In 1789, the French Revolution began when citizens stormed the Bastille in Paris.",
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        "Depuis la transition vers la democratie, le pays a connu une croissance.",
        "Calculate the area: A = pi * r^2 where r = 5. So A = 3.14159 * 25 = 78.54.",
    ]

    results = {}
    all_passed = True

    for text in test_texts:
        print(f"\n{'='*60}")
        print(f"  Input: \"{text[:70]}\"")
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs["input_ids"].to("cuda")
        print(f"  Tokens: {input_ids.shape[1]}")

        # Full forward pass — get logits and all hidden states
        logits_full, hidden_states = full_forward(model, input_ids)

        for name, hs_index in split_after_indices.items():
            if hs_index >= len(hidden_states):
                print(f"\n  {name}: SKIP (index {hs_index} >= {len(hidden_states)} hidden states)")
                continue

            # hidden_states[hs_index] is the representation BEFORE transformer layer hs_index
            # (hidden_states[0] = embedding, hidden_states[1] = output of layer 0, etc.)
            # To feed into layer L', we take hidden_states[L'] and run layers L'..end
            intermediate = hidden_states[hs_index]

            # split_forward runs layers[hs_index:] on the intermediate
            # But we named the param split_after_layer, which means layers[split_after_layer+1:]
            # So split_after_layer = hs_index - 1
            logits_split = split_forward(model, intermediate, hs_index - 1, input_ids)

            passed, metrics = compare_logits(logits_full, logits_split, name)
            if not passed:
                all_passed = False
            results[f"{name}|{text[:30]}"] = metrics

    print(f"\n{'='*70}")
    if all_passed:
        print("  ALL SPLITS PASSED")
        print("  Partial forward pass infrastructure is validated.")
    else:
        print("  *** SOME SPLITS FAILED ***")
        print("  Debug the mask/position handling before proceeding.")
    print(f"{'='*70}")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "validation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    del model
    torch.cuda.empty_cache()
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
