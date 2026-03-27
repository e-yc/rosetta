#!/usr/bin/env python3
"""
Additive Injection — Generation Test

Autoregressive generation with additive injection at alpha=0.5.
Llama runs normally with KV cache. At layer 15, the GLM delta is injected
into the residual stream via a forward hook.

Compares: native Llama, native GLM, and injected chimera.
"""

import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoConfig, AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

GRAFT_DIR = os.path.dirname(os.path.abspath(__file__))
HIDDEN_DIM = config.HIDDEN_DIM
GLM_LAYER = 5
LLAMA_SPLICE_LAYER = 15
ALPHA = 0.5
MAX_NEW_TOKENS = 150

PROMPTS = [
    # Code
    ("code", "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n"),
    ("code", "def merge_sort(arr):\n    \"\"\"Sort array using merge sort.\"\"\"\n"),
    ("code", "SELECT name, department, salary FROM employees WHERE"),
    # English
    ("english", "The discovery of penicillin in 1928 revolutionized medicine because"),
    ("english", "The three main branches of the United States government are"),
    # Conversational
    ("conversational", "Explain the difference between a stack and a queue in simple terms."),
    ("conversational", "What are three practical tips for learning a new programming language?"),
    # Math
    ("math", "A store sells apples for $2 each and oranges for $3 each. If Sarah buys 5 apples and 3 oranges, how much does she pay? Let's solve step by step."),
    # Chinese
    ("chinese", "请解释什么是人工智能，以及它在日常生活中的应用。"),
    ("chinese", "中国的首都是哪里？请详细介绍一下这座城市。"),
    # French
    ("french", "La revolution industrielle a commence en Angleterre au"),
    # Mixed
    ("mixed", "Here is a Python function that calculates the area of a circle:\n```python\nimport math\ndef area("),
]


def load_models():
    token = os.environ.get("HF_TOKEN") or True

    print("  Loading GLM (int8)...", flush=True)
    glm_cfg = AutoConfig.from_pretrained(config.MODEL_A_ID, trust_remote_code=True, token=token)
    if not hasattr(glm_cfg, "max_length"):
        glm_cfg.max_length = getattr(glm_cfg, "seq_length", 8192)
    glm_model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_A_ID, config=glm_cfg,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map="cuda", trust_remote_code=True, token=token,
    )
    glm_tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_A_ID, trust_remote_code=True, token=token
    )
    print(f"  GLM: {torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

    print("  Loading Llama (fp16)...", flush=True)
    llama_cfg = AutoConfig.from_pretrained(config.MODEL_B_ID, trust_remote_code=True, token=token)
    if not hasattr(llama_cfg, "max_length"):
        llama_cfg.max_length = getattr(llama_cfg, "seq_length", 8192)
    llama_model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_B_ID, config=llama_cfg,
        torch_dtype=torch.float16, device_map="cuda",
        trust_remote_code=True, token=token,
    )
    llama_tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_B_ID, trust_remote_code=True, token=token
    )
    print(f"  Both: {torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

    return glm_model, glm_tokenizer, llama_model, llama_tokenizer


def load_bridge():
    bridge = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=True).cuda().float()
    for path in [
        os.path.join(GRAFT_DIR, "bridge", "procrustes_init_best.pt"),
        os.path.join(GRAFT_DIR, "bridge", "procrustes_init_epoch3.pt"),
    ]:
        if os.path.exists(path):
            bridge.load_state_dict(torch.load(path, map_location="cuda", weights_only=True))
            print(f"  Bridge: {os.path.basename(path)}")
            break
    bridge.eval()
    return bridge


class InjectionHook:
    """
    Forward hook that injects GLM delta into Llama's residual stream
    at the target layer.

    Call set_delta() before each Llama forward pass with the GLM-derived delta.
    The hook adds alpha * delta to the layer's input hidden states.
    """

    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.delta = None  # (1, seq_len, dim) or None
        self.handle = None

    def set_delta(self, delta):
        """Set the delta to inject. Pass None to disable injection."""
        self.delta = delta

    def hook_fn(self, module, args, kwargs):
        """Pre-forward hook: modify the hidden_states input."""
        if self.delta is None:
            return args, kwargs

        hidden_states = args[0] if args else kwargs.get("hidden_states")
        if hidden_states is None:
            return args, kwargs

        # Align sequence lengths
        min_len = min(hidden_states.shape[1], self.delta.shape[1])
        injection = self.delta[:, :min_len, :].to(hidden_states.dtype)

        modified = hidden_states.clone()
        modified[:, :min_len, :] = hidden_states[:, :min_len, :] + self.alpha * injection

        if args:
            return (modified,) + args[1:], kwargs
        else:
            kwargs["hidden_states"] = modified
            return args, kwargs

    def register(self, layer):
        self.handle = layer.register_forward_pre_hook(self.hook_fn, with_kwargs=True)

    def remove(self):
        if self.handle:
            self.handle.remove()


@torch.no_grad()
def generate_injected(prompt, glm_model, glm_tokenizer, llama_model, llama_tokenizer,
                       bridge, injection_hook, max_new_tokens):
    """
    Generate with additive injection.
    Llama runs its full forward pass normally (with KV cache).
    At layer 15, the hook injects the GLM delta.
    """
    # Tokenize prompt for both models
    llama_ids = llama_tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to("cuda")
    current_text = prompt
    generated_ids = []

    for step in range(max_new_tokens):
        # Get GLM layer 5 hidden states for current text
        glm_input = glm_tokenizer(current_text, return_tensors="pt", add_special_tokens=False)
        glm_ids = glm_input["input_ids"].to("cuda")
        glm_out = glm_model(input_ids=glm_ids, output_hidden_states=True)
        h_glm = glm_out.hidden_states[GLM_LAYER]  # (1, glm_seq, dim)
        del glm_out

        # Project through bridge
        h_glm_projected = bridge(h_glm.float()).half()

        # Get Llama's own layer 15 hidden states (run without injection first)
        injection_hook.set_delta(None)  # disable for this pass
        llama_input = llama_tokenizer(current_text, return_tensors="pt", add_special_tokens=False)
        llama_ids_full = llama_input["input_ids"].to("cuda")
        llama_out = llama_model(input_ids=llama_ids_full, output_hidden_states=True)
        h_llama_15 = llama_out.hidden_states[LLAMA_SPLICE_LAYER]
        del llama_out

        # Compute delta
        min_len = min(h_glm_projected.shape[1], h_llama_15.shape[1])
        delta = h_glm_projected[:, :min_len, :] - h_llama_15[:, :min_len, :]

        # Now run Llama WITH injection
        injection_hook.set_delta(delta)
        llama_out_injected = llama_model(input_ids=llama_ids_full)
        logits = llama_out_injected.logits
        del llama_out_injected

        # Greedy next token
        next_token_id = logits[0, -1, :].argmax().item()
        next_token_str = llama_tokenizer.decode([next_token_id])

        if next_token_id == llama_tokenizer.eos_token_id:
            break
        if next_token_str in ("<|eot_id|>", "<|end|>", "</s>"):
            break

        generated_ids.append(next_token_id)
        current_text += next_token_str

    injection_hook.set_delta(None)
    return current_text, generated_ids


@torch.no_grad()
def generate_native(prompt, model, tokenizer, max_new_tokens):
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"].to("cuda")
    output = model.generate(
        input_ids, max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)


def main():
    print("=" * 70)
    print(f"INJECTION GENERATION TEST (alpha={ALPHA})")
    print("=" * 70)

    print("\nLoading models...", flush=True)
    glm_model, glm_tokenizer, llama_model, llama_tokenizer = load_models()
    bridge = load_bridge()

    # Register injection hook at Llama layer 15
    injection_hook = InjectionHook(alpha=ALPHA)
    target_layer = llama_model.model.layers[LLAMA_SPLICE_LAYER]
    injection_hook.register(target_layer)
    print(f"  Injection hook registered at Llama layer {LLAMA_SPLICE_LAYER}")

    results = []

    for i, (category, prompt) in enumerate(PROMPTS):
        print(f"\n{'='*60}")
        print(f"  Prompt {i+1}/{len(PROMPTS)} [{category}]:")
        print(f"  \"{prompt[:80]}\"")
        print(f"{'='*60}")

        # Injected generation
        t0 = time.time()
        try:
            inject_text, inject_tokens = generate_injected(
                prompt, glm_model, glm_tokenizer, llama_model, llama_tokenizer,
                bridge, injection_hook, MAX_NEW_TOKENS
            )
            inject_time = time.time() - t0
            inject_output = inject_text[len(prompt):]
            print(f"\n  INJECTED a={ALPHA} ({len(inject_tokens)} tok, {inject_time:.1f}s):")
            print(f"  {inject_output[:300]}")
        except Exception as e:
            inject_time = time.time() - t0
            inject_output = f"ERROR: {e}"
            inject_tokens = []
            print(f"\n  INJECTED ERROR: {e}")
            import traceback; traceback.print_exc()

        # Native Llama
        injection_hook.set_delta(None)
        t0 = time.time()
        llama_output = generate_native(prompt, llama_model, llama_tokenizer, MAX_NEW_TOKENS)
        llama_time = time.time() - t0
        print(f"\n  LLAMA ({llama_time:.1f}s):")
        print(f"  {llama_output[:300]}")

        # Native GLM
        t0 = time.time()
        glm_output = generate_native(prompt, glm_model, glm_tokenizer, MAX_NEW_TOKENS)
        glm_time = time.time() - t0
        print(f"\n  GLM ({glm_time:.1f}s):")
        print(f"  {glm_output[:300]}")

        results.append({
            "category": category,
            "prompt": prompt,
            "injected_output": inject_output,
            "llama_output": llama_output,
            "glm_output": glm_output,
            "injected_tokens": len(inject_tokens),
            "injected_time": inject_time,
            "llama_time": llama_time,
            "glm_time": glm_time,
        })

        torch.cuda.empty_cache()

    # Save
    out_path = os.path.join(GRAFT_DIR, "injection_generation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    for r in results:
        tps = r["injected_tokens"] / max(r["injected_time"], 0.01)
        print(f"  [{r['category']}] injected: {r['injected_tokens']} tok, "
              f"{tps:.1f} tok/s | llama: {r['llama_time']:.1f}s | glm: {r['glm_time']:.1f}s")

    print(f"\n  Results saved to {out_path}")

    injection_hook.remove()
    del glm_model, llama_model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
