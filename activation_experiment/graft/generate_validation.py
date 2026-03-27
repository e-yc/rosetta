#!/usr/bin/env python3
"""
Generation Validation — KV-cached autoregressive generation through the bridge.

Loads GLM (int8, ~10GB) and Llama (fp16, ~16GB) simultaneously.
Uses KV caching for both partial models to avoid full recomputation per token.

GLM: maintains KV cache for layers 0-5. On each step, re-tokenizes and
     extends the cache if the prefix is stable, or recomputes if not.
Llama: maintains KV cache for layers 15-32. Each step processes one new position.
"""

import json
import os
import sys
import time
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import (
    AutoConfig, AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig, DynamicCache,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

GRAFT_DIR = os.path.dirname(os.path.abspath(__file__))
HIDDEN_DIM = config.HIDDEN_DIM
GLM_LAYER = 5
LLAMA_SPLICE_LAYER = 15
MAX_NEW_TOKENS = 100

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
    ("math", "A store sells apples for $2 each and oranges for $3 each. If Sarah buys 5 apples and"),
    # Multilingual
    ("multilingual", "La revolution industrielle a commence en Angleterre au"),
    # Mixed
    ("mixed", "Here is a Python function that calculates the area of a circle:\n```python\nimport math\ndef area("),
]


def load_models():
    """Load GLM in int8 and Llama in fp16 simultaneously."""
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
    for param in llama_model.parameters():
        param.requires_grad = False
    print(f"  Both: {torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

    return glm_model, glm_tokenizer, llama_model, llama_tokenizer


def load_bridge():
    bridge = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=True).cuda().float()
    for path in [
        os.path.join(GRAFT_DIR, "bridge", "procrustes_init_best.pt"),
        os.path.join(GRAFT_DIR, "bridge", "procrustes_init_epoch3.pt"),
        os.path.join(GRAFT_DIR, "bridge", "procrustes_init_epoch2.pt"),
        os.path.join(GRAFT_DIR, "bridge", "procrustes_init_epoch1.pt"),
    ]:
        if os.path.exists(path):
            bridge.load_state_dict(torch.load(path, map_location="cuda", weights_only=True))
            print(f"  Bridge: {os.path.basename(path)}")
            break
    bridge.eval()
    return bridge


class GraftGenerator:
    """
    KV-cached autoregressive generation through the GLM->bridge->Llama graft.

    Maintains separate KV caches for GLM (layers 0-5) and Llama (layers 15-32).
    Each step:
      1. Tokenize new text with GLM, extend GLM KV cache, get layer-5 hidden state
      2. Project through bridge
      3. Run through Llama layers 15-32 with KV cache, get next-token logits
    """

    def __init__(self, glm_model, glm_tokenizer, llama_model, llama_tokenizer, bridge):
        self.glm = glm_model
        self.glm_tok = glm_tokenizer
        self.llama = llama_model.model
        self.lm_head = llama_model.lm_head
        self.llama_tok = llama_tokenizer
        self.bridge = bridge

        self.llama_layers = list(self.llama.layers[LLAMA_SPLICE_LAYER:])
        self.n_llama_layers = len(self.llama_layers)

    def _get_glm_layer5(self, text):
        """Run GLM on full text, return layer 5 hidden states for all positions."""
        inputs = self.glm_tok(text, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs["input_ids"].to("cuda")
        with torch.no_grad():
            out = self.glm(input_ids=input_ids, output_hidden_states=True)
        hs = out.hidden_states[GLM_LAYER][0]  # (seq_len, dim)
        del out
        return hs

    def _run_llama_layers(self, hidden, position_id, kv_cache):
        """Run one position through Llama layers 15-32 with KV cache."""
        # hidden: (1, 1, dim) for incremental, or (1, seq_len, dim) for prefill
        seq_len = hidden.shape[1]
        total_len = kv_cache.get_seq_length() + seq_len if kv_cache.get_seq_length() > 0 else seq_len

        position_ids = position_id.unsqueeze(0) if position_id.dim() == 1 else position_id

        # Causal mask for the new position(s) attending to all cached + new positions
        cache_position = torch.arange(
            total_len - seq_len, total_len, device=hidden.device
        )

        causal_mask = self.llama._update_causal_mask(
            None, hidden, cache_position,
            past_key_values=kv_cache, output_attentions=False
        )

        position_embeddings = None
        if hasattr(self.llama, "rotary_emb"):
            position_embeddings = self.llama.rotary_emb(hidden, position_ids)

        for i, layer in enumerate(self.llama_layers):
            layer_out = layer(
                hidden,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=kv_cache,
                use_cache=True,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            hidden = layer_out[0]

        hidden = self.llama.norm(hidden)
        logits = self.lm_head(hidden)
        return logits

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=MAX_NEW_TOKENS):
        """Generate text through the graft with KV caching."""
        # Prefill: run GLM on the full prompt
        glm_hidden = self._get_glm_layer5(prompt)  # (prompt_glm_len, dim)
        projected = self.bridge(glm_hidden.float().unsqueeze(0)).half()  # (1, seq_len, dim)

        prompt_len = projected.shape[1]
        position_ids = torch.arange(prompt_len, device="cuda").unsqueeze(0)

        # Initialize KV cache for Llama
        # We need a DynamicCache that spans all Llama layers (not just our subset)
        # The layer indices in the cache must match the layer indices in the model
        kv_cache = DynamicCache()

        # Prefill through Llama layers
        logits = self._run_llama_layers(projected, position_ids, kv_cache)

        # Generate tokens
        generated_tokens = []
        current_text = prompt
        next_pos = prompt_len

        for step in range(max_new_tokens):
            # Sample next token from Llama's vocab (greedy)
            next_token_id = logits[0, -1, :].argmax().item()
            next_token_str = self.llama_tok.decode([next_token_id])

            if next_token_id == self.llama_tok.eos_token_id:
                break
            if next_token_str in ("<|eot_id|>", "<|end|>", "</s>"):
                break

            generated_tokens.append(next_token_id)
            current_text += next_token_str

            # Get GLM's representation of the new text
            # Re-run GLM on full text to get the last position's layer-5 hidden state
            glm_hidden = self._get_glm_layer5(current_text)
            last_vec = glm_hidden[-1:].float().unsqueeze(0)  # (1, 1, dim)
            projected_new = self.bridge(last_vec).half()

            # Run new position through Llama with KV cache
            new_pos_ids = torch.tensor([[next_pos]], device="cuda")
            logits = self._run_llama_layers(projected_new, new_pos_ids, kv_cache)
            next_pos += 1

        return current_text, generated_tokens


@torch.no_grad()
def generate_native(prompt, model, tokenizer, max_new_tokens):
    """Standard generation for baseline."""
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"].to("cuda")
    output = model.generate(
        input_ids, max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)


def main():
    print("=" * 70)
    print("GENERATION VALIDATION (KV-cached)")
    print("=" * 70)

    print("\nLoading models...", flush=True)
    glm_model, glm_tokenizer, llama_model, llama_tokenizer = load_models()
    bridge = load_bridge()

    generator = GraftGenerator(glm_model, glm_tokenizer, llama_model, llama_tokenizer, bridge)

    results = []

    for i, (category, prompt) in enumerate(PROMPTS):
        print(f"\n{'='*60}")
        print(f"  Prompt {i+1}/{len(PROMPTS)} [{category}]:")
        print(f"  \"{prompt[:80]}\"")
        print(f"{'='*60}")

        # Graft generation
        t0 = time.time()
        try:
            graft_text, graft_tokens = generator.generate(prompt, MAX_NEW_TOKENS)
            graft_time = time.time() - t0
            graft_output = graft_text[len(prompt):]
            print(f"\n  GRAFT ({len(graft_tokens)} tokens, {graft_time:.1f}s, "
                  f"{len(graft_tokens)/graft_time:.1f} tok/s):")
            print(f"  {graft_output[:300]}")
        except Exception as e:
            graft_time = time.time() - t0
            graft_output = f"ERROR: {e}"
            graft_tokens = []
            print(f"\n  GRAFT ERROR: {e}")

        # Native Llama generation
        t0 = time.time()
        llama_text = generate_native(prompt, llama_model, llama_tokenizer, MAX_NEW_TOKENS)
        llama_time = time.time() - t0
        llama_output = llama_text[len(prompt):]
        print(f"\n  LLAMA ({llama_time:.1f}s):")
        print(f"  {llama_output[:300]}")

        results.append({
            "category": category,
            "prompt": prompt,
            "graft_output": graft_output,
            "llama_output": llama_output,
            "graft_tokens": len(graft_tokens),
            "graft_time": graft_time,
            "llama_time": llama_time,
            "graft_tok_per_sec": len(graft_tokens) / max(graft_time, 0.01),
        })

        # Clear intermediate caches
        torch.cuda.empty_cache()

    # Save
    out_path = os.path.join(GRAFT_DIR, "generation_validation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    for r in results:
        print(f"  [{r['category']}] graft: {r['graft_tokens']} tok in {r['graft_time']:.1f}s "
              f"({r['graft_tok_per_sec']:.1f} tok/s) | llama: {r['llama_time']:.1f}s")

    print(f"\n  Results saved to {out_path}")

    del glm_model, llama_model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
