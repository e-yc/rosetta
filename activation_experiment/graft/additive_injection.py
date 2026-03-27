#!/usr/bin/env python3
"""
Additive Injection Experiment

Phase 0: Benchmark both models on Chinese/English/code/math/multilingual
Phase 1: Alpha sweep — inject GLM delta into Llama's residual stream at layer 15

Uses both trained bridge and untrained Procrustes for the delta.
Both models loaded simultaneously (GLM int8 + Llama fp16).
"""

import json
import math
import os
import sys
import time
from collections import defaultdict

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
RESULTS_DIR = os.path.join(GRAFT_DIR, "additive_injection")
os.makedirs(RESULTS_DIR, exist_ok=True)

HIDDEN_DIM = config.HIDDEN_DIM
GLM_LAYER = 5
LLAMA_SPLICE_LAYER = 15
MAX_SEQ_LEN = 128

# Chinese eval prompts (simple comprehension + knowledge)
CHINESE_PROMPTS = [
    "中国的首都是哪里？请详细介绍一下这座城市。",
    "请解释什么是人工智能，以及它在日常生活中的应用。",
    "长城是中国最著名的建筑之一。请介绍长城的历史。",
    "请用中文写一段关于春节的介绍。",
    "什么是量子计算？它与传统计算机有什么不同？",
    "请介绍中国的四大发明。",
    "请解释光合作用的过程。",
    "中国有哪些主要的河流？请介绍长江。",
    "请写一首关于秋天的短诗。",
    "请解释什么是区块链技术。",
]

ENGLISH_PROMPTS = [
    "What is the capital of France? Describe the city briefly.",
    "Explain what artificial intelligence is and its daily applications.",
    "The Great Wall of China is one of the most famous structures. Describe its history.",
    "Write a paragraph about the celebration of Christmas.",
    "What is quantum computing? How does it differ from classical computing?",
    "Describe the four great inventions of ancient China.",
    "Explain the process of photosynthesis.",
    "What are the major rivers in China? Describe the Yangtze.",
    "Write a short poem about autumn.",
    "Explain what blockchain technology is.",
]

CODE_PROMPTS = [
    "def binary_search(arr, target):\n    \"\"\"Search for target in sorted array. Return index or -1.\"\"\"\n",
    "def is_palindrome(s):\n    \"\"\"Check if string is a palindrome.\"\"\"\n",
    "def flatten(lst):\n    \"\"\"Flatten a nested list.\"\"\"\n",
    "SELECT department, AVG(salary) FROM employees GROUP BY",
    "def gcd(a, b):\n    \"\"\"Return greatest common divisor.\"\"\"\n",
]

MATH_PROMPTS = [
    "A baker makes 60 cookies. She sells 2/3 of them at $2 each. How much money does she make?",
    "If a train travels 120 miles in 2 hours, what is its average speed in miles per hour?",
    "A rectangle has a length of 12 cm and a width of 8 cm. What is its area?",
    "John has 3 times as many marbles as Mary. Mary has 15 marbles. How many do they have together?",
    "A shirt costs $40. It's on sale for 25% off. What is the sale price?",
]

FRENCH_PROMPTS = [
    "Expliquez le concept de la photosynthese en termes simples.",
    "Quelle est la capitale de la France? Decrivez la ville brievement.",
    "Quels sont les principaux fleures de France?",
    "Expliquez ce qu'est l'intelligence artificielle.",
    "Ecrivez un court paragraphe sur la Revolution francaise.",
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
    for param in llama_model.parameters():
        param.requires_grad = False
    print(f"  Both: {torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

    return glm_model, glm_tokenizer, llama_model, llama_tokenizer


def load_projections():
    """Load both trained bridge and untrained Procrustes."""
    projections = {}

    # Trained bridge
    bridge = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=True).cuda().float()
    for path in [
        os.path.join(GRAFT_DIR, "bridge", "procrustes_init_best.pt"),
        os.path.join(GRAFT_DIR, "bridge", "procrustes_init_epoch3.pt"),
    ]:
        if os.path.exists(path):
            bridge.load_state_dict(torch.load(path, map_location="cuda", weights_only=True))
            break
    bridge.eval()
    projections["trained_bridge"] = bridge

    # Untrained Procrustes
    sp_dir = os.path.join(GRAFT_DIR, "projections", "S2")
    procrustes = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=True).cuda().float()
    W_proc = np.load(os.path.join(sp_dir, "P4_W.npy"))
    mean_glm = np.load(os.path.join(sp_dir, "P4_mean_glm.npy"))
    mean_llama = np.load(os.path.join(sp_dir, "P4_mean_llama.npy"))
    bias_init = mean_llama - mean_glm @ W_proc
    procrustes.weight.data = torch.from_numpy(W_proc.T.copy()).float().cuda()
    procrustes.bias.data = torch.from_numpy(bias_init.copy()).float().cuda()
    procrustes.eval()
    projections["procrustes"] = procrustes

    return projections


# -------------------------------------------------------------------------
# Phase 0: Baseline benchmarks
# -------------------------------------------------------------------------

@torch.no_grad()
def benchmark_model(model, tokenizer, prompts, label, max_new_tokens=80):
    """Generate from a model and compute per-token loss against itself."""
    print(f"\n  {label}:", flush=True)
    results = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False,
                           max_length=MAX_SEQ_LEN, truncation=True)
        input_ids = inputs["input_ids"].to("cuda")

        # Loss (self-evaluation: how confident is the model on this input?)
        out = model(input_ids=input_ids)
        logits = out.logits
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1)
        ).item()

        # Generate
        try:
            gen_out = model.generate(
                input_ids, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            generated = tokenizer.decode(gen_out[0][input_ids.shape[1]:], skip_special_tokens=True)
        except Exception as e:
            generated = f"ERROR: {e}"

        results.append({"prompt": prompt[:60], "loss": loss, "output": generated[:200]})

    avg_loss = np.mean([r["loss"] for r in results])
    print(f"    Avg loss: {avg_loss:.3f}", flush=True)
    for r in results[:3]:
        print(f"    [{r['loss']:.2f}] {r['output'][:80]}", flush=True)

    return results, avg_loss


def run_phase0(glm_model, glm_tokenizer, llama_model, llama_tokenizer):
    """Benchmark both models across categories."""
    print(f"\n{'='*70}")
    print(f"  PHASE 0: BASELINE BENCHMARKS")
    print(f"{'='*70}")

    categories = {
        "chinese": CHINESE_PROMPTS,
        "english": ENGLISH_PROMPTS,
        "code": CODE_PROMPTS,
        "math": MATH_PROMPTS,
        "french": FRENCH_PROMPTS,
    }

    all_results = {}

    for cat, prompts in categories.items():
        print(f"\n  --- {cat.upper()} ({len(prompts)} prompts) ---")

        glm_results, glm_loss = benchmark_model(
            glm_model, glm_tokenizer, prompts, f"GLM on {cat}"
        )
        llama_results, llama_loss = benchmark_model(
            llama_model, llama_tokenizer, prompts, f"Llama on {cat}"
        )

        gap = glm_loss - llama_loss
        winner = "GLM" if gap < 0 else "Llama"
        print(f"    Gap: {gap:+.3f} ({winner} wins by {abs(gap):.3f})")

        all_results[cat] = {
            "glm_loss": glm_loss,
            "llama_loss": llama_loss,
            "gap": gap,
            "winner": winner,
            "glm_details": glm_results,
            "llama_details": llama_results,
        }

    # Summary table
    print(f"\n  {'Category':<12} {'GLM loss':>10} {'Llama loss':>12} {'Gap':>8} {'Winner':>8}")
    print(f"  {'-'*55}")
    for cat, r in all_results.items():
        print(f"  {cat:<12} {r['glm_loss']:>10.3f} {r['llama_loss']:>12.3f} "
              f"{r['gap']:>+8.3f} {r['winner']:>8}")

    return all_results


# -------------------------------------------------------------------------
# Phase 1: Alpha sweep
# -------------------------------------------------------------------------

@torch.no_grad()
def run_injection_eval(llama_model, llama_tokenizer, glm_model, glm_tokenizer,
                        projection, prompts, alpha, category):
    """
    Run additive injection at a given alpha.
    Returns per-prompt loss and generated text.
    """
    llama = llama_model.model
    lm_head = llama_model.lm_head
    results = []

    for prompt in prompts:
        # Tokenize for both models
        llama_inputs = llama_tokenizer(prompt, return_tensors="pt",
                                        add_special_tokens=False, max_length=MAX_SEQ_LEN, truncation=True)
        llama_ids = llama_inputs["input_ids"].to("cuda")

        glm_inputs = glm_tokenizer(prompt, return_tensors="pt",
                                    add_special_tokens=False, max_length=MAX_SEQ_LEN, truncation=True)
        glm_ids = glm_inputs["input_ids"].to("cuda")

        # Get Llama hidden states at layer 15 (via full forward with output_hidden_states)
        llama_out = llama_model(input_ids=llama_ids, output_hidden_states=True)
        h_llama_15 = llama_out.hidden_states[LLAMA_SPLICE_LAYER]  # (1, llama_seq, dim)

        # Get GLM hidden states at layer 5
        glm_out = glm_model(input_ids=glm_ids, output_hidden_states=True)
        h_glm_5 = glm_out.hidden_states[GLM_LAYER]  # (1, glm_seq, dim)

        # Project GLM through bridge/procrustes
        h_glm_projected = projection(h_glm_5.float()).half()  # (1, glm_seq, dim)

        # Align sequence lengths (truncate to shorter)
        min_len = min(h_llama_15.shape[1], h_glm_projected.shape[1])
        h_llama = h_llama_15[:, :min_len, :]
        h_glm_p = h_glm_projected[:, :min_len, :]

        # Compute delta and inject
        delta = h_glm_p - h_llama
        h_injected = h_llama + alpha * delta

        # Run injected hidden states through Llama layers 15-32
        hidden = h_injected
        seq_len = hidden.shape[1]
        position_ids = torch.arange(seq_len, device="cuda").unsqueeze(0)
        cache_position = torch.arange(seq_len, device="cuda")
        causal_mask = llama._update_causal_mask(
            None, hidden, cache_position, past_key_values=None, output_attentions=False
        )
        position_embeddings = None
        if hasattr(llama, "rotary_emb"):
            position_embeddings = llama.rotary_emb(hidden, position_ids)

        for layer in llama.layers[LLAMA_SPLICE_LAYER:]:
            kwargs = {"attention_mask": causal_mask, "position_ids": position_ids}
            if position_embeddings is not None:
                kwargs["position_embeddings"] = position_embeddings
            layer_out = layer(hidden, **kwargs)
            hidden = layer_out[0]

        hidden = llama.norm(hidden)
        logits = lm_head(hidden)

        # Compute loss against Llama's tokenization
        shift_logits = logits[:, :min_len-1, :]
        shift_labels = llama_ids[:, 1:min_len]
        if shift_logits.shape[1] > 0 and shift_labels.shape[1] > 0:
            n = min(shift_logits.shape[1], shift_labels.shape[1])
            loss = F.cross_entropy(
                shift_logits[:, :n].reshape(-1, shift_logits.size(-1)),
                shift_labels[:, :n].reshape(-1)
            ).item()
        else:
            loss = float("inf")

        # Also compute KL vs native Llama logits
        native_logits = llama_out.logits[:, :min_len, :]
        if min_len > 1:
            p = F.softmax(native_logits[:, :min_len-1].float(), dim=-1)
            q = F.log_softmax(logits[:, :min_len-1].float(), dim=-1)
            kl = F.kl_div(q, p, reduction="batchmean", log_target=False).item()
        else:
            kl = float("inf")

        # Top-1 agreement with native Llama
        if min_len > 1:
            native_top1 = native_logits[:, :min_len-1].argmax(dim=-1)
            inject_top1 = logits[:, :min_len-1].argmax(dim=-1)
            top1_agree = (native_top1 == inject_top1).float().mean().item()
        else:
            top1_agree = 0.0

        results.append({
            "loss": loss,
            "kl_vs_native": kl,
            "top1_vs_native": top1_agree,
        })

        del llama_out, glm_out

    avg_loss = np.mean([r["loss"] for r in results])
    avg_kl = np.mean([r["kl_vs_native"] for r in results])
    avg_top1 = np.mean([r["top1_vs_native"] for r in results])

    return {
        "loss": avg_loss,
        "kl_vs_native": avg_kl,
        "top1_vs_native": avg_top1,
        "per_prompt": results,
    }


def run_phase1(glm_model, glm_tokenizer, llama_model, llama_tokenizer, projections):
    """Alpha sweep for additive injection."""
    print(f"\n{'='*70}")
    print(f"  PHASE 1: ALPHA SWEEP")
    print(f"{'='*70}")

    alphas = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

    categories = {
        "chinese": CHINESE_PROMPTS,
        "english": ENGLISH_PROMPTS,
        "code": CODE_PROMPTS,
        "math": MATH_PROMPTS,
        "french": FRENCH_PROMPTS,
    }

    all_results = {}

    for proj_name, projection in projections.items():
        print(f"\n  === Projection: {proj_name} ===", flush=True)
        all_results[proj_name] = {}

        for alpha in alphas:
            print(f"\n  alpha={alpha}:", flush=True)
            all_results[proj_name][str(alpha)] = {}

            for cat, prompts in categories.items():
                result = run_injection_eval(
                    llama_model, llama_tokenizer,
                    glm_model, glm_tokenizer,
                    projection, prompts, alpha, cat
                )
                all_results[proj_name][str(alpha)][cat] = {
                    "loss": result["loss"],
                    "kl": result["kl_vs_native"],
                    "top1": result["top1_vs_native"],
                }
                print(f"    {cat:<10} loss={result['loss']:.3f}  "
                      f"kl={result['kl_vs_native']:.3f}  "
                      f"top1={result['top1_vs_native']:.3f}", flush=True)

            torch.cuda.empty_cache()

    return all_results


def main():
    print("=" * 70)
    print("ADDITIVE INJECTION EXPERIMENT")
    print("=" * 70)

    # Load models
    print("\nLoading models...", flush=True)
    glm_model, glm_tokenizer, llama_model, llama_tokenizer = load_models()
    projections = load_projections()

    # Phase 0: Baselines
    phase0 = run_phase0(glm_model, glm_tokenizer, llama_model, llama_tokenizer)
    with open(os.path.join(RESULTS_DIR, "phase0_baselines.json"), "w", encoding="utf-8") as f:
        json.dump(phase0, f, indent=2, ensure_ascii=False)

    # Check: does GLM beat Llama on anything?
    glm_wins = [cat for cat, r in phase0.items() if r["winner"] == "GLM"]
    print(f"\n  GLM wins on: {glm_wins if glm_wins else 'NOTHING'}")
    if not glm_wins:
        print("  WARNING: GLM doesn't beat Llama on any category.")
        print("  Injection may not have useful signal. Proceeding anyway for the data.")

    # Phase 1: Alpha sweep
    phase1 = run_phase1(glm_model, glm_tokenizer, llama_model, llama_tokenizer, projections)
    with open(os.path.join(RESULTS_DIR, "phase1_alpha_sweep.json"), "w", encoding="utf-8") as f:
        json.dump(phase1, f, indent=2, ensure_ascii=False)

    # Summary table
    print(f"\n{'='*70}")
    print(f"  ALPHA SWEEP SUMMARY")
    print(f"{'='*70}")

    for proj_name in projections:
        print(f"\n  {proj_name}:")
        print(f"  {'alpha':<8}", end="")
        cats = list(phase1[proj_name]["0.0"].keys())
        for cat in cats:
            print(f"  {cat:<12}", end="")
        print()

        for alpha in [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]:
            a_str = str(alpha)
            if a_str not in phase1[proj_name]:
                continue
            print(f"  {alpha:<8.2f}", end="")
            for cat in cats:
                loss = phase1[proj_name][a_str][cat]["loss"]
                print(f"  {loss:<12.3f}", end="")
            print()

    print(f"\n  Results saved to {RESULTS_DIR}/")

    del glm_model, llama_model
    torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print(f"  ADDITIVE INJECTION EXPERIMENT COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
