"""Diagnoses why fixed-bin padding still shows a 0.25 corpus-dependent residual despite identical
(rows, padded_len) batch shape.

Hypothesis: PyTorch's scaled_dot_product_attention auto-selects between backend kernels
(flash/efficient/math) based on runtime conditions (e.g. available memory), not just tensor
shape -- so identical-shape batches issued from different points in the same process's CUDA
allocator lifetime can silently hit different kernels and produce different bf16 rounding.

Tests, all on the exact same single sequence, scored via a batch of the exact same (rows, len):
  T1  two calls back-to-back, fresh process, nothing else allocated in between -> expect 0 diff
  T2  same, but with a large dummy allocation + free in between (mimics "large corpus" pressure)
  T3  same as T2, but forcing the SDPA math backend explicitly for both calls

Run: .venv/Scripts/python.exe -m src.judges.validity_padmode_diagnosis2
"""
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gc

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.build_prompts import load_cfg
from src.judges import margins as mm

JUDGE = "Qwen/Qwen2.5-1.5B-Instruct"


def score_once(model, tok, template, pair, ans_ids, device):
    _, _, d, _ = mm.judge_margins_batched(model, tok, template, [pair], ans_ids, device)
    return float(d[0])


def main():
    cfg = load_cfg()
    template = cfg["judge_templates"]["template_train"]
    pairs = mm.load_probe_pairs("probe", 5)
    pair = pairs[0]

    mm.PAD_MODE = "fixed_bin"

    tok = AutoTokenizer.from_pretrained(JUDGE)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ans = mm.get_answer_token_ids(tok)
    model = AutoModelForCausalLM.from_pretrained(JUDGE, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
    model.eval()

    print("T1: back-to-back, no intervening allocation")
    d1a = score_once(model, tok, template, pair, ans, "cuda")
    d1b = score_once(model, tok, template, pair, ans, "cuda")
    print(f"  {d1a!r} vs {d1b!r}  diff={abs(d1a-d1b):.4e}")

    print("\nT2: with a large dummy allocation (~1GB) + free in between")
    d2a = score_once(model, tok, template, pair, ans, "cuda")
    dummy = torch.zeros((1024, 1024, 256), dtype=torch.bfloat16, device="cuda")  # ~512MB
    del dummy
    gc.collect()
    torch.cuda.empty_cache()
    d2b = score_once(model, tok, template, pair, ans, "cuda")
    print(f"  {d2a!r} vs {d2b!r}  diff={abs(d2a-d2b):.4e}")

    print("\nT2b: with a large dummy allocation left ALLOCATED (not freed) before the second call")
    d3a = score_once(model, tok, template, pair, ans, "cuda")
    dummy2 = torch.zeros((1024, 1024, 256), dtype=torch.bfloat16, device="cuda")
    d3b = score_once(model, tok, template, pair, ans, "cuda")
    del dummy2
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  {d3a!r} vs {d3b!r}  diff={abs(d3a-d3b):.4e}")

    print("\nT3: forcing SDPA math backend explicitly for both calls (T2b repeated)")
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        d4a = score_once(model, tok, template, pair, ans, "cuda")
        dummy3 = torch.zeros((1024, 1024, 256), dtype=torch.bfloat16, device="cuda")
        d4b = score_once(model, tok, template, pair, ans, "cuda")
        del dummy3
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  {d4a!r} vs {d4b!r}  diff={abs(d4a-d4b):.4e}")

    print("\nVERDICT:")
    if abs(d1a - d1b) < 1e-9 and abs(d2a - d2b) > 1e-6:
        print("  Memory pressure/allocator state (not shape) causes the residual -- confirmed.")
    if abs(d4a - d4b) < 1e-9 and abs(d3a - d3b) > 1e-6:
        print("  Forcing the math backend eliminates it -- backend auto-selection is the cause.")
    elif abs(d4a - d4b) > 1e-6:
        print("  Math backend does NOT eliminate it -- cause is elsewhere.")


if __name__ == "__main__":
    main()
