"""Diagnosis for the D3a antisymmetry gate failing at the 1e-6 tolerance.

Hypothesis: the residual is bf16 quantization of the model's logits, amplified by the fact that
the swapped run re-buckets the texts into different batches (different padding -> different
rounding), NOT a logic error in the margin computation. An orientation/indexing bug would make
the margin symmetric instead of antisymmetric, i.e. residual |d' + d| ~ 2|d| ~ 2 sd, not ~0.06.

Three controls, all on the same small subset:
  C1  same pairs, same token budget      -> expect exact 0 (pure determinism, already known)
  C2  same pairs, DIFFERENT token budget -> isolates batch-composition rounding alone,
                                            with no swapping involved at all
  C3  swapped pairs in float32 on CPU    -> if the residual collapses to ~1e-6 in fp32 while
                                            being ~1e-1 in bf16, bf16 is demonstrably the cause

Run: .venv/Scripts/python.exe -m src.judges.validity_d3a_diagnosis
"""
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.build_prompts import load_cfg
from src.judges import margins as margins_mod
from src.judges.margins import get_answer_token_ids, judge_margins_batched, load_probe_pairs

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
JUDGE = "Qwen/Qwen2.5-1.5B-Instruct"
N_GPU = 48
N_CPU = 6  # fp32 CPU forward passes are slow; a handful is enough to be decisive


def swap(pairs):
    return [{"prompt": p["prompt"], "response_a": p["response_b"], "response_b": p["response_a"]} for p in pairs]


def main() -> None:
    cfg = load_cfg()
    template = cfg["judge_templates"]["template_train"]
    pairs = load_probe_pairs("probe", N_GPU)

    tok = AutoTokenizer.from_pretrained(JUDGE)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ans_ids = get_answer_token_ids(tok)

    model = AutoModelForCausalLM.from_pretrained(
        JUDGE, dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to("cuda")
    model.eval()

    default_budget = margins_mod.TOKEN_BUDGET

    # C1: identical call twice, identical batching
    _, _, d_a, _ = judge_margins_batched(model, tok, template, pairs, ans_ids, "cuda")
    _, _, d_a2, _ = judge_margins_batched(model, tok, template, pairs, ans_ids, "cuda")
    c1 = np.abs(d_a - d_a2)
    print(f"C1 same budget, repeat:        max={c1.max():.3e}  median={np.median(c1):.3e}")

    # C2: same (unswapped) pairs, different token budget -> different batch composition only
    margins_mod.TOKEN_BUDGET = 1536
    _, _, d_a_diffbatch, _ = judge_margins_batched(model, tok, template, pairs, ans_ids, "cuda")
    margins_mod.TOKEN_BUDGET = default_budget
    c2 = np.abs(d_a - d_a_diffbatch)
    print(f"C2 different batch composition: max={c2.max():.3e}  median={np.median(c2):.3e}   "
          f"<- no swapping involved at all")

    # D3a residual on the same subset, for scale comparison
    _, _, d_swapped, _ = judge_margins_batched(model, tok, template, swap(pairs), ans_ids, "cuda")
    d3a = np.abs(d_swapped + d_a)
    print(f"D3a residual (bf16, GPU):       max={d3a.max():.3e}  median={np.median(d3a):.3e}")
    print(f"   |d| scale for reference:     sd={np.std(d_a):.3e}  median|d|={np.median(np.abs(d_a)):.3e}")
    print(f"   symmetric-bug signature would be ~2*median|d| = {2*np.median(np.abs(d_a)):.3e}")

    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # C3: float32 on CPU, decisive test
    print(f"\nC3: float32 on CPU, {N_CPU} pairs (slow but decisive)...")
    model32 = AutoModelForCausalLM.from_pretrained(
        JUDGE, dtype=torch.float32, attn_implementation="sdpa",
    )
    model32.eval()
    sub = pairs[:N_CPU]
    _, _, d32, _ = judge_margins_batched(model32, tok, template, sub, ans_ids, "cpu")
    _, _, d32_swapped, _ = judge_margins_batched(model32, tok, template, swap(sub), ans_ids, "cpu")
    c3 = np.abs(d32_swapped + d32)
    print(f"C3 D3a residual (fp32, CPU):    max={c3.max():.3e}  median={np.median(c3):.3e}")

    print("\nVERDICT:")
    print(f"  bf16 batch-composition alone (C2, no swap) explains residuals up to {c2.max():.3e}")
    print(f"  fp32 antisymmetry residual is {c3.max():.3e}")
    if c3.max() < 1e-4 and c2.max() > 1e-3:
        print("  => bf16 precision is demonstrably the cause; the identity holds in fp32.")
    else:
        print("  => NOT explained by bf16 alone. Investigate the margin computation.")


if __name__ == "__main__":
    main()
