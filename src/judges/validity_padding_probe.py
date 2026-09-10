"""Isolates whether left-padding amount changes the verdict logits, in bf16 vs float32.

Motivation: D3a's antisymmetry residual was exactly 0 when both sides were computed fresh in the
same process, but ~0.06 (median) when the swapped side was compared against margins STORED from
the 600-pair Track B run. The only difference between those two situations is how the texts were
bucketed into batches, i.e. how much left-padding each sequence received. This script tests that
directly: score the same comparison text (a) essentially unpadded and (b) padded out by batching
it with a much longer sequence, and report the change in the A/B verdict log-prob difference.

Run: .venv/Scripts/python.exe -m src.judges.validity_padding_probe
"""
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gc

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.build_prompts import load_cfg
from src.judges.margins import build_comparison_text, get_answer_token_ids, load_probe_pairs

JUDGE = "Qwen/Qwen2.5-1.5B-Instruct"
N = 8


def ell_for_texts(model, tok, texts, ans_ids, device):
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    with torch.no_grad():
        logits = model(**enc, logits_to_keep=1).logits[:, -1, :].float()
    logp = torch.log_softmax(logits, dim=-1)
    return (logp[:, ans_ids["A"]] - logp[:, ans_ids["B"]]).cpu().numpy(), enc["input_ids"].shape[1]


def run_for_dtype(dtype, device):
    cfg = load_cfg()
    template = cfg["judge_templates"]["template_train"]
    pairs = load_probe_pairs("probe", 200)

    tok = AutoTokenizer.from_pretrained(JUDGE)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ans_ids = get_answer_token_ids(tok)

    model = AutoModelForCausalLM.from_pretrained(JUDGE, dtype=dtype, attn_implementation="sdpa").to(device)
    model.eval()

    texts = [build_comparison_text(tok, template, p["prompt"], p["response_a"], p["response_b"]) for p in pairs]
    lens = [len(tok.encode(t)) for t in texts]
    order = np.argsort(lens)
    short_idx = [int(i) for i in order[:N]]          # the N shortest
    long_text = texts[int(order[-1])]                 # the single longest, used only as a padding driver

    # (a) the N short texts batched only with each other -> minimal padding
    ell_min_pad, len_min = ell_for_texts(model, tok, [texts[i] for i in short_idx], ans_ids, device)
    # (b) same N short texts, but batched alongside the longest text -> heavy left-padding
    ell_max_pad_all, len_max = ell_for_texts(model, tok, [texts[i] for i in short_idx] + [long_text], ans_ids, device)
    ell_max_pad = ell_max_pad_all[:N]

    diff = np.abs(ell_min_pad - ell_max_pad)
    print(f"  padded length: {len_min} -> {len_max} tokens")
    print(f"  |delta ell| across padding change: max={diff.max():.3e} median={np.median(diff):.3e}")

    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return diff


def main() -> None:
    print(f"bf16 on GPU ({JUDGE}):")
    d_bf16 = run_for_dtype(torch.bfloat16, "cuda")
    print(f"\nfloat32 on CPU ({JUDGE}):")
    d_fp32 = run_for_dtype(torch.float32, "cpu")

    print("\nVERDICT:")
    print(f"  bf16  padding sensitivity: max |delta ell| = {d_bf16.max():.3e}")
    print(f"  fp32  padding sensitivity: max |delta ell| = {d_fp32.max():.3e}")
    if d_bf16.max() > 100 * max(d_fp32.max(), 1e-12):
        print("  => padding-induced variation is a bf16 precision effect, not a logic error.")
    else:
        print("  => bf16 does NOT explain it; the padding/masking path itself is suspect.")


if __name__ == "__main__":
    main()
