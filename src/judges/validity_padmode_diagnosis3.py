"""Follow-up to diagnosis2: T1-T3 there used a single real pair + repeated filler, which stayed
perfectly reproducible even under memory pressure and backend switching. That rules out simple
allocator-state explanations, but doesn't yet test the actual failing scenario in
padmode_check.py: REAL heterogeneous batches (different pairs sharing a batch), compared across
a 24-pair vs 200-pair corpus. This tests that scenario directly, with and without forcing the
SDPA math backend, to see whether backend auto-selection (chosen differently depending on what
else is batched) is the actual cause.

Run: .venv/Scripts/python.exe -m src.judges.validity_padmode_diagnosis3
"""
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.build_prompts import load_cfg
from src.judges import margins as mm

JUDGE = "Qwen/Qwen2.5-1.5B-Instruct"
N_SMALL, N_LARGE = 24, 200


def run(model, tok, template, ans, force_math: bool):
    pairs_large = mm.load_probe_pairs("probe", N_LARGE)
    pairs_small = pairs_large[:N_SMALL]
    ctx = torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH) if force_math else _noop()
    with ctx:
        _, _, d_small, _ = mm.judge_margins_batched(model, tok, template, pairs_small, ans, "cuda")
        _, _, d_large, _ = mm.judge_margins_batched(model, tok, template, pairs_large, ans, "cuda")
    gap = np.abs(d_small - d_large[:N_SMALL])
    return gap


class _noop:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def main():
    cfg = load_cfg()
    template = cfg["judge_templates"]["template_train"]
    mm.PAD_MODE = "fixed_bin"

    tok = AutoTokenizer.from_pretrained(JUDGE)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ans = mm.get_answer_token_ids(tok)
    model = AutoModelForCausalLM.from_pretrained(JUDGE, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
    model.eval()

    print("default SDPA backend (auto-selected), real heterogeneous batches:")
    gap_default = run(model, tok, template, ans, force_math=False)
    print(f"  max={gap_default.max():.4e}  median={np.median(gap_default):.4e}  "
          f"n_nonzero={(gap_default > 1e-9).sum()}/{len(gap_default)}")

    print("\nforced MATH backend, same comparison:")
    gap_math = run(model, tok, template, ans, force_math=True)
    print(f"  max={gap_math.max():.4e}  median={np.median(gap_math):.4e}  "
          f"n_nonzero={(gap_math > 1e-9).sum()}/{len(gap_math)}")

    print("\nVERDICT:")
    if gap_math.max() < 1e-6 and gap_default.max() > 1e-3:
        print("  MATH backend eliminates corpus-dependence -- backend auto-selection confirmed as cause.")
    else:
        print("  MATH backend does NOT eliminate it -- cause is elsewhere (needs further isolation).")


if __name__ == "__main__":
    main()
