"""Validates the fixed-bin padding mode before it is used for any reported measurement.

Two properties must hold, and both are measured rather than assumed:

  1. CORPUS-INDEPENDENCE (the property the whole change exists for): scoring the same pairs as
     part of a small corpus and as part of a large one must give identical margins. This is what
     the original token-budget bucketing violated, injecting ~0.125 median noise.
  2. RESIDUAL GAP vs exact padding-free scoring: fixed-bin still pads, so it is not identical to
     the unpadded value. Quantify that gap against the signal SD so the size of the remaining
     approximation is on the record.

Run: .venv/Scripts/python.exe -m src.judges.validity_padmode_check
"""
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gc
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.build_prompts import load_cfg
from src.judges import margins as mm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "results"
JUDGE = "Qwen/Qwen2.5-1.5B-Instruct"
N_SMALL = 24
N_LARGE = 200


def main() -> None:
    cfg = load_cfg()
    template = cfg["judge_templates"]["template_train"]
    pairs_large = mm.load_probe_pairs("probe", N_LARGE)
    pairs_small = pairs_large[:N_SMALL]

    tok = AutoTokenizer.from_pretrained(JUDGE)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ans = mm.get_answer_token_ids(tok)
    model = AutoModelForCausalLM.from_pretrained(
        JUDGE, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
    model.eval()

    out = {}

    # --- 1. corpus independence under fixed_bin ---
    mm.PAD_MODE = "fixed_bin"
    _, _, d_small, _ = mm.judge_margins_batched(model, tok, template, pairs_small, ans, "cuda")
    _, _, d_large, _ = mm.judge_margins_batched(model, tok, template, pairs_large, ans, "cuda")
    gap_corpus = np.abs(d_small - d_large[:N_SMALL])
    out["fixed_bin_corpus_independence"] = {
        "max_abs": float(gap_corpus.max()), "median_abs": float(np.median(gap_corpus)),
    }
    print(f"fixed_bin, same pairs in a {N_SMALL}-pair vs {N_LARGE}-pair corpus:")
    print(f"  max|diff|={gap_corpus.max():.3e}  median={np.median(gap_corpus):.3e}")

    # --- 1b. same test under the legacy token-budget bucketing, for contrast ---
    mm.PAD_MODE = "legacy"
    _, _, l_small, _ = mm.judge_margins_batched(model, tok, template, pairs_small, ans, "cuda")
    _, _, l_large, _ = mm.judge_margins_batched(model, tok, template, pairs_large, ans, "cuda")
    gap_legacy = np.abs(l_small - l_large[:N_SMALL])
    out["legacy_corpus_independence"] = {
        "max_abs": float(gap_legacy.max()), "median_abs": float(np.median(gap_legacy)),
    }
    print(f"legacy bucketing, same comparison:")
    print(f"  max|diff|={gap_legacy.max():.3e}  median={np.median(gap_legacy):.3e}")

    # --- 2. residual gap of fixed_bin vs exact padding-free ---
    mm.PAD_MODE = "free"
    _, _, d_free, _ = mm.judge_margins_batched(model, tok, template, pairs_small, ans, "cuda")
    mm.PAD_MODE = "fixed_bin"
    gap_free = np.abs(d_small - d_free)
    signal_sd = float(np.std(d_free))
    out["fixed_bin_vs_padfree"] = {
        "max_abs": float(gap_free.max()), "median_abs": float(np.median(gap_free)),
        "signal_sd": signal_sd,
        "median_as_frac_of_signal_sd": float(np.median(gap_free) / signal_sd) if signal_sd else None,
    }
    print(f"fixed_bin vs exact padding-free (n={N_SMALL}):")
    print(f"  max|diff|={gap_free.max():.3e}  median={np.median(gap_free):.3e}  "
          f"signal_sd={signal_sd:.3f}  median/sd={np.median(gap_free)/signal_sd:.3f}")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "padmode_check.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {RESULTS_DIR / 'padmode_check.json'}")


if __name__ == "__main__":
    main()
