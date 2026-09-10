"""Authoritative padding-mode validation, per the diagnosis in RESULTS.md: each mode is scored
in its OWN process invocation (never toggling PAD_MODE mid-process), matching how the real
pipeline (src/judges/margins.py) actually runs -- PAD_MODE is a fixed module constant for the
whole run, never switched at runtime. `validity_padmode_check.py`'s multi-mode-in-one-process
test methodology was itself the source of the earlier spurious 0.25 corpus-dependent residual;
this script avoids that confound by construction.

Run three times with --mode {fixed_bin,legacy,free}, each a fresh process; a driver combines
the three output files.

Run: .venv/Scripts/python.exe -m src.judges.validity_padmode_final --mode fixed_bin
     .venv/Scripts/python.exe -m src.judges.validity_padmode_final --mode legacy
     .venv/Scripts/python.exe -m src.judges.validity_padmode_final --mode free
     .venv/Scripts/python.exe -m src.judges.validity_padmode_final --combine
"""
import argparse
import gc
import json
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.build_prompts import load_cfg
from src.judges import margins as mm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "results"
JUDGE = "Qwen/Qwen2.5-1.5B-Instruct"
N_SMALL, N_LARGE = 24, 200


def score(mode: str) -> dict:
    mm.PAD_MODE = mode
    cfg = load_cfg()
    template = cfg["judge_templates"]["template_train"]
    pairs_large = mm.load_probe_pairs("probe", N_LARGE)
    pairs_small = pairs_large[:N_SMALL]

    tok = AutoTokenizer.from_pretrained(JUDGE)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ans = mm.get_answer_token_ids(tok)
    model = AutoModelForCausalLM.from_pretrained(JUDGE, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
    model.eval()

    _, _, d_small, _ = mm.judge_margins_batched(model, tok, template, pairs_small, ans, "cuda")
    _, _, d_large, _ = mm.judge_margins_batched(model, tok, template, pairs_large, ans, "cuda")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {"d_small": d_small.tolist(), "d_large_prefix": d_large[:N_SMALL].tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fixed_bin", "legacy", "free"], default=None)
    parser.add_argument("--combine", action="store_true")
    args = parser.parse_args()

    if args.combine:
        results = {}
        for mode in ["fixed_bin", "legacy", "free"]:
            p = RESULTS_DIR / f"padmode_final_{mode}.json"
            assert p.exists(), f"missing {p} -- run --mode {mode} first"
            results[mode] = json.load(open(p))

        out = {}
        for mode in ["fixed_bin", "legacy"]:
            small = np.array(results[mode]["d_small"])
            large_prefix = np.array(results[mode]["d_large_prefix"])
            gap = np.abs(small - large_prefix)
            out[f"{mode}_corpus_independence"] = {"max_abs": float(gap.max()), "median_abs": float(np.median(gap))}
            print(f"{mode} corpus independence: max={gap.max():.4e} median={np.median(gap):.4e}")

        free_small = np.array(results["free"]["d_small"])
        signal_sd = float(np.std(free_small))
        for mode in ["fixed_bin", "legacy"]:
            small = np.array(results[mode]["d_small"])
            gap = np.abs(small - free_small)
            out[f"{mode}_vs_padfree"] = {
                "max_abs": float(gap.max()), "median_abs": float(np.median(gap)),
                "signal_sd": signal_sd,
                "median_as_frac_of_signal_sd": float(np.median(gap) / signal_sd) if signal_sd else None,
            }
            print(f"{mode} vs exact padfree: max={gap.max():.4e} median={np.median(gap):.4e} "
                  f"(signal_sd={signal_sd:.4f}, frac={np.median(gap)/signal_sd:.4f})")

        with open(RESULTS_DIR / "padmode_final.json", "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {RESULTS_DIR / 'padmode_final.json'}")
        return

    assert args.mode, "pass --mode or --combine"
    result = score(args.mode)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / f"padmode_final_{args.mode}.json", "w") as f:
        json.dump(result, f)
    print(f"wrote results/padmode_final_{args.mode}.json")


if __name__ == "__main__":
    main()
