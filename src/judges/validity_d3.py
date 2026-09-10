"""Step 1 of NEXT_STEPS_FINAL.md: D3a antisymmetry gate + D3b exact-null gate and
surface-sensitivity diagnostics. Uses the existing 600 probe pairs and the judges already
scored in Track B. Does not regenerate probe pairs.

All four conditions go through the SAME public margin-scoring path (`judge_margins_batched`),
so the comparison is re-templated, re-tokenized, re-bucketed and re-forwarded from scratch --
that is what gives D3a teeth. A cached-single-forward-pass implementation would satisfy the
antisymmetry identity vacuously; this one would catch a verdict-letter orientation bug, because
if `ell_BA` were indexed with the wrong letter the debiased margin would measure position bias
instead of content and would come out SYMMETRIC (d' = +d) rather than antisymmetric (d' = -d).

Conditions per judge (all under `template_train`):
  D3a   swapped:   d_m(x, y', y)  vs stored d_m(x, y, y')   -- gate: |sum| ~ 0
  D3b-0 duplicate: d_m(x, y, y)                             -- gate: |d| ~ 0
  D3b-1 format:    d_m(x, y, y^fmt)   (line-wrap/spacing only, lexical content preserved)
  D3b-2 paraphrase: d_m(x, y, y^para) (conservative deterministic contraction/abbrev expansion)

Run: .venv/Scripts/python.exe -m src.judges.validity_d3
"""
import argparse
import json
import os
import re
import textwrap

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common.vram import free_vram_gb, reclaim
from src.data.build_prompts import load_cfg
from src.judges.margins import get_answer_token_ids, judge_margins_batched, load_probe_pairs

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = REPO_ROOT / "data_cache"
RESULTS_DIR = REPO_ROOT / "results"

# Gate tolerance. Both gates are algebraically exact identities; the only reason they would not
# be bitwise zero is bf16 kernel nondeterminism across different batch compositions/padding,
# since the swapped/duplicate runs re-bucket the texts independently.
GATE_TOL = 1e-6


def format_variant(text: str) -> str:
    """Presentation-only change: normalize intra-line whitespace and re-wrap paragraphs at a
    different width. Words and their order are preserved exactly; only line breaks and spacing
    change. Token identity is NOT claimed to be preserved (re-wrapping changes tokenization).
    """
    paragraphs = re.split(r"\n\s*\n", text)
    out = []
    for para in paragraphs:
        collapsed = re.sub(r"\s+", " ", para).strip()
        if not collapsed:
            continue
        out.append(textwrap.fill(collapsed, width=72))
    return "\n\n".join(out)


# Conservative, meaning-preserving expansions only. Anything whose expansion could change
# meaning (possessive "'s", "ain't", ambiguous "'d"/"'ll" on nouns) is deliberately excluded.
_PARAPHRASE_SUBS = [
    (r"\bdon't\b", "do not"), (r"\bDon't\b", "Do not"),
    (r"\bdoesn't\b", "does not"), (r"\bDoesn't\b", "Does not"),
    (r"\bdidn't\b", "did not"), (r"\bDidn't\b", "Did not"),
    (r"\bisn't\b", "is not"), (r"\bIsn't\b", "Is not"),
    (r"\baren't\b", "are not"), (r"\bAren't\b", "Are not"),
    (r"\bwasn't\b", "was not"), (r"\bWasn't\b", "Was not"),
    (r"\bweren't\b", "were not"), (r"\bWeren't\b", "Were not"),
    (r"\bwon't\b", "will not"), (r"\bWon't\b", "Will not"),
    (r"\bcan't\b", "cannot"), (r"\bCan't\b", "Cannot"),
    (r"\bcouldn't\b", "could not"), (r"\bCouldn't\b", "Could not"),
    (r"\bshouldn't\b", "should not"), (r"\bShouldn't\b", "Should not"),
    (r"\bwouldn't\b", "would not"), (r"\bWouldn't\b", "Would not"),
    (r"\bhaven't\b", "have not"), (r"\bHaven't\b", "Have not"),
    (r"\bhasn't\b", "has not"), (r"\bHasn't\b", "Has not"),
    (r"\bhadn't\b", "had not"), (r"\bHadn't\b", "Had not"),
    (r"\bI'm\b", "I am"),
    (r"\byou're\b", "you are"), (r"\bYou're\b", "You are"),
    (r"\bthey're\b", "they are"), (r"\bThey're\b", "They are"),
    (r"\bwe're\b", "we are"), (r"\bWe're\b", "We are"),
    (r"\bI've\b", "I have"),
    (r"\byou've\b", "you have"), (r"\bYou've\b", "You have"),
    (r"\bwe've\b", "we have"), (r"\bWe've\b", "We have"),
    (r"\bthey've\b", "they have"), (r"\bThey've\b", "They have"),
    (r"\be\.g\.", "for example"), (r"\bi\.e\.", "that is"),
]


def paraphrase_variant(text: str) -> tuple[str, int]:
    """Deterministic conservative paraphrase. Returns (new_text, n_substitutions). Rows with
    zero substitutions are skipped by the caller rather than counted as a null (per the spec's
    instruction not to pretend a no-op is a paraphrase).
    """
    out = text
    n = 0
    for pattern, repl in _PARAPHRASE_SUBS:
        out, k = re.subn(pattern, repl, out)
        n += k
    return out, n


def build_condition_pairs(pairs: list[dict], condition: str, limit: int | None = None) -> tuple[list[dict], list[int]]:
    """Returns (condition_pairs, source_indices). source_indices maps each returned pair back to
    its index in `pairs` (needed because paraphrase skips rows with no applicable substitution).
    """
    out, idxs = [], []
    for i, p in enumerate(pairs):
        if limit is not None and len(out) >= limit:
            break
        if condition == "swapped":
            out.append({"prompt": p["prompt"], "response_a": p["response_b"], "response_b": p["response_a"]})
            idxs.append(i)
        elif condition == "duplicate":
            out.append({"prompt": p["prompt"], "response_a": p["response_a"], "response_b": p["response_a"]})
            idxs.append(i)
        elif condition == "format":
            variant = format_variant(p["response_a"])
            if not variant.strip():
                continue
            out.append({"prompt": p["prompt"], "response_a": p["response_a"], "response_b": variant})
            idxs.append(i)
        elif condition == "paraphrase":
            variant, n_sub = paraphrase_variant(p["response_a"])
            if n_sub == 0:
                continue  # no applicable substitution -> not a paraphrase, skip rather than fake a null
            out.append({"prompt": p["prompt"], "response_a": p["response_a"], "response_b": variant})
            idxs.append(i)
        else:
            raise ValueError(condition)
    return out, idxs


def load_stored_margins(judge_name: str, template_name: str, tag: str | None = "fixedbin") -> dict[int, float]:
    safe = judge_name.replace("/", "__")
    prefix = f"judge_margins_{tag}_" if tag else "judge_margins_"
    path = CACHE_DIR / f"{prefix}{safe}.jsonl"
    assert path.exists(), f"{path} missing -- run src.judges.margins --tag {tag} first"
    out = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if row["template_name"] == template_name:
                out[row["prompt_idx"]] = row["d_m"]
    return out


def run_judge(judge_name: str, pairs: list[dict], template: str, diag_limit: int, tag: str | None = "fixedbin") -> dict:
    baseline_vram = free_vram_gb()
    tokenizer = AutoTokenizer.from_pretrained(judge_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        judge_name, dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to("cuda")
    model.eval()
    ans_ids = get_answer_token_ids(tokenizer)

    stored = load_stored_margins(judge_name, "template_train", tag)
    result: dict = {"judge": judge_name}

    # ---- D3a: antisymmetry, full 600 pairs, full re-forward ----
    t0 = time.time()
    swapped_pairs, swapped_idx = build_condition_pairs(pairs, "swapped")
    _, _, d_swapped, _ = judge_margins_batched(model, tokenizer, template, swapped_pairs, ans_ids, "cuda")
    d_orig = np.array([stored[i] for i in swapped_idx])
    antisym_residual = np.abs(d_swapped + d_orig)
    result["d3a"] = {
        "n": int(len(antisym_residual)),
        "max_abs": float(antisym_residual.max()),
        "median_abs": float(np.median(antisym_residual)),
        "passes": bool(antisym_residual.max() <= GATE_TOL),
        "wall_clock_s": time.time() - t0,
    }
    print(f"  D3a antisymmetry: max={result['d3a']['max_abs']:.3e} "
          f"median={result['d3a']['median_abs']:.3e} pass={result['d3a']['passes']}")

    # ---- D3b-0: exact-duplicate null, full 600 pairs ----
    t0 = time.time()
    dup_pairs, _ = build_condition_pairs(pairs, "duplicate")
    _, _, d_dup, _ = judge_margins_batched(model, tokenizer, template, dup_pairs, ans_ids, "cuda")
    result["d3b0"] = {
        "n": int(len(d_dup)),
        "max_abs": float(np.abs(d_dup).max()),
        "median_abs": float(np.median(np.abs(d_dup))),
        "passes": bool(np.abs(d_dup).max() <= GATE_TOL),
        "wall_clock_s": time.time() - t0,
    }
    print(f"  D3b-0 duplicate null: max={result['d3b0']['max_abs']:.3e} "
          f"median={result['d3b0']['median_abs']:.3e} pass={result['d3b0']['passes']}")

    signal_sd = float(np.std([stored[i] for i in sorted(stored)]))
    result["signal_sd"] = signal_sd

    # ---- D3b-1: formatting sensitivity (diagnostic) ----
    t0 = time.time()
    fmt_pairs, fmt_idx = build_condition_pairs(pairs, "format", limit=diag_limit)
    _, _, d_fmt, _ = judge_margins_batched(model, tokenizer, template, fmt_pairs, ans_ids, "cuda")
    format_sd = float(np.std(d_fmt))
    result["d3b1_format"] = {
        "n": int(len(d_fmt)),
        "coverage": len(fmt_idx) / len(pairs),
        "format_sd": format_sd,
        "median_abs": float(np.median(np.abs(d_fmt))),
        "snr_format": float(signal_sd / format_sd) if format_sd > 0 else float("inf"),
        "wall_clock_s": time.time() - t0,
    }
    print(f"  D3b-1 format: sd={format_sd:.4f} snr={result['d3b1_format']['snr_format']:.3f} "
          f"(signal_sd={signal_sd:.4f}, n={len(d_fmt)})")

    # ---- D3b-2: paraphrase sensitivity (diagnostic) ----
    t0 = time.time()
    para_pairs, para_idx = build_condition_pairs(pairs, "paraphrase", limit=diag_limit)
    if para_pairs:
        _, _, d_para, _ = judge_margins_batched(model, tokenizer, template, para_pairs, ans_ids, "cuda")
        para_sd = float(np.std(d_para))
        result["d3b2_paraphrase"] = {
            "n": int(len(d_para)),
            "coverage": len(para_idx) / len(pairs),
            "paraphrase_sd": para_sd,
            "median_abs": float(np.median(np.abs(d_para))),
            "snr_paraphrase": float(signal_sd / para_sd) if para_sd > 0 else float("inf"),
            "wall_clock_s": time.time() - t0,
        }
        print(f"  D3b-2 paraphrase: sd={para_sd:.4f} "
              f"snr={result['d3b2_paraphrase']['snr_paraphrase']:.3f} (n={len(d_para)}, "
              f"coverage={result['d3b2_paraphrase']['coverage']:.2f})")
    else:
        result["d3b2_paraphrase"] = {"n": 0, "coverage": 0.0, "note": "no rows had an applicable substitution"}

    del model
    after = reclaim()
    print(f"  VRAM baseline={baseline_vram:.3f}GB after={after:.3f}GB")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=600)
    parser.add_argument("--diag-limit", type=int, default=300,
                        help="rows for the D3b-1/D3b-2 diagnostics (gates always use all --n)")
    parser.add_argument("--judges", nargs="*", default=None)
    parser.add_argument("--tag", default="fixedbin",
                        help="which stored margin file to compare against (must match how the "
                             "stored margins were scored, or D3a measures batching, not logic)")
    args = parser.parse_args()

    cfg = load_cfg()
    template = cfg["judge_templates"]["template_train"]
    pairs = load_probe_pairs("probe", args.n)
    print(f"loaded {len(pairs)} probe pairs; gate tolerance {GATE_TOL:g}")

    if args.judges:
        judges = args.judges
    else:
        prefix = f"judge_margins_{args.tag}_" if args.tag else "judge_margins_"
        judges = sorted({p.name.replace(prefix, "").replace(".jsonl", "").replace("__", "/")
                         for p in CACHE_DIR.glob(f"{prefix}*.jsonl")})

    results = []
    for judge_name in judges:
        print(f"judge: {judge_name}")
        results.append(run_judge(judge_name, pairs, template, args.diag_limit, args.tag))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "validity_d3.json"
    with open(out_path, "w") as f:
        json.dump({"gate_tol": GATE_TOL, "n_pairs": len(pairs), "results": results}, f, indent=2)
    print(f"\nwrote {out_path}")

    gates_pass = all(r["d3a"]["passes"] and r["d3b0"]["passes"] for r in results)
    print(f"\nSTEP 1 HARD GATES: {'PASS' if gates_pass else 'FAIL'}")
    for r in results:
        print(f"  {r['judge']}: d3a_max={r['d3a']['max_abs']:.3e} dup_max={r['d3b0']['max_abs']:.3e}")


if __name__ == "__main__":
    main()
