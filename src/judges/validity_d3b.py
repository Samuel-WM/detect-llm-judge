"""NEXT_STEPS_ROUND_2.md Step 1: D3b, done properly. The prior round's D3b-0 (identical-token
duplicate, `d_m(x,y,y)`) is degenerate -- for identical token sequences, `ell_AB = p`,
`ell_BA = -p`, so `d_m = 0.5*(ell_AB + ell_BA) = 0` bitwise BY CONSTRUCTION of the debiasing
algebra, regardless of whether the judge perceives anything. It re-confirms D3a's algebra, not
content sensitivity.

This constructs two NON-identical, semantically-equivalent variants of each response and measures
whether the judge is actually responding to content:

  whitespace: reflow line breaks / spacing / list-bullet spacing only. No content token changed
              in meaning, but the TOKEN SEQUENCE must differ from the original (asserted; pairs
              where it doesn't are dropped rather than silently counted as null).
  reworded:   a uniform mechanical substitution (contraction expansion, e.g. "don't" -> "do not")
              applied identically across every pair. Listed in full below and in RESULTS.md.

Per judge, per construction (never pooled together):

    floor_m  = sd_i( d_m(x_i, y_i, y_tilde_i) )     null pairs, true content difference ~zero
    signal_m = sd_i( d_m(x_i, y_i, y_i') )           real probe pairs
    snr_m    = signal_m / floor_m
    mean_i( d_m(x_i, y_i, y_tilde_i) )               second check: should be near zero

Run: .venv/Scripts/python.exe -m src.judges.validity_d3b
"""
import argparse
import json
import os
import re
import textwrap

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common.vram import free_vram_gb, reclaim
from src.data.build_prompts import load_cfg
from src.judges.margins import get_answer_token_ids, judge_margins_batched, load_probe_pairs
from tests.test_templating import run as run_templating_check

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "results"


def whitespace_variant(tokenizer, text: str) -> str | None:
    """Reflow only: normalize intra-paragraph whitespace, re-wrap at a different width than the
    original's natural formatting. Returns None if the result would be empty.
    """
    paragraphs = re.split(r"\n\s*\n", text)
    out = []
    for para in paragraphs:
        collapsed = re.sub(r"\s+", " ", para).strip()
        if not collapsed:
            continue
        out.append(textwrap.fill(collapsed, width=58))  # deliberately different width than validity_d3's 72
    result = "\n\n".join(out)
    return result if result.strip() else None


# Same substitution table as the prior round's D3b diagnostic -- conservative, meaning-preserving
# expansions only (nothing that could change meaning, like possessive "'s" or ambiguous "'d").
REWORDED_SUBS = [
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


def reworded_variant(text: str) -> tuple[str, int]:
    out, n = text, 0
    for pattern, repl in REWORDED_SUBS:
        out, k = re.subn(pattern, repl, out)
        n += k
    return out, n


def build_null_pairs(tokenizer, pairs: list[dict], construction: str) -> tuple[list[dict], list[int]]:
    """Returns (null_pairs, source_indices). null_pairs[i] = {"prompt", "response_a": y_i,
    "response_b": y_tilde_i}. Rows dropped where the variant doesn't produce a differing token
    sequence (whitespace) or has no applicable substitution (reworded).
    """
    out, idxs = [], []
    for i, p in enumerate(pairs):
        y = p["response_a"]
        if construction == "whitespace":
            variant = whitespace_variant(tokenizer, y)
            if variant is None:
                continue
            orig_ids = tokenizer.encode(y)
            var_ids = tokenizer.encode(variant)
            if orig_ids == var_ids:
                continue  # assert-and-drop: must be a genuinely different token sequence
        elif construction == "reworded":
            variant, n_sub = reworded_variant(y)
            if n_sub == 0:
                continue
        else:
            raise ValueError(construction)
        out.append({"prompt": p["prompt"], "response_a": y, "response_b": variant})
        idxs.append(i)
    return out, idxs


def run_judge(judge_name: str, pairs: list[dict], template: str, signal_d_m: dict[int, float]) -> dict:
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

    result = {"judge": judge_name}
    for construction in ("whitespace", "reworded"):
        null_pairs, idxs = build_null_pairs(tokenizer, pairs, construction)
        _, _, d_null, _ = judge_margins_batched(model, tokenizer, template, null_pairs, ans_ids, "cuda")

        signal_vals = np.array([signal_d_m[i] for i in idxs])
        floor_m = float(np.std(d_null))
        signal_m = float(np.std(signal_vals))
        snr_m = signal_m / floor_m if floor_m > 0 else float("inf")
        mean_null = float(np.mean(d_null))

        result[construction] = {
            "n": len(null_pairs), "coverage": len(idxs) / len(pairs),
            "floor_m": floor_m, "signal_m": signal_m, "snr_m": snr_m, "mean_null": mean_null,
        }
        print(f"  {construction}: n={len(null_pairs)} (coverage {len(idxs)/len(pairs):.2f})  "
              f"floor_m={floor_m:.4f}  signal_m={signal_m:.4f}  snr_m={snr_m:.3f}  "
              f"mean_null={mean_null:.4f}")

    del model
    after = reclaim()
    print(f"  VRAM baseline={baseline_vram:.3f}GB after={after:.3f}GB")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=600)
    parser.add_argument("--tag", default="fixedbin")
    args = parser.parse_args()

    cfg = load_cfg()
    template = cfg["judge_templates"]["template_train"]
    pairs = load_probe_pairs("probe", args.n)
    print(f"loaded {len(pairs)} probe pairs")
    print(f"reworded substitution rule: {len(REWORDED_SUBS)} conservative contraction expansions "
          f"(full list in this file / RESULTS.md)")

    results = []
    for judge_name in cfg["models"]["judge_pool"]:
        print(f"\njudge: {judge_name}")
        run_templating_check(judge_name)
        safe = judge_name.replace("/", "__")
        margins_path = REPO_ROOT / "data_cache" / f"judge_margins_{args.tag}_{safe}.jsonl"
        assert margins_path.exists(), f"{margins_path} missing -- run src.judges.margins --tag {args.tag} first"
        signal_d_m = {}
        with open(margins_path) as f:
            for line in f:
                row = json.loads(line)
                if row["template_name"] == "template_train":
                    signal_d_m[row["prompt_idx"]] = row["d_m"]

        results.append(run_judge(judge_name, pairs, template, signal_d_m))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "validity_d3b.json"
    with open(out_path, "w") as f:
        json.dump({"n_pairs": len(pairs), "reworded_rule": REWORDED_SUBS, "results": results}, f, indent=2)
    print(f"\nwrote {out_path}")

    print("\n=== D3b summary ===")
    for r in results:
        ws, rw = r["whitespace"], r["reworded"]
        print(f"  {r['judge']}: snr_ws={ws['snr_m']:.3f}  snr_rw={rw['snr_m']:.3f}")


if __name__ == "__main__":
    main()
