"""NEXT_STEPS_ROUND_2.md Step 2: D1 large-gap diagnostic.

Secondary number -- the pre-registered D1 verdict from the prior round does NOT change regardless
of this outcome (see prohibitions: "Do not revisit the D1 retention threshold").

Reruns D1 restricted to the subset of the same UltraFeedback pool with `score_chosen -
score_rejected >= 3` (an "easy" gap by the dataset's own GPT-4-derived quality scores). Reports
subset size, per-judge accuracy on it, distance from chance, and a two-sided exact binomial p vs
0.5, alongside the full-set numbers already in `results/d1_external.json`.

Interpretation stated in advance, not fitted after seeing numbers (the document's own framing):
  - still at chance on easy pairs -> the Step 4 verdict is unambiguous.
  - jumps to >=0.65 -> the judges have real content sensitivity but can't resolve fine gaps,
    which changes what stage 3 of the pool ladder requires. Does not change Step 4.

Run: .venv/Scripts/python.exe -m src.judges.d1_large_gap
"""
import json
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from scipy.stats import binomtest
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common.vram import free_vram_gb, reclaim
from src.data.build_prompts import build_prompt_splits, load_cfg
from src.judges.d1_external import _text_of
from src.judges.margins import get_answer_token_ids, judge_margins_batched
from tests.test_templating import run as run_templating_check

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = REPO_ROOT / "data_cache"
RESULTS_DIR = REPO_ROOT / "results"
GAP_THRESHOLD = 3.0
N_PAIRS = 500  # matches the original D1's N for a comparable-power diagnostic


def build_large_gap_pairs(cfg: dict) -> list[dict]:
    """Same held-out pool D1 drew from (prompts disjoint from train/probe), filtered to
    score_chosen - score_rejected >= GAP_THRESHOLD. Cached separately from the original
    (unscored) D1 pairs file so neither consumer is disturbed.
    """
    cache_path = CACHE_DIR / "d1_large_gap_pairs.jsonl"
    if cache_path.exists():
        rows = [json.loads(l) for l in open(cache_path)]
        print(f"reusing cached large-gap pairs ({len(rows)}) from {cache_path}")
        return rows[:N_PAIRS]

    splits = build_prompt_splits(cfg, cfg["seed"])
    used = set(splits["train_prompts"]) | set(splits["probe_prompts"])
    ds = load_dataset(cfg["data"]["dataset"], split="train_prefs")

    rows = []
    for row in ds:
        if len(rows) >= N_PAIRS:
            break
        if row["prompt"] in used:
            continue
        gap = row.get("score_chosen") - row.get("score_rejected") if row.get("score_chosen") is not None else None
        if gap is None or gap < GAP_THRESHOLD:
            continue
        chosen, rejected = _text_of(row["chosen"]), _text_of(row["rejected"])
        if not chosen.strip() or not rejected.strip():
            continue
        rows.append({
            "prompt": row["prompt"], "response_a": chosen, "response_b": rejected,
            "score_chosen": row["score_chosen"], "score_rejected": row["score_rejected"], "gap": gap,
        })

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"built {len(rows)} large-gap (>= {GAP_THRESHOLD}) pairs -> {cache_path}")
    return rows


def main() -> None:
    cfg = load_cfg()
    pairs = build_large_gap_pairs(cfg)
    template = cfg["judge_templates"]["template_train"]
    null_acc = cfg["judge_validity"]["d1_external_structure"]["null_accuracy"]

    # load full-set D1 numbers for the side-by-side report
    full_set_path = RESULTS_DIR / "d1_external.json"
    full_set = {}
    if full_set_path.exists():
        for r in json.load(open(full_set_path))["results"]:
            if "accuracy" in r:
                full_set[r["judge"]] = r

    results = []
    for judge_name in cfg["models"]["judge_pool"]:
        print(f"\njudge: {judge_name}")
        try:
            run_templating_check(judge_name)
            baseline = free_vram_gb()
            tok = AutoTokenizer.from_pretrained(judge_name)
            tok.padding_side = "left"
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                judge_name, dtype=torch.bfloat16, attn_implementation="sdpa",
            ).to("cuda")
            model.eval()
            ans_ids = get_answer_token_ids(tok)

            t0 = time.time()
            _, _, d_m, _ = judge_margins_batched(model, tok, template, pairs, ans_ids, "cuda")
            n_correct = int(np.sum(d_m > 0))
            acc = n_correct / len(d_m)
            bt = binomtest(n_correct, len(d_m), null_acc, alternative="two-sided")
            dist = abs(acc - null_acc)

            full = full_set.get(judge_name, {})
            results.append({
                "judge": judge_name, "n": len(d_m), "n_correct": n_correct,
                "large_gap_accuracy": acc, "large_gap_abs_distance": dist,
                "large_gap_binomial_p": float(bt.pvalue),
                "full_set_accuracy": full.get("accuracy"),
                "full_set_abs_distance": full.get("abs_distance_from_chance"),
                "wall_clock_s": time.time() - t0,
            })
            print(f"  large-gap accuracy={acc:.4f}  |acc-0.5|={dist:.4f}  p={bt.pvalue:.3e}  "
                  f"(full-set was {full.get('accuracy', 'n/a')})  ({time.time()-t0:.0f}s)")

            del model
            after = reclaim()
            print(f"  VRAM baseline={baseline:.3f}GB after={after:.3f}GB")
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {str(e)[:200]}")
            results.append({"judge": judge_name, "error": f"{type(e).__name__}: {str(e)[:200]}"})
            reclaim()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "d1_large_gap.json"
    with open(out_path, "w") as f:
        json.dump({"gap_threshold": GAP_THRESHOLD, "n_pairs": len(pairs), "results": results}, f, indent=2)
    print(f"\nwrote {out_path}")

    print("\n=== D1 large-gap summary (secondary diagnostic; does not change the Step 4 verdict) ===")
    for r in results:
        if "error" in r:
            print(f"  {r['judge']}: ERROR")
            continue
        print(f"  {r['judge']}: full_set_acc={r['full_set_accuracy']}  "
              f"large_gap_acc={r['large_gap_accuracy']:.4f}  p={r['large_gap_binomial_p']:.3e}")


if __name__ == "__main__":
    main()
