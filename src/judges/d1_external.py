"""Step 3 of NEXT_STEPS_FINAL.md: D1 external preference structure.

Scores each judge on held-out UltraFeedback pairs with known chosen/rejected labels, through the
existing `judges/margins.py` path (same templating, same debiasing, same padding-free scoring).

    accuracy_m = mean_i 1[ d_m(x_i, chosen_i, rejected_i) > 0 ]

The statistic is DISTANCE FROM CHANCE, not distance from 1.0 -- a stable anti-correlated judge
still defines an attributable preference function. The retention rule is pre-registered in
configs/base.yaml (judge_validity.d1_external_structure) and is read from there, not hardcoded
here, so it cannot be adjusted after seeing the numbers.

Prompts are drawn from UltraFeedback rows whose prompt appears in NEITHER the train nor the probe
split, so this is held out from everything else in the project.

Run: .venv/Scripts/python.exe -m src.judges.d1_external
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
from src.judges.margins import get_answer_token_ids, judge_margins_batched
from tests.test_templating import run as run_templating_check

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = REPO_ROOT / "data_cache"
RESULTS_DIR = REPO_ROOT / "results"


def _text_of(msg_field) -> str:
    """UltraFeedback 'chosen'/'rejected' are message lists; take the assistant turn."""
    if isinstance(msg_field, str):
        return msg_field
    for turn in reversed(msg_field):
        if turn.get("role") == "assistant":
            return turn.get("content", "")
    return msg_field[-1].get("content", "") if msg_field else ""


def build_d1_pairs(cfg: dict, n_pairs: int) -> list[dict]:
    cache_path = CACHE_DIR / "d1_external_pairs.jsonl"
    if cache_path.exists():
        rows = [json.loads(l) for l in open(cache_path)]
        if len(rows) >= n_pairs:
            print(f"reusing cached D1 pairs ({len(rows)}) from {cache_path}")
            return rows[:n_pairs]

    splits = build_prompt_splits(cfg, cfg["seed"])
    used = set(splits["train_prompts"]) | set(splits["probe_prompts"])
    ds = load_dataset(cfg["data"]["dataset"], split="train_prefs")

    rows = []
    for row in ds:
        if len(rows) >= n_pairs:
            break
        if row["prompt"] in used:
            continue
        chosen, rejected = _text_of(row["chosen"]), _text_of(row["rejected"])
        if not chosen.strip() or not rejected.strip():
            continue
        # response_a = chosen, response_b = rejected  ->  d_m > 0 means the judge agrees
        rows.append({"prompt": row["prompt"], "response_a": chosen, "response_b": rejected})

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"built {len(rows)} held-out D1 pairs -> {cache_path}")
    return rows


def main() -> None:
    cfg = load_cfg()
    rule = cfg["judge_validity"]["d1_external_structure"]
    n_pairs = rule["n_pairs"]
    min_dist = rule["min_abs_distance_from_chance"]
    p_thresh = rule["binomial_p_threshold"]
    null_acc = rule["null_accuracy"]
    print(f"PRE-REGISTERED D1 rule (from configs/base.yaml): |acc - {null_acc}| >= {min_dist} "
          f"AND two-sided exact binomial p < {p_thresh}, N={n_pairs}")

    pairs = build_d1_pairs(cfg, n_pairs)
    template = cfg["judge_templates"]["template_train"]

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
            passes = bool(dist >= min_dist and bt.pvalue < p_thresh)
            results.append({
                "judge": judge_name, "n": len(d_m), "n_correct": n_correct,
                "accuracy": acc, "abs_distance_from_chance": dist,
                "binomial_p": float(bt.pvalue), "passes": passes,
                "wall_clock_s": time.time() - t0,
            })
            print(f"  accuracy={acc:.4f}  |acc-0.5|={dist:.4f}  p={bt.pvalue:.3e}  "
                  f"verdict={'PASS' if passes else 'FAIL'}  ({time.time()-t0:.0f}s)")

            del model
            after = reclaim()
            print(f"  VRAM baseline={baseline:.3f}GB after={after:.3f}GB")
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {str(e)[:200]}")
            results.append({"judge": judge_name, "error": f"{type(e).__name__}: {str(e)[:200]}"})
            reclaim()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "d1_external.json"
    with open(out_path, "w") as f:
        json.dump({"rule": rule, "results": results}, f, indent=2)
    print(f"\nwrote {out_path}")

    print("\n=== D1 summary ===")
    for r in results:
        if "error" in r:
            print(f"  {r['judge']}: ERROR {r['error']}")
        else:
            print(f"  {r['judge']}: acc={r['accuracy']:.4f} |d|={r['abs_distance_from_chance']:.4f} "
                  f"p={r['binomial_p']:.3e} -> {'PASS' if r['passes'] else 'FAIL'}")


if __name__ == "__main__":
    main()
