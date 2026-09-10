"""NEXT_STEPS_ROUND_4_FINAL.md §2.4 / round-3 §3.7: D1 for the reward-model pool.

Pre-registered in `configs/base.yaml:reward_model_stage.d1_pre_registration` BEFORE any
reward-model scoring, and not modified after seeing results:

    primary   = large-gap UltraFeedback subset (score_chosen - score_rejected >= 3)
    secondary = full held-out UltraFeedback set
    retain if |accuracy - 0.5| >= 0.10 AND two-sided exact binomial p < 0.001

The statistic is distance FROM CHANCE, so a stable anti-correlated model passes; a below-chance
result satisfying the rule is not reinterpreted as failure.

Prospective per-model expectations, written before scoring (round-4 §2.4):
    OpenAssistant  - expected positive structure
    gpt2-helpful   - expected positive structure
    gpt2-harmless  - direction unknown on generic helpfulness labels; result is a measurement
    internlm2      - unknown

One process per model (in-process sequencing of these models produced an all-NaN artifact).

Run: .venv/Scripts/python.exe -m src.judges.rm_d1 --only <repo>
     .venv/Scripts/python.exe -m src.judges.rm_d1 --combine
"""
import argparse
import json
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import time
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from src.common.vram import reclaim
from src.data.build_prompts import load_cfg
from src.judges import rm_scorer
from src.judges.rm_harness import CTX, INTERNLM, POOL, SHORT, load_internlm, load_model, score_internlm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = REPO_ROOT / "data_cache"
RESULTS_DIR = REPO_ROOT / "results"

EXPECTATION = {
    "OpenAssistant/reward-model-deberta-v3-large-v2": "expected positive structure",
    "Ray2333/gpt2-large-helpful-reward_model": "expected positive structure",
    "Ray2333/gpt2-large-harmless-reward_model": "direction unknown on generic helpfulness labels",
    INTERNLM: "unknown",
}


def length_ok(tok, repo: str, pairs: list[dict], is_int: bool, model=None) -> list[int]:
    """Indices whose (prompt, response) inputs fit this model's context under the exact
    serialization it will be scored with. Never truncates to force coverage."""
    cap = CTX[repo]
    keep = []
    for i, p in enumerate(pairs):
        lens = []
        for which in ("response_a", "response_b"):
            if is_int:
                conv = [{"role": "user", "content": p["prompt"]},
                        {"role": "assistant", "content": p[which]}]
                text = tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
                lens.append(len(tok.encode(text, add_special_tokens=False)) + 1)
            else:
                lens.append(len(tok.encode(rm_scorer.serialize(p["prompt"], p[which]),
                                           add_special_tokens=True)))
        if max(lens) <= cap:
            keep.append(i)
    return keep


def evaluate(repo: str, pairs: list[dict], label: str, rule: dict) -> dict:
    is_int = repo == INTERNLM
    model, tok = (load_internlm(repo) if is_int else load_model(repo))
    keep = length_ok(tok, repo, pairs, is_int)
    sub = [pairs[i] for i in keep]
    t0 = time.time()
    if is_int:
        r_a = score_internlm(model, tok, sub, "response_a")
        r_b = score_internlm(model, tok, sub, "response_b")
        d = r_a - r_b
    else:
        _, _, d = rm_scorer.score_pairs(model, tok, sub, "cuda", CTX[repo])
        d = d.numpy()
    del model
    reclaim()

    n = len(d)
    n_correct = int(np.sum(d > 0))
    acc = n_correct / n
    dist = abs(acc - rule["null_accuracy"])
    bt = binomtest(n_correct, n, rule["null_accuracy"], alternative=rule["binomial_alternative"])
    passes = bool(dist >= rule["min_abs_distance_from_chance"] and bt.pvalue < rule["binomial_p_threshold"])
    return {"set": label, "n_scored": n, "n_available": len(pairs),
            "coverage": n / len(pairs), "accuracy": acc, "abs_distance": dist,
            "p_value": float(bt.pvalue), "direction": "above chance" if acc > 0.5 else "below chance",
            "passes": passes, "wall_clock_s": time.time() - t0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    ap.add_argument("--combine", action="store_true")
    args = ap.parse_args()

    cfg = load_cfg()
    rule = cfg["reward_model_stage"]["d1_pre_registration"]

    if args.combine:
        import pandas as pd
        rows = []
        for repo in POOL:
            p = RESULTS_DIR / f"rm_d1_{repo.replace('/', '__')}.json"
            if not p.exists():
                print(f"missing {p.name}")
                continue
            r = json.load(open(p))
            lg, fs = r["large_gap"], r["full_set"]
            rows.append({
                "model": SHORT[repo],
                "D1_large_gap_acc": round(lg["accuracy"], 4), "lg_|d|": round(lg["abs_distance"], 4),
                "lg_p": f"{lg['p_value']:.2e}", "lg_n": lg["n_scored"], "lg_pass": lg["passes"],
                "D1_full_acc": round(fs["accuracy"], 4), "full_|d|": round(fs["abs_distance"], 4),
                "full_p": f"{fs['p_value']:.2e}", "full_n": fs["n_scored"], "full_pass": fs["passes"],
                "expectation": EXPECTATION.get(repo, ""),
            })
        df = pd.DataFrame(rows)
        print(f"PRE-REGISTERED RULE: |acc-0.5| >= {rule['min_abs_distance_from_chance']} "
              f"AND two-sided exact binomial p < {rule['binomial_p_threshold']}; "
              f"PRIMARY = large-gap subset\n")
        print(df.to_string(index=False))
        df.to_csv(RESULTS_DIR / "rm_d1_summary.csv", index=False)
        print(f"\nwrote {RESULTS_DIR / 'rm_d1_summary.csv'}")
        return

    repo = args.only
    print(f"=== D1: {SHORT.get(repo, repo)} ===")
    print(f"prospective expectation (written before scoring): {EXPECTATION.get(repo)}")
    large_gap = [json.loads(l) for l in open(CACHE_DIR / "d1_large_gap_pairs.jsonl")]
    full_set = [json.loads(l) for l in open(CACHE_DIR / "d1_external_pairs.jsonl")]

    res = {"model": repo, "expectation": EXPECTATION.get(repo)}
    res["large_gap"] = evaluate(repo, large_gap, "large_gap_primary", rule)
    print(f"  PRIMARY large-gap: n={res['large_gap']['n_scored']} acc={res['large_gap']['accuracy']:.4f} "
          f"|d|={res['large_gap']['abs_distance']:.4f} p={res['large_gap']['p_value']:.3e} "
          f"-> {'PASS' if res['large_gap']['passes'] else 'FAIL'}")
    res["full_set"] = evaluate(repo, full_set, "full_set_secondary", rule)
    print(f"  secondary full-set: n={res['full_set']['n_scored']} acc={res['full_set']['accuracy']:.4f} "
          f"|d|={res['full_set']['abs_distance']:.4f} p={res['full_set']['p_value']:.3e} "
          f"-> {'PASS' if res['full_set']['passes'] else 'FAIL'}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / f"rm_d1_{repo.replace('/', '__')}.json", "w") as f:
        json.dump(res, f)
    print(f"  wrote results/rm_d1_{repo.replace('/', '__')}.json")


if __name__ == "__main__":
    main()
