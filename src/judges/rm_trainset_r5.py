"""NEXT_STEPS_ROUND_5_FINAL.md 3: training-pair precheck and frozen training set.

A training pair is retained only if EVERY retained candidate reward-model scorer and the DPO policy
path can process it under its exact final serialization without truncation or context violation.

The policy path uses `build_scoring_inputs`, which is the identical serialization applied at
generation, DPO training and Delta scoring; its cap is the standing dpo.max_length = 640.

No response is truncated and no context limit is relaxed to preserve sample size (3.2). If fewer
than 250 pairs survive, the run stops before DPO training.

Run: .venv/Scripts/python.exe -m src.judges.rm_trainset_r5
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from src.common.templating import build_scoring_inputs
from src.data.build_prompts import load_cfg
from src.judges import rm_scorer
from src.judges.rm_harness import CTX, SHORT

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = REPO_ROOT / "data_cache"
RESULTS_DIR = REPO_ROOT / "results"
MIN_TRAIN_PAIRS = 250


def main() -> None:
    cfg = load_cfg()
    policy_cap = cfg["dpo"]["max_length"]
    S = json.load(open(RESULTS_DIR / "rm_calibration_r5.json"))["S"]
    repos = [r for r, s in SHORT.items() if s in S]
    pairs = [json.loads(l) for l in open(CACHE_DIR / "response_pairs_train.jsonl")]
    print(f"retained pool = {S}   policy cap = {policy_cap}   starting pairs = {len(pairs)}")

    toks = {r: AutoTokenizer.from_pretrained(r) for r in repos}
    ptok = AutoTokenizer.from_pretrained(cfg["models"]["base_policy"])

    lens = {SHORT[r]: np.zeros(len(pairs), dtype=int) for r in repos}
    lens["policy(Qwen)"] = np.zeros(len(pairs), dtype=int)
    for i, p in enumerate(pairs):
        for r in repos:
            lens[SHORT[r]][i] = max(
                len(toks[r].encode(rm_scorer.serialize(p["prompt"], p[w]), add_special_tokens=True))
                for w in ("response_a", "response_b"))
        lens["policy(Qwen)"][i] = max(
            build_scoring_inputs(ptok, p["prompt"], p[w])[0].shape[0]
            for w in ("response_a", "response_b"))

    caps = {SHORT[r]: CTX[r] for r in repos}
    caps["policy(Qwen)"] = policy_cap
    over = {k: lens[k] > caps[k] for k in lens}
    excluded_any = np.zeros(len(pairs), dtype=bool)
    for v in over.values():
        excluded_any |= v
    keep = [i for i in range(len(pairs)) if not excluded_any[i]]

    tbl = pd.DataFrame([{
        "scorer": k, "context_cap": caps[k], "max_realized_len": int(lens[k].max()),
        "median_len": int(np.median(lens[k])), "p99_len": int(np.percentile(lens[k], 99)),
        "n_excluded_by_this_scorer": int(over[k].sum()),
        "n_excluded_uniquely_by_this_scorer": int((over[k] & (sum(over.values()) == 1)).sum()),
    } for k in lens])
    print("\n=== 3.2 exact-serialization length check ===")
    print(tbl.to_string(index=False))

    frac = len(keep) / len(pairs)
    print(f"\nstarting pairs      = {len(pairs)}")
    print(f"retained pairs      = {len(keep)}")
    print(f"retained fraction   = {frac:.4f}")
    print(f"excluded overall    = {int(excluded_any.sum())}")
    ok = len(keep) >= MIN_TRAIN_PAIRS
    print(f"\nGate (>= {MIN_TRAIN_PAIRS} pairs): {'PASS' if ok else 'STOP'}")

    ids = [pairs[i].get("pair_id", i) for i in keep]
    manifest = hashlib.sha256(
        json.dumps([{"i": i, "prompt": pairs[i]["prompt"], "a": pairs[i]["response_a"],
                     "b": pairs[i]["response_b"]} for i in keep], sort_keys=True).encode()
    ).hexdigest()
    out = {"starting_pairs": len(pairs), "retained_pairs": len(keep), "retained_fraction": frac,
           "excluded_overall": int(excluded_any.sum()), "min_required": MIN_TRAIN_PAIRS,
           "gate_pass": ok, "policy_cap": policy_cap, "pool": S,
           "per_scorer": tbl.to_dict(orient="records"),
           "retained_indices": keep, "retained_pair_ids": ids,
           "frozen_manifest_sha256": manifest,
           "no_truncation": "every retained pair fits every required scorer without truncation"}
    with open(RESULTS_DIR / "rm_trainset_r5.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"frozen training manifest sha256 = {manifest}")
    print(f"wrote {RESULTS_DIR / 'rm_trainset_r5.json'}")


if __name__ == "__main__":
    main()
