"""NEXT_STEPS_ROUND_5_FINAL.md 5.4: the J_perp feature-judge control.

J_perp is a linear reward over the standing 7 handcrafted response features whose margin is
constructed to be uncorrelated with response-length difference. It supplies an EXACT known teacher
margin for the transmission / map-calibration analysis, which no real reward model can.

Constructed from TRAINING DATA ONLY (5.4): feature standardization constants, the orthogonality
projection and the label standardization all come from the frozen training pairs. Correlations on
the 487 calibration probes and on the fresh final holdout are computed afterwards and are
REPORTING-ONLY -- nothing about J_perp is tuned on them.

Requirement: |corr(d_true_train, len_diff_train)| < 0.05.

J_len is deliberately NOT built in this round (5.4): length is addressed by C4 and by the mandatory
raw-versus-residualized attribution analysis instead.

Run: .venv/Scripts/python.exe -m src.judges.jperp_r5 --build
"""
import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from src.data.build_prompts import load_cfg
from src.phase1_feature_judges import (FEATURE_NAMES, build_feature_matrix, build_judges,
                                       select_judge)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = REPO_ROOT / "data_cache"
RESULTS_DIR = REPO_ROOT / "results"
MAX_ABS_CORR = 0.05
SEED = 0


def len_diff(ptok, pairs: list[dict]) -> np.ndarray:
    return np.array([len(ptok.encode(p["response_a"])) - len(ptok.encode(p["response_b"]))
                     for p in pairs], dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.parse_args()

    cfg = load_cfg()
    ptok = AutoTokenizer.from_pretrained(cfg["models"]["base_policy"])
    train_all = [json.loads(l) for l in open(CACHE_DIR / "response_pairs_train.jsonl")]
    keep = json.load(open(RESULTS_DIR / "rm_trainset_r5.json"))["retained_indices"]
    train = [train_all[i] for i in keep]
    print(f"J_perp built on the frozen training set: {len(train)} pairs")

    # feature standardization from TRAINING responses only
    texts = [t for p in train for t in (p["response_a"], p["response_b"])]
    phi = build_feature_matrix(texts)
    mu, sd = phi.mean(axis=0), phi.std(axis=0)
    sd[sd == 0] = 1.0

    def phidiff(pairs):
        a = (build_feature_matrix([p["response_a"] for p in pairs]) - mu) / sd
        b = (build_feature_matrix([p["response_b"] for p in pairs]) - mu) / sd
        return a - b

    P_tr = phidiff(train)
    x_tr = len_diff(ptok, train)

    # Start from the standing orthonormal judge basis for continuity, then project the chosen
    # direction onto the subspace orthogonal to the training length direction. corr(P@w, x) = 0
    # exactly when w is orthogonal to v = P^T (x - xbar).
    W = build_judges(seed=SEED)
    chosen, _ = select_judge(W)
    w0 = W[chosen]
    v = P_tr.T @ (x_tr - x_tr.mean())
    w = w0 - (w0 @ v) / (v @ v) * v
    w = w / np.linalg.norm(w)

    d_tr = P_tr @ w
    corr_tr = float(np.corrcoef(d_tr, x_tr)[0, 1])
    ok = abs(corr_tr) < MAX_ABS_CORR
    print(f"corr(d_true, len_diff) on TRAINING = {corr_tr:+.6f}  "
          f"(requirement |corr| < {MAX_ABS_CORR}): {'PASS' if ok else 'FAIL'}")
    print(f"cosine to the pre-projection direction = {float(w0 @ w):.4f}")
    assert ok, "J_perp construction failed its own training-set orthogonality requirement"

    mu_train, sd_train = float(d_tr.mean()), float(d_tr.std())

    # reporting-only correlations, computed after construction is frozen
    report = {}
    for name, path in [("calibration_487", CACHE_DIR / "response_pairs_probe.jsonl"),
                       ("final_holdout", CACHE_DIR / "final_holdout.jsonl")]:
        if not path.exists():
            continue
        pairs = [json.loads(l) for l in open(path)]
        if name == "calibration_487":
            idx = json.load(open(RESULTS_DIR / "rm_probe_filter.json"))["retained_indices"]
            pairs = [pairs[i] for i in idx]
        d = phidiff(pairs) @ w
        c = float(np.corrcoef(d, len_diff(ptok, pairs))[0, 1])
        report[name] = {"n": len(pairs), "corr_d_true_len_diff": c, "d_true": d.tolist()}
        print(f"corr(d_true, len_diff) on {name} = {c:+.4f}   [REPORTING ONLY, not tuned on]")

    out = {"seed": SEED, "feature_names": FEATURE_NAMES, "w": w.tolist(),
           "base_judge_index": chosen, "feature_mu": mu.tolist(), "feature_sd": sd.tolist(),
           "standardization_source": "frozen training pairs only",
           "corr_train": corr_tr, "max_abs_corr_requirement": MAX_ABS_CORR, "pass": ok,
           "mu_train": mu_train, "sd_train": sd_train,
           "d_true_train": d_tr.tolist(), "reporting_only": report,
           "len_diff_convention": "base-policy tokens"}
    with open(CACHE_DIR / "jperp_r5.json", "w") as f:
        json.dump(out, f)
    print(f"wrote {CACHE_DIR / 'jperp_r5.json'}")


if __name__ == "__main__":
    main()
