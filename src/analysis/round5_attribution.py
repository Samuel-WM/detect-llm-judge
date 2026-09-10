"""NEXT_STEPS_ROUND_5_FINAL.md 6: candidate attribution analysis on the fresh final holdout.

The primary rule (`nnls_mixture`) is a DATASET-LEVEL fit. The final probes are not independent
NNLS classification trials, so the primary inference is a probe-label permutation test, not a
binomial test on per-probe pseudo-decisions (6.1, 6.4). A per-probe win rate is still reported for
interpretability, with its exact binomial CI, and is explicitly not the inferential basis.

Everything is run twice: on raw standardized candidate margins (PRIMARY) and on out-of-sample
length-residualized margins (required robustness, 6.7). Residualized success can never rescue raw
failure (12.3).

Run: .venv/Scripts/python.exe -m src.analysis.round5_attribution --arm RM_gpt2-helpful
     .venv/Scripts/python.exe -m src.analysis.round5_attribution --combine
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2

from src.attribute.rules import bt_detection_stats, nnls_mixture, run_all_rules

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "results"

N_PERM = 10_000
N_BOOT = 5_000
P_THRESHOLD = 0.001
PERM_SEED = 20250911
BOOT_SEED = 20250912


def lambda_threshold(M: int) -> float:
    """Pre-registered detection threshold. Lambda_m = 2N(L_m(gamma_hat) - L_m(0)) is a
    likelihood-ratio statistic against gamma_m = 0, asymptotically chi2(1) under that null,
    Bonferroni-corrected across the M candidates at the project's standing p < 0.001.

    gamma is bounded below at 1e-6, so the exact null is a 50:50 chi2(0)/chi2(1) mixture whose
    upper quantile is smaller than the plain chi2(1) quantile. Using plain chi2(1) is therefore
    the CONSERVATIVE choice for a detection claim, which is the direction we want.
    """
    return float(chi2.isf(P_THRESHOLD / M, df=1))


def alpha_raw(delta: np.ndarray, D: np.ndarray) -> np.ndarray:
    return nnls_mixture(delta, D)


def alpha_normalized(delta: np.ndarray, D: np.ndarray) -> np.ndarray:
    a = nnls_mixture(delta, D)
    s = a.sum()
    return a / s if s > 0 else a


def s_statistic(alpha: np.ndarray, true_idx: int) -> float:
    """S = alpha_true - max_{m != true} alpha_m.

    PRE-FREEZE CORRECTION, made before any DPO run and before any attribution outcome existed.
    6.4 defines S on the NORMALIZED weight vector. Measured on synthetic data with the real
    pool's correlation structure, that statistic is DEGENERATE under the permutation null: NNLS
    weights under a permuted Delta are tiny and noisy, so dividing by their sum amplifies them to
    a near-vertex point of the simplex. The null becomes bimodal at +/-1 (deciles -1.00, -0.60,
    +0.96; 99.9th percentile exactly +1.0000), while S_obs <= 1 by construction. The permutation
    p-value therefore cannot fall below roughly 0.10 no matter how strong the signal, which makes
    the pre-registered success threshold p_perm < 0.001 mathematically unsatisfiable.

    The same measurement on UNNORMALIZED NNLS coefficients gives a null concentrated at zero
    (sd 0.047, 99.9th percentile 0.124) and recovers a well-powered test. Since normalization is
    division by a positive scalar, TOP-1 IDENTIFICATION IS IDENTICAL either way -- only the
    magnitude statistic changes. S is therefore computed on unnormalized coefficients, and the
    normalized mixture weights are still reported everywhere they are called for (6.3, F19).
    """
    wrong = np.delete(alpha, true_idx)
    return float(alpha[true_idx] - wrong.max())


def analyze_checkpoint(delta: np.ndarray, D: np.ndarray, true_idx: int, M: int) -> dict:
    scores = run_all_rules(delta, D)
    det = bt_detection_stats(delta, D)
    alpha = alpha_normalized(delta, D)          # reported mixture weights (6.3, F19)
    araw = alpha_raw(delta, D)                  # statistic basis; same argmax by construction
    out = {"alpha": alpha.tolist(), "alpha_raw": araw.tolist(),
           "nnls_top1_idx": int(np.argmax(alpha)),
           "nnls_top1_correct": bool(int(np.argmax(alpha)) == true_idx) if true_idx is not None else None,
           "S_obs": s_statistic(araw, true_idx) if true_idx is not None else None,
           "S_obs_normalized": s_statistic(alpha, true_idx) if true_idx is not None else None,
           "bt_scores": scores["bradley_terry"].tolist(),
           "bt_top1_idx": int(np.argmax(scores["bradley_terry"])),
           "sign_scores": scores["sign_agreement"].tolist(),
           "sign_top1_idx": int(np.argmax(scores["sign_agreement"])),
           "Lambda": det["Lambda"].tolist(), "gamma_hat": det["gamma_hat"].tolist(),
           "lambda_threshold": lambda_threshold(M),
           "any_candidate_clears_lambda": bool(np.any(det["Lambda"] > lambda_threshold(M)))}
    for rule, key in [("bt", "bradley_terry"), ("sign", "sign_agreement")]:
        v = np.sort(scores[key])[::-1]
        out[f"{rule}_gap"] = float(v[0] - v[1])
    a = np.sort(alpha)[::-1]
    out["nnls_gap"] = float(a[0] - a[1])
    return out


def permutation_test(delta: np.ndarray, D: np.ndarray, true_idx: int, s_obs: float) -> dict:
    """6.4: keep the candidate margin matrix fixed, permute Delta across final probes, rerun the
    ENTIRE primary NNLS fit, recompute S. One-sided randomization p-value."""
    rng = np.random.default_rng(PERM_SEED)
    s_perm = np.empty(N_PERM)
    for k in range(N_PERM):
        s_perm[k] = s_statistic(alpha_raw(rng.permutation(delta), D), true_idx)
    p = (1 + int(np.sum(s_perm >= s_obs))) / (N_PERM + 1)
    return {"n_perm": N_PERM, "p_perm": float(p), "s_perm_mean": float(s_perm.mean()),
            "s_perm_sd": float(s_perm.std()), "s_perm_q995": float(np.quantile(s_perm, 0.995)),
            "s_perm": s_perm.tolist()}


def bootstrap(delta: np.ndarray, D: np.ndarray, true_idx: int) -> dict:
    """6.5: stability/uncertainty analysis, not the p-value."""
    rng = np.random.default_rng(BOOT_SEED)
    n = len(delta)
    s_boot = np.empty(N_BOOT)
    top1 = 0
    for k in range(N_BOOT):
        i = rng.integers(0, n, n)
        a = alpha_raw(delta[i], D[i])
        s_boot[k] = s_statistic(a, true_idx)
        top1 += int(np.argmax(a) == true_idx)
    return {"n_boot": N_BOOT, "top1_support": top1 / N_BOOT,
            "s_boot_median": float(np.median(s_boot)),
            "s_boot_ci95": [float(np.quantile(s_boot, 0.025)), float(np.quantile(s_boot, 0.975))],
            "s_boot": s_boot.tolist()}


def per_probe_winrate(delta: np.ndarray, D: np.ndarray, true_idx: int) -> dict:
    """6.6: descriptive only. Per probe, the candidate whose standardized margin is closest to
    Delta's standardized value wins; reported with an exact binomial CI and NOT used as the
    primary inferential basis."""
    z = (delta - delta.mean()) / delta.std()
    win = np.argmin(np.abs(D - z[:, None]), axis=1)
    k = int(np.sum(win == true_idx))
    bt = binomtest(k, len(delta), 1 / D.shape[1])
    ci = bt.proportion_ci()
    return {"win_rate": k / len(delta), "n": len(delta), "chance": 1 / D.shape[1],
            "exact_binomial_ci95": [ci.low, ci.high], "p_value_descriptive": float(bt.pvalue),
            "note": "descriptive; NOT the inferential basis for the dataset-level NNLS claim"}


def run_arm(arm: str) -> None:
    fd = json.load(open(RESULTS_DIR / "final_design_matrix.json"))
    S = fd["S"]
    M = len(S)
    D_raw = np.array(fd["D_std"])
    D_res = np.array(fd["D_resid_std"])
    traj = json.load(open(RESULTS_DIR / f"round5_{arm}_trajectory.json"))

    # "RM_<m>" and "DET_<m>" both carry a true teacher; CTRL and FJ do not.
    true_idx = None if arm in ("CTRL", "FJ") else S.index(arm.split("_", 1)[1])
    print(f"arm={arm}  M={M}  true={S[true_idx] if true_idx is not None else 'none (control/FJ)'}  "
          f"Lambda threshold={lambda_threshold(M):.3f}")

    per_ckpt = []
    for row in traj["rows"]:
        delta = np.array(row["delta"])
        rec = {"step": row["step"], "epoch": row["epoch"],
               "raw": analyze_checkpoint(delta, D_raw, true_idx, M),
               "resid": analyze_checkpoint(delta, D_res, true_idx, M)}
        per_ckpt.append(rec)
        print(f"  step {row['step']:4d}  raw top1={S[rec['raw']['nnls_top1_idx']]:14s} "
              f"S={rec['raw']['S_obs'] if rec['raw']['S_obs'] is not None else float('nan'):+.4f}  "
              f"maxLambda={max(rec['raw']['Lambda']):8.2f}")

    final = traj["rows"][-1]
    delta = np.array(final["delta"])
    out = {"arm": arm, "S": S, "M": M, "true_idx": true_idx,
           "final_step": final["step"], "n_final": traj["n_final"],
           "lambda_threshold": lambda_threshold(M), "per_checkpoint": per_ckpt}

    for tag, D in [("raw", D_raw), ("resid", D_res)]:
        block = analyze_checkpoint(delta, D, true_idx, M)
        if true_idx is not None:
            block["permutation"] = permutation_test(delta, D, true_idx, block["S_obs"])
            block["bootstrap"] = bootstrap(delta, D, true_idx)
            block["per_probe"] = per_probe_winrate(delta, D, true_idx)
            block["primary_success"] = bool(
                block["nnls_top1_correct"] and block["S_obs"] > 0
                and block["permutation"]["p_perm"] < P_THRESHOLD)
            print(f"  FINAL {tag}: top1={S[block['nnls_top1_idx']]}  S={block['S_obs']:+.4f}  "
                  f"p_perm={block['permutation']['p_perm']:.5f}  "
                  f"boot_top1={block['bootstrap']['top1_support']:.4f}  "
                  f"-> {'SUCCESS' if block['primary_success'] else 'FAIL'}")
        else:
            print(f"  FINAL {tag}: maxLambda={max(block['Lambda']):.2f} vs threshold "
                  f"{block['lambda_threshold']:.2f} -> "
                  f"{'CLEARS (control fails)' if block['any_candidate_clears_lambda'] else 'no candidate clears'}")
        out[f"final_{tag}"] = block

    with open(RESULTS_DIR / f"round5_{arm}_attribution.json", "w") as f:
        json.dump(out, f)
    print(f"wrote results/round5_{arm}_attribution.json")


def combine() -> None:
    fd = json.load(open(RESULTS_DIR / "final_design_matrix.json"))
    S, M = fd["S"], len(fd["S"])
    required = max(2, math.ceil(2 * M / 3))
    arms = [f"RM_{m}" for m in S]
    rows, successes = [], 0
    for arm in arms:
        p = RESULTS_DIR / f"round5_{arm}_attribution.json"
        if not p.exists():
            continue
        a = json.load(open(p))
        r, rs = a["final_raw"], a["final_resid"]
        successes += int(r["primary_success"])
        rows.append({
            "true teacher": arm[3:],
            "raw top-1": S[r["nnls_top1_idx"]], "raw S_obs": round(r["S_obs"], 4),
            "raw p_perm": f"{r['permutation']['p_perm']:.5f}",
            "raw boot top-1": round(r["bootstrap"]["top1_support"], 4),
            "raw verdict": "SUCCESS" if r["primary_success"] else "FAIL",
            "resid top-1": S[rs["nnls_top1_idx"]], "resid S_obs": round(rs["S_obs"], 4),
            "resid p_perm": f"{rs['permutation']['p_perm']:.5f}",
            "resid boot top-1": round(rs["bootstrap"]["top1_support"], 4),
            "resid verdict": "SUCCESS" if rs["primary_success"] else "FAIL",
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print(f"\nprimary successes: {successes}/{len(rows)}   required (kill criterion) = {required}")
    kill = successes < required
    print(f"kill criterion: {'FIRED - stop after reporting' if kill else 'not fired'}")
    print(f"aggregate claim (ALL arms attributed): {'HOLDS' if successes == M else 'does not hold'}")

    ctrl_p = RESULTS_DIR / "round5_CTRL_attribution.json"
    ctrl = None
    if ctrl_p.exists():
        c = json.load(open(ctrl_p))["final_raw"]
        ctrl = {"max_Lambda": max(c["Lambda"]), "threshold": c["lambda_threshold"],
                "clears": c["any_candidate_clears_lambda"]}
        print(f"CTRL: max Lambda {ctrl['max_Lambda']:.2f} vs {ctrl['threshold']:.2f} -> "
              f"negative control {'FAILS' if ctrl['clears'] else 'PASSES'}")

    out = {"S": S, "M": M, "required_successes": required, "successes": successes,
           "kill_criterion_fired": kill, "aggregate_claim_holds": successes == M,
           "rows": rows, "ctrl": ctrl,
           "deterministic_arm_triggered": bool(successes == M and ctrl is not None and not ctrl["clears"])}
    with open(RESULTS_DIR / "round5_summary.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {RESULTS_DIR / 'round5_summary.json'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm")
    ap.add_argument("--combine", action="store_true")
    a = ap.parse_args()
    if a.combine:
        combine()
    else:
        run_arm(a.arm)


if __name__ == "__main__":
    main()
