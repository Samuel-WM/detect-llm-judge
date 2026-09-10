"""Track A: Phase 0c revisions per NEXT_STEPS_TRACKS_A_B_C.md. CPU only, target < 45 min.

A0/A1 (report existing marginals + per-rule Fig 1) need no new computation -- see RESULTS.md
and src/analysis/figures_phase0c.py:fig1b_heatmap_per_rule. This script covers A2-A5, which do.

Interpretation / correction notes:

  - A5's literal construction ("apply phi(x2)=phi(x1)+delta*u to a controlled fraction of probe
    rows", u presumably random per row) does NOT push kappa_D past ~3.7 no matter how small
    delta gets or how large the fraction, because a handful of full-scale isotropic rows
    already span R^16 -- adding rows that are merely small in EVERY direction doesn't create a
    direction-specific deficiency, it just becomes numerically negligible as delta shrinks
    (verified empirically before committing to a sweep design). Fixed by projecting a single
    shared random direction `v` out of every row, then adding it back at one *fixed, identical*
    magnitude `delta` (no per-row variance) -- now the design's only information about `v` is a
    constant offset whose contribution to the smallest eigenvalue of X^T X is exactly
    N * delta^2, so kappa_D grows smoothly and unboundedly as delta -> 0. Verified this spans
    ~3.4 to ~5.7e4 across delta in {1, 0.1, 0.01, 0.001, 0.0001}, comfortably several orders of
    magnitude as intended.
  - A2/A3 regenerate the primary-map "positive" replicates fresh at every cell (200 new draws
    per cell) rather than reusing the original run's stored rows, since gamma_hat/Lambda need
    the raw (Delta, D) per replicate, which was not persisted (only scalar summaries were).
    Replicate-to-replicate noise between this run and the original primary map is expected and
    is itself informative (see RESULTS.md note on Marginal C's two independently-drawn sigma=0.5
    rows).
  - AUC computed via the rank-sum (Mann-Whitney U) identity, no sklearn dependency:
    AUC = (sum of positive-class ranks - n_pos*(n_pos+1)/2) / (n_pos * n_neg).

Run: .venv/Scripts/python.exe -m src.phase0c_track_a
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import rankdata

from src.attribute.rules import RULE_NAMES, bt_detection_stats, run_all_rules
from src.common.io import write_manifest
from src.common.seeds import set_all_seeds
from src.phase0c_synthetic import (
    add_noise, build_judges, condition_number, sample_phi_diff_pool,
)
from src.probe.select import select_disagreement_max, select_pairwise_coverage, select_random

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_cfg() -> dict:
    with open(REPO_ROOT / "configs" / "phase0c.yaml") as f:
        return yaml.safe_load(f)


def auc_rank_sum(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    pos_scores = pos_scores[np.isfinite(pos_scores)]
    neg_scores = neg_scores[np.isfinite(neg_scores)]
    n_pos, n_neg = len(pos_scores), len(neg_scores)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    combined = np.concatenate([pos_scores, neg_scores])
    ranks = rankdata(combined)
    rank_sum_pos = ranks[:n_pos].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def f1_optimal_threshold(pos_scores: np.ndarray, neg_scores: np.ndarray) -> dict:
    pos_scores = pos_scores[np.isfinite(pos_scores)]
    neg_scores = neg_scores[np.isfinite(neg_scores)]
    all_scores = np.concatenate([pos_scores, neg_scores])
    labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    candidates = np.unique(all_scores)
    best = {"f1": -1.0, "tau": None, "tpr": None, "fpr": None}
    for tau in candidates:
        pred = (all_scores > tau).astype(int)
        tp = int(np.sum((pred == 1) & (labels == 1)))
        fp = int(np.sum((pred == 1) & (labels == 0)))
        fn = int(np.sum((pred == 0) & (labels == 1)))
        tn = int(np.sum((pred == 0) & (labels == 0)))
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        if f1 > best["f1"]:
            best.update({
                "f1": f1, "tau": float(tau),
                "tpr": tp / (tp + fn) if (tp + fn) > 0 else float("nan"),
                "fpr": fp / (fp + tn) if (fp + tn) > 0 else float("nan"),
            })
    return best


# ------------------------------------------------------------------------- A2 + A3 --------


def run_one_group(rng, d, M, psi_deg, sigma, lam, N, pool_mult, group: str) -> dict:
    """group in {"positive", "control1", "control2"}. Returns per-rule margin/gamma_hat/Lambda
    plus top1_correct (positive group only)."""
    w, u = build_judges(rng, d, M, psi_deg)
    true_idx = int(rng.integers(M))
    pool_size = pool_mult * N
    phi_diff_pool = sample_phi_diff_pool(rng, pool_size, d, u, lam)
    D_clean_pool = phi_diff_pool @ w.T

    if group == "control2":
        # Delta has no dependence on any w_m at all; the sigma_D noise mechanism doesn't apply
        # since there's no clean signal to add noise to -- draw it directly at a matched scale.
        typical_sd = D_clean_pool.std()
        Delta_noisy_pool = typical_sd * rng.standard_normal(pool_size)
    else:
        Delta_clean_pool = phi_diff_pool @ w[true_idx]
        Delta_noisy_pool = add_noise(rng, Delta_clean_pool, sigma)

    D_noisy_pool = add_noise(rng, D_clean_pool, sigma)
    idx = select_random(N)
    Delta_sel = Delta_noisy_pool[idx]
    D_sel = D_noisy_pool[idx]
    if group == "control1":
        D_sel = np.delete(D_sel, true_idx, axis=1)

    scores_by_rule = run_all_rules(Delta_sel, D_sel)
    bt_stats = bt_detection_stats(Delta_sel, D_sel)

    out = {}
    for rule_name in RULE_NAMES:
        scores = scores_by_rule[rule_name]
        order = np.argsort(scores)[::-1]
        top_m = int(order[0])
        margin = float(scores[top_m] - scores[order[1]]) if len(order) > 1 else float("nan")
        out[rule_name] = {
            "margin": margin,
            "gamma_hat": float(bt_stats["gamma_hat"][top_m]),
            "Lambda": float(bt_stats["Lambda"][top_m]),
            "top1_correct": int(top_m == true_idx) if group == "positive" else None,
        }
    return out


def run_detection_grid(rng, cfg, n_reps: int) -> pd.DataFrame:
    d, M = cfg["d"], cfg["M"]
    pool_mult = cfg["pool_multiplier"]
    lam = cfg["primary_map"]["lambda_aniso"]
    N = cfg["primary_map"]["N"]

    rows = []
    for psi_deg in cfg["psi_grid_deg"]:
        for sigma in cfg["sigma_grid"]:
            groups = {"positive": [], "control1": [], "control2": []}
            for _rep in range(n_reps):
                for group in ("positive", "control1", "control2"):
                    groups[group].append(run_one_group(rng, d, M, psi_deg, sigma, lam, N, pool_mult, group))

            for rule_name in RULE_NAMES:
                pos = groups["positive"]
                neg = groups["control1"] + groups["control2"]
                pos_margin = np.array([r[rule_name]["margin"] for r in pos])
                neg_margin = np.array([r[rule_name]["margin"] for r in neg])
                pos_gamma = np.array([r[rule_name]["gamma_hat"] for r in pos])
                neg_gamma = np.array([r[rule_name]["gamma_hat"] for r in neg])
                pos_lambda = np.array([r[rule_name]["Lambda"] for r in pos])
                neg_lambda = np.array([r[rule_name]["Lambda"] for r in neg])
                top1_acc = float(np.mean([r[rule_name]["top1_correct"] for r in pos]))

                thr = f1_optimal_threshold(pos_margin, neg_margin)
                rows.append({
                    "psi_deg": psi_deg, "sigma": sigma, "rule": rule_name,
                    "top1_accuracy": top1_acc,
                    "margin_tau": thr["tau"], "margin_f1": thr["f1"],
                    "margin_tpr": thr["tpr"], "margin_fpr": thr["fpr"],
                    "auc_margin": auc_rank_sum(pos_margin, neg_margin),
                    "auc_gamma_hat": auc_rank_sum(pos_gamma, neg_gamma),
                    "auc_Lambda": auc_rank_sum(pos_lambda, neg_lambda),
                    "n_pos": len(pos), "n_neg": len(neg),
                })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------- A4 --------


def run_marginal_a_three_strategies(rng, cfg, n_reps: int) -> pd.DataFrame:
    d, M = cfg["d"], cfg["M"]
    ma = cfg["marginal_a"]
    pool_mult = cfg["pool_multiplier"]
    strategies = ["random", "per_row_topk", "pairwise_coverage"]
    rows = []
    for N in cfg["N_grid"]:
        for strategy in strategies:
            for _rep in range(n_reps):
                w, u = build_judges(rng, d, M, ma["psi_deg"])
                true_idx = int(rng.integers(M))
                pool_size = pool_mult * N
                phi_diff_pool = sample_phi_diff_pool(rng, pool_size, d, u, ma["lambda_aniso"])
                D_clean_pool = phi_diff_pool @ w.T
                Delta_clean_pool = phi_diff_pool @ w[true_idx]
                Delta_noisy_pool = add_noise(rng, Delta_clean_pool, ma["sigma"])
                D_noisy_pool = add_noise(rng, D_clean_pool, ma["sigma"])

                if strategy == "random":
                    idx = select_random(N)
                elif strategy == "per_row_topk":
                    idx = select_disagreement_max(D_noisy_pool, N)
                elif strategy == "pairwise_coverage":
                    idx = select_pairwise_coverage(D_noisy_pool, N)
                else:
                    raise ValueError(strategy)

                Delta_sel, D_sel = Delta_noisy_pool[idx], D_noisy_pool[idx]
                scores_by_rule = run_all_rules(Delta_sel, D_sel)
                for rule_name in RULE_NAMES:
                    scores = scores_by_rule[rule_name]
                    m_hat = int(np.argmax(scores))
                    rows.append({
                        "N": N, "strategy": strategy, "rule": rule_name,
                        "top1_correct": int(m_hat == true_idx),
                    })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------- A5 --------


def run_conditioning_fix(rng, cfg, n_reps: int) -> pd.DataFrame:
    d, M = cfg["d"], cfg["M"]
    psi_deg = cfg["marginal_a"]["psi_deg"]
    sigma = cfg["marginal_a"]["sigma"]
    lam = cfg["marginal_a"]["lambda_aniso"]
    N = 600  # marginal_a config doesn't set N (that's swept in A4); matches the spec's Fig 4 point
    pool_mult = cfg["pool_multiplier"]
    deltas = [1.0, 0.1, 0.01, 0.001, 0.0001]

    rows = []
    for delta in deltas + ["baseline_sigma0"]:
        for _rep in range(n_reps):
            w, u = build_judges(rng, d, M, psi_deg)
            true_idx = int(rng.integers(M))
            pool_size = pool_mult * N
            phi_diff_pool = sample_phi_diff_pool(rng, pool_size, d, u, lam)

            if delta != "baseline_sigma0":
                v = rng.standard_normal(d)
                v /= np.linalg.norm(v)
                phi_no_v = phi_diff_pool - (phi_diff_pool @ v)[:, None] * v[None, :]
                phi_diff_pool = phi_no_v + delta * v[None, :]

            D_clean_pool = phi_diff_pool @ w.T
            Delta_clean_pool = phi_diff_pool @ w[true_idx]
            use_sigma = 0.0 if delta == "baseline_sigma0" else sigma
            Delta_sel = add_noise(rng, Delta_clean_pool, use_sigma)[:N]
            D_sel = add_noise(rng, D_clean_pool, use_sigma)[:N]
            phi_diff_sel = phi_diff_pool[:N]

            w_hat, *_ = np.linalg.lstsq(phi_diff_sel, Delta_sel, rcond=None)
            denom = np.linalg.norm(w_hat) * np.linalg.norm(w[true_idx])
            theta = float(np.arccos(np.clip(np.dot(w_hat, w[true_idx]) / denom, -1, 1))) if denom > 0 else float("nan")
            kappa = condition_number(phi_diff_sel)

            rows.append({"delta": str(delta), "sigma": use_sigma, "kappa_D": kappa, "theta": theta})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------- main -------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sections", nargs="*", default=None)
    parser.add_argument("--reps", type=int, default=200)
    args = parser.parse_args()
    sections = set(args.sections) if args.sections else {"a2a3", "a4", "a5"}

    cfg = load_cfg()
    set_all_seeds(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"] + 1000)  # offset from phase0c_synthetic's stream

    t0 = time.time()
    results_dir = REPO_ROOT / "results"
    timings = {}

    if "a2a3" in sections:
        t = time.time()
        df_grid = run_detection_grid(rng, cfg, args.reps)
        timings["a2a3_detection_grid"] = time.time() - t
        df_grid.to_parquet(results_dir / "phase0c_track_a_detection_grid.parquet", index=False)
        print(f"A2/A3 detection grid: {len(df_grid)} rows in {timings['a2a3_detection_grid']:.1f}s")

    if "a4" in sections:
        t = time.time()
        df_ma = run_marginal_a_three_strategies(rng, cfg, args.reps)
        timings["a4_marginal_a"] = time.time() - t
        df_ma.to_parquet(results_dir / "phase0c_track_a_marginal_a_strategies.parquet", index=False)
        print(f"A4 marginal A (3 strategies): {len(df_ma)} rows in {timings['a4_marginal_a']:.1f}s")

    if "a5" in sections:
        t = time.time()
        df_cond = run_conditioning_fix(rng, cfg, args.reps)
        timings["a5_conditioning"] = time.time() - t
        df_cond.to_parquet(results_dir / "phase0c_track_a_conditioning.parquet", index=False)
        print(f"A5 conditioning fix: {len(df_cond)} rows in {timings['a5_conditioning']:.1f}s")

    wall_clock = time.time() - t0
    print(f"\nTrack A total wall clock: {wall_clock:.1f}s")

    write_manifest(
        f"phase0c_track_a_{int(time.time())}", config={"phase0c": cfg}, results_dir=results_dir,
        seed=cfg["seed"], extra={"wall_clock_s": wall_clock, "timings": timings, "sections_run": list(sections)},
    )


if __name__ == "__main__":
    main()
