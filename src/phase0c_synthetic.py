"""Phase 0c: identifiability map. Addendum to Phase 0 per PHASE_0C_SPEC.md. CPU only, no GPU,
no network. Target under 30 minutes wall clock.

Interpretation calls flagged explicitly (the spec leaves these open at the individual-probe
level, same as the main document does for Phase 2):

  - Noise scale (Section 2): sigma_D, sigma_d are fractions of the standard deviation of the
    CLEAN quantity across the full candidate POOL (size pool_multiplier * N), computed before
    selection, not the standard deviation of the final N-subset. This keeps the noise scale
    comparable between the `random` and `disagreement_max` selection strategies, which would
    otherwise see different post-selection variances by construction.
  - Selection (Section 5): "greedily select" is implemented as a per-row score
    s_i = min_{m!=m'} |d_tilde_{m,i} - d_tilde_{m',i}| ranked and top-N taken -- the formula as
    written has no cross-row interaction term, so this is a literal reading, not an approximation
    of a set-interaction greedy algorithm. Same convention will be used for the real Phase 4.
  - Negative controls + Section 4.6 battery (Sections 9, "one representative cell"): run at the
    fixed point shared with marginals A/B/C (psi=20deg, sigma_D=sigma_d=0.5, lambda=4, N=600,
    select=random), configs/phase0c.yaml `representative_cell`. "Positive cells" for the
    threshold fit = the primary map's replicates at that same (psi, sigma) cell.
  - Section 4.6 test battery needs PER-PROBE decisions; our replicate-level rules aggregate over
    all N probes into one scalar per candidate. Decomposed sign_agreement into its per-probe
    term t_i^m = 1[sign(Delta_i) == sign(d_{m,i})] for the identification test (per-probe
    decision = argmax_m d_tilde_{m,i} * sign(Delta_tilde_i), i.e. sign-agreement weighted by
    margin magnitude, avoiding tie problems with the raw binary term), and used the paired
    per-probe difference t_i^top - t_i^runnerup (top/runner-up chosen by the replicate's
    aggregate sign_agreement score) for the margin significance test. Exercises the code path
    per the spec's instruction; a real per-probe statistic for the other two rules is not
    otherwise used in this project.
  - Mixture identifiability (Section 10) fixes lambda_aniso=4, matching every other section's
    default operating point; the spec doesn't state a lambda for this section.

Run: .venv/Scripts/python.exe -m src.phase0c_synthetic
"""
import argparse
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.attribute.rules import (
    RULE_NAMES, angular_error, nnls_mixture, recover_weights_ols, run_all_rules,
)
from src.common.io import append_results, write_manifest
from src.common.seeds import set_all_seeds
from src.probe.select import select_disagreement_max, select_random
from src.stats.tests import identification_binomial_test, margin_significance_test

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_config() -> dict:
    with open(REPO_ROOT / "configs" / "phase0c.yaml") as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------ construction primitives ---

def sample_unit_vector(rng: np.random.Generator, d: int) -> np.ndarray:
    v = rng.standard_normal(d)
    return v / np.linalg.norm(v)


def build_judges(rng: np.random.Generator, d: int, M: int, psi_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """w_m = cos(a) u + sin(a) v_m, v_m mutually orthonormal and orthogonal to u, so every pair
    sits at angle psi = arccos(cos(a)^2). Requires d >= M + 1. Returns (w (M, d), u (d,)).
    """
    assert d >= M + 1, f"need d >= M+1 for M orthonormal directions orthogonal to u, got d={d}, M={M}"
    psi = np.radians(psi_deg)
    a = np.arccos(np.sqrt(np.cos(psi)))

    u = sample_unit_vector(rng, d)
    G = rng.standard_normal((d, M))
    G -= np.outer(u, u @ G)  # project out the u-component from every column
    V, _r = np.linalg.qr(G)  # (d, M) orthonormal columns, orthogonal to u by construction
    w = (np.cos(a) * u[:, None] + np.sin(a) * V).T  # (M, d)
    return w, u


def sample_probe_pool(rng: np.random.Generator, pool_size: int, d: int, u: np.ndarray, lam: float) -> np.ndarray:
    """z ~ Normal(0, Sigma), Sigma = I + (lambda-1) u u^T, phi = z / ||z||. Sampled via
    Sigma^(1/2) = I + (sqrt(lambda)-1) u u^T (exact, since u is an eigenvector with eigenvalue
    lambda and the orthogonal complement has eigenvalue 1).
    """
    z0 = rng.standard_normal((pool_size, d))
    z = z0 + (np.sqrt(lam) - 1.0) * (z0 @ u)[:, None] * u[None, :]
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    return z


def sample_phi_diff_pool(rng: np.random.Generator, pool_size: int, d: int, u: np.ndarray, lam: float) -> np.ndarray:
    phi1 = sample_probe_pool(rng, pool_size, d, u, lam)
    phi2 = sample_probe_pool(rng, pool_size, d, u, lam)
    return phi1 - phi2


def add_noise(rng: np.random.Generator, clean: np.ndarray, sigma: float) -> np.ndarray:
    if sigma == 0.0:
        return clean.copy()
    if clean.ndim == 1:
        sd = clean.std()
        return clean + sigma * sd * rng.standard_normal(clean.shape)
    sd = clean.std(axis=0, keepdims=True)
    return clean + sigma * sd * rng.standard_normal(clean.shape)


def condition_number(X: np.ndarray) -> float:
    s = np.linalg.svd(X, compute_uv=False)
    s = s[s > 1e-15]
    return float(s.max() / s.min()) if len(s) > 0 else float("inf")


def empirical_agreement(D: np.ndarray) -> float:
    M = D.shape[1]
    iu = np.triu_indices(M, k=1)
    signs = np.sign(D)
    return float((signs[:, iu[0]] == signs[:, iu[1]]).mean())


def min_pairwise_margin_sep(D: np.ndarray) -> float:
    M = D.shape[1]
    iu = np.triu_indices(M, k=1)
    mean_abs_gap = np.abs(D[:, iu[0]] - D[:, iu[1]]).mean(axis=0)
    return float(mean_abs_gap.min())


def psi_to_agreement(psi_deg: float) -> float:
    """agreement(psi) = 1 - psi/pi (psi in radians): lower bound under isotropic probe features."""
    return 1.0 - np.radians(psi_deg) / np.pi


def agreement_to_psi_deg(agreement: float) -> float:
    return float(np.degrees(np.pi * (1.0 - agreement)))


# --------------------------------------------------------------------------- one replicate ---

def run_one_replicate(
    rng: np.random.Generator, d: int, M: int, psi_deg: float, sigma_D: float, sigma_d: float,
    lam: float, N: int, select: str, pool_multiplier: int, true_idx: int | None = None,
    w_true_override: np.ndarray | None = None,
) -> dict:
    w, u = build_judges(rng, d, M, psi_deg)
    if true_idx is None:
        true_idx = int(rng.integers(M))

    pool_size = pool_multiplier * N
    phi_diff_pool = sample_phi_diff_pool(rng, pool_size, d, u, lam)
    D_clean_pool = phi_diff_pool @ w.T  # (pool_size, M)
    w_true_vec = w[true_idx] if w_true_override is None else w_true_override
    Delta_clean_pool = phi_diff_pool @ w_true_vec  # (pool_size,)

    Delta_noisy_pool = add_noise(rng, Delta_clean_pool, sigma_D)
    D_noisy_pool = add_noise(rng, D_clean_pool, sigma_d)

    if select == "random":
        idx = select_random(N)
    elif select == "disagreement_max":
        idx = select_disagreement_max(D_noisy_pool, N)
    else:
        raise ValueError(select)

    Delta_sel = Delta_noisy_pool[idx]
    D_sel = D_noisy_pool[idx]
    phi_diff_sel = phi_diff_pool[idx]

    scores_by_rule = run_all_rules(Delta_sel, D_sel)
    w_hat = recover_weights_ols(Delta_sel, phi_diff_sel)
    theta = angular_error(w_hat, w_true_vec)

    per_rule = {}
    for rule_name in RULE_NAMES:
        scores = scores_by_rule[rule_name]
        m_hat = int(np.argmax(scores))
        sorted_scores = np.sort(scores)[::-1]
        margin = float(sorted_scores[0] - sorted_scores[1])
        per_rule[rule_name] = {
            "candidate_judge": m_hat,
            "top1_correct": int(m_hat == true_idx),
            "score": float(scores[m_hat]),
            "margin": margin,
        }

    return {
        "true_judge": true_idx,
        "w": w,
        "u": u,
        "theta": theta,
        "kappa_D": condition_number(phi_diff_sel),
        "empirical_agreement": empirical_agreement(D_sel),
        "min_pairwise_margin_sep": min_pairwise_margin_sep(D_sel),
        "per_rule": per_rule,
        "Delta_sel": Delta_sel,
        "D_sel": D_sel,
    }


def rows_from_replicate(rep_result: dict, meta: dict) -> list[dict]:
    rows = []
    for rule_name in RULE_NAMES:
        pr = rep_result["per_rule"][rule_name]
        row = dict(meta)
        row.update({
            "rule": rule_name,
            "true_judge": rep_result["true_judge"],
            "candidate_judge": pr["candidate_judge"],
            "top1_correct": pr["top1_correct"],
            "score": pr["score"],
            "margin": pr["margin"],
            "angular_error_rad": rep_result["theta"],
            "kappa_D": rep_result["kappa_D"],
            "empirical_agreement": rep_result["empirical_agreement"],
            "min_pairwise_margin_sep": rep_result["min_pairwise_margin_sep"],
        })
        rows.append(row)
    return rows


# --------------------------------------------------------------------------------- sections ---

def run_primary_map(rng, cfg) -> list[dict]:
    d, M = cfg["d"], cfg["M"]
    pm = cfg["primary_map"]
    rows = []
    for psi_deg, sigma in itertools.product(cfg["psi_grid_deg"], cfg["sigma_grid"]):
        for rep in range(cfg["replicates"]):
            result = run_one_replicate(
                rng, d, M, psi_deg, sigma, sigma, pm["lambda_aniso"], pm["N"], pm["select"],
                cfg["pool_multiplier"],
            )
            meta = {
                "section": "primary_map", "psi_deg": psi_deg, "sigma_D": sigma, "sigma_d": sigma,
                "lambda_aniso": pm["lambda_aniso"], "n_probes": pm["N"],
                "probe_selection": pm["select"], "replicate": rep,
            }
            rows.extend(rows_from_replicate(result, meta))
    return rows


def run_marginal_a(rng, cfg) -> list[dict]:
    d, M = cfg["d"], cfg["M"]
    ma = cfg["marginal_a"]
    rows = []
    for N, select in itertools.product(cfg["N_grid"], ["random", "disagreement_max"]):
        for rep in range(cfg["replicates"]):
            result = run_one_replicate(
                rng, d, M, ma["psi_deg"], ma["sigma"], ma["sigma"], ma["lambda_aniso"], N, select,
                cfg["pool_multiplier"],
            )
            meta = {
                "section": "marginal_a_N_select", "psi_deg": ma["psi_deg"], "sigma_D": ma["sigma"],
                "sigma_d": ma["sigma"], "lambda_aniso": ma["lambda_aniso"], "n_probes": N,
                "probe_selection": select, "replicate": rep,
            }
            rows.extend(rows_from_replicate(result, meta))
    return rows


def run_marginal_b(rng, cfg) -> list[dict]:
    d, M = cfg["d"], cfg["M"]
    mb = cfg["marginal_b"]
    rows = []
    for lam, select in itertools.product(cfg["lambda_grid"], ["random", "disagreement_max"]):
        for rep in range(cfg["replicates"]):
            result = run_one_replicate(
                rng, d, M, mb["psi_deg"], mb["sigma"], mb["sigma"], lam, mb["N"], select,
                cfg["pool_multiplier"],
            )
            meta = {
                "section": "marginal_b_anisotropy", "psi_deg": mb["psi_deg"], "sigma_D": mb["sigma"],
                "sigma_d": mb["sigma"], "lambda_aniso": lam, "n_probes": mb["N"],
                "probe_selection": select, "replicate": rep,
            }
            rows.extend(rows_from_replicate(result, meta))
    return rows


def run_marginal_c(rng, cfg) -> list[dict]:
    d, M = cfg["d"], cfg["M"]
    mc = cfg["marginal_c"]
    fixed = mc["fixed_sigma"]
    rows = []
    for sigma_D in cfg["sigma_grid"]:
        for rep in range(cfg["replicates"]):
            result = run_one_replicate(
                rng, d, M, mc["psi_deg"], sigma_D, fixed, mc["lambda_aniso"], mc["N"], "random",
                cfg["pool_multiplier"],
            )
            meta = {
                "section": "marginal_c_decoupled_D", "psi_deg": mc["psi_deg"], "sigma_D": sigma_D,
                "sigma_d": fixed, "lambda_aniso": mc["lambda_aniso"], "n_probes": mc["N"],
                "probe_selection": "random", "replicate": rep,
            }
            rows.extend(rows_from_replicate(result, meta))
    for sigma_d in cfg["sigma_grid"]:
        for rep in range(cfg["replicates"]):
            result = run_one_replicate(
                rng, d, M, mc["psi_deg"], fixed, sigma_d, mc["lambda_aniso"], mc["N"], "random",
                cfg["pool_multiplier"],
            )
            meta = {
                "section": "marginal_c_decoupled_d", "psi_deg": mc["psi_deg"], "sigma_D": fixed,
                "sigma_d": sigma_d, "lambda_aniso": mc["lambda_aniso"], "n_probes": mc["N"],
                "probe_selection": "random", "replicate": rep,
            }
            rows.extend(rows_from_replicate(result, meta))
    return rows


def run_mixture_identifiability(rng, cfg) -> list[dict]:
    d, M = cfg["d"], cfg["M"]
    mi = cfg["mixture_identifiability"]
    rows = []
    for psi_deg, (a1, a2) in itertools.product(mi["psi_deg_grid"], mi["alphas"]):
        for rep in range(cfg["replicates"]):
            w, u = build_judges(rng, d, M, psi_deg)
            w_true_vec = a1 * w[0] + a2 * w[1]
            alpha_true = np.zeros(M)
            alpha_true[0] = a1
            alpha_true[1] = a2

            pool_size = cfg["pool_multiplier"] * mi["N"]
            phi_diff_pool = sample_phi_diff_pool(rng, pool_size, d, u, mi["lambda_aniso"])
            D_clean_pool = phi_diff_pool @ w.T
            Delta_clean_pool = phi_diff_pool @ w_true_vec

            Delta_noisy = add_noise(rng, Delta_clean_pool, mi["sigma"])
            D_noisy = add_noise(rng, D_clean_pool, mi["sigma"])

            idx = select_random(mi["N"])
            Delta_sel, D_sel = Delta_noisy[idx], D_noisy[idx]

            alpha_hat = nnls_mixture(Delta_sel, D_sel)
            l1 = alpha_hat.sum()
            alpha_hat_norm = alpha_hat / l1 if l1 > 0 else alpha_hat
            recovery_l1_error = float(np.sum(np.abs(alpha_hat_norm - alpha_true)))

            rows.append({
                "section": "mixture_identifiability", "psi_deg": psi_deg, "sigma_D": mi["sigma"],
                "sigma_d": mi["sigma"], "lambda_aniso": mi["lambda_aniso"], "n_probes": mi["N"],
                "probe_selection": "random", "replicate": rep, "rule": "nnls_mixture",
                "alpha_1": a1, "alpha_2": a2, "recovery_l1_error": recovery_l1_error,
            })
    return rows


def run_negative_controls_and_stat_battery(rng, cfg, primary_rows: list[dict]) -> tuple[list[dict], dict]:
    d, M = cfg["d"], cfg["M"]
    rc = cfg["representative_cell"]
    n_reps = cfg["replicates"]
    rows = []

    # --- Control 1: true judge removed from the candidate pool ---
    control1_margins = {r: [] for r in RULE_NAMES}
    for rep in range(n_reps):
        w, u = build_judges(rng, d, M, rc["psi_deg"])
        true_idx = int(rng.integers(M))
        pool_size = cfg["pool_multiplier"] * rc["N"]
        phi_diff_pool = sample_phi_diff_pool(rng, pool_size, d, u, rc["lambda_aniso"])
        D_clean_pool = phi_diff_pool @ w.T
        Delta_clean_pool = phi_diff_pool @ w[true_idx]

        Delta_noisy = add_noise(rng, Delta_clean_pool, rc["sigma"])
        D_noisy = add_noise(rng, D_clean_pool, rc["sigma"])
        idx = select_random(rc["N"])
        Delta_sel = Delta_noisy[idx]
        D_sel_full = D_noisy[idx]
        D_sel_reduced = np.delete(D_sel_full, true_idx, axis=1)  # M-1 candidates, true judge excluded

        scores_by_rule = run_all_rules(Delta_sel, D_sel_reduced)
        for rule_name in RULE_NAMES:
            scores = scores_by_rule[rule_name]
            sorted_scores = np.sort(scores)[::-1]
            margin = float(sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) > 1 else float("nan")
            control1_margins[rule_name].append(margin)
            rows.append({
                "section": "control1_true_judge_removed", "psi_deg": rc["psi_deg"],
                "sigma_D": rc["sigma"], "sigma_d": rc["sigma"], "lambda_aniso": rc["lambda_aniso"],
                "n_probes": rc["N"], "probe_selection": rc["select"], "replicate": rep,
                "rule": rule_name, "margin": margin, "true_judge": true_idx,
            })

    # --- Control 2: Delta is pure noise, unrelated to any w_m ---
    control2_margins = {r: [] for r in RULE_NAMES}
    for rep in range(n_reps):
        w, u = build_judges(rng, d, M, rc["psi_deg"])
        pool_size = cfg["pool_multiplier"] * rc["N"]
        phi_diff_pool = sample_phi_diff_pool(rng, pool_size, d, u, rc["lambda_aniso"])
        D_clean_pool = phi_diff_pool @ w.T
        # Delta with no dependence on any w_m: scale matched to a typical clean Delta for
        # comparability, but drawn independently of phi_diff_pool / w entirely.
        typical_sd = D_clean_pool.std()
        Delta_pure_noise = typical_sd * rng.standard_normal(pool_size)

        D_noisy = add_noise(rng, D_clean_pool, rc["sigma"])
        idx = select_random(rc["N"])
        Delta_sel = Delta_pure_noise[idx]
        D_sel = D_noisy[idx]

        scores_by_rule = run_all_rules(Delta_sel, D_sel)
        for rule_name in RULE_NAMES:
            scores = scores_by_rule[rule_name]
            sorted_scores = np.sort(scores)[::-1]
            margin = float(sorted_scores[0] - sorted_scores[1])
            control2_margins[rule_name].append(margin)
            rows.append({
                "section": "control2_no_signal", "psi_deg": rc["psi_deg"], "sigma_D": rc["sigma"],
                "sigma_d": rc["sigma"], "lambda_aniso": rc["lambda_aniso"], "n_probes": rc["N"],
                "probe_selection": rc["select"], "replicate": rep, "rule": rule_name,
                "margin": margin,
            })

    # --- Threshold fit: positive cells = primary map replicates at the matching (psi, sigma) ---
    threshold_report = {}
    for rule_name in RULE_NAMES:
        positive_margins = [
            r["margin"] for r in primary_rows
            if r["rule"] == rule_name and r["psi_deg"] == rc["psi_deg"]
            and r["sigma_D"] == rc["sigma"] and r["sigma_d"] == rc["sigma"]
        ]
        neg_margins = [m for m in control1_margins[rule_name] if np.isfinite(m)] + \
                      [m for m in control2_margins[rule_name] if np.isfinite(m)]
        pos = np.array(positive_margins)
        neg = np.array(neg_margins)
        all_margins = np.concatenate([pos, neg])
        labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])

        candidate_taus = np.unique(all_margins)
        best_f1, best_tau, best_tpr, best_fpr = -1.0, None, None, None
        for tau in candidate_taus:
            pred = (all_margins > tau).astype(int)
            tp = int(np.sum((pred == 1) & (labels == 1)))
            fp = int(np.sum((pred == 1) & (labels == 0)))
            fn = int(np.sum((pred == 0) & (labels == 1)))
            tn = int(np.sum((pred == 0) & (labels == 0)))
            f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
            if f1 > best_f1:
                best_f1 = f1
                best_tau = float(tau)
                best_tpr = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
                best_fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")

        threshold_report[rule_name] = {
            "tau": best_tau, "f1": best_f1, "tpr": best_tpr, "fpr": best_fpr,
            "n_positive": len(pos), "n_negative": len(neg),
        }

    # --- Section 4.6 test battery on one representative replicate ---
    w, u = build_judges(rng, d, M, rc["psi_deg"])
    true_idx = int(rng.integers(M))
    pool_size = cfg["pool_multiplier"] * rc["N"]
    phi_diff_pool = sample_phi_diff_pool(rng, pool_size, d, u, rc["lambda_aniso"])
    D_clean_pool = phi_diff_pool @ w.T
    Delta_clean_pool = phi_diff_pool @ w[true_idx]
    Delta_noisy_pool = add_noise(rng, Delta_clean_pool, rc["sigma"])
    D_noisy_pool = add_noise(rng, D_clean_pool, rc["sigma"])
    idx = select_random(rc["N"])
    Delta_sel, D_sel = Delta_noisy_pool[idx], D_noisy_pool[idx]

    # per-probe identification decision: argmax_m d_tilde_{m,i} * sign(Delta_tilde_i)
    per_probe_scores = D_sel * np.sign(Delta_sel)[:, None]  # (N, M)
    per_probe_decision = np.argmax(per_probe_scores, axis=1)
    X_i = (per_probe_decision == true_idx).astype(int)
    identification_result = identification_binomial_test(int(X_i.sum()), len(X_i), M)

    aggregate_scores = run_all_rules(Delta_sel, D_sel)["sign_agreement"]
    order = np.argsort(aggregate_scores)[::-1]
    top_m, runnerup_m = int(order[0]), int(order[1])
    t_i = (np.sign(Delta_sel)[:, None] == np.sign(D_sel)).astype(float)  # (N, M) per-probe sign match
    paired_diffs = t_i[:, top_m] - t_i[:, runnerup_m]
    margin_result = margin_significance_test(paired_diffs, M)

    stat_battery = {
        "representative_cell": rc,
        "identification_test": identification_result,
        "margin_test": margin_result,
        "top_candidate": top_m,
        "runnerup_candidate": runnerup_m,
        "true_judge": true_idx,
    }

    return rows, {"threshold_report": threshold_report, "stat_battery": stat_battery}


# ------------------------------------------------------------------------------------- main ---

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sections", nargs="*", default=None,
                         help="subset of {primary,marginal_a,marginal_b,marginal_c,mixture,controls} to run")
    args = parser.parse_args()
    sections = set(args.sections) if args.sections else {
        "primary", "marginal_a", "marginal_b", "marginal_c", "mixture", "controls",
    }

    cfg = load_config()
    seed = cfg["seed"]
    set_all_seeds(seed)
    rng = np.random.default_rng(seed)

    run_id = f"{cfg['output']['run_id_prefix']}_{int(time.time())}"
    t0 = time.time()
    all_rows: list[dict] = []
    timings = {}

    if "primary" in sections:
        t = time.time()
        primary_rows = run_primary_map(rng, cfg)
        timings["primary_map"] = time.time() - t
        print(f"primary_map: {len(primary_rows)} rows in {timings['primary_map']:.1f}s")
        all_rows.extend(primary_rows)
    else:
        primary_rows = []

    if "marginal_a" in sections:
        t = time.time()
        rows = run_marginal_a(rng, cfg)
        timings["marginal_a"] = time.time() - t
        print(f"marginal_a: {len(rows)} rows in {timings['marginal_a']:.1f}s")
        all_rows.extend(rows)

    if "marginal_b" in sections:
        t = time.time()
        rows = run_marginal_b(rng, cfg)
        timings["marginal_b"] = time.time() - t
        print(f"marginal_b: {len(rows)} rows in {timings['marginal_b']:.1f}s")
        all_rows.extend(rows)

    if "marginal_c" in sections:
        t = time.time()
        rows = run_marginal_c(rng, cfg)
        timings["marginal_c"] = time.time() - t
        print(f"marginal_c: {len(rows)} rows in {timings['marginal_c']:.1f}s")
        all_rows.extend(rows)

    if "mixture" in sections:
        t = time.time()
        rows = run_mixture_identifiability(rng, cfg)
        timings["mixture"] = time.time() - t
        print(f"mixture_identifiability: {len(rows)} rows in {timings['mixture']:.1f}s")
        all_rows.extend(rows)

    extras = {}
    if "controls" in sections:
        t = time.time()
        control_rows, extras = run_negative_controls_and_stat_battery(rng, cfg, primary_rows)
        timings["controls"] = time.time() - t
        print(f"controls+battery: {len(control_rows)} rows in {timings['controls']:.1f}s")
        all_rows.extend(control_rows)

    for r_ in all_rows:
        r_["run_id"] = run_id
        r_["seed"] = seed
        r_["phase"] = "phase0c"
        r_["timestamp"] = time.time()

    wall_clock = time.time() - t0
    print(f"\nTotal rows: {len(all_rows)}, wall clock: {wall_clock:.1f}s")

    results_path = REPO_ROOT / cfg["output"]["results_path"]
    append_results(all_rows, results_path)

    write_manifest(
        run_id, config={"phase0c": cfg}, results_dir=REPO_ROOT / "results", seed=seed,
        extra={"wall_clock_s": wall_clock, "timings": timings, "extras": extras, "sections_run": list(sections)},
    )

    # Acceptance check (Section 11): regression gate only, at the Phase-0-equivalent cell.
    if "primary" in sections:
        df = pd.DataFrame(primary_rows)
        gate = df[
            (df["psi_deg"] == 90) & (df["lambda_aniso"] == cfg["primary_map"]["lambda_aniso"]) &
            (df["sigma_D"] == 0.0) & (df["n_probes"] == cfg["primary_map"]["N"]) &
            (df["probe_selection"] == "random") &
            (df["rule"].isin(["sign_agreement", "bradley_terry"]))
        ]
        acc_by_rule = gate.groupby("rule")["top1_correct"].mean()
        print("\nACCEPTANCE CHECK (Section 11 regression gate):")
        print(acc_by_rule)
        passed = bool((acc_by_rule == 1.0).all())
        print(f"PASS" if passed else "FAIL")

    if extras:
        print("\n--- Threshold report (Section 9) ---")
        for rule_name, rep in extras["threshold_report"].items():
            print(f"  {rule_name}: tau={rep['tau']:.4f} F1={rep['f1']:.3f} "
                  f"TPR={rep['tpr']:.3f} FPR={rep['fpr']:.3f} "
                  f"(n_pos={rep['n_positive']}, n_neg={rep['n_negative']})")
        print("\n--- Section 4.6 stat battery (one representative cell) ---")
        sb = extras["stat_battery"]
        print(f"  identification: rho_hat={sb['identification_test']['rho_hat']:.4f} "
              f"p={sb['identification_test']['p_value']:.3e} "
              f"(chance={sb['identification_test']['chance_rate']:.3f})")
        print(f"  margin: test={sb['margin_test']['test_used']} "
              f"p_one_sided={sb['margin_test']['p_one_sided']:.3e} "
              f"p_star={sb['margin_test']['p_star_bonferroni']:.3e} "
              f"shapiro_p={sb['margin_test']['shapiro_p']:.3e}")


if __name__ == "__main__":
    main()
