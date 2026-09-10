"""Phase 0: synthetic identifiability check. CPU only, no language models.

Interpretation notes (flagged for review, not specified numerically in Section 5):
  - beta is fixed to 1.0 for the exponential tilt. Attribution is scale-free (Section 0), so
    this choice does not affect any reported accuracy or angular error; it only sets the scale
    of `w_hat` recovered by OLS, which cosine-similarity comparison to `w_true` removes anyway.
  - "empirical distribution from N draws" is implemented as: draw N samples from the analytic
    categorical policy pi_m(y|x) over the n=20 candidate responses per prompt, take Laplace-
    smoothed relative frequencies as a plug-in estimate of pi_m(y|x), and use that in place of
    the exact analytic log-probability when forming Delta. This is a finite-sample-estimation
    robustness/convergence diagnostic for the estimator itself; it does not simulate DPO
    training (the real pipeline reads exact log-probs via a forward pass, so this noise source
    does not literally arise there). Candidate margins d_m stay exact throughout, matching how
    a real auditor reads judge logits exactly at attribution time.

Run: .venv/Scripts/python.exe -m src.phase0_synthetic
"""
import argparse
import itertools
import time
from pathlib import Path

import numpy as np
import yaml

from src.attribute.rules import (
    RULE_NAMES, angular_error, recover_weights_ols, run_all_rules,
)
from src.common.io import append_results, write_manifest
from src.common.seeds import set_all_seeds

BETA = 1.0
REPO_ROOT = Path(__file__).resolve().parent.parent


def load_configs() -> tuple[dict, dict]:
    with open(REPO_ROOT / "configs" / "base.yaml") as f:
        base_cfg = yaml.safe_load(f)
    with open(REPO_ROOT / "configs" / "phase0.yaml") as f:
        phase_cfg = yaml.safe_load(f)
    return base_cfg, phase_cfg


def sample_unit_sphere(rng: np.random.Generator, n: int, d: int) -> np.ndarray:
    v = rng.normal(size=(n, d))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def build_instance(
    rng: np.random.Generator, n_prompts: int, n_responses: int, feature_dim: int, n_judges: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Draws M judge weight vectors w (M, d) and, per prompt, n_responses feature vectors phi
    (n_prompts, n_responses, d), all uniform on the unit sphere.
    """
    w = sample_unit_sphere(rng, n_judges, feature_dim)
    phi = sample_unit_sphere(rng, n_prompts * n_responses, feature_dim).reshape(
        n_prompts, n_responses, feature_dim
    )
    return w, phi


def pair_indices(n_responses: int) -> np.ndarray:
    return np.array(list(itertools.combinations(range(n_responses), 2)))


def compute_rewards(w: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """r[m, p, k] = <w_m, phi_p_k>. Returns shape (M, n_prompts, n_responses)."""
    return np.einsum("md,pkd->mpk", w, phi)


def pairwise_diffs(
    values: np.ndarray, pairs: np.ndarray, is_feature: bool,
) -> np.ndarray:
    """Vectorized y-vs-y' differences over all (prompt, pair) combinations.

    values: (M, n_prompts, n_responses) for rewards, or (n_prompts, n_responses) for a scalar
        per-response quantity (e.g. s_hat), or (n_prompts, n_responses, d) for features.
    Returns: (n_prompts * n_pairs, M) for rewards, (n_prompts * n_pairs,) for scalars, or
        (n_prompts * n_pairs, d) for features.
    """
    k1, k2 = pairs[:, 0], pairs[:, 1]
    if is_feature:
        diff = values[:, k1, :] - values[:, k2, :]  # (n_prompts, P, d)
        return diff.reshape(-1, diff.shape[-1])
    if values.ndim == 3:  # rewards: (M, n_prompts, n_responses)
        diff = values[:, :, k1] - values[:, :, k2]  # (M, n_prompts, P)
        return diff.transpose(1, 2, 0).reshape(-1, diff.shape[0])
    diff = values[:, k1] - values[:, k2]  # (n_prompts, P)
    return diff.reshape(-1)


def run_analytic_case(rng: np.random.Generator, phase_cfg: dict) -> list[dict]:
    n_prompts = phase_cfg["n_prompts_analytic"]
    n_responses = phase_cfg["n_responses"]
    d = phase_cfg["feature_dim"]
    M = phase_cfg["n_judges"]
    pairs = pair_indices(n_responses)

    w, phi = build_instance(rng, n_prompts, n_responses, d, M)
    r = compute_rewards(w, phi)  # (M, n_prompts, n_responses)

    D = pairwise_diffs(r, pairs, is_feature=False)  # (n_prompts*P, M)
    phi_diff = pairwise_diffs(phi, pairs, is_feature=True)  # (n_prompts*P, d)

    rows = []
    for true_idx in range(M):
        delta = D[:, true_idx] / BETA  # exact identity: pi_ref cancels, exponential tilt exact
        scores_by_rule = run_all_rules(delta, D)

        w_hat = recover_weights_ols(delta, phi_diff)
        theta = angular_error(w_hat, w[true_idx])

        for rule_name in RULE_NAMES:
            scores = scores_by_rule[rule_name]
            m_hat = int(np.argmax(scores))
            rows.append({
                "phase": "phase0_analytic",
                "true_judge": true_idx,
                "candidate_judge": m_hat,
                "rule": rule_name,
                "n_probes": len(delta),
                "top1_correct": int(m_hat == true_idx),
                "score": float(scores[m_hat]),
                "angular_error_rad": float(theta),
            })
    return rows


def run_empirical_case(rng: np.random.Generator, phase_cfg: dict) -> list[dict]:
    n_responses = phase_cfg["n_responses"]
    d = phase_cfg["feature_dim"]
    M = phase_cfg["n_judges"]
    N_values = phase_cfg["empirical"]["N_values"]
    n_replicates = phase_cfg["empirical"]["n_replicates"]
    n_prompts = phase_cfg["empirical"]["n_prompts"]
    pairs = pair_indices(n_responses)
    alpha_smooth = 1.0

    rows = []
    for N in N_values:
        t_start = time.time()
        for rep in range(n_replicates):
            w, phi = build_instance(rng, n_prompts, n_responses, d, M)
            r = compute_rewards(w, phi)  # (M, n_prompts, n_responses)
            D = pairwise_diffs(r, pairs, is_feature=False)
            phi_diff = pairwise_diffs(phi, pairs, is_feature=True)

            true_idx = rep % M
            logits = r[true_idx] / BETA  # (n_prompts, n_responses)
            logits = logits - logits.max(axis=1, keepdims=True)
            probs = np.exp(logits)
            probs /= probs.sum(axis=1, keepdims=True)

            counts = np.stack([rng.multinomial(N, probs[p]) for p in range(n_prompts)])
            freq = (counts + alpha_smooth) / (N + n_responses * alpha_smooth)
            s_hat = np.log(freq) - np.log(1.0 / n_responses)  # (n_prompts, n_responses)

            delta = pairwise_diffs(s_hat, pairs, is_feature=False)
            scores_by_rule = run_all_rules(delta, D)

            w_hat = recover_weights_ols(delta, phi_diff)
            theta = angular_error(w_hat, w[true_idx])

            for rule_name in RULE_NAMES:
                scores = scores_by_rule[rule_name]
                m_hat = int(np.argmax(scores))
                rows.append({
                    "phase": "phase0_empirical",
                    "N": N,
                    "replicate": rep,
                    "true_judge": true_idx,
                    "candidate_judge": m_hat,
                    "rule": rule_name,
                    "n_probes": len(delta),
                    "top1_correct": int(m_hat == true_idx),
                    "score": float(scores[m_hat]),
                    "angular_error_rad": float(theta),
                })
        print(f"  N={N:>9,d}  {n_replicates} replicates  {time.time() - t_start:6.1f}s")
    return rows


def summarize(rows: list[dict], group_cols: list[str]) -> None:
    import pandas as pd
    df = pd.DataFrame(rows)
    g = df.groupby(group_cols)
    summary = g.agg(
        top1_median=("top1_correct", "median"),
        top1_mean=("top1_correct", "mean"),
        theta_median=("angular_error_rad", "median"),
        theta_q1=("angular_error_rad", lambda x: x.quantile(0.25)),
        theta_q3=("angular_error_rad", lambda x: x.quantile(0.75)),
        n=("top1_correct", "count"),
    )
    print(summary.to_string())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-empirical", action="store_true")
    args = parser.parse_args()

    base_cfg, phase_cfg = load_configs()
    seed = phase_cfg.get("seed", base_cfg["seed"])
    set_all_seeds(seed)
    rng = np.random.default_rng(seed)

    run_id = f"{phase_cfg['output']['run_id_prefix']}_{int(time.time())}"
    t0 = time.time()

    print("Phase 0a: analytic exact-recovery case")
    analytic_rows = run_analytic_case(rng, phase_cfg)
    for r_ in analytic_rows:
        r_["run_id"] = run_id
        r_["seed"] = seed
        r_["wall_clock_s"] = None
        r_["timestamp"] = time.time()
    print(f"  {len(analytic_rows)} rows")
    print()
    print("--- Phase 0a summary (rule) ---")
    summarize(analytic_rows, ["rule"])
    print()
    print("--- Phase 0a summary (rule, true_judge) ---")
    summarize(analytic_rows, ["rule", "true_judge"])

    results_path = REPO_ROOT / phase_cfg["output"]["results_path"]
    append_results(analytic_rows, results_path)

    empirical_rows: list[dict] = []
    if not args.skip_empirical:
        print()
        print("Phase 0b: empirical finite-N estimation case")
        empirical_rows = run_empirical_case(rng, phase_cfg)
        for r_ in empirical_rows:
            r_["run_id"] = run_id
            r_["seed"] = seed
            r_["wall_clock_s"] = None
            r_["timestamp"] = time.time()
        print(f"  {len(empirical_rows)} rows")
        print()
        print("--- Phase 0b summary (rule, N) ---")
        summarize(empirical_rows, ["rule", "N"])
        append_results(empirical_rows, results_path)

    wall_clock = time.time() - t0
    write_manifest(
        run_id, config={"base": base_cfg, "phase0": phase_cfg}, results_dir=REPO_ROOT / "results",
        seed=seed, extra={"wall_clock_s": wall_clock, "beta": BETA},
    )

    print()
    print(f"Total wall clock: {wall_clock:.1f}s")

    # Acceptance check: 100% top-1 in the analytic case, angular error at numerical precision.
    analytic_top1 = np.mean([r_["top1_correct"] for r_ in analytic_rows])
    analytic_theta_max = np.max([r_["angular_error_rad"] for r_ in analytic_rows])
    passed = analytic_top1 == 1.0 and analytic_theta_max < 1e-6
    print()
    print(f"ACCEPTANCE CHECK: {'PASS' if passed else 'FAIL'}")
    print(f"  analytic top-1 accuracy: {analytic_top1:.6f} (need 1.0)")
    print(f"  analytic max angular error: {analytic_theta_max:.3e} rad (need < 1e-6)")


if __name__ == "__main__":
    main()
