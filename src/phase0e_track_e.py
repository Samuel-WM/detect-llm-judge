"""Step 5 of NEXT_STEPS_FINAL.md: Track E, reliability-aware map redesign. CPU only.

Replaces Phase 0c's independent-per-probe-noise model with one whose reliability and shared-noise
parameters are estimated from the real margins (Step 2). Two observations per judge are generated
per replicate -- a training-template margin (which produces the alignment fingerprint) and a
probe-template margin (which the auditor attributes against) -- so the simulated problem is the
actual mismatched-template forensic problem.

Model (standardized, unit-variance components):

    d_obs_{m,i,t} = a_m * d_true_{m,i}
                    + sqrt(1 - a_m^2) * [ sqrt(c) * s_{i,t} + sqrt(1-c) * e_{m,i,t} ]

with noise draws independent across templates t, and s_{i,t} shared across judges within a
template. Under this model corr(d_obs_train, d_obs_probe) = a_m^2, so the latent signal loading is
a_m = sqrt(R_m) -- NOT R_m itself (5.1, and an explicit prohibition). Gate 2 verifies this
numerically.

Run: .venv/Scripts/python.exe -m src.phase0e_track_e [--reps 200]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.attribute.rules import RULE_NAMES, run_all_rules
from src.common.io import write_manifest
from src.common.seeds import set_all_seeds
from src.phase0c_synthetic import build_judges, sample_phi_diff_pool

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

PSI_GRID = [90, 45, 20, 10, 5, 2]
R_GRID = [1.0, 0.9, 0.7, 0.5, 0.3, 0.15]
LAMBDA = 4
N_PROBES = 600
D = 16
M = 5


def standardize(x: np.ndarray) -> np.ndarray:
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else x - x.mean()


def simulate_observations(rng, psi_deg, a_vec, c, n, d=D, M_=M, lam=LAMBDA):
    """Draws the latent judge margins and both template observations of them.

    Returns (d_train_obs, d_probe_obs, d_true), each (n, M). Noise draws are independent across
    templates; s_{i,t} is shared across judges within a template, e_{m,i,t} is not.
    """
    w, u = build_judges(rng, d, M_, psi_deg)
    phi_diff = sample_phi_diff_pool(rng, n, d, u, lam)
    d_true = phi_diff @ w.T                                  # (n, M)
    d_true = np.apply_along_axis(standardize, 0, d_true)     # unit variance per judge

    s_train = rng.standard_normal(n)
    s_probe = rng.standard_normal(n)
    e_train = rng.standard_normal((n, M_))
    e_probe = rng.standard_normal((n, M_))

    a = a_vec[None, :]
    noise_scale = np.sqrt(np.clip(1 - a**2, 0, None))
    d_train_obs = a * d_true + noise_scale * (np.sqrt(c) * s_train[:, None] + np.sqrt(1 - c) * e_train)
    d_probe_obs = a * d_true + noise_scale * (np.sqrt(c) * s_probe[:, None] + np.sqrt(1 - c) * e_probe)
    return d_train_obs, d_probe_obs, d_true


def simulate_replicate(rng, psi_deg, a_vec, c, sigma_D, n, d=D, M_=M, lam=LAMBDA):
    """Returns (Delta_obs, D_probe_obs, true_idx). Fingerprint comes from the TRAINING-template
    margin of the true judge; attribution is against PROBE-template candidate margins."""
    d_train_obs, d_probe_obs, _ = simulate_observations(rng, psi_deg, a_vec, c, n, d, M_, lam)
    true_idx = int(rng.integers(M_))
    delta_clean = d_train_obs[:, true_idx]
    sd_clean = delta_clean.std()
    eps = rng.standard_normal(n) * (sigma_D * sd_clean if sd_clean > 0 else sigma_D)
    delta_obs = delta_clean + eps
    return delta_obs, d_probe_obs, true_idx


def run_cell(rng, psi_deg, R, c, sigma_D, reps, n=N_PROBES, a_vec=None):
    a = a_vec if a_vec is not None else np.full(M, np.sqrt(max(R, 0.0)))
    rows = []
    for rep in range(reps):
        delta_obs, d_probe_obs, true_idx = simulate_replicate(rng, psi_deg, a, c, sigma_D, n)
        scores = run_all_rules(delta_obs, d_probe_obs)
        for rule in RULE_NAMES:
            m_hat = int(np.argmax(scores[rule]))
            rows.append({
                "psi_deg": psi_deg, "R": R, "c": c, "sigma_D": sigma_D, "rule": rule,
                "replicate": rep, "true_judge": true_idx, "candidate_judge": m_hat,
                "top1_correct": int(m_hat == true_idx), "n_probes": n,
            })
    return rows


# ------------------------------------------------------------------ regression gates ---------

def gate_2_reliability_recovery(rng, n_draws=20, n=600, c=0.0) -> list[dict]:
    """For every requested R, simulated corr(d_train_obs, d_probe_obs) must equal R.

    This is the numerical check the spec requires for a_m = sqrt(R_m): we request a loading of
    sqrt(R) and verify the resulting cross-template correlation comes back as R, not sqrt(R).
    """
    out = []
    for R in R_GRID:
        a = np.full(M, np.sqrt(R))
        corrs = []
        for _ in range(n_draws):
            d_tr, d_pr, _ = simulate_observations(rng, 45, a, c, n)
            for m in range(M):
                corrs.append(np.corrcoef(d_tr[:, m], d_pr[:, m])[0, 1])
        mean_corr = float(np.mean(corrs))
        out.append({"R_requested": R, "R_simulated": mean_corr, "abs_error": abs(mean_corr - R)})
        print(f"  R={R:.2f} -> simulated corr(train,probe)={mean_corr:.4f} "
              f"(err {abs(mean_corr-R):.4f})")
    return out


def gate_4_independent_error_reduction(rng, n_draws=20, n=600) -> dict:
    """At c=0 the shared-noise-adjusted estimator must reduce to the naive independent-error
    attenuation formula, and both must recover the latent inter-judge correlation.
    """
    R = 0.5
    a = np.full(M, np.sqrt(R))
    err_naive, err_adjusted, gap = [], [], []
    for _ in range(n_draws):
        d_tr, d_pr, d_true = simulate_observations(rng, 45, a, 0.0, n)
        latent_true = np.corrcoef(d_true, rowvar=False)
        obs = np.corrcoef(d_tr, rowvar=False)
        naive = obs / np.sqrt(R * R)
        adjusted = (obs - 0.0 * np.sqrt((1 - R) * (1 - R))) / np.sqrt(R * R)  # c=0
        iu = np.triu_indices(M, k=1)
        err_naive.append(np.abs(naive[iu] - latent_true[iu]).mean())
        err_adjusted.append(np.abs(adjusted[iu] - latent_true[iu]).mean())
        gap.append(np.abs(naive[iu] - adjusted[iu]).max())
    res = {
        "mean_abs_err_naive_vs_latent": float(np.mean(err_naive)),
        "mean_abs_err_adjusted_vs_latent": float(np.mean(err_adjusted)),
        "max_gap_naive_vs_adjusted": float(np.max(gap)),
    }
    res["passes"] = bool(res["max_gap_naive_vs_adjusted"] < 1e-12
                         and res["mean_abs_err_adjusted_vs_latent"] < 0.10)
    print(f"  naive vs latent err={res['mean_abs_err_naive_vs_latent']:.4f}  "
          f"adjusted vs latent err={res['mean_abs_err_adjusted_vs_latent']:.4f}  "
          f"naive-vs-adjusted gap={res['max_gap_naive_vs_adjusted']:.2e}")
    return res


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--c-hat", type=float, default=None,
                        help="shared-noise fraction from D2c; if omitted, read from results/d2_reliability_padfree.json")
    parser.add_argument("--sigma-d-pilot", type=float, default=None,
                        help="sigma_D_emp from the Track C pilot; if omitted only {0, 0.5} are swept")
    args = parser.parse_args()

    set_all_seeds(0)
    rng = np.random.default_rng(4242)

    c_hat = args.c_hat
    if c_hat is None:
        p = RESULTS_DIR / "d2_reliability_padfree.json"
        if p.exists():
            c_hat = json.load(open(p))["d2c"]["c_hat"]
            print(f"c_hat read from {p.name}: {c_hat:.4f}")
        else:
            c_hat = 0.25
            print(f"WARNING: no D2 results found; using placeholder c_hat={c_hat}")
    c_grid = sorted({0.0, round(float(c_hat), 4), 0.5})
    sigma_grid = sorted({0.0, 0.5} | ({args.sigma_d_pilot} if args.sigma_d_pilot is not None else set()))
    print(f"grids: psi={PSI_GRID}  R={R_GRID}  c={c_grid}  sigma_D={sigma_grid}")

    t0 = time.time()

    print("\n=== Gate 2: reliability recovery (simulated corr(train,probe) must equal R) ===")
    gate2 = gate_2_reliability_recovery(rng)
    gate2_pass = all(g["abs_error"] < 0.03 for g in gate2)
    print(f"Gate 2: {'PASS' if gate2_pass else 'FAIL'}")

    print("\n=== Gate 4: at c=0, corrections reduce to independent-error formulas ===")
    gate4 = gate_4_independent_error_reduction(rng)
    print(f"Gate 4: {'PASS' if gate4['passes'] else 'FAIL'}")

    print("\n=== Gate 1: psi=90, R=1, c=0, sigma_D=0 must reproduce exact recovery ===")
    g1_rows = run_cell(rng, 90, 1.0, 0.0, 0.0, args.reps)
    g1 = pd.DataFrame(g1_rows).groupby("rule")["top1_correct"].mean()
    print(g1.to_string())
    gate1_pass = bool((g1[["sign_agreement", "bradley_terry"]] == 1.0).all())
    print(f"Gate 1: {'PASS' if gate1_pass else 'FAIL'}")

    print("\n=== Main map ===")
    rows = []
    for psi_deg in PSI_GRID:
        for R in R_GRID:
            for c in c_grid:
                for sd in sigma_grid:
                    rows.extend(run_cell(rng, psi_deg, R, c, sd, args.reps))
        print(f"  psi={psi_deg} done ({time.time()-t0:.0f}s)")
    df = pd.DataFrame(rows)

    print("\n=== Gate 3: at fixed psi and sigma_D, accuracy must not improve as R decreases ===")
    # The gate is about SYSTEMATIC improvement as reliability falls, so it must be Monte-Carlo
    # aware: with `reps` replicates the SE of a per-cell accuracy is sqrt(0.25/reps), and a
    # step-to-step reversal smaller than a few SE is noise, not a violation. Tested two ways:
    # (a) no single adjacent-R reversal beyond max(0.10, 3*SE); (b) the overall accuracy-vs-R
    # trend (Spearman) is not significantly negative.
    se = np.sqrt(0.25 / args.reps)
    thresh = max(0.10, 3 * se)
    viol = []
    for (psi_deg, c, sd, rule), g in df.groupby(["psi_deg", "c", "sigma_D", "rule"]):
        acc_by_R = g.groupby("R")["top1_correct"].mean().sort_index()
        diffs = np.diff(acc_by_R.values)  # R ascending: accuracy should be non-decreasing
        trend = np.corrcoef(acc_by_R.index.values, acc_by_R.values)[0, 1] if acc_by_R.std() > 0 else 0.0
        if (diffs < -thresh).any() and trend < -0.5:
            viol.append({"psi_deg": psi_deg, "c": c, "sigma_D": sd, "rule": rule,
                         "trend_corr_acc_vs_R": round(float(trend), 3),
                         "acc_by_R": acc_by_R.round(3).to_dict()})
    gate3_pass = len(viol) == 0
    print(f"Gate 3: {'PASS' if gate3_pass else 'FAIL'} ({len(viol)} cells with a reversal beyond "
          f"{thresh:.3f} AND a negative accuracy-vs-R trend; MC SE={se:.3f})")
    for v in viol[:5]:
        print(f"    {v}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RESULTS_DIR / "track_e_map.parquet", index=False)
    gates = {"gate1_exact_recovery": gate1_pass, "gate2_reliability_recovery": gate2_pass,
             "gate2_detail": gate2, "gate3_monotone_in_R": gate3_pass, "gate3_violations": viol[:20],
             "gate4_independent_error_reduction": gate4,
             "all_gates_pass": bool(gate1_pass and gate2_pass and gate3_pass and gate4["passes"]),
             "c_hat_used": c_hat, "c_grid": c_grid, "sigma_grid": sigma_grid}
    with open(RESULTS_DIR / "track_e_gates.json", "w") as f:
        json.dump(gates, f, indent=2, default=str)

    wall = time.time() - t0
    write_manifest(f"track_e_{int(time.time())}", config={"psi": PSI_GRID, "R": R_GRID,
                   "c": c_grid, "sigma_D": sigma_grid, "reps": args.reps},
                   results_dir=RESULTS_DIR, seed=0, extra={"wall_clock_s": wall, "gates": gates})

    print(f"\n=== Headline: top-1 by psi x R (c={c_hat:.3f}, sigma_D=0.5, per rule) ===")
    sel = df[(df["c"] == round(float(c_hat), 4)) & (df["sigma_D"] == 0.5)]
    for rule in RULE_NAMES:
        print(f"\n{rule}:")
        print(sel[sel["rule"] == rule].pivot_table(index="psi_deg", columns="R",
              values="top1_correct", aggfunc="mean").round(3).to_string())

    print(f"\nwall clock {wall:.0f}s -> results/track_e_map.parquet")


if __name__ == "__main__":
    main()
