"""Section B of NEXT_STEPS_ROUND_3_FINAL.md: Track E reliability-aware synthetic map. CPU only.

Two analyses, reported separately and never merged:

  E1  generic homogeneous-reliability sensitivity map. Every candidate gets the same target
      reliability R within a cell. This is theory/sensitivity, NOT a real-pool prediction.
  E2  heterogeneous diagnostic using the measured per-judge R_m vector. Also NOT a real-pool
      prediction -- the real pool's latent separation psi is not identified (the raw
      sign-agreement -> psi conversion was voided in an earlier round and is not used here).

Section 0.1 correction, which changes the math: the simulator's latent signal LOADING is
`a`, and the measured test-retest correlation is `R`. Under parallel two-template measurement
with independently resampled noise, corr(d^(1), d^(2)) = a^2, so:

    a_m = sqrt(R_m)          NOT a_m = R_m

Using R directly as the loading over-attenuates and makes the reliability axis unreadable
against real measurements. Gate F9 verifies realized test-retest correlation matches target R.

Shared noise `c` is shared across judges WITHIN one template but independently resampled for the
second template, so it contributes to same-template inter-judge covariance but not to test-retest
covariance -- which is why R = a^2 survives its introduction.

Run: .venv/Scripts/python.exe -m src.phase0e_track_e2 [--reps 200] [--analysis e1|e2|both]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.attribute.rules import RULE_NAMES, run_all_rules
from src.common.io import write_manifest
from src.common.seeds import set_all_seeds
from src.phase0c_synthetic import build_judges, sample_phi_diff_pool

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

PSI_GRID = [90, 45, 20, 10, 5, 2]
R_GRID = [1.0, 0.9, 0.7, 0.573, 0.5, 0.478, 0.3, 0.15]
C_HAT = 0.0489
C_GRID = [0.0, C_HAT, 0.25, 0.5]
SIGMA_D_GRID = [0.0, 0.5, 1.0]
LAMBDA = 4
N_PROBES = 600
D = 16
M = 5

# Measured cross-template reliabilities from the corrected fixed_bin margins (D2a, this project).
MEASURED_R = {
    "Qwen2.5-1.5B": 0.573,
    "gemma-3-1b": 0.478,
    "SmolLM2-1.7B": 0.219,
    "TinyLlama-1.1B": 0.216,
    "Llama-3.2-1B": 0.152,
}


def standardize(x: np.ndarray) -> np.ndarray:
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else x - x.mean()


def simulate_one_template(rng, d_true: np.ndarray, a_vec: np.ndarray, c: float) -> np.ndarray:
    """One measurement/template: d_obs = a*d_true + sqrt(1-a^2)*[sqrt(c)*s_i + sqrt(1-c)*e_mi].
    `s_i` is shared across judges within this template; a fresh `s` is drawn per template."""
    n, M_ = d_true.shape
    s = rng.standard_normal(n)
    e = rng.standard_normal((n, M_))
    a = a_vec[None, :]
    noise_scale = np.sqrt(np.clip(1 - a**2, 0, None))
    return a * d_true + noise_scale * (np.sqrt(c) * s[:, None] + np.sqrt(1 - c) * e)


def draw_latent(rng, psi_deg: float, n: int, d: int = D, M_: int = M, lam: float = LAMBDA) -> np.ndarray:
    w, u = build_judges(rng, d, M_, psi_deg)
    phi_diff = sample_phi_diff_pool(rng, n, d, u, lam)
    d_true = phi_diff @ w.T
    return np.apply_along_axis(standardize, 0, d_true)


def run_cell(rng, psi_deg, a_vec, c, sigma_D, reps, n=N_PROBES):
    """Fingerprint comes from the TRAINING-template observation of the true judge; attribution
    runs against the PROBE-template observations of all candidates (mismatched-template setting)."""
    rows = []
    for rep in range(reps):
        d_true = draw_latent(rng, psi_deg, n)
        d_train_obs = simulate_one_template(rng, d_true, a_vec, c)
        d_probe_obs = simulate_one_template(rng, d_true, a_vec, c)

        true_idx = int(rng.integers(len(a_vec)))
        delta_clean = d_train_obs[:, true_idx]
        sd_clean = delta_clean.std()
        eps = rng.standard_normal(n) * (sigma_D * sd_clean if sd_clean > 0 else sigma_D)
        delta_obs = delta_clean + eps

        scores = run_all_rules(delta_obs, d_probe_obs)
        for rule in RULE_NAMES:
            m_hat = int(np.argmax(scores[rule]))
            rows.append({
                "psi_deg": psi_deg, "c": c, "sigma_D": sigma_D, "rule": rule, "replicate": rep,
                "true_judge": true_idx, "candidate_judge": m_hat,
                "top1_correct": int(m_hat == true_idx), "n_probes": n,
            })
    return rows


def calibration_gate(rng, reps=30, n=N_PROBES, c=C_HAT) -> list[dict]:
    """F9 / gate: realized corr(d_obs^(1), d_obs^(2)) must match target R. This is the numerical
    check that a = sqrt(R) is the right parameterization -- if the loading were set to R itself,
    realized correlation would come back as R^2, not R."""
    out = []
    for R in R_GRID:
        a = np.full(M, np.sqrt(R))
        corrs = []
        for _ in range(reps):
            d_true = draw_latent(rng, 45, n)
            o1 = simulate_one_template(rng, d_true, a, c)
            o2 = simulate_one_template(rng, d_true, a, c)
            for m in range(M):
                corrs.append(np.corrcoef(o1[:, m], o2[:, m])[0, 1])
        realized = float(np.mean(corrs))
        se = float(np.std(corrs) / np.sqrt(len(corrs)))
        out.append({"R_target": R, "R_realized": realized, "se": se,
                    "abs_error": abs(realized - R), "within_mc_error": abs(realized - R) < max(3 * se, 0.02)})
        print(f"  R={R:.3f} -> realized {realized:.4f} (se {se:.4f})  "
              f"{'OK' if out[-1]['within_mc_error'] else 'MISMATCH'}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--analysis", choices=["e1", "e2", "both"], default="both")
    parser.add_argument("--sigma-d-extra", type=float, default=None,
                        help="Arm A tau=2 sigma_D_ratio from Section A, added to the sigma_D grid")
    args = parser.parse_args()

    set_all_seeds(0)
    rng = np.random.default_rng(20260909)
    sigma_grid = sorted(set(SIGMA_D_GRID) | ({args.sigma_d_extra} if args.sigma_d_extra else set()))
    t0 = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Calibration gate (F9): realized test-retest corr vs target R, a = sqrt(R) ===")
    calib = calibration_gate(rng)
    calib_pass = all(c["within_mc_error"] for c in calib)
    print(f"Calibration gate: {'PASS' if calib_pass else 'FAIL'}")

    print("\n=== Regression gate: psi=90, R=1, c=0, sigma_D=0 -> top-1 must be 1.0 ===")
    g = pd.DataFrame(run_cell(rng, 90, np.ones(M), 0.0, 0.0, args.reps))
    acc = g.groupby("rule")["top1_correct"].mean()
    print(acc.to_string())
    reg_pass = bool((acc == 1.0).all())
    print(f"Regression gate: {'PASS' if reg_pass else 'FAIL'}")

    gates = {"calibration": calib, "calibration_pass": calib_pass,
             "regression_gate_pass": reg_pass, "regression_acc": acc.to_dict()}
    with open(RESULTS_DIR / "track_e2_gates.json", "w") as f:
        json.dump(gates, f, indent=2)

    if args.analysis in ("e1", "both"):
        print(f"\n=== E1: generic homogeneous-reliability map "
              f"({len(PSI_GRID)}x{len(R_GRID)}x{len(C_GRID)}x{len(sigma_grid)} cells) ===")
        rows = []
        for psi_deg in PSI_GRID:
            for R in R_GRID:
                a_vec = np.full(M, np.sqrt(R))
                for c in C_GRID:
                    for sd in sigma_grid:
                        cell = run_cell(rng, psi_deg, a_vec, c, sd, args.reps)
                        for r in cell:
                            r["R"] = R
                        rows.extend(cell)
            print(f"  psi={psi_deg} done ({time.time()-t0:.0f}s)")
        df1 = pd.DataFrame(rows)
        df1.to_parquet(RESULTS_DIR / "track_e2_E1_generic_map.parquet", index=False)
        print(f"E1: {len(df1)} rows -> results/track_e2_E1_generic_map.parquet")

    if args.analysis in ("e2", "both"):
        print(f"\n=== E2: heterogeneous measured-R diagnostic (NOT a real-pool prediction) ===")
        judges = list(MEASURED_R)
        a_vec = np.sqrt(np.array([MEASURED_R[j] for j in judges]))
        print(f"  measured R: {MEASURED_R}")
        print(f"  loadings a = sqrt(R): {dict(zip(judges, np.round(a_vec, 4)))}")
        rows = []
        for psi_deg in PSI_GRID:
            for c in C_GRID:
                for sd in sigma_grid:
                    cell = run_cell(rng, psi_deg, a_vec, c, sd, args.reps)
                    for r in cell:
                        r["R"] = "heterogeneous"
                        r["true_judge_name"] = judges[r["true_judge"]]
                    rows.extend(cell)
            print(f"  psi={psi_deg} done ({time.time()-t0:.0f}s)")
        df2 = pd.DataFrame(rows)
        df2.to_parquet(RESULTS_DIR / "track_e2_E2_heterogeneous.parquet", index=False)
        print(f"E2: {len(df2)} rows -> results/track_e2_E2_heterogeneous.parquet")

    wall = time.time() - t0
    write_manifest(f"track_e2_{int(time.time())}",
                   config={"psi": PSI_GRID, "R": R_GRID, "c": C_GRID, "sigma_D": sigma_grid,
                           "measured_R": MEASURED_R, "reps": args.reps, "a_parameterization": "a = sqrt(R)"},
                   results_dir=RESULTS_DIR, seed=0,
                   extra={"wall_clock_s": wall, "gates": gates})
    print(f"\nTrack E wall clock: {wall:.0f}s")


if __name__ == "__main__":
    main()
