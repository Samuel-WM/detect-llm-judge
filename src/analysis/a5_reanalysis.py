"""A5 reanalysis, per NEXT_STEPS_FINAL.md Section 1 correction 3.

The original A5 fit reported a log-log slope of 0.363 across a regime that visibly reaches the
angular-error ceiling (theta cannot exceed pi/2 for a cosine-based angular error between two
directions, once w_hat is effectively uncorrelated with w_true). Fitting an unbounded power law
through saturated points biases the exponent downward.

Refits without selecting on the dependent variable:
  1. median theta and kappa_D per CONSTRUCTION LEVEL (delta), not per row;
  2. a capped power law theta = min(A * kappa^b, pi/2) and a smooth saturating equivalent
     theta = (pi/2) * tanh(A * kappa^b / (pi/2));
  3. ordinary log-log slopes over PRE-SATURATION construction levels only, cutting by design
     level (delta), never by observed theta;
  4. saturation onset and the range of slopes across reasonable cutoffs.

Run: .venv/Scripts/python.exe -m src.analysis.a5_reanalysis
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "results"
CEILING = np.pi / 2

# Fraction of the ceiling above which a construction level is called saturated.
SATURATION_FRAC = 0.90


def capped_power_law(kappa, A, b):
    return np.minimum(A * np.power(kappa, b), CEILING)


def smooth_saturating(kappa, A, b):
    return CEILING * np.tanh(A * np.power(kappa, b) / CEILING)


def main() -> None:
    df = pd.read_parquet(RESULTS_DIR / "phase0c_track_a_conditioning.parquet")
    # sigma=0 baseline excluded from every fit: it is a separate error mechanism (machine-
    # precision recovery), as established in the original A5 note.
    sub = df[df["delta"] != "baseline_sigma0"].copy()
    sub["delta_f"] = sub["delta"].astype(float)

    levels = (
        sub.groupby("delta_f")
        .agg(
            kappa_median=("kappa_D", "median"),
            kappa_q1=("kappa_D", lambda x: x.quantile(0.25)),
            kappa_q3=("kappa_D", lambda x: x.quantile(0.75)),
            theta_median=("theta", "median"),
            theta_q1=("theta", lambda x: x.quantile(0.25)),
            theta_q3=("theta", lambda x: x.quantile(0.75)),
            n=("theta", "count"),
        )
        .sort_values("kappa_median")
    )
    levels["frac_of_ceiling"] = levels["theta_median"] / CEILING
    levels["saturated"] = levels["frac_of_ceiling"] >= SATURATION_FRAC

    print("=== Per-construction-level medians (IQR), sigma=0.5 only ===")
    print(levels.round(4).to_string())
    print(f"\nceiling pi/2 = {CEILING:.4f}; saturation flag at >= {SATURATION_FRAC:.0%} of ceiling")

    unsat = levels[~levels["saturated"]]
    sat = levels[levels["saturated"]]
    if len(sat):
        onset_kappa = float(sat["kappa_median"].min())
        onset_delta = float(sat.index.min()) if len(sat) else None
        print(f"\nSaturation onset: first saturated construction level is delta={sat.index[0]:g} "
              f"at kappa_D median {onset_kappa:.1f} "
              f"({sat['frac_of_ceiling'].iloc[0]:.1%} of ceiling)")
    else:
        onset_kappa, onset_delta = None, None
        print("\nNo construction level reached the saturation threshold.")

    # ---- capped / smooth saturating fits on all replicates (no row-level selection) ----
    kappa_all = sub["kappa_D"].to_numpy()
    theta_all = sub["theta"].to_numpy()
    fits = {}
    for name, fn in (("capped_min", capped_power_law), ("smooth_tanh", smooth_saturating)):
        try:
            popt, _ = curve_fit(fn, kappa_all, theta_all, p0=[0.05, 0.4], maxfev=20000)
            resid = theta_all - fn(kappa_all, *popt)
            fits[name] = {
                "A": float(popt[0]), "b": float(popt[1]),
                "rmse": float(np.sqrt(np.mean(resid**2))),
            }
            print(f"\n{name} fit on all {len(kappa_all)} replicates: "
                  f"A={popt[0]:.4f}  b={popt[1]:.4f}  rmse={fits[name]['rmse']:.4f}")
        except RuntimeError as e:
            fits[name] = {"error": str(e)}
            print(f"\n{name} fit failed: {e}")

    # same fits on the per-level medians (design points, equal weight per level)
    k_med = levels["kappa_median"].to_numpy()
    t_med = levels["theta_median"].to_numpy()
    for name, fn in (("capped_min", capped_power_law), ("smooth_tanh", smooth_saturating)):
        try:
            popt, _ = curve_fit(fn, k_med, t_med, p0=[0.05, 0.4], maxfev=20000)
            fits[f"{name}_on_level_medians"] = {"A": float(popt[0]), "b": float(popt[1])}
            print(f"{name} fit on {len(k_med)} level medians: A={popt[0]:.4f}  b={popt[1]:.4f}")
        except RuntimeError as e:
            fits[f"{name}_on_level_medians"] = {"error": str(e)}

    # ---- sensitivity: ordinary log-log slope over pre-saturation LEVELS only ----
    print("\n=== Sensitivity: ordinary log-log slope, cutting by construction level (never by theta) ===")
    slopes = {}
    ordered = levels.sort_values("kappa_median")
    for n_levels in range(2, len(ordered) + 1):
        used = ordered.iloc[:n_levels]
        slope, intercept = np.polyfit(np.log10(used["kappa_median"]), np.log10(used["theta_median"]), 1)
        label = f"lowest_{n_levels}_levels (deltas {', '.join(f'{d:g}' for d in used.index)})"
        slopes[label] = float(slope)
        flag = " [includes saturated level(s)]" if used["saturated"].any() else ""
        print(f"  {label}: slope={slope:.3f}{flag}")

    presat_slopes = [v for k, v in slopes.items()
                     if not ordered.iloc[:int(k.split('_')[1])]["saturated"].any()]
    if presat_slopes:
        print(f"\nPre-saturation slope range: {min(presat_slopes):.3f} to {max(presat_slopes):.3f}")
    else:
        print("\nNo cutoff excludes all saturated levels.")

    out = {
        "ceiling": CEILING,
        "saturation_frac_threshold": SATURATION_FRAC,
        "levels": json.loads(levels.reset_index().to_json(orient="records")),
        "saturation_onset_kappa_median": onset_kappa,
        "saturation_onset_delta": onset_delta,
        "saturating_fits": fits,
        "loglog_slopes_by_cutoff": slopes,
        "presaturation_slope_range": [min(presat_slopes), max(presat_slopes)] if presat_slopes else None,
        "original_reported_slope": 0.363,
    }
    out_path = RESULTS_DIR / "a5_reanalysis.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
