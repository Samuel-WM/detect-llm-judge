"""NEXT_STEPS_ROUND_5_FINAL.md 10 / F27: map validation at two levels.

Track E must NOT be validated using only the scalar minimum psi_eff and called "the real
geometry". Two predictions are produced instead:

  F27A  conservative HOMOGENEOUS REFERENCE -- the standing Track E parameterization at
        psi = min raw pairwise calibration psi_eff, R = 1 for candidates passing cross-process C1,
        N = N_final, M = final candidate count. Labelled a reference prediction, never the
        measured real operating point.

  F27B  EMPIRICAL-GEOMETRY prediction -- the ACTUAL retained candidate margin matrix on the fresh
        final holdout, standardized by the frozen calibration transforms, with synthetic aligned-
        policy signal Delta_sim_z = d_true_z + sigma_D * eps. This preserves the real anisotropic
        candidate relationships instead of collapsing them to one angle, and is the preferred
        calibration comparison.

sigma_D comes from converged FJ transmission if the 5.7 plateau rule passes; otherwise from the
final-four sensitivity range, and every prediction is reported as a band.

Both simulations assume an additive independent noise approximation. Neither proves that the DPO
dynamics follow the synthetic model, and this is stated in the figure itself.

Run: .venv/Scripts/python.exe -m src.analysis.round5_map_validation
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.analysis.figures_round5 import convergence_check
from src.attribute.rules import RULE_NAMES, run_all_rules
from src.phase0e_track_e2 import draw_latent, standardize

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = REPO_ROOT / "results"
FIG = REPO_ROOT / "figures"
REPS = 400
SEED = 20250913
PRIMARY_RULE = "nnls_mixture"


def sigma_band() -> dict:
    """5.7: a single sigma point may be used only if the FJ plateau rule passes."""
    p = RESULTS / "round5_FJ_trajectory.json"
    if not p.exists():
        raise FileNotFoundError(str(p))
    rows = json.load(open(p))["rows"]
    conv = convergence_check(rows)
    last4 = [r["sigma_D_ratio"] for r in rows[-4:]]
    if conv["converged"]:
        return {"converged": True, "point": rows[-1]["sigma_D_ratio"],
                "band": [rows[-1]["sigma_D_ratio"]], "detail": conv}
    return {"converged": False, "point": None,
            "band": [float(np.min(last4)), float(np.median(last4)), float(np.max(last4))],
            "detail": conv,
            "note": "FJ magnitude transmission is NOT demonstrably converged; a single measured "
                    "sigma_D point is not placed on Track E. A sensitivity band over the "
                    "final-four checkpoint range is used instead."}


def f27a_homogeneous(psi_deg: float, n: int, M: int, sigmas: list[float]) -> dict:
    """Standing Track E parameterization. R = 1 for every retained candidate (all pass
    cross-process C1), so a = sqrt(R) = 1, the measurement-noise term vanishes and the shared-noise
    parameter c drops out of the model entirely."""
    rng = np.random.default_rng(SEED)
    out = {}
    for sd_ratio in sigmas:
        correct = {r: 0 for r in RULE_NAMES}
        for _ in range(REPS):
            d_true = draw_latent(rng, psi_deg, n, M_=M)      # (n, M), standardized columns
            true_idx = int(rng.integers(M))
            clean = d_true[:, true_idx]
            delta = clean + rng.standard_normal(n) * sd_ratio * clean.std()
            scores = run_all_rules(delta, d_true)
            for r in RULE_NAMES:
                correct[r] += int(np.argmax(scores[r]) == true_idx)
        out[sd_ratio] = {r: correct[r] / REPS for r in RULE_NAMES}
        out[sd_ratio]["se_primary"] = float(
            np.sqrt(out[sd_ratio][PRIMARY_RULE] * (1 - out[sd_ratio][PRIMARY_RULE]) / REPS))
    return out


def f27b_empirical(D: np.ndarray, S: list[str], sigmas: list[float]) -> dict:
    """The real candidate geometry, held fixed; only the aligned-policy signal is simulated."""
    rng = np.random.default_rng(SEED + 1)
    n, M = D.shape
    Dz = np.apply_along_axis(standardize, 0, D)
    out = {}
    for sd_ratio in sigmas:
        per = {m: {r: 0 for r in RULE_NAMES} for m in range(M)}
        for _ in range(REPS):
            for m in range(M):
                clean = Dz[:, m]
                delta = clean + rng.standard_normal(n) * sd_ratio * clean.std()
                scores = run_all_rules(delta, Dz)
                for r in RULE_NAMES:
                    per[m][r] += int(np.argmax(scores[r]) == m)
        block = {}
        for m in range(M):
            block[S[m]] = {r: per[m][r] / REPS for r in RULE_NAMES}
        acc = float(np.mean([block[S[m]][PRIMARY_RULE] for m in range(M)]))
        block["pooled_primary"] = acc
        block["se_pooled_primary"] = float(np.sqrt(acc * (1 - acc) / (REPS * M)))
        out[sd_ratio] = block
    return out


def observed() -> dict:
    fd = json.load(open(RESULTS / "final_design_matrix.json"))
    obs = {}
    for m in fd["S"]:
        p = RESULTS / f"round5_RM_{m}_attribution.json"
        if p.exists():
            a = json.load(open(p))
            obs[m] = {"top1_correct": a["final_raw"]["nnls_top1_correct"],
                      "S_obs": a["final_raw"]["S_obs"],
                      "p_perm": a["final_raw"]["permutation"]["p_perm"],
                      "boot_top1": a["final_raw"]["bootstrap"]["top1_support"],
                      "primary_success": a["final_raw"]["primary_success"]}
    obs["observed_top1_fraction"] = (
        float(np.mean([v["top1_correct"] for k, v in obs.items() if isinstance(v, dict)]))
        if any(isinstance(v, dict) for v in obs.values()) else None)
    return obs


def main() -> None:
    fd = json.load(open(RESULTS / "final_design_matrix.json"))
    cal = json.load(open(RESULTS / "rm_calibration_r5.json"))
    S, M = fd["S"], len(fd["S"])
    D = np.array(fd["D_std"])
    n = D.shape[0]
    psi = cal["min_raw_psi_deg"]

    sb = sigma_band()
    sigmas = sb["band"]
    print(f"sigma_D source: {'converged FJ point' if sb['converged'] else 'FJ final-four sensitivity band'}"
          f" -> {[round(s, 4) for s in sigmas]}")
    print(f"F27A homogeneous reference: psi={psi:.2f} deg, R=1, N={n}, M={M}, reps={REPS}")
    a = f27a_homogeneous(psi, n, M, sigmas)
    print(f"F27B empirical geometry: real {n}x{M} final-holdout margin matrix, reps={REPS} per teacher")
    b = f27b_empirical(D, S, sigmas)
    obs = observed()

    out = {"psi_used_deg": psi, "N": n, "M": M, "reps": REPS, "sigma_band": sb,
           "F27A_homogeneous": {str(k): v for k, v in a.items()},
           "F27B_empirical": {str(k): v for k, v in b.items()},
           "observed": obs,
           "caveat": "both simulations assume an additive independent noise approximation and "
                     "neither proves the DPO dynamics follow the synthetic model"}
    with open(RESULTS / "round5_map_validation.json", "w") as f:
        json.dump(out, f, indent=1)

    for s in sigmas:
        print(f"  sigma_D={s:.4f}  F27A top-1 (primary) = {a[s][PRIMARY_RULE]:.3f}  "
              f"F27B pooled top-1 = {b[s]['pooled_primary']:.3f}")
    if obs.get("observed_top1_fraction") is not None:
        print(f"  OBSERVED real-RM top-1 fraction = {obs['observed_top1_fraction']:.3f} "
              f"({sum(1 for k, v in obs.items() if isinstance(v, dict) and v['top1_correct'])}"
              f"/{sum(1 for v in obs.values() if isinstance(v, dict))} arms)")

    plot(out, S, sigmas, a, b, obs, psi, n, M, sb)
    print(f"wrote {RESULTS / 'round5_map_validation.json'}")


def plot(out, S, sigmas, a, b, obs, psi, n, M, sb) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6))
    x = np.arange(len(sigmas))

    ax = axes[0]
    ax.errorbar(x, [a[s][PRIMARY_RULE] for s in sigmas],
                yerr=[a[s]["se_primary"] for s in sigmas], fmt="-o", lw=2, capsize=4,
                color="tab:blue", label="F27A homogeneous reference (nnls_mixture)")
    ax.errorbar(x, [b[s]["pooled_primary"] for s in sigmas],
                yerr=[b[s]["se_pooled_primary"] for s in sigmas], fmt="-s", lw=2, capsize=4,
                color="tab:purple", label="F27B empirical geometry, pooled (nnls_mixture)")
    if obs.get("observed_top1_fraction") is not None:
        ax.axhline(obs["observed_top1_fraction"], color="crimson", lw=2.4,
                   label=f"OBSERVED real-RM top-1 fraction = {obs['observed_top1_fraction']:.3f}")
    ax.axhline(1.0 / M, color="k", ls=":", lw=1.2, label=f"chance = 1/M = {1 / M:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s:.3f}" for s in sigmas])
    ax.set_xlabel("sigma_D_ratio" + ("" if sb["converged"] else "  (FJ sensitivity band, NON-CONVERGED)"))
    ax.set_ylabel("predicted / observed top-1")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"predicted vs observed\npsi = {psi:.2f} deg, R = 1, N = {n}, M = {M}", fontsize=10)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)

    ax = axes[1]
    w = 0.8 / max(len(sigmas), 1)
    xs = np.arange(len(S))
    for k, s in enumerate(sigmas):
        ax.bar(xs + (k - (len(sigmas) - 1) / 2) * w, [b[s][m]["nnls_mixture"] for m in S], w,
               label=f"F27B predicted, sigma_D={s:.3f}")
    if any(isinstance(v, dict) for v in obs.values()):
        ax.scatter(xs, [1.0 if obs[m]["top1_correct"] else 0.0 for m in S if m in obs],
                   marker="*", s=320, color="crimson", zorder=5,
                   label="OBSERVED top-1 (1 = correct)")
    ax.axhline(1.0 / M, color="k", ls=":", lw=1.2, label=f"chance = {1 / M:.3f}")
    ax.set_xticks(xs)
    ax.set_xticklabels(S, rotation=12)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("predicted top-1 probability")
    ax.set_title("per-teacher empirical-geometry prediction vs observed", fontsize=10)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("F27: map validation at two levels. F27A is a homogeneous REFERENCE prediction, "
                 "not the measured real operating point.\nBoth assume additive independent noise "
                 "and neither proves the DPO dynamics follow the synthetic model.", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIG / "F27_map_validation.png", dpi=150)
    plt.close(fig)
    print("wrote F27_map_validation.png")


if __name__ == "__main__":
    main()
