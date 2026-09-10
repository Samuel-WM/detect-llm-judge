"""Round 5 figures. Each function reads stored results only and is generated only when the
corresponding data exists (10, "Generate figures only when the corresponding data exist").

Run: .venv/Scripts/python.exe -m src.analysis.figures_round5 [--only F26]
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = REPO_ROOT / "results"
FIG = REPO_ROOT / "figures"


def f26_candidate_geometry() -> None:
    """F26: raw vs length-residualized calibration psi_eff on the retained pool."""
    cal = json.load(open(RESULTS / "rm_calibration_r5.json"))
    S = cal["S"]
    panels = [("raw", np.array(cal["corr_raw"]), np.array(cal["psi_raw"]), cal["min_raw_psi_deg"],
               cal["min_raw_pair"]),
              ("length-residualized", np.array(cal["corr_resid"]), np.array(cal["psi_resid"]),
               cal["min_resid_psi_deg"], cal["min_resid_pair"])]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))
    for ax, (label, C, P, mn, pair) in zip(axes, panels):
        im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(S)))
        ax.set_xticklabels(S, rotation=15, ha="right", fontsize=9)
        ax.set_yticks(range(len(S)))
        ax.set_yticklabels(S, fontsize=9)
        for i in range(len(S)):
            for j in range(len(S)):
                ax.text(j, i, f"r={C[i, j]:.3f}\npsi={P[i, j]:.1f}deg", ha="center", va="center",
                        fontsize=9, color="white" if abs(C[i, j]) > 0.55 else "black")
        ax.set_title(f"{label}\nmin pairwise psi_eff = {mn:.2f} deg ({pair})", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(
        f"F26: retained-pool candidate geometry, {cal['n_calibration_probes']} calibration probes. "
        f"C5 gate = min RAW psi_eff >= 45 deg -> {'PASS' if cal['C5_pass'] else 'FAIL'}. "
        "Residualized panel is a diagnostic, not a gate.", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "F26_candidate_geometry.png", dpi=150)
    plt.close(fig)
    print("wrote F26_candidate_geometry.png")


def _arms():
    fd = json.load(open(RESULTS / "final_design_matrix.json"))
    return fd["S"], [f"RM_{m}" for m in fd["S"]]


def _load(arm, kind):
    p = RESULTS / f"round5_{arm}_{kind}.json"
    return json.load(open(p)) if p.exists() else None


def f17_attribution_trajectory() -> None:
    """F17: score-gap trajectories per true teacher. One trajectory per teacher, so this is NOT
    a replicate-level accuracy estimate and is deliberately not labelled 'accuracy'."""
    S, arms = _arms()
    have = [(a, _load(a, "attribution")) for a in arms]
    have = [(a, d) for a, d in have if d]
    if not have:
        raise FileNotFoundError(str(RESULTS / "round5_RM_*_attribution.json"))
    fig, axes = plt.subplots(1, len(have), figsize=(5.6 * len(have), 4.8), squeeze=False)
    for ax, (arm, d) in zip(axes[0], have):
        steps = [c["step"] for c in d["per_checkpoint"]]
        s_obs = [c["raw"]["S_obs"] for c in d["per_checkpoint"]]
        bt = [c["raw"]["bt_gap"] for c in d["per_checkpoint"]]
        sg = [c["raw"]["sign_gap"] for c in d["per_checkpoint"]]
        ok = [c["raw"]["nnls_top1_correct"] for c in d["per_checkpoint"]]
        ax.plot(steps, s_obs, "-o", lw=2.2, ms=5, color="tab:blue",
                label="PRIMARY: NNLS true-vs-runner-up gap S")
        ax.plot(steps, bt, "--s", lw=1.3, ms=4, color="tab:orange", label="BT score gap")
        ax.plot(steps, sg, "--^", lw=1.3, ms=4, color="tab:green", label="sign-agreement gap")
        for st, s, o in zip(steps, s_obs, ok):
            ax.scatter([st], [s], s=110, facecolors="none",
                       edgecolors="tab:green" if o else "tab:red", lw=1.8, zorder=5)
        ax.axhline(0, color="k", lw=0.8)
        ax.axvline(steps[-1], color="gray", ls=":", lw=2)
        ax.annotate("final pre-registered\ncheckpoint", (steps[-1], ax.get_ylim()[1]),
                    ha="right", va="top", fontsize=7, color="gray")
        ax.set_title(f"true teacher: {arm.split("_", 1)[1]}\n(circle: green = top-1 correct, red = incorrect)",
                     fontsize=9)
        ax.set_xlabel("optimizer step")
        ax.set_ylabel("score gap")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle("F17: attribution trajectories, one run per teacher (not a replicate-level "
                 "accuracy estimate)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "F17_attribution_trajectory.png", dpi=150)
    plt.close(fig)
    print("wrote F17_attribution_trajectory.png")


def f18_inferential_stability() -> None:
    """F18: permutation null and bootstrap distribution at the final checkpoint. Replaces the
    old binomial-per-probe headline (6.4)."""
    S, arms = _arms()
    have = [(a, _load(a, "attribution")) for a in arms]
    have = [(a, d) for a, d in have if d]
    if not have:
        raise FileNotFoundError(str(RESULTS / "round5_RM_*_attribution.json"))
    fig, axes = plt.subplots(2, len(have), figsize=(5.4 * len(have), 7.4), squeeze=False)
    for j, (arm, d) in enumerate(have):
        b = d["final_raw"]
        ax = axes[0][j]
        ax.hist(b["permutation"]["s_perm"], bins=60, color="lightsteelblue",
                edgecolor="none", label="permutation null")
        ax.axvline(b["S_obs"], color="crimson", lw=2.4, label=f"S_obs = {b['S_obs']:+.4f}")
        ax.set_title(f"{arm.split("_", 1)[1]}: permutation test\np_perm = {b['permutation']['p_perm']:.5f} "
                     f"({b['permutation']['n_perm']:,} perms)", fontsize=9)
        ax.set_xlabel("S under permuted Delta")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        ax = axes[1][j]
        ax.hist(b["bootstrap"]["s_boot"], bins=60, color="darkseagreen", edgecolor="none")
        lo, hi = b["bootstrap"]["s_boot_ci95"]
        ax.axvline(lo, color="k", ls="--", lw=1.2)
        ax.axvline(hi, color="k", ls="--", lw=1.2, label=f"95% CI [{lo:+.3f}, {hi:+.3f}]")
        ax.axvline(b["bootstrap"]["s_boot_median"], color="darkgreen", lw=2,
                   label=f"median {b['bootstrap']['s_boot_median']:+.3f}")
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(f"{arm.split("_", 1)[1]}: bootstrap stability\ntop-1 support = "
                     f"{b['bootstrap']['top1_support']:.4f} ({b['bootstrap']['n_boot']:,} resamples)",
                     fontsize=9)
        ax.set_xlabel("S under bootstrap resampling")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle("F18: inferential stability at the final checkpoint. S is computed on "
                 "UNNORMALIZED NNLS coefficients (see the pre-registration deviation note).",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "F18_inferential_stability.png", dpi=150)
    plt.close(fig)
    print("wrote F18_inferential_stability.png")


def f19_candidate_profiles() -> None:
    """F19: normalized NNLS weights, BT scores and sign-agreement scores at the final checkpoint."""
    S, arms = _arms()
    have = [(a, _load(a, "attribution")) for a in arms]
    have = [(a, d) for a, d in have if d]
    if not have:
        raise FileNotFoundError(str(RESULTS / "round5_RM_*_attribution.json"))
    fig, axes = plt.subplots(3, len(have), figsize=(4.6 * len(have), 9.2), squeeze=False)
    rows = [("normalized NNLS mixture weight", "alpha"),
            ("Bradley-Terry score", "bt_scores"),
            ("sign-agreement score", "sign_scores")]
    x = np.arange(len(S))
    for j, (arm, d) in enumerate(have):
        b = d["final_raw"]
        true_i = d["true_idx"]
        for i, (label, key) in enumerate(rows):
            ax = axes[i][j]
            v = np.array(b[key])
            order = np.argsort(v)[::-1]
            runner = int(order[1]) if order[0] == true_i else int(order[0])
            cols = ["tab:blue"] * len(S)
            cols[true_i] = "tab:green"
            cols[runner] = "tab:orange"
            ax.bar(x, v, color=cols)
            ax.set_xticks(x)
            ax.set_xticklabels(S, rotation=15, ha="right", fontsize=8)
            ax.set_ylabel(label, fontsize=8)
            if i == 0:
                ax.set_title(f"true teacher: {arm.split("_", 1)[1]}\ngreen = true, orange = runner-up "
                             f"({S[runner]})", fontsize=9)
            ax.grid(alpha=0.3, axis="y")
    fig.suptitle("F19: candidate score profiles at the final checkpoint", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "F19_candidate_profiles.png", dpi=150)
    plt.close(fig)
    print("wrote F19_candidate_profiles.png")


def _traj_arms():
    S, arms = _arms()
    out = []
    for a in arms + ["FJ", "CTRL"]:
        d = _load(a, "trajectory")
        if d:
            out.append((a, d))
    if not out:
        raise FileNotFoundError(str(RESULTS / "round5_*_trajectory.json"))
    return out


def f20_optimization_vs_generalization() -> None:
    """F20: fitting the training labels versus transmitting the teacher's ordering held out."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for arm, d in _traj_arms():
        steps = [r["step"] for r in d["rows"]]
        acc = [r.get("train_acc", float("nan")) for r in d["rows"]]
        sty = {"ls": "--", "color": "gray"} if arm == "CTRL" else {}
        axes[0].plot(steps, acc, "-o", ms=3, label=arm, **sty)
        if "latent_sign_agreement" in d["rows"][-1]:
            axes[1].plot(steps, [r["latent_sign_agreement"] for r in d["rows"]], "-o", ms=3,
                         label=arm, **sty)
    axes[0].axhline(0.5, color="k", ls=":", lw=1)
    axes[0].axhline(0.60, color="tab:red", ls="--", lw=1,
                    label="0.60 standing low-fit threshold (12.5)")
    axes[0].set_xlabel("optimizer step")
    axes[0].set_ylabel("train_acc (DPO rewards/accuracies)")
    axes[0].set_title("optimization: fitting the training labels")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)
    axes[1].axhline(0.5, color="k", ls=":", lw=1, label="chance")
    axes[1].set_xlabel("optimizer step")
    axes[1].set_ylabel("held-out latent sign agreement")
    axes[1].set_title("generalization: transmitting the teacher's ordering")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)
    fig.suptitle("F20: optimization versus generalization (CTRL shown dashed for context)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "F20_optimization_vs_generalization.png", dpi=150)
    plt.close(fig)
    print("wrote F20_optimization_vs_generalization.png")


def f21_judge_signal_vs_length() -> None:
    """F21: standardized gamma_1 (teacher margin) and gamma_2 (length) with SE bands."""
    have = [(a, d) for a, d in _traj_arms() if "gamma_1" in d["rows"][-1]]
    if not have:
        raise FileNotFoundError("no arm has gamma_1")
    fig, axes = plt.subplots(1, len(have), figsize=(5.2 * len(have), 4.6), squeeze=False)
    for ax, (arm, d) in zip(axes[0], have):
        steps = np.array([r["step"] for r in d["rows"]])
        for key, se_key, col, lab in [("gamma_1", "se_gamma_1", "tab:blue", "gamma_1 (d_true)"),
                                      ("gamma_2", "se_gamma_2", "tab:red", "gamma_2 (len_diff)")]:
            g = np.array([r[key] for r in d["rows"]])
            se = np.array([r[se_key] for r in d["rows"]])
            ax.plot(steps, g, "-o", ms=3.5, color=col, label=lab)
            ax.fill_between(steps, g - 1.96 * se, g + 1.96 * se, color=col, alpha=0.2)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(f"{arm}", fontsize=9)
        ax.set_xlabel("optimizer step")
        ax.set_ylabel("standardized coefficient")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle("F21: Delta = c + gamma_1*d_true + gamma_2*len_diff, both predictors "
                 "standardized; bands are +/-1.96 SE", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "F21_judge_signal_vs_length.png", dpi=150)
    plt.close(fig)
    print("wrote F21_judge_signal_vs_length.png")


TRACK_E_SIGMA_GRID = (0.0, 1.0)


def convergence_check(rows: list[dict]) -> dict:
    """5.7: a final checkpoint is a CONVERGED magnitude-transmission measurement only if all four
    conditions hold over the final four checkpoints."""
    last = rows[-4:]
    pr = np.array([r["pearson"] for r in last])
    r2 = np.array([r["r_squared"] for r in last])
    c = {"range_pearson": float(pr.max() - pr.min()), "range_r_squared": float(r2.max() - r2.min()),
         "abs_final_minus_mean_pearson": float(abs(pr[-1] - pr.mean())),
         "abs_final_minus_mean_r_squared": float(abs(r2[-1] - r2.mean()))}
    c["converged"] = bool(c["range_pearson"] <= 0.03 and c["range_r_squared"] <= 0.02
                          and c["abs_final_minus_mean_pearson"] <= 0.015
                          and c["abs_final_minus_mean_r_squared"] <= 0.010)
    c["sigma_D_ratio_last4"] = [r["sigma_D_ratio"] for r in last]
    return c


def f22_sigma_trajectory() -> None:
    """F22: sigma_D_ratio(t). The struck pilot value 5.35 is not plotted or referenced."""
    have = [(a, d) for a, d in _traj_arms() if "sigma_D_ratio" in d["rows"][-1]]
    if not have:
        raise FileNotFoundError("no arm has sigma_D_ratio")
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    ax.axhspan(max(TRACK_E_SIGMA_GRID[0], 1e-3), TRACK_E_SIGMA_GRID[1], color="gold", alpha=0.18,
               label="Track E simulated sigma_D grid (0.0-1.0)")
    for arm, d in have:
        rows = d["rows"]
        conv = convergence_check(rows)
        ax.plot([r["step"] for r in rows], [r["sigma_D_ratio"] for r in rows], "-o", ms=4,
                label=f"{arm} ({'CONVERGED' if conv['converged'] else 'NON-CONVERGED'})")
        ax.annotate(f"{rows[-1]['sigma_D_ratio']:.2f}"
                    + ("" if conv["converged"] else "\nnon-converged\ncheckpoint value"),
                    (rows[-1]["step"], rows[-1]["sigma_D_ratio"]), fontsize=7,
                    textcoords="offset points", xytext=(6, 0))
    ax.set_yscale("log")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("sigma_D_ratio = sqrt((1 - R^2) / R^2)")
    ax.set_title("F22: magnitude-transmission trajectory.\nA final value is called converged only "
                 "if the 5.7 four-part plateau rule passes.", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIG / "F22_sigma_trajectory.png", dpi=150)
    plt.close(fig)
    print("wrote F22_sigma_trajectory.png")


def f23_detection_statistic() -> None:
    """F23: Lambda against step for every run including CTRL, with the pre-registered threshold."""
    S, arms = _arms()
    have = [(a, _load(a, "attribution")) for a in arms + ["CTRL", "FJ"]]
    have = [(a, d) for a, d in have if d]
    if not have:
        raise FileNotFoundError(str(RESULTS / "round5_*_attribution.json"))
    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    thr = have[0][1]["lambda_threshold"]
    for arm, d in have:
        steps = [c["step"] for c in d["per_checkpoint"]]
        lam = [max(c["raw"]["Lambda"]) for c in d["per_checkpoint"]]
        sty = {"ls": "--", "color": "gray", "lw": 2.4} if arm == "CTRL" else {}
        ax.plot(steps, lam, "-o", ms=4, label=f"{arm} (max over candidates)", **sty)
    ax.axhline(thr, color="crimson", ls="-.", lw=2,
               label=f"pre-registered detection threshold = {thr:.2f}")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("Lambda (max over candidates)")
    ax.set_yscale("symlog")
    ax.set_title("F23: detection statistic. Negative-control claim is only that no candidate "
                 "clears the threshold in CTRL.", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIG / "F23_detection_statistic.png", dpi=150)
    plt.close(fig)
    print("wrote F23_detection_statistic.png")


def f24_raw_vs_residualized() -> None:
    """F24: the direct length-confound robustness figure."""
    S, arms = _arms()
    have = [(a, _load(a, "attribution")) for a in arms]
    have = [(a, d) for a, d in have if d]
    if not have:
        raise FileNotFoundError(str(RESULTS / "round5_RM_*_attribution.json"))
    labels = [a[3:] for a, _ in have]
    x = np.arange(len(have))
    w = 0.38
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    for k, (key, title, fmt) in enumerate([
            ("S_obs", "S_obs (unnormalized NNLS)", "{:+.3f}"),
            ("p_perm", "permutation p-value", "{:.4f}"),
            ("top1_support", "bootstrap top-1 support", "{:.3f}")]):
        ax = axes[k]
        for off, tag, col in [(-w / 2, "final_raw", "tab:blue"), (w / 2, "final_resid", "tab:purple")]:
            vals = []
            for _, d in have:
                b = d[tag]
                vals.append(b["S_obs"] if key == "S_obs"
                            else b["permutation"]["p_perm"] if key == "p_perm"
                            else b["bootstrap"]["top1_support"])
            bars = ax.bar(x + off, vals, w, color=col,
                          label="raw (PRIMARY)" if tag == "final_raw" else "length-residualized")
            for b_, v in zip(bars, vals):
                ax.annotate(fmt.format(v), (b_.get_x() + b_.get_width() / 2, v), ha="center",
                            va="bottom", fontsize=7)
        if key == "p_perm":
            ax.axhline(0.001, color="crimson", ls="--", lw=1.4, label="p = 0.001")
            ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=10)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("F24: raw versus length-residualized final attribution. Raw is primary; "
                 "residualized success cannot rescue raw failure (12.3).", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "F24_raw_vs_residualized.png", dpi=150)
    plt.close(fig)
    print("wrote F24_raw_vs_residualized.png")


def f25_deterministic_arm() -> None:
    """F25: generated only if the 9 trigger fired. Compares the deterministic-teacher arm with its
    corresponding stochastic arm. This is a realism bridge; it does not alter the stochastic
    verdict."""
    det = [d for d in RESULTS.glob("round5_DET_*_attribution.json")]
    if not det:
        raise FileNotFoundError(str(RESULTS / "round5_DET_*_attribution.json"))
    arm_det = det[0].name[len("round5_"):-len("_attribution.json")]
    teacher = arm_det.split("_", 1)[1]
    arm_sto = f"RM_{teacher}"
    a_det, a_sto = _load(arm_det, "attribution"), _load(arm_sto, "attribution")
    t_det, t_sto = _load(arm_det, "trajectory"), _load(arm_sto, "trajectory")

    panels = [
        ("attribution score gap S", lambda a, t: a["final_raw"]["S_obs"], "{:+.4f}"),
        ("permutation p", lambda a, t: a["final_raw"]["permutation"]["p_perm"], "{:.5f}"),
        ("bootstrap top-1 support", lambda a, t: a["final_raw"]["bootstrap"]["top1_support"], "{:.4f}"),
        ("final train_acc", lambda a, t: t["rows"][-1].get("train_acc", float("nan")), "{:.4f}"),
        ("Pearson transmission", lambda a, t: t["rows"][-1].get("pearson", float("nan")), "{:+.4f}"),
        ("latent sign transmission", lambda a, t: t["rows"][-1].get("latent_sign_agreement", float("nan")), "{:.4f}"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8))
    for ax, (title, fn, fmt) in zip(axes.ravel(), panels):
        vals = [fn(a_sto, t_sto), fn(a_det, t_det)]
        bars = ax.bar(["stochastic BT\n(tau=2)", "deterministic\n1[d>0]"], vals,
                      color=["tab:blue", "tab:orange"])
        for b_, v in zip(bars, vals):
            ax.annotate(fmt.format(v), (b_.get_x() + b_.get_width() / 2, v), ha="center",
                        va="bottom", fontsize=9)
        if title == "permutation p":
            ax.axhline(0.001, color="crimson", ls="--", lw=1.4, label="p = 0.001")
            ax.set_yscale("log")
            ax.legend(fontsize=7)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle(f"F25: deterministic realism arm, teacher = {teacher}. "
                 "A realism bridge -- it does not alter the stochastic-stage verdict.", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "F25_deterministic_arm.png", dpi=150)
    plt.close(fig)
    print("wrote F25_deterministic_arm.png")


def f28_summary_panel() -> None:
    """F28: generated only after all triggered work is complete."""
    summ = RESULTS / "round5_summary.json"
    if not summ.exists():
        raise FileNotFoundError(str(summ))
    sm = json.load(open(summ))
    cal = json.load(open(RESULTS / "rm_calibration_r5.json"))
    fd = json.load(open(RESULTS / "final_design_matrix.json"))
    S = sm["S"]
    mv = RESULTS / "round5_map_validation.json"
    mvd = json.load(open(mv)) if mv.exists() else None
    fj = _load("FJ", "trajectory")

    fig = plt.figure(figsize=(15.5, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.28)

    ax = fig.add_subplot(gs[0, 0])
    ax.axis("off")
    lines = [f"FINAL RETAINED POOL (M = {sm['M']}, chance = {1 / sm['M']:.3f})"]
    lines += [f"   - {m}" for m in S]
    lines += ["",
              f"min psi_eff raw          {cal['min_raw_psi_deg']:.2f} deg  (calibration)",
              f"min psi_eff residualized {cal['min_resid_psi_deg']:.2f} deg  (calibration)",
              f"min psi_eff raw          {fd['min_psi_raw_final']:.2f} deg  (final holdout)",
              "",
              f"kill criterion   required {sm['required_successes']}, "
              f"observed {sm['successes']}",
              f"                 {'FIRED' if sm['kill_criterion_fired'] else 'not fired'}",
              f"aggregate claim  {'HOLDS' if sm['aggregate_claim_holds'] else 'does not hold'}"]
    if sm.get("ctrl"):
        lines += ["", f"CTRL max Lambda  {sm['ctrl']['max_Lambda']:.2f} vs {sm['ctrl']['threshold']:.2f}",
                  f"negative control {'FAILS' if sm['ctrl']['clears'] else 'PASSES'}"]
    if fj:
        c = convergence_check(fj["rows"])
        lines += ["", f"FJ sigma_D       {'CONVERGED' if c['converged'] else 'NON-CONVERGED'}",
                  f"                 final {fj['rows'][-1]['sigma_D_ratio']:.3f}"]
    lines += ["", f"deterministic arm {'triggered' if sm.get('deterministic_arm_triggered') else 'not triggered'}"]
    ax.text(0, 1, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=8.4)
    ax.set_title("Round 5 verdicts", fontsize=11, loc="left")

    ax = fig.add_subplot(gs[0, 1])
    rows = sm["rows"]
    x = np.arange(len(rows))
    ax.bar(x - 0.2, [r["raw S_obs"] for r in rows], 0.4, color="tab:blue", label="raw (PRIMARY)")
    ax.bar(x + 0.2, [r["resid S_obs"] for r in rows], 0.4, color="tab:purple", label="residualized")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([r["true teacher"] for r in rows], rotation=12, fontsize=8)
    ax.set_ylabel("S_obs")
    ax.set_title("final attribution statistic", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")

    ax = fig.add_subplot(gs[0, 2])
    ax.bar(x - 0.2, [float(r["raw p_perm"]) for r in rows], 0.4, color="tab:blue", label="raw")
    ax.bar(x + 0.2, [float(r["resid p_perm"]) for r in rows], 0.4, color="tab:purple",
           label="residualized")
    ax.axhline(0.001, color="crimson", ls="--", lw=1.5, label="p = 0.001")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([r["true teacher"] for r in rows], rotation=12, fontsize=8)
    ax.set_ylabel("permutation p")
    ax.set_title("inferential outcome", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")

    ax = fig.add_subplot(gs[1, 0])
    ax.bar(x - 0.2, [r["raw boot top-1"] for r in rows], 0.4, color="tab:blue", label="raw")
    ax.bar(x + 0.2, [r["resid boot top-1"] for r in rows], 0.4, color="tab:purple",
           label="residualized")
    ax.axhline(1 / sm["M"], color="k", ls=":", lw=1.2, label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels([r["true teacher"] for r in rows], rotation=12, fontsize=8)
    ax.set_ylabel("bootstrap top-1 support")
    ax.set_title("stability", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")

    ax = fig.add_subplot(gs[1, 1])
    for arm, d in _traj_arms():
        sty = {"ls": "--", "color": "gray"} if arm == "CTRL" else {}
        ax.plot([r["step"] for r in d["rows"]],
                [r.get("train_acc", float("nan")) for r in d["rows"]], "-o", ms=3, label=arm, **sty)
    ax.axhline(0.60, color="tab:red", ls="--", lw=1, label="0.60 low-fit threshold")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("train_acc")
    ax.set_title("optimization", fontsize=10)
    ax.legend(fontsize=6.5)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 2])
    if mvd:
        sig = mvd["sigma_band"]["band"]
        xs = np.arange(len(sig))
        ax.plot(xs, [mvd["F27A_homogeneous"][str(s)]["nnls_mixture"] for s in sig], "-o",
                color="tab:blue", label="F27A homogeneous ref")
        ax.plot(xs, [mvd["F27B_empirical"][str(s)]["pooled_primary"] for s in sig], "-s",
                color="tab:purple", label="F27B empirical geometry")
        if mvd["observed"].get("observed_top1_fraction") is not None:
            ax.axhline(mvd["observed"]["observed_top1_fraction"], color="crimson", lw=2,
                       label="observed top-1 fraction")
        ax.axhline(1 / sm["M"], color="k", ls=":", lw=1.2, label="chance")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{s:.3f}" for s in sig], fontsize=8)
        ax.set_xlabel("sigma_D_ratio" + ("" if mvd["sigma_band"]["converged"] else " (band)"))
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=6.5)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "map validation not run", ha="center", va="center", fontsize=10)
    ax.set_title("observed vs predicted", fontsize=10)
    ax.grid(alpha=0.3)

    fig.suptitle("F28: Round 5 final summary", fontsize=12)
    fig.savefig(FIG / "F28_summary_panel.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote F28_summary_panel.png")


FIGURES = {"F17": f17_attribution_trajectory,
           "F18": f18_inferential_stability,
           "F19": f19_candidate_profiles,
           "F20": f20_optimization_vs_generalization,
           "F21": f21_judge_signal_vs_length,
           "F22": f22_sigma_trajectory,
           "F23": f23_detection_statistic,
           "F24": f24_raw_vs_residualized,
           "F25": f25_deterministic_arm,
           "F26": f26_candidate_geometry,
           "F28": f28_summary_panel}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    for name, fn in FIGURES.items():
        if args.only and name not in args.only:
            continue
        try:
            fn()
        except FileNotFoundError as e:
            print(f"{name}: skipped, missing input ({e.filename})")


if __name__ == "__main__":
    main()
