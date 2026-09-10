"""Phase 0c figures (Section 12 of PHASE_0C_SPEC.md). Reads only from
results/phase0c_synthetic.parquet; never recomputes anything.

Run: .venv/Scripts/python.exe -m src.analysis.figures_phase0c
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_PATH = REPO_ROOT / "results" / "phase0c_synthetic.parquet"
FIGURES_DIR = REPO_ROOT / "figures"

REFERENCE_PSI_AGREEMENT = {90: 0.750, 45: 0.875, 20: 0.889, 10: 0.944, 5: 0.972, 2: 0.989}


def fig1_heatmap(df: pd.DataFrame) -> None:
    prim = df[df["section"] == "primary_map"]
    piv = prim.pivot_table(index="psi_deg", columns="sigma_D", values="top1_correct", aggfunc="mean")
    piv = piv.sort_index(ascending=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(piv.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([f"{c:g}" for c in piv.columns])
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([f"{r:g}°" for r in piv.index])
    ax.set_xlabel("sigma (= sigma_D = sigma_d)")
    ax.set_ylabel("psi (judge pairwise separation)")
    ax.set_title("Phase 0c primary map: pooled top-1 accuracy\n(sign_agreement + bradley_terry + nnls_mixture, mean)")

    cs = ax.contour(
        range(len(piv.columns)), range(len(piv.index)), piv.values, levels=[0.60],
        colors="red", linewidths=2,
    )
    ax.clabel(cs, fmt={0.60: "0.60"})

    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center",
                     color="white" if piv.values[i, j] < 0.5 else "black", fontsize=8)

    fig.colorbar(im, ax=ax, label="pooled top-1 accuracy")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "phase0c_fig1_heatmap_psi_sigma.png", dpi=150)
    plt.close(fig)


def fig1b_heatmap_per_rule(df: pd.DataFrame) -> None:
    """NEXT_STEPS_TRACKS_A_B_C.md A1: the pooled map's 0.60 contour is not a decision boundary
    for any single rule (nnls_mixture clears 0.60 at psi=20/sigma=0.5 where the pooled map says
    0.54). Redraw as 3 per-rule panels plus the pooled panel for continuity.
    """
    prim = df[df["section"] == "primary_map"]
    rules = ["sign_agreement", "bradley_terry", "nnls_mixture"]
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))

    def _draw(ax, piv, title):
        piv = piv.sort_index(ascending=False)
        im = ax.imshow(piv.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels([f"{c:g}" for c in piv.columns])
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels([f"{r:g}°" for r in piv.index])
        ax.set_xlabel("sigma")
        ax.set_title(title)
        try:
            cs = ax.contour(range(len(piv.columns)), range(len(piv.index)), piv.values,
                             levels=[0.60], colors="red", linewidths=2)
            ax.clabel(cs, fmt={0.60: "0.60"})
        except ValueError:
            pass
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                ax.text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center",
                        color="white" if piv.values[i, j] < 0.5 else "black", fontsize=7)
        return im

    for ax, rule in zip(axes[:3], rules):
        piv = prim[prim["rule"] == rule].pivot_table(index="psi_deg", columns="sigma_D", values="top1_correct", aggfunc="mean")
        im = _draw(ax, piv, rule)
    axes[0].set_ylabel("psi")

    piv_pooled = prim.pivot_table(index="psi_deg", columns="sigma_D", values="top1_correct", aggfunc="mean")
    im = _draw(axes[3], piv_pooled, "pooled (for continuity)")

    fig.colorbar(im, ax=axes, label="top-1 accuracy", fraction=0.02, pad=0.02)
    fig.suptitle("Phase 0c primary map, per rule (0.60 contour is rule-specific)")
    fig.savefig(FIGURES_DIR / "phase0c_fig1b_heatmap_per_rule.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig2_n_vs_accuracy(df: pd.DataFrame) -> None:
    ma = df[df["section"] == "marginal_a_N_select"]
    piv = ma.pivot_table(index="n_probes", columns="probe_selection", values="top1_correct", aggfunc="mean")

    fig, ax = plt.subplots(figsize=(6, 5))
    for select in ["random", "disagreement_max"]:
        ax.plot(piv.index, piv[select], marker="o", label=select)
    ax.set_xscale("log")
    ax.set_xlabel("N (probe count)")
    ax.set_ylabel("pooled top-1 accuracy")
    ax.set_title("Marginal A: probe count and selection strategy\n(psi=20deg, sigma=0.5, lambda=4, pooled over rules)")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "phase0c_fig2_N_vs_accuracy_by_selection.png", dpi=150)
    plt.close(fig)


def fig3_agreement_scatter(df: pd.DataFrame) -> None:
    cells = df[df["section"].isin([
        "primary_map", "marginal_a_N_select", "marginal_b_anisotropy",
        "marginal_c_decoupled_D", "marginal_c_decoupled_d",
    ])]
    group_cols = ["section", "psi_deg", "sigma_D", "sigma_d", "lambda_aniso", "n_probes", "probe_selection", "rule"]
    g = cells.groupby(group_cols).agg(
        top1=("top1_correct", "mean"), agreement=("empirical_agreement", "mean"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"sign_agreement": "tab:blue", "bradley_terry": "tab:orange", "nnls_mixture": "tab:green"}
    for rule, sub in g.groupby("rule"):
        ax.scatter(sub["agreement"], sub["top1"], s=14, alpha=0.5, label=rule, color=colors.get(rule))

    for psi_deg, ref_agreement in REFERENCE_PSI_AGREEMENT.items():
        ax.axvline(ref_agreement, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)
        ax.text(ref_agreement, 1.02, f"{psi_deg}°", ha="center", va="bottom", fontsize=7, color="gray")

    ax.set_xlabel("empirical pairwise judge agreement rate (this probe set)")
    ax.set_ylabel("pooled top-1 accuracy")
    ax.set_title("Top-1 accuracy vs measured judge agreement, all cells\n(gray lines: analytic agreement(psi) reference)")
    ax.set_ylim(-0.02, 1.08)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "phase0c_fig3_accuracy_vs_agreement.png", dpi=150)
    plt.close(fig)


def fig4_theta_vs_kappa(df: pd.DataFrame) -> None:
    all_rows = df[df["section"] != "mixture_identifiability"].dropna(subset=["kappa_D", "angular_error_rad"])
    # one row per (rule) per replicate carries the same theta/kappa_D; dedupe to one per replicate
    dedup = all_rows.drop_duplicates(subset=[
        "section", "psi_deg", "sigma_D", "sigma_d", "lambda_aniso", "n_probes", "probe_selection", "replicate",
    ])
    kappa = dedup["kappa_D"].to_numpy()
    theta = dedup["angular_error_rad"].to_numpy()
    mask = (kappa > 0) & np.isfinite(kappa) & (theta > 0) & np.isfinite(theta)
    kappa, theta = kappa[mask], theta[mask]

    log_kappa = np.log10(kappa)
    bins = np.linspace(log_kappa.min(), log_kappa.max(), 25)
    bin_idx = np.digitize(log_kappa, bins)
    bin_centers, bin_medians = [], []
    for b in range(1, len(bins)):
        sel = bin_idx == b
        if sel.sum() >= 5:
            bin_centers.append(np.median(log_kappa[sel]))
            bin_medians.append(np.median(theta[sel]))
    bin_centers, bin_medians = np.array(bin_centers), np.array(bin_medians)
    log_bin_medians = np.log10(bin_medians)

    slope, intercept = np.polyfit(bin_centers, log_bin_medians, 1)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(log_kappa, np.log10(theta), s=4, alpha=0.08, color="steelblue", label="replicates")
    ax.plot(bin_centers, log_bin_medians, "o-", color="black", label="binned median")
    fit_x = np.array([bin_centers.min(), bin_centers.max()])
    ax.plot(fit_x, slope * fit_x + intercept, "r--", label=f"fit slope = {slope:.2f}")
    ax.set_xlabel("log10 kappa_D (condition number)")
    ax.set_ylabel("log10 theta (angular error, rad)")
    ax.set_title("Angular error vs probe-design conditioning (log-log)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "phase0c_fig4_theta_vs_kappa.png", dpi=150)
    plt.close(fig)
    return slope


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(RESULTS_PATH)

    fig1_heatmap(df)
    print("wrote phase0c_fig1_heatmap_psi_sigma.png")
    fig1b_heatmap_per_rule(df)
    print("wrote phase0c_fig1b_heatmap_per_rule.png")
    fig2_n_vs_accuracy(df)
    print("wrote phase0c_fig2_N_vs_accuracy_by_selection.png")
    fig3_agreement_scatter(df)
    print("wrote phase0c_fig3_accuracy_vs_agreement.png")
    slope = fig4_theta_vs_kappa(df)
    print(f"wrote phase0c_fig4_theta_vs_kappa.png (fitted slope = {slope:.3f})")


if __name__ == "__main__":
    main()
