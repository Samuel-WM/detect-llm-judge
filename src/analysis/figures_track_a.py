"""Track A figures (NEXT_STEPS_TRACKS_A_B_C.md). Reads only from
results/phase0c_track_a_*.parquet; never recomputes anything.

Run: .venv/Scripts/python.exe -m src.analysis.figures_track_a
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"


def fig_a4_three_strategies(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    piv = df.pivot_table(index="N", columns="strategy", values="top1_correct", aggfunc="mean")
    styles = {"random": "o-", "per_row_topk": "s--", "pairwise_coverage": "^-"}
    for strategy in ["random", "per_row_topk", "pairwise_coverage"]:
        ax.plot(piv.index, piv[strategy], styles[strategy], label=strategy)
    ax.set_xscale("log")
    ax.set_xlabel("N (probe count)")
    ax.set_ylabel("pooled top-1 accuracy")
    ax.set_title("A4: probe selection strategy comparison\n(psi=20deg, sigma=0.5, lambda=4, pooled over rules)")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "track_a_fig_A4_three_strategies.png", dpi=150)
    plt.close(fig)


def fig_a5_conditioning_fixed(df: pd.DataFrame) -> None:
    sub = df[df["delta"] != "baseline_sigma0"].copy()
    sub["delta_f"] = sub["delta"].astype(float)
    log_k = np.log10(sub["kappa_D"])
    log_t = np.log10(sub["theta"].clip(lower=1e-300))
    slope, intercept = np.polyfit(log_k, log_t, 1)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(log_k, log_t, s=6, alpha=0.15, color="steelblue", label="replicates (sigma=0.5 only)")

    medians = sub.groupby("delta_f").agg(k=("kappa_D", "median"), t=("theta", "median")).sort_index()
    ax.plot(np.log10(medians["k"]), np.log10(medians["t"]), "o-", color="black", label="median per delta")

    fit_x = np.array([log_k.min(), log_k.max()])
    ax.plot(fit_x, slope * fit_x + intercept, "r--", label=f"fit slope = {slope:.3f}")
    ax.plot(fit_x, 1.0 * fit_x + (intercept + (slope - 1.0) * fit_x.mean()), "g:",
             label="slope = 1 (spec prediction, shifted for comparison)")

    ax.set_xlabel("log10 kappa_D (condition number)")
    ax.set_ylabel("log10 theta (angular error, rad)")
    ax.set_title("A5: angular error vs conditioning, fixed construction\n(sigma=0 excluded from fit; kappa_D now spans ~4 orders of magnitude)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "track_a_fig_A5_conditioning_fixed.png", dpi=150)
    plt.close(fig)
    return slope


def fig_a2a3_grid(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, rule in zip(axes, ["sign_agreement", "bradley_terry", "nnls_mixture"]):
        sub = df[df["rule"] == rule]
        ax.scatter(sub["top1_accuracy"], sub["margin_fpr"], s=18, alpha=0.7, label="margin FPR@F1-tau")
        ax.axhline(0.1, color="red", linestyle="--", linewidth=1, label="FPR=0.1")
        ax.axvline(0.9, color="gray", linestyle="--", linewidth=1, label="accuracy=0.9")
        ax.set_xlabel("top-1 accuracy at this cell")
        ax.set_ylabel("margin FPR at F1-optimal tau")
        ax.set_title(rule)
        ax.set_ylim(-0.02, 1.0)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle("A2: does margin-threshold FPR fall below 0.1 in the easy (accuracy>0.9) regime?")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "track_a_fig_A2_fpr_vs_accuracy.png", dpi=150)
    plt.close(fig)


def fig_a3_auc_comparison(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    means = df.groupby("rule")[["auc_margin", "auc_gamma_hat", "auc_Lambda"]].mean()
    means = means.loc[["sign_agreement", "bradley_terry", "nnls_mixture"]]
    x = np.arange(len(means))
    width = 0.25
    for i, stat in enumerate(["auc_margin", "auc_gamma_hat", "auc_Lambda"]):
        ax.bar(x + (i - 1) * width, means[stat], width, label=stat)
    ax.set_xticks(x)
    ax.set_xticklabels(means.index)
    ax.set_ylabel("mean AUC across all 30 grid cells")
    ax.set_title("A3: detection statistic comparison (margin vs gamma_hat vs Lambda)")
    ax.set_ylim(0.5, 1.0)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "track_a_fig_A3_auc_comparison.png", dpi=150)
    plt.close(fig)


def main() -> None:
    df_grid = pd.read_parquet(RESULTS_DIR / "phase0c_track_a_detection_grid.parquet")
    df_ma = pd.read_parquet(RESULTS_DIR / "phase0c_track_a_marginal_a_strategies.parquet")
    df_cond = pd.read_parquet(RESULTS_DIR / "phase0c_track_a_conditioning.parquet")

    fig_a2a3_grid(df_grid)
    print("wrote track_a_fig_A2_fpr_vs_accuracy.png")
    fig_a3_auc_comparison(df_grid)
    print("wrote track_a_fig_A3_auc_comparison.png")
    fig_a4_three_strategies(df_ma)
    print("wrote track_a_fig_A4_three_strategies.png")
    slope = fig_a5_conditioning_fixed(df_cond)
    print(f"wrote track_a_fig_A5_conditioning_fixed.png (fitted slope, sigma>0 only = {slope:.3f})")


if __name__ == "__main__":
    main()
