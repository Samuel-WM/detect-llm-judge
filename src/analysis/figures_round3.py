"""Figures for NEXT_STEPS_ROUND_3_FINAL.md Section 4. Reads long-format results only; recomputes
nothing.

F11 (reward-model dataset-overlap matrix) is produced before any scoring, per Section 3.1.
Track C (F1-F6) and Track E (F7-F10) figures are produced from their respective result files
once those sections finish.

Run: .venv/Scripts/python.exe -m src.analysis.figures_round3 --which f11
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"

SHORT_RM = {
    "OpenAssistant/reward-model-deberta-v3-large-v2": "OA-deberta-v3-large",
    "internlm/internlm2-1_8b-reward": "InternLM2-1.8B",
    "Ray2333/gpt2-large-helpful-reward_model": "gpt2-large-helpful",
    "Ray2333/gpt2-large-harmless-reward_model": "gpt2-large-harmless",
}


def fig11_dataset_overlap() -> None:
    """F11: models x documented training corpora, with the blocked candidate marked."""
    pre = json.load(open(RESULTS_DIR / "rm_precheck.json"))
    provenance = pre["provenance"]
    blocked = {r["repo"] for r in pre["results"] if not r.get("all_ok", False)}

    corpora = sorted({c for v in provenance.values() for c in v})
    models = list(provenance)
    mat = np.zeros((len(models), len(corpora)))
    for i, m in enumerate(models):
        for c in provenance[m]:
            mat[i, corpora.index(c)] = 1.0

    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.imshow(mat, cmap="Blues", vmin=0, vmax=1.4, aspect="auto")
    ax.set_xticks(range(len(corpora)))
    ax.set_xticklabels([c.replace("/", "/\n") for c in corpora], fontsize=7, rotation=30, ha="right")
    labels = [SHORT_RM.get(m, m) + ("  [BLOCKED]" if m in blocked else "") for m in models]
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(len(models)):
        for j in range(len(corpora)):
            if mat[i, j]:
                ax.text(j, i, "x", ha="center", va="center", fontsize=11, color="white")
    ax.set_title("F11: reward-model training-corpus overlap\n"
                 "(three of four include Anthropic/hh-rlhf; no independence claim)", fontsize=10)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "F11_rm_dataset_overlap.png", dpi=150)
    plt.close(fig)

    # pairwise shared-corpus count, the actual confound measure
    n = len(models)
    overlap = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            overlap[i, j] = len(set(provenance[models[i]]) & set(provenance[models[j]]))
    print("F11 pairwise shared-corpus counts:")
    print(pd.DataFrame(overlap, index=[SHORT_RM.get(m, m) for m in models],
                       columns=[SHORT_RM.get(m, m) for m in models]).to_string())
    print(f"\nwrote {FIGURES_DIR / 'F11_rm_dataset_overlap.png'}")


def fig9_calibration() -> None:
    """F9 gate figure: realized test-retest correlation vs target R."""
    gates = json.load(open(RESULTS_DIR / "track_e2_gates.json"))
    calib = pd.DataFrame(gates["calibration"])
    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="y = x")
    ax.errorbar(calib["R_target"], calib["R_realized"], yerr=3 * calib["se"],
                fmt="o", capsize=3, label="realized (3 SE)")
    ax.set_xlabel("target R")
    ax.set_ylabel("realized corr(d_obs$^{(1)}$, d_obs$^{(2)}$)")
    ax.set_title("F9: Track E calibration gate\nloading a = sqrt(R) reproduces test-retest R")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "F9_track_e_calibration.png", dpi=150)
    plt.close(fig)
    print(f"wrote {FIGURES_DIR / 'F9_track_e_calibration.png'}")


def fig10_heterogeneous() -> None:
    """F10: accuracy vs sigma_D using the measured heterogeneous R_m vector, panelled by psi.
    Explicitly NOT labeled as a real operating point."""
    path = RESULTS_DIR / "track_e2_E2_heterogeneous.parquet"
    if not path.exists():
        print(f"skip F10: {path} not present yet")
        return
    df = pd.read_parquet(path)
    c_hat = 0.0489
    sub = df[np.isclose(df["c"], c_hat)]
    psis = sorted(df["psi_deg"].unique(), reverse=True)
    fig, axes = plt.subplots(1, len(psis), figsize=(3.1 * len(psis), 3.6), sharey=True)
    for ax, psi in zip(np.atleast_1d(axes), psis):
        s = sub[sub["psi_deg"] == psi]
        for rule, g in s.groupby("rule"):
            acc = g.groupby("sigma_D")["top1_correct"].mean()
            ax.plot(acc.index, acc.values, marker="o", label=rule)
        ax.axhline(0.60, color="red", linestyle="--", linewidth=1)
        ax.axhline(1 / 5, color="gray", linestyle=":", linewidth=1)
        ax.set_title(f"psi={psi}°", fontsize=9)
        ax.set_xlabel("sigma_D")
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("top-1 accuracy")
    np.atleast_1d(axes)[-1].legend(fontsize=7)
    fig.suptitle("F10: heterogeneous measured-R diagnostic (c = c_hat = 0.0489). "
                 "NOT a real-pool prediction — latent psi is not identified.", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "F10_heterogeneous_diagnostic.png", dpi=150)
    plt.close(fig)
    print(f"wrote {FIGURES_DIR / 'F10_heterogeneous_diagnostic.png'}")


def fig7_generic_map() -> None:
    """F7: per-rule psi x R heatmaps at c = c_hat, at the measured Arm A tau=2 sigma_D_ratio."""
    df = pd.read_parquet(RESULTS_DIR / "track_e2_E1_generic_map.parquet")
    c_hat = 0.0489
    sigma = max(df["sigma_D"].unique())  # the measured Arm A sigma_D_ratio (largest in grid)
    sub = df[np.isclose(df["c"], c_hat) & np.isclose(df["sigma_D"], sigma)]
    rules = ["sign_agreement", "bradley_terry", "nnls_mixture"]

    fig, axes = plt.subplots(1, 4, figsize=(23, 5.2))

    def draw(ax, piv, title):
        piv = piv.sort_index(ascending=False)
        im = ax.imshow(piv.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels([f"{c:g}" for c in piv.columns], fontsize=8)
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels([f"{r:g}°" for r in piv.index], fontsize=8)
        ax.set_xlabel("R (measured reliability)")
        ax.set_title(title, fontsize=10)
        try:
            cs = ax.contour(range(len(piv.columns)), range(len(piv.index)), piv.values,
                            levels=[0.60], colors="red", linewidths=2)
            ax.clabel(cs, fmt={0.60: "0.60"})
        except (ValueError, IndexError):
            pass
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                ax.text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center",
                        color="white" if piv.values[i, j] < 0.5 else "black", fontsize=6.5)
        return im

    for ax, rule in zip(axes[:3], rules):
        im = draw(ax, sub[sub["rule"] == rule].pivot_table(index="psi_deg", columns="R",
                                                            values="top1_correct", aggfunc="mean"), rule)
    axes[0].set_ylabel("psi (latent separation)")
    im = draw(axes[3], sub.pivot_table(index="psi_deg", columns="R", values="top1_correct",
                                        aggfunc="mean"), "pooled (continuity only)")
    fig.colorbar(im, ax=axes, label="top-1 accuracy", fraction=0.015, pad=0.02)
    fig.suptitle(f"F7: E1 generic homogeneous-reliability map, c = c_hat = {c_hat}, "
                 f"sigma_D = {sigma:.3f} (measured Arm A tau=2 sigma_D_ratio). "
                 f"Sensitivity map, NOT a real-pool prediction.", fontsize=11)
    fig.savefig(FIGURES_DIR / "F7_E1_generic_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIGURES_DIR / 'F7_E1_generic_map.png'}")


def fig8_c_sensitivity() -> None:
    """F8: how the 0.60 boundary moves with shared-noise fraction c."""
    df = pd.read_parquet(RESULTS_DIR / "track_e2_E1_generic_map.parquet")
    sigmas = sorted(df["sigma_D"].unique())
    fig, axes = plt.subplots(1, len(sigmas), figsize=(4.6 * len(sigmas), 4.4), sharey=True)
    for ax, sd in zip(np.atleast_1d(axes), sigmas):
        for c in sorted(df["c"].unique()):
            s = df[np.isclose(df["c"], c) & np.isclose(df["sigma_D"], sd)]
            # smallest psi that still clears 0.60, per R
            acc = s.pivot_table(index="psi_deg", columns="R", values="top1_correct", aggfunc="mean")
            Rs, thresh = [], []
            for R in acc.columns:
                ok = acc.index[acc[R] >= 0.60]
                Rs.append(R)
                thresh.append(min(ok) if len(ok) else np.nan)
            ax.plot(Rs, thresh, marker="o", label=f"c={c:g}")
        ax.set_xlabel("R"); ax.set_title(f"sigma_D={sd:g}", fontsize=10); ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("smallest psi (deg) still clearing 0.60")
    np.atleast_1d(axes)[-1].legend(fontsize=8)
    fig.suptitle("F8: sensitivity of the 0.60 boundary to shared-noise fraction c "
                 "(higher curve = harder; NaN gaps = never clears 0.60)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "F8_E1_c_sensitivity.png", dpi=150)
    plt.close(fig)
    print(f"wrote {FIGURES_DIR / 'F8_E1_c_sensitivity.png'}")


def fig22_probe_filter() -> None:
    """F22 (round 4, 2.1): probe-length filter -- retained fraction and length distributions,
    filtered vs full, per model tokenizer."""
    d = json.load(open(RESULTS_DIR / "rm_probe_filter.json"))
    worst = np.array(d["worst_case_lengths"])
    retained = np.zeros(len(worst), dtype=bool)
    retained[d["retained_indices"]] = True
    cap = d["max_len"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))

    ax = axes[0]
    for repo, lens in d["per_model_lengths"].items():
        ax.hist(np.array(lens), bins=np.linspace(0, 1200, 60), histtype="step", linewidth=1.6,
                label=SHORT_RM.get(repo, repo))
    ax.axvline(cap, color="red", ls="--", linewidth=2, label=f"cap = {cap}")
    ax.axvline(512, color="darkred", ls=":", linewidth=1.5, label="DeBERTa ctx = 512")
    ax.set_xlabel("(prompt, response) tokens, max over both responses")
    ax.set_ylabel("probes")
    ax.set_title("Per-tokenizer length distributions")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    bins = np.linspace(0, 1200, 60)
    ax.hist(worst[retained], bins=bins, alpha=0.65, label=f"retained (n={retained.sum()})")
    ax.hist(worst[~retained], bins=bins, alpha=0.65, label=f"dropped (n={(~retained).sum()})")
    ax.axvline(cap, color="red", ls="--", linewidth=2)
    ax.set_xlabel("worst-case tokens across all 4 tokenizers")
    ax.set_title(f"Retained {d['retained_fraction']:.1%} >= {d['retain_threshold']:.0%} gate "
                 f"-> keep all FOUR models")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("F22: reward-model probe-length filter. Reward models score (prompt, ONE response), "
                 "so the round-3 640-token block did not apply.", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "F22_rm_probe_filter.png", dpi=150)
    plt.close(fig)
    print(f"wrote {FIGURES_DIR / 'F22_rm_probe_filter.png'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", nargs="*", default=["f11"])
    args = parser.parse_args()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fns = {"f11": fig11_dataset_overlap, "f9": fig9_calibration, "f10": fig10_heterogeneous,
           "f7": fig7_generic_map, "f8": fig8_c_sensitivity, "f22": fig22_probe_filter}
    for w in args.which:
        fns[w]()


if __name__ == "__main__":
    main()
