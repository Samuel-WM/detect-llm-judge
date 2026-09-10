"""Round 4 figures F23-F25 (reward-model calibration stage).

Reads result JSONs only; recomputes no scores. Section A/C/D figures (F17-F22, F26-F28)
are NOT generated here: they are gated behind the Section B retention outcome.

Run: .venv/Scripts/python.exe -m src.analysis.figures_round4
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = REPO_ROOT / "results"
FIG = REPO_ROOT / "figures"

POOL = ["OpenAssistant/reward-model-deberta-v3-large-v2", "internlm/internlm2-1_8b-reward",
        "Ray2333/gpt2-large-helpful-reward_model", "Ray2333/gpt2-large-harmless-reward_model"]
SHORT = {POOL[0]: "OA-deberta", POOL[1]: "InternLM2-1.8B",
         POOL[2]: "gpt2-helpful", POOL[3]: "gpt2-harmless"}
HARD_CELL = {"gpt2-helpful", "gpt2-harmless"}


def f23_margin_matrix():
    psi = json.load(open(RESULTS / "rm_harness.json"))["psi_eff"]
    names = psi["names"]
    C = np.array(psi["correlation"])
    P = np.array(psi["psi_eff_deg"])
    fig, ax = plt.subplots(figsize=(7.8, 6.6))
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    for i in range(len(names)):
        for j in range(len(names)):
            is_hard = names[i] in HARD_CELL and names[j] in HARD_CELL and i != j
            ax.text(j, i, f"r={C[i, j]:.3f}\npsi={P[i, j]:.1f}deg", ha="center", va="center",
                    fontsize=8, color="white" if abs(C[i, j]) > 0.55 else "black",
                    fontweight="bold" if is_hard else "normal")
            if is_hard:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                           edgecolor="lime", lw=3))
    fig.colorbar(im, ax=ax, label="pairwise reward-margin correlation")
    ax.set_title("F23: reward-model margin correlation and psi_eff\n"
                 "487 retained probes; psi_eff = arccos(r) is EFFECTIVE separation on THIS probe\n"
                 "distribution, not interchangeable with synthetic isotropic psi. "
                 "Green = pre-registered hard cell.", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "F23_rm_margin_matrix.png", dpi=150)
    plt.close(fig)
    print("wrote F23_rm_margin_matrix.png")


def f24_harness_panel():
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    for repo in POOL:
        p = RESULTS / f"rm_harness_{repo.replace('/', '__')}.json"
        if not p.exists():
            continue
        r = json.load(open(p))["result"]
        d = np.array(r["d_m"])
        d = d / d.std()
        axes[0].hist(d, bins=50, histtype="step", linewidth=1.6, density=True,
                     label=f"{SHORT[repo]} (sd={np.array(r['d_m']).std():.3f})")
        f = r["H6"]["frac_preferring_intact"]
        axes[1].bar(SHORT[repo], f, color="tab:green" if f > 0.5 else "tab:red")
        axes[1].annotate(f"{f:.3f}", (SHORT[repo], f), ha="center",
                         textcoords="offset points", xytext=(0, 3), fontsize=8)
    axes[0].set_xlabel("margin d_m, standardized by its own sd")
    axes[0].set_ylabel("density")
    axes[0].set_title("H1 (CORRECTNESS GATE): margin distributions are non-degenerate")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].axhline(0.5, color="k", ls="--", label="indifference")
    axes[1].set_ylabel("fraction preferring INTACT over token-shuffled")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("H6 (BEHAVIORAL DIAGNOSTIC ONLY -- not a correctness gate)")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.suptitle("F24: reward-model harness panel", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "F24_rm_harness_panel.png", dpi=150)
    plt.close(fig)
    print("wrote F24_rm_harness_panel.png")


def f25_d1():
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    for repo in POOL:
        p = RESULTS / f"rm_d1_{repo.replace('/', '__')}.json"
        if not p.exists():
            continue
        r = json.load(open(p))
        full, lg = r["full_set"], r["large_gap"]
        ax.plot([0, 1], [full["accuracy"], lg["accuracy"]], "-o", linewidth=2, markersize=8,
                label=f"{SHORT[repo]}  primary {'PASS' if lg['passes'] else 'FAIL'}"
                      f"  (|d|={lg['abs_distance']:.4f}, p={lg['p_value']:.2e})")
        ax.annotate(f"{lg['accuracy']:.4f}", (1, lg["accuracy"]), textcoords="offset points",
                    xytext=(9, -3), fontsize=8)
    ax.axhline(0.5, color="k", ls="--", linewidth=1)
    ax.axhspan(0.4, 0.6, color="gray", alpha=0.18)
    ax.text(1.03, 0.412, "chance +/- 0.10\nfail band", fontsize=8, va="center", color="dimgray")
    ax.text(1.03, 0.503, "chance", fontsize=8, va="bottom", color="black")
    ax.set_xticks([0, 1])
    ax.set_xlim(-0.08, 1.35)
    ax.set_xticklabels(["full held-out set\n(SECONDARY)", "large-gap subset\n(PRIMARY)"])
    ax.set_ylabel("D1 accuracy (agreement with UltraFeedback preference)")
    ax.set_ylim(0.38, 0.76)
    ax.set_title("F25: D1 external-validity check, pre-registered rule applied unmodified\n"
                 "retain iff |acc - 0.5| >= 0.10 AND two-sided exact binomial p < 0.001 "
                 "on the PRIMARY set", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "F25_rm_d1.png", dpi=150)
    plt.close(fig)
    print("wrote F25_rm_d1.png")


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    f23_margin_matrix()
    f24_harness_panel()
    f25_d1()
