"""Track C figures F1-F6 (NEXT_STEPS_ROUND_3_FINAL.md Section 4). Reads results/track_c_pilot.json;
recomputes nothing.

Interpretation-class labelling is baked into the figures per Section 7: only the stochastic
feature-judge arm's magnitude regression is a primary magnitude estimate; the deterministic arm
and the Qwen arm are marked diagnostic/exploratory for magnitude.

Run: .venv/Scripts/python.exe -m src.analysis.figures_track_c
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = REPO_ROOT / "results" / "track_c_pilot.json"
CACHE = REPO_ROOT / "data_cache"
FIG = REPO_ROOT / "figures"

CLASS = {
    "A_tau2": ("primary magnitude", "tab:blue"),
    "A_det": ("diagnostic magnitude (contrast)", "tab:orange"),
    "B_qwen": ("primary rank / exploratory magnitude", "tab:green"),
}


def load():
    d = json.load(open(RESULTS))
    real = {r["tag"]: r for r in d["results"] if not r["shuffled"]}
    shuf = {r["tag"].replace("_shuffled", ""): r for r in d["results"] if r["shuffled"]}
    fj = json.load(open(CACHE / "feature_judges.json"))
    return d, real, shuf, fj


def f1_scatter(real, fj):
    d_std = np.array(fj["d_std_probe"])
    qwen = None
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, arm in zip(axes, ["A_tau2", "A_det", "B_qwen"]):
        r = real.get(arm)
        if not r:
            continue
        delta = np.array(r["delta"])
        if arm.startswith("A_"):
            x = d_std
            xlabel = "d_std_probe (feature judge, known ground truth)"
        else:
            import json as _j
            m = {}
            for line in open(CACHE / "judge_margins_fixedbin_Qwen__Qwen2.5-1.5B-Instruct.jsonl"):
                row = _j.loads(line)
                if row["template_name"] == "template_train":
                    m[row["prompt_idx"]] = row["d_m"]
            x = np.array([m[i] for i in range(len(delta))])
            xlabel = "d_Qwen_obs (observed, not ground truth)"
        mm = r["metrics"]
        ax.scatter(x, delta, s=8, alpha=0.35, color=CLASS[arm][1])
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, mm["intercept"] + mm["gamma"] * xs, "r-", linewidth=2)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("Delta")
        ax.set_title(f"{arm}\nR2={mm['r_squared']:.4f}  sigma_D_ratio={mm['sigma_D_ratio']:.2f}  "
                     f"sigma_D_total={mm['sigma_D_total']:.3f}\n[{CLASS[arm][0]}]", fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle("F1: Delta vs generating margin. Only A_tau2 is a primary magnitude estimate.", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "F1_track_c_delta_scatter.png", dpi=150)
    plt.close(fig)


def f3_rank(real):
    arms = [a for a in ["A_tau2", "A_det", "B_qwen"] if a in real]
    sign = [real[a]["metrics"]["sign_agreement"] for a in arms]
    spear = [real[a]["metrics"]["spearman"] for a in arms]
    sign_lr = [real[a]["metrics_length_residualized"]["sign_agreement"] for a in arms]
    spear_lr = [real[a]["metrics_length_residualized"]["spearman"] for a in arms]

    x = np.arange(len(arms))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].bar(x - 0.2, sign, 0.4, label="raw")
    axes[0].bar(x + 0.2, sign_lr, 0.4, label="length-residualized")
    axes[0].axhline(0.5, color="red", ls="--", label="chance")
    axes[0].set_xticks(x); axes[0].set_xticklabels(arms); axes[0].set_ylabel("sign agreement")
    axes[0].set_title("Sign agreement (rank-level)"); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    axes[1].bar(x - 0.2, spear, 0.4, label="raw")
    axes[1].bar(x + 0.2, spear_lr, 0.4, label="length-residualized")
    axes[1].axhline(0.0, color="red", ls="--")
    axes[1].set_xticks(x); axes[1].set_xticklabels(arms); axes[1].set_ylabel("Spearman")
    axes[1].set_title("Spearman (rank-level)"); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    fig.suptitle("F3: rank-level transmission -- the headline cross-arm comparison", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "F3_track_c_rank_transmission.png", dpi=150)
    plt.close(fig)


def f4_densities(real, shuf):
    arms = [a for a in ["A_tau2", "A_det", "B_qwen"] if a in real and a in shuf]
    fig, axes = plt.subplots(1, len(arms), figsize=(5.2 * len(arms), 4.2), sharey=True)
    for ax, arm in zip(np.atleast_1d(axes), arms):
        dr = np.array(real[arm]["delta"]); ds = np.array(shuf[arm]["delta"])
        bins = np.linspace(min(dr.min(), ds.min()), max(dr.max(), ds.max()), 60)
        ax.hist(dr, bins=bins, alpha=0.55, label=f"real (mean|D|={np.mean(np.abs(dr)):.2f})", density=True)
        ax.hist(ds, bins=bins, alpha=0.55, label=f"shuffled (mean|D|={np.mean(np.abs(ds)):.2f})", density=True)
        ax.set_title(f"{arm}  ratio={np.mean(np.abs(dr))/np.mean(np.abs(ds)):.2f}x", fontsize=10)
        ax.set_xlabel("Delta"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("density")
    fig.suptitle("F4: Delta scale, real vs matched shuffled-label control", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "F4_track_c_shuffled_control.png", dpi=150)
    plt.close(fig)


def f5_length(real, fj):
    len_diff = np.array(fj["len_diff_probe"])
    arms = [a for a in ["A_tau2", "A_det", "B_qwen"] if a in real]
    fig, axes = plt.subplots(1, len(arms), figsize=(5.2 * len(arms), 4.2))
    for ax, arm in zip(np.atleast_1d(axes), arms):
        delta = np.array(real[arm]["delta"])
        ax.scatter(len_diff, delta, s=8, alpha=0.35, color=CLASS[arm][1])
        c = np.corrcoef(len_diff, delta)[0, 1]
        lr = real[arm]["metrics_length_residualized"]
        ax.set_title(f"{arm}  corr(Delta, len_diff)={c:+.3f}\n"
                     f"len-resid: R2={lr['r_squared']:.4f} sign={lr['sign_agreement']:.3f}", fontsize=9)
        ax.set_xlabel("standardized length difference"); ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("Delta")
    fig.suptitle("F5: Delta vs response-length difference, with length-residualized summaries", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "F5_track_c_length.png", dpi=150)
    plt.close(fig)


def f6_qwen(real):
    r = real.get("B_qwen")
    if not r:
        return
    obs = r["metrics"]["pearson"]; dis = r["disattenuated_correlation"]
    fig, ax = plt.subplots(figsize=(6, 4.6))
    ax.bar(["observed\nPearson", "disattenuated\n(/ sqrt(R_m))"], [obs, dis],
           color=["tab:green", "tab:gray"])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("correlation with d_Qwen_obs")
    ax.set_title("F6: Qwen arm, measurement-error sensitivity analysis\n"
                 f"R_m=0.573, sqrt(R_m)=0.757. NO headline sigma_D is derived from the\n"
                 f"corrected value -- DPO still received only binary labels.", fontsize=9)
    for i, v in enumerate([obs, dis]):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "F6_track_c_qwen_disattenuation.png", dpi=150)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    d, real, shuf, fj = load()
    f1_scatter(real, fj); print("wrote F1")
    f3_rank(real); print("wrote F3")
    f4_densities(real, shuf); print("wrote F4")
    f5_length(real, fj); print("wrote F5")
    f6_qwen(real); print("wrote F6")
    print("(F2 sigma_D_ratio-vs-tau omitted: only tau=2 was run, per the doc's "
          "'if time is short' allowance -- a one-point sweep is not a figure)")


if __name__ == "__main__":
    main()
