"""NEXT_STEPS_ROUND_6_SAMEDAY_FINAL.md Section E: Round 6 figures.

Only figures whose sections actually completed are generated (Section E). Deferred sections
produce no figure rather than an empty one.

Run: .venv/Scripts/python.exe -m src.analysis.figures_round6
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
RESULTS = REPO_ROOT / "results"
FIG = REPO_ROOT / "figures"
ARMS = ["RM_OA-deberta", "RM_gpt2-helpful", "RM_gpt2-harmless"]
SHORT = {a: a.split("_", 1)[1] for a in ARMS}


def f29_length_mechanism() -> None:
    a2 = pd.read_csv(RESULTS / "round6_a2_decomposition.csv")
    a3 = pd.read_csv(RESULTS / "round6_a3_gamma2.csv")
    audit = json.load(open(RESULTS / "round6_a5_labelaudit.json"))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    ax = axes[0][0]
    for arm in ARMS:
        s = a3[a3.arm == arm]
        ax.plot(s.step, s.c_bar, "-o", label=SHORT[arm])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("c_bar = mean per-token log(pi_theta/pi_ref)")
    ax.set_title("c_bar(t): the accumulating KL cost, always negative and growing", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0][1]
    x = np.arange(len(a3))
    ax.bar(x - 0.2, a3.gamma_2_hat_raw, 0.4, color="tab:red",
           label="predicted: c_bar * sd(len_diff)")
    ax.bar(x + 0.2, a3.gamma_2_observed_r5, 0.4, color="tab:blue", label="observed gamma_2")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{SHORT[r.arm]}\n{r.step}" for r in a3.itertuples()], fontsize=7)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("coefficient (Delta units per SD of len_diff)")
    ax.set_title("mechanical accumulation OVER-predicts gamma_2 by 2.6-13.5x\n"
                 "(R^2 of Delta by c_bar*len_diff alone is negative in 8 of 9 cells)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1][0]
    for arm, m in zip(ARMS, ["o", "s", "^"]):
        s = a2[a2.arm == arm]
        ax.plot(s.step, s.sd_len_accum, "-" + m, color="tab:red",
                label=f"{SHORT[arm]}: sd(length_accum)" if arm == ARMS[0] else None)
        ax.plot(s.step, s.sd_token_diff, "--" + m, color="tab:green",
                label=f"{SHORT[arm]}: sd(token_diff)" if arm == ARMS[0] else None)
        ax.plot(s.step, s.sd_Delta, ":" + m, color="k",
                label=f"{SHORT[arm]}: sd(Delta)" if arm == ARMS[0] else None)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("standard deviation")
    ax.set_title("both components are ~2x sd(Delta) and cancel\n"
                 f"corr(accum, token_diff) = {a2.corr_accum_tokdiff.min():.2f} to "
                 f"{a2[a2.step > 51].corr_accum_tokdiff.max():.2f}", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1][1]
    n = audit["null"]
    xs = np.arange(len(n))
    ax.bar(xs, [r["corr_percentile"] for r in n], color="tab:purple", width=0.5)
    for i, r in enumerate(n):
        ax.annotate(f"obs {r['corr_obs']:+.3f}\nnull {r['corr_null_mean']:+.3f}\n"
                    f"tail p={r['corr_two_sided_tail_p']:.2f}",
                    (i, r["corr_percentile"]), ha="center", va="bottom", fontsize=7)
    ax.axhline(50, color="k", ls=":", lw=1, label="null median")
    ax.axhspan(2.5, 97.5, color="gray", alpha=0.15, label="central 95% of the null")
    ax.set_xticks(xs)
    ax.set_xticklabels([r["arm"].split("_", 1)[1] for r in n], fontsize=8)
    ax.set_ylim(0, 118)
    ax.set_ylabel("percentile of realized corr(label_sign, len_diff)")
    ax.set_title("A5: the frozen label draw is UNremarkable under its own null\n"
                 "(10,000 label-only BT resamples from the same frozen margins)", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("F29: the Round 5 length mechanism. RETROSPECTIVE / EXPLORATORY -- nothing here "
                 "revises a Round 5 number or verdict.", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "F29_length_mechanism.png", dpi=150)
    plt.close(fig)
    print("wrote F29_length_mechanism.png")


def _extra_arms() -> list[str]:
    return [p.name[len("round5_"):-len("_trajectory.json")]
            for p in RESULTS.glob("round5_*_trajectory.json")
            if any(k in p.name for k in ("CTRL", "FJ_PERP", "J_SHORT"))]


def f30_control_mechanisms() -> None:
    extra = _extra_arms()
    if not extra:
        raise FileNotFoundError("no CTRL / FJ_PERP / J_SHORT arm ran")
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    for arm in ARMS + extra:
        p = RESULTS / f"round5_{arm}_trajectory.json"
        if not p.exists():
            continue
        rows = json.load(open(p))["rows"]
        st = [r["step"] for r in rows]
        ctrl = arm in extra
        sty = {"lw": 2.6, "ls": "--"} if ctrl else {"lw": 1.4, "alpha": 0.75}
        axes[0].plot(st, [r["gamma_2"] for r in rows], "-o", ms=3, label=arm, **sty)
        axes[1].plot(st, [r.get("pearson", np.nan) for r in rows], "-o", ms=3, label=arm, **sty)
        axes[2].plot(st, [r["sd_delta"] for r in rows], "-o", ms=3, label=arm, **sty)
    for ax, t, yl in zip(axes,
                         ["gamma_2 (length coefficient)", "Pearson(Delta, d_true)", "sd(Delta)"],
                         ["gamma_2", "Pearson", "sd(Delta)"]):
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel("optimizer step")
        ax.set_ylabel(yl)
        ax.set_title(t, fontsize=10)
        ax.legend(fontsize=6.5)
        ax.grid(alpha=0.3)
    fig.suptitle("F30: control mechanisms (dashed) against the Round 5 real-RM arms (thin). "
                 "Does negative length drift appear without coherent teacher structure?",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "F30_control_mechanisms.png", dpi=150)
    plt.close(fig)
    print("wrote F30_control_mechanisms.png")


def f31_kappa() -> None:
    kl = pd.read_csv(RESULTS / "round6_b2_kl.csv")
    kap = json.load(open(RESULTS / "round6_b3_kappa.json"))
    kappa = kap["kappa_R6"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for arm in ARMS:
        s = kl[kl.arm == arm].sort_values("step")
        ax.plot(s.step, s.KL_bar, "-o", ms=4, label=SHORT[arm])
    ax.axhline(kappa, color="crimson", ls="--", lw=2, label=f"kappa_R6 = {kappa:.4f}")
    for r in kap["peak_table"]:
        ax.scatter([r["peak_S_step"]], [r["KL_bar_at_peak"]], marker="*", s=300,
                   color="gold", edgecolors="k", zorder=5)
    ax.set_yscale("log")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("training-side KL_bar (exact, full vocabulary)")
    ax.set_title("B2: KL(pi_theta || pi_ref) on 497 fixed training contexts\n"
                 "stars = each arm's Round 5 peak-S checkpoint", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    for arm in ARMS:
        att = json.load(open(RESULTS / f"round5_{arm}_attribution.json"))
        pc = att["per_checkpoint"]
        ax.plot([c["step"] for c in pc], [c["raw"]["S_obs"] for c in pc], "-o", ms=4,
                label=SHORT[arm])
    for r in kap["round5_selection"]:
        att = json.load(open(RESULTS / f"round5_{r['arm']}_attribution.json"))
        pc = {c["step"]: c["raw"]["S_obs"] for c in att["per_checkpoint"]}
        st = r["kappa_selected_step"]
        ax.scatter([st], [pc[st]], marker="D", s=120, facecolors="none", edgecolors="crimson",
                   lw=2.2, zorder=5)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("raw primary S")
    ax.set_title("kappa-selected checkpoints (red diamonds) against the S trajectory\n"
                 "selection uses TRAINING-side KL only", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("F31: the training-only kappa stopping rule. kappa is derived from Round 5 "
                 "holdout S once, retrospectively -- a development-set procedure, not holdout-free.",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "F31_kappa_stopping_rule.png", dpi=150)
    plt.close(fig)
    print("wrote F31_kappa_stopping_rule.png")


def f32_n1000() -> None:
    p = RESULTS / "round6_d3_n1000.json"
    if not p.exists():
        raise FileNotFoundError(str(p))
    d = json.load(open(p))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    steps = d["steps"]
    for ax, key, lab in zip(axes, ["pearson", "gamma_2", "raw_S"],
                            ["Pearson(Delta, d_true)", "gamma_2 (length)", "raw primary S"]):
        ax.plot(steps, d["n259"][key], "-o", label=f"N=259 (Round 5)")
        ax.plot(steps, d["nbig"][key], "-s", label=f"N={d['n_big']} (Round 6)")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel("optimizer step")
        ax.set_ylabel(lab)
        ax.set_title(lab, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle(f"F32: N={d['n_big']} versus N=259 at matched optimizer steps, "
                 f"gpt2-helpful teacher, single seed", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "F32_n1000_vs_n259.png", dpi=150)
    plt.close(fig)
    print("wrote F32_n1000_vs_n259.png")


FIGURES = {"F29": f29_length_mechanism, "F30": f30_control_mechanisms,
           "F31": f31_kappa, "F32": f32_n1000}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    for name, fn in FIGURES.items():
        if a.only and name not in a.only:
            continue
        try:
            fn()
        except FileNotFoundError as e:
            print(f"{name}: skipped, section did not run ({e})")


if __name__ == "__main__":
    main()
