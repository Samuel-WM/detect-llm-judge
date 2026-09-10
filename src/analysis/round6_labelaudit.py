"""NEXT_STEPS_ROUND_6_SAMEDAY_FINAL.md A5: frozen-label length-bias audit.

Asks whether the Round 5 hard labels were themselves length-biased, and whether the realized bias
is unusual under the label model that generated them. No training, no GPU: the continuous teacher
margins and the sampled labels are already frozen on disk.

The null is label-sampling only: resample hard labels from the SAME frozen continuous margins at
tau = 2 and ask where the realized Round 5 draw sits. This holds the teacher fixed and varies only
the Bernoulli draw, which is exactly the randomness the frozen seed collapsed.

RETROSPECTIVE / EXPLORATORY.

Run: .venv/Scripts/python.exe -m src.analysis.round6_labelaudit
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from src.data.build_prompts import load_cfg
from src.train.round5_attribution import TAU, frozen_training_pairs, teacher_margins_train

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = REPO_ROOT / "results"
CKPT = REPO_ROOT / "checkpoints" / "round5"
ARMS = ["RM_OA-deberta", "RM_gpt2-helpful", "RM_gpt2-harmless"]
N_RESAMPLE = 10_000
SEED = 20250914


def logistic_fit(x: np.ndarray, y: np.ndarray, iters: int = 50) -> dict:
    """IRLS logistic regression of y on [1, x]; returns slope, SE and a two-sided Wald p."""
    from scipy.stats import norm
    X = np.column_stack([np.ones_like(x), x])
    b = np.zeros(2)
    for _ in range(iters):
        eta = X @ b
        p = 1 / (1 + np.exp(-eta))
        W = np.clip(p * (1 - p), 1e-9, None)
        z = eta + (y - p) / W
        XtW = X.T * W
        b_new = np.linalg.solve(XtW @ X, XtW @ z)
        if np.max(np.abs(b_new - b)) < 1e-10:
            b = b_new
            break
        b = b_new
    p = 1 / (1 + np.exp(-(X @ b)))
    W = np.clip(p * (1 - p), 1e-9, None)
    cov = np.linalg.inv((X.T * W) @ X)
    se = float(np.sqrt(cov[1, 1]))
    zstat = b[1] / se
    return {"slope": float(b[1]), "se": se, "z": float(zstat),
            "p_value": float(2 * norm.sf(abs(zstat)))}


def stats_for(labels: np.ndarray, len_diff: np.ndarray) -> tuple[float, float]:
    """(corr(label_sign, len_diff), fraction choosing the shorter response)."""
    sign = 2 * labels - 1
    corr = float(np.corrcoef(sign, len_diff)[0, 1])
    chosen_len = np.where(labels == 1, len_diff, -len_diff)   # chosen minus rejected length
    nz = chosen_len != 0
    frac_short = float(np.mean(chosen_len[nz] < 0)) if nz.any() else float("nan")
    return corr, frac_short


def main() -> None:
    cfg = load_cfg()
    tok = AutoTokenizer.from_pretrained(cfg["models"]["base_policy"])
    pairs = frozen_training_pairs()
    len_diff = np.array([len(tok.encode(p["response_a"])) - len(tok.encode(p["response_b"]))
                         for p in pairs], dtype=float)
    z_len = (len_diff - len_diff.mean()) / len_diff.std()
    print(f"A5: {len(pairs)} frozen training pairs, len_diff sd = {len_diff.std():.2f} tokens")

    rows, nulls, labels_by_arm = [], [], {}
    for arm in ARMS:
        meta = json.load(open(CKPT / arm / "run_meta.json"))
        labels = np.array(meta["labels"])
        labels_by_arm[arm] = labels
        corr, frac_short = stats_for(labels, len_diff)
        chosen_minus_rej = float(np.mean(np.where(labels == 1, len_diff, -len_diff)))
        lf = logistic_fit(z_len, labels.astype(float))

        # null: resample hard labels from the SAME frozen continuous margins at tau = 2
        d = teacher_margins_train(arm)
        zz = (d - d.mean()) / d.std()
        p = 1 / (1 + np.exp(-TAU * zz))
        rng = np.random.default_rng(SEED + ARMS.index(arm))
        u = rng.random((N_RESAMPLE, len(p)))
        L = (u < p[None, :]).astype(int)
        sign = 2 * L - 1
        lc = len_diff - len_diff.mean()
        null_corr = (sign @ lc) / (np.sqrt((sign ** 2).sum(1) - (sign.sum(1) ** 2) / len(lc))
                                   * np.sqrt((lc ** 2).sum()))
        cl = np.where(L == 1, len_diff[None, :], -len_diff[None, :])
        nz = len_diff != 0
        null_short = (cl[:, nz] < 0).mean(axis=1)

        pct_corr = float((null_corr < corr).mean() * 100)
        pct_short = float((null_short < frac_short).mean() * 100)
        tail_corr = float(2 * min((null_corr <= corr).mean(), (null_corr >= corr).mean()))
        tail_short = float(2 * min((null_short <= frac_short).mean(), (null_short >= frac_short).mean()))

        rows.append({"arm": arm, "corr_labelsign_lendiff": corr,
                     "mean_chosen_minus_rejected_len": chosen_minus_rej,
                     "frac_choosing_shorter": frac_short,
                     "logit_slope_on_z_lendiff": lf["slope"], "logit_se": lf["se"],
                     "logit_p": lf["p_value"]})
        nulls.append({"arm": arm, "corr_obs": corr, "corr_null_mean": float(null_corr.mean()),
                      "corr_null_sd": float(null_corr.std()), "corr_percentile": pct_corr,
                      "corr_two_sided_tail_p": min(tail_corr, 1.0),
                      "short_obs": frac_short, "short_null_mean": float(null_short.mean()),
                      "short_null_sd": float(null_short.std()), "short_percentile": pct_short,
                      "short_two_sided_tail_p": min(tail_short, 1.0)})

    A = pd.DataFrame(rows)
    N = pd.DataFrame(nulls)
    pd.set_option("display.width", 250)
    print("\n=== A5: realized frozen-label length bias ===")
    print(A.round(4).to_string(index=False))
    print(f"\n=== A5 null: {N_RESAMPLE:,} label-only BT resamples from the frozen margins (tau={TAU}) ===")
    print(N.round(4).to_string(index=False))

    print("\n=== pairwise hard-label agreement between teacher arms ===")
    ag = pd.DataFrame(
        [[float(np.mean(labels_by_arm[a] == labels_by_arm[b])) for b in ARMS] for a in ARMS],
        index=ARMS, columns=ARMS)
    print(ag.round(4).to_string())

    json.dump({"n_pairs": len(pairs), "len_diff_sd": float(len_diff.std()),
               "realized": rows, "null": nulls, "n_resample": N_RESAMPLE, "tau": TAU,
               "pairwise_label_agreement": ag.to_dict()},
              open(RESULTS / "round6_a5_labelaudit.json", "w"), indent=1)
    A.to_csv(RESULTS / "round6_a5_realized.csv", index=False)
    N.to_csv(RESULTS / "round6_a5_null.csv", index=False)
    print(f"\nwrote round6_a5_labelaudit.json, round6_a5_realized.csv, round6_a5_null.csv")


if __name__ == "__main__":
    main()
