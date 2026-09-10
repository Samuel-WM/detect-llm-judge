"""Step 2 of NEXT_STEPS_FINAL.md: D2 direct reliability, shared noise, latent correlation,
length loading. CPU only -- uses stored d_train / d_probe margins, no inversion of summary
statistics.

  D2a  R_m = pearson(d_train_m, d_probe_m), plus spearman and sign agreement
  D2b  naive attenuation correction corr(d_m,d_m')/sqrt(R_m R_m'), flag entries > 1
  D2c  delta_m = d_probe_m - d_train_m; inter-judge corr matrix; c_hat_raw, c_hat
  D2d  shared-noise-adjusted latent correlation, flagging out-of-range values (never clipping)
  D2e  length loading: d_m ~ c_m + gamma_m * (len(y) - len(y')), R^2, residual corr matrix

Run: .venv/Scripts/python.exe -m src.analysis.d2_reliability [--tag padfree]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from transformers import AutoTokenizer

from src.data.build_prompts import load_cfg

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = REPO_ROOT / "data_cache"
RESULTS_DIR = REPO_ROOT / "results"

SHORT = {
    "Qwen/Qwen2.5-1.5B-Instruct": "Qwen2.5-1.5B",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": "SmolLM2-1.7B",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "TinyLlama-1.1B",
    "meta-llama/Llama-3.2-1B-Instruct": "Llama-3.2-1B",
    "google/gemma-3-1b-it": "gemma-3-1b",
}


def load_margins(tag: str | None) -> pd.DataFrame:
    prefix = f"judge_margins_{tag}_" if tag else "judge_margins_"
    rows = []
    for path in CACHE_DIR.glob(f"{prefix}*.jsonl"):
        # don't let the untagged glob pick up tagged files
        if tag is None and path.name.count("_") > 2 and "judge_margins_padfree" in path.name:
            continue
        with open(path) as f:
            for line in f:
                rows.append(json.loads(line))
    assert rows, f"no {prefix}*.jsonl files found in {CACHE_DIR}"
    return pd.DataFrame(rows)


def pivot_by_template(df: pd.DataFrame, template_name: str, judges: list[str]) -> pd.DataFrame:
    sub = df[df["template_name"] == template_name]
    piv = sub.pivot(index="prompt_idx", columns="judge", values="d_m")
    return piv[[j for j in judges if j in piv.columns]]


def response_length_diff(n_pairs: int) -> pd.Series:
    """len(y) - len(y') in base-policy tokens, matching how the responses were generated."""
    cfg = load_cfg()
    tok = AutoTokenizer.from_pretrained(cfg["models"]["base_policy"])
    path = CACHE_DIR / "response_pairs_probe.jsonl"
    diffs = {}
    with open(path) as f:
        for idx, line in enumerate(f):
            if idx >= n_pairs:
                break
            row = json.loads(line)
            diffs[idx] = len(tok.encode(row["response_a"])) - len(tok.encode(row["response_b"]))
    return pd.Series(diffs, name="len_diff")


def fmt_matrix(mat: np.ndarray, labels: list[str]) -> str:
    return pd.DataFrame(mat, index=labels, columns=labels).round(3).to_string()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="fixedbin")
    args = parser.parse_args()

    df = load_margins(args.tag)
    judges = sorted(df["judge"].unique())
    labels = [SHORT.get(j, j) for j in judges]
    print(f"judges ({len(judges)}): {labels}")
    print(f"margin source: tag={args.tag!r}")

    train = pivot_by_template(df, "template_train", judges)
    probe = pivot_by_template(df, "template_probe", judges)
    common = train.dropna().index.intersection(probe.dropna().index)
    train, probe = train.loc[common], probe.loc[common]
    print(f"N common probe pairs with both templates for all judges: {len(common)}")

    out: dict = {"tag": args.tag, "judges": judges, "n": int(len(common))}

    # ---------------- D2a ----------------
    print("\n=== D2a: cross-template reliability (directly measured) ===")
    R = {}
    d2a_rows = []
    for j, lab in zip(judges, labels):
        r_p, p_p = pearsonr(train[j], probe[j])
        r_s, _ = spearmanr(train[j], probe[j])
        sign_agree = float(np.mean(np.sign(train[j]) == np.sign(probe[j])))
        R[j] = float(r_p)
        d2a_rows.append({"judge": lab, "R_pearson": r_p, "p_value": p_p,
                         "spearman": float(r_s), "sign_agreement": sign_agree})
        flag = "  <-- R <= 0: attenuation correction and Track E mapping NOT valid" if r_p <= 0 else ""
        print(f"  {lab:16s} R={r_p:+.4f} (p={p_p:.2e})  spearman={r_s:+.4f}  "
              f"sign_agree={sign_agree:.3f}{flag}")
    out["d2a"] = d2a_rows

    # ---------------- D2b ----------------
    print("\n=== D2b: naive attenuation correction (diagnostic only) ===")
    corr_obs = train.corr(method="pearson").values
    M = len(judges)
    naive = np.full((M, M), np.nan)
    for a in range(M):
        for b in range(M):
            ra, rb = R[judges[a]], R[judges[b]]
            if ra > 0 and rb > 0:
                naive[a, b] = corr_obs[a, b] / np.sqrt(ra * rb)
    print("observed inter-judge margin correlation (template_train):")
    print(fmt_matrix(corr_obs, labels))
    print("\nnaive attenuation-corrected:")
    print(fmt_matrix(naive, labels))
    above_one = [(labels[a], labels[b], float(naive[a, b]))
                 for a in range(M) for b in range(a + 1, M)
                 if np.isfinite(naive[a, b]) and abs(naive[a, b]) > 1.0]
    if above_one:
        print("\n  entries with |value| > 1 (independent-error model violated, consistent with"
              "\n  shared template/format response -- NOT 'super-correlation'):")
        for a, b, v in above_one:
            print(f"    {a} vs {b}: {v:+.3f}")
    out["d2b"] = {"corr_obs": corr_obs.tolist(), "naive_corrected": naive.tolist(),
                  "entries_above_one": above_one}

    # ---------------- D2c ----------------
    print("\n=== D2c: shared-noise fraction ===")
    delta = (probe - train)
    delta_corr = delta.corr(method="pearson").values
    print("inter-judge correlation of template differences (delta_m):")
    print(fmt_matrix(delta_corr, labels))
    offdiag = [delta_corr[a, b] for a in range(M) for b in range(a + 1, M)]
    c_hat_raw = float(np.mean(offdiag))
    c_hat = float(np.clip(c_hat_raw, 0.0, 1.0))
    print(f"\n  c_hat_raw = {c_hat_raw:+.4f}   c_hat (clipped for the simulator) = {c_hat:.4f}")
    print("  (interpretable as a shared fraction only under the Track E standardized noise model)")
    out["d2c"] = {"delta_corr": delta_corr.tolist(), "c_hat_raw": c_hat_raw, "c_hat": c_hat}

    # ---------------- D2d ----------------
    print("\n=== D2d: shared-noise-adjusted latent correlation ===")
    latent = np.full((M, M), np.nan)
    out_of_range = []
    for a in range(M):
        for b in range(M):
            ra, rb = R[judges[a]], R[judges[b]]
            if ra > 0 and rb > 0:
                val = (corr_obs[a, b] - c_hat * np.sqrt((1 - ra) * (1 - rb))) / np.sqrt(ra * rb)
                latent[a, b] = val
                if a < b and (val < -1 or val > 1):
                    out_of_range.append((labels[a], labels[b], float(val)))
    print(fmt_matrix(latent, labels))
    if out_of_range:
        print("\n  OUT OF RANGE (flagged, not clipped -- simplified model inadequate for these pairs):")
        for a, b, v in out_of_range:
            print(f"    {a} vs {b}: {v:+.3f}")
    out["d2d"] = {"corr_latent_hat": latent.tolist(), "out_of_range": out_of_range}

    # ---------------- D2e ----------------
    print("\n=== D2e: length loading ===")
    len_diff = response_length_diff(int(max(common)) + 1).loc[common]
    x = len_diff.to_numpy(dtype=float)
    resid = {}
    d2e_rows = []
    for j, lab in zip(judges, labels):
        y = train[j].to_numpy(dtype=float)
        A = np.vstack([np.ones_like(x), x]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ coef
        r2 = 1.0 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
        resid[j] = y - pred
        d2e_rows.append({"judge": lab, "intercept": float(coef[0]), "gamma": float(coef[1]),
                         "r_squared": float(r2)})
        print(f"  {lab:16s} gamma={coef[1]:+.5f} per token   R^2={r2:.4f}")
    resid_df = pd.DataFrame(resid)
    resid_corr = resid_df.corr(method="pearson").values
    print("\nresidual (length-removed) inter-judge correlation:")
    print(fmt_matrix(resid_corr, labels))
    print("\ncompare with raw margin correlation above: if raw is moderate but residual collapses"
          "\ntoward zero, the shared structure is primarily length-driven.")
    out["d2e"] = {"per_judge": d2e_rows, "residual_corr": resid_corr.tolist()}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"d2_reliability_{args.tag or 'padded'}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
