"""NEXT_STEPS_ROUND_5_FINAL.md 2.2-2.7: calibration-stage retention C1-C5, candidate geometry,
and the frozen calibration transforms consumed out-of-sample in 4.5.

C1  deterministic within-process AND bitwise reproducible across two fresh processes
C2  non-degenerate margins (existing H1 criterion, no new sd threshold invented here)
C3  harness correctness (H3 batch-invariance, H4 margin identity, H5 input reaches model)
C4  not length-dominated: R^2 of d_m on len_diff < 0.50
C5  POOL-level: min pairwise RAW psi_eff over the C1-C4-passing set S >= 45 degrees

C5 is evaluated on S as a whole. If it fails, the run stops -- the closest pair is not broken up
and no subset search is performed (2.6).

Run: .venv/Scripts/python.exe -m src.judges.rm_calibration_r5
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from src.data.build_prompts import load_cfg
from src.judges.rm_harness import INTERNLM, POOL, SHORT

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = REPO_ROOT / "data_cache"
RESULTS_DIR = REPO_ROOT / "results"

C4_R2_MAX = 0.50
C5_MIN_PSI_DEG = 45.0


def len_diff_calibration(idxs: list[int]) -> np.ndarray:
    """len(y) - len(y') in base-policy tokens -- the standing project convention."""
    tok = AutoTokenizer.from_pretrained(load_cfg()["models"]["base_policy"])
    pairs = [json.loads(l) for l in open(CACHE_DIR / "response_pairs_probe.jsonl")]
    return np.array([len(tok.encode(pairs[i]["response_a"])) - len(tok.encode(pairs[i]["response_b"]))
                     for i in idxs], dtype=float)


def ols_length(d: np.ndarray, x: np.ndarray) -> dict:
    """d = c + gamma*x + eps. Returns coefficients, gamma's SE, R^2 and the residuals."""
    n = len(d)
    xbar, dbar = x.mean(), d.mean()
    sxx = float(((x - xbar) ** 2).sum())
    gamma = float(((x - xbar) * (d - dbar)).sum() / sxx)
    c = float(dbar - gamma * xbar)
    resid = d - (c + gamma * x)
    sse = float((resid ** 2).sum())
    sst = float(((d - dbar) ** 2).sum())
    sigma2 = sse / (n - 2)
    return {"c": c, "gamma": gamma, "se_gamma": float(np.sqrt(sigma2 / sxx)),
            "r_squared": float(1 - sse / sst), "resid": resid}


def psi_matrix(D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    C = np.corrcoef(D)
    return C, np.degrees(np.arccos(np.clip(C, -1, 1)))


def show(mat: np.ndarray, names: list[str], nd: int = 3) -> str:
    return pd.DataFrame(mat, index=names, columns=names).round(nd).to_string()


def main() -> None:
    idxs = json.load(open(RESULTS_DIR / "rm_probe_filter.json"))["retained_indices"]
    x = len_diff_calibration(idxs)
    cross = json.load(open(RESULTS_DIR / "rm_c1_crossproc.json"))

    rows, margins, transforms = [], {}, {}
    for repo in POOL:
        short = SHORT[repo]
        h = json.load(open(RESULTS_DIR / f"rm_harness_{repo.replace('/', '__')}.json"))["result"]

        # margins: prefer the fresh C1 run1 scores; fall back to the harness run for InternLM2,
        # which is not re-scored in this round (2.3).
        p_c1 = RESULTS_DIR / f"rm_c1_{repo.replace('/', '__')}_run1.json"
        if p_c1.exists():
            d = np.array(json.load(open(p_c1))["d_m"])
            assert json.load(open(p_c1))["probe_indices"] == idxs
            drift = float(np.max(np.abs(d - np.array(h["d_m"]))))
        else:
            d = np.array(h["d_m"])
            drift = None
        margins[short] = d

        c1_within = bool(h["H2"]["pass"])
        c1_cross = bool(cross[short]["C1_cross_process_pass"]) if short in cross else False
        c1 = c1_within and c1_cross
        c2 = bool(h["H1"]["pass"]) and np.isfinite(d.std()) and d.std() > 0
        c3 = all(bool(h[g]["pass"]) for g in ("H3", "H4", "H5"))
        fit = ols_length(d, x)
        c4 = bool(fit["r_squared"] < C4_R2_MAX)

        transforms[short] = {"mu_cal": float(d.mean()), "sd_cal": float(d.std(ddof=1)),
                             "c_len": fit["c"], "gamma_len": fit["gamma"],
                             "resid_mean_cal": float(fit["resid"].mean()),
                             "resid_sd_cal": float(fit["resid"].std(ddof=1))}
        rows.append({
            "model": short,
            "C1_within": "PASS" if c1_within else "FAIL",
            "C1_cross": ("PASS" if c1_cross else "FAIL") if short in cross else "FAIL (not re-tested; 2.3)",
            "C2": "PASS" if c2 else "FAIL",
            "C3": "PASS" if c3 else "FAIL",
            "C4_gamma": round(fit["gamma"], 5), "C4_se": round(fit["se_gamma"], 5),
            "C4_R2_len": round(fit["r_squared"], 4), "C4": "PASS" if c4 else "FAIL",
            "retained_C1_C4": bool(c1 and c2 and c3 and c4),
            "_resid": fit["resid"], "_drift_vs_harness": drift,
        })

    d1 = {SHORT[r]: json.load(open(RESULTS_DIR / f"rm_d1_{r.replace('/', '__')}.json")) for r in POOL}
    tbl = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
    tbl["D1_full_cov"] = [round(d1[m]["full_set"]["accuracy"], 4) for m in tbl["model"]]
    tbl["D1_lg_cov"] = [round(d1[m]["large_gap"]["accuracy"], 4) for m in tbl["model"]]

    print("=== 2.7 model-wise calibration retention (D1 columns are REPORTED COVARIATES, not gates) ===")
    print(tbl.to_string(index=False))
    for r in rows:
        if r["_drift_vs_harness"] is not None:
            print(f"  consistency {r['model']}: max |C1 run1 - Round 4 harness| = {r['_drift_vs_harness']:.6g}")

    S = [r["model"] for r in rows if r["retained_C1_C4"]]
    print(f"\nS (C1-C4 passing) = {S}   |S| = {len(S)}")
    out = {"C4_threshold": C4_R2_MAX, "C5_min_psi_deg": C5_MIN_PSI_DEG,
           "table": tbl.to_dict(orient="records"), "S": S,
           "calibration_transforms": transforms, "len_diff_convention": "base-policy tokens",
           "n_calibration_probes": len(idxs)}

    if len(S) < 2:
        out["gate"] = "STOP: |S| < 2"
        print("STOP: fewer than two candidates pass C1-C4.")
    else:
        Draw = np.vstack([margins[m] for m in S])
        Dres = np.vstack([r["_resid"] for r in rows if r["model"] in S])
        Craw, Praw = psi_matrix(Draw)
        Cres, Pres = psi_matrix(Dres)
        iu = np.triu_indices(len(S), k=1)
        min_raw, min_res = float(Praw[iu].min()), float(Pres[iu].min())
        pair_raw = f"{S[iu[0][Praw[iu].argmin()]]} vs {S[iu[1][Praw[iu].argmin()]]}"
        pair_res = f"{S[iu[0][Pres[iu].argmin()]]} vs {S[iu[1][Pres[iu].argmin()]]}"
        c5 = bool(min_raw >= C5_MIN_PSI_DEG)

        print(f"\n=== 2.5 RAW calibration geometry (n={len(idxs)}) ===")
        print(show(Craw, S)); print(); print(show(Praw, S, 2))
        print(f"\n=== 2.5 LENGTH-RESIDUALIZED calibration geometry ===")
        print(show(Cres, S)); print(); print(show(Pres, S, 2))
        print(f"\nmin RAW pairwise psi_eff = {min_raw:.2f} deg ({pair_raw})")
        print(f"min RESIDUALIZED pairwise psi_eff = {min_res:.2f} deg ({pair_res})  [diagnostic only]")
        print(f"\nC5 (pool-level, min raw psi_eff >= {C5_MIN_PSI_DEG}): {'PASS' if c5 else 'FAIL'}")
        print(f"M = {len(S)}   chance = 1/M = {1 / len(S):.4f}")
        out.update({"M": len(S), "chance": 1 / len(S),
                    "corr_raw": Craw.tolist(), "psi_raw": Praw.tolist(),
                    "corr_resid": Cres.tolist(), "psi_resid": Pres.tolist(),
                    "min_raw_psi_deg": min_raw, "min_raw_pair": pair_raw,
                    "min_resid_psi_deg": min_res, "min_resid_pair": pair_res,
                    "C5_pass": c5, "gate": "PROCEED" if c5 else "STOP: C5 failed"})

    with open(RESULTS_DIR / "rm_calibration_r5.json", "w") as f:
        json.dump(out, f, indent=1)
    np.save(RESULTS_DIR / "rm_calibration_len_diff.npy", x)
    print(f"\nwrote {RESULTS_DIR / 'rm_calibration_r5.json'}")


if __name__ == "__main__":
    main()
