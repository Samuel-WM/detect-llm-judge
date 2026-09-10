"""NEXT_STEPS_ROUND_6_SAMEDAY_FINAL.md A4: the Delta_norm diagnostic.

    Delta_norm = c1 - c2          (beta omitted; A1 established there is no beta in Delta)

Length-normalized: the per-token mean log-ratio difference rather than the summed one. Re-runs the
full attribution stack at the FINAL checkpoint (272) only, raw and length-residualized, using the
frozen calibration transforms.

CAVEAT, stated once and plainly: `Delta_norm` breaks the sequence-level KL-regularized equilibrium
identity that the summed statistic satisfies. The density-ratio derivation applies to the summed
log-ratio, not to its per-token mean. This is a MECHANISM DIAGNOSTIC. It cannot reverse the Round 5
verdict and is not reported as doing so.

RETROSPECTIVE / EXPLORATORY.

Run: .venv/Scripts/python.exe -m src.analysis.round6_deltanorm
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.round5_attribution import (P_THRESHOLD, analyze_checkpoint, bootstrap,
                                             permutation_test)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = REPO_ROOT / "results"
ARMS = ["RM_OA-deberta", "RM_gpt2-helpful", "RM_gpt2-harmless"]
FINAL_STEP = 272


def main() -> None:
    fd = json.load(open(RESULTS / "final_design_matrix.json"))
    S, M = fd["S"], len(fd["S"])
    D_raw = np.array(fd["D_std"])
    D_res = np.array(fd["D_resid_std"])
    comp = json.load(open(RESULTS / "round6_components.json"))

    rows, out = [], {}
    for arm in ARMS:
        c = comp[f"{arm}@{FINAL_STEP}"]
        dn = np.array(c["c1"]) - np.array(c["c2"])
        delta_sum = np.array(c["delta"])
        true_idx = S.index(arm.split("_", 1)[1])
        rec = {"arm": arm, "true": S[true_idx],
               "corr_deltanorm_deltasum": float(np.corrcoef(dn, delta_sum)[0, 1]),
               "sd_delta_norm": float(dn.std())}
        for tag, D in [("raw", D_raw), ("resid", D_res)]:
            b = analyze_checkpoint(dn, D, true_idx, M)
            b["permutation"] = permutation_test(dn, D, true_idx, b["S_obs"])
            b["bootstrap"] = bootstrap(dn, D, true_idx)
            b["primary_success"] = bool(b["nnls_top1_correct"] and b["S_obs"] > 0
                                        and b["permutation"]["p_perm"] < P_THRESHOLD)
            out[f"{arm}_{tag}"] = b
            rec[f"{tag}_top1"] = S[b["nnls_top1_idx"]]
            rec[f"{tag}_S"] = b["S_obs"]
            rec[f"{tag}_p_perm"] = b["permutation"]["p_perm"]
            rec[f"{tag}_boot"] = b["bootstrap"]["top1_support"]
            rec[f"{tag}_maxLambda"] = max(b["Lambda"])
            rec[f"{tag}_verdict"] = "SUCCESS" if b["primary_success"] else "FAIL"
        rows.append(rec)
        print(f"  {arm}: raw top1={rec['raw_top1']} S={rec['raw_S']:+.4f} p={rec['raw_p_perm']:.5f} "
              f"-> {rec['raw_verdict']}   |   resid top1={rec['resid_top1']} "
              f"S={rec['resid_S']:+.4f} p={rec['resid_p_perm']:.5f} -> {rec['resid_verdict']}")

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print("\n=== A4: Delta_norm = c1 - c2, final checkpoint 272 ===")
    print(df.round(4).to_string(index=False))
    df.to_csv(RESULTS / "round6_a4_deltanorm.csv", index=False)
    json.dump({"caveat": "Delta_norm breaks the sequence-level KL-regularized equilibrium identity "
                         "the summed statistic satisfies; mechanism diagnostic only, cannot "
                         "reverse the Round 5 verdict",
               "final_step": FINAL_STEP, "rows": rows, "detail": out},
              open(RESULTS / "round6_a4_deltanorm.json", "w"))
    print("wrote round6_a4_deltanorm.csv / .json")


if __name__ == "__main__":
    main()
