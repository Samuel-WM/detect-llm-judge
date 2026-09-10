"""Section B.4: mechanical application of the pre-registered retention criterion.

The criterion is applied exactly as written in configs/base.yaml before any reward-model
scoring. Nothing here is tuned on the results.

    retain(m) = H1..H5 all pass  AND  D1 PRIMARY (large-gap) passes
    D1 pass   = |acc - 0.5| >= 0.10  AND  two-sided exact binomial p < 0.001

H6 is a behavioral diagnostic and is deliberately NOT part of the criterion.
"""
import json
from pathlib import Path

import pandas as pd

from src.data.build_prompts import load_cfg
from src.judges.rm_harness import POOL, SHORT

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = REPO_ROOT / "results"
GATES = ["H1", "H2", "H3", "H4", "H5"]


def main() -> None:
    rule = load_cfg()["reward_model_stage"]["d1_pre_registration"]
    crit = (f"H1-H5 all pass AND D1 PRIMARY (large-gap): "
            f"|acc-0.5| >= {rule['min_abs_distance_from_chance']} AND p < {rule['binomial_p_threshold']}")
    rows = []
    for repo in POOL:
        h = json.load(open(RESULTS / f"rm_harness_{repo.replace('/', '__')}.json"))["result"]
        d = json.load(open(RESULTS / f"rm_d1_{repo.replace('/', '__')}.json"))
        gate_flags = {g: bool(h[g]["pass"]) for g in GATES}
        h_all = all(gate_flags.values())
        lg, fs = d["large_gap"], d["full_set"]
        rows.append({
            "model": SHORT[repo],
            "H1-H5": "PASS" if h_all else "FAIL(" + ",".join(g for g, v in gate_flags.items() if not v) + ")",
            "D1 full": f"{fs['accuracy']:.4f} (|d|={fs['abs_distance']:.4f}, p={fs['p_value']:.2e}, "
                       f"n={fs['n_scored']}) {'PASS' if fs['passes'] else 'FAIL'}",
            "D1 large-gap": f"{lg['accuracy']:.4f} (|d|={lg['abs_distance']:.4f}, p={lg['p_value']:.2e}, "
                            f"n={lg['n_scored']}) {'PASS' if lg['passes'] else 'FAIL'}",
            "retention criterion": crit,
            "retained?": "YES" if (h_all and lg["passes"]) else "NO",
        })
    df = pd.DataFrame(rows)
    n_ret = int((df["retained?"] == "YES").sum())
    print(df.to_string(index=False))
    print(f"\nretained: {n_ret} of {len(df)} -> "
          f"{'>=2 survivors, Section A may proceed' if n_ret >= 2 else 'FEWER THAN TWO SURVIVORS: STOP after Section B'}")
    df.to_csv(RESULTS / "rm_retention.csv", index=False)
    json.dump({"criterion": crit, "n_retained": n_ret, "n_candidates": len(df),
               "retained_models": df.loc[df["retained?"] == "YES", "model"].tolist(),
               "rows": rows,
               "gate_outcome": "BLOCKED: fewer than two models survive" if n_ret < 2 else "PROCEED"},
              open(RESULTS / "rm_retention.json", "w"), indent=1)
    print(f"wrote {RESULTS / 'rm_retention.csv'} and rm_retention.json")


if __name__ == "__main__":
    main()
