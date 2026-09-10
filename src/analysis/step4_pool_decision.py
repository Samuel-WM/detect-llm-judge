"""Step 4 of NEXT_STEPS_FINAL.md: the pool-decision table.

One row per judge, columns exactly as specified. Criteria are read from configs/base.yaml
(`judge_validity`), which was written before D1/D2 were run, so nothing here can be tuned after
seeing the numbers.

    hard gates:      D3a antisymmetry passes AND D3b exact-duplicate null passes
    retention:       D1 external-structure rule passes AND R_m >= min_reliability_R
    diagnostics:     formatting / paraphrase SNRs are NOT exclusion criteria

Run: .venv/Scripts/python.exe -m src.analysis.step4_pool_decision
"""
import json
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "results"

SHORT = {
    "Qwen/Qwen2.5-1.5B-Instruct": "Qwen2.5-1.5B",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": "SmolLM2-1.7B",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "TinyLlama-1.1B",
    "meta-llama/Llama-3.2-1B-Instruct": "Llama-3.2-1B",
    "google/gemma-3-1b-it": "gemma-3-1b",
}


def load_json(name):
    p = RESULTS_DIR / name
    return json.load(open(p)) if p.exists() else None


def main() -> None:
    cfg = yaml.safe_load(open(REPO_ROOT / "configs" / "base.yaml"))
    jv = cfg["judge_validity"]
    min_R = jv["d2_reliability"]["min_reliability_R"]
    min_pool = jv["min_pool_size_before_ladder"]

    d3 = load_json("validity_d3.json")
    d1 = load_json("d1_external.json")
    d2 = load_json("d2_reliability_padfree.json") or load_json("d2_reliability_fixedbin.json")

    if d3 is None or d1 is None or d2 is None:
        missing = [n for n, v in [("validity_d3.json", d3), ("d1_external.json", d1),
                                  ("d2_reliability_*.json", d2)] if v is None]
        print(f"MISSING inputs: {missing}")
        print("Step 4 requires Steps 1-3 all reported; not producing a partial pool decision.")
        return

    d3_by_judge = {r["judge"]: r for r in d3["results"]}
    d1_by_judge = {r["judge"]: r for r in d1["results"]}
    r_by_judge = {}
    for row, judge in zip(d2["d2a"], d2["judges"]):
        r_by_judge[judge] = row
    len_by_judge = {}
    for row, judge in zip(d2["d2e"]["per_judge"], d2["judges"]):
        len_by_judge[judge] = row

    rows = []
    for judge in sorted(set(d3_by_judge) | set(d1_by_judge) | set(r_by_judge)):
        g3 = d3_by_judge.get(judge, {})
        g1 = d1_by_judge.get(judge, {})
        rm = r_by_judge.get(judge, {})
        ln = len_by_judge.get(judge, {})

        antisym = g3.get("d3a", {}).get("max_abs")
        dup = g3.get("d3b0", {}).get("max_abs")
        gates_ok = bool(g3.get("d3a", {}).get("passes") and g3.get("d3b0", {}).get("passes"))

        d1_pass = bool(g1.get("passes")) if "passes" in g1 else None
        R = rm.get("R_pearson")
        r_pass = (R is not None and R >= min_R)

        if not gates_ok:
            verdict = "BLOCKED (hard gate)"
        elif d1_pass and r_pass:
            verdict = "RETAIN"
        else:
            reasons = []
            if not d1_pass:
                reasons.append("D1")
            if not r_pass:
                reasons.append(f"R<{min_R}")
            verdict = "DROP (" + ",".join(reasons) + ")"

        rows.append({
            "judge": SHORT.get(judge, judge),
            "D1_acc": g1.get("accuracy"),
            "|acc-0.5|": g1.get("abs_distance_from_chance"),
            "binom_p": g1.get("binomial_p"),
            "D2_R_m": R,
            "len_R2": ln.get("r_squared"),
            "D3a_max": antisym,
            "D3b_dup_max": dup,
            "SNR_format": g3.get("d3b1_format", {}).get("snr_format"),
            "SNR_para": g3.get("d3b2_paraphrase", {}).get("snr_paraphrase"),
            "verdict": verdict,
        })

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 240)
    print("=== Step 4: pool decision ===")
    print(f"criteria (pre-registered): D1 |acc-0.5| >= {jv['d1_external_structure']['min_abs_distance_from_chance']} "
          f"and p < {jv['d1_external_structure']['binomial_p_threshold']}; R_m >= {min_R}; "
          f"both hard gates must pass")
    print(df.to_string(index=False))

    retained = df[df["verdict"] == "RETAIN"]
    print(f"\nretained: {len(retained)} judge(s): {list(retained['judge'])}")
    if len(retained) < min_pool:
        print(f"FEWER THAN {min_pool} JUDGES SURVIVE -> Step 6 pool ladder applies.")
    else:
        print(f"pool size {len(retained)} >= {min_pool}; Step 6 not triggered.")

    df.to_csv(RESULTS_DIR / "step4_pool_decision.csv", index=False)
    print(f"\nwrote {RESULTS_DIR / 'step4_pool_decision.csv'}")


if __name__ == "__main__":
    main()
