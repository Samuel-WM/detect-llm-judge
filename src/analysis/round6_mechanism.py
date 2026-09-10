"""NEXT_STEPS_ROUND_6_SAMEDAY_FINAL.md Section A: forensic analysis of the Round 5 length mechanism.

A1 (settled by inspection, recorded here so the convention is unambiguous):

    s(x,y) = sum_t [ log pi_theta(y_t | prefix) - log pi_ref(y_t | prefix) ]     -- a SUM
    Delta  = s(x, y1) - s(x, y2)

There is no beta anywhere in the scoring path (`src/score/delta.py`); the only beta in the project
is DPOConfig's training beta = 0.1. So beta is omitted below, equivalently beta = 1, and the
decomposition identity is exact:

    c1 = s_a / n1,  c2 = s_b / n2
    c_avg = (c1+c2)/2   c_diff = (c1-c2)/2   len_diff = n1-n2   len_sum = n1+n2
    c_avg*len_diff + c_diff*len_sum  ==  c1*n1 - c2*n2  ==  Delta

RETROSPECTIVE / EXPLORATORY: this analyzes already-observed Round 5 outcomes. Nothing here can
revise a Round 5 number, verdict or threshold.

Run: .venv/Scripts/python.exe -m src.analysis.round6_mechanism --benchmark
     .venv/Scripts/python.exe -m src.analysis.round6_mechanism --run --steps 17,51,68,136,204,272
"""
import argparse
import json
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common.vram import reclaim
from src.data.build_prompts import load_cfg
from src.judges.rm_harness import SHORT
from src.score import delta as delta_mod

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE = REPO_ROOT / "data_cache"
RESULTS = REPO_ROOT / "results"
CKPT = REPO_ROOT / "checkpoints" / "round5"


def probes() -> list[dict]:
    return [json.loads(l) for l in open(CACHE / "final_holdout.jsonl")]


def score_components(arm: str, step: int, pairs: list[dict]) -> dict:
    """Re-score one stored checkpoint, keeping the per-response sums and token counts that the
    Round 5 trajectory did not persist. Uses the identical scoring path (fixed_bin, adapter
    on/off on the same padded tensor)."""
    cfg = load_cfg()
    tok = AutoTokenizer.from_pretrained(cfg["models"]["base_policy"])
    base = AutoModelForCausalLM.from_pretrained(
        cfg["models"]["base_policy"], dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
    model = PeftModel.from_pretrained(base, str(CKPT / arm / f"checkpoint-{step}")).eval()
    t0 = time.time()
    recs = delta_mod.compute_deltas(model, tok, pairs, "cuda")
    del model, base
    reclaim()
    return {"arm": arm, "step": step, "n": len(pairs), "wall_clock_s": time.time() - t0,
            "delta": [r["delta"] for r in recs], "s_a": [r["s_a"] for r in recs],
            "s_b": [r["s_b"] for r in recs], "len_a": [r["len_a"] for r in recs],
            "len_b": [r["len_b"] for r in recs]}


def decompose(comp: dict) -> dict:
    s_a, s_b = np.array(comp["s_a"]), np.array(comp["s_b"])
    n1, n2 = np.array(comp["len_a"], float), np.array(comp["len_b"], float)
    delta = np.array(comp["delta"])
    c1, c2 = s_a / n1, s_b / n2
    c_avg, c_diff = (c1 + c2) / 2, (c1 - c2) / 2
    len_diff, len_sum = n1 - n2, n1 + n2
    accum = c_avg * len_diff            # Delta_length_accum
    tokdiff = c_diff * len_sum          # Delta_token_diff
    recon = accum + tokdiff
    c_all = np.concatenate([c1, c2])
    len_all = np.concatenate([n1, n2])
    return {"c1": c1, "c2": c2, "c_all": c_all, "len_all": len_all, "delta": delta,
            "accum": accum, "tokdiff": tokdiff, "len_diff_scorer": len_diff,
            "max_recon_err": float(np.max(np.abs(delta - recon))),
            "rel_recon_err": float(np.max(np.abs(delta - recon)) / (np.abs(delta).max() + 1e-12))}


def a2_row(arm: str, step: int, d: dict, d_true: np.ndarray) -> dict:
    delta, accum, tok = d["delta"], d["accum"], d["tokdiff"]
    return {
        "arm": arm, "step": step,
        "mean_c": float(d["c_all"].mean()), "sd_c": float(d["c_all"].std()),
        "corr_c_resp_len": float(np.corrcoef(d["c_all"], d["len_all"])[0, 1]),
        "sd_Delta": float(delta.std()),
        "sd_len_accum": float(accum.std()), "sd_token_diff": float(tok.std()),
        "corr_Delta_accum": float(np.corrcoef(delta, accum)[0, 1]),
        "corr_Delta_tokdiff": float(np.corrcoef(delta, tok)[0, 1]),
        "corr_accum_dtrue": float(np.corrcoef(accum, d_true)[0, 1]),
        "corr_tokdiff_dtrue": float(np.corrcoef(tok, d_true)[0, 1]),
        "cov_accum_tokdiff": float(np.cov(accum, tok)[0, 1]),
        "corr_accum_tokdiff": float(np.corrcoef(accum, tok)[0, 1]),
        "max_recon_err": d["max_recon_err"], "rel_recon_err": d["rel_recon_err"],
    }


def a3_row(arm: str, step: int, d: dict, len_diff_r5: np.ndarray, gamma_2_r5: float) -> dict:
    """gamma_2_hat_raw = c_bar * sd(len_diff), the coefficient a purely mechanical accumulation
    Delta ~= c_bar * len_diff would produce when regressed on the STANDARDIZED length difference
    -- the same parameterization the Round 5 two-predictor fit used."""
    c_bar = float(d["c_all"].mean())
    pred = c_bar * len_diff_r5
    delta = d["delta"]
    ss_res = float(((delta - pred) ** 2).sum())
    ss_tot = float(((delta - delta.mean()) ** 2).sum())
    g_hat = c_bar * float(len_diff_r5.std())
    return {"arm": arm, "step": step, "c_bar": c_bar,
            "gamma_2_hat_raw": g_hat, "gamma_2_observed_r5": gamma_2_r5,
            "ratio_hat_over_observed": g_hat / gamma_2_r5 if gamma_2_r5 != 0 else float("nan"),
            "gamma_2_hat_over_sdDelta": g_hat / delta.std(),
            "gamma_2_obs_over_sdDelta": gamma_2_r5 / delta.std(),
            "R2_delta_by_cbar_lendiff": 1 - ss_res / ss_tot,
            "corr_delta_pred": float(np.corrcoef(delta, pred)[0, 1])}


def teacher_margin(arm: str) -> np.ndarray:
    short = arm.split("_", 1)[1]
    repo = [r for r, s in SHORT.items() if s == short][0]
    return np.array(json.load(open(RESULTS / f"final_margins_{repo.replace('/', '__')}.json"))["d_m"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--steps", default="17,51,68,136,204,272")
    ap.add_argument("--arms", default="RM_OA-deberta,RM_gpt2-helpful,RM_gpt2-harmless")
    a = ap.parse_args()
    arms = a.arms.split(",")
    steps = [int(s) for s in a.steps.split(",")]
    P = probes()

    if a.benchmark:
        sub = P[:50]
        t0 = time.time()
        comp = score_components(arms[0], 272, sub)
        el = time.time() - t0
        d = decompose(comp)
        per_probe = el / len(sub)
        proj_full = per_probe * len(P) * len(steps) * len(arms)
        print(f"A1.5 benchmark: 50 probes, 1 checkpoint = {el:.1f}s ({per_probe:.3f}s/probe)")
        print(f"  reconstruction max abs err = {d['max_recon_err']:.3e} "
              f"(relative {d['rel_recon_err']:.3e})")
        print(f"  projected A2+A3 = {len(steps)} steps x {len(arms)} arms x {len(P)} probes "
              f"= {proj_full / 60:.1f} min")
        keep = steps if proj_full <= 3600 else [51, 136, 272]
        print(f"  budget rule (<= 60 min): checkpoint set selected = {keep}")
        json.dump({"benchmark_s": el, "n_probes_benchmark": len(sub),
                   "s_per_probe": per_probe, "projected_full_s": proj_full,
                   "projected_full_min": proj_full / 60,
                   "checkpoints_selected": keep,
                   "reason": ("projection <= 60 min, full six-checkpoint schedule kept"
                              if proj_full <= 3600 else
                              "projection > 60 min, reduced to 51/136/272 per the budget rule"),
                   "reconstruction_max_abs_err": d["max_recon_err"]},
                  open(RESULTS / "round6_a15_benchmark.json", "w"), indent=1)
        print(f"wrote {RESULTS / 'round6_a15_benchmark.json'}")
        return

    if not a.run:
        return

    fh = json.load(open(RESULTS / "final_holdout.json"))
    len_diff_r5 = np.array(fh["len_diff"])
    a2_rows, a3_rows, store = [], [], {}
    for arm in arms:
        traj = json.load(open(RESULTS / f"round5_{arm}_trajectory.json"))
        g2 = {r["step"]: r["gamma_2"] for r in traj["rows"]}
        dt = teacher_margin(arm)
        for st in steps:
            comp = score_components(arm, st, P)
            d = decompose(comp)
            a2_rows.append(a2_row(arm, st, d, dt))
            a3_rows.append(a3_row(arm, st, d, len_diff_r5, g2[st]))
            store[f"{arm}@{st}"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                    for k, v in d.items()}
            print(f"  {arm} step {st}: c_bar={d['c_all'].mean():+.5f} "
                  f"sd(Delta)={d['delta'].std():.3f} recon_err={d['max_recon_err']:.2e} "
                  f"({comp['wall_clock_s']:.0f}s)")

    A2 = pd.DataFrame(a2_rows)
    A3 = pd.DataFrame(a3_rows)
    pd.set_option("display.width", 250)
    print("\n=== A2: exact pairwise decomposition ===")
    print(A2.round(4).to_string(index=False))
    print("\n=== A3: does mechanical accumulation explain gamma_2 ===")
    print(A3.round(4).to_string(index=False))
    A2.to_csv(RESULTS / "round6_a2_decomposition.csv", index=False)
    A3.to_csv(RESULTS / "round6_a3_gamma2.csv", index=False)
    json.dump(store, open(RESULTS / "round6_components.json", "w"))
    print(f"\nmax reconstruction error across all cells: {A2['max_recon_err'].max():.3e}")
    print(f"wrote round6_a2_decomposition.csv, round6_a3_gamma2.csv, round6_components.json")


if __name__ == "__main__":
    main()
