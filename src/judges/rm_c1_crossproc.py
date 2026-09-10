"""NEXT_STEPS_ROUND_5_FINAL.md 2.4: cross-process determinism test (criterion C1).

C1 requires a candidate's reward margins to be bitwise reproducible across two FRESH, independent
processes on identical inputs. The Round 4 H2 gate only tested within-process determinism, which is
strictly weaker: every number in this project is assembled from multiple processes, because one
process per model was forced by the InternLM2 all-NaN artifact.

Bitwise equality is the criterion. An approximate correlation threshold may not be substituted for
it after seeing the result (2.4).

Run: .venv/Scripts/python.exe -m src.judges.rm_c1_crossproc --only <repo> --run 1
     .venv/Scripts/python.exe -m src.judges.rm_c1_crossproc --only <repo> --run 2
     .venv/Scripts/python.exe -m src.judges.rm_c1_crossproc --combine
"""
import argparse
import json
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import time
from pathlib import Path

import numpy as np

from src.common.vram import reclaim
from src.judges import rm_scorer
from src.judges.rm_harness import CTX, INTERNLM, SHORT, load_model

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = REPO_ROOT / "data_cache"
RESULTS_DIR = REPO_ROOT / "results"


def calibration_probes() -> tuple[list[dict], list[int]]:
    """The identical 487 retained calibration probes used throughout the reward-model stage."""
    pairs = [json.loads(l) for l in open(CACHE_DIR / "response_pairs_probe.jsonl")]
    idxs = json.load(open(RESULTS_DIR / "rm_probe_filter.json"))["retained_indices"]
    return [pairs[i] for i in idxs], idxs


def score_once(repo: str, run: int) -> Path:
    assert repo != INTERNLM, (
        "InternLM2 is marked C1 FAIL on already-reproduced evidence (2.3); re-testing it here "
        "would be an attempt to rescue it, which this round prohibits.")
    sub, idxs = calibration_probes()
    model, tok = load_model(repo)
    t0 = time.time()
    _, _, d_m = rm_scorer.score_pairs(model, tok, sub, "cuda", CTX[repo])
    del model
    reclaim()
    out = {"model": repo, "run": run, "n": len(sub), "probe_indices": idxs,
           "d_m": d_m.numpy().tolist(), "pid": os.getpid(), "wall_clock_s": time.time() - t0}
    p = RESULTS_DIR / f"rm_c1_{repo.replace('/', '__')}_run{run}.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(out, f)
    print(f"{SHORT[repo]} run{run}: scored {len(sub)} probes in {out['wall_clock_s']:.0f}s "
          f"(pid {out['pid']}) -> {p.name}")
    return p


def compare(repo: str) -> dict | None:
    p1 = RESULTS_DIR / f"rm_c1_{repo.replace('/', '__')}_run1.json"
    p2 = RESULTS_DIR / f"rm_c1_{repo.replace('/', '__')}_run2.json"
    if not (p1.exists() and p2.exists()):
        return None
    r1, r2 = json.load(open(p1)), json.load(open(p2))
    assert r1["probe_indices"] == r2["probe_indices"], "runs scored different probes"
    assert r1["pid"] != r2["pid"], "both runs share a pid: not independent processes"
    a, b = np.array(r1["d_m"]), np.array(r2["d_m"])
    d = np.abs(a - b)
    return {
        "model": repo, "n": int(d.size), "pids": [r1["pid"], r2["pid"]],
        "frac_bitwise_equal": float((d == 0).mean()),
        "max_abs_diff": float(d.max()),
        "median_abs_diff": float(np.median(d)),
        "median_diff_over_margin_sd": float(np.median(d) / a.std()),
        "pearson_between_processes": float(np.corrcoef(a, b)[0, 1]),
        "C1_cross_process_pass": bool(d.max() == 0.0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    ap.add_argument("--run", type=int, choices=[1, 2])
    ap.add_argument("--combine", action="store_true")
    args = ap.parse_args()

    if args.combine:
        out = {}
        for repo in [r for r in CTX if r != INTERNLM]:
            c = compare(repo)
            if c is None:
                continue
            out[SHORT[repo]] = c
            print(f"{SHORT[repo]}: bitwise-equal {c['frac_bitwise_equal']:.4f}  "
                  f"max {c['max_abs_diff']:.6g}  median {c['median_abs_diff']:.6g}  "
                  f"({c['median_diff_over_margin_sd']:.2%} of sd)  "
                  f"r={c['pearson_between_processes']:.6f}  pids={c['pids']}  "
                  f"-> C1 cross-process {'PASS' if c['C1_cross_process_pass'] else 'FAIL'}")
        with open(RESULTS_DIR / "rm_c1_crossproc.json", "w") as f:
            json.dump(out, f, indent=1)
        print(f"wrote {RESULTS_DIR / 'rm_c1_crossproc.json'}")
        return

    score_once(args.only, args.run)


if __name__ == "__main__":
    main()
