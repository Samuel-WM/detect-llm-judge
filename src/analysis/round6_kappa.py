"""NEXT_STEPS_ROUND_6_SAMEDAY_FINAL.md Section B: a training-only early-stopping rule (kappa).

B1  a fixed context set chosen by frozen seed from the TRAINING pairs, never from KL or from any
    attribution outcome
B2  exact conditional forward KL( pi_theta || pi_ref ) over the FULL vocabulary at those contexts,
    at all 16 stored checkpoints of each Round 5 real-RM arm
B3  kappa_R6 = median of training-side KL_bar at the three Round 5 peak-S checkpoints

B3 is explicitly a development-set procedure and is NOT holdout-free: Round 5 holdout S is used
once, retrospectively, to locate the peak checkpoints. Once frozen, kappa is applied to new arms
using TRAINING-SIDE KL ONLY, without consulting that arm's holdout outcome.

Run: .venv/Scripts/python.exe -m src.analysis.round6_kappa --contexts
     .venv/Scripts/python.exe -m src.analysis.round6_kappa --kl
     .venv/Scripts/python.exe -m src.analysis.round6_kappa --freeze
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

from src.common.templating import build_scoring_inputs
from src.common.vram import reclaim
from src.common.gpulock import gpu_lock
from src.data.build_prompts import load_cfg
from src.train.round5_attribution import frozen_training_pairs

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE = REPO_ROOT / "data_cache"
RESULTS = REPO_ROOT / "results"
CKPT = REPO_ROOT / "checkpoints" / "round5"
ARMS = ["RM_OA-deberta", "RM_gpt2-helpful", "RM_gpt2-harmless"]
N_EXAMPLES = 64
N_POSITIONS = 8
CONTEXT_SEED = 20250915
VOCAB_CHUNK = 8


def build_contexts() -> None:
    """B1: 64 training examples, one response each, 8 token positions per response. Selection is by
    frozen seed over indices only -- never by KL, loss, length or any attribution outcome."""
    cfg = load_cfg()
    tok = AutoTokenizer.from_pretrained(cfg["models"]["base_policy"])
    pairs = frozen_training_pairs()
    rng = np.random.default_rng(CONTEXT_SEED)
    ex_idx = sorted(rng.choice(len(pairs), size=N_EXAMPLES, replace=False).tolist())
    which = rng.integers(0, 2, size=N_EXAMPLES)

    items = []
    for k, i in enumerate(ex_idx):
        resp = pairs[i]["response_a" if which[k] == 0 else "response_b"]
        ids, plen = build_scoring_inputs(tok, pairs[i]["prompt"], resp)
        n_resp = ids.shape[0] - plen
        if n_resp < 1:
            continue
        take = min(N_POSITIONS, n_resp)
        pos = sorted(rng.choice(n_resp, size=take, replace=False).tolist())
        items.append({"pair_index": i, "which": int(which[k]), "prompt_len": int(plen),
                      "n_response_tokens": int(n_resp), "positions": pos,
                      "input_ids": ids.tolist()})
    total = sum(len(it["positions"]) for it in items)
    out = {"seed": CONTEXT_SEED, "n_examples": len(items), "n_contexts": total,
           "selection": "frozen-seed index sampling over the 259 frozen training pairs; no KL, "
                        "loss, length or attribution outcome consulted",
           "items": items}
    json.dump(out, open(CACHE / "round6_kappa_contexts.json", "w"))
    print(f"B1: {len(items)} examples, {total} fixed token contexts -> "
          f"data_cache/round6_kappa_contexts.json")


@torch.no_grad()
def kl_for_checkpoint(model, ids_list, pos_list, device="cuda") -> float:
    """Exact conditional forward KL( pi_theta || pi_ref ) over the FULL vocabulary, averaged over
    the fixed contexts. Adapter-on and adapter-off logits come from the identical input tensor."""
    tot, n = 0.0, 0
    for ids, pos in zip(ids_list, pos_list):
        x = ids.unsqueeze(0).to(device)
        on = model(input_ids=x).logits[0]
        with model.disable_adapter():
            off = model(input_ids=x).logits[0]
        rows = [p - 1 for p in pos]          # logits at t-1 predict token t
        for start in range(0, len(rows), VOCAB_CHUNK):
            sel = rows[start:start + VOCAB_CHUNK]
            lp = torch.log_softmax(on[sel].float(), dim=-1)
            lq = torch.log_softmax(off[sel].float(), dim=-1)
            kl = (lp.exp() * (lp - lq)).sum(dim=-1)
            tot += float(kl.sum())
            n += len(sel)
            del lp, lq, kl
        del on, off
    return tot / max(n, 1)


def run_kl(arms: list[str] | None = None, out: str = "round6_b2_kl.csv") -> None:
    with gpu_lock("round6:kl"):
        _run_kl(arms or ARMS, out)


def _run_kl(arms: list[str], out: str) -> None:
    ctx = json.load(open(CACHE / "round6_kappa_contexts.json"))
    cfg = load_cfg()
    ids_list = [torch.tensor(it["input_ids"], dtype=torch.long) for it in ctx["items"]]
    pos_list = [[it["prompt_len"] + p for p in it["positions"]] for it in ctx["items"]]
    print(f"B2: {ctx['n_contexts']} contexts, full vocabulary, exact KL")

    rows = []
    t_start = time.time()
    for arm in arms:
        steps = sorted(int(p.name.split("-")[1]) for p in (CKPT / arm).glob("checkpoint-*"))
        for st in steps:
            base = AutoModelForCausalLM.from_pretrained(
                cfg["models"]["base_policy"], dtype=torch.bfloat16,
                attn_implementation="sdpa").to("cuda")
            model = PeftModel.from_pretrained(base, str(CKPT / arm / f"checkpoint-{st}")).eval()
            t0 = time.time()
            kl = kl_for_checkpoint(model, ids_list, pos_list)
            del model, base
            reclaim()
            rows.append({"arm": arm, "step": st, "KL_bar": kl, "wall_clock_s": time.time() - t0})
            print(f"  {arm} step {st:4d}: KL_bar = {kl:.5f}  ({time.time() - t0:.0f}s)")
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / out, index=False)
    print(f"KL total {time.time() - t_start:.0f}s -> results/{out}")


def freeze_kappa() -> None:
    """B3: locate each arm's Round 5 peak raw primary S (development use of the holdout, declared),
    read the TRAINING-side KL_bar there, and take the median."""
    kl = pd.read_csv(RESULTS / "round6_b2_kl.csv")
    rows = []
    for arm in ARMS:
        att = json.load(open(RESULTS / f"round5_{arm}_attribution.json"))
        pc = att["per_checkpoint"]
        s = [c["raw"]["S_obs"] for c in pc]
        i = int(np.argmax(s))
        peak_step = pc[i]["step"]
        k = kl[(kl.arm == arm) & (kl.step == peak_step)]["KL_bar"]
        rows.append({"arm": arm, "peak_S_step": peak_step, "peak_S": s[i],
                     "KL_bar_at_peak": float(k.iloc[0]) if len(k) else float("nan")})
    df = pd.DataFrame(rows)
    kappa = float(np.median(df["KL_bar_at_peak"]))
    print("=== B3: peak-S checkpoints and their TRAINING-side KL ===")
    print(df.round(5).to_string(index=False))
    print(f"\nkappa_R6 = median(KL_bar at peak-S checkpoints) = {kappa:.6f}")

    sel = []
    for arm in ARMS:
        sub = kl[kl.arm == arm].sort_values("step")
        under = sub[sub.KL_bar <= kappa]
        chosen = int(under.step.iloc[-1]) if len(under) else int(sub.step.iloc[0])
        sel.append({"arm": arm, "kappa_selected_step": chosen,
                    "KL_bar_there": float(sub[sub.step == chosen].KL_bar.iloc[0])})
    seldf = pd.DataFrame(sel)
    print("\n=== kappa applied retrospectively to the Round 5 arms (training-side KL only) ===")
    print(seldf.to_string(index=False))

    json.dump({"kappa_R6": kappa, "rule": "last checkpoint whose training-side KL_bar <= kappa_R6",
               "derivation": "median of training-side KL_bar at the three Round 5 peak-raw-S "
                             "checkpoints; Round 5 holdout S used ONCE, retrospectively, to locate "
                             "those checkpoints -- this is a development-set procedure and is not "
                             "holdout-free",
               "peak_table": rows, "round5_selection": sel,
               "frozen_before": "any Tier 1+ Round 6 arm trains"},
              open(RESULTS / "round6_b3_kappa.json", "w"), indent=1)
    print(f"\nwrote results/round6_b3_kappa.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", action="store_true")
    ap.add_argument("--kl", action="store_true")
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--arms", default=None, help="comma-separated; default = the three Round 5 arms")
    ap.add_argument("--out", default="round6_b2_kl.csv")
    a = ap.parse_args()
    if a.contexts:
        build_contexts()
    elif a.kl:
        run_kl(a.arms.split(",") if a.arms else None, a.out)
    elif a.freeze:
        freeze_kappa()


if __name__ == "__main__":
    main()
