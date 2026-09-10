"""NEXT_STEPS_ROUND_2.md Step 0: regression test for the Delta scorer's batch-composition
invariance -- the same class of bug found in the judge-margin scorer (RESULTS.md), audited here
because it would land directly in Delta, the entire estimand, if left unchecked.

Asserts Delta over a fixed set of pairs is invariant to:
  1. shuffling the order of pairs within the batch
  2. changing the batch size / token budget

    report:  max_i | Delta_i^(config A) - Delta_i^(config B) |, as a fraction of sd_i(Delta_i)
    pass:    max deviation < 0.01 * sd(Delta)

Run: .venv/Scripts/python.exe -m tests.test_scorer_invariance
"""
import json
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.build_prompts import load_cfg
from src.score import delta as delta_mod

N_PAIRS = 24


def build_perturbed_lora_model(base_policy: str):
    """A fresh LoRA adapter has B initialized to zero (PEFT default), so adapter-on == adapter-off
    exactly until trained -- useless for testing batch invariance of a REAL difference. Nudges
    the B matrices with small random noise instead of running real training, purely so the
    on/off passes differ nontrivially.
    """
    tokenizer = AutoTokenizer.from_pretrained(base_policy)
    model = AutoModelForCausalLM.from_pretrained(base_policy, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.add_(0.02 * torch.randn_like(param))
    model.eval()
    return model, tokenizer


def load_pairs(n: int) -> list[dict]:
    import pathlib
    path = pathlib.Path(__file__).resolve().parent.parent / "data_cache" / "response_pairs_probe.jsonl"
    assert path.exists(), f"{path} missing -- run src.gen.sample_pairs first"
    rows = [json.loads(l) for l in open(path)]
    return rows[:n]


def run() -> None:
    cfg = load_cfg()
    model, tokenizer = build_perturbed_lora_model(cfg["models"]["base_policy"])
    pairs = load_pairs(N_PAIRS)

    baseline = delta_mod.compute_deltas(model, tokenizer, pairs, "cuda")
    delta_baseline = np.array([r["delta"] for r in baseline])
    sd = float(delta_baseline.std())
    print(f"baseline computed, N={len(pairs)}, sd(Delta)={sd:.4f}")

    # --- 1. shuffled order ---
    rng = np.random.default_rng(0)
    order = rng.permutation(len(pairs))
    shuffled_pairs = [pairs[i] for i in order]
    shuffled = delta_mod.compute_deltas(model, tokenizer, shuffled_pairs, "cuda")
    delta_shuffled_realigned = np.empty(len(pairs))
    for local_i, orig_i in enumerate(order):
        delta_shuffled_realigned[orig_i] = shuffled[local_i]["delta"]
    diff_shuffle = np.abs(delta_baseline - delta_shuffled_realigned)
    frac_shuffle = diff_shuffle.max() / sd if sd > 0 else float("nan")
    print(f"shuffled-order max|diff|={diff_shuffle.max():.4e}  as frac of sd={frac_shuffle:.4f}  "
          f"pass={frac_shuffle < 0.01}")

    # --- 2. different batch size / token budget ---
    orig_max_rows, orig_budget = delta_mod.MAX_BATCH_ROWS, delta_mod.TOKEN_BUDGET
    delta_mod.MAX_BATCH_ROWS = 2
    delta_mod.TOKEN_BUDGET = 1024
    smaller_batch = delta_mod.compute_deltas(model, tokenizer, pairs, "cuda")
    delta_mod.MAX_BATCH_ROWS, delta_mod.TOKEN_BUDGET = orig_max_rows, orig_budget
    delta_small = np.array([r["delta"] for r in smaller_batch])
    diff_batch = np.abs(delta_baseline - delta_small)
    frac_batch = diff_batch.max() / sd if sd > 0 else float("nan")
    print(f"different batch/token-budget max|diff|={diff_batch.max():.4e}  as frac of sd={frac_batch:.4f}  "
          f"pass={frac_batch < 0.01}")

    overall_pass = frac_shuffle < 0.01 and frac_batch < 0.01
    print(f"\ntests/test_scorer_invariance.py: {'PASS' if overall_pass else 'FAIL'}")
    if not overall_pass:
        raise AssertionError(
            f"scorer is not batch-composition invariant: shuffle_frac={frac_shuffle:.4f}, "
            f"batch_frac={frac_batch:.4f} (threshold 0.01)"
        )
    return {"sd_delta": sd, "shuffle_frac": frac_shuffle, "batch_frac": frac_batch, "pass": overall_pass}


if __name__ == "__main__":
    run()
