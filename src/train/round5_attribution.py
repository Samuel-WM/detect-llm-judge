"""NEXT_STEPS_ROUND_5_FINAL.md 5: the attribution/DPO experiment.

One arm per label-generating teacher. All arms share the identical frozen training pairs, DPO
configuration, checkpoint cadence and final holdout; only the labels differ.

    RM_<model>   stochastic Bradley-Terry labels from a retained reward model, tau = 2
    FJ           stochastic BT labels from J_perp, the length-orthogonal feature judge
    CTRL         shuffled-label negative control, a permutation of the reference arm's labels

Three phases, run separately so a reward model and the policy never share a process:

    --labels --only <repo>   score the teacher on the frozen training pairs (one process per model)
    --train  --arm <arm>     DPO training with 16 evenly spaced checkpoints
    --score  --arm <arm>     Delta on the fresh final holdout at every checkpoint, plus 5.6 metrics

Run: .venv/Scripts/python.exe -m src.train.round5_attribution --train --arm RM_gpt2-helpful
"""
import argparse
import json
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import math
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from src.common.vram import reclaim
from src.common.gpulock import gpu_lock
from src.data.build_prompts import load_cfg
from src.judges import rm_scorer
from src.judges.rm_harness import CTX, SHORT, load_model
from src.score import delta as delta_mod

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = REPO_ROOT / "data_cache"
RESULTS_DIR = REPO_ROOT / "results"
CKPT_DIR = REPO_ROOT / "checkpoints" / "round5"

TAU = 2.0
EPOCHS = 16
N_CHECKPOINTS = 16
LABEL_SEED = 20250909      # frozen before any label is sampled
CTRL_SEED = 20250910
CTRL_REFERENCE = "gpt2-helpful"   # frozen: the arm whose label vector CTRL permutes


def frozen_training_pairs() -> list[dict]:
    allp = [json.loads(l) for l in open(CACHE_DIR / "response_pairs_train.jsonl")]
    keep = json.load(open(RESULTS_DIR / "rm_trainset_r5.json"))["retained_indices"]
    return [allp[i] for i in keep]


def teacher_margins_train(arm: str) -> np.ndarray:
    if arm in ("FJ", "FJ_PERP"):
        return np.array(json.load(open(CACHE_DIR / "jperp_r5.json"))["d_true_train"])
    short = arm[len("RM_"):]
    repo = [r for r, s in SHORT.items() if s == short][0]
    return np.array(json.load(open(RESULTS_DIR / f"train_margins_{repo.replace('/', '__')}.json"))["d_m"])


def bt_labels(d_train: np.ndarray, seed: int) -> tuple[np.ndarray, dict]:
    """5.3: standardize on the TRAINING pairs only, then sample hard preference labels from
    P(a > b) = sigmoid(tau * d_std). The RNG seed is frozen before sampling."""
    mu, sd = float(d_train.mean()), float(d_train.std())
    z = (d_train - mu) / sd
    p = 1.0 / (1.0 + np.exp(-TAU * z))
    labels = (np.random.default_rng(seed).random(len(p)) < p).astype(int)
    flip = float(np.mean(labels != (d_train > 0).astype(int)))
    # Bayes ceiling: the best achievable agreement with a FRESH draw from this same label model
    bayes = float(np.mean(np.maximum(p, 1 - p)))
    return labels, {"mu_train": mu, "sd_train": sd, "tau": TAU, "seed": seed,
                    "label_balance": float(labels.mean()),
                    "empirical_flip_rate_vs_sign": flip,
                    "mean_abs_standardized_margin": float(np.abs(z).mean()),
                    "bayes_label_ceiling": bayes}


def train_len_diff() -> np.ndarray:
    """len(y1) - len(y2) in base-policy tokens over the frozen training pairs."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(load_cfg()["models"]["base_policy"])
    return np.array([len(tok.encode(p["response_a"])) - len(tok.encode(p["response_b"]))
                     for p in frozen_training_pairs()], dtype=float)


def jshort_transform() -> tuple[float, float]:
    """Frozen affine transform from the N=259 training pairs ONLY, reused unchanged out of sample:
    d_short = -(len_diff - mu_259) / sd_259, so corr(d_short, len_diff) = -1 by construction."""
    x = train_len_diff()
    return float(x.mean()), float(x.std())


def arm_labels(arm: str, n: int) -> tuple[np.ndarray, dict]:
    if arm == "J_SHORT":
        x = train_len_diff()
        mu, sd = jshort_transform()
        d = -(x - mu) / sd
        labels, diag = bt_labels(d, LABEL_SEED)
        diag["label_process"] = f"stochastic BT, tau={TAU}, teacher = -z_259(len_diff)"
        diag["corr_teacher_len_diff_train"] = float(np.corrcoef(d, x)[0, 1])
        diag["frozen_transform"] = {"mu_259": mu, "sd_259": sd}
        return labels, diag
    if arm.startswith("DET_"):
        # 9: deterministic realism arm. label = 1[d_m_train > 0]. Tie handling is fixed BEFORE
        # training: an exact zero margin yields label 0 (response_b preferred), which is what the
        # strict `>` gives. Teacher choice was frozen in the pre-registration.
        d = teacher_margins_train("RM_" + arm[len("DET_"):])
        labels = (d > 0).astype(int)
        return labels, {"label_process": "deterministic 1[d_m_train > 0]",
                        "tie_rule": "exact zero -> label 0, fixed before training",
                        "n_exact_ties": int(np.sum(d == 0)),
                        "label_balance": float(labels.mean())}
    if arm == "CTRL":
        ref, diag = bt_labels(teacher_margins_train(f"RM_{CTRL_REFERENCE}"), LABEL_SEED)
        perm = np.random.default_rng(CTRL_SEED).permutation(len(ref))
        labels = ref[perm]
        diag = {"label_process": f"permutation of the {CTRL_REFERENCE} label vector",
                "ctrl_seed": CTRL_SEED, "reference_arm": CTRL_REFERENCE,
                "label_balance": float(labels.mean()),
                "positives_preserved_exactly": bool(labels.sum() == ref.sum()),
                "agreement_with_reference_labels": float(np.mean(labels == ref))}
        return labels, diag
    labels, diag = bt_labels(teacher_margins_train(arm), LABEL_SEED)
    diag["label_process"] = f"stochastic BT, tau={TAU}"
    return labels, diag


def build_dataset(pairs: list[dict], labels: np.ndarray) -> Dataset:
    return Dataset.from_list([
        {"prompt": p["prompt"],
         "chosen": p["response_a"] if z == 1 else p["response_b"],
         "rejected": p["response_b"] if z == 1 else p["response_a"]}
        for p, z in zip(pairs, labels)])


def do_labels(repo: str) -> None:
    pairs = frozen_training_pairs()
    model, tok = load_model(repo)
    _, _, d = rm_scorer.score_pairs(model, tok, pairs, "cuda", CTX[repo])
    del model
    reclaim()
    p = RESULTS_DIR / f"train_margins_{repo.replace('/', '__')}.json"
    with open(p, "w") as f:
        json.dump({"model": repo, "n": len(pairs), "d_m": d.numpy().tolist()}, f)
    print(f"{SHORT[repo]}: scored {len(pairs)} frozen training pairs -> {p.name}")


def do_train(arm: str) -> None:
    with gpu_lock(f"train:{arm}"):
        _do_train(arm)


def _do_train(arm: str) -> None:
    cfg = load_cfg()
    d = cfg["dpo"]
    pairs = frozen_training_pairs()
    labels, diag = arm_labels(arm, len(pairs))
    ds = build_dataset(pairs, labels)
    steps_per_epoch = math.ceil(len(ds) / d["gradient_accumulation_steps"])
    total_steps = steps_per_epoch * EPOCHS
    save_steps = max(1, total_steps // N_CHECKPOINTS)
    print(f"arm={arm}  N={len(ds)}  steps/epoch={steps_per_epoch}  total={total_steps}  "
          f"save_steps={save_steps}  ({total_steps // save_steps} checkpoints)")
    print(f"labels: {diag}")

    out_dir = CKPT_DIR / arm
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(cfg["models"]["base_policy"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["models"]["base_policy"], dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
    lora = LoraConfig(r=d["lora"]["r"], lora_alpha=d["lora"]["alpha"],
                      lora_dropout=d["lora"]["dropout"],
                      target_modules=d["lora"]["target_modules"], task_type="CAUSAL_LM")
    args = DPOConfig(
        output_dir=str(out_dir), per_device_train_batch_size=d["per_device_train_batch_size"],
        gradient_accumulation_steps=d["gradient_accumulation_steps"], num_train_epochs=EPOCHS,
        learning_rate=d["learning_rate"], lr_scheduler_type=d["lr_scheduler_type"],
        warmup_steps=max(1, round(d["warmup_ratio"] * total_steps)), beta=d["beta"],
        max_length=d["max_length"], truncation_mode="keep_start", bf16=True,
        gradient_checkpointing=True, dataloader_num_workers=0, logging_steps=1,
        save_strategy="steps", save_steps=save_steps, save_total_limit=None,
        report_to=[], seed=cfg["seed"],
        precompute_ref_log_probs=True, precompute_ref_batch_size=1,
        torch_empty_cache_steps=1,
    )
    torch.cuda.set_per_process_memory_fraction(0.90)
    trainer = DPOTrainer(model=model, args=args, train_dataset=ds, processing_class=tok,
                         peft_config=lora)
    t0 = time.time()
    trainer.train()
    hist = [h for h in trainer.state.log_history]
    with open(out_dir / "run_meta.json", "w") as f:
        json.dump({"arm": arm, "labels": labels.tolist(), "label_diag": diag,
                   "steps_per_epoch": steps_per_epoch, "total_steps": total_steps,
                   "save_steps": save_steps, "log_history": hist,
                   "train_wall_clock_s": time.time() - t0,
                   "peak_vram_alloc_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
                   "peak_vram_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 3)}, f)
    del trainer, model
    reclaim()
    print(f"arm={arm} trained in {time.time() - t0:.0f}s -> {out_dir}")


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def two_predictor_fit(delta: np.ndarray, d_true: np.ndarray, len_diff: np.ndarray) -> dict:
    """5.6: Delta = c + gamma_1*d_true + gamma_2*len_diff + eps, BOTH predictors standardized."""
    X = np.column_stack([np.ones_like(delta),
                         (d_true - d_true.mean()) / d_true.std(),
                         (len_diff - len_diff.mean()) / len_diff.std()])
    beta, *_ = np.linalg.lstsq(X, delta, rcond=None)
    resid = delta - X @ beta
    n, k = X.shape
    s2 = float(resid @ resid) / (n - k)
    cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    return {"gamma_1": float(beta[1]), "se_gamma_1": float(se[1]),
            "gamma_2": float(beta[2]), "se_gamma_2": float(se[2]),
            "intercept": float(beta[0])}


def do_score(arm: str) -> None:
    with gpu_lock(f"score:{arm}"):
        _do_score(arm)


def _do_score(arm: str) -> None:
    cfg = load_cfg()
    out_dir = CKPT_DIR / arm
    meta = json.load(open(out_dir / "run_meta.json"))
    probes = [json.loads(l) for l in open(CACHE_DIR / "final_holdout.jsonl")]
    fh = json.load(open(RESULTS_DIR / "final_holdout.json"))
    x = np.array(fh["len_diff"])

    d_true, ceiling, noisy = teacher_on_final(arm, probes)
    ckpts = sorted(out_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    print(f"arm={arm}: {len(ckpts)} checkpoints, {len(probes)} final probes")

    tok = AutoTokenizer.from_pretrained(cfg["models"]["base_policy"])
    # ROUND 6 engineering fix: pi_ref is the frozen base model with the adapter disabled, so its
    # log-probs are identical at every checkpoint. Memoizing them turns 16 reference passes into
    # one. Validated as BITWISE identical to the unoptimized path (max |Delta diff| = 0.000e+00
    # over 128 probes) before use; measured speedup 3.71x on the cached pass.
    ref_cache: dict = {}
    rows = []
    for ck in ckpts:
        step = int(ck.name.split("-")[1])
        base = AutoModelForCausalLM.from_pretrained(
            cfg["models"]["base_policy"], dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
        model = PeftModel.from_pretrained(base, str(ck)).eval()
        t0 = time.time()
        recs = delta_mod.compute_deltas(model, tok, probes, "cuda", ref_cache=ref_cache)
        del model, base
        reclaim()
        delta = np.array([r["delta"] for r in recs])
        row = {"step": step, "epoch": step / meta["steps_per_epoch"],
               "sd_delta": float(delta.std()), "delta": delta.tolist(),
               # ROUND 6: also persist the per-response sums and token counts. Round 5 discarded
               # them, forcing a full re-scoring pass for the Section A decomposition. Additive
               # only -- `delta` and every Round 5 metric are computed exactly as before.
               "s_a": [r["s_a"] for r in recs], "s_b": [r["s_b"] for r in recs],
               "len_a": [r["len_a"] for r in recs], "len_b": [r["len_b"] for r in recs],
               "score_wall_clock_s": time.time() - t0}
        row.update(train_metrics_at(meta, step))
        if d_true is not None:
            row.update(transmission(delta, d_true, x, ceiling, noisy))
        rows.append({k: v for k, v in row.items()})
        print(f"  step {step:4d}  sd(Delta)={row['sd_delta']:.4f}  "
              f"train_acc={row.get('train_acc', float('nan')):.4f}  "
              f"r={row.get('pearson', float('nan')):.4f}  ({row['score_wall_clock_s']:.0f}s)")

    with open(RESULTS_DIR / f"round5_{arm}_trajectory.json", "w") as f:
        json.dump({"arm": arm, "meta": {k: v for k, v in meta.items() if k != "log_history"},
                   "n_final": len(probes), "rows": rows,
                   "d_true_final": None if d_true is None else d_true.tolist(),
                   "bayes_ceiling_final": ceiling}, f)
    print(f"wrote results/round5_{arm}_trajectory.json")


def teacher_on_final(arm: str, probes: list[dict]):
    """d_true on the fresh final holdout, plus a FRESH stochastic label draw and its Bayes
    ceiling (5.6). Standardization uses TRAINING statistics only."""
    if arm == "CTRL":
        return None, None, None
    if arm.startswith("DET_"):
        # deterministic labels: the stochastic-label metrics (fresh noisy agreement, Bayes ceiling,
        # normalized noisy transmission) are undefined, so they are not computed rather than
        # computed against a label model this arm never used.
        short = arm[len("DET_"):]
        repo = [r for r, s in SHORT.items() if s == short][0]
        d = np.array(json.load(open(RESULTS_DIR / f"final_margins_{repo.replace('/', '__')}.json"))["d_m"])
        return d, None, None
    if arm == "J_SHORT":
        fh = json.load(open(RESULTS_DIR / "final_holdout.json"))
        mu, sd = jshort_transform()          # FROZEN training transform, never refit
        d = -(np.array(fh["len_diff"]) - mu) / sd
        dt = -(train_len_diff() - mu) / sd
        mu_t, sd_t = float(dt.mean()), float(dt.std())
        p = 1.0 / (1.0 + np.exp(-TAU * (d - mu_t) / sd_t))
        fresh = (np.random.default_rng(LABEL_SEED + 7).random(len(p)) < p).astype(int)
        return d, float(np.mean(np.maximum(p, 1 - p))), fresh
    if arm in ("FJ", "FJ_PERP"):
        j = json.load(open(CACHE_DIR / "jperp_r5.json"))
        d = np.array(j["reporting_only"]["final_holdout"]["d_true"])
        mu, sd = j["mu_train"], j["sd_train"]
    else:
        short = arm[len("RM_"):]
        repo = [r for r, s in SHORT.items() if s == short][0]
        d = np.array(json.load(open(RESULTS_DIR / f"final_margins_{repo.replace('/', '__')}.json"))["d_m"])
        tr = np.array(json.load(open(RESULTS_DIR / f"train_margins_{repo.replace('/', '__')}.json"))["d_m"])
        mu, sd = float(tr.mean()), float(tr.std())
    p = 1.0 / (1.0 + np.exp(-TAU * (d - mu) / sd))
    fresh = (np.random.default_rng(LABEL_SEED + 7).random(len(p)) < p).astype(int)
    return d, float(np.mean(np.maximum(p, 1 - p))), fresh


def transmission(delta, d_true, x, ceiling, noisy) -> dict:
    r = float(np.corrcoef(delta, d_true)[0, 1])
    r2 = r * r
    n = len(delta)
    k_lat = int(np.sum(np.sign(delta) == np.sign(d_true)))
    lat = k_lat / n
    lo_l, hi_l = wilson_ci(k_lat, n)
    out = {"pearson": r, "r_squared": r2,
           "sigma_D_ratio": float(np.sqrt((1 - r2) / r2)) if r2 > 0 else float("inf"),
           "spearman": float(stats.spearmanr(delta, d_true).statistic),
           "latent_sign_agreement": lat, "latent_sign_ci": [lo_l, hi_l]}
    if noisy is not None:
        k = int(np.sum((delta > 0).astype(int) == noisy))
        a = k / n
        lo, hi = wilson_ci(k, n)
        out.update({"fresh_noisy_label_agreement": a, "fresh_noisy_ci": [lo, hi],
                    "bayes_ceiling": ceiling,
                    "normalized_noisy_transmission": (a - 0.5) / (ceiling - 0.5)})
    out.update(two_predictor_fit(delta, d_true, x))
    return out


def train_metrics_at(meta: dict, step: int) -> dict:
    """train_loss and train_acc averaged over the logging window ending at this checkpoint."""
    lo = step - meta["save_steps"]
    win = [h for h in meta["log_history"] if lo < h.get("step", -1) <= step and "loss" in h]
    if not win:
        return {}
    # trl logs BOTH `rewards/accuracies` (the DPO pairwise preference accuracy -- the train_acc
    # 5.6 asks for) and `mean_token_accuracy` (next-token accuracy, a different quantity that
    # happens to appear first in the log dict). Name the intended key explicitly.
    acc_key = "rewards/accuracies"
    accs = [h[acc_key] for h in win if acc_key in h]
    return {"train_loss": float(np.mean([h["loss"] for h in win])),
            "train_acc": float(np.mean(accs)) if accs else float("nan"),
            "train_token_acc": float(np.mean([h["mean_token_accuracy"] for h in win
                                              if "mean_token_accuracy" in h])) if win else float("nan")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", action="store_true")
    ap.add_argument("--only")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--arm")
    a = ap.parse_args()
    if a.labels:
        do_labels(a.only)
    elif a.train:
        do_train(a.arm)
    elif a.score:
        do_score(a.arm)


if __name__ == "__main__":
    main()
