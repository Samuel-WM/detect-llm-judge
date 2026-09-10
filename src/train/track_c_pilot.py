"""Section A of NEXT_STEPS_ROUND_3_FINAL.md: the Track C pilot.

Arms (all share ONE generation of 300 train prompt pairs; only labels differ):

    A_tau2    feature judge, stochastic BT labels, tau=2       PRIMARY (magnitude)
    A_det     feature judge, deterministic sign(d_true)        designated contrast
    B_qwen    Qwen2.5-1.5B observed margins, binary labels     PRIMARY (rank only)
  + a matched one-epoch shuffled-label control for every arm.

Key interpretation rules baked in here so they can't drift (0.2, 0.3, 1.2, 1.3):
  - `sigma_D_ratio = sqrt((1-R2)/R2)` is the Phase-0c-compatible quantity Track E consumes.
    `sigma_D_total = sqrt(1-R2)` is descriptive only.
  - Only the STOCHASTIC feature-judge arm's sigma_D_ratio is Track-E eligible. The deterministic
    arm's and Arm B's are recorded but explicitly not eligible, because deterministic binary
    labels do not identify continuous margin magnitude.
  - Arm B's headline is rank-level (sign agreement, Spearman); its regression is exploratory.
  - Probe margins reuse TRAINING standardization constants, never renormalized on probes.

Run: .venv/Scripts/python.exe -m src.train.track_c_pilot --arms A_tau2 A_det B_qwen
"""
import argparse
import json
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig
from scipy.stats import pearsonr, spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import DPOConfig, DPOTrainer

from src.common.vram import free_vram_gb, reclaim
from src.data.build_prompts import load_cfg
from src.score import delta as delta_mod
from tests.test_scorer_invariance import run as run_scorer_invariance_check

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = REPO_ROOT / "data_cache"
RESULTS_DIR = REPO_ROOT / "results"
LOGS_DIR = REPO_ROOT / "logs"
CKPT_DIR = REPO_ROOT / "checkpoints"


class StepTiming(TrainerCallback):
    def __init__(self, path: Path, tag: str):
        self.path, self.tag, self.t_last = path, tag, None

    def on_step_end(self, args, state, control, **kwargs):
        now = time.time()
        if self.t_last is not None:
            rec = {"arm": self.tag, "step": state.global_step,
                   "seconds": round(now - self.t_last, 4),
                   "mem_reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 3),
                   "loss": state.log_history[-1].get("loss") if state.log_history else None}
            with open(self.path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        self.t_last = now
        return control


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def make_labels(arm: str, fj: dict, qwen_d: np.ndarray | None, tau: float, seed: int):
    """Returns (labels, diagnostics). label=1 means response_a preferred."""
    rng = np.random.default_rng(seed)
    diag = {}
    if arm.startswith("A_"):
        d_std_train = np.array(fj["d_std_train"])
        if arm == "A_det":
            labels = (d_std_train > 0).astype(int)
            diag["label_process"] = "deterministic sign(d_true)"
        else:
            p = sigmoid(tau * d_std_train)
            labels = (rng.random(len(p)) < p).astype(int)
            diag["label_process"] = f"stochastic BT, tau={tau}"
            diag["tau"] = tau
            diag["flip_rate"] = float(np.mean(labels != (d_std_train > 0).astype(int)))
    elif arm == "B_qwen":
        labels = (qwen_d > 0).astype(int)
        diag["label_process"] = "deterministic sign(d_Qwen_obs)"
    else:
        raise ValueError(arm)
    return labels, diag


def build_dataset(train_pairs, labels, shuffle: bool, seed: int) -> Dataset:
    rng = np.random.default_rng(seed + 999)
    recs = []
    for p, z in zip(train_pairs, labels):
        z_eff = int(rng.integers(2)) if shuffle else int(z)
        chosen = p["response_a"] if z_eff == 1 else p["response_b"]
        rejected = p["response_b"] if z_eff == 1 else p["response_a"]
        recs.append({"prompt": p["prompt"], "chosen": chosen, "rejected": rejected})
    return Dataset.from_list(recs)


def train_one(cfg, ds: Dataset, out_dir: Path, tag: str, seed: int) -> dict:
    d = cfg["dpo"]
    tokenizer = AutoTokenizer.from_pretrained(cfg["models"]["base_policy"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["models"]["base_policy"], dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
    lora = LoraConfig(r=d["lora"]["r"], lora_alpha=d["lora"]["alpha"], lora_dropout=d["lora"]["dropout"],
                      target_modules=d["lora"]["target_modules"], task_type="CAUSAL_LM")
    # trl 1.12.0's DPOConfig has warmup_steps, not warmup_ratio. Section 4.4 fixes "5% warmup";
    # with 300 pairs / (batch 1 x accum 16) = ~19 optimizer steps, 5% is 0.95 -> 1 step, so the
    # fixed value's intent is preserved exactly. Only the parameterization changed, not the value.
    total_steps = max(1, len(ds) // d["gradient_accumulation_steps"])
    warmup_steps = max(1, round(d["warmup_ratio"] * total_steps))
    args = DPOConfig(
        output_dir=str(out_dir), per_device_train_batch_size=d["per_device_train_batch_size"],
        gradient_accumulation_steps=d["gradient_accumulation_steps"], num_train_epochs=1,
        learning_rate=d["learning_rate"], lr_scheduler_type=d["lr_scheduler_type"],
        warmup_steps=warmup_steps, beta=d["beta"], max_length=d["max_length"],
        truncation_mode="keep_start", bf16=True, gradient_checkpointing=True,
        dataloader_num_workers=0, logging_steps=1, save_strategy="no", report_to=[], seed=seed,
        # The first attempt reserved 7.96 GB on a 6 GB card -- Windows/WDDM silently spilled to
        # host memory, and optimizer steps took 204s/107s/49s instead of seconds. The reference
        # policy is frozen (adapter-disabled base weights), so caching its log-probs once is
        # mathematically identical to recomputing them every step and removes an entire forward
        # pass (plus its activations) from the training loop. Not a change to any fixed value in
        # Section 4.4 -- same beta, same max_length, same LoRA config, same optimum.
        precompute_ref_log_probs=True,
        precompute_ref_batch_size=1,
        # Measured (scripts/probe_dpo_memory.py, results/dpo_memory_probe.json): actual demand is
        # only ~3.0 GB allocated, but the allocator over-RESERVES to 5.2 GB and Windows/WDDM will
        # silently spill past the 6 GB card into host memory rather than raising OOM. Emptying the
        # cache each step holds peak reserved to 4.48 GB with no speed cost (44.9 vs 48.1 s/step).
        torch_empty_cache_steps=1,
    )
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    trainer = DPOTrainer(model=model, ref_model=None, args=args, train_dataset=ds,
                         processing_class=tokenizer, peft_config=lora,
                         callbacks=[StepTiming(LOGS_DIR / "track_c_step_timing.jsonl", tag)])
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    wall = time.time() - t0
    peak_alloc = torch.cuda.max_memory_allocated() / 1024**3
    peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_dir))
    loss_traj = [h for h in trainer.state.log_history if "loss" in h]
    del trainer, model
    reclaim()
    return {"wall_clock_s": wall, "loss_trajectory": loss_traj,
            "peak_vram_alloc_gb": round(peak_alloc, 3), "peak_vram_reserved_gb": round(peak_reserved, 3)}


def score_deltas(cfg, adapter_dir: Path, probe_pairs: list[dict]) -> list[dict]:
    from peft import PeftModel
    tokenizer = AutoTokenizer.from_pretrained(cfg["models"]["base_policy"])
    base = AutoModelForCausalLM.from_pretrained(
        cfg["models"]["base_policy"], dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
    model = PeftModel.from_pretrained(base, str(adapter_dir)).eval()
    out = delta_mod.compute_deltas(model, tokenizer, probe_pairs, "cuda")
    del model, base
    reclaim()
    return out


def regression_metrics(delta: np.ndarray, d: np.ndarray) -> dict:
    """OLS Delta = c + gamma*d + eps, plus the two sigma_D normalizations (0.2)."""
    A = np.vstack([np.ones_like(d), d]).T
    coef, *_ = np.linalg.lstsq(A, delta, rcond=None)
    pred = A @ coef
    resid = delta - pred
    ss_tot = np.sum((delta - delta.mean()) ** 2)
    r2 = 1.0 - np.sum(resid**2) / ss_tot if ss_tot > 0 else float("nan")
    sd_fit = float(np.std(pred))
    sigma_ratio = float(np.std(resid) / sd_fit) if sd_fit > 0 else float("inf")
    return {
        "gamma": float(coef[1]), "intercept": float(coef[0]), "r_squared": float(r2),
        "sigma_D_ratio": sigma_ratio,                      # Track E consumes this
        "sigma_D_ratio_from_r2": float(np.sqrt((1 - r2) / r2)) if 0 < r2 < 1 else float("nan"),
        "sigma_D_total": float(np.sqrt(max(1 - r2, 0))),   # descriptive only
        "pearson": float(pearsonr(delta, d)[0]), "spearman": float(spearmanr(delta, d)[0]),
        "sign_agreement": float(np.mean(np.sign(delta) == np.sign(d))),
    }


def residualize(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ coef


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="*", default=["A_tau2", "A_det", "B_qwen"])
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-invariance", action="store_true")
    args = parser.parse_args()

    # Hard allocator cap: on Windows/WDDM the driver will silently spill past physical VRAM into
    # host memory instead of raising OOM, which turns an over-subscription bug into a 100x
    # slowdown that looks like a hang. Capping makes over-subscription fail loudly and fast.
    # Measured demand is ~3.0 GB allocated / 4.5 GB reserved, so 90% of a 6 GB card is ample
    # headroom -- this bounds the failure mode, it does not constrain the run.
    torch.cuda.set_per_process_memory_fraction(0.90)
    print(f"allocator capped at 90% of {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")

    cfg = load_cfg()
    if not args.skip_invariance:
        print("=== scorer invariance gate (run at entry, per Step 0) ===")
        run_scorer_invariance_check()

    train_pairs = [json.loads(l) for l in open(CACHE_DIR / "response_pairs_train.jsonl")]
    probe_pairs = [json.loads(l) for l in open(CACHE_DIR / "response_pairs_probe.jsonl")]
    fj = json.load(open(CACHE_DIR / "feature_judges.json"))
    d_std_probe = np.array(fj["d_std_probe"])
    d_true_probe = np.array(fj["d_true_probe"])
    len_diff_probe = np.array(fj["len_diff_probe"])

    qwen_margins = {}
    for line in open(CACHE_DIR / "judge_margins_fixedbin_Qwen__Qwen2.5-1.5B-Instruct.jsonl"):
        r = json.loads(line)
        if r["template_name"] == "template_train":
            qwen_margins[r["prompt_idx"]] = r["d_m"]
    qwen_probe = np.array([qwen_margins[i] for i in range(len(probe_pairs))])

    # Qwen train-pair margins are NOT available (margins were scored on probe pairs), so Arm B's
    # training labels need Qwen scored on the 300 TRAIN pairs. Handled by the caller script;
    # here we read it if present.
    qwen_train_path = CACHE_DIR / "judge_margins_fixedbin_train_Qwen__Qwen2.5-1.5B-Instruct.jsonl"
    qwen_train = None
    if qwen_train_path.exists():
        m = {}
        for line in open(qwen_train_path):
            r = json.loads(line)
            if r["template_name"] == "template_train":
                m[r["prompt_idx"]] = r["d_m"]
        qwen_train = np.array([m[i] for i in range(len(train_pairs))])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    for arm in args.arms:
        if arm == "B_qwen" and qwen_train is None:
            print(f"\n!! SKIPPING {arm}: Qwen train-pair margins not found at {qwen_train_path}")
            print("   run: src.judges.margins --pairs train --n 300 --template train --tag fixedbin_train")
            continue

        print(f"\n{'='*70}\nARM {arm}\n{'='*70}")
        labels, diag = make_labels(arm, fj, qwen_train, args.tau, args.seed)
        print(f"  {diag}")

        for shuffled in (False, True):
            tag = f"{arm}{'_shuffled' if shuffled else ''}"
            ds = build_dataset(train_pairs, labels, shuffled, args.seed)
            out_dir = CKPT_DIR / f"track_c_{tag}"
            print(f"\n-- training {tag} ({len(ds)} pairs) --")
            baseline_vram = free_vram_gb()
            info = train_one(cfg, ds, out_dir, tag, args.seed)
            print(f"   trained in {info['wall_clock_s']:.0f}s, peak-free VRAM baseline {baseline_vram:.2f}GB")

            print(f"-- scoring {len(probe_pairs)} probes for {tag} --")
            t0 = time.time()
            deltas = score_deltas(cfg, out_dir, probe_pairs)
            delta = np.array([r["delta"] for r in deltas])
            print(f"   scored in {time.time()-t0:.0f}s   mean|Delta|={np.mean(np.abs(delta)):.4f}")

            rec = {"arm": arm, "shuffled": shuffled, "tag": tag, **diag,
                   "train_wall_clock_s": info["wall_clock_s"],
                   "peak_vram_alloc_gb": info["peak_vram_alloc_gb"],
                   "peak_vram_reserved_gb": info["peak_vram_reserved_gb"],
                   "mean_abs_delta": float(np.mean(np.abs(delta))),
                   "final_loss": info["loss_trajectory"][-1]["loss"] if info["loss_trajectory"] else None,
                   "delta": delta.tolist()}

            if not shuffled:
                target = d_std_probe if arm.startswith("A_") else qwen_probe
                target_name = "d_std_probe" if arm.startswith("A_") else "d_Qwen_obs"
                rec["target"] = target_name
                rec["metrics"] = regression_metrics(delta, target)
                rec["metrics_length_residualized"] = regression_metrics(
                    residualize(delta, len_diff_probe), residualize(target, len_diff_probe))
                if arm.startswith("A_"):
                    rec["rank_vs_d_true"] = {
                        "sign_agreement": float(np.mean(np.sign(delta) == np.sign(d_true_probe))),
                        "spearman": float(spearmanr(delta, d_true_probe)[0]),
                    }
                if arm == "A_tau2":
                    p = sigmoid(args.tau * d_std_probe)
                    rng = np.random.default_rng(args.seed + 7)
                    sampled = (rng.random(len(p)) < p).astype(int) * 2 - 1
                    rec["bayes_probe_accuracy"] = float(np.mean(np.maximum(p, 1 - p)))
                    rec["agreement_vs_sampled_probe_label"] = float(np.mean(np.sign(delta) == sampled))
                if arm == "B_qwen":
                    R_m = 0.573
                    obs = rec["metrics"]["pearson"]
                    rec["disattenuated_correlation"] = float(obs / np.sqrt(R_m))
                    rec["disattenuation_note"] = (
                        "sensitivity analysis only; NOT converted into a headline sigma_D, and the "
                        "correction does not undo that DPO received only binary labels")
                    rec["assumptions_violated"] = bool(abs(rec["disattenuated_correlation"]) > 1)

                m = rec["metrics"]
                print(f"   R2={m['r_squared']:.4f}  gamma={m['gamma']:.4f}  "
                      f"sigma_D_ratio={m['sigma_D_ratio']:.4f}  sigma_D_total={m['sigma_D_total']:.4f}")
                print(f"   sign_agree={m['sign_agreement']:.4f}  spearman={m['spearman']:.4f}  "
                      f"pearson={m['pearson']:.4f}")

            all_results.append(rec)
            with open(RESULTS_DIR / "track_c_pilot.json", "w") as f:
                json.dump({"tau": args.tau, "seed": args.seed, "results": all_results}, f, indent=2)

    print(f"\nwrote {RESULTS_DIR / 'track_c_pilot.json'}")


if __name__ == "__main__":
    main()
