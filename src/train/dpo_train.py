"""Track C pilot (NEXT_STEPS_FINAL.md): one small DPO run to measure alignment-transmission
noise. Deliberately small so failure is cheap.

    judge:   Qwen/Qwen2.5-1.5B-Instruct (the only already-measured judge with substantial
             cross-template reproducibility)
    train:   300 prompts, 2 responses each from pi_ref, 1 epoch, batch 1 x grad-accum 16
             (~19 optimizer steps)
    probes:  the existing 600, not regenerated

Section 4.4 settings otherwise unchanged. `max_prompt_length` no longer exists in this trl;
prompts are already capped at 384 tokens at generation time and max_length=640 covers
384 + 256 (see RESULTS.md environment notes).

Per-step wall-clock logging is on, per the "Loose end: runtime anomaly" instruction.

Run: .venv/Scripts/python.exe -m src.train.dpo_train --pilot
"""
import argparse
import json
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import DPOConfig, DPOTrainer

from src.common.vram import free_vram_gb, reclaim
from src.data.build_prompts import load_cfg

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = REPO_ROOT / "data_cache"
LOGS_DIR = REPO_ROOT / "logs"


class StepTimingCallback(TrainerCallback):
    """Per-optimizer-step wall clock + GPU memory, written to logs/dpo_step_timing.jsonl."""

    def __init__(self, path: Path):
        self.path = path
        self.t_last = None
        self.records = []

    def on_step_end(self, args, state, control, **kwargs):
        now = time.time()
        if self.t_last is not None:
            rec = {
                "step": state.global_step,
                "seconds": round(now - self.t_last, 4),
                "mem_alloc_gb": round(torch.cuda.memory_allocated() / 1024**3, 3),
                "mem_reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 3),
                "loss": state.log_history[-1].get("loss") if state.log_history else None,
            }
            self.records.append(rec)
            with open(self.path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        self.t_last = now
        return control


def build_dpo_dataset(judge_name: str, n_prompts: int, shuffle_labels: bool, seed: int) -> Dataset:
    """Training pairs labeled by the judge's own margin sign (z_m). With shuffle_labels, the
    preference direction is randomized -- the negative-control model."""
    pairs_path = CACHE_DIR / "response_pairs_train.jsonl"
    assert pairs_path.exists(), "run src.gen.sample_pairs --prompts train first"
    safe = judge_name.replace("/", "__")
    label_path = CACHE_DIR / f"train_labels_{safe}.jsonl"
    assert label_path.exists(), f"{label_path} missing -- run src.judges.margins --pairs train"

    pairs = [json.loads(l) for l in open(pairs_path)][:n_prompts]
    labels = {}
    for line in open(label_path):
        row = json.loads(line)
        if row["template_name"] == "template_train":
            labels[row["prompt_idx"]] = row

    import random
    rng = random.Random(seed)

    recs = []
    for i, p in enumerate(pairs):
        if i not in labels:
            continue
        z = labels[i]["z_m"]
        if shuffle_labels:
            z = rng.randint(0, 1)
        chosen = p["response_a"] if z == 1 else p["response_b"]
        rejected = p["response_b"] if z == 1 else p["response_a"]
        recs.append({"prompt": p["prompt"], "chosen": chosen, "rejected": rejected})
    return Dataset.from_list(recs)


def train(cfg: dict, judge_name: str, n_prompts: int, epochs: int, shuffle_labels: bool,
          out_dir: Path, seed: int) -> dict:
    d = cfg["dpo"]
    tokenizer = AutoTokenizer.from_pretrained(cfg["models"]["base_policy"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["models"]["base_policy"], dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to("cuda")

    ds = build_dpo_dataset(judge_name, n_prompts, shuffle_labels, seed)
    print(f"  DPO dataset: {len(ds)} pairs (shuffle_labels={shuffle_labels})")

    lora = LoraConfig(
        r=d["lora"]["r"], lora_alpha=d["lora"]["alpha"], lora_dropout=d["lora"]["dropout"],
        target_modules=d["lora"]["target_modules"], task_type="CAUSAL_LM",
    )
    args = DPOConfig(
        output_dir=str(out_dir),
        per_device_train_batch_size=d["per_device_train_batch_size"],
        gradient_accumulation_steps=d["gradient_accumulation_steps"],
        num_train_epochs=epochs,
        learning_rate=d["learning_rate"],
        lr_scheduler_type=d["lr_scheduler_type"],
        warmup_ratio=d["warmup_ratio"],
        beta=d["beta"],
        max_length=d["max_length"],
        truncation_mode="keep_start",
        bf16=d["bf16"],
        gradient_checkpointing=d["gradient_checkpointing"],
        dataloader_num_workers=0,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        seed=seed,
    )
    timing = StepTimingCallback(LOGS_DIR / "dpo_step_timing.jsonl")
    trainer = DPOTrainer(model=model, ref_model=None, args=args, train_dataset=ds,
                         processing_class=tokenizer, peft_config=lora, callbacks=[timing])
    t0 = time.time()
    result = trainer.train()
    wall = time.time() - t0

    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_dir))
    loss_traj = [h for h in trainer.state.log_history if "loss" in h]
    with open(out_dir / "loss_trajectory.json", "w") as f:
        json.dump(loss_traj, f, indent=2)

    del trainer, model
    reclaim()
    return {"wall_clock_s": wall, "train_result": str(result), "n_pairs": len(ds),
            "loss_trajectory": loss_traj}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--n-prompts", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--shuffled-control", action="store_true",
                        help="train the shuffled-label control model instead")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = load_cfg()
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    tag = "shuffled" if args.shuffled_control else "pilot"
    out_dir = REPO_ROOT / "checkpoints" / f"dpo_{tag}_{args.judge.replace('/', '__')}"

    print(f"free VRAM before: {free_vram_gb():.3f} GB")
    info = train(cfg, args.judge, args.n_prompts, args.epochs, args.shuffled_control, out_dir, args.seed)
    print(f"trained in {info['wall_clock_s']:.0f}s on {info['n_pairs']} pairs -> {out_dir}")
    print(f"free VRAM after: {free_vram_gb():.3f} GB")


if __name__ == "__main__":
    main()
