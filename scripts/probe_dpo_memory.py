"""Bounded, instrumented probe: how much VRAM does one DPO optimizer step actually need?

Motivation: the first pilot attempt reserved 7.96 GB on a 6 GB card and steps took 204s/107s/49s.
Adding `precompute_ref_log_probs` only moved it to 7.45 GB / 121s. Windows/WDDM silently spills
past physical VRAM into host memory instead of raising OOM, so the failure mode is 100x slowdown
rather than an error -- which is exactly why this needs measuring rather than guessing.

`torch.cuda.set_per_process_memory_fraction` installs a HARD cap so the allocator raises OOM
instead of spilling. Each config is probed for 2 optimizer steps only.

Run: .venv/Scripts/python.exe -m scripts.probe_dpo_memory
"""
import json
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gc
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from src.data.build_prompts import load_cfg

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "data_cache"
CAP_FRACTION = 0.90  # ~5.4 GB of the 6 GB card; beyond this we want a loud OOM, not host spill
N_PROBE_PAIRS = 32
MAX_STEPS = 2


def build_ds(n: int) -> Dataset:
    pairs = [json.loads(l) for l in open(CACHE_DIR / "response_pairs_train.jsonl")][:n]
    return Dataset.from_list([
        {"prompt": p["prompt"], "chosen": p["response_a"], "rejected": p["response_b"]}
        for p in pairs
    ])


def probe(cfg, name: str, **overrides) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    d = cfg["dpo"]
    tok = AutoTokenizer.from_pretrained(cfg["models"]["base_policy"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["models"]["base_policy"], dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
    lora = LoraConfig(r=d["lora"]["r"], lora_alpha=d["lora"]["alpha"], lora_dropout=d["lora"]["dropout"],
                      target_modules=d["lora"]["target_modules"], task_type="CAUSAL_LM")
    base_kwargs = dict(
        output_dir=str(REPO_ROOT / "scratch_probe"), per_device_train_batch_size=1,
        gradient_accumulation_steps=d["gradient_accumulation_steps"], max_steps=MAX_STEPS,
        learning_rate=d["learning_rate"], beta=d["beta"], max_length=d["max_length"],
        truncation_mode="keep_start", bf16=True, gradient_checkpointing=True,
        dataloader_num_workers=0, logging_steps=1, save_strategy="no", report_to=[],
    )
    base_kwargs.update(overrides)
    args = DPOConfig(**base_kwargs)
    trainer = DPOTrainer(model=model, ref_model=None, args=args, train_dataset=build_ds(N_PROBE_PAIRS),
                         processing_class=tok, peft_config=lora)

    t0 = time.time()
    status, err = "ok", None
    try:
        trainer.train()
    except torch.cuda.OutOfMemoryError as e:
        status, err = "OOM", str(e)[:160]
    except RuntimeError as e:
        status, err = "RuntimeError", str(e)[:160]
    wall = time.time() - t0

    peak_alloc = torch.cuda.max_memory_allocated() / 1024**3
    peak_res = torch.cuda.max_memory_reserved() / 1024**3
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    rec = {"config": name, "status": status, "error": err, "wall_s": round(wall, 1),
           "sec_per_step": round(wall / MAX_STEPS, 1),
           "peak_alloc_gb": round(peak_alloc, 3), "peak_reserved_gb": round(peak_res, 3)}
    print(f"  {name:34s} {status:12s} {rec['sec_per_step']:6.1f} s/step  "
          f"peak_alloc={rec['peak_alloc_gb']:.2f}GB  peak_reserved={rec['peak_reserved_gb']:.2f}GB")
    if err:
        print(f"     -> {err}")
    return rec


def main() -> None:
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    torch.cuda.set_per_process_memory_fraction(CAP_FRACTION)
    print(f"card total {total:.2f} GB; hard allocator cap at {CAP_FRACTION:.0%} "
          f"= {total*CAP_FRACTION:.2f} GB (OOM instead of host spill)\n")

    cfg = load_cfg()
    results = []
    results.append(probe(cfg, "precompute_ref (current)", precompute_ref_log_probs=True,
                          precompute_ref_batch_size=1))
    results.append(probe(cfg, "precompute_ref + empty_cache", precompute_ref_log_probs=True,
                          precompute_ref_batch_size=1, torch_empty_cache_steps=1))
    results.append(probe(cfg, "precompute_ref + act_offload", precompute_ref_log_probs=True,
                          precompute_ref_batch_size=1, activation_offloading=True))
    results.append(probe(cfg, "no precompute (baseline)", precompute_ref_log_probs=False))

    out = REPO_ROOT / "results" / "dpo_memory_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"cap_fraction": CAP_FRACTION, "card_total_gb": total, "results": results}, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
