"""Smoke test before Phase 1: bf16 forward pass on the real base policy, plus a DPOTrainer
instantiation with LoRA and `ref_model=None` (adapter-toggle reference), matching Section 1
rules 1/3 and Section 4.4. Does not run a full training loop — verifies the stack initializes
and one train step executes without error, against the transformers 5.16.1 / trl 1.12.0 /
peft 0.20.0 actually installed (newer than the doc's implicit baseline).
"""
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gc
import time

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def free_vram_gb() -> float:
    free_b, _total_b = torch.cuda.mem_get_info(0)
    return free_b / 1024**3


def main() -> None:
    print(f"free VRAM before load: {free_vram_gb():.3f} GB")

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to("cuda")
    print(f"loaded {MODEL_NAME} in {time.time() - t0:.1f}s")
    print(f"free VRAM after load: {free_vram_gb():.3f} GB")

    # --- bf16 forward pass, float32 log-softmax reduction ---
    prompt = "Explain what a partition function is in one sentence."
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(**inputs)
    logits = out.logits
    assert logits.dtype == torch.bfloat16, f"expected bf16 logits, got {logits.dtype}"
    logp = torch.log_softmax(logits.float(), dim=-1)
    assert logp.dtype == torch.float32
    print(f"forward pass ok: logits dtype={logits.dtype}, log-softmax dtype={logp.dtype}")

    # bitwise reproducibility check (Section 4.5)
    with torch.no_grad():
        out2 = model(**inputs)
    assert torch.equal(out.logits, out2.logits), "logits not bitwise reproducible across calls"
    print("bitwise reproducibility ok (adapter-off equivalent, base model)")

    del out, out2, logits, logp, inputs
    gc.collect()
    torch.cuda.empty_cache()

    # --- DPOTrainer + LoRA + ref_model=None instantiation ---
    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    tiny_ds = Dataset.from_dict({
        "prompt": ["Say hello.", "What is 2+2?"],
        "chosen": ["Hello! How can I help you today?", "2 + 2 = 4."],
        "rejected": ["No.", "I don't know."],
    })

    # trl 1.12.0's DPOConfig dropped max_prompt_length in favor of a single max_length over the
    # whole tokenized sequence (truncation_mode="keep_start"). This still respects Section 4.4's
    # fixed values without change: max_prompt_tokens=384 is enforced upstream at generation time
    # (Section 4.2), and 384 + max_new_tokens(256) = 640 = max_length, so the two caps compose
    # exactly as designed; only the *enforcement point* moved, not the values.
    dpo_config = DPOConfig(
        output_dir="scratch_smoke_test_dpo",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=1,
        learning_rate=5e-5,
        beta=0.1,
        max_length=640,
        truncation_mode="keep_start",
        bf16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=0,
        report_to=[],
        logging_steps=1,
        save_strategy="no",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=tiny_ds,
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    print("DPOTrainer instantiated ok with ref_model=None + peft_config (adapter-toggle reference)")

    train_result = trainer.train()
    print(f"one training step completed: {train_result}")

    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"free VRAM after cleanup: {free_vram_gb():.3f} GB")

    print()
    print("SMOKE TEST: PASS")


if __name__ == "__main__":
    main()
