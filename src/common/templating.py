"""Consistent prompt/response tokenization across generation, DPO training, and scoring.

PHASE_0C_SPEC.md, "Applies to Phase 1b and everything after it": the chat template must be
applied identically at all three stages. If the scorer computes `log pi_A(y | x)` under a
different prompt serialization than training used, `Delta` is corrupted in a way that produces
plausible-looking but meaningless numbers, and nothing downstream flags it -- so this asserts
loudly rather than warning.
"""
import torch


def build_scoring_inputs(tokenizer, prompt: str, response: str) -> tuple[torch.Tensor, int]:
    """Tokenizes `prompt` alone and `prompt + response` through the tokenizer's chat template
    and returns `(full_input_ids, prompt_len)`, where `full_input_ids[:prompt_len]` is asserted
    to exactly equal the standalone prompt tokenization. Call this identically at generation,
    training, and scoring time so Delta is always computed over the same prompt/response
    boundary under the same serialization.
    """
    # apply_chat_template(..., return_tensors="pt") returns a BatchEncoding (dict-like), not a
    # bare tensor, on transformers>=5 -- index ["input_ids"][0], not [0].
    prompt_messages = [{"role": "user", "content": prompt}]
    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages, add_generation_prompt=True, tokenize=True, return_tensors="pt",
    )["input_ids"][0]

    full_messages = prompt_messages + [{"role": "assistant", "content": response}]
    full_ids = tokenizer.apply_chat_template(
        full_messages, add_generation_prompt=False, tokenize=True, return_tensors="pt",
    )["input_ids"][0]

    prompt_len = prompt_ids.shape[0]
    prefix = full_ids[:prompt_len]
    if not torch.equal(prefix, prompt_ids):
        raise AssertionError(
            "Prompt prefix of the prompt+response tokenization does not exactly match the "
            "standalone prompt tokenization -- chat template applied inconsistently between "
            "stages. This would silently corrupt Delta if left unchecked.\n"
            f"prompt_ids ({len(prompt_ids)} tokens): {prompt_ids.tolist()}\n"
            f"prefix     ({len(prefix)} tokens): {prefix.tolist()}"
        )
    return full_ids, prompt_len
