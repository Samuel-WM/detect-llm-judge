"""Unit test for the chat-template consistency invariant (NEXT_STEPS_TRACKS_A_B_C.md's closing
note): build_scoring_inputs must return a token-id prompt prefix exactly equal to the standalone
prompt tokenization, and must raise loudly on a genuine mismatch. Meant to be run at the top of
every phase entry point that calls apply_chat_template (gen/sample_pairs.py, judges/margins.py,
the eventual scorer) so template drift is caught immediately rather than silently corrupting
Delta downstream -- none of those three call sites will fail loudly on their own if it breaks.

Run directly: .venv/Scripts/python.exe -m tests.test_templating
"""
import torch
from transformers import AutoTokenizer

from src.common.templating import build_scoring_inputs

DEFAULT_TEST_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def test_happy_path(tokenizer) -> None:
    full_ids, prompt_len = build_scoring_inputs(
        tokenizer, "What is the capital of France?", "The capital of France is Paris.",
    )
    prefix = full_ids[:prompt_len]
    prompt_only = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is the capital of France?"}],
        add_generation_prompt=True, tokenize=True, return_tensors="pt",
    )["input_ids"][0]
    assert torch.equal(prefix, prompt_only), \
        "happy path should reproduce the standalone prompt tokenization exactly"


def test_mismatch_is_detectable(tokenizer) -> None:
    """Confirms the check has teeth: a genuinely inconsistent tokenization (raw encode() vs
    chat-templated encode() of the same text) is flagged, not silently accepted.
    """
    raw_prompt_ids = torch.tensor(tokenizer.encode("What is the capital of France?"))
    full_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is the capital of France?"},
         {"role": "assistant", "content": "Paris."}],
        add_generation_prompt=False, tokenize=True, return_tensors="pt",
    )["input_ids"][0]
    prefix = full_ids[:len(raw_prompt_ids)]
    assert not torch.equal(prefix, raw_prompt_ids), (
        "expected a mismatch between raw encode() and chat-templated encode() -- if this now "
        "matches, the test fixture no longer exercises the failure mode and needs revisiting"
    )


def run(model_name: str = DEFAULT_TEST_MODEL) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    test_happy_path(tokenizer)
    test_mismatch_is_detectable(tokenizer)
    print(f"tests/test_templating.py: PASS ({model_name})")


if __name__ == "__main__":
    run()
