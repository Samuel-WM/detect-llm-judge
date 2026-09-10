"""NEXT_STEPS_ROUND_4_FINAL.md 2.1: probe-length filter for the reward-model stage.

Reward models score (x, y) SINGLY -- one prompt plus one response -- not a prompt plus two
responses like the A/B judge templates. So the binding length is per (prompt, response), and the
640 = 384 + 256 figure that hard-blocked OpenAssistant (512 context) in round 3 was the wrong
quantity for this model class. Most real probes will be well under it.

Retain a probe only if its max tokenized length across ALL four model tokenizers and BOTH
responses is <= 500 (margin under DeBERTa's 512). Then:

    retained >= 0.60  ->  keep all four models, use the filtered subset everywhere
    retained <  0.60  ->  drop OpenAssistant, three models on the full 600, chance = 1/3

Tokenizers only -- no model weights are downloaded here.

Run: .venv/Scripts/python.exe -m src.judges.rm_probe_filter
"""
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = REPO_ROOT / "data_cache"
RESULTS_DIR = REPO_ROOT / "results"

MAX_LEN = 500
RETAIN_THRESHOLD = 0.60
POOL = [
    "OpenAssistant/reward-model-deberta-v3-large-v2",
    "internlm/internlm2-1_8b-reward",
    "Ray2333/gpt2-large-helpful-reward_model",
    "Ray2333/gpt2-large-harmless-reward_model",
]


def make_length_fn(repo: str):
    """Returns a function text -> token count.

    internlm2 ships only a legacy sentencepiece `tokenizer.model` (no tokenizer.json), and
    transformers 5.16's converter misidentifies it as a tiktoken file and dies parsing it
    (`Error parsing line b'\\x0e'`). That's a converter bug, not a property of the model, so we
    read the sentencepiece model directly instead -- same tokenizer, just bypassing the broken
    conversion path. +1 for the BOS token the model tokenizer would prepend (conservative).
    """
    try:
        tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
        return lambda text: len(tok.encode(text, add_special_tokens=True)), "AutoTokenizer"
    except Exception as e:
        from huggingface_hub import hf_hub_download
        import sentencepiece as spm
        path = hf_hub_download(repo_id=repo, filename="tokenizer.model")
        sp = spm.SentencePieceProcessor(model_file=path)
        print(f"   [AutoTokenizer failed: {type(e).__name__}; using sentencepiece directly]")
        return lambda text: len(sp.encode(text)) + 1, "sentencepiece-direct"


def serialize(prompt: str, response: str) -> str:
    """Single (prompt, response) serialization used for scalar reward scoring. Deliberately
    plain: reward models are trained on their own conventions, and the round-3 prohibition
    forbids constructing an artificial second input template for them."""
    return f"{prompt}\n\n{response}"


def main() -> None:
    pairs = [json.loads(l) for l in open(CACHE_DIR / "response_pairs_probe.jsonl")]
    print(f"probe pairs: {len(pairs)}   cap: {MAX_LEN} tokens   retain gate: {RETAIN_THRESHOLD:.0%}\n")

    per_model_lengths = {}
    tokenizer_source = {}
    for repo in POOL:
        print(f"{repo}")
        length_of, source = make_length_fn(repo)
        tokenizer_source[repo] = source
        lens = []
        for p in pairs:
            la = length_of(serialize(p["prompt"], p["response_a"]))
            lb = length_of(serialize(p["prompt"], p["response_b"]))
            lens.append(max(la, lb))
        per_model_lengths[repo] = np.array(lens)
        arr = per_model_lengths[repo]
        print(f"   max-of-both-responses length: median={np.median(arr):.0f} "
              f"p90={np.percentile(arr,90):.0f} max={arr.max()}  "
              f"over_{MAX_LEN}={int((arr > MAX_LEN).sum())}")

    all_len = np.vstack([per_model_lengths[r] for r in POOL])   # (n_models, n_pairs)
    worst = all_len.max(axis=0)
    retained = worst <= MAX_LEN
    frac = float(retained.mean())

    print(f"\nworst-case across all 4 tokenizers x both responses:")
    print(f"   median={np.median(worst):.0f} p90={np.percentile(worst,90):.0f} max={worst.max()}")
    print(f"   RETAINED {int(retained.sum())}/{len(pairs)} = {frac:.3f}")
    print(f"   retained length dist: median={np.median(worst[retained]):.0f} max={worst[retained].max()}")
    if (~retained).any():
        print(f"   dropped  length dist: median={np.median(worst[~retained]):.0f} max={worst[~retained].max()}")

    if frac >= RETAIN_THRESHOLD:
        decision = "KEEP_FOUR_FILTERED"
        print(f"\nDECISION: retained {frac:.3f} >= {RETAIN_THRESHOLD} -> keep all FOUR models, "
              f"use the {int(retained.sum())}-probe filtered subset for the entire reward-model stage. "
              f"Four-way chance = 1/4 = 0.25.")
    else:
        decision = "DROP_OPENASSISTANT"
        print(f"\nDECISION: retained {frac:.3f} < {RETAIN_THRESHOLD} -> drop OpenAssistant, "
              f"three models on the full 600. Three-way chance = 1/3 = 0.333.")

    out = {
        "max_len": MAX_LEN, "retain_threshold": RETAIN_THRESHOLD,
        "n_pairs": len(pairs), "n_retained": int(retained.sum()), "retained_fraction": frac,
        "decision": decision,
        "tokenizer_source": tokenizer_source,
        "retained_indices": [int(i) for i in np.flatnonzero(retained)],
        "worst_case_lengths": [int(x) for x in worst],
        "per_model_lengths": {r: [int(x) for x in v] for r, v in per_model_lengths.items()},
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "rm_probe_filter.json", "w") as f:
        json.dump(out, f)
    print(f"\nwrote {RESULTS_DIR / 'rm_probe_filter.json'}")


if __name__ == "__main__":
    main()
