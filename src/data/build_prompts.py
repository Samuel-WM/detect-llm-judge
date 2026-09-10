"""Section 4.2: prompts from HuggingFaceH4/ultrafeedback_binarized, prompt field only,
deduplicated. Splits into disjoint train / probe prompt sets and asserts disjointness -- probe
leakage silently inflates every downstream number.

Track B (NEXT_STEPS_TRACKS_A_B_C.md) only needs 600 probe prompts; the full n_probe_prompts=1200
is built here anyway (cheap, CPU/network only) so Phase 2 can reuse the same checkpointed file.

Run: .venv/Scripts/python.exe -m src.data.build_prompts
"""
import json
from pathlib import Path

import yaml
from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = REPO_ROOT / "data_cache"


def load_cfg() -> dict:
    with open(REPO_ROOT / "configs" / "base.yaml") as f:
        return yaml.safe_load(f)


def build_prompt_splits(cfg: dict, seed: int) -> dict:
    out_path = CACHE_DIR / "prompt_splits.json"
    if out_path.exists():
        with open(out_path) as f:
            cached = json.load(f)
        if cached.get("seed") == seed and cached.get("n_train_prompts") == cfg["data"]["n_train_prompts"] \
                and cached.get("n_probe_prompts") == cfg["data"]["n_probe_prompts"]:
            print(f"reusing cached prompt split at {out_path}")
            return cached

    ds = load_dataset(cfg["data"]["dataset"], split="train_prefs")
    seen = set()
    prompts = []
    for row in ds:
        p = row["prompt"]
        if p not in seen:
            seen.add(p)
            prompts.append(p)

    n_train = cfg["data"]["n_train_prompts"]
    n_probe = cfg["data"]["n_probe_prompts"]
    total_needed = n_train + n_probe
    assert len(prompts) >= total_needed, (
        f"only {len(prompts)} unique prompts available, need {total_needed} "
        f"(n_train={n_train} + n_probe={n_probe})"
    )

    import random
    rng = random.Random(seed)
    rng.shuffle(prompts)
    train_prompts = prompts[:n_train]
    probe_prompts = prompts[n_train:n_train + n_probe]

    assert set(train_prompts).isdisjoint(set(probe_prompts)), \
        "train/probe prompt sets are not disjoint -- probe leakage, aborting"

    result = {
        "seed": seed, "n_train_prompts": n_train, "n_probe_prompts": n_probe,
        "train_prompts": train_prompts, "probe_prompts": probe_prompts,
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f)
    print(f"wrote {len(train_prompts)} train + {len(probe_prompts)} probe prompts to {out_path}")
    return result


def main() -> None:
    cfg = load_cfg()
    build_prompt_splits(cfg, cfg["seed"])


if __name__ == "__main__":
    main()
