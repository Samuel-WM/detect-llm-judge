"""NEXT_STEPS_ROUND_3_FINAL.md Section C, steps 3.1 + 3.5: reward-model provenance matrix and
pre-download compatibility checks. Metadata only -- config.json / tokenizer_config.json, never
weights. Per 3.9, this must be reported and reviewed BEFORE any weight download.

Checks per candidate:
  context length : config.max_position_embeddings, tokenizer.model_max_length, vs the intended
                   384 + 256 = 640 token scoring input. Truncation is NOT acceptable silently.
  scalar head    : verified from config (architectures / num_labels), not from the repo name.
  size           : parameter footprint vs the one-model-at-a-time VRAM budget.

Run: .venv/Scripts/python.exe -m src.judges.rm_precheck
"""
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "results"

REQUIRED_LEN = 640  # max_prompt_tokens 384 + max_new_tokens 256
VRAM_BUDGET_GB = 3.5

POOL = [
    "OpenAssistant/reward-model-deberta-v3-large-v2",
    "internlm/internlm2-1_8b-reward",
    "Ray2333/gpt2-large-helpful-reward_model",
    "Ray2333/gpt2-large-harmless-reward_model",
]

# Documented training corpora, from Hub dataset tags (verified, not recalled).
# Corpus identity is normalized to the BASE corpus so the overlap matrix is truthful: the two
# Ray2333 models use different SPLITS/objectives of Anthropic/hh-rlhf, but it is the same corpus,
# and recording the splits as distinct strings would have falsely shown zero overlap. The
# objective difference is recorded separately in OBJECTIVE below -- that's the deliberate
# helpful/harmless hard cell, not an independence claim.
PROVENANCE = {
    "OpenAssistant/reward-model-deberta-v3-large-v2": [
        "openai/summarize_from_feedback", "openai/webgpt_comparisons",
        "Dahoas/instruct-synthetic-prompt-responses", "Anthropic/hh-rlhf",
    ],
    "internlm/internlm2-1_8b-reward": ["InternLM internal RLHF preference data (arXiv 2403.17297)"],
    "Ray2333/gpt2-large-helpful-reward_model": ["Anthropic/hh-rlhf"],
    "Ray2333/gpt2-large-harmless-reward_model": ["Anthropic/hh-rlhf"],
}

OBJECTIVE = {
    "OpenAssistant/reward-model-deberta-v3-large-v2": "general preference (mixed corpora)",
    "internlm/internlm2-1_8b-reward": "general preference (InternLM internal)",
    "Ray2333/gpt2-large-helpful-reward_model": "helpfulness",
    "Ray2333/gpt2-large-harmless-reward_model": "harmlessness",
}

CONTEXT_KEYS = [
    "max_position_embeddings", "n_positions", "max_seq_length", "n_ctx", "seq_length",
]


def get_meta(repo: str) -> dict:
    api = HfApi()
    info = api.model_info(repo)
    out = {"repo": repo, "gated": info.gated, "pipeline_tag": info.pipeline_tag}

    cfg_path = hf_hub_download(repo_id=repo, filename="config.json")
    cfg = json.load(open(cfg_path))
    out["architectures"] = cfg.get("architectures")
    out["model_type"] = cfg.get("model_type")
    out["num_labels"] = cfg.get("num_labels")
    out["id2label_size"] = len(cfg.get("id2label", {})) if cfg.get("id2label") else None
    out["config_context"] = {k: cfg.get(k) for k in CONTEXT_KEYS if cfg.get(k) is not None}
    out["hidden_size"] = cfg.get("hidden_size") or cfg.get("n_embd")
    out["num_layers"] = cfg.get("num_hidden_layers") or cfg.get("n_layer")

    try:
        tok_path = hf_hub_download(repo_id=repo, filename="tokenizer_config.json")
        tok_cfg = json.load(open(tok_path))
        out["tokenizer_model_max_length"] = tok_cfg.get("model_max_length")
    except Exception as e:
        out["tokenizer_model_max_length"] = f"unavailable: {type(e).__name__}"

    # size: prefer the safetensors index total, else the single-file metadata
    size_bytes = None
    for fname in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        try:
            idx = json.load(open(hf_hub_download(repo_id=repo, filename=fname)))
            size_bytes = idx.get("metadata", {}).get("total_size")
            break
        except Exception:
            continue
    if size_bytes is None:
        for s in api.model_info(repo, files_metadata=True).siblings:
            if s.rfilename in ("model.safetensors", "pytorch_model.bin") and s.size:
                size_bytes = s.size
                break
    out["stored_size_bytes"] = size_bytes
    out["stored_size_gb"] = round(size_bytes / 1024**3, 3) if size_bytes else None
    return out


def evaluate(meta: dict) -> dict:
    ctx_vals = [v for v in meta["config_context"].values() if isinstance(v, int)]
    config_ctx = max(ctx_vals) if ctx_vals else None
    tok_max = meta.get("tokenizer_model_max_length")
    tok_ctx = tok_max if isinstance(tok_max, int) and tok_max < 10**6 else None

    effective = min([c for c in (config_ctx, tok_ctx) if c is not None], default=None)
    ctx_ok = effective is not None and effective >= REQUIRED_LEN

    arch = " ".join(meta.get("architectures") or [])
    scalar_head = any(k in arch for k in ("SequenceClassification", "RewardModel", "ForReward"))

    size_gb = meta.get("stored_size_gb")
    size_ok = size_gb is not None and size_gb <= VRAM_BUDGET_GB

    return {
        "config_context": config_ctx, "tokenizer_context": tok_ctx,
        "effective_context": effective, "context_ok": ctx_ok,
        "scalar_head_from_arch": scalar_head, "size_gb": size_gb, "size_ok": size_ok,
        "all_ok": bool(ctx_ok and scalar_head and size_ok),
    }


def main() -> None:
    print(f"Pre-download compatibility checks. Required scoring length: {REQUIRED_LEN} tokens "
          f"(384 prompt + 256 response). VRAM budget: {VRAM_BUDGET_GB} GB.\n")

    rows = []
    for repo in POOL:
        try:
            meta = get_meta(repo)
            verdict = evaluate(meta)
            rows.append({**meta, **verdict, "training_corpora": PROVENANCE.get(repo)})
            print(f"=== {repo} ===")
            print(f"  architectures      : {meta['architectures']}  (model_type={meta['model_type']})")
            print(f"  scalar head        : {verdict['scalar_head_from_arch']}  "
                  f"(num_labels={meta['num_labels']}, id2label={meta['id2label_size']})")
            print(f"  config context     : {meta['config_context']}")
            print(f"  tokenizer max len  : {meta['tokenizer_model_max_length']}")
            print(f"  effective context  : {verdict['effective_context']}  "
                  f"-> {'OK' if verdict['context_ok'] else 'FAIL (< 640)'}")
            print(f"  stored size        : {meta['stored_size_gb']} GB  "
                  f"-> {'OK' if verdict['size_ok'] else 'FAIL (> 3.5GB)'}")
            print(f"  gated              : {meta['gated']}")
            print(f"  VERDICT            : {'PASS' if verdict['all_ok'] else 'BLOCKED'}\n")
        except Exception as e:
            rows.append({"repo": repo, "error": f"{type(e).__name__}: {str(e)[:200]}"})
            print(f"=== {repo} ===\n  ERROR: {type(e).__name__}: {str(e)[:200]}\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "rm_precheck.json"
    with open(out_path, "w") as f:
        json.dump({"required_context": REQUIRED_LEN, "vram_budget_gb": VRAM_BUDGET_GB,
                   "provenance": PROVENANCE, "objective": OBJECTIVE, "results": rows}, f, indent=2)
    print(f"wrote {out_path}")

    blocked = [r["repo"] for r in rows if r.get("error") or not r.get("all_ok")]
    print("\n=== SUMMARY ===")
    for r in rows:
        if r.get("error"):
            print(f"  {r['repo']}: ERROR")
        else:
            print(f"  {r['repo']}: ctx={r['effective_context']} size={r['size_gb']}GB "
                  f"scalar_head={r['scalar_head_from_arch']} -> {'PASS' if r['all_ok'] else 'BLOCKED'}")
    if blocked:
        print(f"\n{len(blocked)} candidate(s) BLOCKED -- per 3.5, stop and report before downloading:")
        for b in blocked:
            print(f"  - {b}")
    else:
        print("\nAll candidates pass pre-download checks.")


if __name__ == "__main__":
    main()
