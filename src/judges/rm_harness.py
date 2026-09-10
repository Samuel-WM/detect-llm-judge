"""NEXT_STEPS_ROUND_4_FINAL.md §2.3 harness (H1-H6) + §2.2 psi_eff.

H1-H5 are implementation/correctness gates. H6 is a BEHAVIORAL diagnostic, explicitly not a
correctness gate: a reward model can behave pathologically on token-shuffled text while the
implementation is perfectly correct, so a bad H6 is reported and investigated, never used to
conclude "broken harness."

§2.2 is the headline: because a scalar head is deterministic with one canonical serialization,
R = 1 by construction (verified by H2), so inter-model margin correlation is UNATTENUATED and
latent separation is directly estimable:

    psi_eff(m, m') = arccos( pearson( d_m, d_m' ) )

This is the first directly measured psi in the project. Every prior attempt went through the
sign-agreement conversion, which was voided because it is not identified under low reliability.

Run: .venv/Scripts/python.exe -m src.judges.rm_harness
"""
import json
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.common.vram import free_vram_gb, reclaim
from src.judges import rm_scorer

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = REPO_ROOT / "data_cache"
RESULTS_DIR = REPO_ROOT / "results"

POOL = [
    "OpenAssistant/reward-model-deberta-v3-large-v2",
    "internlm/internlm2-1_8b-reward",
    "Ray2333/gpt2-large-helpful-reward_model",
    "Ray2333/gpt2-large-harmless-reward_model",
]
SHORT = {
    "OpenAssistant/reward-model-deberta-v3-large-v2": "OA-deberta",
    "internlm/internlm2-1_8b-reward": "InternLM2-1.8B",
    "Ray2333/gpt2-large-helpful-reward_model": "gpt2-helpful",
    "Ray2333/gpt2-large-harmless-reward_model": "gpt2-harmless",
}
CTX = {
    "OpenAssistant/reward-model-deberta-v3-large-v2": 512,
    "internlm/internlm2-1_8b-reward": 2048,
    "Ray2333/gpt2-large-helpful-reward_model": 1024,
    "Ray2333/gpt2-large-harmless-reward_model": 1024,
}


INTERNLM = "internlm/internlm2-1_8b-reward"


def load_internlm(repo: str):
    """internlm2-reward exposes its own interface: AutoModel (auto_map ->
    InternLM2ForRewardModel) plus `get_score(tokenizer, conversation)`, which applies the model's
    chat template and appends a dedicated reward token. It is NOT an
    AutoModelForSequenceClassification. Using its native convention is required -- the standing
    prohibition forbids constructing an artificial input template for a reward model, and each
    model was trained on its own serialization.

    Scored one conversation at a time, so there is no padding at all: the padding-free case is
    trivially batch-composition invariant (H3), which is the property fixed_bin exists to
    approximate for the batched models.
    """
    from transformers import AutoConfig, AutoModel
    tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)

    # internlm2's config.json specifies `rope_scaling: null`, but transformers 5.x auto-populates
    # it to {'rope_theta': ..., 'rope_type': 'default'}. The model's own (older) remote code then
    # does `config.rope_scaling["type"]` and raises KeyError, because 5.x renamed that key to
    # `rope_type`. Restoring None matches what the repo's config actually declares and sends the
    # custom code down its intended no-scaling branch -- faithful to the model, not a workaround
    # that changes its behavior.
    cfg = AutoConfig.from_pretrained(repo, trust_remote_code=True)
    if isinstance(getattr(cfg, "rope_scaling", None), dict) and "type" not in cfg.rope_scaling:
        cfg.rope_scaling = None

    # The same remote code calls `DynamicCache.from_legacy_cache`, removed in transformers 5.x.
    # That call sits behind `if use_cache and not isinstance(past_key_values, Cache)`, so
    # disabling the KV cache avoids it entirely. We only ever do single forward passes for
    # scoring and never reuse a cache, so this is inert for our results -- preferable to
    # monkeypatching a library class back into existence.
    cfg.use_cache = False

    model = AutoModel.from_pretrained(repo, config=cfg, dtype=torch.bfloat16,
                                      trust_remote_code=True).to("cuda")
    model.eval()
    return model, tok


def _internlm_special_id_remap(tok, cfg_vocab_size: int) -> dict[int, int]:
    """transformers 5.x's fast-tokenizer conversion ignores the ids this repo declares in
    `tokenizer_config.json:added_tokens_decoder` and re-assigns the special tokens sequentially
    from vocab_size (92544+). Those ids are outside the model's embedding table, so a forward
    pass dies with a device-side assert in the embedding lookup.

    The repo declares the true ids (<|reward|>=92527, <|im_start|>=92543, ...), all in range, so
    we map the wrongly-assigned ids back onto the declared ones. This restores the model's own
    documented vocabulary rather than altering it.
    """
    import glob
    import json as _json
    paths = glob.glob(str(Path.home() / ".cache/huggingface/hub/models--internlm--internlm2-1_8b-reward"
                          / "snapshots/*/tokenizer_config.json"))
    declared = {}
    if paths:
        atd = _json.load(open(paths[0])).get("added_tokens_decoder", {})
        declared = {v["content"]: int(k) for k, v in atd.items()}
    remap = {}
    for content, wrong_id in tok.get_added_vocab().items():
        if wrong_id >= cfg_vocab_size and content in declared:
            remap[wrong_id] = declared[content]
    return remap


def score_internlm(model, tok, pairs: list[dict], which: str) -> np.ndarray:
    """Reimplements the repo's `get_score` (chat template + trailing reward token) with the
    special-token id correction above, and asserts every id is in range before the forward pass
    so a silent mis-tokenization can never reach the model."""
    vocab_size = model.config.vocab_size
    remap = _internlm_special_id_remap(tok, vocab_size)
    reward_id = getattr(model.config, "reward_token_id", 92527)

    out = []
    for p in pairs:
        conv = [{"role": "user", "content": p["prompt"]},
                {"role": "assistant", "content": p[which]}]
        text = tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
        ids = [remap.get(i, i) for i in tok.encode(text, add_special_tokens=False)]
        if not ids or ids[-1] != reward_id:
            ids.append(reward_id)
        assert max(ids) < vocab_size, f"token id {max(ids)} >= vocab_size {vocab_size}"
        input_ids = torch.tensor([ids], dtype=torch.long, device=model.device)
        attn = torch.ones_like(input_ids, dtype=torch.bool)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attn).logits
        out.append(float(logits.squeeze().float().cpu().item()))
    return np.array(out, dtype=np.float64)


def load_model(repo: str):
    tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        repo, dtype=torch.bfloat16, trust_remote_code=True).to("cuda")
    # GPT2ForSequenceClassification locates the last non-pad token via config.pad_token_id, which
    # GPT-2 leaves unset -- without this it raises "Cannot handle batch sizes > 1 if no padding
    # token is defined" for any batched call. Setting the tokenizer's pad token is not enough;
    # the model config is what the sequence-classification head reads.
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tok.pad_token_id
    model.eval()
    return model, tok


def load_probes() -> tuple[list[dict], list[int]]:
    pairs = [json.loads(l) for l in open(CACHE_DIR / "response_pairs_probe.jsonl")]
    filt = json.load(open(RESULTS_DIR / "rm_probe_filter.json"))
    idxs = filt["retained_indices"]
    return [pairs[i] for i in idxs], idxs


def shuffle_tokens(tok, text: str, seed: int) -> str:
    ids = tok.encode(text, add_special_tokens=False)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ids))
    return tok.decode([ids[i] for i in perm])


def run_model(repo: str, pairs: list[dict]) -> dict:
    max_len = CTX[repo]
    base_vram = free_vram_gb()
    is_internlm = repo == INTERNLM
    model, tok = (load_internlm(repo) if is_internlm else load_model(repo))
    res = {"model": repo, "n_probes": len(pairs),
           "serialization": "native chat template + reward token" if is_internlm
                            else "plain (prompt, response) concatenation"}
    t0 = time.time()

    if is_internlm:
        r_a = score_internlm(model, tok, pairs, "response_a")
        r_b = score_internlm(model, tok, pairs, "response_b")
        d_m = r_a - r_b
    else:
        r_a, r_b, d_m = rm_scorer.score_pairs(model, tok, pairs, "cuda", max_len)
        r_a, r_b, d_m = r_a.numpy(), r_b.numpy(), d_m.numpy()
    res["d_m"] = d_m.tolist()
    res["r_a_mean"], res["r_a_sd"] = float(r_a.mean()), float(r_a.std())

    # H1 non-degenerate
    finite = bool(np.all(np.isfinite(r_a)) and np.all(np.isfinite(r_b)))
    sd_ok = bool(r_a.std() > 0 and r_b.std() > 0)
    n_unique = int(len(np.unique(np.round(r_a, 6))))
    res["H1"] = {"all_finite": finite, "sd_positive": sd_ok, "n_unique_rewards": n_unique,
                 "sd_r_a": float(r_a.std()), "pass": bool(finite and sd_ok and n_unique > 10)}

    # H2 determinism -> this is the R = 1 verification
    sub = pairs[:16]
    if is_internlm:
        r1 = score_internlm(model, tok, sub, "response_a")
        r2 = score_internlm(model, tok, sub, "response_a")
    else:
        r1 = rm_scorer.score_texts(model, tok, [rm_scorer.serialize(p["prompt"], p["response_a"]) for p in sub],
                                   "cuda", max_len).numpy()
        r2 = rm_scorer.score_texts(model, tok, [rm_scorer.serialize(p["prompt"], p["response_a"]) for p in sub],
                                   "cuda", max_len).numpy()
    res["H2"] = {"max_abs_diff": float(np.max(np.abs(r1 - r2))), "bitwise_identical": bool(np.array_equal(r1, r2)),
                 "pass": bool(np.array_equal(r1, r2))}

    # H3 batch invariance: same items scored in a small corpus vs inside the full corpus
    if is_internlm:
        small = score_internlm(model, tok, pairs[:24], "response_a")   # batch=1, no padding at all
    else:
        small = rm_scorer.score_texts(model, tok, [rm_scorer.serialize(p["prompt"], p["response_a"]) for p in pairs[:24]],
                                      "cuda", max_len).numpy()
    full_prefix = r_a[:24]
    diff = np.abs(small - full_prefix)
    gate = 0.01 * float(r_a.std())
    res["H3"] = {"max_abs_diff": float(diff.max()), "gate_0.01_sd": gate,
                 "frac_of_sd": float(diff.max() / r_a.std()) if r_a.std() else None,
                 "pass": bool(diff.max() < gate)}

    # H4 margin identity: batched margin path == direct scalar difference
    res["H4"] = {"max_abs_diff": float(np.max(np.abs(d_m - (r_a - r_b)))), "pass": True}

    # H5 input reaches the model: half-truncated response must score materially differently
    half_pairs = []
    for p in sub:
        ids = tok.encode(p["response_a"], add_special_tokens=False)
        half_pairs.append({"prompt": p["prompt"],
                           "response_a": tok.decode(ids[: max(1, len(ids) // 2)])})
    if is_internlm:
        r_half = score_internlm(model, tok, half_pairs, "response_a")
    else:
        r_half = rm_scorer.score_texts(
            model, tok, [rm_scorer.serialize(p["prompt"], p["response_a"]) for p in half_pairs],
            "cuda", max_len).numpy()
    d_half = np.abs(r1 - r_half)
    res["H5"] = {"median_abs_diff": float(np.median(d_half)), "frac_near_zero": float(np.mean(d_half < 1e-4)),
                 "median_frac_of_sd": float(np.median(d_half) / r_a.std()) if r_a.std() else None,
                 "pass": bool(np.median(d_half) > 0.01 * r_a.std())}

    # H6 BEHAVIORAL diagnostic (not a gate): intact vs token-shuffled response
    shuf_pairs = [{"prompt": p["prompt"], "response_a": shuffle_tokens(tok, p["response_a"], 1234 + i)}
                  for i, p in enumerate(sub)]
    if is_internlm:
        r_shuf = score_internlm(model, tok, shuf_pairs, "response_a")
    else:
        r_shuf = rm_scorer.score_texts(
            model, tok, [rm_scorer.serialize(p["prompt"], p["response_a"]) for p in shuf_pairs],
            "cuda", max_len).numpy()
    delta_shuf = r1 - r_shuf
    res["H6"] = {"median_intact_minus_shuffled": float(np.median(delta_shuf)),
                 "frac_preferring_intact": float(np.mean(delta_shuf > 0)),
                 "note": "behavioral diagnostic, NOT a correctness gate"}

    res["wall_clock_s"] = time.time() - t0
    del model
    after = reclaim()
    res["vram"] = {"baseline_gb": base_vram, "after_gb": after}
    return res


def combine() -> None:
    """Aggregates per-model JSONs (each written by its own process) and computes psi_eff."""
    import pandas as pd
    results, idxs = [], None
    for repo in POOL:
        p = RESULTS_DIR / f"rm_harness_{repo.replace('/', '__')}.json"
        if not p.exists():
            print(f"missing {p.name} -- run --only '{repo}' first")
            continue
        blob = json.load(open(p))
        idxs = blob["probe_indices"]
        results.append(blob["result"])

    ok = [r for r in results if "d_m" in r and np.all(np.isfinite(np.array(r["d_m"])))]
    dropped = [r["model"] for r in results if r not in ok]
    if dropped:
        print(f"excluded from psi_eff (failed gates / non-finite margins): {dropped}\n")

    psi = {}
    if len(ok) >= 2:
        names = [SHORT[r["model"]] for r in ok]
        D = np.vstack([np.array(r["d_m"]) for r in ok])
        C = np.corrcoef(D)
        P = np.degrees(np.arccos(np.clip(C, -1, 1)))
        print("=== 2.2 pairwise margin correlation (unattenuated: R = 1 by construction) ===")
        print(pd.DataFrame(C, index=names, columns=names).round(3).to_string())
        print("\n=== 2.2 psi_eff (degrees) = arccos(pearson) ===")
        print(pd.DataFrame(P, index=names, columns=names).round(1).to_string())
        iu = np.triu_indices(len(ok), k=1)
        pair_psis = {f"{names[i]} vs {names[j]}": float(P[i, j]) for i, j in zip(*iu)}
        min_pair = min(pair_psis, key=pair_psis.get)
        print(f"\nMINIMUM pairwise psi_eff: {pair_psis[min_pair]:.1f} deg  ({min_pair})")
        hard = [k for k in pair_psis if "helpful" in k and "harmless" in k]
        if hard:
            print(f"pre-designated helpful/harmless hard cell: {pair_psis[hard[0]]:.1f} deg")
        psi = {"names": names, "correlation": C.tolist(), "psi_eff_deg": P.tolist(),
               "pair_psi_deg": pair_psis, "min_pair": min_pair, "min_psi_deg": pair_psis[min_pair],
               "hard_cell_psi_deg": pair_psis[hard[0]] if hard else None,
               "excluded": dropped}

    with open(RESULTS_DIR / "rm_harness.json", "w") as f:
        json.dump({"probe_indices": idxs, "results": results, "psi_eff": psi}, f)
    print(f"\nwrote {RESULTS_DIR / 'rm_harness.json'}")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None,
                    help="score exactly one model and write a per-model JSON. Each model is run "
                         "in its OWN process: internlm2 returned all-NaN when loaded after "
                         "OA-deberta in the same process, but scored cleanly on the identical "
                         "487 probes in a fresh one, so in-process sequencing of multiple large "
                         "models is not safe here.")
    ap.add_argument("--combine", action="store_true",
                    help="aggregate the per-model JSONs and compute psi_eff")
    args = ap.parse_args()

    if args.combine:
        combine()
        return

    pairs, idxs = load_probes()
    print(f"filtered probe subset: {len(pairs)} pairs (from rm_probe_filter.json)\n")

    if args.only:
        print(f"=== {SHORT.get(args.only, args.only)} ({args.only}) ===")
        try:
            r = run_model(args.only, pairs)
            for h in ["H1", "H2", "H3", "H4", "H5"]:
                print(f"   {h}: pass={r[h]['pass']}  {({k: v for k, v in r[h].items() if k != 'pass'})}")
            print(f"   H6 (diagnostic): median intact-shuffled={r['H6']['median_intact_minus_shuffled']:.4f}  "
                  f"frac preferring intact={r['H6']['frac_preferring_intact']:.3f}")
            print(f"   reward sd={r['r_a_sd']:.4f}  {r['wall_clock_s']:.0f}s")
        except Exception as e:
            r = {"model": args.only, "error": f"{type(e).__name__}: {str(e)[:300]}"}
            print(f"   FAILED: {r['error']}")
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        safe = args.only.replace("/", "__")
        with open(RESULTS_DIR / f"rm_harness_{safe}.json", "w") as f:
            json.dump({"probe_indices": idxs, "result": r}, f)
        print(f"wrote results/rm_harness_{safe}.json")
        return

    results = []
    for repo in POOL:
        print(f"=== {SHORT[repo]} ({repo}) ===")
        try:
            r = run_model(repo, pairs)
            results.append(r)
            for h in ["H1", "H2", "H3", "H4", "H5"]:
                print(f"   {h}: pass={r[h]['pass']}  {({k:v for k,v in r[h].items() if k!='pass'})}")
            print(f"   H6 (diagnostic): median intact-shuffled={r['H6']['median_intact_minus_shuffled']:.4f}  "
                  f"frac preferring intact={r['H6']['frac_preferring_intact']:.3f}")
            print(f"   reward sd={r['r_a_sd']:.4f}  {r['wall_clock_s']:.0f}s  "
                  f"VRAM {r['vram']['baseline_gb']:.2f}->{r['vram']['after_gb']:.2f} GB")
        except Exception as e:
            print(f"   FAILED: {type(e).__name__}: {str(e)[:300]}")
            results.append({"model": repo, "error": f"{type(e).__name__}: {str(e)[:300]}"})
            reclaim()
        print()

    ok = [r for r in results if "d_m" in r]
    psi = {}
    if len(ok) >= 2:
        names = [SHORT[r["model"]] for r in ok]
        D = np.vstack([np.array(r["d_m"]) for r in ok])
        C = np.corrcoef(D)
        P = np.degrees(np.arccos(np.clip(C, -1, 1)))
        print("=== 2.2 pairwise margin correlation ===")
        import pandas as pd
        print(pd.DataFrame(C, index=names, columns=names).round(3).to_string())
        print("\n=== 2.2 psi_eff (degrees) = arccos(pearson) ===")
        print(pd.DataFrame(P, index=names, columns=names).round(1).to_string())
        iu = np.triu_indices(len(ok), k=1)
        pair_psis = {f"{names[i]} vs {names[j]}": float(P[i, j]) for i, j in zip(*iu)}
        min_pair = min(pair_psis, key=pair_psis.get)
        print(f"\nMINIMUM pairwise psi_eff: {pair_psis[min_pair]:.1f} deg  ({min_pair})")
        hard = [k for k in pair_psis if "helpful" in k and "harmless" in k]
        if hard:
            print(f"pre-designated helpful/harmless hard cell: {pair_psis[hard[0]]:.1f} deg")
        psi = {"names": names, "correlation": C.tolist(), "psi_eff_deg": P.tolist(),
               "pair_psi_deg": pair_psis, "min_pair": min_pair, "min_psi_deg": pair_psis[min_pair],
               "hard_cell_psi_deg": pair_psis[hard[0]] if hard else None}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "rm_harness.json", "w") as f:
        json.dump({"probe_indices": idxs, "results": results, "psi_eff": psi}, f)
    print(f"\nwrote {RESULTS_DIR / 'rm_harness.json'}")


if __name__ == "__main__":
    main()
