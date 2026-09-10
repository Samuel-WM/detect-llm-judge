"""NEXT_STEPS_ROUND_4_FINAL.md §2.1 step 5: filtered-vs-full bias check.

The 487/600 length filter selects on length, and length is a known confound in this project, so
the filter could in principle change the pool's geometry rather than just trimming a tail. This
compares pairwise reward-margin correlations (and the psi_eff derived from them) on the filtered
subset versus the full 600.

OpenAssistant is deliberately EXCLUDED from the full-set arm: its context is 512 tokens and 82 of
the 600 probes exceed the 500-token cap, so scoring it on the full set would require truncating
beyond its supported context. The instruction is explicit that we must never force it past its
context just to manufacture a comparison. So the full-set arm covers the three models that can be
validly scored on all 600, and OA appears only in the filtered arm.

Run: .venv/Scripts/python.exe -m src.judges.rm_bias_check --only <repo>   (one process per model)
     .venv/Scripts/python.exe -m src.judges.rm_bias_check --combine
"""
import argparse
import json
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import time
from pathlib import Path

import numpy as np

from src.common.vram import reclaim
from src.judges import rm_scorer
from src.judges.rm_harness import CTX, INTERNLM, SHORT, load_internlm, load_model, score_internlm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = REPO_ROOT / "data_cache"
RESULTS_DIR = REPO_ROOT / "results"

# Models that can be scored on the FULL 600 without exceeding their context.
FULL_SET_OK = [INTERNLM, "Ray2333/gpt2-large-helpful-reward_model",
               "Ray2333/gpt2-large-harmless-reward_model"]
OA = "OpenAssistant/reward-model-deberta-v3-large-v2"


def score_full(repo: str) -> dict:
    pairs = [json.loads(l) for l in open(CACHE_DIR / "response_pairs_probe.jsonl")]
    is_int = repo == INTERNLM
    model, tok = (load_internlm(repo) if is_int else load_model(repo))
    t0 = time.time()
    if is_int:
        r_a = score_internlm(model, tok, pairs, "response_a")
        r_b = score_internlm(model, tok, pairs, "response_b")
        d_m = r_a - r_b
        realized = realized_lengths_internlm(tok, model, pairs)
    else:
        r_a, r_b, d_m = rm_scorer.score_pairs(model, tok, pairs, "cuda", CTX[repo])
        d_m = d_m.numpy()
        realized = None
    out = {"model": repo, "n": len(pairs), "d_m_full": d_m.tolist(),
           "wall_clock_s": time.time() - t0}
    if realized is not None:
        out["realized_lengths_full"] = realized
    del model
    reclaim()
    return out


def realized_lengths_internlm(tok, model, pairs: list[dict]) -> list[int]:
    """§B.2: realized input_ids length under the EXACT final scoring serialization (chat template
    + appended reward token), not the preliminary plain-concatenation SentencePiece estimate."""
    from src.judges.rm_harness import _internlm_special_id_remap
    vocab = model.config.vocab_size
    remap = _internlm_special_id_remap(tok, vocab)
    reward_id = getattr(model.config, "reward_token_id", 92527)
    lens = []
    for p in pairs:
        for which in ("response_a", "response_b"):
            conv = [{"role": "user", "content": p["prompt"]},
                    {"role": "assistant", "content": p[which]}]
            text = tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
            ids = [remap.get(i, i) for i in tok.encode(text, add_special_tokens=False)]
            if not ids or ids[-1] != reward_id:
                ids.append(reward_id)
            lens.append(len(ids))
    return lens


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    ap.add_argument("--combine", action="store_true")
    args = ap.parse_args()

    if args.combine:
        combine()
        return

    repo = args.only
    assert repo in FULL_SET_OK, f"{repo} cannot be validly scored on the full 600 (context limit)"
    res = score_full(repo)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    p = RESULTS_DIR / f"rm_full_{repo.replace('/', '__')}.json"
    with open(p, "w") as f:
        json.dump(res, f)
    print(f"{SHORT[repo]}: scored {res['n']} full-set pairs in {res['wall_clock_s']:.0f}s -> {p.name}")


def combine() -> None:
    import pandas as pd
    filt = json.load(open(RESULTS_DIR / "rm_probe_filter.json"))
    retained_idx = filt["retained_indices"]

    full, filtered = {}, {}
    for repo in FULL_SET_OK:
        p = RESULTS_DIR / f"rm_full_{repo.replace('/', '__')}.json"
        if not p.exists():
            print(f"missing {p.name}; run --only '{repo}'")
            continue
        full[repo] = np.array(json.load(open(p))["d_m_full"])
    for repo in FULL_SET_OK + [OA]:
        p = RESULTS_DIR / f"rm_harness_{repo.replace('/', '__')}.json"
        if p.exists():
            r = json.load(open(p))["result"]
            if "d_m" in r:
                filtered[repo] = np.array(r["d_m"])

    names = [SHORT[r] for r in FULL_SET_OK if r in full and r in filtered]
    keep = [r for r in FULL_SET_OK if r in full and r in filtered]
    if len(keep) < 2:
        print("not enough models with both arms")
        return

    # filtered-subset correlations (already on retained probes)
    Dfilt = np.vstack([filtered[r] for r in keep])
    Cfilt = np.corrcoef(Dfilt)
    # full-set correlations
    Dfull = np.vstack([full[r] for r in keep])
    Cfull = np.corrcoef(Dfull)
    # full-set restricted to the retained rows -> isolates "which probes" from "which rows"
    Dfull_ret = np.vstack([full[r][retained_idx] for r in keep])
    Cfull_ret = np.corrcoef(Dfull_ret)

    Pfilt = np.degrees(np.arccos(np.clip(Cfilt, -1, 1)))
    Pfull = np.degrees(np.arccos(np.clip(Cfull, -1, 1)))

    print("=== pairwise margin correlation: FULL 600 ===")
    print(pd.DataFrame(Cfull, index=names, columns=names).round(3).to_string())
    print("\n=== pairwise margin correlation: FILTERED 487 ===")
    print(pd.DataFrame(Cfilt, index=names, columns=names).round(3).to_string())
    print("\n=== delta (filtered - full), correlation ===")
    print(pd.DataFrame(Cfilt - Cfull, index=names, columns=names).round(3).to_string())
    print("\n=== psi_eff delta (filtered - full), degrees ===")
    print(pd.DataFrame(Pfilt - Pfull, index=names, columns=names).round(2).to_string())

    iu = np.triu_indices(len(keep), k=1)
    corr_deltas = np.abs((Cfilt - Cfull)[iu])
    psi_deltas = np.abs((Pfilt - Pfull)[iu])
    print(f"\nmax |corr delta| = {corr_deltas.max():.4f}   max |psi_eff delta| = {psi_deltas.max():.2f} deg")

    # consistency check: same probes, scored in each arm, should agree exactly
    agree = {SHORT[r]: float(np.max(np.abs(filtered[r] - full[r][retained_idx]))) for r in keep}
    print(f"same-probe score agreement across arms (max abs diff): {agree}")

    material = bool(psi_deltas.max() > 5.0)
    print(f"\nDoes the length filter materially change pool geometry? "
          f"{'YES' if material else 'NO'} (threshold: any pairwise psi_eff shift > 5 deg)")

    out = {"names": names, "corr_full": Cfull.tolist(), "corr_filtered": Cfilt.tolist(),
           "corr_full_restricted_to_retained": Cfull_ret.tolist(),
           "psi_full": Pfull.tolist(), "psi_filtered": Pfilt.tolist(),
           "max_abs_corr_delta": float(corr_deltas.max()),
           "max_abs_psi_delta_deg": float(psi_deltas.max()),
           "same_probe_agreement_max_abs": agree,
           "materially_changes_geometry": material,
           "openassistant_excluded_from_full": "context 512 < required; never truncated to force a comparison"}
    with open(RESULTS_DIR / "rm_bias_check.json", "w") as f:
        json.dump(out, f)
    print(f"\nwrote {RESULTS_DIR / 'rm_bias_check.json'}")


if __name__ == "__main__":
    main()
