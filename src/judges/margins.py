"""Section 4.3: judge margin extraction. No generation from the judge -- read logits.

Batched forward passes per (template, order): comparison prompt presents the prompt and two
responses labeled A/B, ending so the next token is the verdict; reads logits at the final
position for the tokens "A"/"B" (leading-space variant handled, asserted single-token).

    ell_AB = logp(A) - logp(B)   with y=A=response_a, y'=B=response_b
    ell_BA = logp(B) - logp(A)   with y=B=response_a, y'=A=response_b   (swapped labeling)
    d_m    = 0.5 * (ell_AB + ell_BA)     position-debiased margin
    z_m    = 1[d_m > 0]                   binary preference (DPO label)

Batched, length-bucketed (batch_size=32), left-padded -- same fix as gen/sample_pairs.py after
its single-sequence generation was measured at ~7-9 tok/s from WDDM per-kernel-launch overhead.
A forward pass is one launch (not one per generated token), so the effect is smaller here, but
still real (~4.7 calls/s at batch=1, measured directly), and each judge needs
600 pairs x 2 templates x 2 orders = 2400 forward calls -- batching is cheap insurance,
especially given generation separately showed this GPU can slow down by ~27x mid-job (see
RESULTS.md; still investigating whether that's thermal throttling).

Loops outer over judges (load one at a time, `del` + reclaim before the next -- Section 1 rule
2), inner over probe pairs, matched then mismatched template. Checkpointed per judge.

Run: .venv/Scripts/python.exe -m src.judges.margins --pairs probe --n 600 --template both
"""
import argparse
import json
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common.vram import assert_baseline, free_vram_gb, reclaim
from src.data.build_prompts import load_cfg
from tests.test_templating import run as run_templating_check

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = REPO_ROOT / "data_cache"
MAX_BATCH_ROWS = 32
TOKEN_BUDGET = 4096  # rows_in_batch * max_len_in_batch stays under this (padding cost dominates)

# Padding-free scoring. Measured directly (src/judges/validity_padding_probe.py): in bf16, how
# much left-padding a sequence receives shifts the A/B verdict log-prob difference by up to 0.25
# (median 0.125), while the identical computation in float32 shifts by 5e-5. The masking/position
# path is therefore mathematically correct -- this is purely bf16 kernel numerics varying with
# tensor shape. But 0.125 is ~13% of Qwen's margin SD and ~47-49% of SmolLM2's and TinyLlama's,
# i.e. batch composition was injecting a noise term comparable to the signal for the smaller
# judges. With PAD_FREE, every sequence is scored alone with no padding at all, making margins
# exactly reproducible and independent of what else happens to be in the corpus. Costs ~1.53x
# the batched path (measured) -- these sequences are long enough to be compute-bound, so batching
# was buying much less than it does for token-by-token generation.
# PAD_MODE:
#   "free"      one sequence per batch, zero padding. Exact, but measured ~5.5x the cost of
#               batched scoring across the real length distribution (471s / 200 pairs).
#   "fixed_bin" pad every sequence to the next multiple of PAD_MULTIPLE and batch only sequences
#               sharing a bin. A given sequence then always receives the SAME padded length no
#               matter what else is in the corpus, so margins are reproducible and
#               corpus-independent (which is what the D2 train-vs-probe comparison needs) at
#               close to the original batched speed. Residual difference from the exact
#               padding-free value is measured, not assumed -- see
#               src/judges/validity_padding_probe.py and the padding-mode agreement check.
#   "legacy"    original token-budget bucketing; batch composition depends on the whole corpus.
PAD_MODE = "fixed_bin"
PAD_MULTIPLE = 128
PAD_FREE = False  # retained for the explicit padding-free reference runs


def get_answer_token_ids(tokenizer) -> dict[str, int]:
    """Resolves 'A' and 'B' to single-token ids for this tokenizer, trying the leading-space
    variant if the bare letter isn't a single token. Fails loudly if neither variant is (Section
    4.3's explicit assertion requirement).
    """
    out = {}
    for letter in ("A", "B"):
        resolved = None
        for cand in (letter, " " + letter):
            ids = tokenizer.encode(cand, add_special_tokens=False)
            if len(ids) == 1:
                resolved = ids[0]
                break
        assert resolved is not None, (
            f"neither '{letter}' nor ' {letter}' is a single token for tokenizer "
            f"{tokenizer.name_or_path!r} -- Section 4.3 requires single-token verdict letters"
        )
        out[letter] = resolved
    return out


def build_comparison_text(tokenizer, template: str, prompt: str, response_a: str, response_b: str) -> str:
    content = template.format(prompt=prompt, response_a=response_a, response_b=response_b)
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False,
    )


def _rows_per_batch(bin_len: int, token_budget: int = TOKEN_BUDGET, max_rows: int = MAX_BATCH_ROWS) -> int:
    """Rows per batch for a given padded-length bin. A pure function of the bin, so both the
    batcher and the filler agree and the tensor shape is identical on every run."""
    return max(1, min(max_rows, token_budget // max(bin_len, 1)))


def length_bucketed_batches(lengths: list[int], token_budget: int = TOKEN_BUDGET, max_rows: int = MAX_BATCH_ROWS) -> list[list[int]]:
    """Sorts by length (so batches are near-uniform-length, minimizing padding waste), then
    greedily packs a batch until adding the next row would push rows_in_batch * max_len_so_far
    over `token_budget` (a proxy for the padded batch's total token count, which is what
    actually drives memory) or `max_rows` is hit -- unlike a fixed batch size, this keeps short
    sequences batched large and long sequences batched small automatically.
    """
    if PAD_MODE == "fixed_bin":
        # Group by padded-length bin; batch only within a bin so every member is padded to
        # exactly the same length it would get on its own. Longest bins first, so the largest
        # activation block is allocated once up front and shorter bins reuse it (avoids the
        # allocator thrashing documented below).
        bins: dict[int, list[int]] = {}
        for i, L in enumerate(lengths):
            bin_len = ((L + PAD_MULTIPLE - 1) // PAD_MULTIPLE) * PAD_MULTIPLE
            bins.setdefault(bin_len, []).append(i)
        batches = []
        for bin_len in sorted(bins, reverse=True):
            idxs = bins[bin_len]
            per_batch = _rows_per_batch(bin_len, token_budget, max_rows)
            for k in range(0, len(idxs), per_batch):
                batches.append(idxs[k:k + per_batch])
        return batches

    if PAD_FREE or PAD_MODE == "free":
        # One sequence per batch -> no padding anywhere, exactly reproducible margins. Iterate in
        # length-sorted order: with a single sequence per batch there is no padding interaction,
        # so ordering cannot change any result, but it makes the allocation sizes monotone rather
        # than random. That matters here because `expandable_segments` is unavailable on Windows,
        # so the caching allocator fragments under randomly-varying shapes -- the per-batch log
        # from the first padding-free run showed reserved memory climbing 3.49 -> 4.29 GB with
        # per-batch time swinging 650 -> 12689 ms as it approached the ~5 GB ceiling.
        # Descending, specifically: the largest activation block is allocated on the very first
        # forward, and every subsequent (shorter) sequence is served from that already-cached
        # block instead of triggering a fresh cudaMalloc. Ascending would do the opposite --
        # a monotonically growing series of allocations that the freed smaller blocks cannot
        # satisfy, which is the thrashing pattern the log above shows.
        order = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
        return [[i] for i in order]
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batches = []
    current: list[int] = []
    current_max_len = 0
    for idx in order:
        candidate_max_len = max(current_max_len, lengths[idx])
        if current and (len(current) + 1) * candidate_max_len > token_budget:
            batches.append(current)
            current, current_max_len = [], 0
            candidate_max_len = lengths[idx]
        current.append(idx)
        current_max_len = candidate_max_len
        if len(current) >= max_rows:
            batches.append(current)
            current, current_max_len = [], 0
    if current:
        batches.append(current)
    return batches


def judge_margins_batched(
    model, tokenizer, template: str, pairs: list[dict], ans_ids: dict[str, int], device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (ell_AB, ell_BA, d_m, z_m), each shape (len(pairs),), in the same order as `pairs`."""
    n = len(pairs)
    texts_ab = [build_comparison_text(tokenizer, template, p["prompt"], p["response_a"], p["response_b"]) for p in pairs]
    texts_ba = [build_comparison_text(tokenizer, template, p["prompt"], p["response_b"], p["response_a"]) for p in pairs]
    # Bin on the per-pair MAX over both orderings: the swapped text is the same content but can
    # tokenize a token or two longer, and padding to a bin derived from the AB text alone could
    # silently truncate the BA text. Taking the max keeps the bin a property of the pair.
    lengths = [max(len(tokenizer.encode(a)), len(tokenizer.encode(b)))
               for a, b in zip(texts_ab, texts_ba)]

    ell_AB = np.zeros(n)
    ell_BA = np.zeros(n)
    batches = length_bucketed_batches(lengths)
    batch_log = []
    t_start = time.time()
    for b_i, batch_idx in enumerate(batches):
        t_batch = time.time()
        # logits_to_keep=1: only compute the LM head for the final position. Without this, HF
        # materializes logits for every position of every sequence in the batch -- with a ~150k
        # vocab and sequences up to ~1500 tokens (full prompt + two 256-token responses), that's
        # tens of GB for a batch of 32 and was the actual OOM cause, not the batch size itself.
        if PAD_MODE == "fixed_bin":
            # Every member of this batch shares a bin, so this bin length is a property of the
            # sequence itself, not of the batch it landed in.
            bin_len = ((max(lengths[i] for i in batch_idx) + PAD_MULTIPLE - 1) // PAD_MULTIPLE) * PAD_MULTIPLE
            pad_kwargs = {"padding": "max_length", "max_length": bin_len}
            # ...and pad the ROW count to a constant too. Measured: fixing only the sequence
            # length still left occasional 0.25 discrepancies, because the final partial batch of
            # a bin has a different row count in a 24-pair corpus than in a 200-pair one, and a
            # different (rows, len) shape selects a different bf16 kernel. With both dimensions
            # fixed, a sequence's margin no longer depends on what else is in the corpus.
            n_fill = _rows_per_batch(bin_len) - len(batch_idx)
        else:
            pad_kwargs = {"padding": True}
            n_fill = 0

        batch_ab = [texts_ab[i] for i in batch_idx] + [texts_ab[batch_idx[-1]]] * n_fill
        enc_ab = tokenizer(batch_ab, return_tensors="pt", add_special_tokens=False, **pad_kwargs).to(device)
        with torch.no_grad():
            logits_ab = model(**enc_ab, logits_to_keep=1).logits[:, -1, :].float()
        logp_ab = torch.log_softmax(logits_ab, dim=-1)
        vals_ab = (logp_ab[:, ans_ids["A"]] - logp_ab[:, ans_ids["B"]]).cpu().numpy()

        batch_ba = [texts_ba[i] for i in batch_idx] + [texts_ba[batch_idx[-1]]] * n_fill
        enc_ba = tokenizer(batch_ba, return_tensors="pt", add_special_tokens=False, **pad_kwargs).to(device)
        with torch.no_grad():
            logits_ba = model(**enc_ba, logits_to_keep=1).logits[:, -1, :].float()
        logp_ba = torch.log_softmax(logits_ba, dim=-1)
        vals_ba = (logp_ba[:, ans_ids["B"]] - logp_ba[:, ans_ids["A"]]).cpu().numpy()

        for local_i, global_i in enumerate(batch_idx):
            ell_AB[global_i] = vals_ab[local_i]
            ell_BA[global_i] = vals_ba[local_i]

        # Per-batch wall-clock + memory + telemetry logging, per NEXT_STEPS_FINAL.md's
        # "Loose end: runtime anomaly" -- so any recurrence of the ~27x generation slowdown can
        # be localized to a specific batch rather than inferred after the fact.
        if device == "cuda":
            batch_log.append({
                "batch": b_i, "rows": len(batch_idx), "max_len": int(enc_ab["input_ids"].shape[1]),
                "seconds": round(time.time() - t_batch, 4),
                "mem_alloc_gb": round(torch.cuda.memory_allocated() / 1024**3, 3),
                "mem_reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 3),
            })
            if b_i % 100 == 0 and b_i > 0:
                recent = batch_log[-100:]
                print(f"      batch {b_i}/{len(batches)}: "
                      f"{sum(r['seconds'] for r in recent)/len(recent)*1000:.0f} ms/batch (last 100), "
                      f"elapsed {time.time()-t_start:.0f}s, "
                      f"reserved {batch_log[-1]['mem_reserved_gb']:.2f}GB", flush=True)

    if batch_log:
        log_path = REPO_ROOT / "logs" / "margins_batch_timing.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as lf:
            for rec in batch_log:
                lf.write(json.dumps(rec) + "\n")

    d_m = 0.5 * (ell_AB + ell_BA)
    z_m = (d_m > 0).astype(int)
    return ell_AB, ell_BA, d_m, z_m


def load_probe_pairs(pairs_name: str, n: int | None) -> list[dict]:
    path = CACHE_DIR / f"response_pairs_{pairs_name}.jsonl"
    assert path.exists(), f"{path} does not exist -- run src.gen.sample_pairs first"
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    if n is not None:
        rows = rows[:n]
    return rows


def run_one_judge(judge_name: str, pairs: list[dict], templates: dict[str, str], out_path: Path) -> None:
    # A template only counts as done if EVERY pair has a row -- a template with a partial count
    # (e.g. from an interrupted run) must be treated as not-done and fully redone, since there's
    # no per-pair index to resume from mid-template (judge_margins_batched processes all pairs
    # for a template in one call).
    template_counts: dict[str, int] = {}
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                row = json.loads(line)
                template_counts[row["template_name"]] = template_counts.get(row["template_name"], 0) + 1
        done_templates = {t for t, count in template_counts.items() if count == len(pairs)}
        partial_templates = {t: c for t, c in template_counts.items() if c != len(pairs)}
        if partial_templates:
            print(f"  {judge_name}: discarding partial template rows (not a full {len(pairs)}-pair set): {partial_templates}")
            kept_rows = []
            with open(out_path) as f:
                for line in f:
                    row = json.loads(line)
                    if row["template_name"] in done_templates:
                        kept_rows.append(line)
            with open(out_path, "w") as f:
                f.writelines(kept_rows)
        print(f"  resuming {judge_name}: templates already done = {done_templates}")
    else:
        done_templates = set()

    remaining_templates = {k: v for k, v in templates.items() if k not in done_templates}
    if not remaining_templates:
        print(f"  {judge_name}: nothing to do, all requested templates already done")
        return

    baseline_vram = free_vram_gb()
    tokenizer = AutoTokenizer.from_pretrained(judge_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        judge_name, dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to("cuda")
    model.eval()
    ans_ids = get_answer_token_ids(tokenizer)

    t0 = time.time()
    n_done = 0
    with open(out_path, "a") as f:
        for template_name, template_str in remaining_templates.items():
            t_tmpl = time.time()
            ell_AB, ell_BA, d_m, z_m = judge_margins_batched(
                model, tokenizer, template_str, pairs, ans_ids, "cuda",
            )
            for idx in range(len(pairs)):
                f.write(json.dumps({
                    "prompt_idx": idx, "template_name": template_name, "judge": judge_name,
                    "ell_AB": float(ell_AB[idx]), "ell_BA": float(ell_BA[idx]),
                    "d_m": float(d_m[idx]), "z_m": int(z_m[idx]),
                }) + "\n")
            n_done += len(pairs)
            print(f"    template={template_name}: {len(pairs)} pairs in {time.time()-t_tmpl:.1f}s")
        f.flush()

    del model
    after = reclaim()
    print(f"  {judge_name}: {n_done} new rows in {time.time()-t0:.1f}s, "
          f"VRAM baseline={baseline_vram:.3f}GB after={after:.3f}GB")
    assert_baseline(baseline_vram, tol_gb=0.5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", choices=["train", "probe"], required=True)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--template", choices=["train", "probe", "both"], default="both")
    parser.add_argument("--judges", nargs="*", default=None,
                        help="subset of the configured judge pool to score; defaults to all")
    parser.add_argument("--tag", default=None,
                        help="output file tag, e.g. 'padfree'; keeps re-scored runs separate from "
                             "the original padded Track B files instead of clobbering them")
    args = parser.parse_args()

    cfg = load_cfg()
    pairs = load_probe_pairs(args.pairs, args.n)
    print(f"loaded {len(pairs)} response pairs from {args.pairs}")

    templates_all = cfg["judge_templates"]
    if args.template == "both":
        templates = {"template_train": templates_all["template_train"], "template_probe": templates_all["template_probe"]}
    else:
        key = f"template_{args.template}"
        templates = {key: templates_all[key]}

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    failures = {}
    judge_list = args.judges if args.judges else cfg["models"]["judge_pool"]
    for judge_name in judge_list:
        print(f"judge: {judge_name}")
        try:
            run_templating_check(judge_name)
            safe_name = judge_name.replace("/", "__")
            prefix = f"judge_margins_{args.tag}_" if args.tag else "judge_margins_"
            out_path = CACHE_DIR / f"{prefix}{safe_name}.jsonl"
            run_one_judge(judge_name, pairs, templates, out_path)
        except Exception as e:
            # A gated-repo 401 (or any other single-judge failure) must not take down the rest
            # of the pool -- outer loop is judges, and one inaccessible judge shouldn't block
            # the others from being measured.
            print(f"  FAILED: {type(e).__name__}: {str(e)[:300]}")
            failures[judge_name] = f"{type(e).__name__}: {str(e)[:300]}"
            reclaim()

    if failures:
        print(f"\n{len(failures)}/{len(cfg['models']['judge_pool'])} judges failed:")
        for name, err in failures.items():
            print(f"  {name}: {err}")


if __name__ == "__main__":
    main()
