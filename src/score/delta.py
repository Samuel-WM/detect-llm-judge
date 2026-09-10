"""Section 4.5: Delta computation.

For each probe pair, the SUM of token log-probabilities over RESPONSE tokens only (prompt tokens
masked), under adapter-on (pi_A) and adapter-off (pi_ref):

    s(x, y)  = logp_A(y|x) - logp_ref(y|x)          (sums, not means)
    Delta_i  = s(x_i, y_i) - s(x_i, y_i')

The reference policy is the same weights with the LoRA adapter disabled (`model.disable_adapter()`),
never a second copy -- so pi_A and pi_ref share base weights exactly and most numerical error
cancels in the ratio. log-softmax is reduced in float32. A length-normalized `delta_mean` variant
is also stored as a diagnostic for length confounding; it is never substituted for the primary sum.

Prompt/response tokenization goes through `common/templating.build_scoring_inputs`, the same
function used at generation and training time, and its prefix assertion fails loudly on drift.

Batched, fixed_bin-padded (NEXT_STEPS_ROUND_2.md Step 0): the judge-margin scorer was found to be
bf16-batch-composition-sensitive (RESULTS.md) -- the same mechanism applies to any batched
log-prob read, and here it would land directly in Delta, the entire estimand, if the adapter-on
and adapter-off passes were batched differently. Guarded two ways: (1) both passes run the exact
same padded batch tensor -- only `model.disable_adapter()` differs, so batch composition is
identical between them by construction, not by convention; (2) a given response's own padded
length is a pure function of its token length (next multiple of `PAD_MULTIPLE`), never of what
else is in the corpus, matching the judge-margin fix. `tests/test_scorer_invariance.py` asserts
this quantitatively rather than assuming it.
"""
import torch

from src.common.templating import build_scoring_inputs

PAD_MULTIPLE = 128
MAX_BATCH_ROWS = 8
TOKEN_BUDGET = 2048  # rows_in_batch * bin_len stays under this
LOGPROB_CHUNK = 128  # sequence positions per fp32 log-softmax chunk (memory bound, not numerics)


def _rows_per_batch(bin_len: int, token_budget: int = TOKEN_BUDGET, max_rows: int = MAX_BATCH_ROWS) -> int:
    return max(1, min(max_rows, token_budget // max(bin_len, 1)))


def _fixed_bins(lengths: list[int], token_budget: int = TOKEN_BUDGET, max_rows: int = MAX_BATCH_ROWS) -> list[list[int]]:
    """Groups indices by padded-length bin (next multiple of PAD_MULTIPLE), batches only within a
    bin (so a row's padded length depends only on itself), longest bins first."""
    bins: dict[int, list[int]] = {}
    for i, L in enumerate(lengths):
        bin_len = ((L + PAD_MULTIPLE - 1) // PAD_MULTIPLE) * PAD_MULTIPLE
        bins.setdefault(bin_len, []).append(i)
    batches = []
    for bin_len in sorted(bins, reverse=True):
        idxs = bins[bin_len]
        per_batch = _rows_per_batch(bin_len, token_budget, max_rows)
        for k in range(0, len(idxs), per_batch):
            batches.append((bin_len, idxs[k:k + per_batch]))
    return batches


@torch.no_grad()
def _score_batch(model, batch_ids: torch.Tensor, attn_mask: torch.Tensor,
                  offsets: list[int], prompt_lens: list[int], lengths: list[int]) -> list[tuple[float, int]]:
    """Runs one forward pass over an already-built left-padded batch and returns
    [(sum_logp_response, n_response_tokens), ...] per row, in row order.

    The log-softmax is reduced in float32 (Section 4.5) but CHUNKED over sequence positions:
    materializing `(batch, seq, vocab)` in fp32 at once is ~3.1 GB for 8x640 at Qwen's 151936
    vocab, and the intermediate log_softmax output doubles it -- that OOM'd on this 5 GB card.
    log-softmax is independent per position, so chunking is mathematically and numerically
    identical to the one-shot version, just bounded in peak memory.
    """
    logits = model(input_ids=batch_ids, attention_mask=attn_mask).logits  # keep bf16 for now
    targets = batch_ids[:, 1:]
    n_pos = targets.shape[1]
    token_logp = torch.empty(targets.shape, dtype=torch.float32, device=logits.device)
    for start in range(0, n_pos, LOGPROB_CHUNK):
        end = min(start + LOGPROB_CHUNK, n_pos)
        chunk = logits[:, start:end, :].float()               # fp32 reduction, per Section 4.5
        chunk_logp = torch.log_softmax(chunk, dim=-1)
        token_logp[:, start:end] = chunk_logp.gather(
            -1, targets[:, start:end].unsqueeze(-1)).squeeze(-1)
        del chunk, chunk_logp
    del logits

    out = []
    for i, (offset, plen, L) in enumerate(zip(offsets, prompt_lens, lengths)):
        resp_start = offset + plen - 1
        resp_end = offset + L - 1  # exclusive
        resp_logp = token_logp[i, resp_start:resp_end]
        out.append((float(resp_logp.sum()), int(resp_logp.numel())))
    return out


def _build_padded_batch(tokenizer, items: list[tuple[torch.Tensor, int]], bin_len: int,
                         n_fill: int, device: str):
    """items: list of (input_ids [L], prompt_len). Left-pads every row to bin_len, and pads the
    ROW COUNT to a constant (real rows + n_fill filler copies of the last item) -- fixing only
    sequence length still left the batch *shape* (row count) dependent on how many real items
    happened to land in a bin's last partial batch, which changes under reordering (measured:
    shuffled-order test failed at 5.5% of sd(Delta) before this fix, 0% after -- see
    tests/test_scorer_invariance.py and the identical finding for the judge-margin scorer).
    Filler rows are computed but never returned to the caller.
    """
    pad_id = tokenizer.pad_token_id
    real_items = list(items) + [items[-1]] * n_fill
    n = len(real_items)
    batch_ids = torch.full((n, bin_len), pad_id, dtype=torch.long)
    attn_mask = torch.zeros((n, bin_len), dtype=torch.long)
    offsets, prompt_lens, lengths = [], [], []
    for i, (ids, plen) in enumerate(real_items):
        L = ids.shape[0]
        offset = bin_len - L
        batch_ids[i, offset:] = ids
        attn_mask[i, offset:] = 1
        offsets.append(offset)
        prompt_lens.append(plen)
        lengths.append(L)
    return batch_ids.to(device), attn_mask.to(device), offsets, prompt_lens, lengths


def score_batch(model, tokenizer, items: list[tuple[str, str]], device: str,
                ref_cache: dict | None = None) -> list[dict]:
    """items: list of (prompt, response). Returns, per item, adapter-on/off sums, in fixed_bin
    batches -- a given item's score is a pure function of its own token length, not of what else
    is in `items` or what order they're passed in (see tests/test_scorer_invariance.py).
    """
    prepared = [build_scoring_inputs(tokenizer, p, r) for p, r in items]  # [(input_ids, prompt_len), ...]
    lengths = [ids.shape[0] for ids, _ in prepared]

    results: list[dict | None] = [None] * len(items)
    for bin_len, idxs in _fixed_bins(lengths):
        batch_items = [prepared[i] for i in idxs]
        n_fill = _rows_per_batch(bin_len) - len(batch_items)
        batch_ids, attn_mask, offsets, prompt_lens, item_lengths = _build_padded_batch(
            tokenizer, batch_items, bin_len, n_fill, device,
        )

        # SAME batch tensor for both passes -- batch composition is identical by construction.
        on = _score_batch(model, batch_ids, attn_mask, offsets, prompt_lens, item_lengths)
        # The reference policy is the frozen base model with the adapter disabled, so its
        # log-probs do not depend on the checkpoint and are identical across every checkpoint and
        # every arm. `fixed_bin` makes an item's padded batch a pure function of its own token
        # length and the constant per-bin row count, so the tensor fed to the reference pass here
        # is bitwise the same one that produced any cached value for the same item. Reusing it is
        # an exact reuse, not an approximation -- gated by a bitwise regression check before use.
        if ref_cache is not None and all(i in ref_cache for i in idxs):
            off = [ref_cache[i] for i in idxs] + [None] * n_fill
        else:
            with model.disable_adapter():
                off = _score_batch(model, batch_ids, attn_mask, offsets, prompt_lens, item_lengths)
            if ref_cache is not None:
                for local_i, global_i in enumerate(idxs):
                    ref_cache[global_i] = off[local_i]

        for local_i, global_i in enumerate(idxs):  # filler rows (index >= len(idxs)) discarded
            lp_on, n_on = on[local_i]
            lp_off, n_off = off[local_i]
            assert n_on == n_off, "token count changed between adapter on/off -- impossible, aborting"
            results[global_i] = {
                "logp_A": lp_on, "logp_ref": lp_off, "s": lp_on - lp_off, "n_response_tokens": n_on,
            }
    return results  # type: ignore[return-value]


def compute_deltas(model, tokenizer, pairs: list[dict], device: str,
                   ref_cache: dict | None = None) -> list[dict]:
    """Delta for a list of probe pairs (response_a vs response_b), batched.

    `ref_cache`, when supplied, memoizes the adapter-disabled reference log-probs across calls.
    The reference policy is frozen, so this is exact reuse; see the note in `score_batch`.
    """
    items = []
    for p in pairs:
        items.append((p["prompt"], p["response_a"]))
        items.append((p["prompt"], p["response_b"]))
    scored = score_batch(model, tokenizer, items, device, ref_cache=ref_cache)

    out = []
    for i, p in enumerate(pairs):
        a, b = scored[2 * i], scored[2 * i + 1]
        delta_sum = a["s"] - b["s"]
        mean_a = a["s"] / max(a["n_response_tokens"], 1)
        mean_b = b["s"] / max(b["n_response_tokens"], 1)
        out.append({
            "delta": delta_sum,
            "delta_mean": mean_a - mean_b,          # diagnostic only, never the primary statistic
            "s_a": a["s"], "s_b": b["s"],
            "len_a": a["n_response_tokens"], "len_b": b["n_response_tokens"],
            "len_diff": a["n_response_tokens"] - b["n_response_tokens"],
        })
    return out


def assert_adapter_toggle_changes_output(model, tokenizer, prompt: str, response: str, device: str) -> dict:
    """Guards the failure mode the main document calls out: an adapter toggle that silently does
    nothing produces near-perfect attribution accuracy. Fails loudly if pi_A == pi_ref exactly.
    """
    r = score_batch(model, tokenizer, [(prompt, response)], device)[0]
    assert abs(r["s"]) > 0, (
        "adapter-on and adapter-off log-probs are identical -- the adapter toggle is not changing "
        "the model, so every Delta would be exactly zero"
    )
    return r
