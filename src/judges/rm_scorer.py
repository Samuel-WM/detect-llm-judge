"""Scalar reward-model scorer for the calibration stage (NEXT_STEPS_ROUND_4_FINAL.md §2).

A scalar reward head is directly and deterministically observable:

    d_m(x, y, y') = r_m(x, y) - r_m(x, y')

so there is no A/B slot and position bias is structurally zero rather than something to subtract,
and evaluator measurement reliability is R = 1 by construction once determinism (H2) is verified
empirically rather than assumed.

Uses the same fixed_bin construction as the judge/Delta scorers: a sequence's padded length is a
pure function of its own token length, and the row count per batch is padded to a constant, so a
given input's score does not depend on what else is in the corpus. That property is what
`tests/test_scorer_invariance.py`-style checking (H3) verifies here.
"""
import torch

PAD_MULTIPLE = 128
MAX_BATCH_ROWS = 8
TOKEN_BUDGET = 4096


def serialize(prompt: str, response: str) -> str:
    """Single (prompt, response) serialization. Deliberately plain -- the standing prohibition
    forbids constructing an artificial second input template for reward models."""
    return f"{prompt}\n\n{response}"


def _rows_per_batch(bin_len: int, token_budget: int = TOKEN_BUDGET, max_rows: int = MAX_BATCH_ROWS) -> int:
    return max(1, min(max_rows, token_budget // max(bin_len, 1)))


def _fixed_bins(lengths: list[int]) -> list[tuple[int, list[int]]]:
    bins: dict[int, list[int]] = {}
    for i, L in enumerate(lengths):
        b = ((L + PAD_MULTIPLE - 1) // PAD_MULTIPLE) * PAD_MULTIPLE
        bins.setdefault(b, []).append(i)
    out = []
    for b in sorted(bins, reverse=True):
        idxs = bins[b]
        per = _rows_per_batch(b)
        for k in range(0, len(idxs), per):
            out.append((b, idxs[k:k + per]))
    return out


@torch.no_grad()
def score_texts(model, tokenizer, texts: list[str], device: str = "cuda",
                max_length: int | None = None) -> torch.Tensor:
    """Returns a scalar reward per text, in input order."""
    enc_lens = [len(tokenizer.encode(t, add_special_tokens=True)) for t in texts]
    if max_length is not None:
        enc_lens = [min(L, max_length) for L in enc_lens]
    out = torch.empty(len(texts), dtype=torch.float32)

    for bin_len, idxs in _fixed_bins(enc_lens):
        if max_length is not None:
            bin_len = min(bin_len, max_length)
        n_fill = _rows_per_batch(bin_len) - len(idxs)
        batch = [texts[i] for i in idxs] + [texts[idxs[-1]]] * n_fill
        enc = tokenizer(batch, return_tensors="pt", padding="max_length", truncation=True,
                        max_length=bin_len).to(device)
        logits = model(**enc).logits.float()          # (rows, 1) for a scalar head
        vals = logits.squeeze(-1) if logits.shape[-1] == 1 else logits[:, 0]
        for local_i, global_i in enumerate(idxs):
            out[global_i] = vals[local_i].item()
    return out


def score_pairs(model, tokenizer, pairs: list[dict], device: str = "cuda",
                max_length: int | None = None):
    """Returns (r_a, r_b, d_m) over probe pairs, where d_m = r(x,y) - r(x,y')."""
    texts_a = [serialize(p["prompt"], p["response_a"]) for p in pairs]
    texts_b = [serialize(p["prompt"], p["response_b"]) for p in pairs]
    r_a = score_texts(model, tokenizer, texts_a, device, max_length)
    r_b = score_texts(model, tokenizer, texts_b, device, max_length)
    return r_a, r_b, (r_a - r_b)
