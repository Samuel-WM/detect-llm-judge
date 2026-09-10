"""Probe selection strategies. `random` takes the pool as-is (the pool itself is freshly drawn
i.i.d. each replicate, so a prefix slice is already a random sample). `disagreement_max` scores
each candidate pool row by the worst-case pairwise candidate-margin separation and keeps the
top N — a real auditor only has noisy margins to select on, so selection must operate on the
noisy `D`, never the clean one.
"""
import numpy as np


def select_random(N: int) -> np.ndarray:
    return np.arange(N)


def select_disagreement_max(D_noisy: np.ndarray, N: int) -> np.ndarray:
    """score_i = min_{m != m'} |d_tilde_{m,i} - d_tilde_{m',i}|; returns the top-N pool indices
    by this per-row score, maximizing the worst-case candidate separation on each kept probe.

    Kept as the "per_row_topk" ablation (NEXT_STEPS_TRACKS_A_B_C.md A4): this objective has no
    cross-row interaction term, so it can spend the whole budget resolving the single easiest
    pair of candidates while leaving other pairs unresolved. `select_pairwise_coverage` below
    is the fix.
    """
    M = D_noisy.shape[1]
    iu = np.triu_indices(M, k=1)
    pairwise_abs_diff = np.abs(D_noisy[:, iu[0]] - D_noisy[:, iu[1]])  # (pool_size, C(M,2))
    scores = pairwise_abs_diff.min(axis=1)
    n = min(N, len(scores))
    top_idx = np.argpartition(-scores, n - 1)[:n]
    return top_idx


def select_pairwise_coverage(D_noisy: np.ndarray, N: int) -> np.ndarray:
    """Greedy coverage across candidate pairs (NEXT_STEPS_TRACKS_A_B_C.md A4). Maintains
    accumulated separation S_{m,m'} = sum over selected i of |d_tilde_{m,i} - d_tilde_{m',i}|
    per candidate pair; at each step, finds the currently weakest pair (smallest accumulated S)
    and picks the pool row that maximizes *that* pair's separation. Unlike per-row top-k, this
    can't spend the whole budget on one easy pair -- once a pair is well covered it stops being
    the weakest link and attention moves to the next.
    """
    pool_size, M = D_noisy.shape
    iu = np.triu_indices(M, k=1)
    pair_abs_diff = np.abs(D_noisy[:, iu[0]] - D_noisy[:, iu[1]])  # (pool_size, C(M,2))
    n_pairs = pair_abs_diff.shape[1]

    n = min(N, pool_size)
    S = np.zeros(n_pairs)
    selected_mask = np.zeros(pool_size, dtype=bool)
    order = np.empty(n, dtype=np.int64)
    for step in range(n):
        weakest_pair = int(np.argmin(S))
        candidate_scores = pair_abs_diff[:, weakest_pair].copy()
        candidate_scores[selected_mask] = -np.inf
        i_star = int(np.argmax(candidate_scores))
        selected_mask[i_star] = True
        order[step] = i_star
        S[weakest_pair] += pair_abs_diff[i_star, weakest_pair]
    return order
