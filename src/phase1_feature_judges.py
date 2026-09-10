"""Feature judges (CLAUDE_CODE_PROMPT.md Phase 1), used as Track C Arm A's exact-ground-truth
preference functions (NEXT_STEPS_ROUND_3_FINAL.md 1.1).

Judges are explicit linear rewards over cheap handcrafted response features, so `w_m` -- and
therefore the true preference-generating margin `d_true` -- is known exactly. That is the whole
point of this arm: Arm B's LLM judge has no observable ground truth, so its continuous-margin
regression is only exploratory, while here the generating process is known.

    features:  token count, markdown-marker density, mean sentence length, hedging-word rate,
               list-item count, question-mark rate, type-token ratio  (standardized across corpus)
    judges:    five, r_m(y) = <w_m, phi(y)>, w_m constructed well-separated

Label processes (0.3): the PRIMARY arms use a Bradley-Terry process
`P(y > y') = sigmoid(tau * d_std)`, standardized with TRAINING-pair statistics only and reused
unchanged on probes. Deterministic `sign(d_true)` is kept as the designated contrast arm, because
deterministic binary labels do not identify continuous margin magnitude.

Run: .venv/Scripts/python.exe -m src.phase1_feature_judges --build
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "data_cache"
RESULTS_DIR = REPO_ROOT / "results"

FEATURE_NAMES = [
    "token_count", "markdown_marker_density", "mean_sentence_length", "hedging_word_rate",
    "list_item_count", "question_mark_rate", "type_token_ratio",
]

HEDGING_WORDS = {
    "might", "maybe", "perhaps", "possibly", "probably", "seems", "appears", "suggest",
    "suggests", "could", "generally", "typically", "often", "usually", "somewhat", "likely",
    "arguably", "presumably", "roughly", "approximately",
}


def extract_features(text: str) -> np.ndarray:
    words = re.findall(r"\b\w+\b", text.lower())
    n_words = max(len(words), 1)

    token_count = len(words)
    markdown_markers = len(re.findall(r"[*_`#>\[\]]|```", text))
    markdown_density = markdown_markers / max(len(text), 1) * 100

    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    mean_sentence_length = np.mean([len(s.split()) for s in sentences]) if sentences else 0.0

    hedging_rate = sum(1 for w in words if w in HEDGING_WORDS) / n_words * 100
    list_items = len(re.findall(r"^\s*(?:[-*+]|\d+[.)])\s+", text, flags=re.MULTILINE))
    question_rate = text.count("?") / n_words * 100
    type_token_ratio = len(set(words)) / n_words

    return np.array([
        token_count, markdown_density, mean_sentence_length, hedging_rate,
        list_items, question_rate, type_token_ratio,
    ], dtype=np.float64)


def build_feature_matrix(texts: list[str]) -> np.ndarray:
    return np.vstack([extract_features(t) for t in texts])


def standardize_features(phi: np.ndarray, mu: np.ndarray | None = None, sd: np.ndarray | None = None):
    if mu is None:
        mu = phi.mean(axis=0)
    if sd is None:
        sd = phi.std(axis=0)
        sd[sd == 0] = 1.0
    return (phi - mu) / sd, mu, sd


def build_judges(n_judges: int = 5, d: int = 7, seed: int = 0) -> np.ndarray:
    """Well-separated unit-norm weight vectors. Uses the first `n_judges` rows of an orthonormal
    basis of R^d (possible since n_judges <= d), giving exactly-orthogonal ground-truth judges --
    maximal, and equal, pairwise separation by construction.
    """
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    return q[:, :n_judges].T  # (n_judges, d), orthonormal rows


def select_judge(w: np.ndarray) -> int:
    """The judge whose w_m has the largest MINIMUM angle to the other four (1.1)."""
    n = w.shape[0]
    cos = w @ w.T
    np.fill_diagonal(cos, np.nan)
    min_angles = np.degrees(np.arccos(np.clip(np.nanmax(np.abs(cos), axis=1), -1, 1)))
    return int(np.argmax(min_angles)), min_angles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train_path = CACHE_DIR / "response_pairs_train.jsonl"
    assert train_path.exists(), "run src.gen.sample_pairs --prompts train first"
    probe_path = CACHE_DIR / "response_pairs_probe.jsonl"

    train_pairs = [json.loads(l) for l in open(train_path)]
    probe_pairs = [json.loads(l) for l in open(probe_path)]
    print(f"train pairs: {len(train_pairs)}   probe pairs: {len(probe_pairs)}")

    # Standardize features over the full corpus (train + probe responses), per Phase 1's
    # "standardized across the corpus".
    all_texts = []
    for p in train_pairs + probe_pairs:
        all_texts.extend([p["response_a"], p["response_b"]])
    phi_all = build_feature_matrix(all_texts)
    _, mu, sd = standardize_features(phi_all)
    print(f"feature standardization over {len(all_texts)} responses")
    for name, m, s in zip(FEATURE_NAMES, mu, sd):
        print(f"  {name:26s} mean={m:10.4f}  sd={s:10.4f}")

    w = build_judges(seed=args.seed)
    chosen, min_angles = select_judge(w)
    print(f"\njudge min-pairwise-angles (deg): {np.round(min_angles, 2)}")
    print(f"selected judge index {chosen} (largest minimum angle: {min_angles[chosen]:.2f} deg)")

    def margins(pairs):
        a = build_feature_matrix([p["response_a"] for p in pairs])
        b = build_feature_matrix([p["response_b"] for p in pairs])
        a_std = (a - mu) / sd
        b_std = (b - mu) / sd
        return (a_std - b_std) @ w.T, (a_std - b_std)  # (n, n_judges), (n, d)

    d_train_all, phidiff_train = margins(train_pairs)
    d_probe_all, phidiff_probe = margins(probe_pairs)
    d_true_train = d_train_all[:, chosen]
    d_true_probe = d_probe_all[:, chosen]

    # Standardization constants from TRAINING pairs only, reused unchanged on probes (0.3).
    mu_train = float(d_true_train.mean())
    sd_train = float(d_true_train.std())
    d_std_train = (d_true_train - mu_train) / sd_train
    d_std_probe = (d_true_probe - mu_train) / sd_train
    print(f"\nd_true train: mean={mu_train:.4f} sd={sd_train:.4f}")
    print(f"d_std_probe:  mean={d_std_probe.mean():.4f} sd={d_std_probe.std():.4f}  "
          f"(NOT renormalized on probes, by construction)")

    # Length confound reporting (1.1): the judge's own coefficient on standardized token count,
    # and the realized pairwise length loading -- reported as two distinct quantities.
    len_idx = FEATURE_NAMES.index("token_count")
    w_len = float(w[chosen, len_idx])
    len_diff_probe = phidiff_probe[:, len_idx]
    corr_len = float(np.corrcoef(d_true_probe, len_diff_probe)[0, 1])
    print(f"\nw_m[length] (coefficient on standardized token count) = {w_len:+.4f}")
    print(f"corr(d_true, len(y)-len(y')) on probes                 = {corr_len:+.4f}")

    out = {
        "seed": args.seed,
        "feature_names": FEATURE_NAMES,
        "feature_mu": mu.tolist(), "feature_sd": sd.tolist(),
        "w": w.tolist(), "selected_judge": chosen,
        "min_pairwise_angles_deg": min_angles.tolist(),
        "mu_train": mu_train, "sd_train": sd_train,
        "w_len_coefficient": w_len, "corr_d_true_len_diff_probe": corr_len,
        "d_true_train": d_true_train.tolist(), "d_std_train": d_std_train.tolist(),
        "d_true_probe": d_true_probe.tolist(), "d_std_probe": d_std_probe.tolist(),
        "len_diff_probe": len_diff_probe.tolist(),
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CACHE_DIR / "feature_judges.json"
    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
