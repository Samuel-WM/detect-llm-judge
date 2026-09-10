"""Track B report: B1 (judge separation), B2 (judge noise sigma_d), B3 (read the Phase 0c map).
NEXT_STEPS_TRACKS_A_B_C.md. Reads data_cache/judge_margins_*.jsonl; never re-runs a judge.

Run: .venv/Scripts/python.exe -m src.analysis.track_b_report
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = REPO_ROOT / "data_cache"
RESULTS_DIR = REPO_ROOT / "results"


def load_cfg() -> dict:
    with open(REPO_ROOT / "configs" / "base.yaml") as f:
        return yaml.safe_load(f)


def load_all_judge_margins() -> pd.DataFrame:
    rows = []
    for path in CACHE_DIR.glob("judge_margins_*.jsonl"):
        with open(path) as f:
            for line in f:
                rows.append(json.loads(line))
    assert rows, f"no judge_margins_*.jsonl files found in {CACHE_DIR}"
    return pd.DataFrame(rows)


def find_overlength_indices(cfg: dict, judge_name: str) -> dict[str, set[int]]:
    """For a judge whose context window is shorter than the longest comparison prompt, the
    margin at that (prompt_idx, template) is not reliable -- positions beyond
    model_max_length either error or silently degrade (TinyLlama-1.1B: 2048 tokens, 2-3 of 600
    pairs exceed it per template, ~0.3-0.5%). Returns {template_name: {prompt_idx, ...}} to
    exclude from every downstream statistic for this judge.
    """
    tokenizer = AutoTokenizer.from_pretrained(judge_name)
    max_len = tokenizer.model_max_length
    if max_len is None or max_len > 100_000:  # sentinel for "no real limit" in some tokenizers
        return {}

    pairs_path = CACHE_DIR / "response_pairs_probe.jsonl"
    with open(pairs_path) as f:
        pairs = [json.loads(line) for line in f]

    out = {}
    for template_name, template_str in cfg["judge_templates"].items():
        over = set()
        for idx, p in enumerate(pairs):
            content = template_str.format(prompt=p["prompt"], response_a=p["response_a"], response_b=p["response_b"])
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False,
            )
            if len(tokenizer.encode(text)) > max_len:
                over.add(idx)
        if over:
            out[template_name] = over
    return out


def filter_overlength(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    keep_mask = pd.Series(True, index=df.index)
    total_excluded = 0
    for judge_name in df["judge"].unique():
        overlength = find_overlength_indices(cfg, judge_name)
        for template_name, idx_set in overlength.items():
            bad = (df["judge"] == judge_name) & (df["template_name"] == template_name) & (df["prompt_idx"].isin(idx_set))
            total_excluded += int(bad.sum())
            keep_mask &= ~bad
        if overlength:
            counts = {t: len(s) for t, s in overlength.items()}
            print(f"  excluding over-context-length pairs for {judge_name}: {counts}")
    if total_excluded:
        print(f"  total rows excluded for context overflow: {total_excluded}")
    return df[keep_mask].copy()


def b1_judge_separation(df: pd.DataFrame, template_name: str) -> dict:
    sub = df[df["template_name"] == template_name]
    judges = sorted(sub["judge"].unique())
    piv = sub.pivot(index="prompt_idx", columns="judge", values="d_m")[judges]
    n_before = len(piv)
    piv = piv.dropna()  # keep only prompt_idx present for every judge (context-overflow filtering can create gaps)
    if len(piv) < n_before:
        print(f"  [{template_name}] dropped {n_before - len(piv)} prompt_idx not common to all judges")

    signs = np.sign(piv.values)
    M = len(judges)
    agreement = np.ones((M, M))
    for i in range(M):
        for j in range(M):
            agreement[i, j] = np.mean(signs[:, i] == signs[:, j])

    pearson = piv.corr(method="pearson").values
    spearman = piv.corr(method="spearman").values

    corr_for_pca = pearson
    eigvals = np.sort(np.linalg.eigvalsh(corr_for_pca))[::-1]
    var_share_pc1 = float(eigvals[0] / eigvals.sum())

    iu = np.triu_indices(M, k=1)
    pair_agreements = agreement[iu]
    implied_psi_deg = np.degrees(np.pi * (1.0 - pair_agreements))
    pair_names = [f"{judges[i]} vs {judges[j]}" for i, j in zip(*iu)]

    return {
        "judges": judges,
        "agreement_matrix": agreement,
        "pearson_matrix": pearson,
        "spearman_matrix": spearman,
        "eigvals": eigvals,
        "var_share_pc1": var_share_pc1,
        "pair_names": pair_names,
        "pair_agreements": pair_agreements,
        "implied_psi_deg": implied_psi_deg,
        "mean_implied_psi_deg": float(implied_psi_deg.mean()),
        "min_implied_psi_deg": float(implied_psi_deg.min()),
    }


def b2_judge_noise(df: pd.DataFrame) -> dict:
    judges = sorted(df["judge"].unique())
    per_judge = {}
    for judge in judges:
        sub = df[df["judge"] == judge]

        train = sub[sub["template_name"] == "template_train"].set_index("prompt_idx")
        b_pos = 0.5 * (train["ell_AB"] - train["ell_BA"])
        sd_d = train["d_m"].std()
        sigma_d_pos = float(b_pos.std() / sd_d) if sd_d > 0 else float("nan")

        probe = sub[sub["template_name"] == "template_probe"].set_index("prompt_idx")
        sigma_d_tmpl = float("nan")
        if len(probe) > 0:
            common_idx = train.index.intersection(probe.index)
            diff = probe.loc[common_idx, "d_m"] - train.loc[common_idx, "d_m"]
            sd_train = train.loc[common_idx, "d_m"].std()
            sigma_d_tmpl = float(diff.std() / sd_train) if sd_train > 0 else float("nan")

        per_judge[judge] = {
            "sigma_d_pos": sigma_d_pos, "sigma_d_tmpl": sigma_d_tmpl,
            "sigma_d_operating": np.nanmax([sigma_d_pos, sigma_d_tmpl]),
            "dominant": "position_bias" if sigma_d_pos >= (sigma_d_tmpl if not np.isnan(sigma_d_tmpl) else -np.inf) else "template_sensitivity",
        }
    return per_judge


def b3_read_the_map(min_psi_deg: float, sigma_d_operating: float) -> None:
    phase0c_path = RESULTS_DIR / "phase0c_synthetic.parquet"
    if not phase0c_path.exists():
        print("phase0c_synthetic.parquet not found, cannot read the map")
        return
    df = pd.read_parquet(phase0c_path)
    mc = df[df["section"] == "marginal_c_decoupled_d"]
    if len(mc) == 0:
        print("marginal_c_decoupled_d section not found in phase0c parquet")
        return

    sigma_grid = sorted(mc["sigma_d"].unique())
    nearest_sigma = min(sigma_grid, key=lambda s: abs(s - sigma_d_operating))
    print(f"\nB3: reading Marginal C decoupled table at sigma_d~{nearest_sigma} "
          f"(measured sigma_d_operating={sigma_d_operating:.3f}, nearest grid point)")
    print(f"    NOTE: Marginal C was run at psi=20deg; measured min_implied_psi={min_psi_deg:.1f}deg "
          f"-- read this as a proxy, not an exact match unless psi happens to align.")
    cell = mc[(mc["sigma_d"] == nearest_sigma)]
    for rule in ["sign_agreement", "bradley_terry", "nnls_mixture"]:
        acc = cell[cell["rule"] == rule]["top1_correct"].mean()
        gap = acc - 0.60
        print(f"    {rule}: predicted top-1 ~ {acc:.3f} ({'above' if gap >= 0 else 'below'} "
              f"the 0.60 contour by {abs(gap):.3f})")


def main() -> None:
    cfg = load_cfg()
    df = load_all_judge_margins()
    print("Checking for context-window overflow (judge tokenizer max length vs comparison prompt length)...")
    df = filter_overlength(df, cfg)
    judges_available = sorted(df["judge"].unique())
    all_judges = cfg["models"]["judge_pool"]
    missing = [j for j in all_judges if j not in judges_available]

    print(f"Judges available: {judges_available}")
    if missing:
        print(f"Judges MISSING (gated/inaccessible): {missing}")
        print(f"B1 matrices below are {len(judges_available)}x{len(judges_available)}, not "
              f"{len(all_judges)}x{len(all_judges)} -- incomplete pending access to the above.")

    print("\n=== B1: judge separation, template_train (matched) ===")
    b1_train = b1_judge_separation(df, "template_train")
    print("judges:", b1_train["judges"])
    print("pairwise sign-agreement matrix:")
    print(pd.DataFrame(b1_train["agreement_matrix"], index=b1_train["judges"], columns=b1_train["judges"]).round(3))
    print("\npearson correlation matrix:")
    print(pd.DataFrame(b1_train["pearson_matrix"], index=b1_train["judges"], columns=b1_train["judges"]).round(3))
    print("\nspearman correlation matrix:")
    print(pd.DataFrame(b1_train["spearman_matrix"], index=b1_train["judges"], columns=b1_train["judges"]).round(3))
    print(f"\neigenvalues of pearson corr matrix: {np.round(b1_train['eigvals'], 3)}")
    print(f"variance share of PC1: {b1_train['var_share_pc1']:.3f}")
    print("\nper-pair implied psi (degrees), from agreement = 1 - psi/pi:")
    for name, agr, psi in zip(b1_train["pair_names"], b1_train["pair_agreements"], b1_train["implied_psi_deg"]):
        print(f"  {name}: agreement={agr:.3f} -> implied_psi={psi:.1f} deg")
    print(f"\nmean implied psi: {b1_train['mean_implied_psi_deg']:.1f} deg")
    print(f"MIN implied psi (binding pair): {b1_train['min_implied_psi_deg']:.1f} deg")

    if "template_probe" in df["template_name"].unique():
        print("\n=== B1 (mismatched template, template_probe) ===")
        b1_probe = b1_judge_separation(df, "template_probe")
        print(f"mean implied psi: {b1_probe['mean_implied_psi_deg']:.1f} deg")
        print(f"MIN implied psi: {b1_probe['min_implied_psi_deg']:.1f} deg")

    print("\n=== B2: judge noise (sigma_d) ===")
    b2 = b2_judge_noise(df)
    for judge, stats in b2.items():
        print(f"  {judge}: sigma_d_pos={stats['sigma_d_pos']:.4f}  "
              f"sigma_d_tmpl={stats['sigma_d_tmpl']:.4f}  "
              f"operating={stats['sigma_d_operating']:.4f}  dominant={stats['dominant']}")
    sigma_d_pooled = float(np.nanmean([s["sigma_d_operating"] for s in b2.values()]))
    print(f"\npooled sigma_d_operating (mean across judges): {sigma_d_pooled:.4f}")

    b3_read_the_map(b1_train["min_implied_psi_deg"], sigma_d_pooled)


if __name__ == "__main__":
    main()
