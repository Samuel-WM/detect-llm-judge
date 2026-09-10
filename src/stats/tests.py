"""Statistical tests, Section 4.6 of CLAUDE_CODE_PROMPT.md.

Identification: true judge fixed in advance, one-sided exact binomial test of the per-probe
identification rate against chance 1/M, no multiple-comparison correction.

Margin significance: top candidate selected as best of M, so Bonferroni applies. Paired
per-probe (top - runner-up) differences; Shapiro-Wilk normality check; one-sided paired t-test
if normal, else one-sided Wilcoxon signed-rank; p_star = min(1, p * (M - 1)).
"""
import numpy as np
from scipy import stats


def identification_binomial_test(n_correct: int, n_total: int, M: int) -> dict:
    p0 = 1.0 / M
    result = stats.binomtest(n_correct, n_total, p0, alternative="greater")
    return {
        "rho_hat": n_correct / n_total,
        "p_value": float(result.pvalue),
        "n_total": n_total,
        "n_correct": n_correct,
        "chance_rate": p0,
    }


def margin_significance_test(paired_diffs: np.ndarray, M: int) -> dict:
    diffs = np.asarray(paired_diffs, dtype=np.float64)
    diffs = diffs[np.isfinite(diffs)]
    n = len(diffs)

    if n < 3 or np.allclose(diffs, diffs[0]):
        # Shapiro-Wilk needs >=3 distinct values; a degenerate (constant) sample can't be
        # meaningfully tested for normality either.
        sw_p = float("nan")
    else:
        _sw_stat, sw_p = stats.shapiro(diffs)
        sw_p = float(sw_p)

    use_wilcoxon = (not np.isnan(sw_p)) and sw_p < 0.05
    test_used = "wilcoxon" if use_wilcoxon else "t_test"

    if test_used == "t_test":
        t_stat, p_two = stats.ttest_1samp(diffs, popmean=0.0)
        p_one = float(p_two / 2 if t_stat > 0 else 1 - p_two / 2)
    else:
        try:
            _w_stat, p_one_raw = stats.wilcoxon(diffs, alternative="greater")
            p_one = float(p_one_raw)
        except ValueError:
            p_one = float("nan")

    p_star = min(1.0, p_one * (M - 1)) if not np.isnan(p_one) else float("nan")

    return {
        "n": n,
        "shapiro_p": sw_p,
        "test_used": test_used,
        "p_one_sided": p_one,
        "p_star_bonferroni": p_star,
        "margin_mean": float(np.mean(diffs)) if n > 0 else float("nan"),
        "margin_median": float(np.median(diffs)) if n > 0 else float("nan"),
    }
