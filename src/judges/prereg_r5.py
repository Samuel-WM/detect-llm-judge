"""NEXT_STEPS_ROUND_5_FINAL.md 7 / Gate 3: write and print the frozen pre-registration block.

Must run AFTER sections 2-4 are frozen (pool, training set, final holdout) and BEFORE the first
DPO run. Once written, nothing in this block may change after any attribution outcome is observed.

Run: .venv/Scripts/python.exe -m src.judges.prereg_r5
"""
import json
from pathlib import Path

import yaml

from src.analysis.round5_attribution import (BOOT_SEED, N_BOOT, N_PERM, P_THRESHOLD, PERM_SEED,
                                             lambda_threshold)
from src.train.round5_attribution import (CTRL_REFERENCE, CTRL_SEED, EPOCHS, LABEL_SEED,
                                          N_CHECKPOINTS, TAU)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "results"
CFG = REPO_ROOT / "configs" / "base.yaml"


def main() -> None:
    cal = json.load(open(RESULTS_DIR / "rm_calibration_r5.json"))
    tr = json.load(open(RESULTS_DIR / "rm_trainset_r5.json"))
    fh = json.load(open(RESULTS_DIR / "final_holdout.json"))
    S, M = cal["S"], cal["M"]
    required = max(2, -(-2 * M // 3))

    block = {
        "status": "FROZEN before the first DPO run; no field may change after any attribution "
                  "outcome is observed",
        "candidates": S,
        "M": M,
        "chance": 1.0 / M,
        "pool_determined_by": "C1-C4 model-wise plus pool-level C5, in "
                              "reward_model_stage.calibration_retention",
        "min_raw_pairwise_psi_eff_deg": cal["min_raw_psi_deg"],
        "min_residualized_pairwise_psi_eff_deg": cal["min_resid_psi_deg"],

        "primary_identification_rule": "nnls_mixture",
        "secondary_identification_rule": "bradley_terry",
        "additional_reported_rule": "sign_agreement",
        "detection_statistic": "Lambda",
        "lambda_detection_threshold": lambda_threshold(M),
        "lambda_threshold_basis": f"chi2(1) upper quantile at p < {P_THRESHOLD} Bonferroni-corrected "
                                  f"across M = {M} candidates; conservative for a boundary-constrained "
                                  f"gamma, whose exact null is a 50:50 chi2(0)/chi2(1) mixture",

        "primary_evaluation_data": "fresh final attribution holdout (section 4); NEVER the 487 "
                                   "calibration probes",
        "final_holdout_manifest_sha256": fh["manifest_sha256"],
        "N_final": fh["N_final"],
        "training_manifest_sha256": tr["frozen_manifest_sha256"],
        "N_train": tr["retained_pairs"],

        "evaluation_checkpoint": "final checkpoint; full trajectory reported; no checkpoint may be "
                                 "selected on final-holdout performance",
        "epochs": EPOCHS,
        "n_checkpoints": N_CHECKPOINTS,
        "label_process": f"stochastic Bradley-Terry, tau = {TAU}, standardized on training pairs only",
        "label_seed": LABEL_SEED,
        "ctrl_seed": CTRL_SEED,
        "ctrl_construction": f"permutation of the {CTRL_REFERENCE} label vector; preserves the "
                             f"positive/negative count exactly",

        "primary_per_run_statistic": "S = alpha_true - max_{m != true} alpha_m, on the UNNORMALIZED "
                                     "nonnegative NNLS coefficients",
        "s_statistic_deviation_from_6_4": {
            "declared": "before any DPO run and before any attribution outcome existed",
            "spec_text": "6.4 defines S on the normalized mixture weight vector",
            "reason": "measured on synthetic data with this pool's correlation structure, the "
                      "normalized statistic is degenerate under the permutation null: permuted-Delta "
                      "NNLS weights are tiny and noisy, so dividing by their sum drives the vector to "
                      "a simplex vertex. The null becomes bimodal at +/-1 with 99.9th percentile "
                      "exactly +1.0000, while S_obs <= 1 by construction, so p_perm cannot fall below "
                      "about 0.10 and the pre-registered p < 0.001 criterion is unsatisfiable.",
            "effect": "top-1 identification is UNCHANGED (normalization is division by a positive "
                      "scalar); only the magnitude statistic changes. Normalized weights are still "
                      "reported for 6.3 and F19, and S on normalized weights is reported alongside.",
        },
        "primary_per_run_test": f"{N_PERM}-permutation one-sided randomization test, probe-labels "
                                f"permuted, candidate margin matrix held fixed, entire NNLS fit rerun",
        "permutation_seed": PERM_SEED,
        "p_threshold": P_THRESHOLD,
        "primary_per_run_success": "true candidate top-1 AND S > 0 AND p_perm < 0.001",
        "bootstrap": {"n": N_BOOT, "seed": BOOT_SEED,
                      "reports": ["top1_support", "s_boot_median", "s_boot_ci95"],
                      "role": "stability/uncertainty analysis, NOT the p-value"},
        "per_probe_winrate": "reported with exact binomial CI, descriptive only; never the "
                             "inferential basis for the dataset-level NNLS claim",

        "margin_standardization": "frozen 487-probe calibration mean/sd (4.5); candidates are NEVER "
                                  "standardized using final attribution outcomes",
        "residualization": "frozen calibration length coefficients (c_m, gamma_m) applied out of "
                           "sample; residuals standardized with calibration residual mean/sd",

        "aggregate_claim": "every retained real reward-model arm is correctly attributed under the "
                           "primary rule at the final checkpoint",
        "negative_control": "one required shuffled-label CTRL run; no candidate may clear the "
                            "pre-registered Lambda threshold. No claim that a single CTRL run's "
                            "top-1 frequency is 'at chance'.",
        "kill_criterion": {
            "required_successes": required,
            "formula": "max(2, ceil(2*M/3))",
            "on_fire": "rank transmission is insufficient at this scale; report and stop before the "
                       "deterministic realism arm",
        },
        "length_robustness": "raw attribution is primary; residualized attribution is required "
                             "secondary robustness; residualized success cannot rescue raw failure",
        "deterministic_realism_arm": {
            "trigger": "ALL retained stochastic real-RM arms satisfy the primary per-run success "
                       "criterion AND the CTRL detection condition passes",
            "teacher": ("gpt2-helpful" if "gpt2-helpful" in S
                        else "OA-deberta" if "OA-deberta" in S else "gpt2-harmless"),
            "teacher_rule": "frozen in 9 before any DPO result was observed",
            "label": "1[d_m_train > 0], deterministic tie handling fixed before training",
        },
        "sigma_D_convergence_rule": {
            "range_pearson_last4": 0.03, "range_r_squared_last4": 0.02,
            "abs_final_minus_mean_pearson": 0.015, "abs_final_minus_mean_r_squared": 0.010,
            "on_failure": "report sigma_D_ratio(t) as a trajectory; do not place a single measured "
                          "sigma_D point on Track E; use a final-four sensitivity band",
        },
        "prohibited": "sigma_D = 5.35 is struck and is not an input, bound, estimate, or comparison "
                      "target anywhere in this round",
    }

    t = CFG.read_text(encoding="utf-8")
    marker = "\nattribution_stage:\n"
    assert marker not in t, "attribution_stage already exists; refusing to overwrite a frozen block"
    t = t.rstrip() + "\n\n# " + "-" * 90 + "\n" + \
        "# ROUND 5 GATE 3: frozen pre-registration. Written after sections 2-4 were frozen and\n" \
        "# BEFORE the first DPO run. Do not change any field after seeing an attribution outcome.\n" \
        "# " + "-" * 90 + "\n" + \
        yaml.safe_dump({"attribution_stage": {"pre_registration_round5": block}},
                       sort_keys=False, width=100, default_flow_style=False)
    CFG.write_text(t, encoding="utf-8")

    reloaded = yaml.safe_load(CFG.read_text(encoding="utf-8"))["attribution_stage"]["pre_registration_round5"]
    print("=== configs/base.yaml : attribution_stage.pre_registration_round5 ===")
    print(yaml.safe_dump(reloaded, sort_keys=False, width=100))
    print("=== Gate 3 verification ===")
    for k in ["candidates", "M", "chance", "final_holdout_manifest_sha256", "N_final",
              "training_manifest_sha256", "N_train", "primary_identification_rule",
              "secondary_identification_rule", "detection_statistic",
              "lambda_detection_threshold", "primary_per_run_test", "p_threshold",
              "primary_per_run_success"]:
        print(f"  {k}: {reloaded[k]}")
    print(f"  permutation count: {N_PERM}")
    print(f"  bootstrap count: {reloaded['bootstrap']['n']}")
    print(f"  kill criterion: {reloaded['kill_criterion']}")
    print(f"  negative control: {reloaded['negative_control']}")
    print(f"  deterministic arm trigger: {reloaded['deterministic_realism_arm']['trigger']}")


if __name__ == "__main__":
    main()
