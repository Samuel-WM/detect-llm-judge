"""Section B.2 + B cross-process determinism, computed from stored artifacts only (no GPU).

B.2  Validate InternLM2 retained-input lengths under the EXACT final reward-scoring
     serialization and the ACTUAL input_ids (chat template + special-id remap + appended
     reward token), replacing the preliminary plain-concatenation SentencePiece estimate
     that the 500-token probe filter was built on.

B.det  Quantify InternLM2 cross-process reproducibility on the 487 retained probes by
     comparing two independent processes that scored the same probes: the H1-H6 harness
     process and the bias-check full-set process. gpt2 models are included as controls.
"""
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = REPO_ROOT / "results"
INTERNLM = "internlm/internlm2-1_8b-reward"
CTX_INTERNLM = 2048
FILTER_CAP = 500
CONTROLS = ["Ray2333/gpt2-large-helpful-reward_model", "Ray2333/gpt2-large-harmless-reward_model"]
SHORT = {INTERNLM: "InternLM2-1.8B", CONTROLS[0]: "gpt2-helpful", CONTROLS[1]: "gpt2-harmless"}


def b2_lengths() -> dict:
    filt = json.load(open(RESULTS / "rm_probe_filter.json"))
    retained = filt["retained_indices"]
    est = np.array(filt["per_model_lengths"][INTERNLM])          # preliminary SP estimate, per pair
    full = json.load(open(RESULTS / f"rm_full_{INTERNLM.replace('/', '__')}.json"))
    real_flat = np.array(full["realized_lengths_full"])           # 2 rows per pair: a then b
    assert real_flat.size == 2 * est.size, (real_flat.size, est.size)
    real_pair = real_flat.reshape(-1, 2).max(axis=1)              # worst case over both responses

    r_ret, e_ret = real_pair[retained], est[retained]
    delta = r_ret - e_ret
    return {
        "n_retained": len(retained),
        "serialization": "apply_chat_template(user=prompt, assistant=response) + special-id remap "
                         "+ appended <|reward|> token; length = len(actual input_ids)",
        "preliminary_estimate": "plain '{prompt}\n\n{response}' SentencePiece encode",
        "realized": {"max": int(r_ret.max()), "p99": float(np.percentile(r_ret, 99)),
                     "median": float(np.median(r_ret)), "min": int(r_ret.min()),
                     "mean": float(r_ret.mean())},
        "preliminary": {"max": int(e_ret.max()), "median": float(np.median(e_ret))},
        "realized_minus_estimate": {"max": int(delta.max()), "median": float(np.median(delta)),
                                    "min": int(delta.min())},
        "model_context": CTX_INTERNLM,
        "filter_cap_used": FILTER_CAP,
        "n_retained_exceeding_filter_cap": int((r_ret > FILTER_CAP).sum()),
        "n_retained_exceeding_model_context": int((r_ret > CTX_INTERNLM).sum()),
        "max_headroom_used_frac": float(r_ret.max() / CTX_INTERNLM),
        "any_truncation_occurred": bool((r_ret > CTX_INTERNLM).any()),
    }


def cross_process_determinism() -> dict:
    filt = json.load(open(RESULTS / "rm_probe_filter.json"))
    retained = filt["retained_indices"]
    out = {}
    for repo in [INTERNLM] + CONTROLS:
        h = RESULTS / f"rm_harness_{repo.replace('/', '__')}.json"
        f = RESULTS / f"rm_full_{repo.replace('/', '__')}.json"
        if not (h.exists() and f.exists()):
            continue
        a = np.array(json.load(open(h))["result"]["d_m"])                     # process 1
        b = np.array(json.load(open(f))["d_m_full"])[retained]               # process 2, same probes
        d = np.abs(a - b)
        out[SHORT[repo]] = {
            "n": int(d.size),
            "frac_bitwise_equal": float((d == 0).mean()),
            "max_abs_diff": float(d.max()),
            "median_abs_diff": float(np.median(d)),
            "sd_of_margin": float(a.std()),
            "median_abs_diff_as_frac_of_sd": float(np.median(d) / a.std()),
            "pearson_between_processes": float(np.corrcoef(a, b)[0, 1]),
            "reproducible_across_processes": bool(d.max() == 0.0),
        }
    return out


if __name__ == "__main__":
    b2 = b2_lengths()
    det = cross_process_determinism()
    print("=== B.2 InternLM2 retained-input lengths, exact final serialization ===")
    for k, v in b2.items():
        print(f"  {k}: {v}")
    print("\n=== cross-process determinism on the 487 retained probes ===")
    for m, v in det.items():
        print(f"  {m}: bitwise-equal {v['frac_bitwise_equal']:.3f}  max {v['max_abs_diff']:.4f}  "
              f"median {v['median_abs_diff']:.4f} ({v['median_abs_diff_as_frac_of_sd']:.1%} of sd)  "
              f"r={v['pearson_between_processes']:.4f}  "
              f"reproducible={v['reproducible_across_processes']}")
    with open(RESULTS / "rm_b2_lengths_and_determinism.json", "w") as fh:
        json.dump({"b2_lengths_internlm": b2, "cross_process_determinism": det}, fh, indent=1)
    print(f"\nwrote {RESULTS / 'rm_b2_lengths_and_determinism.json'}")
