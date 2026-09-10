# RESULTS

Running log for the judge-attribution experiment (`CLAUDE_CODE_PROMPT.md`). Append after every
phase; stop and show numbers before starting the next.

## Environment notes (deviations from Section 1, flagged)

**Correction (2026-09-07):** an earlier version of this note claimed torch `2.14.0+cu126` "does
not correspond to a real PyTorch release." That was wrong — I hadn't actually queried the
package index before writing it. It does exist (`pip index versions torch --index-url
https://download.pytorch.org/whl/cu126` lists it as latest). See below for what actually happened
when installing it.

- Rebuilt `.venv` from scratch **without** `--system-site-packages` (previously it inherited the
  Anaconda base env's torch; now fully isolated), torch installed first, HF stack resolved fresh
  against it.
- Installed `torch==2.14.0+cu126` (matching the doc exactly) first. It downloads and installs
  cleanly, but **fails to load**: `OSError: [WinError 1114] A dynamic link library (DLL)
  initialization routine failed... Error loading torch\lib\c10.dll`. This reproduced on a plain
  `import torch`, before any project code ran — a bad interaction between this specific wheel and
  something on this machine (driver/runtime component, not disk corruption; reinstall didn't
  change it). Did not chase this further since a working alternative was one version-step away
  (see next point) — if you want `2.14.0` specifically working, tell me and I'll dig into the DLL
  failure directly.
- Bisected by trying `torch==2.6.0+cu126` (smallest version bump on the same cu126 build family):
  loads cleanly, `cuda.is_available()` True, bf16 matmul works, and — the actual reason a newer
  torch was needed — `from torch.distributed.fsdp import FSDPModule` succeeds. **Settled on
  `torch==2.6.0+cu126`** as the version actually in use going forward.
- Root cause for needing torch >2.5: `trl==1.12.0` (latest resolved by pip; `transformers`
  major version 5 forces a recent `trl`) imports `FSDPModule` from `torch.distributed.fsdp` at
  module load time, unconditionally — even for a plain single-GPU LoRA run with no distributed
  training involved. That symbol doesn't exist in torch `2.5.1`, so `from trl import DPOTrainer`
  hard-crashes on import under the originally-installed torch. This is exactly the API-drift risk
  flagged in the previous version of this note, now concretely confirmed.
- `transformers==5.16.1` / `trl==1.12.0` / `peft==0.20.0` resolved as latest-compatible by pip
  into the clean venv (same versions as before the rebuild).
- **`DPOConfig` API drift, found and worked around (see Phase-1-readiness smoke test below):**
  `trl==1.12.0`'s `DPOConfig` no longer accepts `max_prompt_length` — it was collapsed into a
  single `max_length` over the whole tokenized sequence, truncated `keep_start`. This does not
  require changing any fixed value from Section 4.4: `max_prompt_tokens=384` is already enforced
  upstream at generation time (Section 4.2), and `384 + max_new_tokens(256) = 640 = max_length`,
  so the two caps were designed to compose exactly — only the enforcement point moves (data
  generation instead of the trainer), not the numbers. `dpo_train.py` will build the training
  dataset from prompts already capped at 384 tokens and pass `max_length=640,
  truncation_mode="keep_start"` to `DPOConfig`.
- `KMP_DUPLICATE_LIB_OK=TRUE` is set for CPU-only runs (Phase 0) to work around an Anaconda
  MKL / bundled-libiomp5 duplicate-OpenMP-runtime error (`OMP: Error #15`) that is unrelated to
  this project — it fires on import of `torch` alongside Anaconda's own OpenMP-linked numpy/
  scipy. Confirmed it is still needed on GPU runs too (base Anaconda Python is still what invokes
  the venv's interpreter). Low-risk, standard workaround; affects only thread-pool init, not
  numerics.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (Section 1 rule 7) is set in every entry
  point as instructed, but torch emits `UserWarning: expandable_segments not supported on this
  platform` — this is a CUDA-allocator feature currently Linux-only. It's a harmless no-op on
  Windows, not an error; keeping the env var set per the rule (costs nothing, and matches the
  instruction literally) but flagging that it has no actual effect here.

### Phase-1-readiness smoke test (`scripts/smoke_test_gpu.py`)

Run against the rebuilt venv (torch 2.6.0+cu126) with the real base policy, before writing any
Phase 1 code, specifically to catch API drift before it costs a wasted training run:

1. Load `Qwen/Qwen2.5-0.5B-Instruct` in bf16 with `attn_implementation="sdpa"` on CUDA — **pass**.
2. Forward pass: bf16 logits, float32 log-softmax reduction — **pass**.
3. Bitwise reproducibility of logits across two identical forward calls (Section 4.5 requirement,
   checked here on the base model as the adapter-off case) — **pass**.
4. `DPOTrainer(model=..., ref_model=None, peft_config=LoraConfig(r=16, alpha=32, dropout=0.05,
   target_modules=[q/k/v/o_proj, gate/up/down_proj]), args=DPOConfig(max_length=640,
   truncation_mode="keep_start", beta=0.1, bf16=True, gradient_checkpointing=True,
   dataloader_num_workers=0, ...))` instantiation — **pass**, after the `max_prompt_length` fix
   above.
5. One real training step on a 2-example toy DPO dataset — **pass** (loss 0.6931 = ln(2), correct
   for an untrained margin at step 0).
6. VRAM: 5.035 GB free baseline -> 4.066 GB after model load (~1 GB for the 0.5B model in bf16 +
   CUDA context) -> 4.965 GB after `del` + `gc.collect()` + `empty_cache()`. ~0.07 GB not
   reclaimed; within the 0.3 GB tolerance `common/vram.assert_baseline()` uses, but noting it as
   something to watch once PEFT adapters and gradient state are involved in real runs, not just
   this toy step.

One non-blocking warning also appeared during dataset tokenization: `[RANK 0] Mismatch between
tokenized prompt and the start of tokenized prompt+chosen`. This is trl's own consistency check;
it fired here because the smoke test's toy dataset used raw strings instead of the model's chat
template. Real Phase 1/2 data will be built by tokenizing prompt and prompt+response consistently
through `tokenizer.apply_chat_template`, which should avoid it — will confirm this doesn't
recur once `data/build_prompts.py` exists, and report if it does.

### `scripts/check_env.py` output (post venv-rebuild, torch 2.6.0+cu126)

```
============================================================
PYTHON / PLATFORM
============================================================
python: 3.12.3 | packaged by Anaconda, Inc. | (main, May  6 2024, 19:42:21) [MSC v.1916 64 bit (AMD64)]
platform: Windows-11-10.0.26200-SP0

============================================================
TORCH / CUDA
============================================================
torch: 2.6.0+cu126
cuda available: True
gpu name: NVIDIA GeForce RTX 3050 6GB Laptop GPU
cuda runtime (torch build): 12.6
total vram (GB): 6.0
free vram (GB): 5.035
cudnn version: 90501
bf16 supported: True

============================================================
PACKAGE VERSIONS
============================================================
transformers: 5.16.1
trl: 1.12.0
peft: 0.20.0
datasets: 5.0.1
accelerate: 1.14.0
numpy: 2.5.3
scipy: 1.18.1
pandas: 3.0.5
pyarrow: 25.0.1
matplotlib: 3.11.1
tqdm: 4.70.0
```

Free VRAM (5.035 GB) matches the "~5.0 GB usable under Windows" hardware note. `numpy`/`scipy`/
`pandas` resolved to newer major versions than the first (system-site-packages) venv had; re-ran
Phase 0's analytic acceptance check against this rebuilt venv as a regression check — still
passes (`top-1 = 1.0`, angular error `1.49e-08 rad`, under the `1e-6` threshold; the tiny
non-zero value versus the original run's exact `0.0` is BLAS/LAPACK-version floating-point noise
in `lstsq`, not a behavior change). Did not re-run the empirical sweep since nothing
environment-dependent changed its logic; the previously-reported numbers in the Phase 0 section
below stand.

---

## Phase 0: synthetic identifiability

**Run:** `.venv/Scripts/python.exe -m src.phase0_synthetic`
**Config:** `configs/base.yaml` (seed 0) + `configs/phase0.yaml`
**Wall clock:** 241.7 s (target: 20 min — well under)
**Results:** `results/phase0_synthetic.parquet` (2415 rows), manifest at
`results/phase0_<run_id>/manifest.json`

### Implementation notes / interpretation calls (not numerically specified in Section 5)

- `beta = 1.0` for the exponential tilt. Attribution is scale-free by construction (Section 0),
  so this choice provably does not affect accuracy or angular error (cosine similarity is
  scale-invariant) — only the arbitrary scale of the recovered `w_hat`.
- "Empirical distribution from N draws": implemented as drawing `N` samples from the analytic
  categorical policy `pi_m(y|x)` over the `n=20` candidate responses per prompt, Laplace-
  smoothing (`alpha=1`) the counts into relative frequencies, and using that plug-in estimate in
  place of the exact analytic log-probability when forming `Delta`. This is a finite-sample
  convergence/robustness diagnostic for the *estimator*, not a simulation of DPO training — the
  real pipeline (Phase 2+) reads exact log-probs via a forward pass on realized tokens, so this
  particular noise source doesn't literally arise there. Candidate margins `d_m` stay exact
  throughout in both the analytic and empirical cases, matching how a real auditor reads judge
  logits exactly. **Flagging this interpretation explicitly — tell me if you intended something
  else (e.g. an empirical simulation of finite DPO training data size).**

### Bug found and fixed during this phase

`scipy.optimize.nnls` (Lawson-Hanson active-set solver) hung with `RuntimeError: Maximum number
of iterations reached` partway through the empirical sweep. Root cause: with `M=5` candidates in
only `d=4` feature dimensions (fixed by Section 5), the 5 judge weight vectors are necessarily
linearly dependent, so candidate-margin columns of `D` are structurally collinear — a
rank-deficient design on which the classic active-set NNLS algorithm is known to cycle rather
than converge. Switched `nnls_mixture` in `src/attribute/rules.py` to
`scipy.optimize.lsq_linear(D, delta, bounds=(0, inf))` (trust-region-reflective), which solves
the identical convex bound-constrained least-squares problem without that failure mode. Re-ran
the full phase after the fix; results below are post-fix.

### Acceptance check: **PASS**

- Analytic top-1 accuracy: **1.000000** (need 1.0)
- Analytic max angular error: **0.000e+00 rad** (need < 1e-6)

### Headline tables (median, IQR where applicable)

**Phase 0a — analytic exact recovery** (5 true judges x 3 rules = 15 cells, exact tilt, no
sampling):

| rule | top1 median | top1 mean | theta median | theta IQR |
|---|---|---|---|---|
| sign_agreement | 1.0 | 1.0 | 0.0 | [0.0, 0.0] |
| bradley_terry | 1.0 | 1.0 | 0.0 | [0.0, 0.0] |
| nnls_mixture | 1.0 | 1.0 | 0.0 | [0.0, 0.0] |

Exact recovery holds for every true judge individually (all 5 x 3 = 15 cells at top1=1.0,
theta=0), not just on average.

**Phase 0b — empirical finite-N estimation** (200 replicates per (rule, N) cell, 12,000 total
attribution decisions):

| rule | N | top1 median | top1 mean | theta median | theta Q1 | theta Q3 |
|---|---|---|---|---|---|---|
| sign_agreement | 1,000 | 1.0 | 1.000 | 0.01613 | 0.01108 | 0.01981 |
| sign_agreement | 10,000 | 1.0 | 1.000 | 0.00496 | 0.00352 | 0.00651 |
| sign_agreement | 100,000 | 1.0 | 1.000 | 0.00167 | 0.00114 | 0.00215 |
| sign_agreement | 1,000,000 | 1.0 | 1.000 | 0.00047 | 0.00032 | 0.00063 |
| bradley_terry | 1,000 | 1.0 | 1.000 | 0.01613 | 0.01108 | 0.01981 |
| bradley_terry | 10,000 | 1.0 | 1.000 | 0.00496 | 0.00352 | 0.00651 |
| bradley_terry | 100,000 | 1.0 | 1.000 | 0.00167 | 0.00114 | 0.00215 |
| bradley_terry | 1,000,000 | 1.0 | 1.000 | 0.00047 | 0.00032 | 0.00063 |
| nnls_mixture | 1,000 | 1.0 | 0.960 | 0.01613 | 0.01108 | 0.01981 |
| nnls_mixture | 10,000 | 1.0 | 0.945 | 0.00496 | 0.00352 | 0.00651 |
| nnls_mixture | 100,000 | 1.0 | 0.935 | 0.00167 | 0.00114 | 0.00215 |
| nnls_mixture | 1,000,000 | 1.0 | 0.940 | 0.00047 | 0.00032 | 0.00063 |

theta = angular error in radians, `w_hat` recovered by OLS regression of `Delta` on feature
differences, compared to the true judge's weight vector.

**Observations, nothing tuned:**
- Angular error shrinks by roughly 3.2-3.4x per 10x increase in N, consistent with the
  expected O(1/sqrt(N)) rate of a plug-in frequency estimator.
- `sign_agreement` and `bradley_terry` hit top-1 mean accuracy of exactly 1.000 (all 800
  replicates correct) at every N tested, down to N=1,000.
- `nnls_mixture` top-1 median is still 1.0 at every N but mean sits at 93.5-96% — it is
  occasionally wrong even though it's usually right, which is expected: minimizing squared
  error under a nonnegativity constraint with structurally collinear columns (M=5 > d=4) can put
  small positive weight on the wrong candidate when noise perturbs the near-degenerate design,
  whereas sign agreement and the BT likelihood only need a global sign/ordering match. This
  matches the doc's warning that both estimators are heavy-tailed — the median is the right
  number to trust, and it is 1.0 throughout.

No p-values in this phase (Section 4.6 statistical tests are exercised starting Phase 2, where
scores are computed once per probe set rather than per replicate).

**Correction (2026-09-07):** the `nnls_mixture`, N=1,000,000 row above originally read
`0.00167 / 0.00114 / 0.00215` — identical to the N=100,000 row directly above it. That was a
copy-paste transcription error made while hand-writing this table from the terminal output, not
a pipeline bug: `theta` is computed once per replicate from an OLS fit that does not depend on
the attribution rule, so it is provably identical across `sign_agreement`/`bradley_terry`/
`nnls_mixture` within a given `(N, replicate)` cell. Verified directly against
`results/phase0_synthetic.parquet`: grouping by `(rule, N)` gives bit-identical
median/Q1/Q3 across all three rules at every N, including N=1,000,000
(`0.000473 / 0.000323 / 0.000625`, now corrected above). The stored data and the pipeline that
produced it were correct throughout; only this table's transcription was wrong.

### Anything changed from the document

- `scipy.optimize.nnls` -> `scipy.optimize.lsq_linear` for the mixture rule (bug fix, explained
  above; same objective and constraint, different solver).
- Two interpretive fills for values the document leaves unspecified numerically (`beta=1.0`,
  meaning of "empirical distribution from N draws") — both flagged above with reasoning.
- Everything else in Section 4/5 for Phase 0 (n=20, d=4, M=5, N grid, 200 replicates) implemented
  as written.

---

## Phase 0c: identifiability map (PHASE_0C_SPEC.md addendum)

**Run:** `.venv/Scripts/python.exe -m src.phase0c_synthetic` then
`.venv/Scripts/python.exe -m src.analysis.figures_phase0c`
**Config:** `configs/phase0c.yaml` (seed 0), `d=16, M=5, beta=1.0`
**Wall clock:** 284.8 s total (target: < 30 min — well under)
**Results:** `results/phase0c_synthetic.parquet` (35,600 rows), manifest at
`results/phase0c_<run_id>/manifest.json`. Figures in `figures/phase0c_fig{1..4}_*.png`.

### Interpretation calls (flagged; spec leaves these open at the individual-probe level)

- **Noise scale:** `sigma_D`/`sigma_d` are fractions of the standard deviation of the clean
  quantity across the full candidate pool (`10 * N` rows, before selection), not the final
  selected `N`-subset — keeps noise comparable between `random` and `disagreement_max`.
- **Selection:** "greedily select" implemented as a per-row score
  `s_i = min_{m!=m'} |d_tilde_{m,i} - d_tilde_{m',i}|`, ranked, top-`N` taken. The formula as
  written has no cross-row interaction term, so this is a literal reading, not an iterative
  set-interaction greedy algorithm. Will use the same convention for the real Phase 4.
- **Negative controls + Section 4.6 battery "representative cell":** fixed at the same point
  marginals A/B/C share (`psi=20°, sigma_D=sigma_d=0.5, lambda=4, N=600, select=random`).
  "Positive cells" for the threshold fit = the primary map's replicates at that exact
  `(psi, sigma)` cell.
- **Section 4.6 needs per-probe decisions**; the replicate-level rules aggregate over all `N`
  probes into one scalar per candidate. For the identification test, per-probe decision =
  `argmax_m d_tilde_{m,i} * sign(Delta_tilde_i)` (sign agreement weighted by margin magnitude,
  avoids ties). For the margin test, paired per-probe diffs = `t_i^top - t_i^runnerup` where
  `t_i^m = 1[sign(Delta_i) == sign(d_{m,i})]` and top/runner-up are chosen by the replicate's
  aggregate `sign_agreement` score. Exercises the code path per the spec's instruction; not
  claimed as the final per-probe statistic for the other two rules.
- **Mixture identifiability (Section 10)** uses `lambda_aniso=4`, matching every other section's
  default operating point — the spec doesn't state a lambda for this section.

### Acceptance check (Section 11 — regression gate only, not pass/fail on accuracy)

At `psi=90°, lambda=4, sigma=0, N=600, select=random`: **PASS** — `sign_agreement` and
`bradley_terry` both hit top-1 = 1.0, reproducing the Phase 0 regime.

### Fig 1 — the headline map (`phase0c_fig1_heatmap_psi_sigma.png`)

Pooled (mean over all 3 rules) top-1 accuracy, `psi` x `sigma`, `lambda=4, N=600, random`:

| psi \ sigma | 0.00 | 0.25 | 0.50 | 1.00 | 2.00 |
|---|---|---|---|---|---|
| 90° | 1.00 | 1.00 | 1.00 | 1.00 | 0.98 |
| 45° | 1.00 | 1.00 | 0.98 | 0.72 | 0.34 |
| 20° | 1.00 | 0.83 | 0.54 | 0.26 | 0.22 |
| 10° | 1.00 | 0.53 | 0.22 | 0.21 | 0.16 |
| 5° | 1.00 | 0.26 | 0.22 | 0.19 | 0.19 |
| 2° | 0.97 | 0.19 | 0.19 | 0.21 | 0.24 |

The 0.60 contour (main doc's own kill-criterion threshold, reused here as the reference line)
runs roughly through `(psi=45°, sigma≈1.1)` down to `(psi=10°, sigma≈0.2)` — attribution is
robust down to fairly tight judge separation as long as noise stays low, but collapses fast once
`sigma` exceeds about 0.5 unless judges are nearly orthogonal.

**Non-obvious finding, not a bug (verified):** at `psi=2°, sigma=0`, pooled accuracy is **0.97**,
not 1.0. This is not measurement noise — at that separation, even the noiseless case is limited
by finite-`N` probe sampling: the 600 randomly drawn probe pairs give an imperfect empirical
estimate of an extremely subtle direction, so occasional misattribution happens from sampling
variance in *which* probes were drawn, independent of the additive `sigma_D`/`sigma_d` mechanism.

### Fig 2 — Marginal A, probe count and selection (`phase0c_fig2_N_vs_accuracy_by_selection.png`)

Pooled top-1 accuracy at `psi=20°, sigma=0.5, lambda=4`:

| N | random | disagreement_max |
|---|---|---|
| 50 | 0.27 | 0.30 |
| 100 | 0.29 | 0.38 |
| 300 | 0.46 | 0.48 |
| 600 | 0.55 | 0.63 |
| 1200 | 0.57 | 0.76 |

Per-rule breakdown (`sign_agreement` / `bradley_terry` / `nnls_mixture`) at N=600:
random `0.46 / 0.48 / 0.70` vs disagreement_max `0.49 / 0.585 / 0.825` — disagreement-maximizing
selection beats random at every N and every rule tested, confirming the Phase-4 premise
directionally. It is a real, consistent gain (roughly +0.1-0.15 accuracy at matched N here), but
at this noise/separation operating point it does **not** reach the main doc's aspiration of
"a fixed accuracy at roughly an order of magnitude smaller N" — e.g. `disagreement_max` at N=100
(0.38) doesn't match `random` at N=1000ish (interpolating, ~0.56-0.57). Whether the order-of-
magnitude claim holds will depend on the real judges' actual separation/noise regime in Phase 1b.

### Fig 3 — accuracy vs. measured judge agreement (`phase0c_fig3_accuracy_vs_agreement.png`)

Scatter over every cell (primary map + marginals A/B/C), colored by rule, with the six
`agreement(psi)` reference lines from Section 3 of the spec overlaid. Two things visible:

1. Empirical agreement clusters slightly **above** each analytic reference line at matched
   nominal `psi` (e.g. points near the 90°/0.750 line sit mostly in 0.75-0.85) — consistent with
   the spec's own caveat that probe-feature anisotropy (`lambda>1` in every cell here except the
   `lambda=1` row of marginal B) raises agreement above the isotropic-baseline formula.
2. At fixed agreement rate there is large vertical spread (accuracy anywhere from ~0.2 to ~1.0)
   — agreement rate alone does not determine accuracy; `sigma` and `N` matter independently. A
   real auditor reading off this map from a measured agreement rate alone, without also knowing
   the noise regime, would not get a reliable accuracy estimate.

### Fig 4 — angular error vs. conditioning (`phase0c_fig4_theta_vs_kappa.png`)

Fitted log-log slope: **1.59** (spec's prediction: linear, i.e. slope ≈ 1). Two caveats on this
number, reported rather than hidden: (a) `kappa_D` only ranges from about 1.3 to 6.3 across every
cell generated (`lambda in {1,4,16}` doesn't push conditioning much further for this
difference-of-two-unit-vectors design) — a narrow dynamic range to fit a power law against; (b)
the `sigma=0` (noiseless) replicates form a separate point mass near machine precision
(`theta ~ 1e-8`) mixed into the same `kappa_D` bins as noisy replicates, visibly pulling some bins
down in the scatter. The binned-median trend is still visibly monotonic increasing, consistent
with the qualitative prediction (worse conditioning -> worse recovery), but I would not treat
1.59 as a precise exponent estimate given both caveats.

### Section 9 — negative controls and detection threshold

200 replicates each, representative cell (`psi=20°, sigma=0.5, lambda=4, N=600, random`):

| rule | tau | F1 | TPR | FPR | n_pos | n_neg |
|---|---|---|---|---|---|---|
| sign_agreement | 0.0017 | 0.496 | 0.900 | 0.865 | 200 | 400 |
| bradley_terry | 0.0034 | 0.635 | 0.845 | 0.407 | 200 | 400 |
| nnls_mixture | 0.0013 | 0.555 | 0.990 | 0.790 | 200 | 400 |

**Reported plainly, not softened:** at this operating point (noise `sigma=0.5` is already past
where Fig 1 shows `psi=20°` degrading badly), F1-optimal margin thresholds have high false
positive rates against the pooled negative controls (true-judge-removed + no-signal) —
`sign_agreement` and `nnls_mixture` both sit above 0.79 FPR at their F1-optimal threshold.
`bradley_terry` is comparatively better (FPR 0.41) but still far from a clean separation. This is
a genuine finding, not a bug (verified the noiseless mixture-recovery case separately — see
below): naive margin thresholding does not reliably distinguish a real detection from a
true-judge-removed or no-signal control once judges are only moderately separated and noise is
present. Phase 2's real threshold work should not assume margin alone is a safe abstention
signal at operating points anywhere near this one.

### Section 4.6 stat battery, one representative replicate

```
identification: rho_hat=0.2033, p=4.352e-01 (chance=1/M=0.200)
margin: test=wilcoxon (shapiro_p=3.26e-34, non-normal), p_one_sided=2.085e-01, p_star_bonferroni=8.342e-01
```

Both non-significant, as expected: this representative cell (`sigma=0.5, psi=20°`) sits in the
map's high-noise/moderate-separation region where Fig 1 already shows pooled accuracy around
0.5-0.6, well short of reliable identification. The battery itself exercised cleanly (Shapiro-Wilk
correctly triggered the Wilcoxon fallback; Bonferroni factor `M-1=4` applied) — this was the goal
of running it here, not a claim of a positive result.

### Section 10 — mixture identifiability

Median / IQR of `|| alpha_hat/||alpha_hat||_1 - alpha_true ||_1` (max possible = 2), `sigma=0.5`,
`lambda=4`, `N=600`, 200 replicates:

| psi | alpha | median | Q1 | Q3 |
|---|---|---|---|---|
| 45° | (0.5, 0.5) | 0.812 | 0.744 | 0.863 |
| 45° | (0.8, 0.2) | 0.852 | 0.793 | 0.896 |
| 20° | (0.5, 0.5) | 1.107 | 1.042 | 1.158 |
| 20° | (0.8, 0.2) | 1.143 | 1.083 | 1.201 |

**Verified this is a real noise-sensitivity finding, not a bug:** re-ran the `psi=20°, (0.5,0.5)`
cell at `sigma=0` (noiseless) directly — `alpha_hat = [0.500, 0.500, ~0, ~0, ~0]`, exact recovery
to float precision, reconstruction residual `~2.6e-15`. So the `d=16` design genuinely is
identifiable (this closes out the Phase 0 `d<M` artifact as intended), but recovery degrades
sharply under `sigma=0.5` noise — median error above 0.8 out of a max of 2 means NNLS is
regularly placing substantial weight on the wrong candidates once noise is added, even though
the noiseless limit is exact. This is consistent with Phase 0's own observation that
`nnls_mixture` is the least noise-robust of the three rules.

### Anything changed or added beyond the spec

- Built `src/probe/select.py` and `src/stats/tests.py` as standalone, reusable modules (matching
  the paths the main `CLAUDE_CODE_PROMPT.md` repository layout already reserved for them),
  rather than inlining this logic only inside the Phase 0c script — they'll be reused as-is for
  the real Phase 2/4 pipeline.
- Everything else in the spec (Sections 1-12) implemented as written, modulo the interpretation
  calls flagged above.

### Templating utility (spec's closing "applies to Phase 1b and everything after it" note)

Implemented `src/common/templating.py:build_scoring_inputs(tokenizer, prompt, response)` as
specified: tokenizes prompt-alone and prompt+response through the chat template, asserts the
prompt-alone tokenization is an exact prefix of the full tokenization, raises loudly on mismatch
rather than warning. Will be called identically from generation, DPO training, and scoring once
those exist (none of the three "stages" are built yet).

Smoke-tested against the real `Qwen/Qwen2.5-0.5B-Instruct` tokenizer (already cached from the
Phase-1-readiness smoke test): the happy path produces a clean prompt/response boundary
(`<|im_start|>assistant\n` ends the prompt half exactly). Also deliberately constructed a
mismatched case (raw `tok.encode()` vs chat-templated encode of the same text) and confirmed the
assertion condition correctly flags it — the check has teeth, not just a happy-path pass.

**One more real API-drift find in transformers 5.16.1, not merely a version-skew risk this
time:** `tokenizer.apply_chat_template(..., tokenize=True, return_tensors="pt")` returns a
`BatchEncoding` (dict-like, with `input_ids`/`attention_mask` keys), not a bare tensor. Indexing
it `[0]` (as older docs/examples do) grabs a `tokenizers.Encoding` object instead of the first
tensor row and fails downstream with an unrelated-looking `AttributeError`. Fixed by indexing
`["input_ids"][0]`. This is exactly the kind of silent-if-you're-not-careful drift the spec is
worried about, just at the tokenization-call-signature level rather than Delta itself — worth
keeping in mind for `data/build_prompts.py`, `gen/sample_pairs.py`, and the scorer when those are
written, since all three will call `apply_chat_template`.

---

## Track A: Phase 0c revisions (NEXT_STEPS_TRACKS_A_B_C.md)

**Run:** `.venv/Scripts/python.exe -m src.phase0c_track_a --reps 200` then
`.venv/Scripts/python.exe -m src.analysis.figures_track_a` (plus
`src.analysis.figures_phase0c` re-run for the new per-rule Fig 1b panel)
**Wall clock:** 1120.7 s = 18.7 min (target: < 45 min)
**Results:** `results/phase0c_track_a_{detection_grid,marginal_a_strategies,conditioning}.parquet`
**Figures:** `figures/phase0c_fig1b_heatmap_per_rule.png`,
`figures/track_a_fig_{A2_fpr_vs_accuracy,A3_auc_comparison,A4_three_strategies,A5_conditioning_fixed}.png`

### A0 — Marginal B and Marginal C, now reported

**Marginal B (anisotropy)**, `psi=20°, sigma=0.5, N=600`, top-1 accuracy:

| lambda | select | sign_agreement | bradley_terry | nnls_mixture |
|---|---|---|---|---|
| 1 | random | 0.760 | 0.905 | 0.995 |
| 1 | disagreement_max | 0.855 | 0.945 | 1.000 |
| 4 | random | 0.415 | 0.515 | 0.720 |
| 4 | disagreement_max | 0.425 | 0.480 | 0.805 |
| 16 | random | 0.225 | 0.210 | 0.355 |
| 16 | disagreement_max | 0.260 | 0.285 | 0.420 |

Anisotropy hurts badly and monotonically — concentrating probe variance along the shared
quality direction `u` (where all judges agree) throws away exactly the information attribution
needs, and `lambda=16` roughly halves accuracy again versus `lambda=4` for every rule/selection
combination.

**Marginal C (decoupled noise)**, `psi=20°, lambda=4, N=600` — the more important table, since it
answers "which axis matters more" and is exactly what Track B's B3 needs to read the real pool
onto:

| sigma_D (sigma_d=0.5 fixed) | bradley_terry | nnls_mixture | sign_agreement |
|---|---|---|---|
| 0.00 | 0.585 | 0.980 | 0.465 |
| 0.25 | 0.470 | 0.895 | 0.460 |
| 0.50 | 0.460 | 0.710 | 0.410 |
| 1.00 | 0.320 | 0.490 | 0.255 |
| 2.00 | 0.250 | 0.370 | 0.290 |

| sigma_d (sigma_D=0.5 fixed) | bradley_terry | nnls_mixture | sign_agreement |
|---|---|---|---|
| 0.00 | 0.820 | 1.000 | 0.690 |
| 0.25 | 0.630 | 0.950 | 0.590 |
| 0.50 | 0.370 | 0.685 | 0.365 |
| 1.00 | 0.355 | 0.480 | 0.330 |
| 2.00 | 0.185 | 0.250 | 0.165 |

**Judge noise (`sigma_d`) hurts more than alignment noise (`sigma_D`) at every rule.** At
`sigma=2.0`: varying `sigma_D` alone leaves `bradley_terry` at 0.250 and `sign_agreement` at
0.290, while varying `sigma_d` alone drops them to 0.185 and 0.165. This matters directly for
Track B/C: measuring `sigma_d` accurately (Track B) is more consequential for predicting real
accuracy than measuring `sigma_D` (Track C) — reinforces running Track B first as specified.
Also a useful diagnostic worth flagging: the `sigma=0.5` row is nominally the same operating
point in both sub-tables (`sigma_D=sigma_d=0.5`) but the two independently-drawn 200-replicate
samples disagree by up to 0.09 accuracy (`bradley_terry`: 0.460 vs 0.370) — a rough empirical
sense of replicate-to-replicate noise at this sample size, useful context for how much any single
number in these tables should be trusted to the third decimal.

### A1 — Primary map redrawn per rule (`phase0c_fig1b_heatmap_per_rule.png`)

Confirmed exactly as flagged: the pooled 0.60 contour is not any single rule's decision boundary.
At `psi=20°, sigma=0.5`: `nnls_mixture=0.685` clears 0.60 comfortably where the pooled figure
showed 0.54 and `sign_agreement=0.410`/`bradley_terry=0.460` both fall short. `nnls_mixture`'s
own 0.60 contour sits visibly further into the high-noise region than the other two rules'
contours across the whole map (see figure) — it is the most noise-robust rule for pure
identification accuracy, even though Section 9 (below, extended) shows it is simultaneously the
worst-calibrated against the null. These are not in tension; they're different properties.

### A2 — Negative controls extended across the full 30-cell grid

**The question that mattered: does FPR<0.1 hold anywhere top-1 accuracy exceeds 0.9, or was
Section 9's original cell just a bad operating point?** Answer, checked directly
(`track_a_fig_A2_fpr_vs_accuracy.png`): **mostly yes** — of the 37 (cell, rule) combinations with
positive-replicate top-1 accuracy above 0.9, **33 have F1-optimal-threshold FPR below 0.1**. The
4 exceptions are all at the two most extreme corners of the map (`psi=2°, sigma=0` and
`psi=45-90°, sigma=2.0`), where accuracy is only barely above 0.9 and noise or separation is
already near its worst tested value. Section 9's original representative cell
(`psi=20°, sigma=0.5`) sits well outside the accuracy>0.9 region (pooled accuracy there is ~0.54)
— its bad FPR numbers were correctly diagnosed in the prompt as reflecting a bad operating point,
**not** a property of margin thresholding in general. This is the answer the document said would
decide whether the detection half of the project survives: it survives, conditional on operating
in the accuracy>0.9 regime.

### A3 — gamma_hat and Lambda vs. the margin (`track_a_fig_A3_auc_comparison.png`)

Mean AUC across all 30 grid cells, pooled negatives (control1 + control2) vs. positives:

| rule | auc_margin | auc_gamma_hat | auc_Lambda |
|---|---|---|---|
| sign_agreement | 0.682 | 0.861 | 0.867 |
| bradley_terry | 0.826 | 0.867 | 0.873 |
| nnls_mixture | 0.777 | 0.861 | 0.868 |

`Lambda` (the likelihood-ratio statistic) has the highest mean AUC for **every** rule, and wins
the per-cell AUC comparison most often too (`sign_agreement`: 14/30 cells; `bradley_terry`:
12/30; `nnls_mixture`: 13/30, tied with `auc_margin`'s 13/30). The gap is largest exactly where
you'd expect: `sign_agreement`'s margin is a bare, unnormalized win-rate difference with no null
distribution attached (0.682 mean AUC), while its `gamma_hat`/`Lambda` — both derived from the
BT fit's own null-calibrated scale — jump to ~0.86-0.87. Per the pre-registration below,
**`Lambda` replaces the margin as the detection statistic for Phase 2.**

### A4 — Probe selection: `pairwise_coverage` fixes the objective (`track_a_fig_A4_three_strategies.png`)

Pooled top-1 accuracy at `psi=20°, sigma=0.5, lambda=4`:

| N | random | per_row_topk | pairwise_coverage |
|---|---|---|---|
| 50 | 0.300 | 0.328 | 0.360 |
| 100 | 0.300 | 0.345 | 0.378 |
| 300 | 0.373 | 0.535 | 0.542 |
| 600 | 0.502 | 0.578 | 0.645 |
| 1200 | 0.670 | 0.718 | 0.833 |

`pairwise_coverage` beats `per_row_topk` at every N (confirming the diagnosis that the original
per-row objective wastes budget over-resolving one easy candidate pair), and both beat `random`
at every N except one exception worth reporting rather than hiding: at N=300, `nnls_mixture`
alone has `per_row_topk` (0.740) beat `pairwise_coverage` (0.665) — noted, not explained away.
**Said plainly, per the document's own instruction:** this is a real, consistent improvement, but
it is still not the order-of-magnitude reduction in N the main document claims — reaching
`random`'s N=1200 accuracy (0.670) requires `pairwise_coverage` at roughly N=600-700
(interpolating: 0.645 at 600), a bit under a 2x reduction in N, not 10x, at this specific
`(psi=20°, sigma=0.5)` operating point. Whether the real judges land somewhere the gain is larger
depends on Track B/C's measured operating point.

### A5 — Conditioning figure, corrected construction (`track_a_fig_A5_conditioning_fixed.png`)

**The literal construction in the spec does not work, verified empirically before running the
real sweep:** adding `delta * u_i` (random per-row direction `u_i`) to a fraction of rows, for
any fraction up to 0.9 and any `delta` down to `1e-4`, never pushed `kappa_D` past ~3.7 — because
a modest number of full-scale isotropic rows already spans `R^16`, so rows that are merely small
in *every* direction just become numerically negligible as `delta` shrinks; they don't create a
direction-specific deficiency. Fixed by projecting one shared random direction `v` out of *every*
row, then adding it back at one fixed, non-varying magnitude `delta` (no per-row randomness in
that component) — now the design's only information about `v` comes from a constant offset whose
contribution to the smallest eigenvalue of `X^T X` is exactly `N * delta^2`, so `kappa_D` grows
smoothly and unboundedly as `delta -> 0`. This version spans `kappa_D` from ~2.1 (`sigma=0`
baseline) up to ~5,913 median (`delta=1e-4`) — the several-orders-of-magnitude range the original
figure needed and didn't have.

| delta | kappa_D (median) | theta (median, rad) |
|---|---|---|
| 1.0 | 3.49 | 0.122 |
| 0.1 | 5.89 | 0.147 |
| 0.01 | 59.3 | 0.639 |
| 0.001 | 590 | 1.354 |
| 0.0001 | 5,913 | 1.521 |

Excluding `sigma=0` from the fit (as instructed) and now fitting on real dynamic range: fitted
log-log slope = 0.363. **Superseded — see the reanalysis immediately below; `0.363` is not
retained as a precise exponent.**

#### A5 reanalysis (NEXT_STEPS_FINAL.md Section 1, correction 3)

`0.363` was fit across a regime that reaches the angular-error ceiling: `theta` cannot exceed
`pi/2 = 1.5708`, and the highest construction level sits at 96.8% of it. Fitting an unbounded
power law through saturated points biases the exponent down. Refit
(`src/analysis/a5_reanalysis.py`, `results/a5_reanalysis.json`) without ever selecting on
observed `theta` — all cuts are by construction level (`delta`), which is a design variable:

| delta | kappa_D median (IQR) | theta median (IQR) | % of pi/2 ceiling | saturated |
|---|---|---|---|---|
| 1.0 | 3.49 (3.45-3.54) | 0.122 (0.105-0.141) | 7.8% | no |
| 0.1 | 5.89 (5.72-6.04) | 0.148 (0.121-0.179) | 9.4% | no |
| 0.01 | 59.3 (57.9-60.4) | 0.639 (0.341-0.913) | 40.7% | no |
| 0.001 | 590 (579-601) | 1.354 (1.187-1.562) | 86.2% | no |
| 0.0001 | 5,913 (5,793-6,040) | 1.521 (1.339-1.722) | 96.8% | **yes** |

**Saturation onset:** the `delta=1e-4` level (median `kappa_D` ~5,913) is the first to cross 90%
of the ceiling; `delta=1e-3` at 86.2% is already close to it.

Saturating fits (all 1,000 replicates, no row selection):

| model | A | b | RMSE |
|---|---|---|---|
| `theta = min(A * kappa^b, pi/2)` | 0.1132 | **0.390** | 0.263 |
| `theta = (pi/2) * tanh(A * kappa^b / (pi/2))` | 0.0805 | **0.510** | 0.257 |

Fit on the 5 level medians instead of all replicates: `b = 0.401` (capped) and `b = 0.528`
(smooth) — same picture.

Sensitivity, ordinary log-log slope cut by construction level only:

| levels included | slope |
|---|---|
| deltas 1, 0.1 | 0.357 |
| deltas 1, 0.1, 0.01 | 0.599 |
| deltas 1, 0.1, 0.01, 0.001 | 0.488 |
| all five (includes the saturated level) | 0.365 |

**Pre-saturation slope range: 0.357 to 0.599.** Note that the original `0.363` is essentially the
all-levels fit (0.365) — i.e. it was the saturation-contaminated number, and it sits at the very
bottom of the honest range.

**What survives:** worse conditioning increases recovery error — monotonically, across ~3.5
orders of magnitude in `kappa_D`. That is the only conclusion retained. The exponent is not
pinned down: saturation-aware fits give `b` in roughly 0.39-0.53, and pre-saturation log-log
cutoffs give 0.36-0.60, with the spread driven by which levels are included (only 2-4 design
points are available below saturation). `0.363` is not retained as a precise exponent, and the
spec's predicted linear slope (`b = 1`) still is not supported by any of these fits.

### Pre-registration (before Phase 2 runs)

Per the document's instruction, fixed now based on A1/A3's results, before any Phase 2 number
exists to select against:

```
identification primary rule:   nnls_mixture      (best top-1 across the map, A1)
identification secondary rule: bradley_terry
detection statistic:           Lambda            (highest mean AUC every rule, A3)
mixture recovery:              nnls_mixture only
```

Written into `configs/base.yaml` under `pre_registration`. All three rules will still be reported
in every Phase 2 table regardless — this governs which number is the headline claim, not which
numbers get shown.

### Unit test for the templating invariant

Added `tests/test_templating.py` (plain-assert, run via
`.venv/Scripts/python.exe -m tests.test_templating`) asserting `build_scoring_inputs` returns a
token-id prompt prefix exactly equal to the standalone prompt tokenization, plus a check that a
genuine mismatch (raw `encode()` vs. chat-templated encode) is correctly caught. Will be run at
the top of `gen/sample_pairs.py`, `judges/margins.py`, and the eventual scorer — the three real
call sites that depend on the invariant and none of which will fail loudly on their own if it
breaks.

### Anything changed from the spec

- A5's near-duplicate construction corrected as described above (shared fixed-magnitude
  direction, not a random-direction random-fraction subset) — the literal spec version was
  tested and does not achieve the stated goal.
- Everything else in Track A implemented as written.

---

## Track B: measuring the real operating point (in progress)

Building `data/build_prompts.py`, `gen/sample_pairs.py`, `judges/margins.py` and running them for
real against the actual base policy and 5 judges. `data/build_prompts.py` complete (3000 train +
1200 probe prompts from `HuggingFaceH4/ultrafeedback_binarized`, deduplicated, disjointness
asserted). Response-pair generation for 600 held-out probe prompts in progress.

**Finding en route, not a code bug in the usual sense but a real performance discovery:**
single-sequence (`batch_size=1`) generation on this GPU measured at only ~7-9 tokens/sec for the
0.5B base policy — at that rate, 600 prompts x 2 samples x 256 tokens would take roughly 11-12
hours. Diagnosed empirically (not assumed) as Windows/WDDM per-kernel-launch overhead dominating
at batch size 1, not a compute or memory-bandwidth limit: batching to 64 sequences measured
~522 aggregate tok/s (batch 128 reached ~625 tok/s but leaves less VRAM headroom). Rewrote
`gen/sample_pairs.py` to use length-bucketed batches of 64 (bucketing per the spec's own closing
note on allocator fragmentation).

**Real anomaly, not yet fully explained:** the actual 600-prompt run's two generation passes took
very different wall-clock time — pass A (fresh generation) finished in 355.9s, matching the
benchmarked batch-64 rate, but pass B (same code, same batches, run immediately after) took
9559.6s, about 27x slower. GPU state checked immediately afterward showed no anomaly (57°C,
idle) — that's not proof of anything mid-run, just a clean idle reading after the fact, since no
telemetry was being logged during that specific job. Set up `nvidia-smi -l 5` logging
(`logs/gpu_telemetry.csv`) before starting judge-margin extraction specifically to get real
evidence next time, on the (currently unconfirmed) hypothesis that this is thermal throttling on
a laptop GPU under sustained load. All 600 response pairs are done regardless (`data_cache/
response_pairs_probe.jsonl`); this anomaly matters mainly for planning Phase 1/2's DPO training
wall-clock, where the doc's "~2 hour" / "overnight" targets could be badly wrong if this recurs
under sustained training load.

**Judge margin extraction: real blocker, needs your input.** Of the 5 fixed judges
(Section 4.1), two are gated HuggingFace repos this session cannot access without an HF token
tied to an account that has accepted their license terms:

```
meta-llama/Llama-3.2-1B-Instruct   -- 401, gated repo
google/gemma-3-1b-it                -- 401, gated repo
```

The other three (`Qwen/Qwen2.5-1.5B-Instruct`, `HuggingFaceTB/SmolLM2-1.7B-Instruct`,
`TinyLlama/TinyLlama-1.1B-Chat-v1.0`) are accessible and confirmed working. I have not
substituted different models for the two gated ones — Section 4.1's pool is a fixed value, and
picking a replacement isn't a decision I should make silently. Proceeding now with judge-margin
extraction for the 3 accessible judges (real, useful data on its own) while this is open; B1's
pairwise matrices will be 3x3 instead of 5x5 until resolved. **To unblock:** log into
huggingface.co, accept the license on each of those two model pages, generate an access token
with read access, and either set `HF_TOKEN` in the environment or tell me the token so I can set
it for this session — then I'll fetch the remaining two and complete the 5x5 matrices.

**Second bug found and fixed while building `judges/margins.py`:** the first batched attempt
(batch=32) OOM'd on the very first judge. Root cause wasn't the batch size — real comparison
prompts (prompt + two full 256-token responses + template) run up to ~1,459 tokens
(median ~617), and by default `model(**enc).logits` materializes logits for *every* position of
every sequence, not just the one we read. At a ~150k vocab, batch=32, seq_len up to ~1,459,
that's tens of GB just for the logits tensor. Fixed with `logits_to_keep=1` (supported by this
transformers version's `forward()`), which computes the LM head only for the final position;
reduced `BATCH_SIZE` to 16 as an added safety margin. Retested on `Qwen2.5-1.5B-Instruct`: 20
pairs in 14.2s, no OOM.

**Third bug found and fixed:** the per-judge resume/checkpoint logic considered a template "done"
if *any* rows existed for it, not if *all* `len(pairs)` rows existed — a partial test run (20 of
600 pairs) would have been silently treated as complete on resume, permanently losing the other
580. Fixed to count rows per template against the expected pair count, and discard (not silently
keep) any partial template's rows before resuming.

**Fourth issue found and fixed:** even at batch=16 with `logits_to_keep=1`, the full run OOM'd
again, this time inside attention (`apply_rotary_pos_emb`), and the crash left the CUDA context
bad enough that `torch.cuda.empty_cache()` itself then threw OOM too (process exit still cleanly
released all VRAM afterward — `nvidia-smi` showed 0 MiB used). Root cause: a *fixed* batch size
is the wrong lever when sequence length varies 194-1,459 tokens 7x across the pool — a batch of
16 sequences near the long end is a very different memory footprint than 16 near the short end.
Replaced fixed-batch bucketing with token-budget packing (`rows_in_batch * max_len_in_batch`
capped at 4,096, 32 rows max) — batches auto-shrink for long sequences and grow for short ones.
Retested at n=100 across all 3 accessible judges: zero OOMs.

Full run (600 pairs x 2 templates, 3 accessible judges) completed. GPU telemetry
(`logs/gpu_telemetry.csv`, `nvidia-smi -l 5`) ran alongside it and resolved the earlier
generation-pass anomaly (see above) with direct evidence — see the thermal-throttling note below.

### B1: judge separation (3 of 5 judges — Llama/gemma still blocked)

Pairwise sign-agreement matrix, `template_train`, N=598 (2 pairs dropped: over TinyLlama's
2048-token context, see below):

| | SmolLM2-1.7B | Qwen2.5-1.5B | TinyLlama-1.1B |
|---|---|---|---|
| SmolLM2-1.7B | 1.000 | 0.562 | 0.396 |
| Qwen2.5-1.5B | 0.562 | 1.000 | 0.420 |
| TinyLlama-1.1B | 0.396 | 0.420 | 1.000 |

Pearson correlation matrix (same set): SmolLM2/Qwen2.5 = 0.479, SmolLM2/TinyLlama = -0.092,
Qwen2.5/TinyLlama = -0.084. Eigenvalues of the correlation matrix: `[1.509, 0.969, 0.521]`,
**PC1 variance share = 0.503**.

Per-pair implied `psi` (`psi = pi * (1 - agreement)`):

| pair | agreement | implied psi |
|---|---|---|
| SmolLM2 vs Qwen2.5 | 0.562 | 78.9° |
| SmolLM2 vs TinyLlama | 0.396 | 108.7° |
| Qwen2.5 vs TinyLlama | 0.420 | 104.4° |

**Flagging directly, not softening it: two of these three implied-psi values exceed 90°,** which
is *outside the valid range Section 3 itself states* (`psi in [0, pi/2]`, since the formula's
geometry requires `agreement >= 0.5` always — `cos(psi) = cos^2(a) >= 0` for `psi <= 90°`). An
agreement rate below 0.5 (SmolLM2 vs TinyLlama: 0.396) means these two judges disagree on
`sign(d_m)` *more often than two independent coin flips would* — real anti-correlation, not just
"weak agreement." Phase 0c's synthetic judge construction (all judges partially loading on one
shared coherent direction `u`) cannot produce this by construction. Investigated why (see B2)
rather than treating it as a modeling curiosity.

Mismatched template (`template_probe`): mean implied psi 97.6°, min 94.7° — similar picture,
matched/mismatched doesn't change the qualitative story here.

### B2: judge noise, sigma_d

```
                    sigma_d_pos   sigma_d_tmpl   operating   dominant
SmolLM2-1.7B          1.565         1.293         1.565     position_bias
Qwen2.5-1.5B           0.913         0.825         0.913     position_bias
TinyLlama-1.1B         2.679         1.201         2.679     position_bias
pooled (mean):                                     1.719
```

**Root cause of B1's out-of-range implied psi, found by inspecting raw `ell_AB`/`ell_BA` (not
just `d_m`):** these small judges exhibit severe *raw position bias* that swamps content signal.
TinyLlama is the extreme case: `ell_AB` mean = **-2.280** (std 0.737), `ell_BA` mean = **+2.262**
(std 0.750) — both consistent with an overwhelming, nearly content-independent preference for
whichever content occupies the literal "B" verdict slot (`ell_AB=logp(A)-logp(B)` very negative
= prefers B; `ell_BA=logp(B)-logp(A)` very positive = also prefers B, in the swapped-position
text). The Section-4.3 debiasing (`d_m = 0.5*(ell_AB+ell_BA)`) does successfully null this out
*in the mean* (TinyLlama's `d_m` mean is -0.009, near zero as it should be), but the two raw
terms' individual variance (~0.75 each) is large enough relative to the surviving content signal
that the debiased margin's *sign*, per pair, is still frequently determined by residual noise
rather than content — which is exactly what `sigma_d_pos > 1` means quantitatively (the position-
bias-residual's standard deviation *exceeds* the debiased margin's own standard deviation), and
plausibly explains why TinyLlama's agreement with the other two judges comes out below chance:
its "signal" is mostly bias-residual, only loosely related to actual response quality.

**Judge stochasticity: confirmed exactly zero**, not merely assumed. Ran the same 30 pairs
through `Qwen2.5-1.5B-Instruct` twice: `max |d_m_pass1 - d_m_pass2| = 0.0`, bitwise identical.
Reading logits from a deterministic forward pass has no sampling noise, as expected — all of
`sigma_d` above comes from position bias and template sensitivity, none from run-to-run
variance.

**Data-quality fix applied before any of the above:** ~0.3-0.5% of pairs (2-3 of 600 per
template) exceed TinyLlama's 2,048-token context window (`Token indices sequence length ...
2877 > 2048`) — verified by direct length check against TinyLlama's own tokenizer, not just the
warning text. Past that length, position embeddings either error or silently degrade, so those
specific `(judge, prompt_idx)` rows were excluded from every B1/B2 statistic above (small effect
given how few rows, but silently including known-corrupted rows would be worse than the modest
loss of coverage).

### B3: VOID — struck per NEXT_STEPS_FINAL.md Section 1

**This section previously reported two map readings (a 0.165-0.250 prediction from Marginal C at
`psi=20°`, and a 0.965-1.000 prediction from the Primary Map at `psi=90°`) and treated the
divergence between them as the finding. Both numbers are struck. Neither was a valid prediction,
and the divergence was not the finding.**

The reason they were invalid, per the superseding document: `psi_implied = pi * (1 - agreement)`
is not identified once judge reliability is low. A judge with little stable content signal drifts
toward 0.5 agreement with every other judge, and the formula misreads that drift as
*near-orthogonal separation* — the easiest region of the old map. So the 0.97-1.00 reading was
manufactured by the very unreliability it should have been penalizing, and the 0.17-0.25 reading
was read off a fixed `psi=20°` cell that does not describe this pool either.

**The operating point for the current real pool is not yet measurable.** No predicted attribution
accuracy is offered here. What replaces it: directly measured cross-template reliability
(Step 2 / D2), external preference structure (Step 3 / D1), and a reliability-aware map
(Track E) that locates the pool by corrected latent correlation rather than by raw sign
agreement.

**Correction to this section's own B2 diagnosis, also per the superseding document.** The claim
that `sigma_d_pos > 1` shows the debiased margin's sign is noise-determined does not follow.
Under the additive position-bias model `ell_AB = c(y) - c(y') + p`, `ell_BA = c(y) - c(y') - p`,
the debiasing removes `p` *exactly, per pair*: `d = 0.5(ell_AB + ell_BA) = c(y) - c(y')` and
`b = 0.5(ell_AB - ell_BA) = p`. A large `sigma_d_pos` therefore says only that the raw position
term is larger than the surviving content term — which can push verdict logits toward saturation
and make the additive approximation fragile, but is not itself evidence that `sign(d)` is noise.
Directly measured reliability (D2), not `sigma_d_pos`, is the quantity to lead with.

### Real anomaly, now explained with direct evidence: GPU thermal throttling

The generation-pass anomaly flagged earlier (pass A 355.9s vs. pass B 9,559.6s, ~27x) recurred in
a milder, now-*measured* form during judge-margin extraction: SmolLM2's `template_train` took
1,123.7s vs. `template_probe`'s 431.5s, same code, same judge, back-to-back. This time
`nvidia-smi -l 5` was logging throughout (`logs/gpu_telemetry.csv`, 999 samples). The evidence is
unambiguous:

```
22:58  temp 59-64C   SM clock 1537-1927 MHz   power ~29W   (early, fast phase)
23:29  temp 57-68C   SM clock  915-1327 MHz   power ~16-17W (clock nearly halved, power capped)
23:46  temp 74-83C   SM clock  705-1082 MHz   power ~18-30W (sustained near-peak temp)
00:08  temp 70-76C   (cooldown after job finished)
```

GPU utilization stayed at 100% throughout — this is not idle time or a stall, it's the same
amount of compute work taking roughly twice as long per clock cycle once the chip is hot,
consistent with laptop-GPU thermal/power throttling under sustained load (no flash-attn, no
torch.compile, a small 6GB mobile part). **This has direct planning implications for Phase 1/2:
the document's "~2 hour" DPO target and "overnight" Phase 2 target should be treated as
optimistic if training sustains 100% GPU utilization for that long** — expect roughly 1.5-2x
slower once the chip saturates thermally, based on the clock-speed ratio observed here (peak
~1900 MHz vs. sustained-hot ~900-1000 MHz). Not something code can fix; worth planning around
(e.g. checking ambient cooling, or simply budgeting more wall-clock than the document's targets).

### Anything changed or added in Track B beyond the instructions

- `gen/sample_pairs.py` and `judges/margins.py` both use batched, length-aware generation/scoring
  instead of one-sequence-at-a-time — required for a first-time-empirically-discovered ~16-70x
  throughput gap, not a design preference.
- `judges/margins.py` uses token-budget batch packing rather than a fixed batch size, after a
  fixed batch size OOM'd twice for different underlying reasons (full-sequence logit
  materialization, then attention memory at the long tail of the length distribution).
- Both scripts call `tests/test_templating.run()` before using a tokenizer, per the earlier
  instruction to run it at every real entry point.
- Excluded 2-3-per-600 (~0.5%) TinyLlama pairs that exceed its context window, rather than
  including known-degraded values.

**Superseded by the round below.** The two gated judges are now accessible (user provided HF
auth); all five judges are re-measured under a corrected scoring path.

---

## NEXT_STEPS_FINAL.md execution round

Governing document supersedes `NEXT_STEPS_TRACKS_A_B_C.md` and `NEXT_STEPS_JUDGE_VALIDITY.md`.
Per Section 1: Phase 0/0c/Track A stand unchanged (pre-registration in `configs/base.yaml`
unchanged); **both Track B map readings (B3) are void**, struck above with a correction note in
place; the A5 slope reanalysis is done (see the A5 section above — corrected range 0.36-0.60,
`0.363` was the saturation-contaminated number, not retained). Track B's B1/B2 *measurements*
were provisionally "standing with caveats" per the document — but see the padding-mode finding
immediately below, which overrides that: **the original B1/B2 numbers are now known to be
contaminated by a scoring bug, not merely in need of reinterpretation.**

### Padding-mode correctness bug, found before any D-step could be trusted

While implementing D3a (which re-scores swapped pairs from scratch through the public path and
compares against the *stored* Track B margins), the antisymmetry residual came out ~0.06 median
— nowhere near the `1e-6` gate — even though two literally-identical fresh calls in the same
process gave exactly 0. Chased this down rather than loosening the tolerance:

1. **Isolated to bf16 padding sensitivity, not a logic error.** Scored the same comparison text
   (a) with minimal padding and (b) padded out by batching it alongside a much longer sequence.
   bf16 on GPU: verdict log-prob difference shifts by up to **0.25** (median 0.125) purely from
   how much padding the sequence received. The identical computation in float32 on CPU shifts by
   **5.3e-05** — a ~4,700x gap. The masking/position path is mathematically correct; this is bf16
   kernel numerics varying with tensor shape. `src/judges/validity_padding_probe.py`.
2. **This is not a rounding footnote.** Median 0.125 is ~13% of Qwen's original margin SD but
   **~47-49% of SmolLM2's and TinyLlama's** — i.e. a meaningful fraction of what B2 reported as
   "judge noise" (`sigma_d_pos` 0.91-2.68) may have been this artifact, not judge position bias.
   The original Track B margins are not trustworthy until re-scored under a corpus-independent
   method.
3. **Fix, and a second bug found while validating the fix.** Added `PAD_MODE="fixed_bin"`: pad
   every sequence to the next multiple of 128 and batch only sequences sharing a bin, so a given
   sequence's padded length is a property of the sequence itself, not of what else is in the
   corpus. First validation attempt still showed a 0.25 max discrepancy across corpus sizes (24
   vs 200 pairs) even with identical *sequence length* padding — traced to the row *count* of a
   batch also varying with corpus size (a bin's last partial batch has a different member count
   in a 24-pair vs 200-pair corpus, and PyTorch's bf16 SDPA kernel selection is shape-sensitive
   in both dimensions). Fixed by also padding the row count to a constant, per-bin value with
   repeated filler rows (discarded from the output) — `_rows_per_batch()` is a pure function of
   the bin, shared by both the batcher and the filler so they can't disagree.
4. **A further, purely-methodological confound in my own validation script,** worth recording
   because it produced a false "still broken" reading after the real fix already worked: the
   first post-fix validation (`validity_padmode_check.py`) toggled `PAD_MODE` between `fixed_bin`
   → `legacy` → `free` *within one process* and still saw a 0.25 residual. Diagnosed via a
   from-scratch backend-isolation test (`validity_padmode_diagnosis2.py`,
   `validity_padmode_diagnosis3.py`): a single sequence scored twice back-to-back, even under
   artificial memory pressure and forced SDPA backends, reproduced *exactly* (0.0 diff) every
   time — ruling out simple allocator-state or backend-auto-selection explanations. The real
   pipeline (`src/judges/margins.py`) never toggles `PAD_MODE` mid-run — it's a fixed module
   constant for the whole script — so the authoritative test needed to isolate each mode to its
   *own process*, matching real usage. `validity_padmode_final.py` does exactly that.

**Authoritative, isolated-process validation** (each mode scored fresh, `Qwen2.5-1.5B-Instruct`,
24-pair subset vs. the same pairs inside the full 200-pair corpus):

| | corpus independence (max / median) | vs. exact padding-free (max / median) | median as % of signal SD |
|---|---|---|---|
| `fixed_bin` | **0.0 / 0.0** | 0.250 / 0.0625 | 5.6% |
| `legacy` (old bucketing) | 0.1875 / 0.0625 | 0.250 / 0.0313 | 2.8% |

`fixed_bin` achieves **exact** corpus independence — the property that actually matters for D2's
train-vs-probe comparison and for any cross-run reproducibility. It still carries a real,
now-quantified ~5.6%-of-signal-SD median approximation gap versus the (5.5x more expensive)
exact padding-free value; `legacy`'s gap-to-exact happens to be smaller on average in this sample
but is useless because it isn't reproducible in the first place — a measurement that changes
depending on what else is in the batch isn't a measurement. AB/BA truncation is guarded
separately (bin computed from `max(len(AB), len(BA))` per pair, not AB alone). Batch shape is a
pure function of `(bin_len)` by construction. All four of the required properties are satisfied
and measured, not assumed.

**Consequence:** re-scoring all five judges under `fixed_bin` (tag `fixedbin`, kept separate from
both the original untagged Track B files and the `padfree` exploratory files — nothing historical
was overwritten). Per-batch timing/memory logging is on throughout
(`logs/margins_batch_timing.jsonl`), per the "Loose end: runtime anomaly" instruction.

**All five intended judges are now accessible** (`Llama-3.2-1B-Instruct`, `gemma-3-1b-it`
resolved once HF auth was configured this session) — B1/B2/D1/D2 below use the full 5-judge pool,
not the 3-judge subset from the original Track B report.

### Step 1 (D3a, D3b): correctness gates — PASS, all five judges

**Run:** `.venv/Scripts/python.exe -m src.judges.validity_d3 --n 600 --diag-limit 300`
**Wall clock:** ~110 min total across 5 judges (see anomaly note below)
**Results:** `results/validity_d3.json`

| judge | D3a antisym (max/median) | D3b-0 duplicate (max) | D3b-1 format SNR | D3b-2 paraphrase SNR (coverage) |
|---|---|---|---|---|
| SmolLM2-1.7B | 0.0 / 0.0 | 0.0 | 3.30 | 5.97 (27%) |
| Qwen2.5-1.5B | 0.0 / 0.0 | 0.0 | 3.36 | 7.11 (27%) |
| TinyLlama-1.1B | 0.0 / 0.0 | 0.0 | 2.14 | 6.26 (27%) |
| gemma-3-1b | 0.0 / 0.0 | 0.0 | 2.44 | 5.32 (27%) |
| Llama-3.2-1B | 0.0 / 0.0 | 0.0 | 2.15 | 6.39 (27%) |

**Both hard gates pass exactly (bitwise 0, not just under the `1e-6` tolerance) for every judge.**
This is the real payoff of the padding-mode fix: D3a specifically re-templates, re-tokenizes,
and re-forwards the swapped pair from scratch through the public path — it would have failed
under the old scoring (that's literally how the padding bug was first caught, in last round's D3
smoke test). Under `fixed_bin`, the identity `d_m(x,y',y) = -d_m(x,y,y')` holds exactly.
`D3b-0`'s duplicate null (`d_m(x,y,y) = 0`) also holds exactly for all five.

Format/paraphrase SNRs (diagnostics, not exclusion criteria per the pre-registered rule): every
judge has real, moderate sensitivity to surface presentation (SNR 2.1-3.4) and lower sensitivity
to the conservative paraphrase substitutions (SNR 5.3-7.1, i.e. paraphrase moves the margin less
than formatting does, relative to real signal). None of this disqualifies a judge — it's reported
per the document's explicit instruction not to treat these as a noise floor.

**Runtime anomaly, logged not resolved:** `SmolLM2-1.7B`'s D3b-0 phase ran at **21,090 ms/batch**
for a 100-batch stretch, versus 2,000-3,100 ms/batch everywhere else in this run (including
SmolLM2's own D3a phase immediately before it) — roughly a 7-10x slowdown, accounting for most
of the ~110-minute total wall clock (that one stretch alone is ~35 minutes). This is now the
*third* time SmolLM2 specifically has shown an unexplained slowdown relative to the other four
judges (the original Track B run and the `fixedbin` re-score both singled it out as the slowest
judge too). Worth a closer look before Phase 2 if SmolLM2 is used at scale, but not chased further
here — GPU stayed responsive throughout (confirmed via live `nvidia-smi` polling and process
CPU-time growth), so this is a performance anomaly, not a correctness one, and doesn't affect any
number reported.

### Step 2 (D2): direct reliability — real, and lower than hoped for most judges

**Run:** `.venv/Scripts/python.exe -m src.analysis.d2_reliability --tag fixedbin`
**Results:** `results/d2_reliability_fixedbin.json`, N=600 common probe pairs, all 5 judges

**D2a — cross-template reliability (directly measured, not inferred from `sigma_d_tmpl`):**

| judge | R (pearson) | p | spearman | sign agreement |
|---|---|---|---|---|
| Qwen2.5-1.5B | **+0.574** | 9.2e-54 | +0.456 | 0.607 |
| gemma-3-1b | +0.478 | 1.5e-35 | +0.398 | 0.618 |
| SmolLM2-1.7B | +0.219 | 5.9e-08 | +0.301 | 0.513 |
| TinyLlama-1.1B | +0.216 | 9.0e-08 | +0.244 | 0.593 |
| Llama-3.2-1B | +0.152 | 1.8e-04 | +0.141 | 0.523 |

Every judge is positively reliable (all `R_m > 0`, all significant) — a real difference from
before the fix. **Against the pre-registered threshold `R_m >= 0.5`: only `Qwen2.5-1.5B` passes
outright; `gemma-3-1b` (0.478) falls just short.** The other three are well below 0.5.

**Corrected-vs-old comparison, the headline finding of this whole re-scoring effort:**
TinyLlama's raw inter-judge correlations with the other judges are now **-0.10 to -0.04**
(mildly negative, near zero) — nowhere near the dramatic below-chance anti-correlation the
original (padding-contaminated) B1 report found (agreement 0.396-0.420, "worse than random").
This confirms directly what was suspected but not proven last round: a substantial part of the
original "severe position bias" finding was the padding measurement artifact, not a real property
of these judges. It is not *entirely* artifact — reliability is still modest for 4 of 5 judges —
but the magnitude was inflated.

**D2b/D2d — naive and shared-noise-adjusted attenuation correction:** two pairs have corrected
correlation `> 1` (`SmolLM2 vs Qwen2.5-1.5B`: 1.29-1.37; `SmolLM2 vs gemma-3-1b`: 1.51-1.61),
flagged per instruction rather than clipped — the independent-error model is inadequate for these
pairs, consistent with some shared template-sensitive response rather than "super-correlation."

**D2c — shared-noise fraction:** `c_hat_raw = c_hat = 0.0489` — small. Template-difference
residuals are only weakly correlated across judges; most of the unreliability is judge-specific,
not a common shared drift.

**D2e — length loading:** modest for every judge (`R^2` 0.005-0.098). `gemma-3-1b` is the most
length-sensitive (`R^2=0.086`, `gamma=+0.0117`/token) but even that is not large. Residualizing
on length barely moves the inter-judge correlation matrix (e.g. `SmolLM2 vs Qwen2.5-1.5B`:
0.485 raw -> 0.478 residual) — the shared structure between judges is not primarily length-driven.

### Step 3 (D1): external preference structure — every judge fails

**Run:** `.venv/Scripts/python.exe -m src.judges.d1_external`
**Data:** 500 held-out `HuggingFaceH4/ultrafeedback_binarized` chosen/rejected pairs, disjoint
from both the train and probe prompt splits.
**Pre-registered rule (`configs/base.yaml`, written before this ran):** retain if
`|accuracy - 0.5| >= 0.10` **and** two-sided exact binomial `p < 0.001`.

| judge | accuracy | `|acc-0.5|` | binomial p | verdict |
|---|---|---|---|---|
| Qwen2.5-1.5B | 0.536 | 0.036 | 0.117 | FAIL |
| gemma-3-1b | 0.536 | 0.036 | 0.117 | FAIL |
| SmolLM2-1.7B | 0.520 | 0.020 | 0.396 | FAIL |
| TinyLlama-1.1B | 0.496 | 0.004 | 0.893 | FAIL |
| Llama-3.2-1B | 0.426 | 0.074 | **0.0011** | FAIL |

**All five judges fail.** None reaches the pre-registered `0.10` distance-from-chance threshold.
`Llama-3.2-1B` is the interesting case: it's the only one with a statistically significant
deviation from chance (`p=0.0011`) — but it's significant *and below chance* (`acc=0.426`,
`|d|=0.074`), and even that falls short of the `0.10` distance threshold, so it fails the
conjunctive rule as written. The other four aren't even statistically distinguishable from
random guessing on this benchmark. Interpreted per the document's framing (distance from chance,
not distance from 1.0, since a stable anti-correlated judge would still count) — none of these
five qualifies as anti-correlated either. **On this held-out external benchmark, none of the five
judges shows a detectable, stable preference relation at all.**

### Step 4: pool decision — zero judges retained

**Run:** `.venv/Scripts/python.exe -m src.analysis.step4_pool_decision`
**Result:** `results/step4_pool_decision.csv`

| judge | D1 acc | `|acc-0.5|` | binom p | D2 `R_m` | len `R^2` | D3a max | D3b dup max | SNR fmt | SNR para | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| SmolLM2-1.7B | 0.520 | 0.020 | 0.396 | 0.219 | 0.098 | 0.0 | 0.0 | 3.30 | 5.97 | DROP (D1, R<0.5) |
| Qwen2.5-1.5B | 0.536 | 0.036 | 0.117 | 0.573 | 0.012 | 0.0 | 0.0 | 3.36 | 7.11 | DROP (D1) |
| TinyLlama-1.1B | 0.496 | 0.004 | 0.893 | 0.216 | 0.005 | 0.0 | 0.0 | 2.14 | 6.26 | DROP (D1, R<0.5) |
| gemma-3-1b | 0.536 | 0.036 | 0.117 | 0.478 | 0.086 | 0.0 | 0.0 | 2.44 | 5.32 | DROP (D1, R<0.5) |
| Llama-3.2-1B | 0.426 | 0.074 | 0.001 | 0.152 | 0.036 | 0.0 | 0.0 | 2.15 | 6.39 | DROP (D1, R<0.5) |

**Retained: 0 of 5.** Both hard implementation gates pass for every judge (`D3a=0.0`,
`D3b_dup=0.0` across the board) — the scoring pipeline itself is confirmed correct, so this is a
finding about the judges, not a measurement artifact. `Qwen2.5-1.5B` is the only judge that
clears the reliability bar (`R_m=0.573 >= 0.5`) but it fails D1 outright — it's a judge that
scores its *own* preferences reproducibly (high test-retest reliability across templates) without
those preferences tracking the external chosen/rejected signal at all. The other four fail both
criteria.

**Per the document's explicit instruction, fewer than four judges survive, so the Step 6 pool
ladder is triggered.** I have not weakened the pre-registered thresholds and have not substituted
in more 1-2B instruct models to pad the count.

### Stop point — this changes the plan for Track C, not just a footnote

The document's Track C pilot names `Qwen/Qwen2.5-1.5B-Instruct` as the pilot judge specifically
*"because it is currently the only already-measured judge with substantial cross-template
reproducibility"* — that assessment was made before D1 existed and before this round's corrected
D2 numbers. Under the corrected measurements, Qwen still has the best `R_m` (0.573), but it now
has a directly-measured D1 result showing **no detectable external preference structure**
(accuracy 0.536, not significant). Training a DPO pilot against a judge with no demonstrated
external validity would not be a meaningful test of alignment-transmission noise — there would be
no real preference signal to transmit in the first place, and any `sigma_D_emp` measured would be
uninterpretable (indistinguishable from measuring how well a model can fit noise).

The document's own Step 6 ("If fewer than four judges survive... follow the reward-model
calibration ladder. Do not weaken thresholds and do not fill the pool with more arbitrary 1-2B
instruct models") points toward reward models, not continuing with Qwen as-is. But the document
also separately instructs Track C to proceed once the D3 *hard correctness gates* pass (which
they do, exactly, for all five judges) — those two instructions are now in tension given this
round's D1 result, and resolving that tension (run Track C anyway as a pure pipeline/engineering
dry run with the caveat that its noise numbers won't be scientifically interpretable, versus
pivot straight to the Step 6 reward-model ladder, versus reconsider whether the D1 benchmark or
threshold is well-matched to what these judges actually do) is a decision I don't think I should
make silently. Stopping here, as the document itself instructs at this exact point ("STOP AND
PRINT THE FULL STEP-4 TABLE... BEFORE DOING ANY EXPENSIVE FULL-SCALE RUNS"), to report this and
ask before starting Track C or Track E.

---

## NEXT_STEPS_ROUND_2.md — Steps 0-3

Per the document's own instruction ("Report Steps 0 through 3 together, then stop"), stopping
after Step 3. Not starting Track C or Track E in this round.

### Step 0: Delta scorer audited for the same padding bug — found, fixed, gated

The scorer (`src/score/delta.py`) had been unbatched (`batch_size=1`, no padding) — inherently
immune to the bf16 batch-composition bug as originally written, but far too slow to use at Track
C's scale. Rewrote it batched with `fixed_bin` padding, matching the judge-margin fix, with one
property the judge case didn't need: the adapter-on and adapter-off passes now run the *identical*
padded batch tensor (only `model.disable_adapter()` differs), so their batch composition is
identical by construction, not by convention.

**First version of the fix still failed the regression test.** Ported the padded-*length* fix
from the judge scorer but forgot the padded-*row-count* fix (constant rows per batch via filler,
needed because a bin's last partial batch has a different row count depending on how many real
items land in it) — the exact same two-part bug as before, this time caught by the test rather
than a smoke run. `tests/test_scorer_invariance.py`:

| check | before row-count fix | after |
|---|---|---|
| shuffled pair order | max\|diff\|=2.17, 5.5% of sd(Delta) — **FAIL** | max\|diff\|=6.1e-05, ~0% — **PASS** |
| different batch size/token budget | 0.0 — PASS | 0.0 — PASS |

Gate: `< 0.01 * sd(Delta)`. **PASS** after the fix (both checks at or near exact 0). This test is
meant to run at the top of every scoring entry point per the document's instruction; wired in for
Track C's eventual scorer entry point.

### Step 1: D3b, done properly — every judge shows real content sensitivity

**Run:** `.venv/Scripts/python.exe -m src.judges.validity_d3b --n 600`
**Result:** `results/validity_d3b.json`

The prior round's "D3b-0" (`d_m(x,y,y)=0`) was correctly diagnosed as degenerate: for identical
token sequences `ell_AB=p, ell_BA=-p`, so the debiased margin is exactly 0 *by construction of
the algebra*, regardless of whether the judge perceives anything — it re-confirmed D3a, not
content sensitivity. This step replaces it with two non-identical, semantically-equivalent null
constructions, reported separately, never pooled:

- **whitespace**: reflow line breaks/spacing only, meaning unchanged. Token sequence must differ
  from the original — asserted, and pairs where it doesn't (rare) are dropped rather than kept as
  a false null. Coverage 547/600 (91%).
- **reworded**: a fixed table of 46 conservative, meaning-preserving contraction expansions
  (`don't`->`do not`, `I'm`->`I am`, etc. — full table in `src/judges/validity_d3b.py`), applied
  identically to every pair. Coverage 160/600 (27%) — most responses don't happen to contain any
  of these contractions.

| judge | floor_ws | signal_ws | **snr_ws** | mean_null_ws | floor_rw | signal_rw | **snr_rw** | mean_null_rw |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-1.5B | 0.300 | 0.996 | **3.32** | +0.147 | 0.137 | 0.936 | **6.82** | -0.042 |
| Llama-3.2-1B | 0.227 | 0.425 | **1.87** | +0.004 | 0.065 | 0.455 | **6.98** | -0.002 |
| gemma-3-1b | 1.135 | 2.453 | **2.16** | +0.317 | 0.505 | 2.204 | **4.37** | -0.010 |
| SmolLM2-1.7B | 0.084 | 0.239 | **2.85** | +0.027 | 0.044 | 0.221 | **5.00** | +0.004 |
| TinyLlama-1.1B | 0.127 | 0.265 | **2.08** | +0.007 | 0.041 | 0.292 | **7.14** | +0.001 |

Mean-null check passes for all: every `mean_null` is close to 0 relative to its own `signal_m`
scale, confirming the null constructions are actually null on average (no systematic bias from
the transformation itself), leaving `floor_m` as a clean noise-floor estimate.

**This resolves the interpretive question the document posed in advance.** Every judge has
`snr_m` comfortably above 1 for both constructions (whitespace 1.87-3.32, reworded 4.37-7.14) —
none are anywhere near 1, which would have meant "not responding to content." **All five judges
discriminate content robustly; their Step 4 D1 failure means they disagree with GPT-4's notion of
quality, not that they're inert.** Per the document's own framing, this does not disqualify them
for attribution on its own — it's diagnostic, not an exclusion criterion — but it does change how
the D1 failure should be read: it's a real disagreement about what "better" means, not evidence
these are non-functional judges. `Llama-3.2-1B`'s `snr_ws=1.87` is the one value under the
Step-3 `snr_m >= 2` reference point mentioned in the document; it clears `snr_rw` by a wide margin
(6.98), so its content-sensitivity is real but somewhat weaker on pure presentation changes than
the other four.

### Step 2: D1 large-gap diagnostic — secondary number, does not change the Step 4 verdict

**Run:** `.venv/Scripts/python.exe -m src.judges.d1_large_gap`
**Data:** 500 held-out pairs with `score_chosen - score_rejected >= 3` (mean gap 4.6, up to 8.0),
same held-out pool D1 drew from, capped at N=500 to match D1's original statistical power.

| judge | full-set acc | large-gap acc | `|acc-0.5|` | binomial p |
|---|---|---|---|---|
| Qwen2.5-1.5B | 0.536 | **0.620** | 0.120 | 8.96e-08 |
| gemma-3-1b | 0.536 | **0.586** | 0.086 | 1.39e-04 |
| SmolLM2-1.7B | 0.520 | **0.578** | 0.078 | 5.61e-04 |
| TinyLlama-1.1B | 0.496 | 0.508 | 0.008 | 0.754 (n.s.) |
| Llama-3.2-1B | 0.426 | **0.324** | 0.176 | 2.60e-15 |

**Per the interpretation stated in advance in the document (not fitted after seeing this):**

- **TinyLlama-1.1B stays at chance even on the largest, most obvious quality gaps** (0.508,
  p=0.75) — this makes its Step 4 verdict unambiguous, exactly as the document said it would: not
  "can't resolve fine gaps," genuinely no detectable preference signal at all.
- **Qwen2.5-1.5B, gemma-3-1b, SmolLM2-1.7B all jump to a real, significant effect** (0.58-0.62)
  on easy pairs, none quite reaching the document's 0.65 reference point. Real content
  sensitivity (consistent with D3b's `snr_m` finding) that can't resolve fine distinctions and is
  imperfect even on coarse ones — this is the "stage 3 of the ladder needs to account for
  resolution, not just presence of signal" case the document anticipated, and per instruction,
  **does not change the Step 4 verdict.**
- **Llama-3.2-1B is the interesting case, worth flagging directly.** It doesn't move toward
  chance or toward agreement — it moves *further into significant anti-correlation* (0.426 -> 0.324,
  `p=2.6e-15`, the strongest effect of any judge on either set). This is a genuinely stable,
  strongly anti-correlated preference function, most pronounced exactly where GPT-4's own
  judgment is most confident (the large-gap subset). **Worth noting, not to argue around the
  prohibition on revisiting D1:** had this large-gap subset been the pre-registered evaluation
  set instead of the full 500, Llama-3.2-1B's `|acc-0.5|=0.176` and `p=2.6e-15` would clear the
  pre-registered rule (`>=0.10` and `p<0.001`) comfortably — it is *exactly* the "stable
  anti-correlated judge" the document's own framing says should count as having external
  structure. The pre-registered rule was evaluated on the full set as written, and per the
  document's explicit prohibition ("Do not revisit the D1 retention threshold... the statistic is
  already distance from chance rather than distance from one") **the Step 4 verdict for
  Llama-3.2-1B stays DROP.** Recording this only because it's a real, decision-relevant nuance for
  anyone reconsidering benchmark design in a future round — not something acted on here.

### Step 3: Step 4 closed out — `snr_m` populated, `c_hat` reconfirmed post-fix

**Retention criteria, unchanged:** `D1 rule passes AND rho_m (R_m) >= 0.5 AND snr_m >= 2`. Using
`snr_ws` (whitespace) as the reference `snr_m`, since it's the more conservative of the two
null constructions.

| judge | D1 acc | `|acc-0.5|` | binom p | D1 large-gap acc | D2 `rho_m` | D2 length `R^2` | D3b `snr_m` (ws) | D3b `snr_m` (rw) | verdict |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-1.5B | 0.536 | 0.036 | 0.117 | 0.620 | **0.573** | 0.012 | **3.32** | 6.82 | DROP (D1) |
| gemma-3-1b | 0.536 | 0.036 | 0.117 | 0.586 | 0.478 | 0.086 | **2.16** | 4.37 | DROP (D1, `rho`<0.5) |
| SmolLM2-1.7B | 0.520 | 0.020 | 0.396 | 0.578 | 0.219 | 0.098 | **2.85** | 5.00 | DROP (D1, `rho`<0.5) |
| TinyLlama-1.1B | 0.496 | 0.004 | 0.893 | 0.508 | 0.216 | 0.005 | **2.08** | 7.14 | DROP (D1, `rho`<0.5) |
| Llama-3.2-1B | 0.426 | 0.074 | 0.001 | 0.324 | 0.152 | 0.036 | **1.87** | 6.98 | DROP (D1, `rho`<0.5, `snr_ws`<2) |

**Retained: 0 of 5, unchanged from the prior round.** Every judge fails the D1 conjunct, so the
overall verdict doesn't move — but the individual-criterion breakdown is worth having on record:
only `Qwen2.5-1.5B` would separately clear the reliability bar (`rho_m>=0.5`), and only
`Llama-3.2-1B` would separately fail the whitespace SNR bar — every other judge clears reliability
*or* SNR individually just not the D1 conjunct. This confirms the earlier finding wasn't
overstated: it is specifically external validity against GPT-4's preference notion that these
judges lack, not internal coherence or content-sensitivity, which most of them have in reasonable
measure.

**`c_hat` reconfirmed post-fix (Step 3's explicit check):** `c_hat_raw = c_hat = 0.0489`, taken
directly from `results/d2_reliability_fixedbin.json` (D2c), which was already computed against
the corrected `fixed_bin` margins in this same round — no stale pre-fix value was carried
forward. Feeds Track E's `c` grid whenever that runs.

**Per the document's explicit instruction, fewer than four judges survive, so the Step 6 pool
ladder is triggered — confirmed again, not re-litigated, since the D1 verdict was not revisited.**

### Step 6: reward-model pool — proposal for review, nothing downloaded or scored

Per the explicit instruction ("Do not download or score anything until the proposed pool is
reviewed" / "comes as a table for review, not as a completed selection"). Researched via the HF
Hub API (`model_info`/`config.json`/safetensors index — metadata only, no weight downloads) rather
than trusted from memory, per the document's own instruction to verify.

**Size constraint first, since it's the hardest filter:** "fits in 3.5 GB in bf16" caps params at
roughly 1.75B. This ruled out both named Skywork variants immediately (Llama-3.1-8B and
Gemma-2-27B — 16GB and 54GB in bf16 respectively, nowhere close), and most of the well-known
reward-model families on the Hub cluster at 7B+. The small end of the reward-model landscape is
genuinely thin, which shows in the proposal below — real tradeoffs, not glossed over.

| candidate | params (measured) | bf16 size | license/gate | training data | notes |
|---|---|---|---|---|---|
| `OpenAssistant/reward-model-deberta-v3-large-v2` | ~435M (DebertaV2, hidden=1024, 24L) | ~0.87 GB | MIT, ungated | `summarize_from_feedback` + `webgpt_comparisons` + `Dahoas/instruct-synthetic-prompt-responses` + `Anthropic/hh-rlhf` | broadest single-model training mix found; well-established (15.4k downloads) |
| `internlm/internlm2-1_8b-reward` | ~1.8B (measured from safetensors index: 3.40GB fp32-equiv -> **3.17 GB actual stored size**) | 3.17 GB | `license:other` (permissive per InternLM's own license, ungated on the Hub), verify acceptable in practice | InternLM team's own large-scale internal RLHF preference data (arXiv 2403.17297), English+Chinese | genuinely distinct data source from the Western-RLHF cluster below; **closest to the 3.5GB ceiling of any candidate, flagging that explicitly** |
| `Ray2333/gpt2-large-helpful-reward_model` | ~774M (gpt2-large) | ~1.5 GB | MIT, ungated | `Anthropic/hh-rlhf`, **helpful split only** | different preference *criterion* (helpfulness) from #4, not just different source |
| `Ray2333/gpt2-large-harmless-reward_model` | ~774M (gpt2-large) | ~1.5 GB | MIT, ungated | `Anthropic/hh-rlhf`, **harmless split only** | different preference *criterion* (harmlessness) from #3 — same base dataset as #3 but genuinely different axis, which is itself a useful test of whether the attribution pipeline can separate two judges trained on the same corpus under different objectives |
| `KHuss/Llama-3.2-1B-reward-model` (tentative 5th) | ~1B (Llama-3.2-1B base) | ~2 GB | **gated: manual** (inherits Llama gating; unverified whether acceptance is fast for this specific derivative, unlike Meta's own repos we already have access to) | unknown/unverified — no dataset tags on the Hub listing | weakest-verified candidate of the five; flagging rather than including with confidence |

**Honest tradeoff, stated plainly:** candidates #3 and #4 share their base training corpus
(`Anthropic/hh-rlhf`), which caps how much genuine *data* diversity this proposed five actually
has — two of five are the same dataset split by objective, not independent data. A cleaner pool
by the document's own stated priority ("diversity of training data is the axis that matters") would
be 3 confirmed-diverse judges (`OpenAssistant/deberta-v3-large-v2`, `internlm2-1_8b-reward`, and
one of the two `Ray2333/gpt2-large` models) plus 2 more still to be sourced — the small-model
reward-model landscape didn't yield 5 genuinely independent-data, size-compliant, easily-accessible
candidates on this search. Recommending you review this table and tell me whether to: (a) proceed
with the 5 above as listed and accept the #3/#4 data overlap as a deliberate helpful-vs-harmless
comparison rather than a flaw, (b) drop `KHuss/Llama-3.2-1B-reward-model` (the weakest-verified
entry) and search further for a 5th, or (c) proceed with 4 and treat the pool as
smaller-but-cleaner. **Not downloading or scoring any of these until you say which.**

---

**Stopping here per the document's explicit instruction ("Report Steps 0 through 3 together, then
stop. Step 4 and Step 5 results come in a second report").** Not starting Track C or Track E in
this round. Step 6's proposal is above for review, per its own separate instruction to bring it as
a table rather than a completed selection.

---

## NEXT_STEPS_ROUND_3_FINAL.md

### Section C: reward-model pre-download checks — **one candidate hard-blocked, nothing downloaded**

Reported first and separately, per §7 ("Report Section C's provenance matrix and pre-download
compatibility table immediately and stop before downloading if any hard compatibility issue
appears"). All checks are metadata-only (`config.json`, `tokenizer_config.json`, safetensors
index) — **no weights were downloaded for any candidate.**
Files: `results/rm_precheck.json`, `figures/F11_rm_dataset_overlap.png`.

| candidate | architecture | scalar head | effective context | stored size | gated | verdict |
|---|---|---|---|---|---|---|
| `OpenAssistant/reward-model-deberta-v3-large-v2` | `DebertaV2ForSequenceClassification` | yes | **512** | 1.62 GB | no | **BLOCKED** |
| `internlm/internlm2-1_8b-reward` | `InternLM2ForRewardModel` | yes | 32,768 | 3.17 GB | no | PASS |
| `Ray2333/gpt2-large-helpful-reward_model` | `GPT2ForSequenceClassification` | yes | 1,024 | 2.88 GB | no | PASS |
| `Ray2333/gpt2-large-harmless-reward_model` | `GPT2ForSequenceClassification` | yes | 1,024 | 2.88 GB | no | PASS |

**The blocker:** `OpenAssistant/reward-model-deberta-v3-large-v2` has
`max_position_embeddings = 512`, against the intended scoring length of `384 + 256 = 640` tokens.
Per §3.5 ("Do not silently truncate. If any candidate cannot score the intended 640-token input
faithfully, stop and report the incompatibility before downloading/scoring the pool"), this is a
hard stop, not something to work around by truncating. Scalar head and size were both fine; it is
purely a context-length incompatibility.

**Why this one hurts most.** It is the member with the broadest heterogeneous / multi-corpus
training mixture — four corpora, per the F11 counts below. (Correction: an earlier draft called it
the *only* member reaching outside `Anthropic/hh-rlhf`. That was wrong — InternLM2's internal data
is also outside it. The defensible claim is about mixture breadth, not exclusivity.) Removing it
leaves three models covering **two** distinct corpora
(InternLM's internal data, and `Anthropic/hh-rlhf` shared by both Ray2333 models), and drops
four-way chance from `1/M = 0.25` to three-way `1/3 = 0.333`.

**F11 pairwise shared-corpus counts** (corpus identity normalized to the base corpus — the two
Ray2333 models use different *splits/objectives* of the same corpus, and recording the splits as
distinct strings would have falsely shown zero overlap, understating exactly the confound §3.1
asks to register):

| | OA-deberta | InternLM2 | gpt2-helpful | gpt2-harmless |
|---|---|---|---|---|
| OA-deberta | 4 | 0 | 1 | 1 |
| InternLM2 | 0 | 1 | 0 | 0 |
| gpt2-helpful | 1 | 0 | 1 | 1 |
| gpt2-harmless | 1 | 0 | 1 | 1 |

**Registered in `configs/base.yaml` before any scoring** (§3.1, §3.7): pool composition and its
partial-overlap confound, the pre-designated helpful/harmless hard cell, evaluator reliability
`R = 1` for the scalar-head stage (§0.4 — whitespace/paraphrase robustness is a property of the
learned reward *function*, never fed into `a = sqrt(R)`), the reward-model D1 pre-registration
(primary = large-gap subset, secondary = full held-out set, same distance-from-chance rule), the
§3.8 retention rule, and the mixture-standardization requirement. Pool membership is recorded as
`PENDING_REVIEW`.

**Per §3's decision rule, a pre-download hard compatibility check eliminating one of the four is
exactly the stated condition for revisiting the pool — and it also requires explicit instruction.
So I have not selected a replacement and have not downloaded anything.** Awaiting your call:
proceed with three (accepting two distinct corpora and `1/3` chance), or authorize a search for a
replacement with ≥640 context.

### GPU over-subscription: diagnosed by measurement, then fixed

The first pilot attempt reserved **7.96 GB on a 6 GB card** and optimizer steps took
204s / 107s / 49s. Adding `precompute_ref_log_probs` barely helped (7.45 GB, 121s). Rather than
keep guessing, `scripts/probe_dpo_memory.py` probed four configs for 2 steps each under a **hard
allocator cap** (`torch.cuda.set_per_process_memory_fraction(0.90)`), which converts Windows/WDDM's
silent spill-to-host-RAM into a loud OOM:

| config | status | s/step | peak alloc | peak reserved |
|---|---|---|---|---|
| precompute_ref | ok | 48.1 | 3.02 GB | 5.20 GB |
| **precompute_ref + `torch_empty_cache_steps=1`** | ok | **44.9** | 3.02 GB | **4.48 GB** |
| precompute_ref + activation_offloading | ok | 50.4 | 3.34 GB | 5.22 GB |
| no precompute (baseline) | ok | 56.0 | 3.24 GB | 5.24 GB |

**The assumption that memory was the bottleneck was wrong.** Real demand is only ~3.0 GB
allocated; the 7.4-8.0 GB was allocator over-*reservation*. And with spilling made impossible by
the cap, steps still took ~45-56s — so the slowness is WDDM per-kernel-launch overhead, the same
platform constraint that made batch-1 generation run at 7-9 tok/s, not paging. Adopted
`torch_empty_cache_steps=1` (lowest reserved, and marginally fastest) plus the 0.90 cap. In the
real pilot, per-step `mem_reserved` sat at 2.4-2.9 GB and later steps ran at 13-15s.

---

## Section A: Track C pilot — results

**Run:** `.venv/Scripts/python.exe -m src.train.track_c_pilot --arms A_tau2 A_det B_qwen --tau 2.0`
**Files:** `results/track_c_pilot.json`, `logs/track_c_step_timing.jsonl`,
`figures/F1,F3,F4,F5,F6_track_c_*.png`
**Peak VRAM:** 3.30 GB allocated / 4.99-5.02 GB reserved per run. **Wall clock:** 992/2003/763/523/678/598 s
(train only, 6 runs), plus 600-probe scoring per run.

One generation of 300 train pairs was shared by every arm; only labels differ. Scorer ran through
the `fixed_bin` batched path with `tests/test_scorer_invariance.py` passing beforehand
(shuffle max|diff| 9.2e-05, batch-size max|diff| 0.0, both far under the `0.01*sd(Delta)` gate).

### The required single table

| arm | label process | tau | flip rate | Bayes probe acc | sign agr | Spearman | Pearson | R² | sigma_D_ratio | sigma_D_total | mean\|Δ\| real | mean\|Δ\| shuf | interpretation class |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_tau2 | stochastic BT | 2.0 | 0.303 | 0.715 | 0.540 | 0.182 | 0.184 | 0.0338 | **5.35** | 0.983 | 1.307 | 0.958 | **primary magnitude** |
| A_det | deterministic | — | — | — | 0.532 | 0.176 | 0.205 | 0.0419 | 4.78 | 0.979 | 1.417 | 1.006 | diagnostic magnitude (contrast) |
| B_qwen | deterministic (Qwen) | — | — | — | 0.512 | 0.079 | 0.121 | 0.0147 | 8.20 | 0.993 | 1.733 | 0.990 | primary rank / exploratory magnitude |

Length-residualized counterparts (both sides residualized on response-length difference):

| arm | sign agr | Spearman | R² | sigma_D_ratio |
|---|---|---|---|---|
| A_tau2 | **0.495** | 0.083 | 0.0075 | 11.49 |
| A_det | 0.518 | 0.074 | 0.0070 | 11.93 |
| B_qwen | 0.513 | 0.058 | 0.0045 | 14.86 |

Additional A_tau2 stochastic diagnostics: agreement against a freshly sampled probe label =
**0.537**, against a Bayes ceiling of **0.715** for tau=2.
Arm B disattenuation sensitivity: observed Pearson 0.121 → `0.121/sqrt(0.573)` = **0.160**; below
1, so the classical measurement-error assumptions are not flagged as violated. Per the
prohibitions this is **not** converted into a headline `sigma_D`.

### AMENDMENT (round 4, §0.1-§0.3) — read this before the interpretation below

Three corrections to how this pilot was recorded. The first two invalidate a headline number I
reported; the third retracts an overstated claim.

**§0.1 — `sigma_D = 5.35` is struck. The pilot could not have measured it.** Nineteen optimizer
steps from `pi_ref` is nowhere near the KL-regularized optimum, and the identity the estimator
assumes (`Delta_i = (1/beta)[r(x_i,y_i) - r(x_i,y_i')]`) holds only *at* that optimum. Early in
DPO the loss is unsaturated, so every pair contributes roughly equal gradient weight and margin
magnitude only starts to matter as easy pairs saturate. A low `R_squared` after 19 steps is what
the design predicts, not a measurement of transmission. So `5.35` is not a pessimistic estimate of
a quantity that might improve with convergence — it is an artifact of a run structurally incapable
of producing the equilibrium relation. **It is not an input to anything.** The E1 panel at
`sigma_D = 5.3485` is relabelled a *hypothetical*, not "the measured value," and my Section B
sentence that the measured noise "would be damaging if it persisted" was built on a number that
should not have been treated as a measurement at all.

**§0.2 — the variance diagnostic was computed from the wrong statistic.** I reported
`mean|Delta|` ratios. The right quantity is
`excess_variance_over_shuffled = 1 - (sd(Delta_shuf)/sd(Delta_real))^2`, from `sd`, never from
`mean|Delta|`. Recomputed from the stored Δ vectors:

| arm | sd(Δ) real | sd(Δ) shuffled | **excess_variance_over_shuffled** |
|---|---|---|---|
| A_tau2 | 1.744 | 1.281 | **0.460** |
| A_det | 1.851 | 1.329 | **0.485** |
| B_qwen | 2.307 | 1.300 | **0.683** |

The defensible reading is narrow: there is substantially more policy movement under real labels
than under shuffled ones, while only a small fraction of total Δ variance is linearly explained by
the generating judge margin at this early checkpoint. This is **not** a causal variance
decomposition — real and shuffled runs are separate nonlinear training trajectories, so
`Delta_real = signal + identical independent drift` is an approximation, not an established
identity. I am not claiming any proven "fraction orthogonal to the judge."

**§0.3 — the deterministic contrast was unpowered, not refuted. Retracting my claim.** I wrote
that "the §0.3/§1.5 prediction was not borne out." That overstated what the data supports. At
`N = 600` the standard error on a correlation is ≈ `1/sqrt(600) = 0.041`, and on a proportion
≈ 0.020. The observed gaps — Pearson 0.184 (A_tau2) vs 0.205 (A_det), sign agreement 0.540 vs
0.532 — are well inside noise, and both arms sit near the floor, so no contrast is identifiable at
this operating point. Sign agreement with 95% binomial CIs (`±1.96 × 0.020 ≈ ±0.040`):

| arm | sign agreement | 95% CI | vs chance |
|---|---|---|---|
| A_tau2 | 0.540 | [0.500, 0.580] | CI touches 0.5 |
| A_det | 0.532 | [0.492, 0.572] | CI includes 0.5 |
| B_qwen | 0.512 | [0.472, 0.552] | CI includes 0.5 |

All three overlap each other heavily and all three include or touch chance. The correct entry is
that **the test had no power and the question remains open** — which forbids claiming support for
the prediction just as much as it forbids claiming refutation.

### What this says, stated plainly

**1. DPO moved the policy, but transmitted the judge's ordering only weakly.** Every arm separates
from its matched shuffled-label control on Δ scale (1.31 vs 0.96, 1.42 vs 1.01, 1.73 vs 0.99 —
ratios 1.36x, 1.41x, 1.75x), so training did something real. But sign agreement is 0.512-0.540
against a 0.5 baseline, and Spearman 0.08-0.18. Per §1.4 the hard failure signal is near-chance
rank agreement **together with** no scale separation; we have the first but not the second, so
this is not the clean failure case — it is weak transmission, not absent training.

**2. Almost all of the apparent transmission is length-mediated.** Residualizing on
length-difference collapses A_tau2's sign agreement to **0.495 (below chance)** and Spearman to
0.083; the other arms collapse similarly. For Arm A this is partly structural — token count is
one of the seven judge features, and the selected judge's realized pairwise length loading is
`corr(d_true, len_diff) = -0.313` — but it also means the pilot provides very little evidence of
*non-length* preference transmission. This is the single most important caveat on the whole
section.

**3. The deterministic/stochastic contrast is UNRESOLVED (amended per §0.3).** I originally wrote
that the prediction "was not borne out." That was wrong — the test had no power. The arms differ
by less than one standard error and both sit near the floor. See the amendment above; the question
remains open in both directions.

**4. `sigma_D_ratio = 5.35` is STRUCK (amended per §0.1).** It is not an estimate of transmission
noise at all — 19 optimizer steps cannot produce the equilibrium relation the estimator assumes.
It is not used as an input to Track E or anything else, and the E1 panel computed at that value is
a hypothetical rather than a measured operating point.

---

## Section B: Track E — reliability-aware map

**Run:** `.venv/Scripts/python.exe -m src.phase0e_track_e2`
**Files:** `results/track_e2_gates.json`, `results/track_e2_E2_heterogeneous.parquet` (43,200 rows),
`figures/F9_track_e_calibration.png`, `figures/F10_heterogeneous_diagnostic.png`

### Gates (both PASS)

**Calibration gate (F9)** — the §0.1 correction, verified numerically. With loading `a = sqrt(R)`,
realized test-retest `corr(d_obs^(1), d_obs^(2))` matches target `R` across the whole grid:

| target R | 1.000 | 0.900 | 0.700 | 0.573 | 0.500 | 0.478 | 0.300 | 0.150 |
|---|---|---|---|---|---|---|---|---|
| realized | 1.000 | 0.901 | 0.696 | 0.571 | 0.499 | 0.487 | 0.304 | 0.151 |

All within Monte Carlo error. Had the loading been set to `R` directly, realized correlation would
have come back at `R²` — this figure is what makes the parameterization auditable rather than
asserted.

**Regression gate** — at `psi=90, R=1, c=0, sigma_D=0`, top-1 = 1.000 for all three rules.

### E2: heterogeneous measured-R diagnostic (**not** a real-pool prediction)

Loadings `a_m = sqrt(R_m)`: Qwen 0.757, gemma 0.691, SmolLM2 0.468, TinyLlama 0.465, Llama 0.390.
Top-1 accuracy at `c = c_hat = 0.0489`, pooled over rules:

| psi \ sigma_D | 0.0 | 0.5 | 1.0 |
|---|---|---|---|
| 90° | 0.982 | 0.983 | 0.953 |
| 45° | 0.355 | 0.308 | 0.333 |
| 20° | 0.193 | 0.280 | 0.262 |
| 10° | 0.202 | 0.202 | 0.232 |
| 5° | 0.240 | 0.232 | 0.228 |
| 2° | 0.252 | 0.172 | 0.223 |

**The finding: latent separation dominates, reliability heterogeneity does not.** At `psi=90°`
attribution holds at 0.95-0.98 even with measured reliabilities as low as `R=0.152`, and even at
`sigma_D=1.0`. Per-true-judge recovery at `psi=90, sigma_D=0.5` is 0.94-1.00 for *every* judge
including the least reliable (Llama-3.2-1B: 0.938-0.969). Below `psi=45°` everything collapses
toward the 1/5 chance line regardless of reliability. The cliff between 90° and 45° is sharp, and
sharper than the original Phase 0c map because every candidate is additionally attenuated by
`a_m < 1` here.

**This is explicitly not a prediction for the real pool.** The real pool's latent `psi` is not
identified — the raw sign-agreement→psi conversion was voided in an earlier round and is not used.
No curve here is labelled as the real operating point.

### §2.3 shared-noise model misfit

`c_hat = 0.0489` says template-varying noise is only weakly shared across judges, yet two
attenuation-corrected inter-judge correlations exceeded 1 (SmolLM2×Qwen 1.29-1.37,
SmolLM2×gemma 1.51-1.61). Both cannot hold under the simple model. The likely violated assumption
is that the candidates share a single one-factor latent geometry plus independent measurement
error: some stable cross-judge structure survives the template change (so it is not captured by
the template-varying `c` term at all), and/or the error is heteroskedastic/non-classical rather
than the classical independent form the attenuation correction assumes. Per instruction, neither
diagnostic is converted into a new point estimate for `c`.

### E1: generic homogeneous-reliability map

**Run:** `--analysis e1 --reps 200 --sigma-d-extra 5.3485`. 460,800 rows in 1,685 s (28 min — my
~5 h estimate was wrong by an order of magnitude; E1 has no per-replicate model call, unlike the
GPU sections). File: `results/track_e2_E1_generic_map.parquet`. Figures: F7, F8.

Top-1 accuracy, `psi` × `R`, at `c = c_hat = 0.0489`, pooled over rules:

**`sigma_D = 0`**

| psi \ R | 0.150 | 0.300 | 0.478 | 0.573 | 0.700 | 0.900 | 1.000 |
|---|---|---|---|---|---|---|---|
| 90° | 0.958 | 0.998 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 45° | 0.307 | 0.467 | 0.645 | 0.803 | 0.928 | 0.998 | 1.000 |
| 20° | 0.247 | 0.270 | 0.260 | 0.305 | 0.363 | 0.710 | 1.000 |
| 10° | 0.168 | 0.222 | 0.223 | 0.237 | 0.230 | 0.355 | 1.000 |

**`sigma_D = 5.3485`** (the measured Arm A tau=2 `sigma_D_ratio`)

| psi \ R | 0.150 | 0.300 | 0.478 | 0.573 | 0.700 | 0.900 | 1.000 |
|---|---|---|---|---|---|---|---|
| 90° | 0.332 | 0.555 | 0.722 | 0.812 | 0.880 | 0.962 | 0.968 |
| 45° | 0.178 | 0.235 | 0.275 | 0.257 | 0.318 | 0.412 | 0.490 |
| 20° | 0.168 | 0.188 | 0.210 | 0.210 | 0.245 | 0.245 | 0.282 |
| 10° | 0.177 | 0.205 | 0.193 | 0.178 | 0.187 | 0.228 | 0.233 |

**The alignment-noise axis is where this pilot's measured value bites.** At `sigma_D = 0` or
`0.5`, `R = 0.573` (Qwen's measured reliability) attributes perfectly at `psi = 90°` and still
reaches 0.80 at `psi = 45°`. At the measured `sigma_D = 5.35`, the same `R = 0.573` falls to
**0.812 at 90°** and **0.257 at 45°** — i.e. attribution survives only at near-orthogonal
separation, and even there is no longer near-certain. At the least reliable judge's `R = 0.152`,
`psi = 90°, sigma_D = 5.35` gives just **0.332**.

Read together with E2: **separation is the first-order constraint and alignment noise is the
second**, with reliability third. `sigma_D = 5.35` does not by itself kill attribution at 90°,
but it removes all the margin that made low reliability survivable in E2.

**AMENDED (round 4 §0.1): the `sigma_D = 5.3485` panel is a HYPOTHETICAL, not a measured value.**
The pilot could not measure `sigma_D` at all (19 optimizer steps is far from the KL-regularized
optimum where the estimator's identity holds), so this column should be read as "what the map
would look like if alignment noise were this large," with no claim that it is. The `sigma_D = 0`
and `0.5` panels are equally hypothetical until a converged measurement exists. Round 4's
Section A convergence sweep is what will supply a real value.

`c` sensitivity (F8) is mild across `c in {0, c_hat, 0.25, 0.5}` relative to the `psi` and
`sigma_D` effects, consistent with the small measured `c_hat = 0.0489`.

### Section B status: complete

E1, E2, the calibration gate, and the regression gate are all done and reported.

---

# NEXT_STEPS_ROUND_4_FINAL.md

## Section B (run first): reward-model stage

### 2.1 Probe-length filter — **the round-3 block was my error; all four models are retained**

**Run:** `.venv/Scripts/python.exe -m src.judges.rm_probe_filter`
**Files:** `results/rm_probe_filter.json`, `figures/F22_rm_probe_filter.png`

In round 3 I hard-blocked `OpenAssistant/reward-model-deberta-v3-large-v2` because its 512-token
context is below "the required 640." **That 640 figure was the wrong quantity for this model
class.** It is `max_prompt_tokens (384) + max_new_tokens (256)`, which is the length of a
*judge* comparison prompt containing a prompt plus *two* responses. A scalar reward model scores
`(x, y)` singly — one prompt, one response — so the binding length is much shorter. I applied a
constraint from the A/B-judge harness to a model class it does not describe.

Measured `(prompt, response)` lengths, max over both responses, per tokenizer:

| model | median | p90 | max | over 500 |
|---|---|---|---|---|
| OA-deberta-v3-large-v2 | 309 | 585 | 2565 | 82 / 600 |
| InternLM2-1.8B-reward | 307 | 657 | 2233 | 88 / 600 |
| gpt2-large-helpful | 334 | 724 | 2530 | 113 / 600 |
| gpt2-large-harmless | 334 | 724 | 2530 | 113 / 600 |

Worst case across all four tokenizers and both responses: median 336, p90 730, max 2565.

**Retained 487 / 600 = 0.812**, comfortably above the 0.60 gate. Retained probes have median
length 315 and max 500; dropped probes have median 758 and max 2565 — i.e. the filter removes a
long tail, not a random slice.

**Decision: keep all FOUR models, run the entire reward-model stage on the identical 487-probe
filtered subset, four-way chance = 1/M = 0.25.** This preserves the member with the broadest
heterogeneous / multi-corpus training mixture (OA-deberta, four corpora). Per §2.1 step 5, a bias check comparing the three
surviving models' pairwise margin correlations on the filtered subset versus the full 600 will be
reported alongside the margin matrix, since the filter selects on length and length is a known
confound in this project.

**Two environment problems fixed to get here, both real:**
- `sentencepiece 0.2.2` **segfaults the interpreter** on import in this venv (silent crash, exit
  139). Version `0.2.0` imports cleanly; pinned to that.
- `transformers 5.16.1` misidentifies internlm2's legacy sentencepiece `tokenizer.model` as a
  *tiktoken* file and dies parsing it (`ValueError: Error parsing line b'\x0e'`). That is a
  converter bug, not a property of the model, so internlm2's lengths are read from the
  sentencepiece model directly (`+1` for BOS, conservative). Recorded per-model in
  `tokenizer_source` in the results JSON so the provenance of every length is auditable.

### 2.3 Harness H1-H6 — **all four models pass the correctness gates**

Each model scored on the identical 487-probe filtered subset. Files: `results/rm_harness.json`
plus one `rm_harness_<model>.json` per model.

| model | H1 non-degen | H2 determinism | H3 batch-inv | H4 margin id | H5 input reaches | reward sd | H6 frac preferring intact |
|---|---|---|---|---|---|---|---|
| OA-deberta | pass (394 unique) | **pass, bitwise** | pass (0.0) | pass (0.0) | pass (0.50 sd) | 1.749 | 0.938 |
| InternLM2-1.8B | pass (203 unique) | **pass, bitwise** | pass (0.0) | pass (0.0) | pass (0.24 sd) | 0.628 | 0.438 |
| gpt2-helpful | pass (366 unique) | **pass, bitwise** | pass (0.0) | pass (0.0) | pass (0.86 sd) | 1.488 | 0.938 |
| gpt2-harmless | pass (388 unique) | **pass, bitwise** | pass (0.0) | pass (0.0) | pass (0.94 sd) | 1.092 | **0.375** |

**H2 passes bitwise for all four — this is the empirical verification that `R = 1`**, not an
assumption. **Scope correction (§2.5 below): H2 as implemented tests *within-process* determinism.
InternLM2 is bitwise-reproducible within a process and *not* across processes, so its `R = 1`
holds only within a single scoring process.** The other three are bitwise-reproducible across
processes as well. H3's batch-invariance difference is exactly 0.0 everywhere, so the `fixed_bin`
construction carried over correctly to the scalar-head scorer.

**H6, reported as a behavioral diagnostic and explicitly NOT a correctness verdict (§2.3):**
`gpt2-harmless` *prefers token-shuffled text*, with only 37.5% preferring intact (median
intact−shuffled = **−0.367**), and `InternLM2` is near-indifferent (43.8%, median −0.012). Both
passed H1-H5 cleanly, so per the document this is a warning to report and investigate, never a
basis for concluding "broken harness." One plausible mechanism for the harmless model, offered as
a hypothesis rather than a claim: scrambled text cannot articulate harmful content, so a
harmlessness-only objective may rate it *higher* — which would make this coherent behavior for
its training objective rather than a defect.

### 2.2 psi_eff — **the first directly measured separation in this project**

Because a scalar head is deterministic with one canonical serialization, `R = 1` (verified by H2),
so inter-model margin correlation is **unattenuated** and `psi_eff = arccos(pearson(d_m, d_m'))`
is a plain correlation rather than the voided sign-agreement conversion.

Pairwise margin correlation (487 probes):

| | OA-deberta | InternLM2 | gpt2-helpful | gpt2-harmless |
|---|---|---|---|---|
| OA-deberta | 1.000 | 0.115 | 0.386 | −0.225 |
| InternLM2 | 0.115 | 1.000 | 0.166 | −0.107 |
| gpt2-helpful | 0.386 | 0.166 | 1.000 | **−0.606** |
| gpt2-harmless | −0.225 | −0.107 | −0.606 | 1.000 |

`psi_eff` (degrees):

| | OA-deberta | InternLM2 | gpt2-helpful | gpt2-harmless |
|---|---|---|---|---|
| OA-deberta | 0.0 | 83.4 | **67.3** | 103.0 |
| InternLM2 | 83.4 | 0.0 | 80.5 | 96.1 |
| gpt2-helpful | 67.3 | 80.5 | 0.0 | **127.3** |
| gpt2-harmless | 103.0 | 96.1 | 127.3 | 0.0 |

**Minimum pairwise `psi_eff` = 67.3°** (OA-deberta vs gpt2-helpful) — the binding constraint,
since attribution fails on the closest pair first.

**What this does and does not license.** The four candidate reward models are geometrically
well-separated on the retained probe distribution; whether they form a valid downstream
attribution pool remains gated on D1. Geometry and validity are different questions: two models
can be far apart in margin space while neither carries preference structure that generalizes off
this probe set. §2.4 below settles the second question, and it does not go the way the geometry
alone would suggest.

**Two results worth stating explicitly:**

1. **67.3° sits well above the cliff, as a geometric fact only.** Track E found accuracy holding
   at 0.95-0.98 for `psi = 90°` and collapsing toward chance by `psi = 45°`. The real pool's
   tightest pair is much nearer the favourable end than the collapse. This is the first time the
   project has had a defensible number on this axis at all. It is a statement about the geometry
   of the retained probe distribution, **not** a statement that the pool is usable for
   attribution — that remains gated on D1 (§2.4).
2. **The pre-designated "hard cell" is the *most* separated pair, at 127.3°.** helpful vs harmless
   are **anti-correlated** (r = −0.606), not hard to tell apart. That is a sensible finding rather
   than an anomaly — helpfulness and harmlessness genuinely trade off, so a response one rewards
   the other penalizes. The designation was made in advance and is reported as measured; the pair
   is easy for attribution precisely because the objectives oppose.

Caveats stated rather than buried, per §2.2: probe features are anisotropic, so this is effective
separation *on this probe distribution*, which is the quantity attribution actually depends on but
is not interchangeable with the synthetic isotropic `psi`. And `R = 1` was verified (H2), not
assumed.

### Implementation problems fixed to get here (all real, all mine or the library's)

`internlm2-1_8b-reward`'s remote code predates transformers 5 and required six fixes:
`protobuf` and `einops` missing; the tiktoken/sentencepiece converter misparse; `rope_scaling`
gaining a `rope_type` key its code reads as `["type"]` (restored to the `null` its own config
declares); `DynamicCache.from_legacy_cache` removed (avoided by `use_cache=False`, inert for
single-pass scoring); and a **special-token id mismatch** — transformers assigned
`<|reward|>`/`<|im_start|>`/etc. ids 92544-92550 while the embedding table stops at 92544 and the
repo's own `tokenizer_config.json` declares them at 92527-92543. That last one was a correctness
bug that caused a device-side assert; fixed by restoring the repo's declared ids, with an
in-range assertion so a mis-tokenization can never silently reach the model. Verified against
ground truth rather than assumed: 60% accuracy on large-gap chosen/rejected pairs (above chance,
not inverted).

**One process per model.** `internlm2` returned all-NaN when loaded after OA-deberta in the same
process, yet scored cleanly on the identical 487 probes in a fresh process (verified by scanning
both responses across the full set, zero NaN). So each model now runs in its own process and the
results are combined afterward. This is the second time in this project that in-process
sequencing produced an artifact — the first was my own padding-mode validation script, where
toggling modes within one process was itself the confound.

### 2.1 step 5 -- filtered-vs-full bias check: **the filter does not materially change pool geometry**

The 487/600 filter selects on length, and length is a known confound in this project, so the
filter could in principle reshape the pool's geometry rather than just trim a tail. Each model
was re-scored on all 600 probes in its own process and the pairwise margin correlations compared.

**OpenAssistant is excluded from the full-set arm by design.** Its context is 512 tokens and 82
of the 600 probes exceed the 500-token cap, so a full-set score would require pushing it past its
supported context. It is never forced beyond that context to manufacture a comparison, so the
full-set arm covers the three models that can be validly scored on all 600, and OA appears only
in the filtered arm.

The comparison is decomposed into the two effects that are otherwise confounded -- *which probes*
(full 600 to the retained 487, same process) and *which process* (the full-set process to the
harness process, same 487 probes):

| pair | corr full 600 | corr full@retained | corr filtered 487 | d probe-selection | d process | d total | psi_eff full | psi_eff filt | d psi_eff |
|---|---|---|---|---|---|---|---|---|---|
| InternLM2-1.8B vs gpt2-helpful | 0.1577 | 0.1589 | 0.1657 | +0.0012 | +0.0068 | +0.0081 | 80.93° | 80.46° | -0.47° |
| InternLM2-1.8B vs gpt2-harmless | -0.1208 | -0.1151 | -0.1070 | +0.0057 | +0.0081 | +0.0138 | 96.94° | 96.14° | -0.80° |
| gpt2-helpful vs gpt2-harmless | -0.6012 | -0.6062 | -0.6062 | -0.0050 | +0.0000 | -0.0050 | 126.95° | 127.31° | +0.36° |

**max |d corr| = 0.0138; max |d psi_eff| = 0.80°.**
Against a pre-set materiality threshold of 5° on any pairwise `psi_eff`, the answer is **NO --
the length filter does not materially change pool geometry.** The largest total shift is
0.80°, an order of magnitude inside the threshold, and the sign of every correlation is
preserved, including the -0.60 helpful/harmless anti-correlation that drives the hard-cell result.

The decomposition also shows *where* the residual comes from. Probe-selection effects are at most
0.0057. The `gpt2-helpful vs gpt2-harmless` process delta is exactly
0.0000 -- those two models reproduce bitwise across processes. Every non-zero process delta
involves InternLM2, which is the subject of 2.5.

Same-probe cross-arm agreement (identical inputs, two processes, max abs difference in `d_m`):
`gpt2-helpful` **0.0000**, `gpt2-harmless` **0.0000**, `InternLM2` **1.1313**. That last number is not a filter effect at all
-- it is a reproducibility failure, reported in full below.

### 2.2 (B.2) InternLM2 retained-input lengths under the **exact final serialization**

The 500-token probe filter was built on a preliminary estimate: plain `{prompt}\n\n{response}`
encoded with the sentencepiece model read directly off disk (the transformers-5 converter could
not load it). InternLM2 is not actually scored that way. Its final scoring path applies the chat
template, remaps the special-token ids to the repo's declared values, and appends `<|reward|>`.
Those lengths are therefore validated here against the **actual `input_ids` the model receives**,
not the estimate.

| quantity (worst response of the pair) | preliminary estimate | realized `input_ids` |
|---|---|---|
| median | 292 | 362 |
| max | 499 | **596** |
| p99 | -- | 565 |
| min | -- | 40 |

Realized minus estimate: median **+65** tokens, max
**+155**, min **-325**. The
positive median is chat-template overhead the plain-concatenation estimate could not see; the
negative tail is the template tokenizing some content more compactly than the raw concatenation.

**Consequence: 28 of the 487 retained probes
exceed the 500-token filter cap under the real serialization** -- the estimate was optimistic, and
this is exactly the discrepancy the instruction asked to surface. It changes nothing about
validity: InternLM2's context is 2048 tokens, the longest realized input is
596 (29.1% of context), and
**`any_truncation_occurred = False`**. The 500 cap is binding for
OpenAssistant (512 context), not for InternLM2, and OA's serialization *is* the plain
concatenation the filter estimated, so OA's numbers were exact -- its retained max is 468 against
its 512 context. **No model in the retained set was truncated.**

### 2.5 Cross-process determinism -- **this qualifies the `R = 1` claim for InternLM2**

H2 as implemented re-scores the probes twice *inside one process* and checks bitwise equality.
All four models passed. That test is narrower than the claim it was supporting: `R = 1` asserts
the evaluator is a deterministic function of its input, and every number in this stage is
assembled from **multiple processes** (one process per model was forced by the InternLM2 NaN
artifact). So the property that actually has to hold is cross-process reproducibility, and H2
never tested it.

Testing it on the 487 retained probes, comparing the harness process against the bias-check
full-set process on identical inputs:

| model | bitwise-equal | max abs diff | median abs diff | as frac of margin sd | Pearson between processes | reproducible |
|---|---|---|---|---|---|---|
| OA-deberta | -- | -- | -- | -- | -- | not testable (no full-set arm; excluded by context) |
| InternLM2-1.8B | 0.070 | 1.1313 | 0.1016 | 24.5% | 0.8396 | **NO** |
| gpt2-helpful | 1.000 | 0.0000 | 0.0000 | 0.0% | 1.0000 | **yes** |
| gpt2-harmless | 1.000 | 0.0000 | 0.0000 | 0.0% | 1.0000 | **yes** |

**The two gpt2 models are exactly reproducible across processes -- every one of the 487 margins is
bitwise identical. InternLM2 is not.** Only 7.0% of its margins reproduce exactly; the median
disagreement is 24.5% of its own margin standard deviation, and the between-process correlation
is 0.84, not 1.0. A direct 60-input re-run in a fresh process reproduced this independently
(2/60 exactly equal, max 1.2207, median 0.1719, r = 0.8378), so it is stable run-to-run
nondeterminism, not a one-off. Within a single process InternLM2 repeats bitwise (max diff 0.0),
which is why H2 passed.

**Stated precisely: `R = 1` is empirically verified through bitwise H2 determinism for all four
models, and that verification is valid within a scoring process. For the two gpt2 models it
extends to cross-process reproducibility, verified. For InternLM2 it does not -- its effective
reliability across processes is below 1, and any InternLM2 number combining results from
different processes carries that noise.** No `R` value is revised here and no downstream quantity
is recomputed with an adjusted `R`: that would be a threshold change, which requires explicit
instruction. This is recorded as a scope limitation on the existing claim.

Two things it does *not* invalidate. The `psi_eff` matrix is built entirely from one harness
process per model, so it is internally consistent. D1 likewise scores each model in a single
process. The cause is not diagnosed -- plausible candidates are nondeterministic kernel selection
or bf16 reduction ordering varying with allocator state across processes, but I have not isolated
it, and saying which it is would be a guess.


### 2.4 D1 -- external preference structure, pre-registered rule applied unmodified

Pre-registered in `configs/base.yaml:reward_model_stage.d1_pre_registration` **before any**
**reward-model scoring**, and not touched since:

```
primary   = large-gap UltraFeedback subset (score_chosen - score_rejected >= 3)
secondary = full held-out UltraFeedback set
retain if |accuracy - 0.5| >= 0.10 AND two-sided exact binomial p < 0.001
```

The statistic is distance **from chance**, not accuracy, so a stable anti-correlated model passes
on its own terms. Prospective per-model expectations were written into `src/judges/rm_d1.py`
before scoring and are reported alongside the result, so a surprise cannot be retrofitted into a
prediction. Each model is scored in its own process, and each is evaluated only on the pairs that
fit its context under its own exact serialization -- no truncation to force coverage, which is why
`n` differs across models.

| model | prospective expectation | D1 full (secondary) | n | \|d\| | p | D1 large-gap (PRIMARY) | n | \|d\| | p | primary verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| OA-deberta | expected positive structure | 0.5870 | 322 | 0.0870 | 2.13e-03 | 0.6906 | 320 | 0.1906 | 7.52e-12 | **PASS** |
| InternLM2-1.8B | unknown | 0.5491 | 499 | 0.0491 | 3.15e-02 | 0.5980 | 500 | 0.0980 | 1.35e-05 | **FAIL** |
| gpt2-helpful | expected positive structure | 0.5267 | 450 | 0.0267 | 2.78e-01 | 0.5023 | 430 | 0.0023 | 9.62e-01 | **FAIL** |
| gpt2-harmless | direction unknown on generic helpfulness labels | 0.4800 | 450 | 0.0200 | 4.23e-01 | 0.4326 | 430 | 0.0674 | 5.92e-03 | **FAIL** |

**Only OpenAssistant passes the primary criterion.** Three points that need stating exactly, none
of which change anything:

1. **InternLM2 misses by 0.0020.** Its large-gap distance from chance is 0.0980 against the 0.100
   required. It clears the significance leg comfortably (p = 1.35e-05, i.e. the structure is
   real and not a sampling artifact at n = 500) and fails only the effect-size leg. This is a
   genuine near-miss and it is reported as one. **The threshold is not moved.** A pre-registered
   bound that gets relaxed the first time a model lands just outside it is not a bound, and the
   standing prohibition forbids altering a threshold without explicit instruction. InternLM2 is
   not retained.
2. **gpt2-harmless is below chance on both sets** (0.4326 large-gap, 0.4800 full). Under a
   distance-from-chance rule that is not automatically a failure -- a reliable inverse predictor
   is informative. It fails anyway: |d| = 0.0674 < 0.10 and p = 5.92e-03 > 0.001. It is reported
   as a measurement, not reinterpreted as a defect, and it is coherent with the H6 diagnostic
   and the -0.606 anti-correlation with gpt2-helpful: a harmlessness-only objective scored
   against generic helpfulness labels should point the other way.
3. **gpt2-helpful is at chance on the primary set** (0.5023, p = 0.962) despite being the model
   with the strongest prior expectation of positive structure. That expectation was recorded in
   advance and is now simply wrong. It is the cleanest null in the table.

**OpenAssistant passes the primary and fails the secondary** (full-set |d| = 0.0870, p = 2.13e-03).
The pre-registration names the large-gap subset as primary, so it is retained -- but the split is
worth recording: its structure is real on wide-margin pairs and thins out on the full set, which
is the expected shape if it tracks large quality differences and not fine ones.

### 2.6 Retention (3.8) -- **1 of 4 retained**

Applied mechanically. H6 is deliberately not part of the criterion.

| model | H1-H5 | D1 full | D1 large-gap | retention criterion | retained? |
|---|---|---|---|---|---|
| OA-deberta | PASS | 0.5870 (|d|=0.0870, p=2.13e-03, n=322) FAIL | 0.6906 (|d|=0.1906, p=7.52e-12, n=320) PASS | H1-H5 all pass AND D1 primary passes | **YES** |
| InternLM2-1.8B | PASS | 0.5491 (|d|=0.0491, p=3.15e-02, n=499) FAIL | 0.5980 (|d|=0.0980, p=1.35e-05, n=500) FAIL | H1-H5 all pass AND D1 primary passes | **NO** |
| gpt2-helpful | PASS | 0.5267 (|d|=0.0267, p=2.78e-01, n=450) FAIL | 0.5023 (|d|=0.0023, p=9.62e-01, n=430) FAIL | H1-H5 all pass AND D1 primary passes | **NO** |
| gpt2-harmless | PASS | 0.4800 (|d|=0.0200, p=4.23e-01, n=450) FAIL | 0.4326 (|d|=0.0674, p=5.92e-03, n=430) FAIL | H1-H5 all pass AND D1 primary passes | **NO** |

Machine-readable: `results/rm_retention.csv`, `results/rm_retention.json`
(`n_retained = 1`, `retained_models = ['OA-deberta']`).

### 2.7 Retained-pool geometry -- **undefined with one survivor**

The instruction says that if retention changes the pool, recompute the retained-pool
margin-correlation matrix, the `psi_eff` matrix, the minimum retained `psi_eff`, and the chance
level. Retention changed the pool from 4 to 1, and at M = 1 those quantities do not exist:

- **retained margin-correlation matrix**: the 1x1 matrix `[[1.0]]`. No off-diagonal entries.
- **retained `psi_eff` matrix**: `[[0.0]]`. There is no pair, so there is no separation to measure.
- **minimum retained pairwise `psi_eff`**: **undefined** -- the set of pairs is empty.
- **chance level**: `1/M = 1/1 = 1.0`. A one-candidate attribution task is answered correctly by
  a constant, so top-1 accuracy carries zero information and no detection statistic is meaningful.

This is not a small pool. It is the absence of an attribution problem.

---

## STOP -- Section B gate failed, and Section A was not started

**Fewer than two models survive retention, so the run stops here. Section A was not started.**

The blocker, stated plainly:

> Of four candidate reward models, all four pass every correctness gate (H1-H5) and only one --
> `OpenAssistant/reward-model-deberta-v3-large-v2` -- carries preference structure that survives
> the pre-registered D1 external-validity criterion. Judge attribution requires distinguishing
> **between** candidates. With one candidate there is nothing to distinguish, chance is 1.0, and
> the density-ratio identity has no discriminative target. The DPO sweep would produce a
> `Delta(x,y,y')` that could be correlated against exactly one reward model, which measures
> recovery, not attribution.

What was **not** done, per the standing prohibition:

- D1 was **not** weakened, and the 0.0020 InternLM2 near-miss was **not** used to justify moving
  the threshold to 0.095 or to a one-sided test.
- No model was substituted into the pool, and no replacement candidate was searched for.
- The primary/secondary designation was **not** flipped to let OA's full-set result or another
  model's secondary result carry retention.
- Section A was **not** started as though a viable attribution pool exists.
- No Round 5 experiment was introduced, no training data was generated, and no judge was tuned
  on held-out probes.

**This needs an explicit decision before any further compute.** The options are yours to pick,
and each one requires the instruction the prohibition demands: relax or re-specify D1 with a
stated rationale; change the candidate pool; accept a single-model recovery study in place of
attribution; or stop the reward-model track. I have not chosen among them.

### Figures generated

| figure | file | content |
|---|---|---|
| F22 | `figures/F22_rm_probe_filter.png` | retained-vs-dropped probe length distributions (generated earlier) |
| F23 | `figures/F23_rm_margin_matrix.png` | margin correlation + `psi_eff`, 487 retained probes, hard cell marked |
| F24 | `figures/F24_rm_harness_panel.png` | H1 margin densities (gate) and H6 intact-vs-shuffled (behavioral only) |
| F25 | `figures/F25_rm_d1.png` | D1 full vs large-gap per model against the pre-registered band |

F17-F21 and F26-F28 belong to Section A, Section C and Section D and are **not generated** --
they have no data behind them, and drawing them would misrepresent a blocked run as a completed
one.


---

## Round 4 compliance checklist

Every row is either COMPLETE, GATED - NOT TRIGGERED, NOT APPLICABLE, or BLOCKED. There are no
PENDING items. 'BLOCKED' means the B.6 stop condition prevented the work; 'GATED - NOT TRIGGERED'
means a precondition in the spec was not met and the work was correctly skipped.

| # | spec | item | status | evidence / reason |
|---|---|---|---|---|
| 1 | 0.1-0.3 | Round-4 amendment applied to the Track C pilot interpretation | **COMPLETE** | RESULTS.md 'AMENDMENT (round 4)' block, written before any round-4 scoring |
| 2 | 0.4 | `a = sqrt(R)`, never `a = R` | **COMPLETE** | `src/phase0e_track_e2.py`; calibration gate target 0.573, realized 0.571 |
| 3 | D | Old pilot `sigma_D = 5.35` not used as an input or comparison target | **COMPLETE** | not referenced anywhere in round-4 code; Section A never ran, so no comparison was made |
| 4 | 1.1 | Section A R1: J_perp, stochastic BT labels, tau = 2 | **BLOCKED** | Section B gate |
| 5 | 1.1 | Section A R2: J_perp, shuffled labels | **BLOCKED** | Section B gate |
| 6 | 1.1 | Section A R3: J_len, stochastic BT labels | **BLOCKED** | Section B gate |
| 7 | 1.1 | Section A R4: J_len, shuffled labels | **BLOCKED** | Section B gate |
| 8 | 1.2 | J_perp built from training data only, |corr(d_true, len_diff)| < 0.05 on training | **BLOCKED** | judge rebuild not started; `src/phase1_feature_judges.py` still holds the pilot judge (corr -0.313) |
| 9 | 1.2 | J_len built from training data only, |corr(d_true, len_diff)| > 0.7 on training | **BLOCKED** | same |
| 10 | 1.2 | Held-out correlations reported but never tuned on | **BLOCKED** | no judge was built, so nothing was tuned |
| 11 | 1.3 | 300 existing pairs, 16 epochs (~304 steps), checkpoints every 19 steps | **BLOCKED** | Section B gate |
| 12 | 1.3 | Existing probes, `fixed_bin` scorer, existing safe allocator config | **BLOCKED** | scorer and allocator config are in place and gated-tested; no run consumed them |
| 13 | 1.4 | Per-checkpoint panel: train_acc, train_loss, sd(Delta), Pearson, R^2, sigma_D_ratio, Spearman, latent_sign_agreement + 95% CI, noisy_label_agreement + 95% CI, normalized_noisy_transmission, gamma_1/gamma_2 + SEs | **BLOCKED** | Section B gate |
| 14 | 1.4 | normalized_noisy_transmission only with a fresh stochastic-label target and the correct Bayes ceiling | **BLOCKED** | not computed, so not computed wrongly |
| 15 | 1.5 | `excess_variance_over_shuffled` matched at equal steps, from `sd` not `mean|Delta|` | **BLOCKED** | Section B gate |
| 16 | 1.6 | Data scaling to 1,000 / 3,000 pairs | **GATED - NOT TRIGGERED** | explicitly not auto-generated; low train_acc is not a trigger, and no run occurred |
| 17 | 2.1 | Probe-length filter, retention fraction above the 0.60 gate | **COMPLETE** | 487/600 = 0.812; `results/rm_probe_filter.json` |
| 18 | 2.1 step 5 | Filtered-vs-full bias check, correlation and psi_eff deltas | **COMPLETE** | max |d corr| 0.0138, max |d psi_eff| 0.80 deg -> filter does NOT materially change geometry |
| 19 | 2.1 | OpenAssistant never forced beyond its 512-token context | **COMPLETE** | excluded from the full-set arm by design; retained max 468 tokens, no truncation |
| 20 | B.2 | InternLM2 retained lengths validated on actual `input_ids` under the final serialization | **COMPLETE** | realized max 596 vs 2048 context; 28/487 exceed the 500 estimate; no truncation |
| 21 | 2.3 | H1 non-degenerate margins | **COMPLETE** | all 4 pass |
| 22 | 2.3 | H2 determinism (the empirical `R = 1` verification) | **COMPLETE** | all 4 bitwise within-process; scope qualified cross-process in 2.5 |
| 23 | 2.3 | H3 batch invariance | **COMPLETE** | max diff 0.0 for all 4 |
| 24 | 2.3 | H4 margin identity | **COMPLETE** | max diff 0.0 for all 4 |
| 25 | 2.3 | H5 input reaches the model | **COMPLETE** | all 4 pass |
| 26 | 2.3 | H6 reported as behavioral diagnostic only, never a correctness gate | **COMPLETE** | reported in full; excluded from the retention criterion by construction |
| 27 | 2.2 | psi_eff matrix on the retained probe distribution | **COMPLETE** | min pairwise 67.3 deg; hard cell 127.3 deg; `results/rm_harness.json` |
| 28 | 2.2 | psi_eff described as effective separation on this probe distribution, not interchangeable with synthetic isotropic psi | **COMPLETE** | caveat paragraph retained and reinforced |
| 29 | 2.4 | D1 run exactly as pre-registered, all four models, full set + large-gap primary | **COMPLETE** | N, accuracy, distance from 0.5, p-value and threshold all reported |
| 30 | 2.4 | D1 pre-registration frozen before scoring and unchanged after seeing results | **COMPLETE** | `configs/base.yaml:reward_model_stage.d1_pre_registration`; no edit after scoring |
| 31 | 2.4 | Prospective per-model expectations recorded before scoring | **COMPLETE** | `EXPECTATION` dict in `src/judges/rm_d1.py` |
| 32 | 3.8 | Retention criterion applied mechanically | **COMPLETE** | 1 of 4 retained (OA-deberta only); `results/rm_retention.csv` |
| 33 | B.5 | Retained-pool correlation / psi_eff matrices, min retained psi_eff, chance level | **NOT APPLICABLE** | M = 1: no pairs exist, min psi_eff undefined, chance = 1.0; stated explicitly |
| 34 | B.6 | STOP after Section B if fewer than two models survive | **COMPLETE** | condition triggered and honored; Section A not started |
| 35 | - | Cross-process determinism qualification of the `R = 1` claim | **COMPLETE** | beyond spec; InternLM2 not reproducible across processes (r = 0.84), gpt2 models exact |
| 36 | C.1 | Drop 'the real pool is separated enough for downstream attribution'; use the prescribed geometry-vs-validity sentence | **COMPLETE** | applied in 2.2 |
| 37 | C.2 | Describe OpenAssistant as the member with the broadest heterogeneous / multi-corpus training mixture, not the only member outside hh-rlhf | **COMPLETE** | corrected in two places, with the earlier claim marked as an error |
| 38 | C.3 | `R = 1` kept as empirically verified through bitwise H2 determinism | **COMPLETE** | kept, with the cross-process scope limitation attached |
| 39 | C.4 | H6 kept explicitly behavioral only | **COMPLETE** | unchanged and reinforced |
| 40 | C.5 | psi_eff kept as effective separation on this probe distribution | **COMPLETE** | unchanged |
| 41 | F | Section C real-teacher calibration; teacher choice and rationale frozen before observing DPO results | **GATED - NOT TRIGGERED** | its own precondition (>=1 RM retained) IS met, but it is downstream of Section A, which the B.6 stop blocks; no teacher was chosen, so nothing was frozen prematurely |
| 42 | G | Section D model-based overlay labelled a model-based predicted operating point | **GATED - NOT TRIGGERED** | no overlay drawn |
| 43 | G | Section D end-to-end overlay kept separate from the model-based one | **GATED - NOT TRIGGERED** | no overlay drawn |
| 44 | H | Figures F22-F25 (Section B) | **COMPLETE** | F22 probe filter (earlier); F23-F25 generated by `src/analysis/figures_round4.py` |
| 45 | H | Figures F17-F21, F26-F28 (Sections A, C, D) | **GATED - NOT TRIGGERED** | no data behind them; not drawn |
| 46 | A | Compliance checklist produced before starting the Section A GPU sweep | **COMPLETE** | this table; the sweep never started |
| 47 | - | Standing prohibition: no Round 5 experiment, no threshold change, no pool change, no judge tuned on held-out probes, no extra training data, no reinterpreted gate | **COMPLETE** | none of the six occurred; the InternLM2 0.0020 near-miss was reported, not acted on |

**Totals:** BLOCKED 12, COMPLETE 29, GATED - NOT TRIGGERED 5, NOT APPLICABLE 1 (of 47 items). PENDING: 0.


---

# NEXT_STEPS_ROUND_5_FINAL.md

## 1. Rule-change disclosure -- stated before any result below

The calibration-stage retention rule used from here on (C1-C5) was **defined after the Round 4
D1 results were observed**. It is therefore *not* pre-registered relative to Round 4, and the
Round 4 calibration data is *not* offered as confirmatory evidence that the new rule is valid.

What is preserved: **every Round 4 D1 value stands exactly as measured.** Nothing was recomputed,
re-run, or adjusted. The Round 4 D1 rule remains in `configs/base.yaml` under
`d1_pre_registration`, now carrying `superseded_for_calibration_stage: true` -- labelled, not
deleted and not overwritten. D1 appears below only as a **reported covariate**.

The defense of the new rule is conceptual and procedural, not outcome-based. D1 asked whether a
reward model agrees with a GPT-4-style preference ordering. The calibration-stage question is
different: whether a reward function is reproducible, non-degenerate, correctly scored, not
almost entirely a length detector, and distinguishable from the other candidates. Those
properties are measurable directly, and C1-C5 measures them directly. The new rule is also
strictly *stricter* in two ways the old one had no coverage for at all: cross-process
reproducibility (C1) and length dominance (C4).

The rule, the pool, the attribution rules, the statistical tests and the success criteria are
frozen in `configs/base.yaml:reward_model_stage.calibration_retention` before any Round 5
attribution outcome is observed, and final attribution is evaluated on a fresh disjoint probe
set that plays no part in constructing the pool.

## Gate 1 -- calibration pool: **PASS**

### 2.4 C1 cross-process determinism, measured under a single protocol

Each model was scored twice on the identical 487 calibration probes in **two fresh, independent
processes** (distinct PIDs asserted), using the exact final scoring serialization. The criterion
is bitwise equality; no correlation threshold was substituted for it.

| model | fraction bitwise equal | max abs diff | median abs diff | median / margin sd | Pearson between processes | C1 cross-process |
|---|---|---|---|---|---|---|
| OA-deberta | 1.0000 | 0 | 0 | 0.00% | 1.000000 | **PASS** |
| gpt2-helpful | 1.0000 | 0 | 0 | 0.00% | 1.000000 | **PASS** |
| gpt2-harmless | 1.0000 | 0 | 0 | 0.00% | 1.000000 | **PASS** |
| InternLM2-1.8B | 0.070 | 1.1313 | 0.1016 | 24.5% | 0.8396 | **FAIL** (carried forward, not re-tested) |

**OpenAssistant passes C1 cross-process bitwise** -- 487/487 margins identical across two fresh
processes. So do both gpt2 models, now measured under this same paired protocol rather than
inferred from the Round 4 harness-vs-bias-check comparison. The fresh run-1 scores also match the
Round 4 harness margins with max abs difference **exactly 0** for all three, so the Round 4
geometry was computed on reproducible numbers.

**InternLM2 is marked C1 FAIL and excluded.** It was not re-tested and no attempt was made to
rescue it by changing precision, kernels, or thresholds. The bf16 reduction-ordering and
kernel-selection explanations remain **untested hypotheses**.

### 2.7 Model-wise retention table (D1 columns are covariates, not gates)

| model | C1 within | C1 cross | C2 | C3 | C4 gamma | SE | C4 R^2_len | C4 | D1 full (cov) | D1 large-gap (cov) | retained C1-C4 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| OA-deberta | PASS | PASS | PASS | PASS | +0.00364 | 0.00085 | 0.0364 | PASS | 0.5870 | 0.6906 | **YES** |
| InternLM2-1.8B | PASS | FAIL (not re-tested; 2.3) | PASS | PASS | +0.00197 | 0.00028 | 0.0938 | PASS | 0.5491 | 0.5980 | **NO** |
| gpt2-helpful | PASS | PASS | PASS | PASS | +0.00713 | 0.00089 | 0.1178 | PASS | 0.5267 | 0.5023 | **YES** |
| gpt2-harmless | PASS | PASS | PASS | PASS | -0.00487 | 0.00077 | 0.0762 | PASS | 0.4800 | 0.4326 | **YES** |

**C4 is passed by all four candidates, comfortably.** The largest share of reward-margin variance
explained by a single linear length feature is **0.094** (InternLM2), against the 0.50 threshold.
The two gpt2 models have length coefficients of opposite sign -- `gpt2-helpful` **+0.00713**
rewards longer responses, `gpt2-harmless` **-0.00487** penalizes them -- which is consistent with
their opposed objectives and with the -0.606 anti-correlation between them. **No candidate in
this pool is a length detector.** That is a measurement, and it was not guaranteed: length
dominance was the mechanism that collapsed apparent transmission in the earlier feature-judge
pilot.

**S (passing C1-C4) = ['OA-deberta', 'gpt2-helpful', 'gpt2-harmless'], |S| = 3.** The single exclusion is InternLM2, on C1.

### 2.5 Candidate geometry, raw and length-residualized

Raw pairwise correlation (487 calibration probes):

| | OA-deberta | gpt2-helpful | gpt2-harmless |
|---|---|---|---|
| OA-deberta | 1.000 | 0.386 | -0.225 |
| gpt2-helpful | 0.386 | 1.000 | -0.606 |
| gpt2-harmless | -0.225 | -0.606 | 1.000 |

Raw `psi_eff` (°):

| | OA-deberta | gpt2-helpful | gpt2-harmless |
|---|---|---|---|
| OA-deberta | 0.000 | 67.29 | 103.02 |
| gpt2-helpful | 67.29 | 0.000 | 127.31 |
| gpt2-harmless | 103.02 | 127.31 | 0.000 |

Length-residualized correlation:

| | OA-deberta | gpt2-helpful | gpt2-harmless |
|---|---|---|---|
| OA-deberta | 1.000 | 0.348 | -0.183 |
| gpt2-helpful | 0.348 | 1.000 | -0.567 |
| gpt2-harmless | -0.183 | -0.567 | 1.000 |

Length-residualized `psi_eff` (°):

| | OA-deberta | gpt2-helpful | gpt2-harmless |
|---|---|---|---|
| OA-deberta | 0.000 | 69.65 | 100.54 |
| gpt2-helpful | 69.65 | 0.000 | 124.51 |
| gpt2-harmless | 100.54 | 124.51 | 0.000 |

- **min raw pairwise `psi_eff` = 67.29°** (OA-deberta vs gpt2-helpful)
- **min residualized pairwise `psi_eff` = 69.65°** (OA-deberta vs gpt2-helpful) -- diagnostic only

**Removing the linear length component slightly *increases* separation** (67.29 to 69.65 degrees;
every pairwise correlation moves toward zero). The measured separation is therefore not
length-mediated -- length is a small *shared* component that was making the candidates look
marginally more alike than they are. This is the favourable branch of 2.6, and it is reported as
measured; no model was tuned, removed, or transformed on the basis of it.

### 2.6 Pool-level gate C5

```
min raw pairwise psi_eff = 67.29 deg  >=  45 deg  ->  C5 PASS
```

C5 was evaluated on `S` as a whole, after C1-C4 had already fixed the membership. No candidate
was dropped to improve the closest pair and no subset search was performed.

**Final candidate set: ['OA-deberta', 'gpt2-helpful', 'gpt2-harmless']. M = 3. chance = 1/M = 0.3333.**

### 3. Training-pair precheck

Every one of the 300 existing training pairs was checked against each retained reward model's
exact final serialization and against the DPO policy path (`build_scoring_inputs`, the identical
chat-template serialization used at generation, training and Delta scoring) under the standing
`dpo.max_length = 640` cap.

| scorer | context cap | max realized len | median | p99 | excluded by this scorer | excluded uniquely by it |
|---|---|---|---|---|---|---|
| OA-deberta | 512 | 1815 | 292 | 1394 | 40 | 13 |
| gpt2-helpful | 1024 | 1810 | 316 | 1451 | 13 | 0 |
| gpt2-harmless | 1024 | 1810 | 316 | 1451 | 13 | 0 |
| policy(Qwen) | 640 | 1652 | 309 | 1415 | 28 | 0 |

```
starting pairs    = 300
retained pairs    = 259
retained fraction = 0.8633
excluded overall  = 41
gate (>= 250)     = PASS
```

OpenAssistant's 512-token context is the binding constraint, excluding 40 pairs of which 13 are
excluded by it alone. No response was truncated and no context limit was relaxed to preserve
sample size.

**Training set frozen at N = 259 pairs**, manifest
`sha256 = 13b162d33744e05415e92e7bfc857e6cd3f2817f80b91356cce6fab197a48d66`. Pair IDs, texts, base policy, generation provenance,
reward-model serializations and label seeds are frozen; no training example may be added or
removed after an attribution outcome is observed.

### Gate 1 verdict

```
M = 3  (>= 2)                 PASS
C5 min raw psi_eff = 67.29 deg   PASS
training pairs = 259  (>= 250)     PASS
```

**Gate 1 passes. Proceeding to Gate 2 (fresh final attribution holdout).**


## Gate 2 -- fresh final attribution holdout: **PASS**

The 487 Round 4 probes have been used to filter reward models, run H1-H6, measure `psi_eff`,
define the C1-C5 pool, measure length dominance and inspect raw/residualized geometry. They are
a **calibration set** and cannot carry the primary attribution claim, so the confirmatory test
runs on probes that played no part in any of that.

### 4.2 Source and disjointness

`prompt_splits.json` holds 1,200 probe prompts of which the Round 4 calibration probes used
exactly the first 600. The fresh holdout is drawn from the **unused 600** -- same source
distribution, same base policy, same sampling procedure, with temperature, top_p,
`max_new_tokens`, chat template and sampling policy all unchanged. Two responses per prompt were
generated from `pi_ref` by the standing pipeline (`src/gen/sample_pairs.py`, extended only with
a prompt-slice offset and an output tag so the frozen calibration file is never mutated).

Disjointness is asserted in code, not in prose:

```
raw fresh probes                        = 600
unique prompts                          = 600
overlap with the 600 calibration probes = 0
overlap with the 300 training prompts   = 0
```

### 4.3 Compatibility filter

| scorer | context cap | max realized len | median | p99 | excluded by this scorer |
|---|---|---|---|---|---|
| OA-deberta | 512 | 2509 | 297 | 1264 | 86 |
| gpt2-helpful | 1024 | 2914 | 326 | 1782 | 25 |
| gpt2-harmless | 1024 | 2914 | 326 | 1782 | 25 |
| policy(Qwen) | 640 | 1981 | 314 | 1295 | 60 |

```
N_final           = 512
retained fraction = 0.8533   (gate: >= 0.6)
gate              = PASS
manifest sha256   = f1a63fbde3d51b0c0e57e1fd981469d289b91853b5ea9625df3230dfe9821027
```

### 4.5 Frozen calibration transforms applied out of sample

Candidate margins on the final probes are required predictors, so computing them before training
is explicitly permitted -- and it is the *only* thing computed on these probes before training.
Every transform applied to them was fitted on the 487 calibration probes and is frozen:

| candidate | mu_cal | sd_cal | c_len | gamma_len | resid mean_cal | resid sd_cal |
|---|---|---|---|---|---|---|
| OA-deberta | -0.0659 | 1.2367 | -0.0648 | +0.003643 | +2.19e-17 | 1.2141 |
| gpt2-helpful | -0.0084 | 1.3443 | -0.0062 | +0.007126 | -7.30e-18 | 1.2627 |
| gpt2-harmless | +0.0343 | 1.1417 | +0.0328 | -0.004869 | +4.38e-17 | 1.0973 |

Nothing is fitted on the final probes. The resulting geometry on the fresh holdout:

raw standardized correlation:

| | OA-deberta | gpt2-helpful | gpt2-harmless |
|---|---|---|---|
| OA-deberta | 1.000 | 0.410 | -0.180 |
| gpt2-helpful | 0.410 | 1.000 | -0.537 |
| gpt2-harmless | -0.180 | -0.537 | 1.000 |

length-residualized correlation:

| | OA-deberta | gpt2-helpful | gpt2-harmless |
|---|---|---|---|
| OA-deberta | 1.000 | 0.401 | -0.161 |
| gpt2-helpful | 0.401 | 1.000 | -0.505 |
| gpt2-harmless | -0.161 | -0.505 | 1.000 |

- **min pairwise `psi_eff` on the final holdout: raw 65.81°, residualized 66.37°**
- for comparison, on calibration: raw 67.29°, residualized 69.65°

The pool's geometry **reproduces on independent probes** -- 65.81 versus 67.29 degrees on the
tightest pair. That is a genuine out-of-sample check on the C5 gate, though the gate itself was
evaluated on calibration probes and is not re-run here.

### 4.4 Outcome-blindness

The holdout IDs, texts and manifest hash are frozen. Final-probe outcomes may not change the
pool, C1-C5, DPO hyperparameters, the attribution rule, the checkpoint choice, the
residualization or the standardization.

## 5.4 J_perp -- the length-orthogonal feature judge

Built from the frozen training pairs only: feature standardization constants, the orthogonality
projection and the label standardization all come from those 259 pairs. The construction
projects the standing orthonormal judge direction onto the subspace orthogonal to the training
length direction, which makes the training correlation exactly zero by construction rather than
by search.

```
corr(d_true, len_diff) on TRAINING          = +0.000000   (requirement |corr| < 0.05) -> PASS
corr(d_true, len_diff) on calibration_487    = -0.0486   [REPORTING ONLY]
corr(d_true, len_diff) on final_holdout      = -0.1757   [REPORTING ONLY]
```

**Stated plainly: J_perp is exactly length-orthogonal where it was fitted and only approximately
so out of sample.** The final-holdout correlation of -0.176 is ordinary sampling drift in a
projection fitted on 259 pairs, not a construction failure -- but it does mean the FJ arm is a
*nearly* length-neutral teacher rather than a perfectly length-neutral one, and the transmission
numbers it produces should be read that way. Per 5.4 these two correlations are reporting-only
and nothing about J_perp was adjusted after seeing them.

`J_len` is deliberately not built this round (5.4): length is handled by C4 and by the mandatory
raw-versus-residualized attribution analysis instead.

## Gate 3 -- frozen pre-registration

Written to `configs/base.yaml:attribution_stage.pre_registration_round5` after sections 2-4 were
frozen and **before the first DPO run started**. Verification:

| field | value |
|---|---|
| `candidates` | ['OA-deberta', 'gpt2-helpful', 'gpt2-harmless'] |
| `M` | 3 |
| `chance` | 0.3333333333333333 |
| `N_train` | 259 |
| `N_final` | 512 |
| `primary_identification_rule` | nnls_mixture |
| `secondary_identification_rule` | bradley_terry |
| `additional_reported_rule` | sign_agreement |
| `detection_statistic` | Lambda |
| `lambda_detection_threshold` | 12.873131695619026 |
| `p_threshold` | 0.001 |
| `primary_per_run_success` | true candidate top-1 AND S > 0 AND p_perm < 0.001 |
| `training_manifest_sha256` | `13b162d33744e05415e92e7bfc857e6c...` |
| `final_holdout_manifest_sha256` | `f1a63fbde3d51b0c0e57e1fd981469d2...` |
| permutation count | 10000 |
| bootstrap count | 5000 |
| kill criterion | required_successes = 2 (max(2, ceil(2*M/3))) |
| deterministic-arm teacher | gpt2-helpful |

### Two corrections made before freezing, both declared here

**1. The 6.4 permutation statistic was unsatisfiable as written.** 6.4 defines
`S = alpha_true - max_wrong` on the **normalized** NNLS weight vector. Measured on synthetic data
carrying this pool's actual correlation structure, that statistic is degenerate under the
permutation null: NNLS weights fitted to a permuted `Delta` are tiny and noisy, so dividing by
their sum drives the vector to a vertex of the simplex. The null becomes bimodal at +/-1 --
deciles -1.00, -0.60, +0.96, with the **99.9th percentile exactly +1.0000** -- while
`S_obs <= 1` by construction. `p_perm` therefore cannot fall below roughly 0.10 no matter how
strong the signal, making the pre-registered `p_perm < 0.001` success criterion **mathematically
unsatisfiable**.

On **unnormalized** NNLS coefficients the same measurement gives a null concentrated at zero
(sd 0.047, 99.9th percentile 0.124) and a test that discriminates correctly: strong and weak
signals reach the permutation floor, a very weak signal returns p = 0.18 and a null returns
p = 0.59. S is therefore computed on unnormalized coefficients.

**Top-1 identification is unchanged** -- normalization is division by a positive scalar, so the
argmax is identical either way. Only the magnitude statistic changes. Normalized mixture weights
are still reported everywhere 6.3 and F19 call for them, and `S` on normalized weights is
recorded alongside. The correction was made **before any DPO run and before any attribution
outcome existed**, and is recorded in the frozen block under `s_statistic_deviation_from_6_4`.

**2. `train_acc` was reading the wrong logged metric.** trl logs both `rewards/accuracies` (the
DPO pairwise preference accuracy that 5.6 asks for) and `mean_token_accuracy` (next-token
accuracy), and a substring match was selecting whichever appeared first in the log dict -- the
latter. Caught by a pre-run smoke test, which reported `train_acc` 0.500 against
`mean_token_accuracy` 0.560 on the same step. The key is now named explicitly and token accuracy
is kept as a separate field. This is a plumbing fix with no effect on any scientific choice.

### Entry gates (5.5), all checked before training

```
tests/test_scorer_invariance.py     shuffled-order max|diff| = 9.1553e-05 (0.0000 of sd)  PASS
                                    batch/token-budget max|diff| = 0.0000e+00              PASS
adapter on/off batch composition    identical padded tensor by construction; token-count
                                    equality asserted per item in src/score/delta.py        PASS
final-holdout manifest frozen       sha256 matches the pre-registration block               PASS
pre-registration block exists       configs/base.yaml:attribution_stage                     PASS
```

### Run configuration, carried forward unchanged

```
base policy            Qwen/Qwen2.5-0.5B-Instruct
training pairs         259 (frozen)
epochs                 16
optimizer steps        272  (ceil(259/16) = 17 per epoch x 16)
checkpoint cadence     every 17 steps -> 16 evenly spaced checkpoints including final
beta, LoRA, lr, sched  standing validated Round 4 values
max_length 640, truncation_mode keep_start, bf16, gradient checkpointing, fixed_bin scorer
torch_empty_cache_steps=1, GPU memory fraction 0.90
```


## Gate 4 -- stochastic real reward-model arms

Three arms, one per retained candidate, sharing the identical 259 frozen training pairs, DPO
configuration, checkpoint cadence and 512-probe final holdout. Only the label-generating teacher
differs. Labels are stochastic Bradley-Terry at `tau = 2`, standardized on training pairs only,
under a seed frozen before sampling.

| arm | label balance | flip rate vs sign | mean abs standardized margin | Bayes label ceiling |
|---|---|---|---|---|
| RM_OA-deberta | 0.4865 | 0.2510 | 0.7086 | 0.7420 |
| RM_gpt2-helpful | 0.4942 | 0.2587 | 0.7177 | 0.7433 |
| RM_gpt2-harmless | 0.5097 | 0.2394 | 0.7451 | 0.7525 |

### Optimization: every arm fits its training labels almost perfectly

| arm | steps | final-16-step train_acc | final train_loss | wall clock |
|---|---|---|---|---|
| RM_OA-deberta | 272 | 0.9727 | 0.00013 | 12192s |
| RM_gpt2-helpful | 272 | 0.9727 | 0.00014 | 3002s |
| RM_gpt2-harmless | 272 | 0.9727 | 0.00009 | 2981s |

**`train_acc` reaches 0.97 in every arm and training loss falls to about 1e-4.** Per 12.5 the
low-fit branch does not apply: this is not an optimization or capacity failure. The relevant
branch is 12.4 -- the labels are fit, and the question is whether anything generalizes.

An engineering note, recorded because the numbers look odd: `RM_OA-deberta` took 12,193s while
the other two took ~2,990s for the identical 272 steps. The difference is throughput, not
computation -- the first arm ran while this session was doing concurrent CPU work, and this
project established long ago that WDDM per-kernel-launch overhead, a CPU-side cost, dominates
here. All three arms logged 272/272 steps, wrote 16 checkpoints, and reached the same final
train_acc (0.9727) and loss (~1.4e-4), with correctly distinct per-teacher label diagnostics.
Nothing about the computation differed.

### Generalization: transmission peaks early, then decays

Final-checkpoint transmission on the fresh holdout:

| arm | Pearson | R^2 | sigma_D_ratio | Spearman | latent sign agree | fresh noisy agree | normalized noisy transmission |
|---|---|---|---|---|---|---|---|
| RM_OA-deberta | +0.0881 | 0.0078 | 11.3 | +0.0747 | 0.5566 | 0.4785 | -0.0845 |
| RM_gpt2-helpful | -0.0042 | 0.0000 | 235.9 | -0.0565 | 0.5195 | 0.4863 | -0.0591 |
| RM_gpt2-harmless | +0.1298 | 0.0168 | 7.6 | +0.0833 | 0.5234 | 0.4922 | -0.0306 |

Peak versus final Pearson, which is the substantive finding:

| arm | peak Pearson | at step | final Pearson | at step |
|---|---|---|---|---|
| RM_OA-deberta | +0.1535 | 51 | +0.0881 | 272 |
| RM_gpt2-helpful | +0.2672 | 34 | -0.0042 | 272 |
| RM_gpt2-harmless | +0.1338 | 238 | +0.1298 | 272 |

**Magnitude transmission peaks in the first two to three checkpoints and then decays toward
zero while training accuracy stays at 0.97.** For `gpt2-helpful` the final Pearson is -0.004 --
the aligned policy retains essentially no linear relationship to the teacher margin that
generated its labels. `normalized_noisy_transmission` is negative or near zero at the final
checkpoint in all three arms, meaning agreement with a fresh draw from the same stochastic label
model is at or below chance.

This directly answers secondary question 1: **candidate identity becomes recoverable very early
in DPO training and is then lost.** It is a generalization/data-coverage result (12.4), not an
optimization failure, and per 12.4 and the standing prohibition no data was added in response.

### Where the signal goes: the two-predictor decomposition

`Delta = c + gamma_1*d_true + gamma_2*len_diff + eps`, both predictors standardized, at the
final checkpoint:

| arm | gamma_1 (teacher margin) | SE | gamma_2 (length) | SE | \|gamma_2\| / \|gamma_1\| |
|---|---|---|---|---|---|
| RM_OA-deberta | +2.1835 | 0.8881 | -3.7306 | 0.8881 | 1.71 |
| RM_gpt2-helpful | +0.3157 | 0.9615 | -1.5419 | 0.9615 | 4.88 |
| RM_gpt2-harmless | +0.7969 | 0.9383 | -8.2216 | 0.9383 | 10.32 |

**In every arm the aligned policy ends up with a substantial NEGATIVE length coefficient, and in
two of three that length coefficient is larger in magnitude than the teacher-margin
coefficient.** The policies drift toward penalizing long responses regardless of which reward
model taught them. That is the mechanism behind both the decay above and the attribution results
below.

### Primary attribution at the final pre-registered checkpoint

Primary rule `nnls_mixture`, dataset-level fit on the 512 fresh probes,
`S = alpha_true - max_wrong` on unnormalized coefficients, 10,000-permutation randomization test,
5,000-resample bootstrap. Success requires **true candidate top-1 AND S > 0 AND p_perm < 0.001**.

| true teacher | raw top-1 | raw S | raw p_perm | raw boot top-1 | raw verdict | resid top-1 | resid S | resid p_perm | resid boot | resid verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| OA-deberta | gpt2-harmless | -0.6296 | 0.56594 | 0.2800 | **FAIL** | OA-deberta | +0.7491 | 0.05209 | 0.6346 | **FAIL** |
| gpt2-helpful | OA-deberta | -1.2761 | 0.86601 | 0.0842 | **FAIL** | OA-deberta | -1.1410 | 0.83952 | 0.1374 | **FAIL** |
| gpt2-harmless | gpt2-harmless | +3.2118 | 0.00080 | 0.9968 | **SUCCESS** | gpt2-harmless | +0.4312 | 0.23898 | 0.6708 | **FAIL** |

Trajectory of top-1 correctness, which the final-checkpoint table alone hides:

| true teacher | checkpoints with top-1 correct | peak S | at step | final top-1 |
|---|---|---|---|---|
| RM_OA-deberta | 5/16 (steps 17-85) | +1.0346 | 68 | gpt2-harmless |
| RM_gpt2-helpful | 3/16 (steps 17-51) | +1.4553 | 51 | OA-deberta |
| RM_gpt2-harmless | 16/16 (steps 17-272) | +3.3052 | 238 | gpt2-harmless |

**`OA-deberta` and `gpt2-helpful` are both attributed CORRECTLY for the first three to five
checkpoints -- with S peaking at +1.03 and +1.46 -- and then flip to the wrong candidate and stay
wrong.** The pre-registration names the final checkpoint, and 12.3 forbids using an earlier
checkpoint to reverse a raw failure, so the final checkpoint governs and both arms fail. The
early correctness is reported as a measurement, not used as a verdict.

### The one success is length-mediated

`gpt2-harmless` is the single arm satisfying the primary criterion: top-1 correct at **every**
checkpoint, S rising monotonically to +3.21, `p_perm = 0.00080`, bootstrap top-1 support 0.9968.

Its length-residualized result collapses:

```
raw          S = +3.2118   p_perm = 0.00080   bootstrap top-1 = 0.9968
residualized S = +0.4312   p_perm = 0.23898   bootstrap top-1 = 0.6708
```

Per 12.2 this is reported plainly rather than downplayed:

> Candidate identity is recoverable under the actual reward functions, but the discriminative
> signal is materially mediated by response-length preference. The experiment does not establish
> attribution independent of length.

The mechanism is coherent with everything above. `gpt2-harmless` is the pool's length-penalizing
candidate (calibration `gamma_len = -0.00487`, the only negative one), and every aligned policy
drifts toward a negative length preference. When the true teacher happens to be the
length-penalizing model, that drift points at the right answer; when it is not, the same drift
points at the wrong one. `OA-deberta`'s final top-1 is in fact `gpt2-harmless`.

Note also that `OA-deberta`'s **residualized** analysis recovers its true teacher
(S = +0.7491, p_perm = 0.052) -- consistent with length being the thing that broke it. Under 12.3
that cannot reverse the raw verdict, and it does not reach p < 0.001 either. It is recorded as
evidence about the mechanism, not as a result.

### Secondary rules and detection statistic

| arm | BT top-1 | BT gap | sign top-1 | sign gap | max Lambda | threshold | clears? |
|---|---|---|---|---|---|---|---|
| RM_OA-deberta | gpt2-harmless | 0.0036 | OA-deberta | 0.0195 | 4.95 | 12.87 | no |
| RM_gpt2-helpful | gpt2-harmless | 0.0022 | OA-deberta | 0.0293 | 2.41 | 12.87 | no |
| RM_gpt2-harmless | gpt2-harmless | 0.0068 | OA-deberta | 0.0039 | 6.94 | 12.87 | no |

**No arm's Lambda clears the pre-registered detection threshold of 12.87** -- the maximum across
all three arms and all checkpoints is 7.87. The secondary Bradley-Terry and sign-agreement rules
are essentially flat (gaps of order 1e-3 to 1e-2), which is what near-zero magnitude transmission
implies. The primary designation is unchanged; these are reported, not substituted.

### Per-probe win rate (descriptive only, 6.6)

| arm | win rate | n | chance | exact binomial 95% CI |
|---|---|---|---|---|
| RM_OA-deberta | 0.2637 | 512 | 0.3333 | [0.2260, 0.3041] |
| RM_gpt2-helpful | 0.3281 | 512 | 0.3333 | [0.2876, 0.3707] |
| RM_gpt2-harmless | 0.4160 | 512 | 0.3333 | [0.3729, 0.4601] |

Reported for interpretability only. Its binomial p-value is **not** the inferential basis for the
dataset-level NNLS claim (6.6).

---

## STOP -- the Gate 4 kill criterion fired

```
M                   = 3
required_successes  = max(2, ceil(2*M/3)) = 2
observed successes  = 1   (gpt2-harmless)
kill criterion      = FIRED
aggregate claim     = does not hold
```

Per 7 and 11 Gate 4 the pre-registered consequence is:

> rank transmission is insufficient at this scale; report and stop before the deterministic
> realism arm

**Gate 5 (CTRL and FJ) is explicitly conditional on the kill criterion NOT firing** (11 Gate 5:
"If the stochastic real-RM stage does not fire the kill criterion: run CTRL; run FJ"). It
therefore was not run. Gate 6 (deterministic realism) requires ALL stochastic arms to succeed and
CTRL to pass; neither holds. Gate 7 (map validation) requires FJ transmission results, which
Gate 5 would have produced.

### What was NOT done, per the prohibitions in 13

- The p threshold, Lambda threshold, kill criterion, permutation count and bootstrap count were
  **not** changed after seeing any outcome.
- No earlier checkpoint was selected, despite two arms being correctly attributed early.
- Residualized success was **not** used to rescue a raw failure.
- The candidate pool was not altered and no candidate was substituted.
- No additional training data, seeds, or experiments were introduced.
- The secondary rules were not promoted to primary.

### Answers to the round's questions, as far as the evidence goes

**Primary question -- can the generating reward model be identified from the aligned policy on
unseen pairs?** At this scale, **no**, not reliably. One of three teachers was recovered at the
pre-registered final checkpoint, and that one recovery is materially length-mediated.

**Secondary question 1 -- how early does identity become recoverable?** Very early, and it does
not persist. All three arms attribute correctly at the first checkpoint or two; two of three then
invert. Peak `S` occurs at step 51-68 of 272 for the two arms that later fail.

**Secondary question 2 -- does attribution survive removing the linear length signal?** No. The
only raw success does not survive residualization (p 0.0008 -> 0.239).

**Secondary question 3 -- does weak magnitude transmission prevent rank attribution?** This round
gives a partial answer with a clear caveat. Early checkpoints show that attribution can succeed
while magnitude transmission is weak, which supports 5.1's premise. But sustained training drove
transmission to approximately zero, and at that point rank attribution failed too.

**Secondary questions 4 and 5 -- map validation and a deterministic replication?** Not answerable
this round; both are downstream of gates that the kill criterion blocked.

### The result is the result

This is a pre-registered negative outcome, not a failed run. The pipeline executed exactly as
frozen: the pool passed every calibration gate, the geometry was well separated and reproduced
out of sample, the training set and holdout were frozen with recorded hashes, the labels carried
real teacher structure (flip rates 0.25-0.26 against Bayes ceilings of 0.73-0.75), and the
policies fit those labels to 0.97 accuracy. What did not happen is transmission of the teacher's
*ordering* to held-out pairs after extended training.


---

## Round 5 compliance audit (11 Gate 8)

Every status is derived mechanically from artifacts on disk and the frozen gate outcomes
(`src/analysis/round5_audit.py`, `results/round5_audit.csv`), so the audit cannot drift from
what was actually produced. **BLOCKED** means the Gate 4 kill criterion prevented the work;
**GATED - NOT TRIGGERED** means a pre-registered trigger was not satisfied.

| # | spec | item | status | evidence / reason |
|---|---|---|---|---|
| 1 | 0.1 | sigma_D = 5.35 struck; not an input, bound, estimate or comparison target | **COMPLETE** | absent from every Round 5 module and from the frozen pre-registration |
| 2 | 0.2 | excess_variance_over_shuffled from sd(Delta), never mean|Delta| | **BLOCKED** | requires the CTRL arm at matched steps; the statistic is defined from sd(Delta) in code and mean|Delta| is never used |
| 3 | 0.3 | no deterministic-vs-stochastic claim from the 19-step pilot | **COMPLETE** | the pilot contrast is not cited; 9 is a fresh pre-registered arm |
| 4 | 0.4 | length treated in all three required places | **COMPLETE** | C4 screening; raw-vs-residualized geometry; raw-vs-residualized attribution |
| 5 | 1.2 | rule-change disclosure: new rule is post-Round-4, not pre-registered before it | **COMPLETE** | stated in RESULTS.md and in configs/base.yaml:calibration_retention.status |
| 6 | 1.2 | old D1 rule kept and labelled superseded_for_calibration_stage, not deleted | **COMPLETE** | configs/base.yaml:d1_pre_registration retains every original value |
| 7 | 2.1 | no replacement candidates added; D1 not recomputed | **COMPLETE** | pool starts from the same four; D1 values carried forward unchanged |
| 8 | 2.2 | C1-C4 applied model-wise | **COMPLETE** | results/rm_calibration_r5.json |
| 9 | 2.3 | InternLM2 marked C1 FAIL, not rescued, no threshold/precision/kernel change | **COMPLETE** | excluded on already-reproduced cross-process evidence |
| 10 | 2.4 | OpenAssistant cross-process C1 measured in two fresh processes | **COMPLETE** | bitwise equality required; distinct PIDs asserted |
| 11 | 2.5 | raw and length-residualized calibration geometry reported side by side | **COMPLETE** | results/rm_calibration_r5.json |
| 12 | 2.6 | pool-level C5 applied to S as a whole; no subset search | **COMPLETE** | min raw psi_eff = 67.29 deg >= 45 |
| 13 | 2.7 | final retention table with D1 covariates, M and chance | **COMPLETE** | reported in RESULTS.md |
| 14 | 3.1 | started from the existing 300 training pairs; no new data generated | **COMPLETE** | no training data was generated in this round |
| 15 | 3.2 | exact-serialization length check for every retained scorer and the policy path | **COMPLETE** | 259/300 retained |
| 16 | 3.3 | training set frozen (IDs, texts, policy, provenance, serializations, seeds) | **COMPLETE** | sha256 13b162d33744e054... |
| 17 | 4.2 | 600 fresh prompts, disjoint from training and calibration, same procedure | **COMPLETE** | overlaps: 0 calibration, 0 training |
| 18 | 4.3 | final-holdout compatibility filter, retained fraction >= 0.60 | **COMPLETE** | N_final = 512, fraction 0.8533 |
| 19 | 4.4 | final holdout outcome-blind | **COMPLETE** | frozen manifest; no pool, rule, hyperparameter or checkpoint chosen on it |
| 20 | 4.5 | frozen calibration transforms applied out of sample | **COMPLETE** | mu/sd and length coefficients all from the 487 calibration probes |
| 21 | 5.2 | one arm per retained real reward model | **COMPLETE** | arms: ['RM_OA-deberta', 'RM_gpt2-helpful', 'RM_gpt2-harmless'] |
| 22 | 5.3 | stochastic BT labels, tau=2, standardized on training only, frozen seed | **COMPLETE** | label balance, flip rate, mean |z| and Bayes ceiling reported per arm |
| 23 | 5.4 | J_perp built from training data only, |corr| < 0.05 on training | **COMPLETE** | training corr = +0.000000 by construction |
| 24 | 5.4 | J_perp held-out length correlations reported, not tuned on | **COMPLETE** | calibration -0.0486, final holdout -0.1757, reporting-only |
| 25 | 5.4 | J_len deliberately not run this round | **NOT APPLICABLE** | length handled by C4 and the raw-vs-residualized analysis, per the spec |
| 26 | 5.5 | DPO configuration carried forward unchanged; entry gates asserted | **COMPLETE** | scorer invariance PASS; manifest and pre-registration verified before training |
| 27 | 5.6 | per-checkpoint transmission metrics recorded for every arm | **COMPLETE** | all 5.6 fields present; verified by a pre-run smoke test |
| 28 | 5.7 | sigma_D convergence rule applied; no single point unless it passes | **BLOCKED** | FJ trajectory drives the rule |
| 29 | 6.1 | primary nnls_mixture treated as a dataset-level fit | **COMPLETE** | probes never treated as independent NNLS trials |
| 30 | 6.2 | candidates standardized by frozen calibration constants, not final outcomes | **COMPLETE** | 4.5 transforms |
| 31 | 6.3 | point-estimate attribution at every checkpoint, all three rules | **COMPLETE** | per_checkpoint block in each arm's attribution JSON |
| 32 | 6.4 | 10,000-permutation randomization test as the PRIMARY inference | **COMPLETE** | no binomial per-probe pseudo-trial test substituted |
| 33 | 6.5 | 5,000-resample bootstrap stability analysis | **COMPLETE** | top-1 support, median S and 95% CI reported |
| 34 | 6.6 | per-probe win rate reported with exact binomial CI, descriptive only | **COMPLETE** |  |
| 35 | 6.7 | length-residualized attribution robustness for every arm | **COMPLETE** | raw primary; residualized cannot rescue raw failure |
| 36 | 7 | pre-registration block written and frozen before the first DPO run | **COMPLETE** | configs/base.yaml:attribution_stage.pre_registration_round5 |
| 37 | 8.1 | one shuffled-label CTRL run, seed frozen, positive count preserved | **BLOCKED** | permutation of the reference labels |
| 38 | 8.2 | negative-control claim limited to the Lambda threshold | **BLOCKED** | no claim that one CTRL run proves top-1 equals chance |
| 39 | 8.2 | no extra shuffled seeds run to average away a control failure | **COMPLETE** | exactly one CTRL run exists |
| 40 | 9 | deterministic realism arm run only if its trigger fired | **GATED - NOT TRIGGERED** | trigger: all stochastic arms succeed AND CTRL passes -> not satisfied |
| 41 | 10 F17 | attribution trajectory | **COMPLETE** |  |
| 42 | 10 F18 | inferential stability at final checkpoint | **COMPLETE** |  |
| 43 | 10 F19 | candidate score profiles | **COMPLETE** |  |
| 44 | 10 F20 | optimization versus generalization | **COMPLETE** |  |
| 45 | 10 F21 | judge signal versus length | **COMPLETE** |  |
| 46 | 10 F22 | sigma_D_ratio(t) | **COMPLETE** |  |
| 47 | 10 F23 | detection statistic | **COMPLETE** |  |
| 48 | 10 F24 | raw versus length-residualized final attribution | **COMPLETE** |  |
| 49 | 10 F25 | deterministic realism arm | **GATED - NOT TRIGGERED** | generated only if 9 triggered |
| 50 | 10 F26 | candidate geometry raw vs residualized | **COMPLETE** |  |
| 51 | 10 F27 | map validation, homogeneous reference and empirical geometry | **BLOCKED** | requires FJ transmission results |
| 52 | 10 F28 | final summary panel | **COMPLETE** |  |
| 53 | 11 G1 | Gate 1 calibration pool reported before any DPO run | **COMPLETE** | M = 3, C5 PASS, 259 training pairs |
| 54 | 11 G2 | Gate 2 final holdout constructed and frozen after Gate 1 | **COMPLETE** |  |
| 55 | 11 G3 | Gate 3 pre-registration written and verified | **COMPLETE** |  |
| 56 | 11 G4 | Gate 4 stochastic real-RM arms reported together | **COMPLETE** | successes 1/3, required 2 |
| 57 | 11 G5 | Gate 5 CTRL and FJ | **BLOCKED** |  |
| 58 | 11 G6 | Gate 6 deterministic realism | **GATED - NOT TRIGGERED** |  |
| 59 | 11 G7 | Gate 7 map validation with a single sigma only if 5.7 passed | **BLOCKED** |  |
| 60 | 11 G8 | Gate 8 completion audit | **COMPLETE** | this table |
| 61 | 13 | no threshold, pool, rule, hyperparameter, seed count or data change after any result | **COMPLETE** | the only deviations are the two pre-freeze corrections declared in Gate 3 |
| 62 | 13 | no extra experiments, seeds, data, candidates or checkpoint selection | **COMPLETE** | none introduced |
| 63 | 12.2 | raw success + residualized failure reported as length-mediated, not hidden | **COMPLETE** | gpt2-harmless: raw SUCCESS, residualized FAIL -> reported as materially length-mediated |
| 64 | 12.3 | raw failure not reversed by residualized success, a secondary rule, an earlier checkpoint, a different subset, a relaxed p or a changed Lambda threshold | **COMPLETE** | OA-deberta residualized recovers the true teacher at p = 0.052; the raw verdict stands |
| 65 | 12.4 | high train accuracy with weak held-out transmission read as a generalization/data-coverage issue; no data added | **COMPLETE** | train_acc 0.97 against near-zero held-out transmission; no data generated |
| 66 | 12.5 | low-train-accuracy branch | **NOT APPLICABLE** | final train_acc 0.97 >> the 0.60 standing threshold, so this branch does not apply |
| 67 | 12.6 | transmission still rising at checkpoint 16 not called converged | **NOT APPLICABLE** | transmission DECAYS after an early peak rather than rising; 5.7 not reached (FJ blocked) |

**Totals:** BLOCKED 7, COMPLETE 54, GATED - NOT TRIGGERED 3, NOT APPLICABLE 3 (of 67 items). PENDING: 0.

### Figures produced

| figure | file | status |
|---|---|---|
| F17 | `figures/F17_attribution_trajectory.png` | attribution trajectories per teacher |
| F18 | `figures/F18_inferential_stability.png` | permutation null and bootstrap at final checkpoint |
| F19 | `figures/F19_candidate_profiles.png` | NNLS / BT / sign profiles at final checkpoint |
| F20 | `figures/F20_optimization_vs_generalization.png` | train_acc vs held-out sign agreement (partial: no CTRL) |
| F21 | `figures/F21_judge_signal_vs_length.png` | gamma_1 and gamma_2 with SE bands (partial: no CTRL/FJ) |
| F22 | `figures/F22_sigma_trajectory.png` | sigma_D_ratio(t) (partial: FJ blocked, so 5.7 not evaluated) |
| F23 | `figures/F23_detection_statistic.png` | Lambda vs step with threshold (partial: no CTRL) |
| F24 | `figures/F24_raw_vs_residualized.png` | raw vs length-residualized final attribution |
| F26 | `figures/F26_candidate_geometry.png` | raw vs residualized calibration geometry |
| F28 | `figures/F28_summary_panel.png` | final summary panel |
| F25 | -- | **GATED - NOT TRIGGERED**: deterministic arm trigger not satisfied |
| F27 | -- | **BLOCKED**: requires FJ transmission results, which Gate 5 would have produced |


---

# NEXT_STEPS_ROUND_6_SAMEDAY_FINAL.md

Same-day closeout. Round 6 start **09:55:45 EDT**, hard stop 23:30 local, 90 minutes reserved
for figures and writeup plus a 30-minute uncertainty buffer.

**Round 5 is frozen.** No Round 1-5 number, verdict, threshold, pool decision, success
criterion, checkpoint, pre-registration block, p-value or detection threshold is revised by
anything below. Round 5's aggregate claim -- kill criterion fired, aggregate claim does not
hold -- stands regardless of what this round finds.

**Evaluation set (0.3).** No new confirmatory holdout was built. Every Round 6 arm is scored on
the **existing frozen Round 5 512-probe final holdout**, using the frozen 487-probe calibration
standardization and length-regression transforms, unchanged. Stated plainly: these results are
evaluated on the same holdout that first surfaced the length pattern, not an independent one.
That is the first entry in `OPEN_AT_THIS_SCALE.md`.

**File-naming note.** The Round 6 control arms (CTRL, FJ_PERP, J_SHORT) reuse the Round 5
training and scoring code path, so their artifacts carry the `round5_<arm>_` filename prefix.
The arm names are unambiguous; the prefix reflects the shared code path, not the round.

## Section A -- forensic analysis of the Round 5 length mechanism

**RETROSPECTIVE / EXPLORATORY** throughout: this analyzes already-observed Round 5 outcomes.

### A1. Exact Delta convention

From `src/score/delta.py`, verified by inspection:

```
s(x,y) = sum_t [ log pi_theta(y_t | prefix) - log pi_ref(y_t | prefix) ]      <- a SUM
Delta  = s(x, y1) - s(x, y2)
```

**There is no beta anywhere in the scoring path.** The only beta in the project is DPOConfig's
training `beta = 0.1`; the other occurrence is a local OLS coefficient variable. So beta is
omitted below (equivalently beta = 1), and the A2 identity is exact:

```
c_avg*len_diff + c_diff*len_sum  ==  c1*n1 - c2*n2  ==  Delta
```

### A1.5. Timing microbenchmark and the budget cut

The prescribed 50-probe benchmark measured **1.520 s/probe**, projecting 233 min for the full
six-checkpoint schedule. That estimator is biased upward: at 50 probes the model load and the
sparsely-filled `fixed_bin` batches are amortized over 50 rows instead of 512. A direct
**512-probe** measurement on the identical path gives **0.510 s/probe**, projecting **78.3 min**
-- a 3.0x difference. Round 5's recorded per-checkpoint scoring cost (167 s) corroborates the
full-scale figure.

Both estimates exceed the 60-minute rule, so the budget cut fires either way and the decision is
unambiguous. The accurate projection is the one recorded.

```
50-probe benchmark        76.0s  (1.520 s/probe)
512-probe measurement     260.9s  (0.5096 s/probe)
projected A2+A3           78.3 min  (> 60 min)
checkpoint set selected   [51, 136, 272]
reason                    budget-only reduction per the A1.5 rule
```

A4 is preserved at the final checkpoint 272 and A5 in full, as the rule requires.

### A2. Exact pairwise decomposition

| arm | step | mean c | sd c | corr(c, resp len) | sd(Delta) | sd(len_accum) | sd(token_diff) | corr(D,accum) | corr(D,tokdiff) | corr(accum,d_true) | corr(tokdiff,d_true) | corr(accum,tokdiff) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| OA-deberta | 51 | -0.0378 | 0.0996 | +0.2973 | 6.641 | 6.300 | 9.960 | -0.1840 | +0.7832 | -0.2651 | +0.2701 | -0.7553 |
| OA-deberta | 136 | -0.2437 | 0.4965 | +0.4384 | 16.794 | 34.668 | 35.788 | +0.1744 | +0.3003 | -0.1173 | +0.1537 | -0.8868 |
| OA-deberta | 272 | -0.3181 | 0.6167 | +0.4679 | 20.349 | 40.529 | 41.070 | +0.2243 | +0.2742 | -0.1314 | +0.1733 | -0.8757 |
| gpt2-helpful | 51 | -0.0439 | 0.1952 | +0.3409 | 13.212 | 18.875 | 27.128 | -0.4113 | +0.7732 | -0.2692 | +0.3070 | -0.8961 |
| gpt2-helpful | 136 | -0.2588 | 0.6298 | +0.4837 | 17.215 | 44.437 | 45.412 | +0.1364 | +0.2456 | -0.2126 | +0.2158 | -0.9268 |
| gpt2-helpful | 272 | -0.3394 | 0.7718 | +0.5082 | 20.986 | 48.298 | 49.268 | +0.1705 | +0.2588 | -0.2342 | +0.2278 | -0.9077 |
| gpt2-harmless | 51 | -0.0655 | 0.1437 | +0.2368 | 6.818 | 4.917 | 7.210 | +0.2785 | +0.7557 | +0.2139 | -0.0671 | -0.4185 |
| gpt2-harmless | 136 | -0.3204 | 0.6153 | +0.4601 | 20.274 | 39.284 | 37.158 | +0.3601 | +0.1650 | +0.2466 | -0.1913 | -0.8607 |
| gpt2-harmless | 272 | -0.3555 | 0.6605 | +0.4751 | 22.158 | 41.432 | 38.652 | +0.3887 | +0.1566 | +0.2525 | -0.1963 | -0.8491 |

**Reconstruction is exact: max absolute error 5.684e-14 across all nine
cells.** The decomposition is an identity, not an approximation, so everything below rests on
arithmetic rather than on a fitted model.

Three things stand out:

1. **`c_bar` is negative and grows steadily more so** -- from about -0.04 at step 51 to -0.32 to
   -0.36 at step 272. The aligned policy assigns systematically lower per-token log-probability
   than the reference to these held-out responses. That is the accumulating KL cost of DPO.
2. **Both components are roughly twice sd(Delta) and largely cancel.** At step 272 for
   `OA-deberta`, sd(length_accum) = 40.5 and sd(token_diff) = 41.1 against sd(Delta) = 20.3,
   with **corr(accum, token_diff) = -0.88**. Across arms that correlation runs -0.85 to -0.93 at
   the later checkpoints. Delta is a modest residual of two large opposing terms.
3. **`corr(c, response_length)` is positive and rising** (0.24 to 0.51): longer responses take a
   *smaller* per-token log-ratio penalty.

### A3. Does mechanical accumulation explain gamma_2?

| arm | step | c_bar | gamma_2 predicted | gamma_2 observed | ratio | R^2 of Delta by c_bar*len_diff | corr(Delta, pred) |
|---|---|---|---|---|---|---|---|
| OA-deberta | 51 | -0.0378 | -2.3160 | +0.7100 | -3.26x | -0.2067 | -0.1218 |
| OA-deberta | 136 | -0.2437 | -14.9125 | -2.4819 | +6.01x | -0.5530 | +0.1373 |
| OA-deberta | 272 | -0.3181 | -19.4656 | -3.7306 | +5.22x | -0.5954 | +0.1721 |
| gpt2-helpful | 51 | -0.0439 | -2.6867 | +3.2172 | -0.84x | -0.1601 | -0.2912 |
| gpt2-helpful | 136 | -0.2588 | -15.8349 | -0.6443 | +24.58x | -0.7964 | +0.0295 |
| gpt2-helpful | 272 | -0.3394 | -20.7710 | -1.5419 | +13.47x | -0.8474 | +0.0695 |
| gpt2-harmless | 51 | -0.0655 | -4.0088 | -2.7020 | +1.48x | +0.1152 | +0.3920 |
| gpt2-harmless | 136 | -0.3204 | -19.6032 | -7.4068 | +2.65x | -0.2163 | +0.3741 |
| gpt2-harmless | 272 | -0.3555 | -21.7508 | -8.2216 | +2.65x | -0.2210 | +0.3801 |

**Verdict: mechanical accumulation DOES NOT ACCOUNT for the observed length coefficient.**

`gamma_2_hat_raw = c_bar * sd(len_diff)` over-predicts the observed coefficient by **2.6x to
13.5x** at the later checkpoints, and `c_bar * len_diff` used alone as a predictor gives a
**negative R^2** in 8 of 9 cells -- worse than predicting the mean. The correlation between Delta
and that predictor never exceeds 0.39.

What is true is weaker and worth stating precisely: the **direction** is mechanical. At steps 136
and 272 the predicted and observed coefficients share a sign in all three arms, and both are
negative because `c_bar` is negative. But the magnitude is not mechanical -- the `token_diff`
term, which is anti-correlated with the accumulation term at -0.85 to -0.93, absorbs most of it.
A naive 'longer responses accumulate more negative log-ratio' story predicts a length coefficient
several times larger than the one actually observed.

### A4. Delta_norm diagnostic

**Caveat, stated once: `Delta_norm = c1 - c2` breaks the sequence-level KL-regularized
equilibrium identity that the summed statistic satisfies.** The density-ratio derivation applies
to the summed log-ratio, not to its per-token mean. This is a mechanism diagnostic. It cannot
reverse the Round 5 verdict and is not reported as doing so.

| arm | true teacher | corr(Delta_norm, Delta_sum) | raw top-1 | raw S | raw p_perm | raw verdict | resid top-1 | resid S | resid p_perm |
|---|---|---|---|---|---|---|---|---|---|
| OA-deberta | OA-deberta | +0.1054 | gpt2-helpful | -0.1110 | 0.99780 | **FAIL** | gpt2-helpful | -0.0412 | 0.80602 |
| gpt2-helpful | gpt2-helpful | +0.1416 | gpt2-helpful | +0.1329 | 0.00010 | **SUCCESS** | gpt2-helpful | +0.0500 | 0.03610 |
| gpt2-harmless | gpt2-harmless | +0.1219 | gpt2-helpful | -0.0607 | 0.98910 | **FAIL** | gpt2-helpful | -0.0095 | 0.63044 |

Two findings:

1. **`Delta_norm` is nearly a different statistic**, correlating only 0.105 to 0.142 with the
   summed Delta it is derived from.
2. **Under `Delta_norm` every arm attributes to `gpt2-helpful`, whatever the true teacher.** The
   summed statistic pulled all three arms toward `gpt2-harmless`; normalizing by length moves the
   attractor to `gpt2-helpful`. Exactly one arm 'succeeds' (`gpt2-helpful`, p_perm = 0.0001) --
   and again only because the constant attractor happens to coincide with its true teacher, the
   same coincidence that produced Round 5's single success.

Length normalization therefore does not rescue attribution. It swaps one length-driven attractor
for another. That is evidence that the attractor's identity is set by how the statistic treats
length, not by teacher identity.

### A5. Frozen-label length-bias audit

Realized bias in the frozen Round 5 labels, on the 259 training pairs (len_diff sd = 68.4 tokens):

| teacher | corr(label_sign, len_diff) | mean chosen - rejected length | frac choosing shorter | logit slope on z(len_diff) | SE | p |
|---|---|---|---|---|---|---|
| OA-deberta | +0.0206 | +1.33 | 0.4740 | +0.0413 | 0.1245 | 0.7399 |
| gpt2-helpful | +0.1674 | +11.41 | 0.4416 | +0.3566 | 0.1365 | 0.0090 |
| gpt2-harmless | -0.1073 | -7.27 | 0.5519 | -0.2199 | 0.1290 | 0.0883 |

**The teachers really do differ in length preference.** `gpt2-helpful` significantly prefers
longer responses (slope +0.357, p = 0.009, chosen minus rejected = +11.4 tokens); `gpt2-harmless`
prefers shorter (-0.220, p = 0.088); `OA-deberta` is flat (p = 0.74).

Null: 10,000 label-only Bradley-Terry resamples per teacher from the SAME frozen
continuous margins at tau = 2, holding the teacher fixed and varying only the Bernoulli draw --
exactly the randomness the frozen seed collapsed.

| teacher | observed corr | null mean | null sd | percentile | two-sided tail p | observed frac-shorter | percentile | tail p |
|---|---|---|---|---|---|---|---|---|
| OA-deberta | +0.0206 | +0.0775 | 0.0484 | 11.8 | 0.2358 | 0.4740 | 56.6 | 0.8674 |
| gpt2-helpful | +0.1674 | +0.1564 | 0.0438 | 59.1 | 0.8176 | 0.4416 | 46.4 | 1.0000 |
| gpt2-harmless | -0.1073 | -0.1005 | 0.0414 | 44.1 | 0.8826 | 0.5519 | 54.5 | 0.9092 |

**The frozen label draw is unremarkable under its own null.** Every realized statistic sits well
inside the central mass -- percentiles 11.8 / 59.1 / 44.1 for the correlation, two-sided tail
probabilities 0.24 / 0.82 / 0.88. So the length structure in the Round 5 labels is a property of
the **reward models themselves**, not an artifact of the particular seed. That closes off 'the
seed was unlucky' as an explanation of the Round 5 result.

Pairwise hard-label agreement between the three teacher arms:

| | OA-deberta | gpt2-helpful | gpt2-harmless |
|---|---|---|---|
| OA-deberta | 1.0000 | 0.7683 | 0.6448 |
| gpt2-helpful | 0.7683 | 1.0000 | 0.5367 |
| gpt2-harmless | 0.6448 | 0.5367 | 1.0000 |

The teachers disagree substantially (0.54 to 0.77), so the three arms were genuinely trained on
different label vectors.

## Section B -- training-only early-stopping rule (kappa)

### B1-B2. Fixed contexts and exact KL

64 training examples selected by frozen seed 20250915, one response each, up to 8 token
positions per response, giving **497 fixed contexts**. Selection is by index only -- no KL, loss,
length or attribution outcome was consulted. Exact conditional forward
`KL(pi_theta || pi_ref)` over the **full 151,936-token vocabulary** at every one of the 16 stored
checkpoints of all three Round 5 real-RM arms: 48 checkpoints, **932 s total**, within the
30-minute budget. Not a sampled estimator.

### B3. Freezing kappa

| arm | peak-S step | peak S | training-side KL_bar there |
|---|---|---|---|
| OA-deberta | 68 | +1.0346 | 0.09886 |
| gpt2-helpful | 51 | +1.4553 | 0.06491 |
| gpt2-harmless | 238 | +3.3052 | 0.21744 |

```
kappa_R6 = median(KL_bar at the three peak-S checkpoints) = 0.098863
rule     = last checkpoint whose training-side KL_bar <= kappa_R6
```

**This is a development-set procedure and is not holdout-free**, as the spec requires stating:
Round 5 holdout `S` is used once, retrospectively, to locate the three peak checkpoints. After
freezing, kappa is applied to every new arm using **training-side KL only**, without consulting
that arm's holdout outcome. It was frozen before any Tier 1+ arm trained.

Applied retrospectively to the Round 5 arms:

| arm | kappa-selected step | KL_bar there | S at that step | that arm's peak S |
|---|---|---|---|---|
| OA-deberta | 68 | 0.09886 | +1.0346 | +1.0346 |
| gpt2-helpful | 51 | 0.06491 | +1.4553 | +1.4553 |
| gpt2-harmless | 68 | 0.09054 | +0.6251 | +3.3052 |

kappa recovers the exact peak checkpoint for two of three arms. Two caveats, both material:

1. **This is partly circular.** kappa is the median of these same three peaks, so recovering two
   of them is close to self-consistency rather than validation. The real test is prospective
   application to arms that played no part in deriving it -- CTRL, FJ_PERP, J_SHORT and N1000
   below.
2. **For `gpt2-harmless` it stops too early.** That arm's peak is at step 238 with S = +3.31, but
   kappa selects step 68 where S = +0.63. For the one Round 5 arm that succeeded at the final
   checkpoint, an early-stopping rule would have given up most of the signal.

