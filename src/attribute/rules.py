"""The three attribution rules (CLAUDE_CODE_PROMPT.md Section 0) plus reward-vector recovery
used for the Phase 0 / Phase 1 angular-error diagnostic. Shared by every phase from 0 onward.

Inputs throughout: `delta` is the length-N vector of Delta_i = s(x,y) - s(x,y') for the audited
policy; `D` is the (N, M) matrix of candidate margins d_m_i, one column per candidate judge.
"""
import numpy as np
from scipy.optimize import lsq_linear, minimize_scalar


def sign_agreement(delta: np.ndarray, D: np.ndarray) -> np.ndarray:
    """A_m = (1/N) sum_i 1[sign(Delta_i) == sign(d_m_i)]. Returns shape (M,)."""
    sign_delta = np.sign(delta)[:, None]
    sign_D = np.sign(D)
    return np.mean(sign_delta == sign_D, axis=0)


def _bt_negloglik(gamma: float, delta: np.ndarray, z: np.ndarray) -> float:
    u = gamma * delta
    logsig_pos = -np.logaddexp(0.0, -u)  # log sigmoid(u)
    logsig_neg = -np.logaddexp(0.0, u)   # log sigmoid(-u)
    ll = z * logsig_pos + (1.0 - z) * logsig_neg
    return -float(np.mean(ll))


def bradley_terry_likelihood(
    delta: np.ndarray, D: np.ndarray, gamma_bounds: tuple[float, float] = (1e-6, 1e4),
) -> tuple[np.ndarray, np.ndarray]:
    """Fits one positive scale gamma_m per candidate maximizing the BT log-likelihood of
    z_m_i = 1[d_m_i > 0] under sigmoid(gamma * Delta_i). Returns (L_m, gamma_m), each shape (M,).
    The negative log-likelihood is convex in gamma for fixed (delta, z) since it is a 1-D
    logistic regression through the origin, so bounded scalar minimization is exact.
    """
    M = D.shape[1]
    L = np.zeros(M)
    gammas = np.zeros(M)
    for m in range(M):
        z = (D[:, m] > 0).astype(np.float64)
        res = minimize_scalar(
            _bt_negloglik, bounds=gamma_bounds, method="bounded", args=(delta, z),
        )
        gammas[m] = res.x
        L[m] = -res.fun
    return L, gammas


def nnls_mixture(delta: np.ndarray, D: np.ndarray) -> np.ndarray:
    """alpha_hat = argmin_{alpha >= 0} sum_i (Delta_i - <alpha, d_i>)^2. Returns shape (M,).

    Uses `lsq_linear` (trust-region reflective) rather than the legacy Lawson-Hanson `nnls`:
    with M candidates spanning a feature space of dimension d < M (Phase 0's d=4, M=5 by
    construction), candidate margin columns of D are structurally collinear, and the classic
    active-set `nnls` cycles indefinitely on such rank-deficient designs. `lsq_linear` solves
    the same convex bound-constrained least-squares problem without that failure mode.
    """
    result = lsq_linear(D, delta, bounds=(0.0, np.inf))
    return result.x


def recover_weights_ols(delta: np.ndarray, phi_diff: np.ndarray) -> np.ndarray:
    """Regress Delta on feature differences phi(x,y) - phi(x,y') to recover w_hat proportional
    to w_true / beta (Phase 0 / Phase 1 reward-recovery diagnostic only).
    """
    w_hat, *_ = np.linalg.lstsq(phi_diff, delta, rcond=None)
    return w_hat


def angular_error(w_hat: np.ndarray, w_true: np.ndarray) -> float:
    """theta(w_hat, w) = arccos( <w_hat, w> / (||w_hat|| ||w||) ), radians."""
    denom = np.linalg.norm(w_hat) * np.linalg.norm(w_true)
    if denom == 0.0:
        return float("nan")
    cos = np.dot(w_hat, w_true) / denom
    cos = np.clip(cos, -1.0, 1.0)
    return float(np.arccos(cos))


RULE_NAMES = ("sign_agreement", "bradley_terry", "nnls_mixture")


def run_all_rules(delta: np.ndarray, D: np.ndarray) -> dict[str, np.ndarray]:
    """Runs all three rules and returns {rule_name: scores (M,)}."""
    A = sign_agreement(delta, D)
    L, _gammas = bradley_terry_likelihood(delta, D)
    alpha = nnls_mixture(delta, D)
    return {"sign_agreement": A, "bradley_terry": L, "nnls_mixture": alpha}


def bt_detection_stats(delta: np.ndarray, D: np.ndarray) -> dict[str, np.ndarray]:
    """Per-candidate fitted BT scale gamma_hat_m and likelihood-ratio statistic
    Lambda_m = 2*N*(L_m(gamma_hat_m) - L_m(0)), with L_m(0) = -log(2) exactly (sigmoid(0)=0.5
    regardless of data, since it's the average log-likelihood at gamma=0). Under no judge
    signal gamma_hat -> 0 and Lambda -> 0, giving both statistics a meaningful null that the
    raw margin (a bare difference of two aggregate scores) does not have
    (NEXT_STEPS_TRACKS_A_B_C.md Section A3). Nearly free given the BT fit already exists.
    """
    L, gammas = bradley_terry_likelihood(delta, D)
    N = len(delta)
    L0 = -np.log(2.0)
    Lambda = 2 * N * (L - L0)
    return {"gamma_hat": gammas, "Lambda": Lambda}
