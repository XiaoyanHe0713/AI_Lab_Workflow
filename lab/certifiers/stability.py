"""Stability-region and SSP-coefficient certificate for explicit Runge-Kutta methods.

A candidate is a method given by its Butcher tableau {"A": ..., "b": ...} or by its Shu-Osher form
{"alpha": ..., "beta": ...} (u^(0) = u^n; u^(i) = sum_k alpha_ik u^(k) + dt sum_k beta_ik F(u^(k));
u^{n+1} = u^(s)). The certifier computes

  * the stability polynomial R(z) = 1 + sum_k (b^T A^{k-1} 1) z^k and checks it against one step of
    the method on u' = z u at seeded random z (the polynomial must agree with the method it claims
    to describe);
  * the region of absolute stability {z : |R(z)| <= 1}: its extent along the negative real axis, along
    the imaginary axis, and its area on a grid;
  * the order by the classical order conditions up to four;
  * the SSP coefficient: the radius of absolute monotonicity R(K) of K = [[A, 0], [b^T, 0]] by bisection
    (the optimal Shu-Osher representation), and, when a Shu-Osher form is given, the coefficient of
    that representation, min alpha_ik / beta_ik over beta_ik > 0 with all coefficients nonnegative.

Classical and SSP methods at the bottom are the known answers for the certifier's tests and the
scoreboard: Euler, Heun, Kutta's third-order rule and classical RK4 reproduce the tabulated real-axis
extents 2, 2, 2.5127, 2.7853 and imaginary extents 0, 0, sqrt(3), 2 sqrt(2); SSPRK(2,2), (3,3), (4,3)
and Spiteri-Ruuth's (5,4) reproduce the SSP coefficients 1, 1, 2 and 1.508; the optimal second-order
family SSPRK(s,2) has coefficient s - 1; classical RK4 is not SSP (coefficient 0).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .base import Certificate, Certifier

KINDS = ("ssp", "real_axis", "imag_axis")


# --- tableaux ------------------------------------------------------------------------------------
def shu_osher_to_butcher(alpha, beta) -> tuple[np.ndarray, np.ndarray]:
    """Butcher (A, b) of the Shu-Osher form: rows r_0 = 0, r_i = sum_k alpha_ik r_k + sum_k beta_ik e_k."""
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    if alpha.shape != beta.shape or alpha.ndim != 2 or alpha.shape[0] != alpha.shape[1]:
        raise ValueError("alpha and beta must be square matrices of the same shape (stages x stages)")
    s = alpha.shape[0]
    rows = [np.zeros(s)]
    for i in range(1, s + 1):
        r = np.zeros(s)
        for k in range(i):
            r += alpha[i - 1, k] * rows[k] + beta[i - 1, k] * np.eye(s)[k]
        rows.append(r)
    A = np.array(rows[:s])
    b = rows[s]
    return A, b


def normalize_method(candidate: Any) -> dict:
    """Accept Butcher or Shu-Osher input; return A, b, c, stages and (if given) the Shu-Osher matrices."""
    if not isinstance(candidate, dict):
        raise ValueError("a method is a dict with A and b (Butcher) or alpha and beta (Shu-Osher)")
    out: dict[str, Any] = {"alpha": None, "beta": None}
    if "alpha" in candidate and "beta" in candidate:
        alpha = np.asarray(candidate["alpha"], dtype=float)
        beta = np.asarray(candidate["beta"], dtype=float)
        A, b = shu_osher_to_butcher(alpha, beta)
        out["alpha"], out["beta"] = alpha, beta
        row_sums = alpha.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-12):
            raise ValueError(f"Shu-Osher alpha rows must sum to one; got {row_sums.tolist()}")
    elif "A" in candidate and "b" in candidate:
        A = np.asarray(candidate["A"], dtype=float)
        b = np.asarray(candidate["b"], dtype=float).reshape(-1)
    else:
        raise ValueError("a method needs A and b, or alpha and beta")
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] != b.shape[0]:
        raise ValueError(f"A must be s x s and b of length s; got {A.shape} and {b.shape}")
    if not (np.all(np.isfinite(A)) and np.all(np.isfinite(b))):
        raise ValueError("the tableau must be finite")
    s = A.shape[0]
    explicit = bool(np.allclose(np.triu(A), 0.0, atol=1e-14))
    out.update({"A": A, "b": b, "c": A.sum(axis=1), "stages": s, "explicit": explicit})
    return out


# --- stability function ------------------------------------------------------------------------
def stability_polynomial(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Coefficients c_0..c_s of R(z) for an explicit method: c_0 = 1, c_k = b^T A^(k-1) 1."""
    s = A.shape[0]
    one = np.ones(s)
    coeffs = [1.0]
    v = one.copy()
    for _ in range(s):
        coeffs.append(float(b @ v))
        v = A @ v
    return np.array(coeffs)


def R_of(coeffs: np.ndarray, z):
    """Evaluate the polynomial with coefficients c_0..c_s at z (scalar or array, real or complex)."""
    z = np.asarray(z)
    out = np.zeros_like(z, dtype=complex)
    for c in coeffs[::-1]:
        out = out * z + c
    return out


def step_test_equation(A: np.ndarray, b: np.ndarray, z: complex) -> complex:
    """One step of the method on u' = z u from u = 1 with dt = 1, by the stage equations themselves."""
    s = A.shape[0]
    k = np.zeros(s, dtype=complex)
    for i in range(s):
        k[i] = z * (1.0 + sum(A[i, j] * k[j] for j in range(i)))   # explicit: stages in order
    return 1.0 + complex(np.dot(b, k))


def extent_polynomial(coeffs: np.ndarray, direction: complex) -> np.ndarray:
    """Real coefficients (ascending in t) of |R(t * direction)|^2 - 1, a polynomial in the real variable t."""
    a = np.asarray(coeffs, dtype=complex) * np.asarray(direction, dtype=complex) ** np.arange(len(coeffs))
    P = np.zeros(2 * len(a) - 1)
    for j in range(len(a)):
        for k in range(len(a)):
            P[j + k] += (a[j] * np.conj(a[k])).real
    P[0] -= 1.0
    return P


def axis_extent(coeffs: np.ndarray, direction: complex, max_t: float = 1e6) -> float:
    """Largest t such that |R(t' * direction)| <= 1 for all 0 <= t' <= t, from the roots of |R|^2 - 1 along the
    axis: zero when the method leaves the unit disc immediately (the lowest-order term is positive), else the first
    positive root after which the polynomial is positive (a root that only touches zero does not end the region)."""
    P = extent_polynomial(coeffs, direction)
    scale = max(1.0, float(np.max(np.abs(P))))
    nz = [i for i, p in enumerate(P) if abs(p) > 1e-12 * scale]
    if not nz:
        return float(max_t)
    if P[nz[0]] > 0:
        return 0.0
    roots = np.roots(P[::-1])
    reals = sorted(float(r.real) for r in roots if abs(r.imag) <= 1e-7 * max(1.0, abs(r)) and r.real > 1e-9)
    for t in reals:
        probe = t * (1 + 1e-6) + 1e-9
        if float(np.polyval(P[::-1], probe)) > 0:
            return t
    return float(max_t)


def region_area(coeffs: np.ndarray, box: float = 5.0, n: int = 801) -> float:
    xs = np.linspace(-box, box, n)
    X, Y = np.meshgrid(xs, xs)
    inside = np.abs(R_of(coeffs, X + 1j * Y)) <= 1.0
    cell = (2 * box / (n - 1)) ** 2
    return float(inside.sum() * cell)


# --- order conditions ------------------------------------------------------------------------
def order_conditions(A: np.ndarray, b: np.ndarray, c: np.ndarray) -> dict[int, list[float]]:
    """Residuals of the classical order conditions, grouped by order (up to four)."""
    Ac = A @ c
    return {
        1: [float(b.sum() - 1.0)],
        2: [float(b @ c - 0.5)],
        3: [float(b @ c ** 2 - 1 / 3), float(b @ Ac - 1 / 6)],
        4: [float(b @ c ** 3 - 0.25), float((b * c) @ Ac - 1 / 8), float(b @ (A @ c ** 2) - 1 / 12),
            float(b @ (A @ Ac) - 1 / 24)],
    }


def classical_order(A: np.ndarray, b: np.ndarray, c: np.ndarray, tol: float = 1e-10) -> int:
    conds = order_conditions(A, b, c)
    p = 0
    for order in (1, 2, 3, 4):
        if all(abs(r) <= tol for r in conds[order]):
            p = order
        else:
            break
    return p


# --- SSP ---------------------------------------------------------------------------------------
def absolutely_monotone(A: np.ndarray, b: np.ndarray, r: float, tol: float = 1e-11) -> bool:
    s = A.shape[0]
    K = np.zeros((s + 1, s + 1))
    K[:s, :s] = A
    K[s, :s] = b
    M = np.eye(s + 1) + r * K
    try:
        Minv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return False
    return bool(np.all(K @ Minv >= -tol) and np.all(Minv @ np.ones(s + 1) >= -tol))


def ssp_coefficient(A: np.ndarray, b: np.ndarray, r_max: float = 1000.0, iters: int = 80) -> float:
    """Radius of absolute monotonicity R(K): the largest r with K (I + rK)^-1 >= 0 and (I + rK)^-1 1 >= 0."""
    if not absolutely_monotone(A, b, 1e-9):
        return 0.0
    if absolutely_monotone(A, b, r_max):
        return float(r_max)
    lo, hi = 0.0, r_max
    for _ in range(iters):
        mid = (lo + hi) / 2
        if absolutely_monotone(A, b, mid):
            lo = mid
        else:
            hi = mid
    return float(lo)


def shu_osher_coefficient(alpha: np.ndarray, beta: np.ndarray) -> float:
    """The SSP coefficient of the given representation: min alpha_ik / beta_ik over beta_ik > 0, zero if a
    coefficient is negative."""
    if np.any(alpha < -1e-14) or np.any(beta < -1e-14):
        return 0.0
    ratios = [alpha[i, k] / beta[i, k] for i in range(alpha.shape[0]) for k in range(alpha.shape[1]) if beta[i, k] > 1e-14]
    return float(min(ratios)) if ratios else 0.0


class StabilityCertifier(Certifier):
    """kind: ssp (score = SSP coefficient), real_axis (score = real-axis extent), imag_axis (score = imaginary-axis
    extent); higher is better. Pass iff the tableau is an explicit method of the claimed order (and stage count),
    its polynomial agrees with the method on seeded test problems, and for ssp the coefficient is positive."""

    name = "stability_region_ssp"
    version = "1.0.0"
    runtime_cap_s = 30.0

    def __init__(self, kind: str = "ssp", order: int = 1, *, stages: int | None = None, best_known: float | None = None,
                 tol: float = 1e-10, n_random: int = 32, min_coefficient: float = 1e-9):
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        self.kind = kind
        self.order = int(order)
        self.stages = stages
        self.best_known = best_known
        self.tol = float(tol)
        self.n_random = int(n_random)
        self.min_coefficient = float(min_coefficient)

    def run(self, candidate: Any, rng, *, target, seed, **kw) -> Certificate:
        m = normalize_method(candidate)
        A, b, c, s = m["A"], m["b"], m["c"], m["stages"]
        if not m["explicit"]:
            return Certificate(self.name, self.version, target, False, score=None, lower_is_better=False,
                               error="only explicit methods are certified (A must be strictly lower triangular)",
                               details={"stages": s})
        coeffs = stability_polynomial(A, b)
        # the polynomial must agree with one step of the method on u' = z u at seeded points
        zs = rng.uniform(-4, 4, size=self.n_random) + 1j * rng.uniform(-4, 4, size=self.n_random)
        max_poly_err = max(abs(step_test_equation(A, b, z) - R_of(coeffs, z)) for z in zs)
        poly_ok = bool(max_poly_err <= 1e-9 * max(1.0, float(np.max(np.abs(coeffs)))))
        real_ext = axis_extent(coeffs, -1.0)
        imag_ext = axis_extent(coeffs, 1j)
        area = region_area(coeffs)
        p = classical_order(A, b, c, self.tol)
        conds = order_conditions(A, b, c)
        C = ssp_coefficient(A, b)
        C_rep = shu_osher_coefficient(m["alpha"], m["beta"]) if m["alpha"] is not None else None
        # sanity: seeded points on the certified real interval are inside the region
        xs = rng.uniform(-real_ext, 0.0, size=self.n_random) if real_ext > 0 else np.zeros(1)
        inside_ok = bool(np.all(np.abs(R_of(coeffs, xs)) <= 1.0 + 1e-9))
        checks = {
            "explicit": True,
            "order_conditions_to_claimed_order": p >= self.order,
            "stage_count": (self.stages is None) or (s == self.stages),
            "polynomial_matches_method": poly_ok,
            "real_interval_inside_region": inside_ok,
        }
        if self.kind == "ssp":
            checks["ssp_coefficient_positive"] = C > self.min_coefficient
        checks = {k: bool(v) for k, v in checks.items()}
        passed = all(checks.values())
        score = {"ssp": C, "real_axis": real_ext, "imag_axis": imag_ext}[self.kind]
        details = {
            "kind": self.kind, "stages": s, "claimed_order": self.order, "measured_order": p,
            "order_condition_residuals": {str(k): v for k, v in conds.items()},
            "stability_polynomial": coeffs.tolist(), "real_axis_extent": real_ext, "imag_axis_extent": imag_ext,
            "region_area": area, "ssp_coefficient": C,
            "shu_osher_coefficient": C_rep, "max_polynomial_mismatch": float(max_poly_err),
            "A": A.tolist(), "b": b.tolist(), "c": c.tolist(), "checks": checks, "best_known": self.best_known, "tol": self.tol,
        }
        optimal = None
        if self.best_known is not None:
            optimal = bool(passed and score >= self.best_known - 1e-6)
        return Certificate(self.name, self.version, target, passed, score=float(score), lower_is_better=False,
                           optimal=optimal, details=details)


# --- known methods (tests and the scoreboard) ----------------------------------------------------
def butcher(name: str) -> dict:
    """Butcher tableaux of classical explicit methods."""
    t = {
        "euler": ([[0.0]], [1.0]),
        "heun2": ([[0, 0], [1, 0]], [0.5, 0.5]),
        "midpoint2": ([[0, 0], [0.5, 0]], [0.0, 1.0]),
        "ralston2": ([[0, 0], [2 / 3, 0]], [0.25, 0.75]),
        "kutta3": ([[0, 0, 0], [0.5, 0, 0], [-1, 2, 0]], [1 / 6, 2 / 3, 1 / 6]),
        "heun3": ([[0, 0, 0], [1 / 3, 0, 0], [0, 2 / 3, 0]], [0.25, 0.0, 0.75]),
        "rk4": ([[0, 0, 0, 0], [0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 1, 0]], [1 / 6, 1 / 3, 1 / 3, 1 / 6]),
        "rk38": ([[0, 0, 0, 0], [1 / 3, 0, 0, 0], [-1 / 3, 1, 0, 0], [1, -1, 1, 0]], [1 / 8, 3 / 8, 3 / 8, 1 / 8]),
    }
    if name not in t:
        raise KeyError(f"unknown method {name!r}; known: {sorted(t)}")
    A, b = t[name]
    return {"A": [list(map(float, r)) for r in A], "b": list(map(float, b))}


def shu_osher(name: str, s: int | None = None) -> dict:
    """Shu-Osher forms of SSP methods. ssprk_s2 (s stages, order 2, coefficient s-1) and ssprk_n2_3
    (n^2 stages, order 3, coefficient n^2 - n) take the stage count."""
    if name == "ssprk22":
        alpha = [[1, 0], [0.5, 0.5]]
        beta = [[1, 0], [0, 0.5]]
    elif name == "ssprk33":
        alpha = [[1, 0, 0], [3 / 4, 1 / 4, 0], [1 / 3, 0, 2 / 3]]
        beta = [[1, 0, 0], [0, 1 / 4, 0], [0, 0, 2 / 3]]
    elif name == "ssprk43":
        alpha = [[1, 0, 0, 0], [0, 1, 0, 0], [2 / 3, 0, 1 / 3, 0], [0, 0, 0, 1]]
        beta = [[0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 1 / 6, 0], [0, 0, 0, 0.5]]
    elif name == "ssprk54":
        # Spiteri & Ruuth (2002), SIAM J. Numer. Anal. 40, the optimal five-stage fourth-order method
        alpha = [[1.0, 0, 0, 0, 0],
                 [0.444370493651235, 0.555629506348765, 0, 0, 0],
                 [0.620101851488403, 0, 0.379898148511597, 0, 0],
                 [0.178079954393132, 0, 0, 0.821920045606868, 0],
                 [0, 0, 0.517231671970585, 0.096059710526147, 0.386708617503269]]
        beta = [[0.391752226571890, 0, 0, 0, 0],
                [0, 0.368410593050371, 0, 0, 0],
                [0, 0, 0.251891774271694, 0, 0],
                [0, 0, 0, 0.544974750228521, 0],
                [0, 0, 0, 0.063692468666290, 0.226007483236906]]
    elif name == "ssprk_s2":
        if s is None or s < 2:
            raise ValueError("ssprk_s2 needs s >= 2")
        alpha = np.zeros((s, s))
        beta = np.zeros((s, s))
        for i in range(s - 1):               # Euler steps of size dt/(s-1)
            alpha[i, i] = 1.0
            beta[i, i] = 1.0 / (s - 1)
        alpha[s - 1, 0] = 1.0 / s            # last stage: a convex combination with u^n
        alpha[s - 1, s - 1] = (s - 1) / s
        beta[s - 1, s - 1] = 1.0 / s
        alpha, beta = alpha.tolist(), beta.tolist()
    elif name == "ssprk_n2_3":
        # Ketcheson (2008): n^2 Euler steps of size dt/(n^2 - n), except stage n(n+1)/2, a convex combination
        # of the step with u^((n-1)(n-2)/2); coefficient n^2 - n, order 3
        n = s
        if n is None or n < 2:
            raise ValueError("ssprk_n2_3 takes n >= 2 (n^2 stages)")
        S = n * n
        alpha = np.zeros((S, S))
        beta = np.zeros((S, S))
        r = n * n - n
        special = n * (n + 1) // 2
        for i in range(1, S + 1):
            prev = i - 1
            if i == special:
                alpha[i - 1, (n - 1) * (n - 2) // 2] = n / (2 * n - 1)
                alpha[i - 1, prev] += (n - 1) / (2 * n - 1)
                beta[i - 1, prev] = (n - 1) / ((2 * n - 1) * r)
            else:
                alpha[i - 1, prev] = 1.0
                beta[i - 1, prev] = 1.0 / r
        alpha, beta = alpha.tolist(), beta.tolist()
    else:
        raise KeyError(f"unknown SSP method {name!r}")
    return {"alpha": alpha, "beta": beta}


# tabulated values (Hairer, Norsett & Wanner 1993; Gottlieb, Shu & Tadmor 2001; Spiteri & Ruuth 2002; Ketcheson 2008)
TABULATED = {
    "euler": {"order": 1, "real": 2.0, "imag": 0.0, "ssp": 1.0},
    "heun2": {"order": 2, "real": 2.0, "imag": 0.0, "ssp": 1.0},
    "kutta3": {"order": 3, "real": 2.5127453266183, "imag": math.sqrt(3.0), "ssp": 0.0},
    "rk4": {"order": 4, "real": 2.7852935634052, "imag": 2.0 * math.sqrt(2.0), "ssp": 0.0},
    "ssprk22": {"order": 2, "ssp": 1.0},
    "ssprk33": {"order": 3, "ssp": 1.0},
    "ssprk43": {"order": 3, "ssp": 2.0},
    "ssprk54": {"order": 4, "ssp": 1.508},
}
