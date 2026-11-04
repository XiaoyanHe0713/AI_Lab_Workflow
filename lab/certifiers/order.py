"""Convergence-order certificate: refinement studies in h or in the time step with a slope fit.

The certifier owns the problems and their exact (manufactured) solutions; the candidate only
produces numerical solutions at the refinement levels the certifier chooses. Errors are measured
here against the exact solution, log(error) is fitted against log(h) by least squares, and the
order is reported with a confidence interval from the regression and with the pairwise orders
between consecutive levels. Levels whose error sits at machine precision are dropped from the fit
and reported. The parameters of each manufactured solution are drawn from the seed, so the same
seed gives the same study and a candidate cannot hard-code answers.

Problem families (a candidate's entry point is solve(problem, hs) -> one solution per h):
  ode_ivp       u' = lam (u - cos(omega t)) - omega sin(omega t), u(0) = 1 on [0, 1]; exact u = cos(omega t);
                the solution is u(1); h is the step.
  poisson1d     -u'' = f on [0, 1], u(0) = 0, u(1) = alpha, with u = sin(k pi x) + alpha x^2, so
                f = (k pi)^2 sin(k pi x) - 2 alpha; the solution is u at x_i = i h, i = 0..N; h = 1/N.
  quadrature    I = int_0^1 exp(a x) cos(b x) dx with the closed form; the solution is the composite
                estimate with panel width h.

Reference solvers at the bottom (Euler, Heun, RK4, second-order finite differences with a correct or a
deliberately wrong boundary term, composite rules) exist for the certifier's own tests.
"""
from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

from .base import Certificate, Certifier

FAMILIES = ("ode_ivp", "poisson1d", "quadrature")
DEFAULT_H0 = {"ode_ivp": 0.1, "poisson1d": 0.125, "quadrature": 0.125}


# --- problems and exact solutions ------------------------------------------------------------
def make_problem(family: str, rng: np.random.Generator) -> dict:
    """A seeded manufactured problem; everything the candidate needs is in the dict."""
    if family == "ode_ivp":
        lam = float(rng.uniform(-3.0, -0.5))
        omega = float(rng.uniform(1.0, 3.0))
        return {"family": family, "lam": lam, "omega": omega, "u0": 1.0, "t_end": 1.0,
                "rhs": "lam * (u - cos(omega * t)) - omega * sin(omega * t)"}
    if family == "poisson1d":
        k = int(rng.integers(1, 4))
        alpha = float(rng.uniform(0.5, 2.0))
        return {"family": family, "k": k, "alpha": alpha, "left": 0.0, "right": alpha,
                "source": "(k*pi)**2 * sin(k*pi*x) - 2*alpha", "equation": "-u'' = f on [0,1], u(0)=left, u(1)=right"}
    if family == "quadrature":
        a = float(rng.uniform(0.5, 2.0))
        b = float(rng.uniform(1.0, 4.0))
        return {"family": family, "a": a, "b": b, "integrand": "exp(a*x) * cos(b*x)", "interval": [0.0, 1.0]}
    raise ValueError(f"unknown family {family!r}")


def exact_solution(problem: dict, h: float):
    fam = problem["family"]
    if fam == "ode_ivp":
        return math.cos(problem["omega"] * problem["t_end"])
    if fam == "poisson1d":
        N = int(round(1.0 / h))
        x = np.linspace(0.0, 1.0, N + 1)
        return np.sin(problem["k"] * math.pi * x) + problem["alpha"] * x ** 2
    if fam == "quadrature":
        a, b = problem["a"], problem["b"]

        def F(x):
            return math.exp(a * x) * (a * math.cos(b * x) + b * math.sin(b * x)) / (a * a + b * b)

        return F(1.0) - F(0.0)
    raise ValueError(fam)


def solution_error(problem: dict, h: float, numerical, norm: str = "max") -> float:
    exact = exact_solution(problem, h)
    got = np.asarray(numerical, dtype=float)
    ex = np.asarray(exact, dtype=float)
    if got.shape != ex.shape:
        raise ValueError(f"solution at h={h:g} has shape {got.shape}; expected {ex.shape}")
    if not np.all(np.isfinite(got)):
        raise ValueError(f"solution at h={h:g} is not finite")
    diff = np.abs(got - ex)
    if norm == "max":
        return float(np.max(diff))
    if norm == "l2":
        return float(math.sqrt(h * np.sum(diff ** 2))) if diff.ndim else float(diff)
    raise ValueError(norm)


# --- the fit ------------------------------------------------------------------------------------
def fit_order(hs: list[float], errors: list[float], floor: float = 1e-12, ci: float = 0.95) -> dict:
    """Least-squares slope of log(error) against log(h) with a t-interval, pairwise orders, and the levels
    dropped because their error sat at the floor."""
    from scipy import stats

    pts = [(h, e) for h, e in zip(hs, errors) if e > floor]
    dropped = [(h, e) for h, e in zip(hs, errors) if e <= floor]
    pairwise = []
    for (h1, e1), (h2, e2) in zip(pts[:-1], pts[1:]):
        pairwise.append(math.log(e1 / e2) / math.log(h1 / h2))
    out = {"levels_used": len(pts), "levels_dropped": len(dropped), "dropped": dropped, "pairwise_orders": pairwise,
           "slope": None, "ci": None, "intercept": None, "residual": None, "floor": floor}
    if len(pts) < 3:
        return out
    x = np.log([p[0] for p in pts])
    y = np.log([p[1] for p in pts])
    A = np.vstack([x, np.ones_like(x)]).T
    coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    slope, intercept = float(coef[0]), float(coef[1])
    n = len(pts)
    resid = y - A @ coef
    s2 = float(np.sum(resid ** 2) / (n - 2))
    sxx = float(np.sum((x - x.mean()) ** 2))
    se = math.sqrt(s2 / sxx) if sxx > 0 else float("inf")
    t = float(stats.t.ppf(0.5 + ci / 2, n - 2))
    out.update({"slope": slope, "intercept": intercept, "ci": [slope - t * se, slope + t * se], "stderr": se,
                "residual": float(np.max(np.abs(resid))), "t": t})
    return out


class ConvergenceOrderCertifier(Certifier):
    """Pass iff the measured order reaches the claimed order within tol_order (slope and the interval's lower
    bound); score is the measured order (higher is better)."""

    name = "convergence_order"
    version = "1.0.0"
    runtime_cap_s = 60.0

    def __init__(self, family: str, claimed_order: float, *, levels: int = 5, h0: float | None = None,
                 norm: str = "max", tol_order: float = 0.25, ci_slack: float = 0.25, floor: float = 1e-12,
                 best_known_order: float | None = None):
        if family not in FAMILIES:
            raise ValueError(f"family must be one of {FAMILIES}")
        self.family = family
        self.claimed_order = float(claimed_order)
        self.levels = int(levels)
        self.h0 = float(h0 if h0 is not None else DEFAULT_H0[family])
        self.norm = norm
        self.tol_order = float(tol_order)
        self.ci_slack = float(ci_slack)
        self.floor = float(floor)
        self.best_known_order = best_known_order

    def hs(self) -> list[float]:
        return [self.h0 / 2 ** j for j in range(self.levels)]

    def problem_for(self, seed: int) -> dict:
        return make_problem(self.family, np.random.default_rng(seed))

    def run(self, candidate: Any, rng, *, target, seed, **kw) -> Certificate:
        hs = self.hs()
        problem = self.problem_for(seed)
        solutions = None
        if callable(candidate):
            solutions = [candidate(problem, h) for h in hs]
        elif isinstance(candidate, dict) and "errors" in candidate:
            errs = [float(e) for e in candidate["errors"]]
            hs = [float(h) for h in candidate.get("h", hs)]
            if len(errs) != len(hs):
                raise ValueError("errors and h must have the same length")
        elif isinstance(candidate, (list, tuple)):
            solutions = list(candidate)
            if len(solutions) != len(hs):
                raise ValueError(f"expected {len(hs)} solutions (one per h in {hs}); got {len(solutions)}")
        else:
            raise ValueError("candidate must be a callable, a list of solutions (one per h) or {'h': ..., 'errors': ...}")
        if solutions is not None:
            errs = [solution_error(problem, h, s, self.norm) for h, s in zip(hs, solutions)]
        fit = fit_order(hs, errs, floor=self.floor)
        slope = fit["slope"]
        if slope is None:
            return Certificate(self.name, self.version, target, False, score=None, lower_is_better=False,
                               error=f"too few levels above the error floor to fit ({fit['levels_used']} of {len(hs)})",
                               details={"h": hs, "errors": errs, "problem": problem, "fit": fit,
                                        "claimed_order": self.claimed_order})
        lo = fit["ci"][0]
        meets = slope >= self.claimed_order - self.tol_order
        ci_ok = lo >= self.claimed_order - self.tol_order - self.ci_slack
        passed = bool(meets and ci_ok)
        details = {"h": hs, "errors": errs, "problem": problem, "fit": fit, "measured_order": slope,
                   "order_interval": fit["ci"], "pairwise_orders": fit["pairwise_orders"], "claimed_order": self.claimed_order,
                   "tol_order": self.tol_order, "ci_slack": self.ci_slack, "norm": self.norm, "family": self.family,
                   "checks": {"slope_meets_claim": bool(meets), "interval_meets_claim": bool(ci_ok)},
                   "best_known_order": self.best_known_order}
        optimal = None
        if self.best_known_order is not None:
            optimal = bool(passed and slope >= self.best_known_order - self.tol_order)
        return Certificate(self.name, self.version, target, passed, score=float(slope), lower_is_better=False,
                           optimal=optimal, details=details)


# --- reference solvers (tests only) ------------------------------------------------------------
def _rhs(problem: dict) -> Callable[[float, float], float]:
    lam, om = problem["lam"], problem["omega"]
    return lambda t, u: lam * (u - math.cos(om * t)) - om * math.sin(om * t)


def euler(problem: dict, h: float) -> float:
    f = _rhs(problem)
    N = int(round(problem["t_end"] / h))
    t, u = 0.0, problem["u0"]
    for _ in range(N):
        u = u + h * f(t, u)
        t += h
    return u


def heun(problem: dict, h: float) -> float:
    f = _rhs(problem)
    N = int(round(problem["t_end"] / h))
    t, u = 0.0, problem["u0"]
    for _ in range(N):
        k1 = f(t, u)
        k2 = f(t + h, u + h * k1)
        u = u + h * (k1 + k2) / 2
        t += h
    return u


def rk4(problem: dict, h: float) -> float:
    f = _rhs(problem)
    N = int(round(problem["t_end"] / h))
    t, u = 0.0, problem["u0"]
    for _ in range(N):
        k1 = f(t, u)
        k2 = f(t + h / 2, u + h / 2 * k1)
        k3 = f(t + h / 2, u + h / 2 * k2)
        k4 = f(t + h, u + h * k3)
        u = u + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        t += h
    return u


def poisson_fd(problem: dict, h: float, boundary: str = "exact") -> np.ndarray:
    """Second-order central differences; boundary='wrong' imposes a boundary value off by O(h)."""
    N = int(round(1.0 / h))
    x = np.linspace(0.0, 1.0, N + 1)
    k, alpha = problem["k"], problem["alpha"]
    f = (k * math.pi) ** 2 * np.sin(k * math.pi * x) - 2 * alpha
    left, right = problem["left"], problem["right"]
    if boundary == "wrong":
        right = right + h  # a deliberately wrong boundary term, first order in h
    A = np.zeros((N - 1, N - 1))
    for i in range(N - 1):
        A[i, i] = 2.0 / h ** 2
        if i > 0:
            A[i, i - 1] = -1.0 / h ** 2
        if i < N - 2:
            A[i, i + 1] = -1.0 / h ** 2
    b = f[1:N].copy()
    b[0] += left / h ** 2
    b[-1] += right / h ** 2
    u = np.empty(N + 1)
    u[0], u[-1] = left, right
    u[1:N] = np.linalg.solve(A, b)
    return u


def composite(problem: dict, h: float, rule: str = "trapezoid") -> float:
    a, b = problem["a"], problem["b"]

    def g(x):
        return math.exp(a * x) * math.cos(b * x)

    N = int(round(1.0 / h))
    if rule == "left":
        return sum(g(i * h) for i in range(N)) * h
    if rule == "midpoint":
        return sum(g((i + 0.5) * h) for i in range(N)) * h
    if rule == "trapezoid":
        return (g(0.0) / 2 + sum(g(i * h) for i in range(1, N)) + g(1.0) / 2) * h
    if rule == "simpson":
        return sum(g(i * h) + 4 * g((i + 0.5) * h) + g((i + 1) * h) for i in range(N)) * h / 6
    raise ValueError(rule)
