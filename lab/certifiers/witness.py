"""Witness check: re-run a claimed break and report whether the failure reproduces.

The breaker's reward is one for a break the witness check confirms and zero for anything else,
including an instance outside the class definition. A witness names the family, the input and the
kind of failure claimed. The check re-runs the candidate on exactly that input under the family's
own certifier criterion and says `reproduces` or not; the claimed magnitude is informational.

Witness kinds:
  * quadrature / exactness_error: the rule integrates the given monomial (exponents) or polynomial
    (basis coefficients) with error above the certifier tolerance;
  * quadrature / exception: evaluating the rule on the input raises;
  * first_order / rate_violation: on the quadratic f(x) = c/2 ||x||^2 with the given curvature c in
    [mu, L], the method contracts the stated distance by more than the claimed rate.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .base import Certificate, Certifier
from .quadrature import DIM, apply_rule, monomial_exact, normalize_rule


def _quadratic_ratio(method: str, params: dict, curvature: float, n: int) -> float:
    """Contraction ratio of a method on f(x) = c/2 x^2 (scalar, exact arithmetic in floats)."""
    c = float(curvature)
    if method == "gradient_descent":
        gamma = float(params["gamma"])
        x = 1.0
        for _ in range(n):
            x = x - gamma * c * x
        return x * x
    if method == "nesterov_strongly_convex":
        mu, L = float(params["mu"]), float(params["L"])
        kappa = mu / L
        beta = (1 - np.sqrt(kappa)) / (1 + np.sqrt(kappa))
        x_old = x = 1.0
        y = 1.0
        for _ in range(n):
            x_new = y - (1 / L) * c * y
            y = x_new + beta * (x_new - x_old)
            x_old, x = x, x_new
        # metric f(x_n) - f* over f(x_0) - f* + mu/2 ||x_0||^2
        return (c / 2 * x * x) / (c / 2 + mu / 2)
    if method in ("douglas_rachford", "davis_yin"):
        alpha, theta = float(params["alpha"]), float(params["theta"])
        # f2 = 0 (prox is identity), f1 = c/2 x^2 (prox_{alpha f1}(v) = v / (1 + alpha c)), f3 = 0
        w = 1.0
        for _ in range(n):
            x = w
            y = (2 * x - w) / (1 + alpha * c)
            w = w + theta * (y - x)
        return w * w
    raise ValueError(f"unknown method {method!r}")


class WitnessCertifier(Certifier):
    name = "witness_check"
    version = "1.0.0"
    runtime_cap_s = 30.0

    def __init__(self, tol: float = 1e-9, rate_tol: float = 1e-6):
        self.tol = float(tol)
        self.rate_tol = float(rate_tol)

    def run(self, candidate: Any, rng, *, target, seed, witness: dict | None = None, **kw) -> Certificate:
        if witness is None:
            raise ValueError("a witness check needs a witness")
        family = witness.get("family")
        kind = witness.get("kind")
        if family == "quadrature":
            return self._quadrature(candidate, witness, target, kind)
        if family == "first_order":
            return self._first_order(candidate, witness, target, kind)
        return Certificate(self.name, self.version, target, False, score=0.0,
                           error=f"instance outside the class definition: family {family!r}", witness=witness)

    def _quadrature(self, candidate, witness, target, kind) -> Certificate:
        domain = witness["domain"]
        if domain not in DIM:
            return Certificate(self.name, self.version, target, False, score=0.0,
                               error=f"instance outside the class definition: domain {domain!r}", witness=witness)
        try:
            nodes, weights = normalize_rule(candidate, domain)
        except Exception as e:
            reproduces = kind == "exception"
            return Certificate(self.name, self.version, target, reproduces, score=float(reproduces),
                               details={"reproduces": reproduces, "error_raised": f"{type(e).__name__}: {e}"},
                               witness=witness)
        inp = witness.get("input", {})
        degree = witness.get("degree")
        if "exponents" in inp:
            alphas = {tuple(int(a) for a in inp["exponents"]): 1.0}
        elif "coefficients" in inp:
            alphas = {tuple(int(a) for a in k.split(",")): float(v) for k, v in inp["coefficients"].items()}
        else:
            return Certificate(self.name, self.version, target, False, score=0.0,
                               error="witness input needs exponents or coefficients", witness=witness)
        if degree is not None and any(sum(a) > int(degree) for a in alphas):
            return Certificate(self.name, self.version, target, False, score=0.0,
                               error="instance outside the class definition: degree above the target degree",
                               witness=witness)
        if any(len(a) != DIM[domain] for a in alphas):
            return Certificate(self.name, self.version, target, False, score=0.0,
                               error="instance outside the class definition: wrong dimension", witness=witness)
        try:
            got = sum(c * apply_rule(nodes, weights, a) for a, c in alphas.items())
            exact = sum(c * monomial_exact(domain, a) for a, c in alphas.items())
        except Exception as e:
            reproduces = kind == "exception"
            return Certificate(self.name, self.version, target, reproduces, score=float(reproduces),
                               details={"reproduces": reproduces, "error_raised": f"{type(e).__name__}: {e}"},
                               witness=witness)
        err = abs(got - exact)
        if kind == "exception":
            reproduces = False  # it evaluated fine
        else:
            reproduces = bool(err > self.tol)
        return Certificate(self.name, self.version, target, reproduces, score=float(reproduces),
                           details={"reproduces": reproduces, "error": err, "tol": self.tol,
                                    "rule_value": got, "exact_value": exact}, witness=witness)

    def _first_order(self, candidate, witness, target, kind) -> Certificate:
        if kind != "rate_violation":
            return Certificate(self.name, self.version, target, False, score=0.0,
                               error=f"unknown witness kind {kind!r}", witness=witness)
        method = candidate["method"]
        params = dict(candidate.get("params", {}))
        n = int(candidate.get("n", 1))
        claimed = float(witness["claimed_rate"])
        c = float(witness["curvature"])
        mu, L = float(params.get("mu", 0.0)), float(params.get("L", 1.0))
        if not (mu - 1e-12 <= c <= L + 1e-12):
            return Certificate(self.name, self.version, target, False, score=0.0,
                               error="instance outside the class definition: curvature not in [mu, L]",
                               witness=witness)
        ratio = _quadratic_ratio(method, params, c, n)
        reproduces = bool(ratio > claimed * (1 + self.rate_tol) + self.rate_tol)
        return Certificate(self.name, self.version, target, reproduces, score=float(reproduces),
                           details={"reproduces": reproduces, "observed_ratio": ratio, "claimed_rate": claimed,
                                    "curvature": c, "iterations": n}, witness=witness)
