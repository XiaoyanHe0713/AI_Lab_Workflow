"""Worst-case rate by performance estimation (PEP), through PEPit with CVXPY and SCS.

For a named first-order method on a stated function class the certifier builds the performance
estimation SDP, solves it, and returns the tight worst-case rate together with the dual certificate:
the multipliers of the interpolation inequalities, the dual slack matrix (which must be positive
semidefinite), the cancellation residual of the function-value terms, and the bound reconstructed
from the multipliers alone (which must equal the primal value). The scoreboard holds the published
rate for each method; the certificate compares to it within a stated tolerance.

Methods (parameters, metric, published rate):
  gradient_descent            mu, L, gamma, n     ||x_n - x*||^2 / ||x_0 - x*||^2
                              max(|1 - gamma mu|, |1 - gamma L|)^(2n), tight (attained on quadratics)
  nesterov_strongly_convex    mu, L, n            (f(x_n) - f*) / (f(x_0) - f* + mu/2 ||x_0 - x*||^2)
                              (1 - sqrt(mu/L))^n, an upper bound (d'Aspremont, Scieur, Taylor 2021, Cor. 4.15)
  douglas_rachford            mu, L, alpha, theta, n   ||w_n - w_n'||^2 / ||w_0 - w_0'||^2
                              theta = 1: max(1/(1 + mu alpha), alpha L/(1 + alpha L))^(2n), tight
                              (Giselsson & Boyd 2017, Thm 2; Ryu, Taylor, Bergeling, Giselsson 2020)
  davis_yin                   mu1, L1, L3, alpha, theta, n   ||w_n - w_n'||^2 / ||w_0 - w_0'||^2
                              no closed form published; the calibration point is PEPit's reference value
                              0.4754523346392658 at mu1=0.1, L1=10, L3=1, alpha=1, theta=1, n=4
                              (PEPit 0.5.1 example, solved with MOSEK), plus the Davis-Yin
                              quadratic-instance lower bound computed here.
Every solve is deterministic (SCS is deterministic for fixed settings), capped at `cap_s` seconds in a
separate process, and cached by method signature.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from .base import Certificate, Certifier, run_with_cap
from .witness import _quadratic_ratio

SOLVER_DEFAULTS = {"SCS": {"eps_abs": 1e-9, "eps_rel": 1e-9, "max_iters": 200_000},
                   "CLARABEL": {}, "MOSEK": {}}


def published_rate(method: str, params: dict) -> tuple[float | None, bool, str]:
    """(rate, tight, source) from the published formulas, or (None, False, note) when no closed form exists."""
    p = params
    if method == "gradient_descent":
        r = max(abs(1 - p["gamma"] * p["mu"]), abs(1 - p["gamma"] * p["L"])) ** (2 * p["n"])
        return r, True, "max(|1-gamma mu|,|1-gamma L|)^(2n); quadratic worst case (Taylor, Hendrickx, Glineur 2017)"
    if method == "nesterov_strongly_convex":
        r = (1 - math.sqrt(p["mu"] / p["L"])) ** p["n"]
        return r, False, "(1-sqrt(mu/L))^n upper bound; d'Aspremont, Scieur, Taylor 2021, Cor. 4.15"
    if method == "douglas_rachford":
        if p["theta"] == 1:
            r = max(1 / (1 + p["mu"] * p["alpha"]), p["alpha"] * p["L"] / (1 + p["alpha"] * p["L"])) ** (2 * p["n"])
            return r, True, "max(1/(1+mu alpha), alpha L/(1+alpha L))^(2n); Giselsson & Boyd 2017, Thm 2 (tight)"
        return None, False, "published closed form only for theta = 1"
    if method == "davis_yin":
        return None, False, "no closed-form published rate; calibrated against PEPit's reference value"
    raise ValueError(f"unknown method {method!r}")


METHODS = {
    "gradient_descent": {"params": ("mu", "L", "gamma", "n"), "metric": "||x_n - x*||^2 / ||x_0 - x*||^2"},
    "nesterov_strongly_convex": {"params": ("mu", "L", "n"),
                                 "metric": "(f(x_n) - f*) / (f(x_0) - f* + mu/2 ||x_0 - x*||^2)"},
    "douglas_rachford": {"params": ("mu", "L", "alpha", "theta", "n"), "metric": "||w_n - w_n'||^2 / ||w_0 - w_0'||^2"},
    "davis_yin": {"params": ("mu1", "L1", "L3", "alpha", "theta", "n"), "metric": "||w_n - w_n'||^2 / ||w_0 - w_0'||^2"},
}

# PEPit 0.5.1 example docstrings, solved with MOSEK: the numerical reference values used for calibration.
PEPIT_REFERENCE = [
    ("nesterov_strongly_convex", {"L": 1, "mu": 0.1, "n": 2}, 0.3476022263165957),
    ("douglas_rachford", {"L": 1, "alpha": 3, "mu": 0.1, "n": 2, "theta": 1}, 0.35012780295429113),
    ("davis_yin", {"L1": 10, "L3": 1, "alpha": 1, "mu1": 0.1, "n": 4, "theta": 1}, 0.4754523346392658),
]


def pepit_reference(method: str, params: dict) -> float | None:
    for m, p, v in PEPIT_REFERENCE:
        if m == method and all(math.isclose(float(params[k]), float(p[k]), rel_tol=1e-12, abs_tol=1e-12) for k in p):
            return v
    return None


def _norm_params(method: str, params: dict) -> dict:
    spec = METHODS[method]
    out = {}
    for k in spec["params"]:
        if k not in params:
            raise ValueError(f"{method} needs parameter {k!r}")
        v = params[k]
        out[k] = int(v) if k == "n" else float(v)
    if out["n"] < 1:
        raise ValueError("n must be >= 1")
    return out


def _build(method: str, p: dict):
    """Build the PEP for a method; returns the PEPit problem."""
    from PEPit import PEP
    from PEPit.functions import ConvexFunction, SmoothConvexFunction, SmoothStronglyConvexFunction
    from PEPit.primitive_steps import proximal_step

    problem = PEP()
    if method == "gradient_descent":
        func = problem.declare_function(SmoothStronglyConvexFunction, mu=p["mu"], L=p["L"])
        xs = func.stationary_point()
        x0 = problem.set_initial_point()
        problem.set_initial_condition((x0 - xs) ** 2 <= 1)
        x = x0
        for _ in range(p["n"]):
            x = x - p["gamma"] * func.gradient(x)
        problem.set_performance_metric((x - xs) ** 2)
    elif method == "nesterov_strongly_convex":
        func = problem.declare_function(SmoothStronglyConvexFunction, mu=p["mu"], L=p["L"])
        xs = func.stationary_point()
        fs = func(xs)
        x0 = problem.set_initial_point()
        problem.set_initial_condition(func(x0) - fs + p["mu"] / 2 * (x0 - xs) ** 2 <= 1)
        kappa = p["mu"] / p["L"]
        beta = (1 - math.sqrt(kappa)) / (1 + math.sqrt(kappa))
        x_new = x0
        y = x0
        for _ in range(p["n"]):
            x_old = x_new
            x_new = y - 1 / p["L"] * func.gradient(y)
            y = x_new + beta * (x_new - x_old)
        problem.set_performance_metric(func(x_new) - fs)
    elif method == "douglas_rachford":
        f1 = problem.declare_function(SmoothStronglyConvexFunction, mu=p["mu"], L=p["L"])
        f2 = problem.declare_function(ConvexFunction)
        w0, w0p = problem.set_initial_point(), problem.set_initial_point()
        problem.set_initial_condition((w0 - w0p) ** 2 <= 1)
        w, wp = w0, w0p
        for _ in range(p["n"]):
            x, _, _ = proximal_step(w, f2, p["alpha"])
            y, _, _ = proximal_step(2 * x - w, f1, p["alpha"])
            w = w + p["theta"] * (y - x)
            xp, _, _ = proximal_step(wp, f2, p["alpha"])
            yp, _, _ = proximal_step(2 * xp - wp, f1, p["alpha"])
            wp = wp + p["theta"] * (yp - xp)
        problem.set_performance_metric((w - wp) ** 2)
    elif method == "davis_yin":
        f1 = problem.declare_function(SmoothStronglyConvexFunction, mu=p["mu1"], L=p["L1"])
        f2 = problem.declare_function(ConvexFunction)
        f3 = problem.declare_function(SmoothConvexFunction, L=p["L3"])
        w0, w0p = problem.set_initial_point(), problem.set_initial_point()
        problem.set_initial_condition((w0 - w0p) ** 2 <= 1)
        w, wp = w0, w0p
        for _ in range(p["n"]):
            x, _, _ = proximal_step(w, f2, p["alpha"])
            gx, _ = f3.oracle(x)
            y, _, _ = proximal_step(2 * x - w - p["alpha"] * gx, f1, p["alpha"])
            w = w + p["theta"] * (y - x)
            xp, _, _ = proximal_step(wp, f2, p["alpha"])
            gxp, _ = f3.oracle(xp)
            yp, _, _ = proximal_step(2 * xp - wp - p["alpha"] * gxp, f1, p["alpha"])
            wp = wp + p["theta"] * (yp - xp)
        problem.set_performance_metric((w - wp) ** 2)
    else:
        raise ValueError(method)
    return problem


def _dual_certificate(problem) -> dict:
    """Rebuild the bound from the multipliers alone and check the dual feasibility conditions.

    Each constraint sent to the solver reads Tr(A_i G) + a_i^T F + c_i <= 0 (or = 0) with multiplier
    lambda_i. The dual certificate is: S = sum_i lambda_i A_i is PSD, sum_i lambda_i a_i equals the
    objective's F-weight (the function-value terms cancel), and tau_dual = -sum_i lambda_i c_i.
    """
    from PEPit.tools.expressions_to_matrices import expression_to_matrices

    cons = list(problem._list_of_constraints_sent_to_wrapper)
    G_obj, F_obj, c_obj = expression_to_matrices(problem.objective)
    S = np.zeros_like(G_obj)
    a_sum = np.zeros_like(F_obj)
    tau_dual = 0.0
    duals = []
    min_ineq_dual = math.inf
    for i, con in enumerate(cons):
        lam = con.eval_dual()
        lam = float(np.asarray(lam).reshape(-1)[0]) if np.ndim(lam) else float(lam)
        A, a, c = expression_to_matrices(con.expression)
        S += lam * A
        a_sum += lam * a
        tau_dual -= lam * c
        kind = con.equality_or_inequality
        if kind == "inequality":
            min_ineq_dual = min(min_ineq_dual, lam)
        duals.append({"index": i, "kind": kind, "value": lam})
    eig_min = float(np.min(np.linalg.eigvalsh((S + S.T) / 2))) if S.size else 0.0
    f_residual = float(np.max(np.abs(F_obj - a_sum))) if F_obj.size else 0.0
    return {
        "tau_dual": float(tau_dual),
        "S_min_eigenvalue": eig_min,
        "f_residual": f_residual,
        "min_inequality_dual": float(min_ineq_dual) if cons else 0.0,
        "n_constraints": len(cons),
        "multipliers": duals,
    }


def _solve_pep(*, method: str, params: dict, solver: str, solver_opts: dict, time_limit_s: float | None) -> dict:
    """Module-level worker so it can run under run_with_cap."""
    import cvxpy as cp

    p = _norm_params(method, params)
    problem = _build(method, p)
    opts = dict(SOLVER_DEFAULTS.get(solver.upper(), {}))
    opts.update(solver_opts or {})
    if solver.upper() == "SCS" and time_limit_s:
        opts.setdefault("time_limit_secs", float(time_limit_s))
    t0 = time.perf_counter()
    tau_primal = problem.solve(wrapper="cvxpy", solver=getattr(cp, solver.upper()), verbose=0,
                               return_primal_or_dual="primal", **opts)
    solve_s = time.perf_counter() - t0
    if tau_primal is None:
        raise RuntimeError("the SDP is unbounded or the solver failed")
    status = str(problem.wrapper.prob.status)
    residual_eig = float(np.min(np.linalg.eigvalsh(problem.residual))) if problem.residual is not None else None
    cert = _dual_certificate(problem)
    return {"tau_primal": float(tau_primal), "status": status, "solve_s": solve_s,
            "pepit_residual_min_eigenvalue": residual_eig, "gram_size": int(problem.G_value.shape[0]), **cert}


def method_signature(method: str, params: dict, solver: str, version: str) -> str:
    p = _norm_params(method, params)
    blob = json.dumps({"method": method, "params": p, "solver": solver.upper(), "version": version}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


class PEPRateCertifier(Certifier):
    """Candidate: {"method": name, "params": {...}} (n inside params). Score: the certified rate (lower is better)."""

    name = "pep_worst_case_rate"
    version = "1.0.0"
    runtime_cap_s = 60.0

    def __init__(self, *, solver: str = "SCS", tol: float = 1e-4, cap_s: float = 60.0,
                 cache_dir: str | Path | None = None, in_process: bool = False, solver_opts: dict | None = None,
                 published: float | None = None, reference: float | None = None):
        self.solver = solver
        self.tol = float(tol)
        self.runtime_cap_s = float(cap_s)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.in_process = in_process
        self.solver_opts = dict(solver_opts or {})
        self.published_override = published
        self.reference_override = reference

    def run(self, candidate, rng, *, target, seed, **kw) -> Certificate:
        method = candidate["method"]
        params = _norm_params(method, candidate.get("params", candidate))
        sig = method_signature(method, params, self.solver, self.version)
        cached = False
        result = None
        if self.cache_dir is not None:
            path = self.cache_dir / f"{sig}.json"
            if path.exists():
                result = json.loads(path.read_text(encoding="utf-8"))
                cached = True
        if result is None:
            kwargs = dict(method=method, params=params, solver=self.solver, solver_opts=self.solver_opts,
                          time_limit_s=self.runtime_cap_s)
            if self.in_process:
                result = _solve_pep(**kwargs)
            else:
                result = run_with_cap(_solve_pep, kwargs, self.runtime_cap_s)
            if self.cache_dir is not None:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                (self.cache_dir / f"{sig}.json").write_text(json.dumps(result), encoding="utf-8")

        tau = result["tau_primal"]
        tau_dual = result["tau_dual"]
        pub, tight, source = published_rate(method, params)
        if self.published_override is not None:
            pub = float(self.published_override)
        ref = self.reference_override
        if ref is None:
            ref = pepit_reference(method, params)

        # the quadratic instances with curvature mu and L are admissible, so the certified worst case
        # can never fall below their ratio: a lower bound every method must respect
        lo = _instance_lower_bound(method, params)

        checks = {
            "solver_optimal": result["status"].startswith("optimal"),
            "dual_matches_primal": abs(tau_dual - tau) <= self.tol * max(1.0, abs(tau)),
            "dual_slack_psd": result["S_min_eigenvalue"] >= -self.tol,
            "function_terms_cancel": result["f_residual"] <= self.tol,
            "multipliers_nonnegative": result["min_inequality_dual"] >= -self.tol,
            "above_instance_lower_bound": tau >= lo - self.tol,
        }
        if pub is not None:
            checks["within_published_upper_bound"] = tau <= pub * (1 + self.tol) + self.tol
            if tight:
                checks["matches_published_tight_rate"] = abs(tau - pub) <= self.tol * max(1.0, pub)
        if ref is not None:
            checks["matches_pepit_reference"] = abs(tau - ref) <= self.tol * max(1.0, ref)
        checks = {k: bool(v) for k, v in checks.items()}
        passed = all(checks.values())
        details = {
            "method": method, "params": params, "metric": METHODS[method]["metric"],
            "rate": tau, "rate_dual": tau_dual, "published_rate": pub, "published_tight": tight,
            "published_source": source, "pepit_reference": ref, "instance_lower_bound": lo,
            "tol": self.tol, "solver": self.solver.upper(), "solver_status": result["status"],
            "solve_s": result["solve_s"], "gram_size": result["gram_size"], "cached": cached,
            "signature": sig, "checks": checks,
            "dual_certificate": {k: result[k] for k in ("tau_dual", "S_min_eigenvalue", "f_residual",
                                                        "min_inequality_dual", "n_constraints", "multipliers",
                                                        "pepit_residual_min_eigenvalue")},
        }
        return Certificate(self.name, self.version, target, passed, score=float(tau), lower_is_better=True,
                           details=details)


def _instance_lower_bound(method: str, params: dict) -> float:
    n = params["n"]
    if method == "davis_yin":
        q = {"mu": params["mu1"], "L": params["L1"], "alpha": params["alpha"], "theta": params["theta"]}
        return float(max(_quadratic_ratio("davis_yin", q, params["mu1"], n),
                         _quadratic_ratio("davis_yin", q, params["L1"], n)))
    return float(max(_quadratic_ratio(method, params, params["mu"], n),
                     _quadratic_ratio(method, params, params["L"], n)))
