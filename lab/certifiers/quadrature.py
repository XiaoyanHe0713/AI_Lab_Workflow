"""Exactness certificate for quadrature and cubature rules.

Domains: the interval [-1, 1], the square [-1, 1]^2, the cube [-1, 1]^3 and the reference triangle
with vertices (0,0), (1,0), (0,1). A rule is a set of nodes and weights. The certifier applies the
rule to every monomial up to the stated degree (a complete basis, so exactness on it is exactness
on all polynomials of that degree) and to seeded random polynomials, and reports the largest error
against the exact integral, the node count, and whether all weights are positive and all nodes
lie inside the domain.

Reference rules at the bottom (Gauss-Legendre, Clenshaw-Curtis, tensor products, a Duffy-collapsed
triangle rule and a few classical triangle rules) exist for the certifier's own tests and for the
scoreboard. Candidates never get them: the judge forbids library quadrature in candidate code.
"""
from __future__ import annotations

import itertools
import math
from fractions import Fraction
from typing import Any

import numpy as np

from .base import Certificate, Certifier

DOMAINS = ("interval", "square", "cube", "triangle")
DIM = {"interval": 1, "square": 2, "cube": 3, "triangle": 2}
VOLUME = {"interval": 2.0, "square": 4.0, "cube": 8.0, "triangle": 0.5}


# --- exact integrals ---------------------------------------------------------------------------
def monomial_exact(domain: str, alpha: tuple[int, ...]) -> float:
    """Exact integral of x^alpha over the domain, computed in rational arithmetic."""
    if domain in ("interval", "square", "cube"):
        val = Fraction(1)
        for k in alpha:
            val *= Fraction(2, k + 1) if k % 2 == 0 else Fraction(0)
        return float(val)
    if domain == "triangle":
        a, b = alpha
        return float(Fraction(math.factorial(a) * math.factorial(b), math.factorial(a + b + 2)))
    raise ValueError(f"unknown domain {domain!r}")


def monomials_up_to(degree: int, dim: int) -> list[tuple[int, ...]]:
    """All exponent tuples with total degree <= degree, ordered by degree then lexicographically."""
    out = []
    for d in range(degree + 1):
        for alpha in itertools.product(range(d + 1), repeat=dim):
            if sum(alpha) == d:
                out.append(alpha)
    return out


# --- rules -----------------------------------------------------------------------------------
def normalize_rule(candidate: Any, domain: str) -> tuple[np.ndarray, np.ndarray]:
    """Accept (nodes, weights), {"nodes":..., "weights":...} or {"x":..., "w":...}; return float arrays."""
    if isinstance(candidate, dict):
        nodes = candidate.get("nodes", candidate.get("x"))
        weights = candidate.get("weights", candidate.get("w"))
    else:
        nodes, weights = candidate
    nodes = np.asarray(nodes, dtype=float)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    dim = DIM[domain]
    if dim == 1:
        nodes = nodes.reshape(-1, 1) if nodes.ndim <= 1 else nodes
    if nodes.ndim != 2 or nodes.shape[1] != dim:
        raise ValueError(f"nodes must have shape (n, {dim}) for the {domain}; got {nodes.shape}")
    if weights.shape[0] != nodes.shape[0]:
        raise ValueError(f"{nodes.shape[0]} nodes but {weights.shape[0]} weights")
    if nodes.shape[0] == 0:
        raise ValueError("a rule needs at least one node")
    if not (np.all(np.isfinite(nodes)) and np.all(np.isfinite(weights))):
        raise ValueError("nodes and weights must be finite")
    return nodes, weights


def apply_rule(nodes: np.ndarray, weights: np.ndarray, alpha: tuple[int, ...]) -> float:
    vals = np.ones(nodes.shape[0])
    for j, k in enumerate(alpha):
        if k:
            vals = vals * nodes[:, j] ** k
    return float(np.dot(weights, vals))


def inside_domain(nodes: np.ndarray, domain: str, tol: float = 1e-12) -> bool:
    if domain == "triangle":
        x, y = nodes[:, 0], nodes[:, 1]
        return bool(np.all(x >= -tol) and np.all(y >= -tol) and np.all(x + y <= 1 + tol))
    return bool(np.all(np.abs(nodes) <= 1 + tol))


def exactness_report(nodes: np.ndarray, weights: np.ndarray, domain: str, degree: int,
                     rng: np.random.Generator | None = None, n_random: int = 32) -> dict:
    """Largest error over the monomial basis up to `degree` and over seeded random polynomials."""
    basis = monomials_up_to(degree, DIM[domain])
    errors = {}
    worst, worst_alpha = 0.0, None
    exact_vals = {}
    rule_vals = {}
    for alpha in basis:
        ex = monomial_exact(domain, alpha)
        got = apply_rule(nodes, weights, alpha)
        exact_vals[alpha] = ex
        rule_vals[alpha] = got
        err = abs(got - ex)
        errors[alpha] = err
        if err > worst:
            worst, worst_alpha = err, alpha
    # random polynomials: coefficients in [-1, 1]; both sides are linear in the coefficients, so the
    # error is a seeded random combination of the monomial errors (a check of the same thing by a
    # different path, and the number the proposal asks for)
    max_random = 0.0
    if rng is not None and n_random > 0:
        ex_vec = np.array([exact_vals[a] for a in basis])
        got_vec = np.array([rule_vals[a] for a in basis])
        for _ in range(n_random):
            coeff = rng.uniform(-1.0, 1.0, size=len(basis))
            max_random = max(max_random, abs(float(np.dot(coeff, got_vec - ex_vec))))
    return {
        "degree": degree,
        "basis_size": len(basis),
        "max_monomial_error": worst,
        "worst_monomial": list(worst_alpha) if worst_alpha is not None else None,
        "max_random_polynomial_error": max_random,
        "n_random": n_random if rng is not None else 0,
    }


def exactness_degree(nodes: np.ndarray, weights: np.ndarray, domain: str, max_degree: int = 40,
                     tol: float = 1e-9) -> int:
    """The largest d <= max_degree such that every monomial of degree <= d is integrated within tol."""
    dim = DIM[domain]
    for d in range(max_degree + 1):
        for alpha in itertools.product(range(d + 1), repeat=dim):
            if sum(alpha) != d:
                continue
            if abs(apply_rule(nodes, weights, alpha) - monomial_exact(domain, alpha)) > tol:
                return d - 1
    return max_degree


class QuadratureExactnessCertifier(Certifier):
    """Pass iff the rule is exact to `degree` (within tol); score is the node count (lower is better)."""

    name = "quadrature_exactness"
    version = "1.0.0"
    runtime_cap_s = 30.0

    def __init__(self, domain: str, degree: int, *, tol: float = 1e-9, n_random: int = 32,
                 require_positive: bool = False, require_interior: bool = False,
                 best_known_nodes: int | None = None):
        if domain not in DOMAINS:
            raise ValueError(f"domain must be one of {DOMAINS}")
        self.domain = domain
        self.degree = int(degree)
        self.tol = float(tol)
        self.n_random = int(n_random)
        self.require_positive = require_positive
        self.require_interior = require_interior
        self.best_known_nodes = best_known_nodes

    def run(self, candidate, rng, *, target, seed, **kw) -> Certificate:
        nodes, weights = normalize_rule(candidate, self.domain)
        rep = exactness_report(nodes, weights, self.domain, self.degree, rng, self.n_random)
        n = int(nodes.shape[0])
        positive = bool(np.all(weights > 0))
        interior = inside_domain(nodes, self.domain)
        exact = rep["max_monomial_error"] <= self.tol and rep["max_random_polynomial_error"] <= self.tol * max(1, rep["basis_size"])
        passed = bool(exact and (positive or not self.require_positive) and (interior or not self.require_interior))
        measured = exactness_degree(nodes, weights, self.domain, max_degree=self.degree + 6, tol=self.tol)
        details = {
            **rep,
            "domain": self.domain,
            "tol": self.tol,
            "node_count": n,
            "all_weights_positive": positive,
            "all_nodes_interior": interior,
            "weight_sum": float(weights.sum()),
            "volume": VOLUME[self.domain],
            "measured_exactness_degree": measured,
            "require_positive": self.require_positive,
            "require_interior": self.require_interior,
            "best_known_nodes": self.best_known_nodes,
        }
        optimal = None
        if self.best_known_nodes is not None:
            optimal = bool(passed and n <= self.best_known_nodes)
        return Certificate(self.name, self.version, target, passed, score=float(n), lower_is_better=True,
                           optimal=optimal, details=details)


# --- reference rules (for tests and the scoreboard; never for candidates) --------------------
def gauss_legendre(n: int) -> tuple[np.ndarray, np.ndarray]:
    x, w = np.polynomial.legendre.leggauss(n)
    return x.reshape(-1, 1), w


def clenshaw_curtis(n: int) -> tuple[np.ndarray, np.ndarray]:
    """n-point Clenshaw-Curtis rule on [-1, 1] (n >= 2), exact to degree n-1 (n even) or n (n odd)."""
    if n < 2:
        raise ValueError("Clenshaw-Curtis needs n >= 2")
    N = n - 1
    k = np.arange(n)
    x = np.cos(np.pi * k / N)
    w = np.zeros(n)
    for i in range(n):
        s = 0.0
        for j in range(1, N // 2 + 1):
            b = 1.0 if 2 * j == N else 2.0
            s += b / (4 * j * j - 1) * math.cos(2 * j * i * math.pi / N)
        c = 1.0 if i in (0, N) else 2.0
        w[i] = c / N * (1 - s)
    return x[::-1].reshape(-1, 1), w[::-1]


def newton_cotes(kind: str) -> tuple[np.ndarray, np.ndarray]:
    if kind == "midpoint":
        return np.array([[0.0]]), np.array([2.0])
    if kind == "trapezoid":
        return np.array([[-1.0], [1.0]]), np.array([1.0, 1.0])
    if kind == "simpson":
        return np.array([[-1.0], [0.0], [1.0]]), np.array([1 / 3, 4 / 3, 1 / 3])
    raise ValueError(kind)


def tensor_rule(rule1d: tuple[np.ndarray, np.ndarray], dim: int) -> tuple[np.ndarray, np.ndarray]:
    x, w = rule1d
    x = x.reshape(-1)
    nodes = np.array(list(itertools.product(x, repeat=dim)), dtype=float)
    weights = np.array([float(np.prod(c)) for c in itertools.product(w, repeat=dim)])
    return nodes, weights


def duffy_triangle(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Collapsed n x n Gauss-Legendre rule on the reference triangle, exact to degree 2n-2."""
    x, w = np.polynomial.legendre.leggauss(n)
    u = (x + 1) / 2
    wu = w / 2
    nodes, weights = [], []
    for i in range(n):
        for j in range(n):
            nodes.append([u[i], u[j] * (1 - u[i])])
            weights.append(wu[i] * wu[j] * (1 - u[i]))
    return np.array(nodes), np.array(weights)


def _bary(groups: list[tuple[float, tuple[float, float, float]]], area: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric triangle rule from (weight, barycentric orbit representative) pairs; weights sum to 1."""
    nodes, weights = [], []
    for w, lam in groups:
        seen = set()
        for perm in itertools.permutations(lam):
            key = tuple(round(v, 14) for v in perm)
            if key in seen:
                continue
            seen.add(key)
            nodes.append([perm[1], perm[2]])  # x = lambda_2, y = lambda_3
            weights.append(w * area)
    return np.array(nodes), np.array(weights)


def triangle_rule(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Classical symmetric rules on the reference triangle, by name."""
    s15 = math.sqrt(15.0)
    rules = {
        # degree 1, 1 point
        "centroid": [(1.0, (1 / 3, 1 / 3, 1 / 3))],
        # degree 2, 3 interior points (Strang & Fix 1973, rule 3)
        "strang-fix-3": [(1 / 3, (2 / 3, 1 / 6, 1 / 6))],
        # degree 2, 3 edge midpoints
        "edge-midpoints": [(1 / 3, (1 / 2, 1 / 2, 0.0))],
        # degree 3, 4 points, one negative weight (Strang & Fix 1973, rule 6; Stroud T2:3-1)
        "strang-fix-4": [(-27 / 48, (1 / 3, 1 / 3, 1 / 3)), (25 / 48, (0.6, 0.2, 0.2))],
        # degree 4, 6 points (Dunavant 1985, rule 4)
        "dunavant-6": [(0.223381589678011, (0.445948490915965, 0.445948490915965, 0.108103018168070)),
                       (0.109951743655322, (0.091576213509771, 0.091576213509771, 0.816847572980459))],
        # degree 5, 7 points (Radon 1948; Dunavant 1985, rule 5)
        "radon-7": [(9 / 40, (1 / 3, 1 / 3, 1 / 3)),
                    ((155 - s15) / 1200, ((6 - s15) / 21, (6 - s15) / 21, (9 + 2 * s15) / 21)),
                    ((155 + s15) / 1200, ((6 + s15) / 21, (6 + s15) / 21, (9 - 2 * s15) / 21))],
    }
    if name not in rules:
        raise KeyError(f"unknown triangle rule {name!r}; known: {sorted(rules)}")
    return _bary(rules[name])


def square_rule(name: str) -> tuple[np.ndarray, np.ndarray]:
    if name == "stroud-4":  # degree 3, 4 points on the coordinate axes (Stroud C2:3-1)
        r = math.sqrt(2 / 3)
        return np.array([[r, 0.0], [-r, 0.0], [0.0, r], [0.0, -r]]), np.array([1.0, 1.0, 1.0, 1.0])
    raise KeyError(name)


def corrupt_weight(rule: tuple[np.ndarray, np.ndarray], index: int = 0, delta: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = rule
    w = weights.copy()
    w[index] += delta
    return nodes.copy(), w
