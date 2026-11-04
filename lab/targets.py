"""Targets: what a role is asked for, which certifier scores it, and the scoreboard entry it is held to.

A quadrature target names a domain and a degree; its candidate is a Python file defining `rule()` that
returns (nodes, weights). A first-order target names a method and its parameters; its candidate defines
`method()` returning {"method": ..., "params": {...}} (week 2 uses these only to test the judge's PEP path).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .certifiers.pep import PEPRateCertifier
from .certifiers.quadrature import DIM, QuadratureExactnessCertifier
from .config import repo_root

DOMAIN_TEXT = {
    "interval": ("the interval [-1, 1]", "floats x_i in [-1, 1]", "int_{-1}^{1} p(x) dx"),
    "square": ("the square [-1, 1]^2", "pairs [x_i, y_i]", "int over [-1,1]^2 of p(x, y) dx dy"),
    "cube": ("the cube [-1, 1]^3", "triples [x_i, y_i, z_i]", "int over [-1,1]^3 of p dx dy dz"),
    "triangle": ("the triangle with vertices (0,0), (1,0), (0,1)", "pairs [x_i, y_i]",
                 "int over the triangle of p(x, y) dx dy"),
}

SYSTEM_PROMPT = (
    "You are a careful numerical analyst writing short, correct Python. You derive what you need from first "
    "principles and never call library quadrature, root-finding or optimisation routines."
)

QUAD_PROMPT = """Task: construct a quadrature rule on {domain_desc} that integrates every polynomial of total degree at most {degree} exactly, with as few nodes as possible.

Write one complete Python file that defines

    def rule():
        ...
        return nodes, weights

where nodes is a list of {node_desc} and weights is a list of floats of the same length, so that
    {integral} = sum_i weights[i] * p(nodes[i])
for every polynomial p of degree <= {degree}. The judge applies the rule to every monomial of degree <= {degree} and to random polynomials; the largest error must be below 1e-9. The score is the number of nodes: fewer is better.{flags}

Rules: use plain Python and NumPy arithmetic only (numpy.linalg is allowed). Library quadrature, root-finding and optimisation functions are forbidden and fail the attempt: numpy.polynomial, scipy.integrate, scipy.optimize, scipy.special, scipy.interpolate, sympy, mpmath. No files, no network, no printing. Reply with exactly one ```python code block containing the whole file."""

PEP_PROMPT = """Task: state the parameters of {method} on the class {cls} for which the lab certifies the worst-case rate by performance estimation.

Write one complete Python file that defines

    def method():
        return {{"method": "{method}", "params": {params}}}

Reply with exactly one ```python code block containing the whole file."""


@dataclass
class Target:
    id: str
    family: str
    title: str
    prompt: str
    entry_point: str
    certifier_name: str
    params: dict = field(default_factory=dict)
    entry_args: list = field(default_factory=list)
    best_known: float | None = None
    best_known_source: str | None = None
    status: str | None = None
    scoreboard_id: str | None = None

    def make_certifier(self, cache_dir=None, sdp_cap_s: float = 60.0, sdp_solver: str = "SCS"):
        if self.family == "quadrature":
            p = self.params
            return QuadratureExactnessCertifier(p["domain"], p["degree"], tol=p.get("tol", 1e-9),
                                                n_random=p.get("n_random", 32),
                                                require_positive=p.get("require_positive", False),
                                                require_interior=p.get("require_interior", False),
                                                best_known_nodes=int(self.best_known) if self.best_known else None)
        if self.family == "first_order":
            return PEPRateCertifier(solver=sdp_solver, cap_s=sdp_cap_s, cache_dir=cache_dir)
        raise ValueError(self.family)

    def scoreboard_context(self) -> dict:
        return {"scoreboard_id": self.scoreboard_id, "best_known": self.best_known, "status": self.status,
                "source": self.best_known_source}


def load_scoreboard(name: str, root: Path | None = None) -> dict:
    path = (root or repo_root()) / "scoreboard" / f"{name}.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def quadrature_best_known(domain: str, degree: int, board: dict | None = None) -> tuple[float | None, str | None, str | None, str | None]:
    board = board or load_scoreboard("quadrature")
    section = board.get(domain, {})
    for e in section.get("entries", []):
        if int(e["degree"]) == int(degree):
            return float(e["best_known"]), str(e.get("source") or section.get("formula_source") or ""), e.get("status"), e.get("id")
    if domain == "interval":
        return float(math.ceil((degree + 1) / 2)), str(section.get("formula_source", "")), "solved", f"interval-d{degree}"
    return None, None, None, None


def quadrature_target(domain: str, degree: int, *, require_positive: bool = False, require_interior: bool = False,
                      board: dict | None = None) -> Target:
    if domain not in DIM:
        raise ValueError(domain)
    best, source, status, sid = quadrature_best_known(domain, degree, board)
    desc, node_desc, integral = DOMAIN_TEXT[domain]
    flags = ""
    if require_positive or require_interior:
        flags = " The rule must have" + (" positive weights" if require_positive else "") + \
                (" and" if require_positive and require_interior else "") + \
                (" nodes inside the domain" if require_interior else "") + "."
    prompt = QUAD_PROMPT.format(domain_desc=desc, degree=degree, node_desc=node_desc, integral=integral, flags=flags)
    tid = f"quad-{domain}-d{degree}" if domain != "interval" else f"quad-d{degree}"
    if require_positive or require_interior:
        tid += "-pi"
    return Target(id=tid, family="quadrature", title=f"exact to degree {degree} on {desc} with the fewest nodes",
                  prompt=prompt, entry_point="rule", certifier_name="quadrature_exactness",
                  params={"domain": domain, "degree": int(degree), "require_positive": require_positive,
                          "require_interior": require_interior},
                  best_known=best, best_known_source=source, status=status, scoreboard_id=sid)


def first_order_target(entry_id: str, board: dict | None = None) -> Target:
    board = board or load_scoreboard("first_order_rates")
    for e in board["entries"]:
        if e["id"] == entry_id:
            params = dict(e["params"])
            prompt = PEP_PROMPT.format(method=e["method"], cls=e["class"], params=params)
            return Target(id=f"pep-{entry_id}", family="first_order", title=f"worst-case rate of {e['method']}",
                          prompt=prompt, entry_point="method", certifier_name="pep_worst_case_rate",
                          params={"method": e["method"], "params": params}, best_known=e.get("published_value"),
                          best_known_source=e.get("source"), status=e.get("status"), scoreboard_id=entry_id)
    raise KeyError(entry_id)


def get_target(target_id: str) -> Target:
    """quad-d7, quad-square-d5, quad-triangle-d4, quad-d7-pi, pep-<scoreboard id>."""
    if target_id.startswith("pep-"):
        return first_order_target(target_id[4:])
    if target_id.startswith("quad-"):
        parts = target_id.split("-")
        pi = parts[-1] == "pi"
        if pi:
            parts = parts[:-1]
        if len(parts) == 2:
            domain, deg = "interval", parts[1]
        elif len(parts) == 3:
            domain, deg = parts[1], parts[2]
        else:
            raise KeyError(target_id)
        if not deg.startswith("d"):
            raise KeyError(target_id)
        return quadrature_target(domain, int(deg[1:]), require_positive=pi, require_interior=pi)
    raise KeyError(f"unknown target {target_id!r}")


def list_targets() -> list[str]:
    out = [f"quad-d{d}" for d in (1, 3, 5, 7, 9, 11)]
    out += [f"quad-square-d{d}" for d in (1, 3, 5, 7)] + [f"quad-triangle-d{d}" for d in (1, 2, 3, 4, 5, 6, 7)]
    out += [f"pep-{e['id']}" for e in load_scoreboard("first_order_rates")["entries"]]
    return out
