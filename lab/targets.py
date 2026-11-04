"""Targets: what a role is asked for, which certifier scores it, the scoreboard entry it is held to, and the
reading-list cutoff the librarian enforces for it.

Families and their candidate contracts (the entry point the judge calls in the sandbox):
  quadrature      rule() -> (nodes, weights); certifier quadrature_exactness
  first_order     method() -> {"method": ..., "params": {...}}; certifier pep_worst_case_rate
  order           solve(problem, hs) -> one numerical solution per h; certifier convergence_order
  opcount         transform(re, im) -> (re, im) on counted scalars; certifier operation_count (instrumented sandbox)
  time_integrator tableau() -> {"A": ..., "b": ...} or {"alpha": ..., "beta": ...}; certifier stability_region_ssp

Target ids: quad-d7, quad-square-d5, quad-triangle-d4, quad-d7-pi, pep-<entry>, order-ode-p4, order-poisson-p2,
order-quad-p4, opcount-dft-n16, ssp-s3-p3, stab-real-s4-p4, stab-imag-s4-p4.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .certifiers.opcount import OperationCountCertifier
from .certifiers.order import ConvergenceOrderCertifier
from .certifiers.pep import PEPRateCertifier
from .certifiers.quadrature import DIM, QuadratureExactnessCertifier
from .certifiers.stability import StabilityCertifier
from .config import repo_root

COUNTED_MODULE = Path(__file__).resolve().parent / "certifiers" / "counted.py"

DOMAIN_TEXT = {
    "interval": ("the interval [-1, 1]", "floats x_i in [-1, 1]", "int_{-1}^{1} p(x) dx"),
    "square": ("the square [-1, 1]^2", "pairs [x_i, y_i]", "int over [-1,1]^2 of p(x, y) dx dy"),
    "cube": ("the cube [-1, 1]^3", "triples [x_i, y_i, z_i]", "int over [-1,1]^3 of p dx dy dz"),
    "triangle": ("the triangle with vertices (0,0), (1,0), (0,1)", "pairs [x_i, y_i]",
                 "int over the triangle of p(x, y) dx dy"),
}

# Reading-list cutoffs for rediscovery (Work package 1): a document dated on or after the cutoff is never shown.
CUTOFFS = {
    "quad-interval": "1814-01-01",       # Gauss (1814): the optimal node count from exactness alone
    "quad-square": "1948-01-01",         # Radon (1948): the 7-point degree-5 rule
    "quad-triangle": "1948-01-01",
    "quad-cube": "1948-01-01",
    "pep-davis-yin-contraction-calibration": "2015-01-01",   # Davis & Yin (arXiv 2015): the two-operator literature only
    "pep-douglas-rachford-contraction": "2014-01-01",        # Giselsson & Boyd (arXiv 2014)
    "pep-gd-strongly-convex-distance": "2014-01-01",         # Drori & Teboulle (2014): before performance estimation
    "pep-nesterov-strongly-convex-function-value": "2021-01-01",
    "ssp": "1988-01-01",                 # Shu & Osher (1988)
    "stab": "1963-01-01",                # Dahlquist (1963)
    "opcount": "1965-01-01",             # Cooley & Tukey (1965)
    "order": "1911-01-01",               # Richardson (1911)
}

SYSTEM_PROMPT = (
    "You are a careful numerical analyst writing short, correct Python. You derive what you need from first "
    "principles and never call library quadrature, root-finding or optimisation routines."
)

RULES_TEXT = ("Rules: use plain Python and NumPy arithmetic only (numpy.linalg is allowed). Library quadrature, root-finding "
              "and optimisation functions are forbidden and fail the attempt: numpy.polynomial, scipy.integrate, "
              "scipy.optimize, scipy.special, scipy.interpolate, sympy, mpmath. No files, no network, no printing. "
              "Reply with exactly one ```python code block containing the whole file.")

QUAD_PROMPT = """Task: construct a quadrature rule on {domain_desc} that integrates every polynomial of total degree at most {degree} exactly, with as few nodes as possible.

Write one complete Python file that defines

    def rule():
        ...
        return nodes, weights

where nodes is a list of {node_desc} and weights is a list of floats of the same length, so that
    {integral} = sum_i weights[i] * p(nodes[i])
for every polynomial p of degree <= {degree}. The judge applies the rule to every monomial of degree <= {degree} and to random polynomials; the largest error must be below 1e-9. The score is the number of nodes: fewer is better.{flags}

""" + RULES_TEXT

PEP_PROMPT = """Task: state the parameters of {method} on the class {cls} for which the lab certifies the worst-case rate by performance estimation.

Write one complete Python file that defines

    def method():
        return {{"method": "{method}", "params": {params}}}

Reply with exactly one ```python code block containing the whole file."""

ORDER_PROMPT = """Task: write a numerical scheme of order at least {order} for the problem family "{family}" and run it at a list of step sizes.

{family_desc}

Write one complete Python file that defines

    def solve(problem, hs):
        ...
        return solutions

where problem is the dict described above and hs is a list of step sizes; solutions must be a list with one entry per h: {solution_desc}. The judge measures the error of every entry against the exact solution, fits the order of convergence from the refinement study, and reports it with an interval; the attempt passes when the measured order reaches {order}. The score is the measured order: higher is better.

""" + RULES_TEXT

OPCOUNT_PROMPT = """Task: compute the discrete Fourier transform of length N = {n} with as few arithmetic operations as possible.

Write one complete Python file that defines

    def transform(re, im):
        ...
        return out_re, out_im

where re and im are lists of N numbers (real and imaginary parts of x_0..x_{{N-1}}) and out_re, out_im are lists of N numbers with out[k] = sum_n x_n exp(-2 pi i n k / N). The inputs are scalars that record every +, -, * and / applied to them: the judge counts the operations that reach the outputs (a shared subexpression counts once; multiplying by a precomputed constant such as a twiddle factor counts as one multiplication; constants computed from math.cos and math.sin are free; negation is free). Use only +, -, *, / on the inputs, no NumPy on them, no comparisons of them, and keep every output an expression of the inputs. The outputs are checked against the exact transform. The score is the total operation count: lower is better; the naive transform costs 8 N^2 = {naive} and a radix-2 FFT 5 N log2 N = {fft}.

""" + RULES_TEXT

STAB_PROMPT = """Task: give an explicit Runge-Kutta method with {stages} stages and order {order} with {objective}.

Write one complete Python file that defines

    def tableau():
        ...
        return {{"alpha": alpha, "beta": beta}}

in Shu-Osher form (u^(0) = u^n; u^(i) = sum_k alpha[i-1][k] u^(k) + dt sum_k beta[i-1][k] F(u^(k)) for i = 1..s; u^(n+1) = u^(s); alpha and beta are s x s lists with rows summing to one for alpha), or return {{"A": A, "b": b}} as a Butcher tableau. The judge checks the order conditions up to order {order}, the stage count, and that the stability polynomial agrees with the method on the test equation; {score_desc}

""" + RULES_TEXT

FAMILY_DESC = {
    "ode_ivp": ("The problem is the scalar initial value problem u'(t) = lam * (u - cos(omega t)) - omega sin(omega t), "
                "u(0) = u0, on [0, t_end], with problem = {'lam': ..., 'omega': ..., 'u0': 1.0, 't_end': 1.0}.",
                "the value u(t_end) obtained with step size h (t_end / h steps)"),
    "poisson1d": ("The problem is -u''(x) = f(x) on [0, 1] with u(0) = problem['left'], u(1) = problem['right'] and "
                  "f(x) = (k pi)^2 sin(k pi x) - 2 alpha, problem = {'k': ..., 'alpha': ..., 'left': ..., 'right': ...}.",
                  "a list of u at the grid points x_i = i h, i = 0..N with N = round(1/h), boundary values included"),
    "quadrature": ("The problem is the integral of exp(a x) cos(b x) over [0, 1], problem = {'a': ..., 'b': ...}.",
                   "the composite estimate of the integral with panel width h"),
}


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
    cutoff: str | None = None                 # reading-list cutoff date (YYYY-MM-DD) for the librarian
    instrument: dict | None = None            # sandbox instrumentation (opcount)

    def make_certifier(self, cache_dir=None, sdp_cap_s: float = 60.0, sdp_solver: str = "SCS"):
        p = self.params
        if self.family == "quadrature":
            return QuadratureExactnessCertifier(p["domain"], p["degree"], tol=p.get("tol", 1e-9),
                                                n_random=p.get("n_random", 32),
                                                require_positive=p.get("require_positive", False),
                                                require_interior=p.get("require_interior", False),
                                                best_known_nodes=int(self.best_known) if self.best_known else None)
        if self.family == "first_order":
            return PEPRateCertifier(solver=sdp_solver, cap_s=sdp_cap_s, cache_dir=cache_dir)
        if self.family == "order":
            return ConvergenceOrderCertifier(p["problem_family"], p["order"], best_known_order=self.best_known)
        if self.family == "opcount":
            return OperationCountCertifier(p["transform"], p["n"], best_known_ops=self.best_known)
        if self.family == "time_integrator":
            return StabilityCertifier(p["kind"], p["order"], stages=p.get("stages"), best_known=self.best_known)
        raise ValueError(self.family)

    def entry_args_for(self, seed: int) -> list:
        """Entry arguments for one attempt; seeded for the families whose inputs the certifier draws."""
        if self.family == "order":
            c = self.make_certifier()
            return [c.problem_for(seed), c.hs()]
        if self.family == "opcount":
            re, im = self.make_certifier().inputs_for(seed)
            return [re, im]
        return list(self.entry_args)

    def scoreboard_context(self) -> dict:
        return {"scoreboard_id": self.scoreboard_id, "best_known": self.best_known, "status": self.status,
                "source": self.best_known_source, "cutoff": self.cutoff}


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
                  best_known=best, best_known_source=source, status=status, scoreboard_id=sid,
                  cutoff=CUTOFFS[f"quad-{domain}"])


def first_order_target(entry_id: str, board: dict | None = None) -> Target:
    board = board or load_scoreboard("first_order_rates")
    for e in board["entries"]:
        if e["id"] == entry_id:
            params = dict(e["params"])
            prompt = PEP_PROMPT.format(method=e["method"], cls=e["class"], params=params)
            return Target(id=f"pep-{entry_id}", family="first_order", title=f"worst-case rate of {e['method']}",
                          prompt=prompt, entry_point="method", certifier_name="pep_worst_case_rate",
                          params={"method": e["method"], "params": params}, best_known=e.get("published_value"),
                          best_known_source=e.get("source"), status=e.get("status"), scoreboard_id=entry_id,
                          cutoff=CUTOFFS.get(f"pep-{entry_id}"))
    raise KeyError(entry_id)


ORDER_FAMILY_IDS = {"ode": "ode_ivp", "poisson": "poisson1d", "quad": "quadrature"}


def order_target(short: str, order: int, board: dict | None = None) -> Target:
    board = board or load_scoreboard("convergence_order")
    fam = ORDER_FAMILY_IDS[short]
    tid = f"order-{short}-p{order}"
    entry = next((e for e in board["entries"] if e["id"] == tid), None)
    desc, sol_desc = FAMILY_DESC[fam]
    prompt = ORDER_PROMPT.format(order=order, family=fam, family_desc=desc, solution_desc=sol_desc)
    return Target(id=tid, family="order", title=f"a scheme of order {order} for {fam}", prompt=prompt, entry_point="solve",
                  certifier_name="convergence_order", params={"problem_family": fam, "order": int(order)},
                  best_known=float(entry["best_known"]) if entry else float(order),
                  best_known_source=entry.get("source") if entry else None, status=entry.get("status") if entry else None,
                  scoreboard_id=tid if entry else None, cutoff=CUTOFFS["order"])


def opcount_target(n: int, board: dict | None = None) -> Target:
    board = board or load_scoreboard("fast_transforms")
    tid = f"opcount-dft-n{n}"
    entry = next((e for e in board["entries"] if e["id"] == tid), None)
    fft = 5 * n * math.log2(n)
    prompt = OPCOUNT_PROMPT.format(n=n, naive=8 * n * n, fft=int(fft))
    return Target(id=tid, family="opcount", title=f"the DFT of length {n} with the fewest operations", prompt=prompt,
                  entry_point="transform", certifier_name="operation_count", params={"transform": "dft", "n": int(n)},
                  best_known=float(entry["best_known"]) if entry else fft, best_known_source=entry.get("source") if entry else None,
                  status=entry.get("status") if entry else None, scoreboard_id=tid if entry else None,
                  cutoff=CUTOFFS["opcount"], instrument={"module": str(COUNTED_MODULE), "wrap_args": [0, 1]})


def stability_target(kind: str, stages: int, order: int, board: dict | None = None) -> Target:
    board = board or load_scoreboard("time_integrators")
    if kind == "ssp":
        tid = f"ssp-s{stages}-p{order}"
        entries = board["ssp"]["entries"]
        objective = "the largest strong-stability-preserving coefficient"
        score_desc = "the score is the SSP coefficient (the radius of absolute monotonicity of the Butcher array): higher is better, and it must be positive."
        title = f"SSP coefficient of an explicit {stages}-stage order-{order} method"
    else:
        axis = "real" if kind == "real_axis" else "imag"
        tid = f"stab-{axis}-s{stages}-p{order}"
        entries = board["region"]["entries"]
        objective = f"the largest extent of the stability region along the {'negative real' if axis == 'real' else 'imaginary'} axis"
        score_desc = f"the score is the extent of the region of absolute stability along the {'negative real' if axis == 'real' else 'imaginary'} axis: higher is better."
        title = f"{axis}-axis stability extent of an explicit {stages}-stage order-{order} method"
    entry = next((e for e in entries if e["id"] == tid), None)
    prompt = STAB_PROMPT.format(stages=stages, order=order, objective=objective, score_desc=score_desc)
    return Target(id=tid, family="time_integrator", title=title, prompt=prompt, entry_point="tableau",
                  certifier_name="stability_region_ssp", params={"kind": kind, "stages": int(stages), "order": int(order)},
                  best_known=float(entry["best_known"]) if entry else None, best_known_source=entry.get("source") if entry else None,
                  status=entry.get("status") if entry else None, scoreboard_id=tid if entry else None,
                  cutoff=CUTOFFS["ssp" if kind == "ssp" else "stab"])


def get_target(target_id: str) -> Target:
    """quad-d7, quad-square-d5, quad-triangle-d4, quad-d7-pi, pep-<scoreboard id>, order-<ode|poisson|quad>-p<k>,
    opcount-dft-n<N>, ssp-s<s>-p<p>, stab-<real|imag>-s<s>-p<p>."""
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
    parts = target_id.split("-")
    try:
        if parts[0] == "order" and len(parts) == 3 and parts[2].startswith("p"):
            return order_target(parts[1], int(parts[2][1:]))
        if parts[0] == "opcount" and len(parts) == 3 and parts[1] == "dft" and parts[2].startswith("n"):
            return opcount_target(int(parts[2][1:]))
        if parts[0] == "ssp" and len(parts) == 3:
            return stability_target("ssp", int(parts[1][1:]), int(parts[2][1:]))
        if parts[0] == "stab" and len(parts) == 4 and parts[1] in ("real", "imag"):
            return stability_target("real_axis" if parts[1] == "real" else "imag_axis", int(parts[2][1:]), int(parts[3][1:]))
    except (ValueError, KeyError) as e:
        raise KeyError(f"malformed target id {target_id!r}: {e}")
    raise KeyError(f"unknown target {target_id!r}")


def list_targets() -> list[str]:
    out = [f"quad-d{d}" for d in (1, 3, 5, 7, 9, 11)]
    out += [f"quad-square-d{d}" for d in (1, 3, 5, 7)] + [f"quad-triangle-d{d}" for d in (1, 2, 3, 4, 5, 6, 7)]
    out += [f"pep-{e['id']}" for e in load_scoreboard("first_order_rates")["entries"]]
    out += [e["id"] for e in load_scoreboard("convergence_order")["entries"]]
    out += [e["id"] for e in load_scoreboard("fast_transforms")["entries"]]
    board = load_scoreboard("time_integrators")
    out += [e["id"] for e in board["ssp"]["entries"]] + [e["id"] for e in board["region"]["entries"]]
    return out


def family_of(target_id: str) -> str:
    return get_target(target_id).family
