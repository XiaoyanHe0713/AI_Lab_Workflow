"""Operation-count certificate: the cost of a straight-line program by instrumented arithmetic.

The candidate is a transform (a program) that maps input numbers to output numbers with +, -, *
and /. The certifier runs it on seeded random inputs made of `Counted` scalars (lab/certifiers/
counted.py): every operation that reaches an output becomes a node of an expression DAG, and the
count is the number of distinct nodes by kind. Negation and operations between constants are free,
as in the published counts. The outputs are checked against the exact transform computed here with
NumPy, so a program that is cheap because it is wrong fails.

A cost model, a formula in n such as "5*n*log2(n)", can be certified against instrumented runs at
several sizes: the model must match the measured count at every size within a stated tolerance.

Reference transforms (naive DFT, radix-2 FFT) live at the bottom for the certifier's own tests and
for the scoreboard; with real arithmetic and precomputed twiddles the radix-2 FFT of size N counts
exactly 2 N log2 N multiplications and 3 N log2 N additions (5 N log2 N flops), and the naive
transform 4 N^2 multiplications and 4 N^2 additions.
"""
from __future__ import annotations

import ast
import math
from typing import Any, Callable

import numpy as np

from .base import Certificate, Certifier
from .counted import count_ops, values, wrap_inputs

TRANSFORMS = ("dft",)


# --- the exact transform ---------------------------------------------------------------------
def exact_transform(name: str, re: list[float], im: list[float]) -> tuple[np.ndarray, np.ndarray]:
    if name == "dft":
        x = np.asarray(re, dtype=float) + 1j * np.asarray(im, dtype=float)
        X = np.fft.fft(x)
        return X.real, X.imag
    raise ValueError(f"unknown transform {name!r}")


def random_inputs(n: int, rng: np.random.Generator) -> tuple[list[float], list[float]]:
    re = rng.standard_normal(n).tolist()
    im = rng.standard_normal(n).tolist()
    return re, im


# --- cost models -------------------------------------------------------------------------------
_ALLOWED_FUNCS = {"log2": math.log2, "log": math.log, "sqrt": math.sqrt, "ceil": math.ceil, "floor": math.floor}


def eval_cost_model(expr: str, n: int) -> float:
    """Evaluate a cost model in n with +, -, *, /, **, parentheses and log2, log, sqrt, ceil, floor."""
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if node.id != "n" and node.id not in _ALLOWED_FUNCS:
                raise ValueError(f"cost model uses an unknown name {node.id!r}")
        elif isinstance(node, ast.Call):
            if not (isinstance(node.func, ast.Name) and node.func.id in _ALLOWED_FUNCS):
                raise ValueError("cost model may only call log2, log, sqrt, ceil, floor")
        elif not isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Add, ast.Sub, ast.Mult,
                                   ast.Div, ast.Pow, ast.USub, ast.UAdd, ast.Load)):
            raise ValueError(f"cost model contains a forbidden construct: {type(node).__name__}")
    return float(eval(compile(tree, "<cost model>", "eval"), {"__builtins__": {}}, {"n": float(n), **_ALLOWED_FUNCS}))


# --- instrumented runs -------------------------------------------------------------------------
def instrumented_run(program: Callable, re: list[float], im: list[float]) -> dict:
    """Run program(re, im) on Counted inputs; return the counts and the output values."""
    out = program(wrap_inputs(re), wrap_inputs(im))
    counts = count_ops(out)
    return {"counts": counts, "outputs": values(out)}


class OperationCountCertifier(Certifier):
    """Candidate: a callable transform(re, im) (in process), or the dict the instrumented sandbox returns
    ({"counts": ..., "outputs": [re, im]} for inputs the certifier generated from the seed), optionally with
    a "cost_model" string. Score: the total operation count at size n (lower is better)."""

    name = "operation_count"
    version = "1.0.0"
    runtime_cap_s = 30.0

    def __init__(self, transform: str = "dft", n: int = 16, *, tol: float = 1e-9, max_ops: float | None = None,
                 best_known_ops: float | None = None, model_sizes: tuple[int, ...] = (8, 16, 32),
                 model_tol: float = 0.05):
        if transform not in TRANSFORMS:
            raise ValueError(f"transform must be one of {TRANSFORMS}")
        self.transform = transform
        self.n = int(n)
        self.tol = float(tol)
        self.max_ops = max_ops                   # pass only if total <= max_ops (default: fewer than the naive 8 n^2)
        self.best_known_ops = best_known_ops     # optimal if total <= best known
        self.model_sizes = tuple(int(s) for s in model_sizes)
        self.model_tol = float(model_tol)

    def inputs_for(self, seed: int, n: int | None = None) -> tuple[list[float], list[float]]:
        rng = np.random.default_rng(seed)
        return random_inputs(n or self.n, rng)

    def run(self, candidate: Any, rng, *, target, seed, **kw) -> Certificate:
        n = self.n
        re, im = self.inputs_for(seed, n)
        cost_model = None
        measured_models = []
        program = None
        if callable(candidate):
            res = instrumented_run(candidate, re, im)
            program = candidate
        elif isinstance(candidate, dict) and "counts" in candidate and "outputs" in candidate:
            res = {"counts": dict(candidate["counts"]), "outputs": candidate["outputs"]}
            cost_model = candidate.get("cost_model")
        elif isinstance(candidate, dict) and callable(candidate.get("program")):
            program = candidate["program"]
            res = instrumented_run(program, re, im)
            cost_model = candidate.get("cost_model")
        else:
            raise ValueError("candidate must be a callable transform, {'program': callable, 'cost_model': str} or the "
                             "instrumented run {'counts', 'outputs'}")
        counts = res["counts"]
        total = int(counts.get("total", 0))
        untraceable = int(counts.get("untraceable_outputs", 0))
        # correctness against the exact transform on the same inputs
        try:
            out_re, out_im = res["outputs"]
            got_re = np.asarray(out_re, dtype=float).reshape(-1)
            got_im = np.asarray(out_im, dtype=float).reshape(-1)
        except Exception as e:
            raise ValueError(f"outputs must be two lists (real, imaginary): {e}")
        ex_re, ex_im = exact_transform(self.transform, re, im)
        if got_re.shape != ex_re.shape or got_im.shape != ex_im.shape:
            raise ValueError(f"outputs must have length {n}; got {got_re.shape[0]} and {got_im.shape[0]}")
        scale = max(1.0, float(np.max(np.abs(ex_re)) + np.max(np.abs(ex_im))))
        max_err = float(max(np.max(np.abs(got_re - ex_re)), np.max(np.abs(got_im - ex_im)))) / scale
        correct = bool(max_err <= self.tol * n and untraceable == 0)
        # the default cap is the naive transform's own count, which a fast transform must beat strictly
        if self.max_ops is not None:
            cap, cap_rule = float(self.max_ops), "total <= max_ops"
            under_cap = bool(total <= cap)
        else:
            cap, cap_rule = float(8 * n * n), "total < naive count 8 n^2"
            under_cap = bool(total < cap)
        # a cost model is checked against instrumented runs at several sizes (needs the program in process)
        model_ok = None
        if cost_model is not None:
            model_ok = True
            if program is not None:
                for m in self.model_sizes:
                    r2, i2 = self.inputs_for(seed + 1, m)
                    meas = instrumented_run(program, r2, i2)["counts"]["total"]
                    pred = eval_cost_model(cost_model, m)
                    ok = abs(pred - meas) <= self.model_tol * max(1.0, meas)
                    measured_models.append({"n": m, "measured": meas, "model": pred, "ok": bool(ok)})
                    model_ok = model_ok and ok
            else:
                pred = eval_cost_model(cost_model, n)
                ok = abs(pred - total) <= self.model_tol * max(1.0, total)
                measured_models.append({"n": n, "measured": total, "model": pred, "ok": bool(ok)})
                model_ok = bool(ok)
        passed = bool(correct and under_cap and (model_ok is None or model_ok))
        details = {
            "transform": self.transform, "n": n, "counts": counts, "total_ops": total, "max_relative_error": max_err,
            "tol": self.tol, "correct": correct, "untraceable_outputs": untraceable, "ops_cap": cap, "cap_rule": cap_rule, "under_cap": under_cap,
            "n_log2_n": n * math.log2(n) if n > 1 else 0.0, "n_squared": n * n,
            "naive_total": 8 * n * n, "fft_total": 5 * n * math.log2(n) if n > 1 else 0.0,
            "cost_model": cost_model, "cost_model_checks": measured_models, "cost_model_ok": model_ok,
            "best_known_ops": self.best_known_ops,
        }
        optimal = None
        if self.best_known_ops is not None:
            optimal = bool(passed and total <= self.best_known_ops + 1e-9)
        return Certificate(self.name, self.version, target, passed, score=float(total), lower_is_better=True,
                           optimal=optimal, details=details)


# --- reference transforms (tests and the scoreboard; never for candidates) --------------------
def naive_dft(re, im):
    """The N^2 transform: X_k = sum_n x_n e^{-2 pi i nk/N} with precomputed twiddles (constants)."""
    n = len(re)
    out_re, out_im = [], []
    for k in range(n):
        sr, si = 0.0, 0.0
        for j in range(n):
            ang = -2.0 * math.pi * j * k / n
            wr, wi = math.cos(ang), math.sin(ang)
            sr = sr + (re[j] * wr - im[j] * wi)
            si = si + (re[j] * wi + im[j] * wr)
        out_re.append(sr)
        out_im.append(si)
    return out_re, out_im


def fft_radix2(re, im):
    """Recursive radix-2 decimation-in-time FFT; n must be a power of two."""
    n = len(re)
    if n == 1:
        return [re[0]], [im[0]]
    if n % 2:
        raise ValueError("radix-2 FFT needs a power-of-two length")
    er, ei = fft_radix2(re[0::2], im[0::2])
    odr, odi = fft_radix2(re[1::2], im[1::2])
    out_re = [None] * n
    out_im = [None] * n
    half = n // 2
    for k in range(half):
        ang = -2.0 * math.pi * k / n
        wr, wi = math.cos(ang), math.sin(ang)
        tr = odr[k] * wr - odi[k] * wi
        ti = odr[k] * wi + odi[k] * wr
        out_re[k] = er[k] + tr
        out_im[k] = ei[k] + ti
        out_re[k + half] = er[k] - tr
        out_im[k + half] = ei[k] - ti
    return out_re, out_im


def fft_formula_counts(n: int) -> dict:
    L = math.log2(n)
    return {"mul": int(round(2 * n * L)), "add": int(round(3 * n * L)), "total": int(round(5 * n * L))}


def naive_formula_counts(n: int) -> dict:
    return {"mul": 4 * n * n, "add": 4 * n * n, "total": 8 * n * n}
