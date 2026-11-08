"""Lemma check: the theorist's immediate reward (Work package 4). A lemma is a statement in words with a
computer-algebra check, a single SymPy expression that must evaluate to True, or simplify to zero, when the
program runs it in the judge's sandbox with SymPy allowed. The check is a pass-or-fail program; nothing about
the statement's wording enters the score.

Validation before anything runs: the check parses as one expression over a fixed namespace (the symbols x, y,
z, t, u, v, the integer symbols n, k, m, i, j, and a list of SymPy functions and methods); it must call at least
one symbolic operation (integrate, diff, Sum, solve, expand, simplify, Matrix, legendre, ...) and mention at
least one symbol, so that a bare constant or a numeric tautology is not a lemma; a check that repeats an
earlier one counts once. Each check runs in the sandbox under its own time budget, the remaining share of the
file's budget divided equally among the checks still to run, so that one slow check cannot stall the rest: a
check cut off is a timeout, a failed check; a check the budget never reaches is skipped and counts for neither;
an exception is a failed check. The certificate: passed when at least one valid lemma check passes; the score is
the fraction of proposed lemmas whose checks pass (higher is better); every lemma's status and its seconds are
in the details.

Known weakness, recorded for the exploit catalogue (Work package 7): a tautology with a symbol in it, such as
simplify(x - x), passes the filter. The delayed credit (the certificate of builds derived from the proposal
minus the frozen baseline) is the signal that separates useful lemmas from true but idle ones; the ten best
theorist attempts per family are read by a person after every generation.
"""
from __future__ import annotations

import ast
from typing import Any

from .base import Certificate, Certifier

SYMBOLS = ("x", "y", "z", "t", "u", "v")
INT_SYMBOLS = ("n", "k", "m", "i", "j")
FUNCTIONS = ("integrate", "Integral", "diff", "Derivative", "Sum", "summation", "Product", "solve", "solveset", "roots",
             "real_roots", "nroots", "Poly", "Matrix", "det", "zeros", "eye", "expand", "factor", "simplify", "cancel", "together",
             "apart", "nsimplify", "binomial", "factorial", "gamma", "limit", "series", "legendre", "jacobi", "chebyshevt",
             "chebyshevu", "hermite", "laguerre", "sqrt", "Rational", "Integer", "Float", "pi", "E", "I", "oo", "exp", "log",
             "cos", "sin", "tan", "acos", "asin", "atan", "cosh", "sinh", "Abs", "Max", "Min", "Eq", "Ne", "Lt", "Le", "Gt", "Ge",
             "And", "Or", "Not", "S", "symbols", "sign", "floor", "ceiling", "re", "im", "conjugate", "KroneckerDelta",
             "Piecewise", "Lambda", "Tuple", "Interval", "oo", "zoo", "nan", "true", "false", "sympify", "ask", "Q",
             "degree", "LT", "LC", "Dummy", "Wild", "trigsimp", "radsimp", "powsimp", "logcombine", "rsolve", "dsolve",
             "Function", "Symbol", "FiniteSet", "Range", "range", "len", "sum", "abs", "min", "max", "all", "any", "int", "float",
             "complex", "list", "tuple", "zip", "enumerate", "round", "True", "False", "None")
METHODS = ("simplify", "expand", "equals", "subs", "doit", "evalf", "n", "factor", "cancel", "together", "apart", "as_poly",
           "coeff", "coeffs", "all_coeffs", "det", "T", "inv", "nroots", "free_symbols", "args", "rewrite", "series", "limit",
           "diff", "integrate", "is_zero", "is_positive", "is_real", "is_number", "lhs", "rhs", "nsimplify", "as_expr",
           "degree", "rank", "nullspace", "eigenvals", "transpose", "trigsimp", "radsimp", "powsimp", "removeO", "sum",
           "replace", "xreplace", "has", "count", "atoms", "is_polynomial", "as_numer_denom", "normal", "is_rational",
           "is_integer", "is_nonnegative", "is_negative", "real_roots", "LC", "LT", "monic", "row", "col", "shape", "tolist",
           "dot", "cross", "norm", "is_true")
SYMBOLIC_OPS = {"integrate", "Integral", "diff", "Derivative", "Sum", "summation", "Product", "solve", "solveset", "roots",
                "real_roots", "nroots", "Poly", "Matrix", "det", "expand", "factor", "simplify", "cancel", "together", "apart",
                "nsimplify", "binomial", "factorial", "gamma", "limit", "series", "legendre", "jacobi", "chebyshevt",
                "chebyshevu", "hermite", "laguerre", "trigsimp", "radsimp", "powsimp", "rsolve", "dsolve", "equals", "doit",
                "subs", "evalf", "coeff", "coeffs", "all_coeffs", "eigenvals", "nullspace", "inv"}
ALLOWED_NODES = (ast.Expression, ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.Compare, ast.Call, ast.Name, ast.Constant, ast.List,
                 ast.Tuple, ast.Dict, ast.Attribute, ast.Subscript, ast.Slice, ast.IfExp, ast.ListComp, ast.GeneratorExp,
                 ast.comprehension, ast.Load, ast.Store, ast.keyword, ast.Lambda, ast.arguments, ast.arg,
                 ast.And, ast.Or, ast.Not, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv, ast.USub, ast.UAdd,
                 ast.MatMult, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn, ast.Is, ast.IsNot)
MAX_CHECK_CHARS = 600
MAX_LEMMAS = 8


def validate_check(expr: Any) -> tuple[bool, str | None, dict]:
    """Parse and whitelist one check. Returns (valid, error, facts) with facts holding the symbolic operations and
    symbols the check uses and a normalized form for duplicate detection."""
    facts: dict = {"ops": [], "symbols": [], "normalized": None}
    if not isinstance(expr, str) or not expr.strip():
        return False, "the check must be a non-empty SymPy expression", facts
    if len(expr) > MAX_CHECK_CHARS:
        return False, f"the check must be at most {MAX_CHECK_CHARS} characters", facts
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as e:
        return False, f"syntax error: {e.msg}", facts
    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.comprehension):
            targets = [node.target] if isinstance(node.target, ast.Name) else list(getattr(node.target, "elts", []))
            bound |= {t.id for t in targets if isinstance(t, ast.Name)}
        if isinstance(node, ast.Lambda):
            bound |= {a.arg for a in node.args.args}
    ops, syms = [], []
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODES):
            return False, f"forbidden construct {type(node).__name__}", facts
        if isinstance(node, ast.Name):
            if node.id in SYMBOLS or node.id in INT_SYMBOLS:
                syms.append(node.id)
            elif node.id not in FUNCTIONS and node.id not in bound:
                return False, f"unknown name {node.id!r}", facts
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or node.attr not in METHODS:
                return False, f"forbidden attribute .{node.attr}", facts
            if node.attr in SYMBOLIC_OPS:
                ops.append(node.attr)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in SYMBOLIC_OPS:
            ops.append(node.func.id)
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float, complex, bool, str, type(None))):
            return False, "only plain constants are allowed", facts
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and len(node.value) > 40:
            return False, "string constants must be short", facts
    facts.update({"ops": sorted(set(ops)), "symbols": sorted(set(syms)), "normalized": ast.dump(tree)})
    if not ops:
        return False, "trivial: the check calls no symbolic operation (integrate, diff, Sum, solve, simplify, expand, ...)", facts
    if not syms:
        return False, "trivial: the check mentions no symbol (x, y, z, t, u, v, n, k, m, i, j)", facts
    return True, None, facts


def validate_lemmas(items: Any) -> list[dict]:
    """Normalize a proposal's lemma list: each entry gets a status of 'valid', 'invalid' (with the reason),
    'trivial' or 'duplicate'; at most MAX_LEMMAS are kept."""
    out: list[dict] = []
    seen: set[str] = set()
    if not isinstance(items, list):
        return out
    for i, item in enumerate(items[:MAX_LEMMAS]):
        if not isinstance(item, dict):
            out.append({"index": i, "statement": "", "check": "", "status": "invalid", "error": "a lemma is an object with statement and check"})
            continue
        statement = " ".join(str(item.get("statement", "")).split())[:400]
        check = str(item.get("check", "")).strip()
        rec = {"index": i, "statement": statement, "check": check[:MAX_CHECK_CHARS], "status": "valid", "error": None}
        if len(statement.split()) < 3:
            rec.update(status="invalid", error="the statement must be a sentence")
            out.append(rec)
            continue
        ok, err, facts = validate_check(check)
        rec.update({"ops": facts["ops"], "symbols": facts["symbols"]})
        if not ok:
            rec.update(status="trivial" if err and err.startswith("trivial") else "invalid", error=err)
        elif facts["normalized"] in seen:
            rec.update(status="duplicate", error="the same check as an earlier lemma")
        else:
            seen.add(facts["normalized"])
        out.append(rec)
    return out


PROGRAM_HEAD = '''import sympy
from sympy import *
import signal as _signal
import time as _time
from sympy.core.relational import Relational, Eq as _Eq
from sympy.logic.boolalg import Boolean as _Boolean
x, y, z, t, u, v = symbols("x y z t u v", real=True)
n, k, m, i, j = symbols("n k m i j", integer=True)
_HAS_TIMER = hasattr(_signal, "setitimer") and hasattr(_signal, "SIGALRM")  # POSIX; the judge runs on Colab


def _verdict(val):
    if isinstance(val, bool):
        return val
    if val is S.true:
        return True
    if val is S.false:
        return False
    if isinstance(val, (int, float)):
        return abs(val) < 1e-12
    if isinstance(val, complex):
        return abs(val) < 1e-12
    if isinstance(val, (list, tuple)):
        return len(val) > 0 and all(_verdict(w) for w in val)
    if isinstance(val, _Eq):
        d = simplify(val.lhs - val.rhs)
        if d == 0:
            return True
        if d.is_number:
            return abs(complex(d.evalf())) < 1e-12
        return d.equals(0) is True
    if isinstance(val, Relational):
        r = simplify(val)
        return r is S.true
    if isinstance(val, _Boolean):
        r = simplify(val)
        return r is S.true
    if isinstance(val, Basic):
        d = simplify(val)
        if d == 0:
            return True
        if d.is_number:
            return abs(complex(d.evalf())) < 1e-12
        return d.equals(0) is True
    return False

'''


DEFAULT_BUDGET_S = 100.0   # the file's time budget under the judge's default cap of 120 s
BUDGET_MARGIN_S = 20.0     # SymPy's import, the file's own start, and the overshoot of a timer that fires inside a C-level call
MIN_CHECK_S = 1.0          # below this much budget left, the remaining checks are skipped rather than started

PROGRAM_CHECK = '''
class _CheckTimeout(Exception):
    pass


def _on_alarm(signum, frame):
    raise _CheckTimeout()


def _run_one(idx, fn, share):
    if _HAS_TIMER:
        _signal.setitimer(_signal.ITIMER_REAL, share)
    try:
        val = fn()
        ok = bool(_verdict(val))
        return {"index": idx, "status": "passed" if ok else "failed", "value": str(val)[:200], "error": None}
    except _CheckTimeout:
        return {"index": idx, "status": "timeout", "value": None, "error": "the check exceeded its time budget of %.1f s" % share}
    except Exception as e:  # a crashing check is a failed check
        return {"index": idx, "status": "error", "value": None, "error": (type(e).__name__ + ": " + str(e))[:300]}
    finally:
        if _HAS_TIMER:
            _signal.setitimer(_signal.ITIMER_REAL, 0)


def check():
    out = []
    if _HAS_TIMER:
        _signal.signal(_signal.SIGALRM, _on_alarm)
    t_start = _time.perf_counter()
    for pos, (idx, fn) in enumerate(CHECKS):
        left = _BUDGET_S - (_time.perf_counter() - t_start)
        if left < _MIN_CHECK_S:
            out.append({"index": idx, "status": "skipped", "value": None, "elapsed_s": 0.0, "budget_s": 0.0,
                        "error": "not run: the file's time budget of %.0f s was spent by the earlier checks" % _BUDGET_S})
            continue
        share = left / (len(CHECKS) - pos)
        t0 = _time.perf_counter()
        try:
            rec = _run_one(idx, fn, share)
        except _CheckTimeout:  # the timer fired after the check's own handling and before it was disarmed
            rec = {"index": idx, "status": "timeout", "value": None, "error": "the check exceeded its time budget of %.1f s" % share}
        rec["elapsed_s"] = round(_time.perf_counter() - t0, 3)
        rec["budget_s"] = round(share, 3)
        out.append(rec)
    return out
'''


def check_budget_s(cap_s: float | None) -> float:
    """The file's time budget under a sandbox wall-clock cap: the cap less BUDGET_MARGIN_S, at least 5 s."""
    cap = float(cap_s) if cap_s else DEFAULT_BUDGET_S + BUDGET_MARGIN_S
    return max(5.0, cap - BUDGET_MARGIN_S)


def lemma_program(lemmas: list[dict], budget_s: float = DEFAULT_BUDGET_S) -> str:
    """The file the sandbox runs: one function per valid check, and check() returning every verdict. Each check gets its
    own time budget, the remaining share of `budget_s` divided equally among the checks still to run (a fast check leaves
    its unused time to the later ones), enforced by an interval timer where the platform has one (POSIX, so the judge on
    Colab; elsewhere the budget only decides what is skipped): a check cut off is `timeout`, a check the budget never
    reaches is `skipped`, and the others run on. Every verdict carries its seconds and its budget."""
    parts = [PROGRAM_HEAD]
    valid = [l for l in lemmas if l["status"] == "valid"]
    for l in valid:
        parts.append(f"def _c{l['index']}():\n    return ({l['check']})\n\n")
    parts.append("CHECKS = [" + ", ".join(f"({l['index']}, _c{l['index']})" for l in valid) + "]\n")
    parts.append(f"_BUDGET_S = {float(budget_s)!r}\n_MIN_CHECK_S = {float(MIN_CHECK_S)!r}\n\n")
    parts.append(PROGRAM_CHECK)
    return "".join(parts)


# signal is the per-check timer of the file's head; a check cannot name it (it is not in FUNCTIONS)
SANDBOX_ALLOW_MODULES = ("sympy", "mpmath", "signal")
SANDBOX_ALLOW_ATTRS = tuple(sorted(set(METHODS) | {"integrate", "polynomial", "special", "optimize", "interpolate"}))


class LemmaCheckCertifier(Certifier):
    """Pass iff at least one valid lemma check passes; the score is the fraction of proposed lemmas that pass. A check
    cut off by its time budget (`timeout`) is a failed check; a check the budget never reached (`skipped`) counts for
    neither. Version 1.1.0: the per-check budget and the two statuses (2026-11-08)."""

    name = "lemma_check"
    version = "1.1.0"
    runtime_cap_s = 120.0

    def run(self, candidate: Any, rng, *, target, seed, lemmas: list[dict] | None = None, **kw) -> Certificate:
        """`candidate` is the sandbox's output of check() (a list of per-check verdicts) or None when the sandbox
        failed; `lemmas` the validated list the program built the file from."""
        lemmas = [dict(l) for l in (lemmas or [])]
        had_valid = any(l["status"] == "valid" for l in lemmas)
        by_index = {int(r["index"]): r for r in (candidate or []) if isinstance(r, dict) and "index" in r}
        for l in lemmas:
            if l["status"] == "valid":
                r = by_index.get(l["index"])
                if r is None:
                    l["status"], l["error"] = "error", "no verdict from the sandbox"
                else:
                    l["status"], l["value"], l["error"] = r.get("status", "error"), r.get("value"), r.get("error")
                    if r.get("elapsed_s") is not None:
                        l["elapsed_s"], l["budget_s"] = r.get("elapsed_s"), r.get("budget_s")
        n = len(lemmas)
        passed_n = sum(1 for l in lemmas if l["status"] == "passed")
        timed_out = sum(1 for l in lemmas if l["status"] == "timeout")
        skipped = sum(1 for l in lemmas if l["status"] == "skipped")
        failed_n = sum(1 for l in lemmas if l["status"] in ("failed", "error", "timeout"))
        rejected = sum(1 for l in lemmas if l["status"] in ("invalid", "trivial", "duplicate"))
        score = passed_n / n if n else 0.0
        details = {"lemmas": lemmas, "proposed": n, "passed": passed_n, "failed": failed_n, "rejected": rejected,
                   "timed_out": timed_out, "skipped": skipped, "sandbox_ok": candidate is not None}
        error = None
        if n == 0:
            error = "no lemmas proposed"
        elif candidate is None and had_valid:
            error = "the lemma checks did not run"
        return Certificate(self.name, self.version, target, passed=bool(passed_n > 0), score=float(score), lower_is_better=False,
                           optimal=None, details=details, error=error)
