"""Known-answer cases for the lemma check (the theorist's immediate reward): true identities pass, false claims fail,
constants and symbol-free checks are trivial, forbidden constructs are invalid, duplicates count once, a crashing
check is a failed check, and the sandbox allows SymPy for this run only."""
import signal

import pytest

from lab.certifiers.lemma import (BUDGET_MARGIN_S, SANDBOX_ALLOW_ATTRS, SANDBOX_ALLOW_MODULES, LemmaCheckCertifier, check_budget_s,
                                  lemma_program, validate_check, validate_lemmas)
from lab.judge import Judge, static_check

TRUE_CHECKS = [
    "integrate(x**2, (x, -1, 1)) - Rational(2, 3)",
    "Eq(integrate(x**3, (x, -1, 1)), 0)",
    "integrate(legendre(2, x) * legendre(3, x), (x, -1, 1))",
    "all(-1 < r < 1 for r in real_roots(legendre(4, x)))",
    "Eq(integrate(integrate(x**2 * y**2, (x, -1, 1)), (y, -1, 1)), Rational(4, 9))",
    "Eq(integrate(integrate(x**2 * y, (y, 0, 1 - x)), (x, 0, 1)), factorial(2) * factorial(1) / factorial(5))",
    "simplify(Sum(k, (k, 1, n)).doit() - n * (n + 1) / 2)",
    "Eq(binomial(n, k), binomial(n, k))",
    "expand((x + y)**2) - (x**2 + 2 * x * y + y**2)",
    "Eq(diff(x**3, x), 3 * x**2)",
    "Eq(det(Matrix([[x, 1], [1, x]])), x**2 - 1)",
    "(integrate(cos(t)**2, (t, 0, 2 * pi)) - pi).simplify()",
    "Eq(len(real_roots(legendre(5, x))), 5)",
    "integrate(x**4, (x, -1, 1)) - Rational(2, 5) == 0",
    "Eq(solve(x**2 - 2, x)[1], sqrt(2))",
]
FALSE_CHECKS = [
    "integrate(x**2, (x, -1, 1)) - Rational(1, 3)",
    "Eq(integrate(x**4, (x, -1, 1)), Rational(2, 3))",
    "all(r > 0 for r in real_roots(legendre(4, x)))",
    "Eq(diff(x**3, x), 2 * x**2)",
    "expand((x + y)**2) - (x**2 + y**2)",
    "Eq(integrate(integrate(1, (y, 0, 1 - x)), (x, 0, 1)), 1)",
]
INVALID_CHECKS = [
    ("__import__('os')", "unknown name"),
    ("open('x')", "unknown name"),
    ("integrate(x, (x, 0, 1)).__class__", "forbidden attribute"),
    ("x.subs(x, 1) if True else exec('1')", "unknown name"),
    ("lambda: 1", None),           # no symbolic op: trivial
    ("", "non-empty"),
    ("integrate(x, (x, 0, 1)", "syntax"),
    ("sympy.integrate(x, (x, 0, 1))", "unknown name"),
    ("integrate(x, (x, 0, 1)); 1", "syntax"),
]
TRIVIAL_CHECKS = ["1 - 1", "Rational(1, 2) - Rational(1, 2)", "simplify(Rational(1, 3) - Rational(1, 3))", "2 + 2 == 4", "sqrt(4) - 2",
                  "x - x", "x**2 - x**2"]


@pytest.fixture(scope="module")
def judge():
    return Judge(timeout_s=90)


def run_checks(judge, checks):
    """One proposal holds at most MAX_LEMMAS checks: longer lists run in batches and the details are merged."""
    from lab.certifiers.lemma import MAX_LEMMAS

    merged = None
    for start in range(0, len(checks), MAX_LEMMAS):
        batch = checks[start:start + MAX_LEMMAS]
        lemmas = validate_lemmas([{"statement": f"lemma number {start + i} about something", "check": c} for i, c in enumerate(batch)])
        src = lemma_program(lemmas)
        sb = judge.run_sandbox(src, "check", allow_modules=SANDBOX_ALLOW_MODULES, allow_attrs=SANDBOX_ALLOW_ATTRS)
        assert sb.ok, sb.error
        cert = LemmaCheckCertifier().certify(sb.result, target="t", seed=0, lemmas=lemmas)
        if merged is None:
            merged = cert
        else:
            merged.details["lemmas"] += cert.details["lemmas"]
            for k in ("proposed", "passed", "failed", "rejected"):
                merged.details[k] += cert.details[k]
            merged.passed = merged.passed or cert.passed
            merged.score = merged.details["passed"] / merged.details["proposed"]
    return merged


def test_true_identities_pass(judge):
    cert = run_checks(judge, TRUE_CHECKS)
    statuses = [l["status"] for l in cert.details["lemmas"]]
    assert statuses == ["passed"] * len(TRUE_CHECKS), list(zip(TRUE_CHECKS, statuses, [l.get("error") for l in cert.details["lemmas"]]))
    assert cert.passed and cert.score == 1.0 and cert.certifier == "lemma_check" and not cert.lower_is_better


def test_false_claims_fail(judge):
    cert = run_checks(judge, FALSE_CHECKS)
    assert [l["status"] for l in cert.details["lemmas"]] == ["failed"] * len(FALSE_CHECKS)
    assert not cert.passed and cert.score == 0.0 and "no lemma" in (cert.error or "") or cert.error is None


@pytest.mark.parametrize("check,fragment", INVALID_CHECKS)
def test_invalid_checks_never_run(check, fragment):
    ok, err, _ = validate_check(check)
    assert not ok
    if fragment:
        assert fragment in err


@pytest.mark.parametrize("check", TRIVIAL_CHECKS)
def test_trivial_checks_are_not_lemmas(check):
    ok, err, _ = validate_check(check)
    assert not ok and err.startswith("trivial"), (check, err)


def test_duplicates_count_once_and_statements_must_be_sentences():
    lemmas = validate_lemmas([{"statement": "the integral of x squared is two thirds", "check": TRUE_CHECKS[0]},
                              {"statement": "same check again in other words", "check": "integrate(x**2, (x, -1, 1)) - Rational(2, 3)"},
                              {"statement": "short", "check": TRUE_CHECKS[1]}, "not a dict", {"statement": "no check here at all"}])
    assert [l["status"] for l in lemmas] == ["valid", "duplicate", "invalid", "invalid", "invalid"]
    assert "sentence" in lemmas[2]["error"] and "non-empty" in lemmas[4]["error"]


def test_mixed_proposal_scores_the_fraction_and_crashes_are_failed_checks(judge):
    checks = [TRUE_CHECKS[0], FALSE_CHECKS[0], "solve(x**2 + 1, x)[9]", "1 - 1", TRUE_CHECKS[2]]
    cert = run_checks(judge, checks)
    st = [l["status"] for l in cert.details["lemmas"]]
    assert st == ["passed", "failed", "error", "trivial", "passed"] and "IndexError" in cert.details["lemmas"][2]["error"]
    assert cert.passed and cert.score == pytest.approx(2 / 5) and cert.details["rejected"] == 1 and cert.details["failed"] == 2


def test_no_lemmas_is_a_failed_certificate():
    cert = LemmaCheckCertifier().certify([], target="t", seed=0, lemmas=[])
    assert not cert.passed and cert.score == 0.0 and "no lemmas" in cert.error
    lemmas = validate_lemmas([{"statement": "a valid lemma whose checks never ran", "check": TRUE_CHECKS[0]}])
    cert = LemmaCheckCertifier().certify(None, target="t", seed=0, lemmas=lemmas)
    assert not cert.passed and cert.details["lemmas"][0]["status"] == "error" and "did not run" in cert.error


def test_sympy_is_forbidden_for_candidates_and_allowed_for_lemma_checks(judge):
    src = "import sympy\n\ndef rule():\n    return [0.0], [2.0]\n"
    assert any("sympy" in v for v in static_check(src))
    assert static_check(src, allow_modules=("sympy", "mpmath")) == []
    sb = judge.run_sandbox(src, "rule")
    assert not sb.ok and sb.error["type"] == "ForbiddenImport"
    sb = judge.run_sandbox(src, "rule", allow_modules=("sympy", "mpmath"))
    assert sb.ok and sb.result == [[0.0], [2.0]]
    # the allowance is per run: os stays forbidden even then
    sb = judge.run_sandbox("import os\n\ndef rule():\n    return 1\n", "rule", allow_modules=("sympy", "mpmath"))
    assert not sb.ok


def test_lemma_program_is_straight_line_code():
    lemmas = validate_lemmas([{"statement": "the integral of x squared over the interval", "check": TRUE_CHECKS[0]}])
    src = lemma_program(lemmas)
    assert "def _c0():" in src and "CHECKS = [(0, _c0)]" in src and "exec(" not in src and "eval(" not in src
    assert static_check(src, allow_modules=SANDBOX_ALLOW_MODULES, allow_attrs=SANDBOX_ALLOW_ATTRS) == []


# --- the per-check time budget (the week-4 defect: one slow lemma stalled the whole file) ------------------------------
def _certify(judge, lemmas, budget_s):
    src = lemma_program(lemmas, budget_s=budget_s)
    assert "_BUDGET_S = " in src and "def check():" in src and "setitimer" in src
    sb = judge.run_sandbox(src, "check", allow_modules=SANDBOX_ALLOW_MODULES, allow_attrs=SANDBOX_ALLOW_ATTRS)
    assert sb.ok, sb.error
    return LemmaCheckCertifier().certify(sb.result, target="t", seed=0, lemmas=lemmas)


def test_budget_spent_skips_the_rest_and_a_skip_is_neither_passed_nor_failed(judge):
    lemmas = validate_lemmas([{"statement": "the integral of x squared over the interval", "check": TRUE_CHECKS[0]},
                              {"statement": "the integral of x cubed is zero", "check": TRUE_CHECKS[1]}])
    cert = _certify(judge, lemmas, budget_s=0.0)
    assert [l["status"] for l in cert.details["lemmas"]] == ["skipped", "skipped"]
    assert cert.details["skipped"] == 2 and cert.details["failed"] == 0 and cert.details["timed_out"] == 0
    assert not cert.passed and cert.score == 0.0 and "spent by the earlier checks" in cert.details["lemmas"][0]["error"]
    assert cert.version == "1.1.0"
    cert = _certify(judge, lemmas, budget_s=30.0)  # with time, both run and every verdict carries its seconds and budget
    assert [l["status"] for l in cert.details["lemmas"]] == ["passed", "passed"] and cert.passed
    assert all(l["elapsed_s"] >= 0.0 and l["budget_s"] > 0 for l in cert.details["lemmas"])
    assert cert.details["lemmas"][0]["budget_s"] == pytest.approx(15.0, abs=0.5)  # half of the budget for the first of two


@pytest.mark.skipif(not hasattr(signal, "setitimer"), reason="the per-check timer needs setitimer (POSIX); here the budget only skips")
def test_a_slow_check_is_cut_off_and_the_rest_still_run(judge):
    slow = "sum(1 for i in range(10**9)) + simplify(x - x)"
    assert validate_check(slow)[0]
    lemmas = validate_lemmas([{"statement": "a check that does not finish in time", "check": slow},
                              {"statement": "the integral of x squared over the interval", "check": TRUE_CHECKS[0]}])
    cert = _certify(judge, lemmas, budget_s=4.0)
    st = [l["status"] for l in cert.details["lemmas"]]
    assert st == ["timeout", "passed"], cert.details["lemmas"]
    assert cert.details["timed_out"] == 1 and cert.details["failed"] == 1 and cert.passed and cert.score == pytest.approx(0.5)
    first = cert.details["lemmas"][0]
    assert "time budget" in first["error"] and first["budget_s"] == pytest.approx(2.0, abs=0.5) and first["elapsed_s"] < 3.5


def test_check_budget_is_the_cap_less_the_margin():
    assert check_budget_s(120.0) == pytest.approx(120.0 - BUDGET_MARGIN_S)
    assert check_budget_s(None) == pytest.approx(100.0) and check_budget_s(10.0) == 5.0
