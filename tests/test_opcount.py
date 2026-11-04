"""Known-answer cases for the operation-count certifier: the radix-2 FFT counts 2 N log2 N multiplications and
3 N log2 N additions, the naive transform 4 N^2 of each, shared subexpressions count once, dead code counts
nothing, a wrong transform fails however cheap, an untraceable output fails, and cost models are checked
against instrumented runs."""
import math

import numpy as np
import pytest

from lab.certifiers.counted import Counted, count_ops, values, wrap_inputs
from lab.certifiers.opcount import (
    OperationCountCertifier,
    eval_cost_model,
    fft_formula_counts,
    fft_radix2,
    instrumented_run,
    naive_dft,
    naive_formula_counts,
)


@pytest.mark.parametrize("n", [2, 4, 8, 16, 32, 64])
def test_fft_counts_n_log_n_to_the_formula(n):
    re, im = np.random.default_rng(n).standard_normal((2, n)).tolist()
    counts = instrumented_run(fft_radix2, re, im)["counts"]
    f = fft_formula_counts(n)
    assert counts["mul"] == f["mul"] == 2 * n * int(math.log2(n))
    assert counts["add"] == f["add"] == 3 * n * int(math.log2(n))
    assert counts["div"] == 0 and counts["total"] == f["total"] and counts["untraceable_outputs"] == 0


@pytest.mark.parametrize("n", [2, 4, 8, 16, 32])
def test_naive_transform_counts_n_squared_to_the_formula(n):
    re, im = np.random.default_rng(n).standard_normal((2, n)).tolist()
    counts = instrumented_run(naive_dft, re, im)["counts"]
    f = naive_formula_counts(n)
    assert counts["mul"] == f["mul"] == 4 * n * n and counts["add"] == f["add"] == 4 * n * n


def test_scaling_between_sizes():
    c16 = instrumented_run(naive_dft, *np.random.default_rng(1).standard_normal((2, 16)).tolist())["counts"]["total"]
    c32 = instrumented_run(naive_dft, *np.random.default_rng(1).standard_normal((2, 32)).tolist())["counts"]["total"]
    assert c32 == 4 * c16
    f16 = instrumented_run(fft_radix2, *np.random.default_rng(1).standard_normal((2, 16)).tolist())["counts"]["total"]
    f32 = instrumented_run(fft_radix2, *np.random.default_rng(1).standard_normal((2, 32)).tolist())["counts"]["total"]
    assert f32 / f16 == pytest.approx((32 * 5) / (16 * 4))


@pytest.mark.parametrize("n", [8, 16, 64])
def test_fft_certificate_passes_and_beats_the_naive_bound(n):
    c = OperationCountCertifier("dft", n, best_known_ops=5 * n * math.log2(n)).certify(fft_radix2, target="t", seed=0)
    assert c.passed and c.optimal and c.score == 5 * n * math.log2(n)
    assert c.details["correct"] and c.details["max_relative_error"] < 1e-12
    assert c.details["under_cap"] and c.details["ops_cap"] == 8 * n * n


@pytest.mark.parametrize("n", [8, 16])
def test_naive_certificate_is_correct_but_over_the_cap(n):
    c = OperationCountCertifier("dft", n).certify(naive_dft, target="t", seed=0)
    assert not c.passed and c.details["correct"] and not c.details["under_cap"]
    assert c.score == 8 * n * n
    # with a cap that admits it, it passes but is not optimal
    c2 = OperationCountCertifier("dft", n, max_ops=1e9, best_known_ops=5 * n * math.log2(n)).certify(naive_dft, target="t", seed=0)
    assert c2.passed and c2.optimal is False


def test_wrong_transform_fails_however_cheap():
    def cheap_wrong(re, im):
        return [x + 0.0 for x in re], [y + 0.0 for y in im]   # N additions, wrong answer

    c = OperationCountCertifier("dft", 8).certify(cheap_wrong, target="t", seed=0)
    assert not c.passed and not c.details["correct"] and c.score == 16


def test_untraceable_output_fails():
    def leaky(re, im):
        out_re, out_im = naive_dft(re, im)
        return [o.value for o in out_re], out_im   # left counted arithmetic: plain floats

    c = OperationCountCertifier("dft", 4, max_ops=1e9).certify(leaky, target="t", seed=0)
    assert not c.passed and c.details["untraceable_outputs"] == 4


def test_shared_subexpressions_count_once_and_dead_code_counts_nothing():
    a, b = wrap_inputs([1.5, 2.5])
    p = a * b
    out = [p + 1.0, p - 1.0]        # the product is one multiplication, used twice
    _ = a * a * a * a * b           # never reaches an output
    counts = count_ops(out)
    assert counts == {"add": 2, "mul": 1, "div": 0, "total": 3, "untraceable_outputs": 0}
    assert values(out) == [1.5 * 2.5 + 1.0, 1.5 * 2.5 - 1.0]


def test_constants_negation_and_powers():
    (x,) = wrap_inputs([3.0])
    assert count_ops([-x])["total"] == 0                     # a sign flip is free
    assert count_ops([2.0 * x])["mul"] == 1                  # a constant factor is one multiplication
    assert count_ops([x ** 3])["mul"] == 2                   # x*x*x
    assert count_ops([x / 2.0 + 1.0])["div"] == 1 and count_ops([x / 2.0 + 1.0])["add"] == 1
    assert count_ops([x ** 0])["total"] == 0 and values([x ** 0]) == [1.0]
    with pytest.raises(TypeError):
        float(x)
    with pytest.raises(TypeError):
        bool(x)
    with pytest.raises(TypeError):
        x ** 0.5
    assert math.isclose(values(x * 2 + 1)[0] if isinstance(values(x * 2 + 1), list) else values(x * 2 + 1), 7.0)
    assert isinstance(Counted(1.0) + 2, Counted)


@pytest.mark.parametrize("expr,n,value", [("5*n*log2(n)", 16, 320.0), ("8*n**2", 4, 128.0), ("n*(n-1)/2", 5, 10.0),
                                          ("sqrt(n)*ceil(n/3)", 9, 9.0)])
def test_cost_model_evaluation(expr, n, value):
    assert eval_cost_model(expr, n) == pytest.approx(value)


@pytest.mark.parametrize("bad", ["__import__('os')", "n + m", "open('x')", "n if n else 1", "[n]"])
def test_cost_model_rejects_everything_but_arithmetic(bad):
    with pytest.raises((ValueError, SyntaxError)):
        eval_cost_model(bad, 4)


def test_cost_model_checked_against_instrumented_runs():
    good = OperationCountCertifier("dft", 16).certify({"program": fft_radix2, "cost_model": "5*n*log2(n)"}, target="t", seed=0)
    assert good.passed and good.details["cost_model_ok"] and all(c["ok"] for c in good.details["cost_model_checks"])
    bad = OperationCountCertifier("dft", 16).certify({"program": fft_radix2, "cost_model": "n*log2(n)"}, target="t", seed=0)
    assert not bad.passed and bad.details["cost_model_ok"] is False


def test_instrumented_sandbox_result_is_accepted():
    cz = OperationCountCertifier("dft", 8)
    re, im = cz.inputs_for(seed=3)
    res = instrumented_run(fft_radix2, re, im)                # what the instrumented sandbox returns
    c = cz.certify({"counts": res["counts"], "outputs": res["outputs"]}, target="t", seed=3)
    assert c.passed and c.score == 5 * 8 * 3
    # the same outputs under a different seed are wrong inputs: fail
    assert not cz.certify({"counts": res["counts"], "outputs": res["outputs"]}, target="t", seed=4).passed


def test_malformed_candidates_fail_without_crashing():
    cz = OperationCountCertifier("dft", 8)
    assert cz.certify("nonsense", target="t", seed=0).error
    assert cz.certify(lambda re, im: ([re[0]], [im[0]]), target="t", seed=0).error     # wrong length
    assert cz.certify({"counts": {"total": 1}, "outputs": [1, 2, 3]}, target="t", seed=0).error


def test_deterministic_and_versioned():
    a = OperationCountCertifier("dft", 16).certify(fft_radix2, target="t", seed=9).to_dict()
    b = OperationCountCertifier("dft", 16).certify(fft_radix2, target="t", seed=9).to_dict()
    a.pop("runtime_s"), b.pop("runtime_s")
    assert a == b and a["certifier"] == "operation_count" and a["version"] == "1.0.0"
