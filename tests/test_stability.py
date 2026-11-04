"""Known-answer cases for the stability-region and SSP certifier: classical Runge-Kutta methods reproduce the
tabulated real- and imaginary-axis extents and orders, SSP methods reproduce their coefficients from the
Shu-Osher form and from the radius of absolute monotonicity, non-SSP methods get zero, and the stability
polynomial agrees with the method on the test equation."""
import math

import numpy as np
import pytest

from lab.certifiers.stability import (
    TABULATED,
    StabilityCertifier,
    butcher,
    classical_order,
    normalize_method,
    shu_osher,
    shu_osher_coefficient,
    shu_osher_to_butcher,
    ssp_coefficient,
    stability_polynomial,
)


@pytest.mark.parametrize("name", ["euler", "heun2", "kutta3", "rk4"])
def test_classical_methods_reproduce_tabulated_regions(name):
    t = TABULATED[name]
    c = StabilityCertifier("real_axis", t["order"]).certify(butcher(name), target=name, seed=0)
    assert c.passed, c.details["checks"]
    assert c.score == pytest.approx(t["real"], abs=1e-6)
    assert c.details["imag_axis_extent"] == pytest.approx(t["imag"], abs=1e-6)
    assert c.details["measured_order"] == t["order"]
    assert c.details["ssp_coefficient"] == pytest.approx(t["ssp"], abs=1e-7)


@pytest.mark.parametrize("name,poly", [
    ("euler", [1, 1]), ("heun2", [1, 1, 0.5]), ("midpoint2", [1, 1, 0.5]), ("ralston2", [1, 1, 0.5]),
    ("kutta3", [1, 1, 0.5, 1 / 6]), ("heun3", [1, 1, 0.5, 1 / 6]), ("rk4", [1, 1, 0.5, 1 / 6, 1 / 24]),
    ("rk38", [1, 1, 0.5, 1 / 6, 1 / 24]),
])
def test_stability_polynomials(name, poly):
    m = normalize_method(butcher(name))
    assert np.allclose(stability_polynomial(m["A"], m["b"]), poly, atol=1e-12)


@pytest.mark.parametrize("name,order", [("euler", 1), ("heun2", 2), ("midpoint2", 2), ("ralston2", 2), ("kutta3", 3),
                                        ("heun3", 3), ("rk4", 4), ("rk38", 4)])
def test_order_conditions(name, order):
    m = normalize_method(butcher(name))
    assert classical_order(m["A"], m["b"], m["c"]) == order


@pytest.mark.parametrize("name,coef,order", [("ssprk22", 1.0, 2), ("ssprk33", 1.0, 3), ("ssprk43", 2.0, 3), ("ssprk54", 1.508, 4)])
def test_ssp_methods_reproduce_their_coefficients(name, coef, order):
    c = StabilityCertifier("ssp", order).certify(shu_osher(name), target=name, seed=0)
    assert c.passed, c.details["checks"]
    assert c.score == pytest.approx(coef, abs=2e-3)
    assert c.details["shu_osher_coefficient"] == pytest.approx(coef, abs=2e-3)
    assert c.details["measured_order"] == order


@pytest.mark.parametrize("s", [2, 3, 4, 5, 6, 8])
def test_optimal_second_order_family_has_coefficient_s_minus_1(s):
    c = StabilityCertifier("ssp", 2, stages=s, best_known=s - 1).certify(shu_osher("ssprk_s2", s), target="t", seed=0)
    assert c.passed and c.optimal and c.score == pytest.approx(s - 1, abs=1e-7)


@pytest.mark.parametrize("n", [2, 3])
def test_n_squared_stage_third_order_family(n):
    c = StabilityCertifier("ssp", 3, stages=n * n).certify(shu_osher("ssprk_n2_3", n), target="t", seed=0)
    assert c.passed, c.details["checks"]
    assert c.score == pytest.approx(n * n - n, abs=1e-7)


@pytest.mark.parametrize("name", ["rk4", "kutta3", "rk38"])
def test_non_ssp_methods_get_zero_and_fail_the_ssp_certificate(name):
    c = StabilityCertifier("ssp", TABULATED.get(name, {}).get("order", 3)).certify(butcher(name), target=name, seed=0)
    assert not c.passed and c.score == 0.0 and c.details["checks"]["ssp_coefficient_positive"] is False


def test_shu_osher_conversion_matches_butcher_for_ssprk33():
    A, b = shu_osher_to_butcher(**shu_osher("ssprk33"))
    assert np.allclose(A, [[0, 0, 0], [1, 0, 0], [0.25, 0.25, 0]]) and np.allclose(b, [1 / 6, 1 / 6, 2 / 3])
    # the Shu-Osher coefficient of the given representation equals the radius of absolute monotonicity here
    so = shu_osher("ssprk33")
    assert shu_osher_coefficient(np.array(so["alpha"]), np.array(so["beta"])) == pytest.approx(1.0)
    assert ssp_coefficient(A, b) == pytest.approx(1.0, abs=1e-7)


def test_heun_in_shu_osher_form_equals_heun_in_butcher_form():
    a = StabilityCertifier("real_axis", 2).certify(shu_osher("ssprk22"), target="t", seed=0)
    b = StabilityCertifier("real_axis", 2).certify(butcher("heun2"), target="t", seed=0)
    assert a.score == pytest.approx(b.score) and a.details["stability_polynomial"] == pytest.approx(b.details["stability_polynomial"])


def test_claiming_too_high_an_order_fails():
    c = StabilityCertifier("real_axis", 3).certify(butcher("heun2"), target="t", seed=0)
    assert not c.passed and c.details["checks"]["order_conditions_to_claimed_order"] is False
    c = StabilityCertifier("ssp", 2, stages=3).certify(shu_osher("ssprk22"), target="t", seed=0)
    assert not c.passed and c.details["checks"]["stage_count"] is False


def test_imag_axis_certificate_and_optimal_flag():
    c = StabilityCertifier("imag_axis", 4, best_known=2 * math.sqrt(2)).certify(butcher("rk4"), target="t", seed=0)
    assert c.passed and c.optimal and c.score == pytest.approx(2 * math.sqrt(2), abs=1e-6)
    c = StabilityCertifier("imag_axis", 3, best_known=2 * math.sqrt(2)).certify(butcher("kutta3"), target="t", seed=0)
    assert c.passed and c.optimal is False and c.score == pytest.approx(math.sqrt(3), abs=1e-6)


def test_region_area_orders_the_classical_methods():
    areas = {n: StabilityCertifier("real_axis", TABULATED[n]["order"]).certify(butcher(n), target=n, seed=0).details["region_area"]
             for n in ("euler", "heun2", "kutta3", "rk4")}
    assert areas["euler"] == pytest.approx(math.pi, abs=0.02)
    assert areas["rk4"] > areas["kutta3"] > areas["heun2"] > 0


def test_implicit_and_malformed_methods_fail_without_crashing():
    implicit = {"A": [[0.5]], "b": [1.0]}   # implicit midpoint
    c = StabilityCertifier("real_axis", 2).certify(implicit, target="t", seed=0)
    assert not c.passed and "explicit" in c.error
    assert StabilityCertifier("ssp", 1).certify({"A": [[0, 0]], "b": [1]}, target="t", seed=0).error
    assert StabilityCertifier("ssp", 1).certify("nonsense", target="t", seed=0).error
    bad_rows = {"alpha": [[0.5, 0], [0.5, 0.5]], "beta": [[1, 0], [0, 0.5]]}
    assert StabilityCertifier("ssp", 1).certify(bad_rows, target="t", seed=0).error
    nan = {"A": [[0, 0], [float("nan"), 0]], "b": [0.5, 0.5]}
    assert StabilityCertifier("ssp", 1).certify(nan, target="t", seed=0).error


def test_deterministic_and_versioned():
    a = StabilityCertifier("ssp", 3).certify(shu_osher("ssprk33"), target="t", seed=4).to_dict()
    b = StabilityCertifier("ssp", 3).certify(shu_osher("ssprk33"), target="t", seed=4).to_dict()
    a.pop("runtime_s"), b.pop("runtime_s")
    assert a == b and a["certifier"] == "stability_region_ssp" and a["version"] == "1.0.0"
    assert a["details"]["checks"]["polynomial_matches_method"]
