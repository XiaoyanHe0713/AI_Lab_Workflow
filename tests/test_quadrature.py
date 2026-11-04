"""Known-answer cases for the exactness certifier: Gauss rules pass, corrupted rules fail, classical
rules have their tabulated degrees, and the certificate is deterministic under a seed."""
import numpy as np
import pytest

from lab.certifiers.quadrature import (
    QuadratureExactnessCertifier,
    clenshaw_curtis,
    corrupt_weight,
    duffy_triangle,
    exactness_degree,
    gauss_legendre,
    monomial_exact,
    monomials_up_to,
    newton_cotes,
    normalize_rule,
    square_rule,
    tensor_rule,
    triangle_rule,
)


@pytest.mark.parametrize("n", range(1, 11))
def test_gauss_legendre_exact_to_2n_minus_1(n):
    cert = QuadratureExactnessCertifier("interval", 2 * n - 1).certify(gauss_legendre(n), target="t", seed=0)
    assert cert.passed, cert.details
    assert cert.score == n
    assert cert.details["max_monomial_error"] < 1e-12
    assert cert.details["all_weights_positive"] and cert.details["all_nodes_interior"]


@pytest.mark.parametrize("n", range(1, 11))
def test_gauss_legendre_not_exact_to_2n(n):
    cert = QuadratureExactnessCertifier("interval", 2 * n).certify(gauss_legendre(n), target="t", seed=0)
    assert not cert.passed
    assert cert.details["worst_monomial"] == [2 * n]
    assert cert.details["measured_exactness_degree"] == 2 * n - 1


@pytest.mark.parametrize("n,index,delta", [(4, 0, 1e-3), (4, 2, -1e-6), (6, 5, 1e-8), (2, 1, 1e-4)])
def test_one_corrupted_weight_fails(n, index, delta):
    cert = QuadratureExactnessCertifier("interval", 2 * n - 1).certify(corrupt_weight(gauss_legendre(n), index, delta),
                                                                       target="t", seed=0)
    assert not cert.passed
    assert cert.details["max_monomial_error"] > 1e-9


@pytest.mark.parametrize("n,degree", [(2, 1), (3, 3), (4, 3), (5, 5), (6, 5), (7, 7), (9, 9)])
def test_clenshaw_curtis_degrees(n, degree):
    x, w = clenshaw_curtis(n)
    assert exactness_degree(x, w, "interval", 20) == degree
    assert QuadratureExactnessCertifier("interval", degree).certify((x, w), target="t", seed=0).passed


@pytest.mark.parametrize("kind,degree", [("midpoint", 1), ("trapezoid", 1), ("simpson", 3)])
def test_newton_cotes_degrees(kind, degree):
    x, w = newton_cotes(kind)
    assert exactness_degree(x, w, "interval", 10) == degree
    assert not QuadratureExactnessCertifier("interval", degree + 1).certify((x, w), target="t", seed=0).passed


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_tensor_gauss_on_square(n):
    rule = tensor_rule(gauss_legendre(n), 2)
    assert QuadratureExactnessCertifier("square", 2 * n - 1).certify(rule, target="t", seed=0).passed
    assert not QuadratureExactnessCertifier("square", 2 * n).certify(rule, target="t", seed=0).passed


def test_tensor_gauss_on_cube():
    rule = tensor_rule(gauss_legendre(2), 3)
    cert = QuadratureExactnessCertifier("cube", 3).certify(rule, target="t", seed=0)
    assert cert.passed and cert.score == 8
    assert abs(cert.details["weight_sum"] - 8.0) < 1e-12


def test_stroud_four_point_square_rule():
    x, w = square_rule("stroud-4")
    assert exactness_degree(x, w, "square", 10) == 3
    cert = QuadratureExactnessCertifier("square", 3, best_known_nodes=4).certify((x, w), target="t", seed=0)
    assert cert.passed and cert.optimal


@pytest.mark.parametrize("name,degree,n", [("centroid", 1, 1), ("strang-fix-3", 2, 3), ("edge-midpoints", 2, 3),
                                           ("strang-fix-4", 3, 4), ("dunavant-6", 4, 6), ("radon-7", 5, 7)])
def test_classical_triangle_rules(name, degree, n):
    x, w = triangle_rule(name)
    assert len(w) == n
    assert abs(w.sum() - 0.5) < 1e-12
    assert exactness_degree(x, w, "triangle", 12) == degree
    cert = QuadratureExactnessCertifier("triangle", degree).certify((x, w), target="t", seed=0)
    assert cert.passed
    assert not QuadratureExactnessCertifier("triangle", degree + 1).certify((x, w), target="t", seed=0).passed


def test_negative_weight_rule_fails_when_positivity_required():
    rule = triangle_rule("strang-fix-4")
    assert QuadratureExactnessCertifier("triangle", 3).certify(rule, target="t", seed=0).passed
    cert = QuadratureExactnessCertifier("triangle", 3, require_positive=True).certify(rule, target="t", seed=0)
    assert not cert.passed and not cert.details["all_weights_positive"]


def test_exterior_node_rule_fails_when_interiority_required():
    nodes, weights = gauss_legendre(2)
    nodes = nodes * 2.0  # still a valid degree-1 rule (symmetric, weights sum to 2) but outside [-1, 1]
    assert QuadratureExactnessCertifier("interval", 1).certify((nodes, weights), target="t", seed=0).passed
    cert = QuadratureExactnessCertifier("interval", 1, require_interior=True).certify((nodes, weights), target="t", seed=0)
    assert not cert.passed and not cert.details["all_nodes_interior"]


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_duffy_collapsed_triangle_rule(n):
    x, w = duffy_triangle(n)
    assert np.all(w > 0)
    assert exactness_degree(x, w, "triangle", 16) == 2 * n - 2


def test_exact_monomial_integrals():
    assert monomial_exact("interval", (0,)) == 2.0
    assert monomial_exact("interval", (2,)) == pytest.approx(2 / 3)
    assert monomial_exact("interval", (3,)) == 0.0
    assert monomial_exact("square", (2, 4)) == pytest.approx((2 / 3) * (2 / 5))
    assert monomial_exact("triangle", (0, 0)) == 0.5
    assert monomial_exact("triangle", (1, 0)) == pytest.approx(1 / 6)
    assert monomial_exact("triangle", (1, 1)) == pytest.approx(1 / 24)
    assert monomial_exact("triangle", (2, 1)) == pytest.approx(2 / 120)


def test_basis_sizes():
    assert len(monomials_up_to(7, 1)) == 8
    assert len(monomials_up_to(5, 2)) == 21
    assert len(monomials_up_to(3, 3)) == 20


def test_certificate_is_deterministic_under_seed():
    c = QuadratureExactnessCertifier("interval", 7, n_random=64)
    a = c.certify(gauss_legendre(4), target="quad-d7", seed=123).to_dict()
    b = c.certify(gauss_legendre(4), target="quad-d7", seed=123).to_dict()
    for k in ("runtime_s",):
        a.pop(k), b.pop(k)
    assert a == b
    assert a["certifier"] == "quadrature_exactness" and a["version"] == "1.0.0" and a["seed"] == 123


def test_optimal_flag_against_best_known():
    c = QuadratureExactnessCertifier("interval", 7, best_known_nodes=4)
    assert c.certify(gauss_legendre(4), target="t", seed=0).optimal is True
    assert c.certify(gauss_legendre(5), target="t", seed=0).optimal is False
    assert c.certify(gauss_legendre(3), target="t", seed=0).optimal is False


@pytest.mark.parametrize("bad", [([0.0, 0.5], [2.0]), ([], []), ([float("nan")], [2.0]), ([[0.0, 0.0]], [2.0])])
def test_malformed_rules_fail_without_crashing(bad):
    cert = QuadratureExactnessCertifier("interval", 1).certify(bad, target="t", seed=0)
    assert not cert.passed and cert.error


def test_dict_candidate_forms_are_accepted():
    x, w = gauss_legendre(3)
    assert QuadratureExactnessCertifier("interval", 5).certify({"nodes": x.ravel().tolist(), "weights": w.tolist()}, target="t", seed=0).passed
    assert QuadratureExactnessCertifier("interval", 5).certify({"x": x, "w": w}, target="t", seed=0).passed
    nodes, weights = normalize_rule({"nodes": [0.0], "weights": [2.0]}, "interval")
    assert nodes.shape == (1, 1)
