"""Known-answer cases for the witness check: synthetic breaks reproduce, synthetic non-breaks do not,
instances outside the class definition score zero."""
import pytest

from lab.certifiers.quadrature import corrupt_weight, gauss_legendre
from lab.certifiers.witness import WitnessCertifier

W = WitnessCertifier()


def _w(domain="interval", degree=7, exponents=None, kind="exactness_error", **extra):
    d = {"family": "quadrature", "kind": kind, "domain": domain, "degree": degree, "input": {}}
    if exponents is not None:
        d["input"]["exponents"] = list(exponents)
    d.update(extra)
    return d


@pytest.mark.parametrize("k", [0, 2, 4, 6])
def test_corrupted_gauss_break_reproduces_on_even_monomials(k):
    cert = W.certify(corrupt_weight(gauss_legendre(4), 0, 1e-3), target="t", seed=0, witness=_w(exponents=[k]))
    assert cert.passed and cert.details["reproduces"] and cert.score == 1.0


@pytest.mark.parametrize("k", [0, 1, 2, 3, 4, 5, 6, 7])
def test_intact_gauss_is_not_broken(k):
    cert = W.certify(gauss_legendre(4), target="t", seed=0, witness=_w(exponents=[k]))
    assert not cert.passed and not cert.details["reproduces"] and cert.score == 0.0
    assert cert.details["error"] < 1e-12


def test_symmetric_corruption_misses_odd_monomial():
    # corrupting a symmetric pair equally keeps odd monomials exact: the claimed break does not reproduce
    nodes, weights = gauss_legendre(4)
    w = weights.copy()
    w[0] += 1e-3
    w[3] += 1e-3
    assert W.certify((nodes, w), target="t", seed=0, witness=_w(exponents=[1])).details["reproduces"] is False
    assert W.certify((nodes, w), target="t", seed=0, witness=_w(exponents=[2])).details["reproduces"] is True


def test_coefficient_polynomial_witness():
    witness = _w()
    witness["input"] = {"coefficients": {"0": 1.0, "2": -0.5, "6": 1.0}}
    cert = W.certify(corrupt_weight(gauss_legendre(4), 1, 1e-4), target="t", seed=0, witness=witness)
    assert cert.passed and cert.details["terms"] == 3 and cert.details["tol"] == pytest.approx(3e-9)
    assert not W.certify(gauss_legendre(4), target="t", seed=0, witness=witness).passed


def test_coefficients_above_one_are_outside_the_class():
    # large coefficients would scale rounding error past the tolerance of any floating-point rule: not a break
    witness = _w()
    witness["input"] = {"coefficients": {"0": 1.0, "2": -0.5, "6": 2.0}}
    cert = W.certify(corrupt_weight(gauss_legendre(4), 1, 1e-4), target="t", seed=0, witness=witness)
    assert not cert.passed and cert.score == 0.0 and "outside the class" in cert.error and "absolute value" in cert.error
    witness["input"] = {"coefficients": {"0": 1e9}}
    assert "outside the class" in W.certify(gauss_legendre(4), target="t", seed=0, witness=witness).error


def test_polynomial_tolerance_scales_with_its_terms_so_rounding_is_not_a_break():
    # the intact rule has rounding-level error on every monomial; a 36-term polynomial with unit coefficients stays below tol * terms
    witness = _w()
    witness["input"] = {"coefficients": {str(k): (1.0 if k % 2 == 0 else -1.0) for k in range(8)}}
    cert = W.certify(gauss_legendre(4), target="t", seed=0, witness=witness)
    assert not cert.passed and cert.details["terms"] == 8 and cert.details["error"] < cert.details["tol"]


@pytest.mark.parametrize("inp", [{"exponents": [-1]}, {"exponents": ["a"]}, {"coefficients": {"x": 1.0}}, {"coefficients": {"0": "nan"}},
                                 {"exponents": []}, {"nothing": 1}])
def test_malformed_inputs_score_zero(inp):
    witness = _w()
    witness["input"] = inp
    cert = W.certify(gauss_legendre(4), target="t", seed=0, witness=witness)
    assert not cert.passed and cert.score == 0.0 and cert.error


def test_degree_above_target_is_outside_the_class():
    # Gauss 4 is not exact for x^8, but a degree-8 witness against a degree-7 target scores zero
    cert = W.certify(gauss_legendre(4), target="t", seed=0, witness=_w(exponents=[8]))
    assert not cert.passed and cert.score == 0.0 and "outside the class" in cert.error


def test_wrong_dimension_is_outside_the_class():
    cert = W.certify(gauss_legendre(4), target="t", seed=0, witness=_w(exponents=[1, 1]))
    assert not cert.passed and "outside the class" in cert.error


def test_unknown_domain_and_family_score_zero():
    assert "outside the class" in W.certify(gauss_legendre(4), target="t", seed=0, witness=_w(domain="disc", exponents=[0])).error
    assert "outside the class" in W.certify(gauss_legendre(4), target="t", seed=0,
                                            witness={"family": "pde", "kind": "x"}).error


def test_missing_witness_is_an_error():
    cert = W.certify(gauss_legendre(4), target="t", seed=0)
    assert not cert.passed and cert.error


def test_exception_witness_reproduces_on_malformed_rule():
    cert = W.certify(([0.0, 1.0], [1.0]), target="t", seed=0, witness=_w(exponents=[0], kind="exception"))
    assert cert.passed and cert.details["reproduces"]


def test_exception_witness_does_not_reproduce_on_sound_rule():
    cert = W.certify(gauss_legendre(2), target="t", seed=0, witness=_w(degree=3, exponents=[0], kind="exception"))
    assert not cert.passed and not cert.details["reproduces"]


@pytest.mark.parametrize("domain,rule_deg,exps", [("square", 3, [4, 0]), ("triangle", 2, [3, 0])])
def test_two_dimensional_witnesses(domain, rule_deg, exps):
    from lab.certifiers.quadrature import square_rule, tensor_rule, triangle_rule

    rule = square_rule("stroud-4") if domain == "square" else triangle_rule("strang-fix-3")
    # the rule is exact only up to rule_deg; a witness at a higher monomial but within a higher target degree reproduces
    witness = _w(domain=domain, degree=rule_deg + 2, exponents=exps)
    assert W.certify(rule, target="t", seed=0, witness=witness).details["reproduces"] is True
    better = tensor_rule(gauss_legendre(3), 2) if domain == "square" else triangle_rule("radon-7")
    assert W.certify(better, target="t", seed=0, witness=witness).details["reproduces"] is False


# --- first-order rate witnesses ---------------------------------------------------------------------
def _fo(method, params, n, claimed, curvature):
    cand = {"method": method, "params": params, "n": n}
    return cand, {"family": "first_order", "kind": "rate_violation", "claimed_rate": claimed, "curvature": curvature}


@pytest.mark.parametrize("gamma,n", [(1.0, 1), (1.0, 3), (1.5, 2)])
def test_gd_claimed_rate_below_quadratic_instance_reproduces(gamma, n):
    params = {"mu": 0.1, "L": 1.0, "gamma": gamma}
    true = max(abs(1 - gamma * 0.1), abs(1 - gamma * 1.0)) ** (2 * n)
    worst_c = 0.1 if abs(1 - gamma * 0.1) >= abs(1 - gamma) else 1.0
    cand, wit = _fo("gradient_descent", params, n, claimed=0.5 * true, curvature=worst_c)
    cert = W.certify(cand, target="t", seed=0, witness=wit)
    assert cert.passed and cert.details["observed_ratio"] == pytest.approx(true)


@pytest.mark.parametrize("gamma,n", [(1.0, 1), (1.0, 3), (1.5, 2)])
def test_gd_correct_rate_is_not_broken(gamma, n):
    params = {"mu": 0.1, "L": 1.0, "gamma": gamma}
    true = max(abs(1 - gamma * 0.1), abs(1 - gamma * 1.0)) ** (2 * n)
    for c in (0.1, 0.5, 1.0):
        cand, wit = _fo("gradient_descent", params, n, claimed=true, curvature=c)
        assert not W.certify(cand, target="t", seed=0, witness=wit).passed


def test_curvature_outside_class_scores_zero():
    cand, wit = _fo("gradient_descent", {"mu": 0.1, "L": 1.0, "gamma": 1.0}, 1, claimed=0.1, curvature=5.0)
    cert = W.certify(cand, target="t", seed=0, witness=wit)
    assert not cert.passed and cert.score == 0.0 and "outside the class" in cert.error


def test_douglas_rachford_quadratic_instance_witness():
    params = {"mu": 0.1, "L": 1.0, "alpha": 3.0, "theta": 1.0}
    tight = max(1 / (1 + 0.1 * 3), 3 / (1 + 3)) ** 2
    cand, wit = _fo("douglas_rachford", params, 1, claimed=0.9 * tight, curvature=0.1)
    assert W.certify(cand, target="t", seed=0, witness=wit).passed
    cand, wit = _fo("douglas_rachford", params, 1, claimed=tight, curvature=0.1)
    assert not W.certify(cand, target="t", seed=0, witness=wit).passed


def test_nesterov_quadratic_instance_witness():
    params = {"mu": 0.1, "L": 1.0}
    # on the quadratic with curvature L the method converges in one step (ratio 0); curvature mu is the slow instance
    cand, wit = _fo("nesterov_strongly_convex", params, 2, claimed=0.01, curvature=1.0)
    assert not W.certify(cand, target="t", seed=0, witness=wit).passed
    cand, wit = _fo("nesterov_strongly_convex", params, 2, claimed=0.01, curvature=0.1)
    cert = W.certify(cand, target="t", seed=0, witness=wit)
    assert cert.passed and 0.29 < cert.details["observed_ratio"] < 0.4676
