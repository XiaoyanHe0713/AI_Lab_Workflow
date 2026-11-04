"""Known-answer cases for the performance-estimation certifier: gradient descent and Douglas-Rachford
reproduce tight published rates, Nesterov stays under its published bound and matches PEPit's reference,
Davis-Yin reproduces PEPit's reference (the calibration point), the dual reconstructs the bound, the cap
and the cache work."""
import json
import math

import pytest

from lab.certifiers.base import CertifierTimeout
from lab.certifiers.pep import PEPRateCertifier, method_signature, pepit_reference, published_rate

TOL = 1e-4


def cert(method, params, **kw):
    c = PEPRateCertifier(in_process=True, tol=TOL, **kw)
    return c.certify({"method": method, "params": params}, target=f"pep-{method}", seed=0)


def _assert_dual(c):
    d = c.details["dual_certificate"]
    assert d["S_min_eigenvalue"] >= -TOL
    assert d["f_residual"] <= TOL
    assert d["min_inequality_dual"] >= -TOL
    assert abs(d["tau_dual"] - c.details["rate"]) <= TOL * max(1.0, c.details["rate"])
    assert len(d["multipliers"]) == d["n_constraints"] > 0


@pytest.mark.parametrize("mu,L,gamma,n", [
    (0.1, 1.0, 1.0, 1), (0.1, 1.0, 1.0, 2), (0.1, 1.0, 1.0, 3), (0.1, 1.0, 1.5, 1), (0.1, 1.0, 2 / 1.1, 1),
    (0.1, 1.0, 2 / 1.1, 2), (0.2, 1.0, 0.5, 2), (0.5, 2.0, 0.4, 1), (0.05, 1.0, 1.0, 4), (0.3, 1.0, 0.9, 3),
    (0.1, 1.0, 0.25, 2), (0.1, 1.0, 1.9, 1),
])
def test_gradient_descent_matches_tight_rate(mu, L, gamma, n):
    c = cert("gradient_descent", {"mu": mu, "L": L, "gamma": gamma, "n": n})
    assert c.passed, c.details.get("checks")
    expected = max(abs(1 - gamma * mu), abs(1 - gamma * L)) ** (2 * n)
    assert c.score == pytest.approx(expected, abs=TOL)
    assert c.details["checks"]["matches_published_tight_rate"]
    _assert_dual(c)


@pytest.mark.parametrize("mu,L,n", [(0.1, 1.0, 1), (0.1, 1.0, 2), (0.01, 1.0, 2), (0.25, 1.0, 3)])
def test_nesterov_is_under_the_published_bound(mu, L, n):
    c = cert("nesterov_strongly_convex", {"mu": mu, "L": L, "n": n})
    assert c.passed, c.details.get("checks")
    bound = (1 - math.sqrt(mu / L)) ** n
    assert c.score <= bound * (1 + TOL)
    assert c.score >= c.details["instance_lower_bound"] - TOL
    _assert_dual(c)


def test_nesterov_matches_pepit_reference():
    c = cert("nesterov_strongly_convex", {"mu": 0.1, "L": 1.0, "n": 2})
    assert c.details["pepit_reference"] == 0.3476022263165957
    assert c.score == pytest.approx(0.3476022263165957, abs=TOL)
    assert c.details["checks"]["matches_pepit_reference"]


@pytest.mark.parametrize("mu,L,alpha,n", [(0.1, 1.0, 3.0, 1), (0.1, 1.0, 3.0, 2), (0.1, 1.0, 1.0, 1), (0.5, 2.0, 0.7, 1)])
def test_douglas_rachford_matches_tight_rate(mu, L, alpha, n):
    c = cert("douglas_rachford", {"mu": mu, "L": L, "alpha": alpha, "theta": 1.0, "n": n})
    assert c.passed, c.details.get("checks")
    expected = max(1 / (1 + mu * alpha), alpha * L / (1 + alpha * L)) ** (2 * n)
    assert c.score == pytest.approx(expected, abs=TOL)
    _assert_dual(c)


def test_douglas_rachford_matches_pepit_reference():
    c = cert("douglas_rachford", {"mu": 0.1, "L": 1.0, "alpha": 3.0, "theta": 1.0, "n": 2})
    assert c.score == pytest.approx(0.35012780295429113, abs=TOL) and c.details["checks"]["matches_pepit_reference"]


def test_douglas_rachford_relaxed_has_no_closed_form_but_is_certified():
    c = cert("douglas_rachford", {"mu": 0.1, "L": 1.0, "alpha": 1.0, "theta": 1.5, "n": 1})
    assert c.passed and c.details["published_rate"] is None
    assert "within_published_upper_bound" not in c.details["checks"]
    _assert_dual(c)


def test_davis_yin_calibration_point():
    """The calibration point: PEPit's documented value for TOS, reproduced with SCS within tolerance."""
    c = cert("davis_yin", {"mu1": 0.1, "L1": 10.0, "L3": 1.0, "alpha": 1.0, "theta": 1.0, "n": 4})
    assert c.passed, c.details.get("checks")
    assert c.score == pytest.approx(0.4754523346392658, abs=TOL)
    assert c.details["checks"]["matches_pepit_reference"]
    assert c.details["solve_s"] < 60
    _assert_dual(c)


def test_davis_yin_one_step_is_above_its_quadratic_instances():
    c = cert("davis_yin", {"mu1": 0.1, "L1": 10.0, "L3": 1.0, "alpha": 1.0, "theta": 1.0, "n": 1})
    assert c.passed and c.score >= c.details["instance_lower_bound"] - TOL
    _assert_dual(c)


def test_published_rate_table_and_reference_lookup():
    assert published_rate("gradient_descent", {"mu": 0.1, "L": 1.0, "gamma": 1.0, "n": 1})[0] == pytest.approx(0.81)
    assert published_rate("douglas_rachford", {"mu": 0.1, "L": 1.0, "alpha": 3.0, "theta": 2.0, "n": 1})[0] is None
    assert published_rate("davis_yin", {})[0] is None
    assert pepit_reference("davis_yin", {"mu1": 0.1, "L1": 10, "L3": 1, "alpha": 1, "theta": 1, "n": 4}) == 0.4754523346392658
    assert pepit_reference("davis_yin", {"mu1": 0.2, "L1": 10, "L3": 1, "alpha": 1, "theta": 1, "n": 4}) is None
    with pytest.raises(ValueError):
        published_rate("foo", {})


def test_a_wrong_published_value_fails_the_certificate():
    c = cert("gradient_descent", {"mu": 0.1, "L": 1.0, "gamma": 1.0, "n": 1}, published=0.5)
    assert not c.passed and not c.details["checks"]["within_published_upper_bound"]


def test_a_wrong_reference_value_fails_the_certificate():
    c = cert("gradient_descent", {"mu": 0.1, "L": 1.0, "gamma": 1.0, "n": 1}, reference=0.7)
    assert not c.passed and not c.details["checks"]["matches_pepit_reference"]


def test_bad_candidates_fail_without_crashing():
    assert cert("gradient_descent", {"mu": 0.1, "L": 1.0, "n": 1}).error  # missing gamma
    assert cert("no_such_method", {}).error
    assert cert("gradient_descent", {"mu": 0.1, "L": 1.0, "gamma": 1.0, "n": 0}).error


def test_deterministic_and_versioned():
    a = cert("gradient_descent", {"mu": 0.1, "L": 1.0, "gamma": 1.0, "n": 2})
    b = cert("gradient_descent", {"mu": 0.1, "L": 1.0, "gamma": 1.0, "n": 2})
    assert a.score == b.score and a.details["rate_dual"] == b.details["rate_dual"]
    assert a.certifier == "pep_worst_case_rate" and a.version == "1.0.0"


def test_cache_by_method_signature(tmp_path):
    c = PEPRateCertifier(in_process=True, cache_dir=tmp_path)
    cand = {"method": "gradient_descent", "params": {"mu": 0.1, "L": 1.0, "gamma": 1.0, "n": 2}}
    first = c.certify(cand, target="t", seed=0)
    second = c.certify(cand, target="t", seed=1)
    assert not first.details["cached"] and second.details["cached"]
    assert first.score == second.score
    sig = method_signature("gradient_descent", cand["params"], "SCS", PEPRateCertifier.version)
    assert (tmp_path / f"{sig}.json").exists()
    assert json.loads((tmp_path / f"{sig}.json").read_text())["tau_primal"] == first.score


def test_runtime_cap_in_a_subprocess():
    c = PEPRateCertifier(in_process=False, cap_s=0.5)
    out = c.certify({"method": "davis_yin", "params": {"mu1": 0.1, "L1": 10.0, "L3": 1.0, "alpha": 1.0, "theta": 1.0, "n": 4}},
                    target="t", seed=0)
    assert not out.passed and "timeout" in out.error


def test_subprocess_path_returns_the_same_rate():
    c = PEPRateCertifier(in_process=False, cap_s=60)
    out = c.certify({"method": "gradient_descent", "params": {"mu": 0.1, "L": 1.0, "gamma": 1.0, "n": 1}}, target="t", seed=0)
    assert out.passed and out.score == pytest.approx(0.81, abs=TOL)
