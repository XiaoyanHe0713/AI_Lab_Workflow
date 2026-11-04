"""Known-answer cases for the convergence-order certifier: manufactured solutions of known order measure
their order with a tight interval, a deliberately wrong boundary term lowers the measured order, the fit
handles the machine-precision floor, and the certificate is deterministic under a seed."""
import numpy as np
import pytest

from lab.certifiers.order import (
    ConvergenceOrderCertifier,
    composite,
    euler,
    exact_solution,
    fit_order,
    heun,
    make_problem,
    poisson_fd,
    rk4,
)


def cert(family, order, solver, seed=0, **kw):
    return ConvergenceOrderCertifier(family, order, **kw).certify(solver, target=f"order-{family}", seed=seed)


@pytest.mark.parametrize("solver,order", [(euler, 1), (heun, 2), (rk4, 4)])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_time_steppers_measure_their_order(solver, order, seed):
    c = cert("ode_ivp", order, solver, seed=seed)
    assert c.passed, c.details
    assert c.score == pytest.approx(order, abs=0.15)
    lo, hi = c.details["order_interval"]
    assert lo - 0.15 <= order <= hi + 0.15
    assert all(abs(p - order) < 0.25 for p in c.details["pairwise_orders"][-2:])


@pytest.mark.parametrize("seed", [0, 3, 7])
def test_second_order_finite_differences(seed):
    c = cert("poisson1d", 2, lambda p, h: poisson_fd(p, h), seed=seed)
    assert c.passed and c.score == pytest.approx(2.0, abs=0.05)


@pytest.mark.parametrize("seed", [0, 3, 7])
def test_wrong_boundary_term_lowers_the_order(seed):
    c = cert("poisson1d", 2, lambda p, h: poisson_fd(p, h, boundary="wrong"), seed=seed)
    assert not c.passed
    assert c.score < 1.3 and c.details["checks"]["slope_meets_claim"] is False
    # the same wrong scheme does meet a first-order claim
    assert cert("poisson1d", 1, lambda p, h: poisson_fd(p, h, boundary="wrong"), seed=seed).passed


@pytest.mark.parametrize("rule,order", [("left", 1), ("midpoint", 2), ("trapezoid", 2), ("simpson", 4)])
def test_composite_quadrature_orders(rule, order):
    c = cert("quadrature", order, lambda p, h: composite(p, h, rule), seed=0)
    assert c.passed, c.details
    assert c.score == pytest.approx(order, abs=0.1)


def test_claiming_too_much_fails():
    c = cert("ode_ivp", 2, euler, seed=0)
    assert not c.passed and c.score == pytest.approx(1.0, abs=0.1)
    c = cert("quadrature", 4, lambda p, h: composite(p, h, "trapezoid"), seed=0)
    assert not c.passed


def test_higher_order_than_claimed_passes_and_is_optimal_against_best_known():
    c = ConvergenceOrderCertifier("ode_ivp", 2, best_known_order=4).certify(rk4, target="t", seed=0)
    assert c.passed and c.optimal is True
    c = ConvergenceOrderCertifier("ode_ivp", 2, best_known_order=4).certify(heun, target="t", seed=0)
    assert c.passed and c.optimal is False


def test_machine_precision_levels_are_dropped_from_the_fit():
    hs = [0.1, 0.05, 0.025, 0.0125, 0.00625]
    errs = [1e-2, 2.5e-3, 6.25e-4, 1e-13, 1e-14]
    f = fit_order(hs, errs)
    assert f["levels_dropped"] == 2 and f["levels_used"] == 3 and f["slope"] == pytest.approx(2.0, abs=1e-9)
    f2 = fit_order(hs, [1e-13] * 5)
    assert f2["slope"] is None
    c = ConvergenceOrderCertifier("ode_ivp", 2).certify({"h": hs, "errors": [1e-13] * 5}, target="t", seed=0)
    assert not c.passed and "too few levels" in c.error


def test_measured_data_candidate_and_interval():
    hs = [0.1, 0.05, 0.025, 0.0125]
    errs = [h ** 3 * 2.0 for h in hs]
    c = ConvergenceOrderCertifier("ode_ivp", 3).certify({"h": hs, "errors": errs}, target="t", seed=0)
    assert c.passed and c.score == pytest.approx(3.0, abs=1e-9)
    lo, hi = c.details["order_interval"]
    assert hi - lo < 1e-6


def test_list_of_solutions_candidate_matches_callable():
    cz = ConvergenceOrderCertifier("ode_ivp", 2)
    prob = cz.problem_for(5)
    sols = [heun(prob, h) for h in cz.hs()]
    a = cz.certify(sols, target="t", seed=5)
    b = cz.certify(heun, target="t", seed=5)
    assert a.passed and a.score == b.score


def test_malformed_candidates_fail_without_crashing():
    cz = ConvergenceOrderCertifier("poisson1d", 2)
    assert cz.certify([np.zeros(3)] * 5, target="t", seed=0).error            # wrong shape
    assert cz.certify({"h": [1, 2], "errors": [1]}, target="t", seed=0).error  # mismatched lengths
    assert cz.certify("nonsense", target="t", seed=0).error
    assert cz.certify([np.full(9, np.nan)] * 5, target="t", seed=0).error      # not finite


def test_seeded_problems_are_deterministic_and_vary_with_the_seed():
    p0 = make_problem("ode_ivp", np.random.default_rng(0))
    assert p0 == make_problem("ode_ivp", np.random.default_rng(0))
    assert p0 != make_problem("ode_ivp", np.random.default_rng(1))
    a = cert("ode_ivp", 2, heun, seed=11).to_dict()
    b = cert("ode_ivp", 2, heun, seed=11).to_dict()
    a.pop("runtime_s"), b.pop("runtime_s")
    assert a == b and a["certifier"] == "convergence_order" and a["version"] == "1.0.0"


def test_exact_solutions_are_what_the_docstring_says():
    p = {"family": "quadrature", "a": 1.0, "b": 2.0}
    from scipy.integrate import quad

    val, _ = quad(lambda x: np.exp(x) * np.cos(2 * x), 0, 1)
    assert exact_solution(p, 0.1) == pytest.approx(val, abs=1e-12)
    q = {"family": "poisson1d", "k": 2, "alpha": 1.5, "left": 0.0, "right": 1.5}
    u = exact_solution(q, 0.25)
    assert u[0] == pytest.approx(0.0) and u[-1] == pytest.approx(1.5) and len(u) == 5
    assert exact_solution({"family": "ode_ivp", "omega": 2.0, "t_end": 1.0}, 0.1) == pytest.approx(np.cos(2.0))
