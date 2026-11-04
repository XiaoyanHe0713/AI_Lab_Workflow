"""The judge path for the week-3 certifiers: a candidate per family runs in the sandbox (the operation count
through the instrumented runner, the refinement study with seeded problems, the tableau), is certified, and
the attempt scores the same twice."""
import pytest

from lab.judge import Judge
from lab.targets import get_target, list_targets

FFT = '''
import math

def transform(re, im):
    n = len(re)
    if n == 1:
        return [re[0]], [im[0]]
    er, ei = transform(re[0::2], im[0::2])
    odr, odi = transform(re[1::2], im[1::2])
    out_re, out_im = [None] * n, [None] * n
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
'''

NAIVE = '''
import math

def transform(re, im):
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
'''

LEAKY = '''
def transform(re, im):
    return [x.value for x in re], [y.value for y in im]
'''

HEUN = '''
import math

def solve(problem, hs):
    lam, om = problem["lam"], problem["omega"]
    f = lambda t, u: lam * (u - math.cos(om * t)) - om * math.sin(om * t)
    out = []
    for h in hs:
        N = int(round(problem["t_end"] / h))
        t, u = 0.0, problem["u0"]
        for _ in range(N):
            k1 = f(t, u)
            k2 = f(t + h, u + h * k1)
            u = u + h * (k1 + k2) / 2
            t += h
        out.append(u)
    return out
'''

SSPRK33 = '''
def tableau():
    alpha = [[1, 0, 0], [0.75, 0.25, 0], [1/3, 0, 2/3]]
    beta = [[1, 0, 0], [0, 0.25, 0], [0, 0, 2/3]]
    return {"alpha": alpha, "beta": beta}
'''

RK4 = '''
def tableau():
    return {"A": [[0, 0, 0, 0], [0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 1, 0]], "b": [1/6, 1/3, 1/3, 1/6]}
'''


@pytest.fixture(scope="module")
def judge():
    return Judge(timeout_s=60, memory_mb=1024)


def test_fft_through_the_instrumented_sandbox(judge):
    t = get_target("opcount-dft-n16")
    assert t.instrument and t.instrument["wrap_args"] == [0, 1] and t.cutoff == "1965-01-01"
    cert, sb = judge.judge(FFT, t, seed=0)
    assert sb.ok and cert.passed, cert.error
    assert cert.score == 320 and cert.optimal is True
    assert cert.details["counts"]["mul"] == 128 and cert.details["counts"]["add"] == 192
    a, _ = judge.judge(FFT, t, seed=3)
    b, _ = judge.judge(FFT, t, seed=3)
    assert a.score == b.score == 320 and a.details["counts"] == b.details["counts"]


def test_naive_dft_is_correct_but_fails_the_cap_and_a_leaky_program_fails(judge):
    t = get_target("opcount-dft-n8")
    cert, _ = judge.judge(NAIVE, t, seed=0)
    assert not cert.passed and cert.details["correct"] and cert.score == 512
    cert, sb = judge.judge(LEAKY, t, seed=0)
    assert sb.ok and not cert.passed and cert.details["untraceable_outputs"] == 16


def test_refinement_study_through_the_sandbox(judge):
    t = get_target("order-ode-p2")
    cert, sb = judge.judge(HEUN, t, seed=1)
    assert sb.ok and cert.passed, cert.error
    assert cert.score == pytest.approx(2.0, abs=0.15) and cert.optimal is True
    assert cert.details["problem"] == t.entry_args_for(1)[0]
    harder = get_target("order-ode-p4")
    cert, _ = judge.judge(HEUN, harder, seed=1)
    assert not cert.passed


def test_tableaux_through_the_sandbox(judge):
    cert, sb = judge.judge(SSPRK33, get_target("ssp-s3-p3"), seed=0)
    assert sb.ok and cert.passed and cert.score == pytest.approx(1.0, abs=1e-7) and cert.optimal is True
    cert, _ = judge.judge(RK4, get_target("stab-real-s4-p4"), seed=0)
    assert cert.passed and cert.score == pytest.approx(2.7853, abs=1e-4) and cert.optimal is True
    cert, _ = judge.judge(RK4, get_target("ssp-s4-p4"), seed=0)
    assert not cert.passed and cert.score == 0.0


def test_new_targets_resolve_and_carry_cutoffs():
    ids = list_targets()
    for tid in ("order-ode-p4", "opcount-dft-n16", "ssp-s3-p3", "stab-real-s4-p4", "stab-imag-s4-p4"):
        assert tid in ids
        t = get_target(tid)
        assert t.prompt and t.entry_point and t.certifier_name
    assert get_target("quad-d7").cutoff == "1814-01-01" and get_target("quad-square-d5").cutoff == "1948-01-01"
    assert get_target("pep-davis-yin-contraction-calibration").cutoff == "2015-01-01"
    assert get_target("ssp-s5-p4").best_known == 1.508 and get_target("ssp-s5-p4").status == "open"
    assert get_target("opcount-dft-n32").best_known == 800
    with pytest.raises(KeyError):
        get_target("ssp-nonsense")
    with pytest.raises(KeyError):
        get_target("order-ode-x")
