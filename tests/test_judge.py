"""The judge: static rule, sandbox limits, determinism, and the PEP path."""
import time

import pytest

from lab.judge import Judge, static_check
from lab.targets import get_target

GOOD = """
import numpy as np
def rule():
    n = 4
    k = np.arange(1, n)
    b = k / np.sqrt(4 * k * k - 1)
    J = np.diag(b, 1) + np.diag(b, -1)
    x, V = np.linalg.eigh(J)
    return x.tolist(), (2 * V[0, :] ** 2).tolist()
"""


@pytest.fixture(scope="module")
def judge():
    return Judge(timeout_s=30, memory_mb=1024)


@pytest.fixture(scope="module")
def target():
    return get_target("quad-d7")


@pytest.mark.parametrize("src,needle", [
    ("from numpy.polynomial.legendre import leggauss\n", "numpy.polynomial"),
    ("import scipy.integrate as si\n", "scipy.integrate"),
    ("from scipy import optimize\n", "scipy.optimize"),
    ("import sympy\n", "sympy"),
    ("import os\n", "os"),
    ("import numpy as np\ndef rule():\n    import mpmath\n", "mpmath"),
    ("import numpy as np\ndef rule():\n    return np.polynomial.legendre.leggauss(4)\n", ".polynomial"),
    ("def rule():\n    return __import__('scipy.special')\n", "__import__"),
    ("def rule():\n    exec('x=1')\n", "exec"),
    ("def rule(:\n", "syntax error"),
])
def test_static_check_flags_forbidden_code(src, needle):
    v = static_check(src)
    assert v and any(needle in s for s in v), v


@pytest.mark.parametrize("src", [GOOD, "import math\nimport numpy.linalg\ndef rule():\n    return [0.0], [2.0]\n",
                                 "from itertools import product\nfrom fractions import Fraction\ndef rule():\n    return [0.0], [2.0]\n"])
def test_static_check_allows_arithmetic_code(src):
    assert static_check(src) == []


def test_good_candidate_passes_and_is_optimal(judge, target):
    cert, sb = judge.judge(GOOD, target, seed=0)
    assert sb.ok and cert.passed and cert.optimal and cert.score == 4
    assert cert.details["sandbox"]["cpu_seconds"] >= 0
    assert cert.details["network_isolation"] in ("unshare", "python-level")
    assert not [m for m in sb.import_log if not m["allowed"]]


def test_same_attempt_scores_the_same_twice(judge, target):
    a, _ = judge.judge(GOOD, target, seed=5)
    b, _ = judge.judge(GOOD, target, seed=5)
    da, db = a.to_dict(), b.to_dict()
    for d in (da, db):
        d.pop("runtime_s")
        d["details"].pop("sandbox")
    assert da == db


def test_forbidden_import_fails_the_attempt(judge, target):
    cert, sb = judge.judge("from numpy.polynomial.legendre import leggauss\ndef rule():\n    x, w = leggauss(4)\n    return x.tolist(), w.tolist()\n", target, seed=0)
    assert not cert.passed and "ForbiddenImport" in cert.error and sb.static_violations


def test_runtime_import_guard_blocks_lazy_submodule_imports(judge, target):
    # the static pass sees only literal imports and attribute names; numpy.polynomial reached through a
    # computed attribute name is stripped from the namespace and its re-import is refused by the guard
    src = "import numpy as np\ndef rule():\n    f = getattr(np, 'poly' + 'nomial')\n    return [0.0], [2.0]\n"
    assert static_check(src) == []
    cert, sb = judge.judge(src, target, seed=0)
    assert not cert.passed and "forbidden" in cert.error.lower()
    assert any(not m["allowed"] and m["module"].startswith("numpy.polynomial") for m in sb.import_log)


def test_runtime_guard_logs_allowed_imports_and_denies_stdlib_outside_the_allowlist(judge, target):
    src = "import numpy as np\ndef rule():\n    import fractions\n    return [0.0], [2.0]\n"
    cert, sb = judge.judge(src, target, seed=0)
    assert sb.ok and any(m["module"] == "fractions" and m["allowed"] for m in sb.import_log)
    assert static_check("def rule():\n    import zlib\n    return [0.0], [2.0]\n")  # outside the allowlist


def test_timeout_kills_the_candidate(target):
    j = Judge(timeout_s=3, memory_mb=1024)
    t0 = time.perf_counter()
    cert, sb = j.judge("def rule():\n    while True:\n        pass\n", target, seed=0)
    assert not cert.passed and sb.timed_out and "timeout" in cert.error
    assert time.perf_counter() - t0 < 15


def test_memory_limit_kills_the_candidate(target):
    j = Judge(timeout_s=30, memory_mb=512)
    cert, sb = j.judge("import numpy as np\ndef rule():\n    keep = [np.ones(60_000_000) for _ in range(20)]\n    return [0.0], [2.0]\n", target, seed=0)
    assert not cert.passed
    assert sb.killed or (sb.error and sb.error["type"] in ("MemoryError", "SandboxFailure"))


def test_network_is_unreachable(judge, target):
    cert, sb = judge.judge("import socket\ndef rule():\n    socket.create_connection(('example.com', 80), timeout=3)\n    return [0.0], [2.0]\n", target, seed=0)
    assert not cert.passed and "socket" in cert.error


def test_crash_and_malformed_outputs_fail(judge, target):
    cert, _ = judge.judge("def rule():\n    return 1 / 0\n", target, seed=0)
    assert not cert.passed and "ZeroDivisionError" in cert.error
    cert, _ = judge.judge("def rule():\n    return [0.0, 0.5], [2.0]\n", target, seed=0)
    assert not cert.passed and cert.error
    cert, _ = judge.judge("def other():\n    return [0.0], [2.0]\n", target, seed=0)
    assert not cert.passed and "no callable" in cert.error
    cert, sb = judge.judge(None, target, seed=0)
    assert not cert.passed and sb is None and "no code" in cert.error


def test_environment_is_scrubbed(judge, target):
    # the candidate cannot import os; the runner clears the environment before executing it
    src = "def rule():\n    import os\n    return [0.0], [2.0]\n"
    assert static_check(src)
    cert, sb = judge.judge("import json\ndef rule():\n    return [0.0], [2.0]\n", target, seed=0)
    assert sb.ok and not cert.passed  # a valid degree-1 rule is not exact to degree 7


def test_pep_target_through_the_judge():
    t = get_target("pep-gd-strongly-convex-distance")
    j = Judge(timeout_s=30, memory_mb=1024, sdp_cap_s=60)
    src = "def method():\n    return {'method': 'gradient_descent', 'params': {'mu': 0.1, 'L': 1.0, 'gamma': 1.0, 'n': 1}}\n"
    cert, sb = j.judge(src, t, seed=0)
    assert sb.ok and cert.passed, cert.error
    assert cert.score == pytest.approx(0.81, abs=1e-4)
    assert cert.details["checks"]["matches_published_tight_rate"]
