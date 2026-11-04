"""Write the synthetic replay fixtures for the demo loop (python tests/make_fixtures.py).

These are NOT recorded model outputs. They are hand-written responses of the kinds a model produces
(correct and optimal, correct but larger, wrong, forbidden, crashing, empty), so that the whole
demo loop, judge and ledger run under pytest with no key and no cost. Recorded fixtures from the real
endpoint are written next to them by RecordingProvider when the lab runs online.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lab.ledger import prompt_hash  # noqa: E402
from lab.targets import SYSTEM_PROMPT, get_target  # noqa: E402

GAUSS4 = """```python
def rule():
    nodes = [-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526]
    weights = [0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538]
    return nodes, weights
```"""

GOLUB_WELSCH = """The nodes are the eigenvalues of the Jacobi matrix of the Legendre recurrence and the weights come from the first components of the eigenvectors.

```python
import numpy as np

def rule():
    n = 4
    k = np.arange(1, n)
    b = k / np.sqrt(4.0 * k * k - 1.0)
    J = np.diag(b, 1) + np.diag(b, -1)
    x, V = np.linalg.eigh(J)
    w = 2.0 * V[0, :] ** 2
    return x.tolist(), w.tolist()
```"""

NEWTON = """```python
import math

def rule():
    n = 4

    def legendre(x):
        p0, p1 = 1.0, x
        for k in range(2, n + 1):
            p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
        dp = n * (x * p1 - p0) / (x * x - 1.0)
        return p1, dp

    nodes, weights = [], []
    for i in range(1, n + 1):
        x = math.cos(math.pi * (i - 0.25) / (n + 0.5))
        for _ in range(100):
            p, dp = legendre(x)
            dx = p / dp
            x -= dx
            if abs(dx) < 1e-15:
                break
        _, dp = legendre(x)
        nodes.append(x)
        weights.append(2.0 / ((1.0 - x * x) * dp * dp))
    return nodes, weights
```"""

GAUSS5 = """```python
def rule():
    a = 0.5384693101056831
    b = 0.9061798459386640
    nodes = [-b, -a, 0.0, a, b]
    weights = [0.2369268850561891, 0.4786286704993665, 0.5688888888888889, 0.4786286704993665, 0.2369268850561891]
    return nodes, weights
```"""

SIMPSON = """Simpson's rule integrates cubics exactly, and with symmetry it should handle the odd terms.

```python
def rule():
    return [-1.0, 0.0, 1.0], [1.0 / 3.0, 4.0 / 3.0, 1.0 / 3.0]
```"""

FORBIDDEN_NP = """```python
from numpy.polynomial.legendre import leggauss

def rule():
    x, w = leggauss(4)
    return x.tolist(), w.tolist()
```"""

FORBIDDEN_SCIPY = """```python
from scipy.special import roots_legendre

def rule():
    x, w = roots_legendre(4)
    return list(x), list(w)
```"""

NEWTON_COTES_8 = """```python
import numpy as np

def rule():
    n = 8
    x = np.linspace(-1.0, 1.0, n)
    V = np.vander(x, n, increasing=True).T
    moments = np.array([2.0 / (k + 1) if k % 2 == 0 else 0.0 for k in range(n)])
    w = np.linalg.solve(V, moments)
    return x.tolist(), w.tolist()
```"""

NO_CODE = "A rule exact to degree 7 needs at least 4 nodes, because an n-point rule cannot integrate the square of its node polynomial. The Gauss-Legendre rule with 4 nodes achieves it; its nodes are the roots of the fourth Legendre polynomial."

SYNTAX_ERROR = """```python
def rule(:
    return [0.0], [2.0]
```"""

CRASH = """```python
def rule():
    return nodes, weights
```"""

WRONG_LENGTHS = """```python
def rule():
    return [-0.5, 0.0, 0.5, 1.0], [0.5, 0.5, 0.5]
```"""

PERTURBED = """```python
def rule():
    nodes = [-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526]
    weights = [0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478558451374538]
    return nodes, weights
```"""

LOBATTO5 = """```python
import math

def rule():
    r = math.sqrt(3.0 / 7.0)
    nodes = [-1.0, -r, 0.0, r, 1.0]
    weights = [1.0 / 10.0, 49.0 / 90.0, 32.0 / 45.0, 49.0 / 90.0, 1.0 / 10.0]
    return nodes, weights
```"""

GAUSS3 = """```python
import math

def rule():
    r = math.sqrt(3.0 / 5.0)
    return [-r, 0.0, r], [5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0]
```"""

GAUSS4_VERBOSE = """We need the 4-point Gauss-Legendre rule. The nodes are the roots of P_4(x) = (35x^4 - 30x^2 + 3)/8, which solve 35t^2 - 30t + 3 = 0 in t = x^2: t = (15 +- 2 sqrt(30))/35.

```python
import math

def rule():
    t1 = (15.0 - 2.0 * math.sqrt(30.0)) / 35.0
    t2 = (15.0 + 2.0 * math.sqrt(30.0)) / 35.0
    x1, x2 = math.sqrt(t1), math.sqrt(t2)
    w1 = (18.0 + math.sqrt(30.0)) / 36.0
    w2 = (18.0 - math.sqrt(30.0)) / 36.0
    return [-x2, -x1, x1, x2], [w2, w1, w1, w2]
```

Both pairs of weights sum to 2."""

RESPONSES = [GAUSS4, GOLUB_WELSCH, NEWTON, GAUSS5, SIMPSON, FORBIDDEN_NP, FORBIDDEN_SCIPY, NEWTON_COTES_8, NO_CODE,
             SYNTAX_ERROR, CRASH, WRONG_LENGTHS, PERTURBED, LOBATTO5, GAUSS3, GAUSS4_VERBOSE]
EXPECT = {"valid": 7, "optimal": 4}


def main() -> None:
    target = get_target("quad-d7")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": target.prompt}]
    key = prompt_hash(messages).split(":")[1][:24]
    out = ROOT / "tests" / "fixtures" / "providers" / "local" / f"{key}.seq.json"
    items = []
    for i, text in enumerate(RESPONSES):
        items.append({"text": text, "prompt_tokens": 410, "completion_tokens": 40 + len(text) // 4,
                      "latency_s": round(1.5 + 0.02 * len(text) / 4, 2), "model": "synthetic-fixture",
                      "finish_reason": "stop", "note": "synthetic, hand-written; not a recorded model output"})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(items, indent=1), encoding="utf-8")
    readme = ROOT / "tests" / "fixtures" / "providers" / "README.md"
    readme.write_text(
        "# Recorded provider responses\n\n"
        "One directory per provider name in `config/lab.yaml` (`local`, `openai`, `anthropic`). Offline mode replays\n"
        "from here and never makes a live call; record mode writes here; a missing fixture fails the test.\n\n"
        "* `<request key>.json`: one recorded call, written by `RecordingProvider`, holding the request (model,\n"
        "  messages, tools, temperature, seed) and the response. The key is the sha256 of those request fields.\n"
        "  GPT and Claude share this format, so a harness test is the same file for both providers.\n"
        "* `<prompt hash>.seq.json`: a list of responses replayed in seed order, so a group of N attempts replays\n"
        "  from one recording. The `local` sequence for quad-d7 is synthetic (hand-written in\n"
        "  `tests/make_fixtures.py`, 16 responses: 7 valid rules, 4 of them optimal), not a model output.\n"
        "  Regenerate it with `python tests/make_fixtures.py` whenever the quad-d7 prompt text changes.\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
