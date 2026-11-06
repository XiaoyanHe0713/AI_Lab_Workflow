"""Synthetic replay fixtures for the week-4 roles and the loop (imported by tests/make_fixtures.py).

Hand-written responses of the kinds the three roles produce, NOT recorded model outputs: builder files for the
square and triangle rungs (valid and optimal, valid but larger, wrong, forbidden, crashing, empty), theorist
proposals (checked lemmas, false lemmas, trivial checks, prose, and the revised proposal of the second round) and
breaker replies (a witness inside the class that does not reproduce, one outside the class, a probe followed by a
witness, prose). Sequences are keyed by the opening prompt; an item's `rounds` holds the reply of each tool round.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures" / "providers"

# --- builder responses, the square (degree 5, best known 7) ------------------------------------------------------------
SQ_TENSOR9 = """A 3 x 3 tensor product of the 3-point Gauss rule is exact to degree 5 in each variable.

```python
import math

def rule():
    r = math.sqrt(3.0 / 5.0)
    x1 = [-r, 0.0, r]
    w1 = [5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0]
    nodes, weights = [], []
    for i in range(3):
        for j in range(3):
            nodes.append([x1[i], x1[j]])
            weights.append(w1[i] * w1[j])
    return nodes, weights
```"""

SQ_RADON7 = """Use the orbit structure centre, two points on the x axis, four symmetric points, and solve the even moment equations up to degree 5.

```python
import math

def rule():
    r = math.sqrt(14.0 / 15.0)
    s = math.sqrt(1.0 / 3.0)
    t = math.sqrt(3.0 / 5.0)
    nodes = [[0.0, 0.0], [r, 0.0], [-r, 0.0], [s, t], [s, -t], [-s, t], [-s, -t]]
    weights = [8.0 / 7.0, 20.0 / 63.0, 20.0 / 63.0, 5.0 / 9.0, 5.0 / 9.0, 5.0 / 9.0, 5.0 / 9.0]
    return nodes, weights
```"""

SQ_RADON7_WRONG = """```python
import math

def rule():
    r = math.sqrt(14.0 / 15.0)
    s = math.sqrt(1.0 / 3.0)
    t = math.sqrt(3.0 / 5.0)
    nodes = [[0.0, 0.0], [r, 0.0], [-r, 0.0], [s, t], [s, -t], [-s, t], [-s, -t]]
    weights = [8.0 / 7.0, 20.0 / 63.0, 20.0 / 63.0, 5.0 / 9.0, 5.0 / 9.0, 5.0 / 9.0, 4.0 / 9.0]
    return nodes, weights
```"""

SQ_TENSOR16 = """```python
import math

def rule():
    a = math.sqrt((3.0 - 2.0 * math.sqrt(6.0 / 5.0)) / 7.0)
    b = math.sqrt((3.0 + 2.0 * math.sqrt(6.0 / 5.0)) / 7.0)
    wa = (18.0 + math.sqrt(30.0)) / 36.0
    wb = (18.0 - math.sqrt(30.0)) / 36.0
    x1 = [-b, -a, a, b]
    w1 = [wb, wa, wa, wb]
    nodes = [[x1[i], x1[j]] for i in range(4) for j in range(4)]
    weights = [w1[i] * w1[j] for i in range(4) for j in range(4)]
    return nodes, weights
```"""

SQ_CENTROID = """```python
def rule():
    return [[0.0, 0.0]], [4.0]
```"""

SQ_FORBIDDEN = """```python
from scipy.special import roots_legendre

def rule():
    x, w = roots_legendre(3)
    return [[a, b] for a in x for b in x], [p * q for p in w for q in w]
```"""

SQ_CRASH = """```python
def rule():
    return nodes, weights
```"""

SQ_NO_CODE = ("Radon showed in 1948 that seven points suffice for degree 5 on the square: the centre, two points on an axis and "
              "four points symmetric about both axes, with weights found from the moment equations.")

SQUARE_RESPONSES = [SQ_TENSOR9, SQ_RADON7, SQ_CENTROID, SQ_RADON7_WRONG, SQ_NO_CODE, SQ_TENSOR16, SQ_FORBIDDEN, SQ_CRASH]
SQUARE_EXPECT = {"valid": 3, "optimal": 1}     # per cycle of 8 attempts

# --- builder responses, the triangle (degree 2, best known 3) -----------------------------------------------------------
TR_STRANG3 = """Three interior points at the barycentric orbit (2/3, 1/6, 1/6) with equal weights integrate quadratics exactly on the reference triangle of area 1/2.

```python
def rule():
    nodes = [[2.0 / 3.0, 1.0 / 6.0], [1.0 / 6.0, 2.0 / 3.0], [1.0 / 6.0, 1.0 / 6.0]]
    weights = [1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0]
    return nodes, weights
```"""

TR_EDGE3 = """```python
def rule():
    nodes = [[0.5, 0.0], [0.5, 0.5], [0.0, 0.5]]
    weights = [1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0]
    return nodes, weights
```"""

TR_CENTROID = """```python
def rule():
    return [[1.0 / 3.0, 1.0 / 3.0]], [0.5]
```"""

TR_AREA_WRONG = """```python
def rule():
    nodes = [[2.0 / 3.0, 1.0 / 6.0], [1.0 / 6.0, 2.0 / 3.0], [1.0 / 6.0, 1.0 / 6.0]]
    weights = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
    return nodes, weights
```"""

TR_DUFFY4 = """Collapse the square's 2 x 2 Gauss rule onto the triangle with the Duffy map.

```python
import math

def rule():
    g = 1.0 / math.sqrt(3.0)
    u = [(1.0 - g) / 2.0, (1.0 + g) / 2.0]
    wu = [0.5, 0.5]
    nodes, weights = [], []
    for i in range(2):
        for j in range(2):
            nodes.append([u[i], u[j] * (1.0 - u[i])])
            weights.append(wu[i] * wu[j] * (1.0 - u[i]))
    return nodes, weights
```"""

TR_DUNAVANT6 = """```python
def rule():
    a, b = 0.445948490915965, 0.108103018168070
    c, d = 0.091576213509771, 0.816847572980459
    wa, wc = 0.223381589678011, 0.109951743655322
    nodes = [[a, a], [a, b], [b, a], [c, c], [c, d], [d, c]]
    weights = [wa * 0.5, wa * 0.5, wa * 0.5, wc * 0.5, wc * 0.5, wc * 0.5]
    return nodes, weights
```"""

TR_NO_CODE = "The three edge midpoints with weight 1/6 each give a degree-2 rule; the centroid alone is only degree 1."

TR_CRASH = """```python
def rule():
    return [[0.5, 0.0], [0.5, 0.5], [0.0, 0.5]], [1.0 / 6.0] * len(points)
```"""

TRIANGLE_RESPONSES = [TR_STRANG3, TR_CENTROID, TR_EDGE3, TR_AREA_WRONG, TR_DUFFY4, TR_NO_CODE, TR_DUNAVANT6, TR_CRASH]
TRIANGLE_EXPECT = {"valid": 4, "optimal": 2}   # per cycle of 8 attempts

# --- theorist proposals (two rounds each: the proposal, then the revision after the lemma checks) ---------------------------
def _j(obj) -> str:
    return "```json\n" + json.dumps(obj, indent=1) + "\n```"


TH_SQUARE_GOOD = _j({"lemmas": [
    {"statement": "The integral of x^2 over the square [-1,1]^2 is 4/3", "check": "integrate(integrate(x**2, (x, -1, 1)), (y, -1, 1)) - Rational(4, 3)"},
    {"statement": "The integral of x^2 y^2 over the square is 4/9", "check": "Eq(integrate(integrate(x**2 * y**2, (x, -1, 1)), (y, -1, 1)), Rational(4, 9))"},
    {"statement": "Every monomial with an odd exponent integrates to zero over the square by symmetry", "check": "integrate(integrate(x**3 * y**2, (x, -1, 1)), (y, -1, 1))"},
], "structure": "the centre, two points on the x axis at distance r, and four points (+-s, +-t): seven nodes, three weights, three parameters r, s, t",
    "hypotheses": ["with that orbit structure the six even moment equations up to degree 5 determine r^2 = 14/15, s^2 = 1/3, t^2 = 3/5",
                   "a tensor product needs 9 points; the orbit rule needs 7"]})
TH_SQUARE_GOOD_R2 = TH_SQUARE_GOOD

TH_SQUARE_FALSE = _j({"lemmas": [
    {"statement": "The integral of x^4 over the square is 4/3", "check": "integrate(integrate(x**4, (x, -1, 1)), (y, -1, 1)) - Rational(4, 3)"},
    {"statement": "A trivial constant check", "check": "Rational(1, 2) - Rational(1, 2)"},
], "structure": "a 3 x 3 tensor product of the 3-point Gauss rule", "hypotheses": ["the tensor product is optimal"]})
TH_SQUARE_FALSE_R2 = _j({"lemmas": [
    {"statement": "The integral of x^4 over the square is 4/5", "check": "integrate(integrate(x**4, (x, -1, 1)), (y, -1, 1)) - Rational(4, 5)"},
], "structure": "a 3 x 3 tensor product of the 3-point Gauss rule", "hypotheses": ["the tensor product is exact to degree 5"]})

TH_PROSE = ("I think the key lemma is that the moments of even monomials over the square factor into one-dimensional moments, "
            "and that symmetric node sets kill the odd moments automatically.")
TH_PROSE_R2 = _j({"lemmas": [
    {"statement": "The moment of x^2 y^2 factors into the product of the one-dimensional moments", "check": "Eq(integrate(integrate(x**2 * y**2, (x, -1, 1)), (y, -1, 1)), integrate(x**2, (x, -1, 1)) * integrate(y**2, (y, -1, 1)))"},
], "structure": "symmetric orbits under the sign changes of x and y", "hypotheses": ["odd moments vanish for symmetric node sets"]})

TH_ERROR = _j({"lemmas": [
    {"statement": "The equation x^2 + 1 = 0 has a tenth real root", "check": "solve(x**2 + 1, x)[9]"},
    {"statement": "The third Legendre polynomial has three real roots, the 3-point Gauss nodes", "check": "Eq(len(real_roots(legendre(3, x))), 3)"},
], "structure": "nodes from Legendre roots", "hypotheses": ["use Gauss nodes in each coordinate"]})
TH_ERROR_R2 = _j({"lemmas": [
    {"statement": "The third Legendre polynomial has three real roots, the 3-point Gauss nodes", "check": "Eq(len(real_roots(legendre(3, x))), 3)"},
], "structure": "nodes from Legendre roots", "hypotheses": ["use Gauss nodes in each coordinate"]})

THEORIST_ROUNDS = [(TH_SQUARE_GOOD, TH_SQUARE_GOOD_R2), (TH_SQUARE_FALSE, TH_SQUARE_FALSE_R2), (TH_PROSE, TH_PROSE_R2), (TH_ERROR, TH_ERROR_R2)]
THEORIST_EXPECT = {"passed_round1": 2, "passed_round2": 4, "proposals_of_4": 4}

TH_TRIANGLE_GOOD = _j({"lemmas": [
    {"statement": "The integral of x^a y^b over the reference triangle is a! b! / (a + b + 2)!", "check": "Eq(integrate(integrate(x**2 * y, (y, 0, 1 - x)), (x, 0, 1)), factorial(2) * factorial(1) / factorial(5))"},
    {"statement": "The area of the reference triangle is one half", "check": "integrate(integrate(1, (y, 0, 1 - x)), (x, 0, 1)) - Rational(1, 2)"},
], "structure": "one symmetric orbit of three points under the permutations of the barycentric coordinates", "hypotheses": ["three points suffice for degree 2"]})
TH_TRIANGLE_FALSE = _j({"lemmas": [
    {"statement": "The integral of x over the reference triangle is 1/3", "check": "integrate(integrate(x, (y, 0, 1 - x)), (x, 0, 1)) - Rational(1, 3)"},
], "structure": "the centroid alone", "hypotheses": ["one point is enough"]})
TH_TRIANGLE_FALSE_R2 = _j({"lemmas": [
    {"statement": "The integral of x over the reference triangle is 1/6", "check": "integrate(integrate(x, (y, 0, 1 - x)), (x, 0, 1)) - Rational(1, 6)"},
], "structure": "one symmetric orbit of three points", "hypotheses": ["three points are needed for degree 2"]})
THEORIST_TRIANGLE_ROUNDS = [(TH_TRIANGLE_GOOD, TH_TRIANGLE_GOOD), (TH_TRIANGLE_FALSE, TH_TRIANGLE_FALSE_R2), (TH_PROSE, TH_PROSE_R2), (TH_ERROR, TH_ERROR_R2)]

TH_INTERVAL_GOOD = _j({"lemmas": [
    {"statement": "The fourth Legendre polynomial has four distinct real roots inside (-1, 1)", "check": "all(-1 < r < 1 for r in real_roots(legendre(4, x))) and len(real_roots(legendre(4, x))) == 4"},
    {"statement": "The Legendre polynomials P2 and P3 are orthogonal on [-1, 1]", "check": "integrate(legendre(2, x) * legendre(3, x), (x, -1, 1))"},
], "structure": "four nodes at the roots of P4, symmetric about 0, with two distinct weights", "hypotheses": ["a 4-node rule with nodes at the roots of P4 is exact to degree 7"]})
THEORIST_INTERVAL_ROUNDS = [(TH_INTERVAL_GOOD, TH_INTERVAL_GOOD), (TH_SQUARE_FALSE, TH_SQUARE_FALSE_R2), (TH_PROSE, TH_PROSE_R2), (TH_ERROR, TH_ERROR_R2)]

# --- breaker replies ---------------------------------------------------------------------------------------------------
BR_INSIDE = _j({"witness": {"kind": "exactness_error", "input": {"exponents": [4, 0]}}, "reason": "the highest even power along one axis is where a rule derived from low-degree moments usually fails"})
BR_OUTSIDE = _j({"witness": {"kind": "exactness_error", "input": {"exponents": [6, 0]}}, "reason": "degree 6 is beyond what seven points can integrate"})
BR_PROBE = _j({"probe": {"exponents": [[2, 0], [4, 0], [2, 2], [0, 4]]}})
BR_PROBE_R2 = _j({"witness": {"kind": "exactness_error", "input": {"coefficients": {"4,0": 1.0, "2,2": -1.0, "0,0": 0.5}}}, "reason": "the probe shows every error below tolerance; this combination is the last try"})
BR_PROSE = "The rule looks exact on every monomial I can think of; I cannot find a breaking input."
BR_LARGE_COEFFICIENTS = _j({"witness": {"kind": "exactness_error", "input": {"coefficients": {"4,0": 1e8, "0,0": -1e8}}}, "reason": "amplify the rounding error"})
BREAKER_ROUNDS = [(BR_INSIDE,), (BR_OUTSIDE,), (BR_PROBE, BR_PROBE_R2), (BR_PROSE,), (BR_LARGE_COEFFICIENTS,)]
BREAKER_EXPECT = {"confirmed": 0, "outside_class": 2}

BR_INTERVAL_INSIDE = _j({"witness": {"kind": "exactness_error", "input": {"exponents": [6]}}, "reason": "x^6 is the highest even power"})
BR_INTERVAL_OUTSIDE = _j({"witness": {"kind": "exactness_error", "input": {"exponents": [8]}}, "reason": "x^8 breaks any 4-node rule"})
BR_INTERVAL_PROBE = _j({"probe": {"exponents": [[2], [4], [6]]}})
BR_INTERVAL_PROBE_R2 = _j({"witness": {"kind": "exactness_error", "input": {"coefficients": {"6": 1.0, "4": -1.0}}}, "reason": "last try"})
BREAKER_INTERVAL_ROUNDS = [(BR_INTERVAL_INSIDE,), (BR_INTERVAL_OUTSIDE,), (BR_INTERVAL_PROBE, BR_INTERVAL_PROBE_R2), (BR_PROSE,), (BR_LARGE_COEFFICIENTS,)]
BR_TRIANGLE_INSIDE = _j({"witness": {"kind": "exactness_error", "input": {"exponents": [2, 0]}}, "reason": "x^2 on the triangle"})
BR_TRIANGLE_OUTSIDE = _j({"witness": {"kind": "exactness_error", "input": {"exponents": [3, 0]}}, "reason": "cubics are beyond degree 2"})
BR_TRIANGLE_PROBE = _j({"probe": {"exponents": [[1, 1], [2, 0], [0, 2]]}})
BR_TRIANGLE_PROBE_R2 = _j({"witness": {"kind": "exactness_error", "input": {"coefficients": {"1,1": 1.0, "2,0": -1.0}}}, "reason": "last try"})
BREAKER_TRIANGLE_ROUNDS = [(BR_TRIANGLE_INSIDE,), (BR_TRIANGLE_OUTSIDE,), (BR_TRIANGLE_PROBE, BR_TRIANGLE_PROBE_R2), (BR_PROSE,), (BR_LARGE_COEFFICIENTS,)]

LOOP_TARGETS = {"quad-d7": None, "quad-square-d5": SQUARE_RESPONSES, "quad-triangle-d2": TRIANGLE_RESPONSES}
LOOP_ATTEMPTS = 8      # attempts per rung in the offline loop test (and in the generator, which mirrors the loop)
THEORIST_BY_TARGET = {"quad-d7": THEORIST_INTERVAL_ROUNDS, "quad-square-d5": THEORIST_ROUNDS, "quad-triangle-d2": THEORIST_TRIANGLE_ROUNDS}
BREAKER_BY_TARGET = {"quad-d7": BREAKER_INTERVAL_ROUNDS, "quad-square-d5": BREAKER_ROUNDS, "quad-triangle-d2": BREAKER_TRIANGLE_ROUNDS}


def items_from_rounds(rounds_list: list[tuple], note: str) -> list[dict]:
    items = []
    for rounds in rounds_list:
        first = rounds[0]
        items.append({"text": first, "rounds": list(rounds), "prompt_tokens": 600, "completion_tokens": 60 + len(first) // 4,
                      "latency_s": round(1.0 + 0.02 * len(first) / 4, 2), "model": "synthetic-fixture", "finish_reason": "stop",
                      "note": "synthetic, hand-written; not a recorded model output", "for": note})
    return items


def items_from_texts(texts: list[str], note: str) -> list[dict]:
    items = []
    for text in texts:
        items.append({"text": text, "rounds": [text, text], "prompt_tokens": 410, "completion_tokens": 40 + len(text) // 4,
                      "latency_s": round(1.5 + 0.02 * len(text) / 4, 2), "model": "synthetic-fixture", "finish_reason": "stop",
                      "note": "synthetic, hand-written; not a recorded model output (round 2: a second try keeps the file)", "for": note})
    return items


def write_items(messages: list[dict], items: list[dict]) -> Path:
    from lab.ledger import prompt_hash

    key = prompt_hash(messages).split(":")[1][:24]
    out = FIXTURES / "local" / f"{key}.seq.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(items, indent=1), encoding="utf-8")
    return out


def write_loop_fixtures(verbose: bool = True) -> list[Path]:
    """The sequences the offline loop test replays: builder rungs, theorist proposals, the builders derived from the
    theorist's checked proposals, and the breaker's attacks on the survivors. The proposal and breaker prompts depend on
    the replayed outputs, so this runs the builder and the theorist offline (temporary ledger) to compute them, in the
    loop's own order (week 5): the plain-prompt group first, the built-in detectors mined on it, then the theorist with
    the retrieval query the diagnosis names (with the test's 8 attempts per rung no detector is kept, so the query is the
    target's title; the order here keeps the generator honest if that changes)."""
    import tempfile

    from lab.config import lab_config
    from lab.demo import prepare, run_group
    from lab.episode import run_episodes
    from lab.harness.checker import diagnosis_query, firing_table, kept_detectors, mine_builtin
    from lab.harness.librarian import Librarian, ReadingList
    from lab.ledger import Ledger
    from lab.loop import survivors_of, theorist_retrieval
    from lab.roles.breaker import BREAKER_SYSTEM, breaker_prompt, candidate_from_row
    from lab.roles.builder import with_proposal
    from lab.roles.theorist import THEORIST_SYSTEM, TheoristRunner, proposal_from_row, theorist_prompt
    from lab.targets import SYSTEM_PROMPT, get_target

    written = []
    tmp = Path(tempfile.mkdtemp(prefix="make_fixtures_loop_"))
    cfg = lab_config(env={"LAB_ROOT": str(tmp)})
    ledger = Ledger(tmp / "ledger.sqlite")
    try:
        # 1. the builder rungs (quad-d7's sequence is the week-2 one, written by make_fixtures.main), run as the loop runs
        #    them (LOOP_ATTEMPTS attempts), then the built-in detectors mined on each group and the diagnosis query computed
        all_rows: dict[str, list] = {}
        queries: dict[str, str] = {}
        for tid, responses in LOOP_TARGETS.items():
            target = get_target(tid)
            if responses is not None:
                written.append(write_items([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": target.prompt}],
                                           items_from_texts(responses, f"loop builder rung {tid}")))
            s = run_group(target_id=tid, attempts=LOOP_ATTEMPTS, seed=0, provider_name="local", role_name="builder", ledger=ledger,
                          ledger_path=ledger.path, gpu="A100", precision=None, units_per_hour=None, temperature=None, max_tokens=None,
                          timeout_s=60, cfg=cfg, fixtures_root=ROOT, quiet=True, cache_dir=tmp / "cache", mode="offline", concurrency=4)
            all_rows[tid] = ledger.attempts(group_id=s["group_id"])
            mine_builtin(ledger, target, all_rows[tid], seed=0)
            table = firing_table(kept_detectors(ledger, tid), all_rows[tid])
            queries[tid] = diagnosis_query(target, table)
            if verbose:
                print(f"builder {tid}: {s['passed']}/{s['attempts']} valid, {s['optimal']} optimal; kept detectors "
                      f"{[d['name'] for d in table['detectors']]}; query {queries[tid]!r}")
        # 2. the theorist, with the librarian's lexical retrieval (under the diagnosis query) in the prompt, two rounds per item
        theorist = prepare(cfg=cfg, role_name="theorist", provider_name="local", mode="offline", fixtures_root=ROOT, ledger=ledger,
                           gpu="A100", temperature=None, system_prompt=THEORIST_SYSTEM, cache_dir=tmp / "cache", timeout_s=60)
        librarian = Librarian(theorist["provider"], ledger, ReadingList(cfg=cfg), cfg=cfg)
        proposals_by_target: dict[str, list[dict]] = {}
        for tid, rounds in THEORIST_BY_TARGET.items():
            target = get_target(tid)
            ret = theorist_retrieval(librarian, target, query=queries[tid])
            prompt = theorist_prompt(target, lessons_text="", docs=ret["documents"])
            written.append(write_items([{"role": "system", "content": THEORIST_SYSTEM}, {"role": "user", "content": prompt}],
                                       items_from_rounds(rounds, f"loop theorist {tid}")))
            runner = TheoristRunner(theorist["role"], target, theorist["judge"], prompt=prompt, tool_calls=1, timeout_s=60)
            s = run_episodes(runner.attempt, role_name="theorist", role=theorist["role"], provider=theorist["provider"], provider_name="local",
                             target=target, attempts=len(rounds), seed=0, ledger=ledger, meter=theorist["meter"], model=theorist["model"],
                             model_version=theorist["model_version"], precision=theorist["precision"], gpu="A100", concurrency=2, quiet=True,
                             tool_calls=1)
            rows = ledger.attempts(group_id=s["group_id"])
            proposals_by_target[tid] = [p for p in (proposal_from_row(r) for r in rows if r.passed) if p]
            if verbose:
                print(f"theorist {tid}: {s['passed']}/{s['attempts']} proposals passed their lemma checks")
        # 3. the builders derived from the proposals (one sequence per distinct proposal prompt)
        for tid, responses in LOOP_TARGETS.items():
            target = get_target(tid)
            texts = responses or None
            if texts is None:
                from make_fixtures import RESPONSES as INTERVAL_RESPONSES
                texts = INTERVAL_RESPONSES
            for p in proposals_by_target.get(tid, []):
                prompt = with_proposal(target.prompt, p)
                written.append(write_items([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
                                           items_from_texts(texts, f"loop builder derived from a proposal on {tid}")))
        # 4. the breaker: one sequence per surviving candidate's prompt (the loop labels candidates by rank)
        for tid, rounds in BREAKER_BY_TARGET.items():
            target = get_target(tid)
            for ci, row in enumerate(survivors_of(all_rows[tid], 4)):
                cand = candidate_from_row(row, None, label=f"candidate {ci + 1}")
                for tool_calls in (0, 1):
                    prompt = breaker_prompt(target, cand, tool_calls=tool_calls)
                    written.append(write_items([{"role": "system", "content": BREAKER_SYSTEM}, {"role": "user", "content": prompt}],
                                               items_from_rounds(rounds, f"loop breaker on {tid} candidate {ci + 1} (tool calls {tool_calls})")))
    finally:
        ledger.close()
    return written


# --- the matrix: synthetic lessons (program-written, no model) and the ledger state the offline matrix test builds ---------
SYNTHETIC_LESSONS = [
    {"target": "quad-d7", "level": "approach", "rule": "Place the nodes at the zeros of the Legendre polynomial of degree n and solve for the weights from the moment equations",
     "conditions": ["interval [-1, 1]", "polynomial exactness"], "evidence": "observed"},
    {"target": "quad-d7", "level": "implementation", "rule": "Compute the Legendre zeros by Newton iteration from the Chebyshev initial guesses cos(pi (i - 1/4) / (n + 1/2)) to full precision",
     "conditions": ["nodes from a polynomial's roots"], "evidence": "observed"},
    {"target": "quad-d7", "level": "process", "rule": "Self-test the rule on every monomial up to the required degree and check that the weights sum to the volume 2 before submission",
     "conditions": ["any quadrature target"], "evidence": "derived"},
    {"target": "quad-d7", "level": "process", "rule": "When a derivation loses a factor, recompute the constant term first: a wrong weight sum shows up on the degree-0 monomial",
     "conditions": ["a certificate reports a constant-monomial error"], "evidence": "proposed"},
    {"target": "ssp-s3-p3", "level": "approach", "rule": "Write the Shu-Osher form with nonnegative coefficients and read the SSP coefficient as the smallest beta/alpha ratio",
     "conditions": ["explicit Runge-Kutta", "strong stability"], "evidence": "proposed"},
    {"target": "ssp-s3-p3", "level": "process", "rule": "Check the order conditions numerically on the test equation before claiming the order",
     "conditions": ["time integrators"], "evidence": "derived"},
]


def seed_bank(bank) -> list[str]:
    """Add the synthetic lessons to a bank (idempotent: ids are content hashes); returns their ids."""
    from lab.harness.lessons import Lesson, lesson_id
    from lab.targets import get_target

    ids = []
    for d in SYNTHETIC_LESSONS:
        fam = get_target(d["target"]).family
        l = Lesson(id=lesson_id(d["rule"], d["level"], fam), level=d["level"], rule=d["rule"], conditions=list(d["conditions"]),
                   evidence_tag=d["evidence"], support=["synthetic"] if d["evidence"] == "observed" else [], family=fam, targets=[d["target"]],
                   source_group="synthetic", provenance={"written_by": "tests/fixtures_week4.py (program), not a model"})
        bank.add(l)
        ids.append(l.id)
    return ids


MATRIX_SOURCES = ["quad-d7"]
MATRIX_TARGETS = ["quad-square-d5", "quad-triangle-d2"]
MATRIX_CARRIERS = ["none", "lessons", "process_only", "unrelated", "skills"]
CONTROL_CARRIERS_TEST = ["none", "second_try", "diagnosis_only"]   # week 5: the two target-side control rows
CONTROL_SEEDS, CONTROL_ATTEMPTS = 2, 8                              # seed blocks x attempts of the offline control test


def matrix_run_kw(tmp: Path) -> dict:
    """The run settings of the offline matrix (the test and the generator must agree, or a prompt is not replayed)."""
    return dict(provider_name="local", mode="offline", gpu="A100", precision=None, units_per_hour=None, temperature=None, max_tokens=None,
                timeout_s=60, fixtures_root=ROOT, cache_dir=tmp / "cache", concurrency=4)


def matrix_state(tmp: Path, ledger, cfg) -> dict:
    """The ledger state the offline matrix measures from: the week-2 demo group on quad-d7 (16 synthetic attempts), its
    distinct certified rules archived by the program with template dossiers, and the synthetic lessons in the bank."""
    from lab.demo import run_group
    from lab.harness.archive import Archive
    from lab.harness.lessons import LessonBank
    from lab.loop import survivors_of, template_dossier
    from lab.targets import get_target

    s = run_group(target_id="quad-d7", attempts=16, seed=0, provider_name="local", role_name="builder", ledger=ledger, ledger_path=ledger.path,
                  gpu="A100", precision=None, units_per_hour=None, temperature=None, max_tokens=None, timeout_s=60, cfg=cfg, fixtures_root=ROOT,
                  quiet=True, cache_dir=tmp / "cache", mode="offline", concurrency=4)
    archive, bank = Archive(ledger, tmp), LessonBank(ledger)
    target = get_target("quad-d7")
    archived = []
    for r in survivors_of(ledger.attempts(group_id=s["group_id"]), 3):
        archived.append(archive.archive_method(r, dossier=template_dossier(r, target))["artifact_id"])
    return {"group": s, "archive": archive, "bank": bank, "lesson_ids": seed_bank(bank), "artifact_ids": archived}


def write_matrix_fixtures(verbose: bool = True) -> list[Path]:
    """The builder sequences for every carrier prompt of the offline matrix test (the prompts are computed exactly as the
    test computes them, from the same synthetic state)."""
    import tempfile

    from lab.config import lab_config
    from lab.harness.carriers import apply_carrier
    from lab.harness.matrix import experience_from_archive, process_only_control, unrelated_control
    from lab.ledger import Ledger
    from lab.targets import SYSTEM_PROMPT, get_target

    tmp = Path(tempfile.mkdtemp(prefix="make_fixtures_matrix_"))
    cfg = lab_config(env={"LAB_ROOT": str(tmp)})
    ledger = Ledger(tmp / "ledger.sqlite")
    written = []
    try:
        st = matrix_state(tmp, ledger, cfg)
        archive, bank = st["archive"], st["bank"]
        exp = experience_from_archive(ledger, archive, bank, MATRIX_SOURCES)
        for tid in MATRIX_TARGETS:
            target = get_target(tid)
            texts = LOOP_TARGETS[tid]
            sets = {"lessons": exp, "skills": exp, "process_only": process_only_control(exp), "unrelated": unrelated_control(ledger, archive, bank, tid)}
            for carrier, e in sets.items():
                if carrier not in MATRIX_CARRIERS:
                    continue
                app = apply_carrier(carrier, target, e, bank=bank, archive=archive, token_budget=2000)
                written.append(write_items([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": app["prompt"]}],
                                           items_from_texts(texts, f"matrix cell {carrier} from {e.name} on {tid}")))
                if verbose:
                    print(f"matrix {carrier} from {e.name} on {tid}: {len(app['context'].get('lesson_ids', []))} lessons, "
                          f"{len(app['context'].get('skill_ids', []))} skills")
        # the diagnosis-only control (week 5): its prompt is the program's diagnosis of the target's own `none` cell, so that
        # cell is run here exactly as the offline control test runs it (CONTROL_SEEDS seed blocks of CONTROL_ATTEMPTS) and the
        # built-in detectors are mined on it; the second-try control replays the plain prompt's sequence (round 2 keeps the file)
        from lab.harness.matrix import diagnosis_for, run_cell

        for tid in MATRIX_TARGETS:
            target = get_target(tid)
            none_cell = run_cell(experience=None, carrier="none", target_id=tid, seeds=CONTROL_SEEDS, attempts=CONTROL_ATTEMPTS, cfg=cfg, ledger=ledger,
                                 bank=bank, archive=archive, matrix_id="mtx_fixtures", run_kw=matrix_run_kw(tmp), quiet=True, seed0=0)
            table, diag = diagnosis_for(ledger, tid, none_cell, seed=0)
            app = apply_carrier("diagnosis_only", target, None, bank=bank, diagnosis_table=table)
            written.append(write_items([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": app["prompt"]}],
                                       items_from_texts(LOOP_TARGETS[tid], f"matrix diagnosis-only control on {tid}")))
            if verbose:
                print(f"matrix diagnosis_only on {tid}: kept {diag['kept']}, fired {app['context']['detectors_named']}")
    finally:
        ledger.close()
    return written
