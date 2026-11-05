"""The lesson bank: three levels, evidence tags, consolidation by add, replace and retire under a token budget, the
notebook text, validated checks applied at use time; and, per recorded provider, lesson extraction from the demo
group and the gate that selects one lesson and writes a check."""
import pytest

from lab.harness.lessons import (
    LEVELS,
    Lesson,
    LessonBank,
    LessonHarness,
    apply_check,
    estimate_tokens,
    lesson_id,
    validate_expression,
)
from lab.ledger import Ledger
from lab.targets import get_target

from conftest import demo_group  # noqa: E402


def L(rule, level="implementation", tag="proposed", support=(), family="quadrature", conds=("interval",)):
    return Lesson(id=lesson_id(rule, level, family), level=level, rule=rule, conditions=list(conds), evidence_tag=tag,
                  support=list(support), family=family)


def test_ids_and_tokens():
    assert lesson_id("A rule", "approach", "q") == lesson_id("a  rule", "approach", "q") != lesson_id("a rule", "process", "q")
    assert estimate_tokens("x" * 40) == 10 and estimate_tokens("") == 1
    assert LEVELS == ("approach", "implementation", "process")


def test_bank_add_active_retire_and_ranking(tmp_path):
    bank = LessonBank(Ledger(tmp_path / "l.sqlite"))
    a = bank.add(L("use the zeros of the Legendre polynomial as nodes", "approach", "observed", support=["att_1", "att_2"]))
    b = bank.add(L("check that the weights sum to the volume", "process", "derived"))
    c = bank.add(L("do not hard-code decimals", "implementation", "proposed"))
    ids = [l.id for l in bank.active("quadrature")]
    assert ids == [a.id, b.id, c.id]                     # observed > derived > proposed
    assert [l.id for l in bank.active("other")] == [b.id]  # process lessons apply to every family
    bank.retire(c.id, "test")
    assert [l.id for l in bank.active("quadrature")] == [a.id, b.id] and bank.get(c.id).retired == 1
    assert bank.total_tokens("quadrature") == a.tokens + b.tokens


def test_consolidation_adds_replaces_skips_and_retires(tmp_path):
    bank = LessonBank(Ledger(tmp_path / "l.sqlite"))
    old = L("compute the nodes as zeros of the Legendre polynomial by Newton iteration", "approach", "proposed")
    log = bank.consolidate([old], token_budget=10_000)
    assert [x["action"] for x in log] == ["add"]
    better = L("compute the nodes as zeros of the Legendre polynomial by Newton iteration from Chebyshev guesses", "approach",
               "observed", support=["att_9"])
    log = bank.consolidate([better], token_budget=10_000)
    assert log[0]["action"] == "replace" and log[0]["replaced"] == old.id
    assert bank.get(old.id).retired == 1 and bank.get(old.id).replaced_by == better.id and bank.get(better.id).support == ["att_9"]
    worse = L("compute the nodes as zeros of the Legendre polynomial by Newton iteration", "approach", "proposed")
    log = bank.consolidate([worse], token_budget=10_000)
    assert log[0]["action"] == "skip" and bank.get(worse.id).retired == 1   # same rule as the retired one: nothing new
    extra = [L(r, "implementation", "proposed") for r in (
        "normalize the weights so they sum to the volume of the domain", "place the nodes symmetrically about the origin",
        "verify exactness on every monomial before returning the rule", "avoid hard-coded decimals with few digits",
        "solve the moment equations with numpy linalg instead of guessing")]
    log = bank.consolidate(extra, token_budget=60, family="quadrature")
    assert sum(1 for x in log if x["action"] == "retire") >= 1 and bank.total_tokens("quadrature") <= 60
    assert bank.get(better.id).retired == 0                # the best-evidenced lesson survives the budget
    text, ids = bank.notebook_text("quadrature", 60)
    assert ids == sorted(ids) and all(f"- [" in line for line in text.splitlines())


@pytest.mark.parametrize("expr,ok", [
    ("abs(sum(weights) - 2.0) < 1e-9", True),
    ("all(-1 <= x <= 1 for x in nodes) and n >= 4", True),
    ("len(weights) == len(nodes)", True),
    ("min(weights) > 0", True),
    ("__import__('os')", False),
    ("weights.append(1)", False),
    ("2 ** 100", False),
    ("open('x')", False),
    ("nodes[0] if nodes else True", True),
    ("", False),
    ("lambda: 1", False),
    ("'a' == 'a'", False),
])
def test_validate_expression(expr, ok):
    assert validate_expression(expr)[0] is ok


def test_apply_check_on_a_rule():
    gauss4 = ([-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526],
              [0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538])
    good = {"expression": "abs(sum(weights) - 2.0) < 1e-9 and n == 4", "valid": True}
    assert apply_check(good, gauss4)["ok"] is True
    bad = {"expression": "min(weights) > 0.5", "valid": True}
    assert apply_check(bad, gauss4)["ok"] is False
    crash = {"expression": "weights[9] > 0", "valid": True}
    assert apply_check(crash, gauss4)["ok"] is None and "IndexError" in apply_check(crash, gauss4)["error"]
    assert apply_check({"expression": "1 < 2", "valid": False}, gauss4) is None and apply_check(None, gauss4) is None
    # a comprehension that calls a function must see the namespace (the gate on Colab wrote exactly this check)
    colab = {"expression": "len(nodes)==4 and len(weights)==4 and n==4 and abs(sum(weights)-2) < 1e-12 and "
                           "all((-1-1e-12) <= x <= (1+1e-12) for x in nodes) and all(abs(w) < 1e6 for w in weights)", "valid": True}
    assert validate_expression(colab["expression"])[0] and apply_check(colab, gauss4) == {"ok": True, "expression": colab["expression"], "error": None}
    three = ([-0.7745966692414834, 0.0, 0.7745966692414834], [5 / 9, 8 / 9, 5 / 9])
    assert apply_check(colab, three)["ok"] is False
    assert apply_check({"expression": "all(abs(w) > 0 for w in weights) and any(x < 0 for x in nodes)", "valid": True}, gauss4)["ok"] is True


def test_extraction_and_gate_replay_per_provider(frontier, tmp_path):
    led = Ledger(tmp_path / "attempts.sqlite")
    s = demo_group(tmp_path, led, attempts=16)
    rows = led.attempts(group_id=s["group_id"])
    target = get_target("quad-d7")
    h = LessonHarness(frontier.provider, frontier.ledger, LessonBank(frontier.ledger))
    lessons = h.extract(target, rows, group_id=s["group_id"])
    assert 1 <= len(lessons) <= 5
    ids_in_group = {r.id for r in rows}
    for l in lessons:
        assert l.level in LEVELS and l.evidence_tag in ("observed", "derived", "proposed") and l.rule and l.family == "quadrature"
        assert set(l.support) <= ids_in_group and (l.evidence_tag == "observed") == bool(l.support)
        assert l.provenance["provider"] == frontier.name and l.provenance["mode"] == frontier.mode and l.tokens > 0
    log = h.bank.consolidate(lessons, 2000, family="quadrature")
    assert all(x["action"] in ("add", "replace", "skip", "retire") for x in log) and h.bank.active("quadrature")
    gate = h.gate(target)
    assert gate["lesson"] is not None and gate["lesson"].id in gate["candidates"]
    chk = gate["check"]
    assert chk["lesson_id"] == gate["lesson"].id and isinstance(chk["valid"], bool)
    if chk["valid"]:
        assert validate_expression(chk["expression"])[0]
    hrows = frontier.ledger.attempts(role="harness:lessons")
    assert [r.context["task"] for r in hrows] == ["extract", "gate"] and all(r.provider == frontier.name for r in hrows)
    assert all(r.mode == frontier.mode for r in hrows) and (all(r.api_dollars == 0 for r in hrows) if frontier.recorded else all(r.api_dollars > 0 for r in hrows))
    # the gate's lesson text and check are what the notebook_gate arm puts in its prompt; nothing here is a score
    assert "One lesson chosen" in LessonHarness.gate_text(gate)
