"""The checker: the attempt view, the detector grammar, the held-out verification rule (kept only when right on
more than nine of ten held-out cases and every firing is a failure), the built-in detectors, the baseline record,
the dossier schema; and, per recorded provider, detector mining and a dossier."""
import pytest

from lab.harness.checker import (
    BUILTIN_DETECTORS,
    Checker,
    attempt_view,
    evaluate_condition,
    evaluate_detector,
    split_pools,
    validate_dossier,
    validate_spec,
    verify_detector,
)
from lab.ledger import Attempt, Ledger
from lab.targets import get_target

from conftest import demo_group  # noqa: E402


@pytest.fixture(scope="module")
def group(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("checker")
    led = Ledger(tmp / "ledger.sqlite")
    s = demo_group(tmp, led, attempts=16)
    rows = led.attempts(group_id=s["group_id"])
    return led, s, rows


def test_attempt_view_fields(group):
    _, _, rows = group
    by_seed = {r.seed: r for r in rows}
    v = attempt_view(by_seed[0])
    assert v["passed"] and v["node_count"] == 4 and abs(v["weight_sum_error"]) < 1e-12 and v["has_code"] and v["sandbox_ok"]
    assert attempt_view(by_seed[8])["has_code"] is False
    assert attempt_view(by_seed[5])["forbidden_import"] is True and attempt_view(by_seed[5])["sandbox_ok"] is False
    assert attempt_view(by_seed[12])["weight_sum_error"] > 1e-7     # the perturbed weight


def test_conditions_and_specs():
    v = {"weight_sum_error": 0.001, "has_code": True, "diagnosis": "not exact: max monomial error", "static_violations": ["forbidden import 'os'"]}
    assert evaluate_condition({"field": "weight_sum_error", "op": "abs_gt", "value": 1e-6}, v)
    assert not evaluate_condition({"field": "weight_sum_error", "op": "<", "value": 1e-6}, v)
    assert evaluate_condition({"field": "has_code", "op": "is_true"}, v) and evaluate_condition({"field": "diagnosis", "op": "contains", "value": "monomial"}, v)
    assert evaluate_condition({"field": "static_violations", "op": "contains", "value": "'os'"}, v)
    assert evaluate_condition({"field": "node_count", "op": "is_null"}, v) and not evaluate_condition({"field": "node_count", "op": ">", "value": 1}, v)
    with pytest.raises(ValueError):
        evaluate_condition({"field": "nope", "op": "==", "value": 1}, v)
    assert validate_spec({"all": [{"field": "has_code", "op": "is_false"}]})[0]
    assert not validate_spec({"all": [{"field": "passed", "op": "is_false"}]})[0]      # may not read the verdict
    assert not validate_spec({"all": []})[0] and not validate_spec("x")[0]
    assert not evaluate_detector({"all": []}, v)


def test_split_pools_is_stable_by_seed(group):
    _, _, rows = group
    mining, heldout = split_pools(rows, seed=0)
    assert sorted(a.seed for a in mining) == list(range(0, 16, 2)) and sorted(a.seed for a in heldout) == list(range(1, 16, 2))
    m2, h2 = split_pools(rows, seed=1)
    assert sorted(a.seed for a in m2) == list(range(1, 16, 2)) and sorted(a.seed for a in h2) == list(range(0, 16, 2))


def test_builtin_detectors_through_the_heldout_gate(group):
    _, _, rows = group
    _, heldout = split_pools(rows, seed=0)
    by_name = {b["name"]: verify_detector(b["spec"], heldout) for b in BUILTIN_DETECTORS}
    crash = by_name["sandbox_crash"]
    assert crash["fires"] == 2 and crash["true_positives"] == 2 and crash["kept"] and crash["right"] == 8
    assert not by_name["no_code_block"]["kept"] and by_name["no_code_block"]["fires"] == 0       # fires too rarely on the held-out pool
    assert not by_name["forbidden_import"]["kept"] and by_name["forbidden_import"]["fires"] == 1
    # a detector that fires on a certified attempt is never kept, however often it fires
    bad = verify_detector({"all": [{"field": "has_code", "op": "is_true"}]}, heldout)
    assert bad["fires"] == 8 and not bad["kept"] and bad["precision"] < 0.9
    # a detector that never fires is vacuously right but not kept
    never = verify_detector({"all": [{"field": "node_count", "op": ">", "value": 1000}]}, heldout)
    assert never["right"] == 8 and never["fires"] == 0 and not never["kept"]


def test_baseline_record_flags_replays(group):
    led, s, rows = group
    rec = Checker.baseline_record(led)
    entries = rec["targets"]["quad-d7"]
    assert len(entries) == 1 and entries[0]["group_id"] == s["group_id"] and entries[0]["attempts"] == 16
    assert entries[0]["passed"] == 7 and entries[0]["optimal"] == 4 and entries[0]["reportable"] is False
    assert Checker.baseline_record(led, reportable_only=True)["targets"] == {}


def test_dossier_schema():
    a = Attempt(role="builder", model="m", model_version="v", precision="bf16", target="t", prompt_hash="h",
                certificate={"passed": True, "score": 4.0})
    ev = {k: None for k in ("attempt_id", "target", "certifier", "certifier_version", "score", "passed", "certificate_sha256",
                            "code_sha256", "summary")}
    ev.update({"attempt_id": a.id, "score": 4.0, "passed": True})
    good = {"claim": "the four node rule is exact to degree seven", "conditions": ["interval"], "evidence": ev}
    assert validate_dossier(good, a) == []
    assert "missing claim" in validate_dossier({"conditions": ["x"], "evidence": ev}, a)
    assert any("sentence" in e for e in validate_dossier({**good, "claim": "ok"}, a))
    assert any("empty" in e for e in validate_dossier({**good, "conditions": []}, a))
    assert any("attempt_id" in e for e in validate_dossier({**good, "evidence": {**ev, "attempt_id": "other"}}, a))
    assert any("score" in e for e in validate_dossier({**good, "evidence": {**ev, "score": 5.0}}, a))
    assert validate_dossier("nonsense") == ["a dossier is a dict"]


def test_mining_and_dossier_replay_per_provider(frontier, group):
    _, s, rows = group
    target = get_target("quad-d7")
    chk = Checker(frontier.provider, frontier.ledger)
    rep = chk.mine(target, rows, seed=0)
    assert rep["mining_n"] == 8 and rep["heldout_n"] == 8 and rep["proposed"] >= 1
    kept = {d["name"] for d in rep["kept"]}
    assert "sandbox_crash" in kept
    for d in rep["kept"]:
        ver = d["verification"]
        assert ver["fires"] >= 2 and ver["true_positives"] == ver["fires"] and ver["right"] > 0.9 * ver["heldout_n"]
    stored = chk.detectors("quad-d7")
    assert {d["name"] for d in stored} == kept and all(d["kept"] == 1 for d in stored)
    assert any(d["origin"] == "model" for d in frontier.ledger.rows("detectors", json_columns=("spec",)))
    by_seed = {r.seed: r for r in rows}
    assert "sandbox_crash" in chk.diagnose(by_seed[9]) and chk.diagnose(by_seed[0]) == []
    dos = chk.dossier(by_seed[0], target)
    assert dos["valid"], dos["errors"]
    assert dos["evidence"]["attempt_id"] == by_seed[0].id and dos["evidence"]["score"] == 4.0 and dos["conditions"]
    assert frontier.ledger.rows("dossiers")[0]["valid"] == 1
    hrows = frontier.ledger.attempts(role="harness:checker")
    assert [r.context["task"] for r in hrows] == ["propose_detectors", "dossier"]
    assert all(r.provider == frontier.name and r.mode == frontier.mode for r in hrows)
    with pytest.raises(ValueError):
        chk.dossier(by_seed[8], target)
