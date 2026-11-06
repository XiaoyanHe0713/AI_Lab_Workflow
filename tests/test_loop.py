"""The loop end to end on replayed fixtures: one generation over three curriculum rungs with the theorist, the builder
and the breaker; every attempt of every role in the ledger with its context, tool log, certificate, diagnosis and
cost; survivors archived with witness records; gate memos; the generation record with its archive snapshot and
ledger pointer; the delayed credit of proposals; and, per recorded frontier provider, the same generation with the
lesson harness extracting lessons (skipped until those fixtures are recorded)."""
import json
from pathlib import Path

import pytest

from lab.config import lab_config
from lab.harness.archive import Archive
from lab.harness.lessons import LessonBank, LessonHarness
from lab.ledger import Ledger
from lab.loop import run_generation, survivors_of
from lab.targets import get_target

from fixtures_week4 import BREAKER_EXPECT, SQUARE_EXPECT, TRIANGLE_EXPECT  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TARGETS = ["quad-d7", "quad-square-d5", "quad-triangle-d2"]


def _generation(tmp_path, led, harness=None, **kw):
    cfg = lab_config(env={"LAB_ROOT": str(tmp_path)})
    args = dict(targets=TARGETS, attempts=8, cfg=cfg, ledger=led, archive=Archive(led, tmp_path), bank=LessonBank(led), harness=harness,
                provider_name="local", mode="offline", carrier="lessons", theorist_attempts=4, proposal_attempts=4, breaker_attempts=5,
                breaker_candidates=2, tool_calls=0, theorist_tool_calls=1, breaker_tool_calls=1, gpu="A100", max_tokens=None, thinking_budget=0,
                timeout_s=60, concurrency=4, fixtures_root=ROOT, cache_dir=tmp_path / "cache", seed=0, out_dir=tmp_path / "out", quiet=True)
    args.update(kw)
    return run_generation(**args)


def test_generation_end_to_end_on_fixtures(tmp_path):
    led = Ledger(tmp_path / "ledger.sqlite")
    rec = _generation(tmp_path, led)
    assert [r["target"] for r in rec["rungs"]] == TARGETS and rec["adapters"] == "none (frozen base)" and rec["mode"] == "offline"
    # the builder's gate groups: the synthetic pass rates, and a gate memo per rung
    base = {r["target"]: r["groups"]["base"] for r in rec["rungs"]}
    assert base["quad-d7"]["passed"] == 5 and base["quad-square-d5"]["passed"] == SQUARE_EXPECT["valid"] and base["quad-triangle-d2"]["passed"] == TRIANGLE_EXPECT["valid"]
    assert base["quad-square-d5"]["optimal"] == SQUARE_EXPECT["optimal"] and base["quad-square-d5"]["verify_first_found"] == 1
    for r in rec["rungs"]:
        memo = Path(r["gate"]["memo"])
        assert memo.exists() and memo.with_suffix(".json").exists() and r["gate"]["verdict"] == "in the band"
        assert json.loads(memo.with_suffix(".json").read_text())["generation_id"] == rec["id"]
        assert "lessons" not in r["groups"]           # the bank is empty without a harness: no lesson group
        # week 5: the base group was mined (built-in detectors, no checker); with 8 attempts none holds on the held-out half
        assert r["detectors"]["mining"] == "builtin" and r["detectors"]["kept"] == [] and r["symptoms"] == {}
        assert r["retrieval_query"] == f"{get_target(r['target']).title} quadrature"
    # the theorist: proposals checked by computer algebra, builds derived from them, the delayed credit
    sq = rec["rungs"][1]
    assert sq["groups"]["theorist"]["proposals"] == 4 and sq["groups"]["theorist"]["lemmas_passed"] >= 5
    assert sq["groups"]["proposals"]["attempts"] == 4 and len(sq["delayed_credit"]) == 4
    credits = {d["proposal_id"]: d for d in sq["delayed_credit"]}
    assert all(d["baseline_best"] == 7.0 and d["builds"] >= 1 for d in credits.values())
    assert any(d["credit"] == 0.0 and d["pass_rate_gain"] > 0 for d in credits.values())
    # the breaker: every reward zero on sound candidates, instances outside the class refused, nothing voided
    for r in rec["rungs"]:
        assert len(r["survivors"]) == 2 and len(r["breaker"]) == 2 and r["voided"] == [] and len(r["archived"]) == 2
        for b in r["breaker"]:
            assert b["attempts"] == 5 and b["confirmed_breaks"] == BREAKER_EXPECT["confirmed"] and b["reward_mean"] == 0.0
            assert b["outside_class"] >= 1 and b["no_witness"] >= 1 and b["dossier_id"] is None
    assert rec["exploit_alarms"] == []
    # the archive snapshot and the ledger pointer
    assert rec["archive_snapshot"]["count"] == 6 and len(rec["archive_snapshot"]["sha256"]) == 16
    assert rec["ledger_pointer"]["rows"] == led.count() and rec["ledger_pointer"]["last_attempt_id"]
    archive = Archive(led, tmp_path)
    tr = archive.trace(rec["rungs"][1]["archived"][0]["artifact_id"])
    assert tr["target"] == "quad-square-d5" and tr["witness_sha"] and tr["dossier_sha"] and tr["certifier"] == "quadrature_exactness"
    wit = archive.get(tr["witness_sha"])
    assert wit["attacks"] == 5 and wit["confirmed_breaks"] == 0 and wit["certifier"] == "witness_check"
    assert archive.get(tr["dossier_sha"])["written_by"]["harness"] == "template"
    # every attempt of every role is a row with the Work-package-4 fields
    rows = led.attempts()
    roles = {r.role for r in rows}
    assert roles == {"builder", "theorist", "breaker"}
    gen_rows = [r for r in rows if r.context.get("generation_id") == rec["id"]]
    assert len(gen_rows) == len(rows) and all(r.context["stage"] in ("base", "proposals", "theorist", "breaker") for r in gen_rows)
    for r in gen_rows:
        assert r.prompt_hash.startswith("sha256:") and r.context["adapter_version"] == "none:frozen-base" and r.context["role"] == r.role
        assert "lesson_ids" in r.context and "skill_ids" in r.context and "reading_list_snapshot" in r.context and r.context["tool_calls"] >= 0
        assert r.certificate and r.tool_log[-1]["tool"] == "judge" and r.seed is not None and r.dollars >= 0 and r.provider == "local"
        assert r.context["transfer"]["family"] == "quadrature" and r.context["carrier"] == "lessons"
        if not r.passed:
            assert r.diagnosis
    th = [r for r in gen_rows if r.role == "theorist"]
    assert all(r.context["citation_ids"] and r.context["reading_list_snapshot"]["count"] > 0 and r.meta["proposal"] is not None for r in th)
    br = [r for r in gen_rows if r.role == "breaker"]
    assert all(r.context["candidate_attempt_id"] and r.meta["witness_reward"] == 0.0 for r in br)
    assert any(r.meta["probe_rounds"] == 1 for r in br)
    derived = [r for r in gen_rows if r.context.get("derived_from_proposal")]
    assert derived and all(r.context["proposal_id"] in {t.id for t in th} for r in derived)
    assert len(led.rows("citations")) >= 9
    # the cost record and the files
    assert set(x["role"] for x in rec["cost"]["by_role"]) == roles and rec["cost"]["dollars"] > 0 and rec["cost"]["api_dollars"] == 0.0
    assert Path(rec["files"]["json"]).exists() and "machinery check" in Path(rec["files"]["markdown"]).read_text(encoding="utf-8")
    assert "Delayed credit on quad-square-d5" in Path(rec["files"]["markdown"]).read_text(encoding="utf-8")
    led.close()


def test_survivors_are_distinct_best_first():
    from lab.ledger import Attempt

    def row(code, score, seed):
        return Attempt(role="builder", model="m", model_version="v", precision="bf16", target="t", prompt_hash="h", seed=seed,
                       certificate={"passed": True, "score": score, "lower_is_better": True}, meta={"code": code})

    rows = [row("a", 9.0, 0), row("b", 7.0, 1), row("a", 9.0, 2), row("c", 16.0, 3),
            Attempt(role="builder", model="m", model_version="v", precision="bf16", target="t", prompt_hash="h", seed=4, certificate={"passed": False})]
    s = survivors_of(rows, 2)
    assert [r.meta["code"] for r in s] == ["b", "a"] and survivors_of(rows, 10)[-1].meta["code"] == "c"


def test_generation_with_lessons_from_a_recorded_provider(frontier, tmp_path):
    """With a harness the loop extracts lessons after each rung and the next rung's builders see the gate's lesson.
    Skipped until the frontier fixtures for these groups are recorded (the notebook's recording cell)."""
    led = Ledger(tmp_path / "ledger.sqlite")
    harness = LessonHarness(frontier.provider, led, LessonBank(led), cfg=lab_config(env={"LAB_ROOT": str(tmp_path)}))
    rec = _generation(tmp_path, led, harness=harness, theorist_attempts=0, proposal_attempts=0, breaker_attempts=0)
    missing = [c for c in harness.calls if c.parse_error and "FileNotFoundError" in c.parse_error]
    if missing:
        pytest.skip(f"no recorded {frontier.name} fixture for the loop's lesson extraction: record it with LAB_FRONTIER_MODE=record ({missing[0].parse_error[:120]})")
    assert rec["rungs"][0].get("lessons_extracted") and rec["lesson_bank"]["active"]
    later = [r for r in rec["rungs"][1:] if r["lessons"]["ids"]]
    assert later, "the second rung should have seen lessons from the first"
    for r in later:
        assert r["lessons"]["gate"] and r["lessons"]["gate"]["harness_attempt_id"]
        if "lessons" in r["groups"]:
            rows = led.attempts(group_id=r["groups"]["lessons"]["group_id"])
            assert rows and all(x.context["lesson_ids"] == r["lessons"]["ids"] and x.context["prompt_override"] for x in rows)
    assert all(x.api_dollars == 0.0 for x in led.attempts()) if frontier.recorded else True
    led.close()
