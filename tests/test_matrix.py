"""The source-target matrix on replayed fixtures: experience sets from the archive, the two controls, the rule that
lessons are never measured on the family they came from, the cell runner at matched attempts with seeds, the ledger
query, the certified gain per cell with its bootstrap interval, and the markdown; the carriers' prompts."""
import json
from pathlib import Path

import pytest

from lab.config import lab_config
from lab.harness.carriers import CARRIERS, apply_carrier, lessons_block, process_only_lessons, redact, skills_from_archive
from lab.harness.matrix import (TransferLeak, analyze, bootstrap_interval, check_transfer, experience_from_archive, matrix_markdown,
                                process_only_control, run_cell, run_matrix, unrelated_control)
from lab.ledger import Ledger
from lab.targets import CONTENT_TERMS, get_target

from fixtures_week4 import MATRIX_CARRIERS, MATRIX_SOURCES, MATRIX_TARGETS, SQUARE_EXPECT, matrix_state  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def state(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("matrix")
    cfg = lab_config(env={"LAB_ROOT": str(tmp)})
    led = Ledger(tmp / "ledger.sqlite")
    st = matrix_state(tmp, led, cfg)
    st.update({"tmp": tmp, "cfg": cfg, "ledger": led})
    yield st
    led.close()


def _run_kw(tmp):
    return dict(provider_name="local", mode="offline", gpu="A100", precision=None, units_per_hour=None, temperature=None, max_tokens=None,
                timeout_s=60, fixtures_root=ROOT, cache_dir=tmp / "cache", concurrency=4)


def test_experience_sets_and_controls(state):
    led, archive, bank = state["ledger"], state["archive"], state["bank"]
    exp = experience_from_archive(led, archive, bank, MATRIX_SOURCES)
    assert exp.kind == "archive" and exp.family_keys == ["quadrature/interval"] and len(exp.lesson_ids) == 4 and len(exp.artifact_ids) == 3
    assert len(exp.attempt_ids) == 16 and "legendre" in exp.content_terms and exp.id.startswith("exp_")
    po = process_only_control(exp)
    assert po.kind == "process_only" and po.lesson_ids == exp.lesson_ids and po.artifact_ids == []
    un = unrelated_control(led, archive, bank, "quad-square-d5")
    assert un.kind == "unrelated" and un.source_targets == ["ssp-s3-p3"] and len(un.lesson_ids) == 2 and un.certificate_families == ["time_integrator"]
    assert unrelated_control(led, archive, bank, "ssp-s3-p3").source_targets == ["quad-d7"]


def test_lessons_are_never_measured_on_the_family_they_came_from(state):
    exp = experience_from_archive(state["ledger"], state["archive"], state["bank"], MATRIX_SOURCES)
    for carrier in ("lessons", "lessons_gate", "skills", "process_only"):
        with pytest.raises(TransferLeak, match="own instances"):
            check_transfer(exp, "quad-d7", carrier)
        with pytest.raises(TransferLeak, match="same family"):
            check_transfer(exp, "quad-d9", carrier)
    st = check_transfer(exp, "quad-d9", "lessons", allow_same_family=True)
    assert st["flag"] == "same family (unreported)" and st["setting"] == "harder_regime"
    assert check_transfer(exp, "quad-square-d5", "lessons")["setting"] == "new_family_same_certificate"
    assert check_transfer(exp, "ssp-s3-p3", "skills")["setting"] == "new_domain"
    assert check_transfer(exp, "quad-d7", "none")["setting"] == "baseline"
    un = unrelated_control(state["ledger"], state["archive"], state["bank"], "quad-square-d5")
    assert check_transfer(un, "quad-square-d5", "unrelated")["setting"] == "new_domain"


def test_carrier_prompts(state):
    led, archive, bank = state["ledger"], state["archive"], state["bank"]
    exp = experience_from_archive(led, archive, bank, MATRIX_SOURCES)
    target = get_target("quad-square-d5")
    none = apply_carrier("none", target, None, bank=bank)
    assert none["prompt"] == target.prompt and not none["prompt_override"] and none["context"]["lesson_ids"] == []
    les = apply_carrier("lessons", target, exp, bank=bank, token_budget=2000)
    assert "Lessons from earlier attempts (the notebook):" in les["prompt"] and len(les["context"]["lesson_ids"]) == 4 and "Legendre" in les["prompt"]
    po = apply_carrier("process_only", target, process_only_control(exp), bank=bank, token_budget=2000)
    assert po["context"]["redacted"] and len(po["context"]["lesson_ids"]) == 2 and "[process]" in po["prompt"]
    assert "Legendre" not in po["prompt"] and "monomial" not in po["prompt"].split("the notebook")[1] and "[...]" in po["prompt"]
    sk = apply_carrier("skills", target, exp, bank=bank, archive=archive)
    assert len(sk["context"]["skill_ids"]) == 3 and "Skills from the archive" in sk["prompt"] and "def rule" in sk["prompt"] and "certified on quad-d7" in sk["prompt"]
    assert "Skill " in sk["prompt"] and not any(aid[:12] in sk["prompt"] for aid in exp.artifact_ids)   # labelled by code hash, not artifact id
    with pytest.raises(ValueError, match="lesson harness"):
        apply_carrier("lessons_gate", target, exp, bank=bank)
    with pytest.raises(ValueError):
        apply_carrier("nope", target, exp, bank=bank)
    tiny, ids = lessons_block([(l, l.text) for l in (bank.get(i) for i in exp.lesson_ids)], token_budget=30)
    assert len(ids) <= 1
    assert CARRIERS == ("none", "lessons", "lessons_gate", "skills", "process_only", "unrelated", "diagnosis_only", "second_try")


def test_redaction_removes_content_and_keeps_craft():
    terms = CONTENT_TERMS["quadrature"]
    assert redact("use the 4-point Gauss-Legendre rule with nodes at the roots of P4", terms) == "use the [...]-point [...] with [...] at the roots of P[...]"
    assert redact("verify the output against a known case before returning", terms) == "verify the output against a known case before returning"
    from lab.harness.lessons import Lesson

    ls = [Lesson(id="a", level="process", rule="self-test on every monomial up to degree 7", conditions=["quadrature"], evidence_tag="derived"),
          Lesson(id="b", level="approach", rule="use Gauss nodes", conditions=[], evidence_tag="observed")]
    out = process_only_lessons(ls, terms)
    assert len(out) == 1 and out[0][1] == "[process] self-test on every [...] up to [...]; applies when: [...]"


def test_matrix_cells_gains_and_intervals(state):
    led, archive, bank, tmp, cfg = state["ledger"], state["archive"], state["bank"], state["tmp"], state["cfg"]
    exp = experience_from_archive(led, archive, bank, MATRIX_SOURCES)
    rec = run_matrix(sources=[exp], carriers=MATRIX_CARRIERS, targets=MATRIX_TARGETS + ["quad-d9"], seeds=2, attempts=8, cfg=cfg, ledger=led,
                     bank=bank, archive=archive, harness=None, run_kw=_run_kw(tmp), quiet=True, out_dir=tmp / "matrices", seed0=0)
    assert len(rec["skipped"]) == 3 and all("same family" in s["reason"] for s in rec["skipped"])
    cells = {(c["carrier"], c["target"]): c for c in rec["cells"]}
    assert ("none", "quad-square-d5") in cells and ("unrelated", "quad-d9") in cells and ("lessons", "quad-d9") not in cells
    base = cells[("none", "quad-square-d5")]
    assert [g["seed"] for g in base["groups"]] == [0, 1000] and all(g["passed"] == SQUARE_EXPECT["valid"] for g in base["groups"])
    an = rec["analysis"]
    by = {(c["carrier"], c["target"]): c for c in an["cells"]}
    les = by[("lessons", "quad-square-d5")]
    assert les["setting"] == "new_family_same_certificate" and les["seeds"] == 2 and les["attempts"] == 16 and les["gain"]["seeds"] == 2
    assert les["gain"]["pass_rate"] == pytest.approx(0.0) and les["gain"]["pass_rate_ci"] == (0.0, 0.0) and "2 seeds" in les["gain"]["method"]
    assert les["gain"]["best_score_gain"] == 0.0 and les["gain"]["matched_dollars_spread"] < 0.01 and les["gain"]["baseline_cell"] == base["cell_id"]
    po = by[("process_only", "quad-triangle-d2")]
    assert po["experience_kind"] == "process_only" and po["gain"]["optimal_rate"] == pytest.approx(0.0)
    un = by[("unrelated", "quad-square-d5")]
    assert un["setting"] == "new_domain" and un["source_targets"] == ["ssp-s3-p3"] and "gain_vs_framing" not in un and an["framing_rows"] == {}
    sk = by[("skills", "quad-triangle-d2")]
    assert sk["pass_rate"] == pytest.approx(0.5) and sk["gain"]["pass_rate"] == pytest.approx(0.0)
    # every cell row in the ledger carries the matrix fields
    rows = [r for r in led.attempts() if r.context.get("matrix_id") == rec["id"]]
    assert len(rows) == len(rec["cells"]) * 2 * 8
    assert all(r.context["cell_id"] and r.context["carrier"] in CARRIERS and r.context["setting"] and r.context["seed_index"] in (0, 1) for r in rows)
    assert all(r.context["lesson_ids"] for r in rows if r.context["carrier"] in ("lessons", "process_only", "unrelated"))
    assert all(r.context["skill_ids"] and r.context["prompt_override"] for r in rows if r.context["carrier"] == "skills")
    md = Path(rec["files"]["markdown"]).read_text(encoding="utf-8")
    assert "| archive: quad-d7 | lessons | quad-square-d5 | new_family_same_certificate | no |" in md and "Cells refused by the program" in md
    assert json.loads(Path(rec["files"]["json"]).read_text(encoding="utf-8"))["id"] == rec["id"]
    # the analysis is a query over the ledger: recomputed from scratch it agrees
    again = analyze(led, rec["id"])
    assert {c["cell_id"]: c["gain"] for c in again["cells"]} == {c["cell_id"]: c["gain"] for c in an["cells"]}
    assert "Transfer matrix" in matrix_markdown(again)


def test_single_seed_cells_bootstrap_over_attempts(state):
    led, archive, bank, tmp, cfg = state["ledger"], state["archive"], state["bank"], state["tmp"], state["cfg"]
    exp = experience_from_archive(led, archive, bank, MATRIX_SOURCES)
    mid = "mtx_single"
    run_cell(experience=None, carrier="none", target_id="quad-square-d5", seeds=1, attempts=8, cfg=cfg, ledger=led, bank=bank, archive=archive,
             matrix_id=mid, run_kw=_run_kw(tmp), quiet=True)
    run_cell(experience=exp, carrier="lessons", target_id="quad-square-d5", seeds=1, attempts=8, cfg=cfg, ledger=led, bank=bank, archive=archive,
             matrix_id=mid, run_kw=_run_kw(tmp), quiet=True)
    an = analyze(led, mid)
    g = [c for c in an["cells"] if c["carrier"] == "lessons"][0]["gain"]
    assert "one seed" in g["method"] and g["pass_rate_ci"][0] <= 0.0 <= g["pass_rate_ci"][1] and g["pass_rate_ci"][0] < g["pass_rate_ci"][1]
    with pytest.raises(TransferLeak):
        run_cell(experience=exp, carrier="lessons", target_id="quad-d7", seeds=1, attempts=1, cfg=cfg, ledger=led, bank=bank, archive=archive,
                 matrix_id=mid, run_kw=_run_kw(tmp), quiet=True)


def test_bootstrap_interval():
    assert bootstrap_interval([]) is None
    lo, hi = bootstrap_interval([0.1, 0.2, 0.3, 0.4, 0.5], n_boot=500)
    assert 0.1 <= lo <= 0.3 <= hi <= 0.5
    assert bootstrap_interval([0.25, 0.25]) == (0.25, 0.25)
