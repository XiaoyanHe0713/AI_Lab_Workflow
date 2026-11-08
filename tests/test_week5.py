"""Week 5: the diagnosis as a program's object (mining without a harness, the firing table, the query the diagnosis
names, the diagnosis-only text, mining from the ledger and its command), the lesson writer told which detectors fired
and the detectors a lesson addresses, the two target-side control rows of the matrix with the symptom-change column,
the loop mining its base groups and recording symptom rates per group, and, after the first run, every carrier's gain
against the framing row beside its gain against the baseline."""
from pathlib import Path

import pytest

from lab.config import lab_config
from lab.harness.carriers import CONTROL_CARRIERS
from lab.harness.checker import (base_groups, diagnosis_query, diagnosis_text, firing_table, kept_detectors, mine_builtin, mine_from_ledger,
                                 mining_markdown, symptom_rates)
from lab.harness.checker import main as checker_main
from lab.harness.lessons import Lesson, LessonBank, extraction_prompt, lesson_id
from lab.harness.matrix import analyze, check_transfer, experience_from_archive, run_matrix, unrelated_control
from lab.harness.matrix import main as matrix_main
from lab.ledger import Ledger
from lab.targets import get_target

from conftest import demo_group  # noqa: E402
from fixtures_week4 import CONTROL_ATTEMPTS, CONTROL_CARRIERS_TEST, CONTROL_SEEDS, MATRIX_SOURCES, MATRIX_TARGETS, matrix_run_kw, matrix_state  # noqa: E402
from test_loop import _generation  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def group(tmp_path_factory):
    """The week-2 demo group on quad-d7 (16 synthetic attempts, 7 valid): the base group a diagnosis is mined from."""
    tmp = tmp_path_factory.mktemp("week5")
    led = Ledger(tmp / "ledger.sqlite")
    s = demo_group(tmp, led, attempts=16)
    rows = sorted(led.attempts(group_id=s["group_id"]), key=lambda a: (a.seed, a.id))
    yield led, s, rows
    led.close()


# --- the checker: mining without a harness, the firing table, the query, the control text, the command ---------------------
def test_builtin_mining_firing_table_and_diagnosis(group):
    led, s, rows = group
    target = get_target("quad-d7")
    rep = mine_builtin(led, target, rows, seed=0)
    assert rep["proposed"] == 0 and rep["harness_attempt_id"] is None and {d["name"] for d in rep["kept"]} == {"sandbox_crash"}
    stored = kept_detectors(led, "quad-d7")
    assert [d["name"] for d in stored] == ["sandbox_crash"] and stored[0]["origin"] == "builtin" and stored[0]["provider"] == "program"
    table = firing_table(stored, rows)
    assert table["n"] == 16 and table["failed"] == 9 and table["detectors"][0]["name"] == "sandbox_crash"
    d = table["detectors"][0]
    assert d["fires"] == d["failures_fired"] >= 2 and d["rate"] == d["fires"] / 16 and len(d["attempt_ids"]) == d["fires"]
    assert all(table["by_attempt"][i] == ["sandbox_crash"] for i in d["attempt_ids"])
    assert symptom_rates(stored, rows)["sandbox_crash"] == {"fires": d["fires"], "n": 16, "rate": d["rate"]}
    plain = f"{target.title} quadrature"
    q = diagnosis_query(target, table)
    assert q.startswith(plain) and "raised or failed to run" in q
    assert diagnosis_query(target, None) == plain and diagnosis_query(target, {"n": 16, "detectors": []}) == plain
    text = diagnosis_text(target, table)
    assert text.startswith("What went wrong") and f"- {d['fires']} of 16 attempts: the candidate raised or failed to run" in text
    assert "no remedy" in text and "no detector fired" in diagnosis_text(target, None)
    # the base groups of the ledger, the command's mining report and its files
    groups = base_groups(led, "quad-d7")
    assert [g for g, _ in groups] == [s["group_id"]] and len(groups[0][1]) == 16
    reports = mine_from_ledger(led, ["quad-d7"])
    assert len(reports) == 1 and reports[0]["mining"] == "builtin" and reports[0]["kept"][0]["name"] == "sandbox_crash"
    assert reports[0]["attempts"] == 16 and reports[0]["query"] == q and reports[0]["diagnosis_text"] == text
    md = mining_markdown(reports)
    assert "| quad-d7 |" in md and "sandbox_crash" in md and "librarian query" in md
    assert checker_main(["--mine", "--targets", "quad-d7", "--ledger", str(led.path)]) == 0
    out = Path(led.path).resolve().parent / "detectors"
    assert any(p.name.startswith("mining-") for p in out.glob("*.md")) and any(p.name.startswith("mining-") for p in out.glob("*.json"))


# --- the lesson writer: the failure named only when detectors are given; the detectors a lesson addresses ------------------
def test_extraction_prompt_names_the_failure_only_when_detectors_are_given(tmp_path):
    led = Ledger(tmp_path / "ledger.sqlite")
    s = demo_group(tmp_path, led, attempts=16)
    rows = sorted(led.attempts(group_id=s["group_id"]), key=lambda a: (a.seed, a.id))
    target = get_target("quad-d7")
    plain, known = extraction_prompt(target, rows)
    assert known == {} and "Detectors" not in plain and "\"detectors\"" not in plain and plain.endswith("\"attempts\": [\"attempt 3\"]}]}")
    mine_builtin(led, target, rows, seed=0)
    kept = kept_detectors(led, "quad-d7")
    named, known = extraction_prompt(target, rows, detectors=kept)
    assert set(known) == {"sandbox_crash"} and "- sandbox_crash: the candidate raised or failed to run; fired on attempt " in named
    assert named.endswith("\"attempts\": [\"attempt 3\"], \"detectors\": [\"detector_name\"]}]}")
    block = named[named.index("\n\nDetectors"):named.index("\n\nWrite at most")]
    assert named.replace(block, "").replace(", \"detectors\": [\"detector_name\"]", "") == plain   # without detectors: the week-3 prompt
    # the detectors a lesson addresses survive the bank and merge on consolidation
    bank = LessonBank(led)
    rule = "check that the file runs before returning it"
    a = Lesson(id=lesson_id(rule, "process", "quadrature"), level="process", rule=rule, conditions=["any target"], evidence_tag="derived",
               family="quadrature", targets=["quad-d7"], detectors=["sandbox_crash"])
    bank.add(a)
    assert bank.get(a.id).detectors == ["sandbox_crash"]
    b = Lesson(id=lesson_id(rule, "process", "quadrature"), level="process", rule=rule, conditions=["any target"], evidence_tag="derived",
               family="quadrature", targets=["quad-d7"], detectors=["no_code_block"], support=["x"])
    bank.consolidate([b], 2000, family="quadrature")
    assert bank.get(a.id).detectors == ["no_code_block", "sandbox_crash"]
    led.close()


# --- the matrix: the two target-side control rows and the symptom-change column ---------------------------------------------
@pytest.fixture(scope="module")
def state(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("week5matrix")
    cfg = lab_config(env={"LAB_ROOT": str(tmp)})
    led = Ledger(tmp / "ledger.sqlite")
    st = matrix_state(tmp, led, cfg)
    st.update({"tmp": tmp, "cfg": cfg, "ledger": led})
    yield st
    led.close()


def test_target_side_controls_and_symptom_change(state):
    led, archive, bank, tmp, cfg = state["ledger"], state["archive"], state["bank"], state["tmp"], state["cfg"]
    for c in CONTROL_CARRIERS:
        st = check_transfer(None, "quad-square-d5", c)
        assert st["allowed"] and st["setting"] == "control" and st["flag"]
    rec = run_matrix(sources=[], carriers=CONTROL_CARRIERS_TEST, targets=MATRIX_TARGETS, seeds=CONTROL_SEEDS, attempts=CONTROL_ATTEMPTS, cfg=cfg,
                     ledger=led, bank=bank, archive=archive, harness=None, run_kw=matrix_run_kw(tmp), quiet=True, out_dir=tmp / "matrices", seed0=0)
    assert rec["skipped"] == [] and set(rec["diagnoses"]) == set(MATRIX_TARGETS)
    by = {(c["carrier"], c["target"]): c for c in rec["analysis"]["cells"]}
    # no unrelated carrier, so no framing row: the gains against it are absent and the report draws no column for them
    assert rec["analysis"]["framing_rows"] == {} and all("gain_vs_framing" not in c for c in rec["analysis"]["cells"])
    for tid in MATRIX_TARGETS:
        kept = {d["name"] for d in kept_detectors(led, tid)}
        assert kept and rec["diagnoses"][tid]["from"] == "none cell" and set(rec["diagnoses"][tid]["kept"]) == kept
        base, second, diag = by[("none", tid)], by[("second_try", tid)], by[("diagnosis_only", tid)]
        assert set(base["symptoms"]) == kept
        assert all(0 <= s["rate"] <= 1 and s["n"] == CONTROL_SEEDS * CONTROL_ATTEMPTS for s in base["symptoms"].values())
        assert second["tool_calls"] == 1 and second["setting"] == "control" and second["gain"]["pass_rate"] == pytest.approx(0.0)
        assert diag["tool_calls"] == 0 and diag["setting"] == "control" and diag["gain"]["pass_rate"] == pytest.approx(0.0)
        assert set(diag["gain"]["symptom_change"]) == kept and all(v == pytest.approx(0.0) for v in diag["gain"]["symptom_change"].values())
        # the rows: a second try made two requests per attempt and kept the file (the replay returns it again); the
        # diagnosis-only prompt names the detectors that fired and carries no lesson
        srows = [r for r in led.attempts() if r.context.get("cell_id") == second["cell_id"]]
        assert srows and all(r.context["tool_calls"] == 1 and r.meta.get("dry_runs") and r.meta["revised_after_dry_run"] is False for r in srows)
        drows = [r for r in led.attempts() if r.context.get("cell_id") == diag["cell_id"]]
        assert drows and all(r.context["carrier"] == "diagnosis_only" and r.context["prompt_override"] and r.context["lesson_ids"] == [] for r in drows)
        fired = [d["name"] for d in rec["diagnoses"][tid]["firing"] if d["fires"] > 0]
        assert fired and all(set(r.context["detectors_named"]) == set(fired) for r in drows)
    md = Path(rec["files"]["markdown"]).read_text(encoding="utf-8")
    assert "| symptom change |" in md and "| control: (target-side) | second_try | quad-square-d5 | control | no |" in md
    assert "| control: (target-side) | diagnosis_only | quad-triangle-d2 | control | no |" in md
    assert "Symptom rates of the baselines" in md and "target-side controls" in md and "vs framing" not in md
    again = analyze(led, rec["id"])
    assert {c["cell_id"]: c["symptoms"] for c in again["cells"]} == {c["cell_id"]: c["symptoms"] for c in rec["analysis"]["cells"]}
    # a second try is a condition of its own: its groups are not base groups, the default mining takes the largest base
    # group, and the cell counts the revisions after the dry run (none here: the replay keeps the file)
    from lab.harness.checker import largest_base_group
    from lab.harness.matrix import ExperienceSet
    for tid in MATRIX_TARGETS:
        cells = [c for c in rec["cells"] if c["target"] == tid]
        second_groups = {g["group_id"] for c in cells if c["carrier"] == "second_try" for g in c["groups"]}
        none_groups = {g["group_id"] for c in cells if c["carrier"] == "none" for g in c["groups"]}
        base = base_groups(led, tid)
        assert second_groups and not (second_groups & {g for g, _ in base}) and none_groups <= {g for g, _ in base}
        big = largest_base_group(base)
        assert big is not None and len(big[1]) == max(len(rows) for _, rows in base)
        assert mine_from_ledger(led, [tid])[0]["group_id"] == big[0]
        assert all(c["revised"] == 0 and c["revised_passed"] == 0 and all(g["revised"] == 0 for g in c["groups"]) for c in cells if c["carrier"] == "second_try")
    # the unrelated control with no other family in the archive is reported as the framing control, never refused
    empty = ExperienceSet(id="x", name="unrelated(empty)", kind="unrelated", source_targets=[], family_keys=[], certificate_families=[])
    fr = check_transfer(empty, "quad-square-d5", "unrelated")
    assert fr["allowed"] and fr["setting"] == "framing" and "framing control" in fr["flag"]


# --- the matrix: every carrier's gain against the framing row beside its gain against the baseline ---------------------------
FRAMING_CARRIERS = ["none", "unrelated", "second_try", "lessons", "process_only"]     # the matrix cell's narrowed set of 2026-11-08


@pytest.fixture(scope="module")
def framing_state(tmp_path_factory):
    """The matrix state with the other family's synthetic lessons retired, so that the archive holds one family and the
    unrelated control runs as the framing row (the plain prompt plus the empty-notebook sentence)."""
    tmp = tmp_path_factory.mktemp("week5framing")
    cfg = lab_config(env={"LAB_ROOT": str(tmp)})
    led = Ledger(tmp / "ledger.sqlite")
    st = matrix_state(tmp, led, cfg)
    fam = get_target(MATRIX_TARGETS[0]).transfer.family
    for l in st["bank"].all():
        if any(get_target(t).transfer.family != fam for t in l.targets):
            st["bank"].retire(l.id, "test: the archive holds one family, so the unrelated control is the framing row")
    st.update({"tmp": tmp, "cfg": cfg, "ledger": led})
    yield st
    led.close()


def test_gains_against_the_framing_row(framing_state, capsys):
    led, archive, bank, tmp, cfg = framing_state["ledger"], framing_state["archive"], framing_state["bank"], framing_state["tmp"], framing_state["cfg"]
    assert all(unrelated_control(led, archive, bank, tid).source_targets == [] for tid in MATRIX_TARGETS)
    exp = experience_from_archive(led, archive, bank, MATRIX_SOURCES)
    rec = run_matrix(sources=[exp], carriers=FRAMING_CARRIERS, targets=MATRIX_TARGETS, seeds=CONTROL_SEEDS, attempts=CONTROL_ATTEMPTS, cfg=cfg, ledger=led,
                     bank=bank, archive=archive, harness=None, run_kw=matrix_run_kw(tmp), quiet=True, out_dir=tmp / "matrices", seed0=0)
    assert rec["skipped"] == [] and len(rec["cells"]) == len(FRAMING_CARRIERS) * len(MATRIX_TARGETS)
    an = rec["analysis"]
    by = {(c["carrier"], c["target"]): c for c in an["cells"]}
    for tid in MATRIX_TARGETS:
        fr, base = by[("unrelated", tid)], by[("none", tid)]
        # the framing row is the reference: it keeps its own gain against the baseline, its gain against itself is None
        assert fr["setting"] == "framing" and an["framing_rows"][tid] == fr["cell_id"] and "framing control" in fr["flag"]
        assert fr["gain"]["baseline_cell"] == base["cell_id"] and fr["gain"]["pass_rate"] == pytest.approx(0.0) and fr["gain_vs_framing"] is None
        # every other cell carries the same three gains against the framing row, paired by seed with the same bootstrap; the
        # replay gives every carrier the same attempts, so every gain is 0.000 with an interval of [0, 0]
        for carrier in FRAMING_CARRIERS:
            if carrier == "unrelated":
                continue
            c = by[(carrier, tid)]
            g = c["gain_vs_framing"]
            assert g["framing_cell"] == fr["cell_id"] and g["seeds"] == CONTROL_SEEDS and f"{CONTROL_SEEDS} seeds" in g["method"]
            assert g["pass_rate"] == pytest.approx(0.0) and g["pass_rate_ci"] == (0.0, 0.0)
            assert g["optimal_rate"] == pytest.approx(0.0) and g["optimal_rate_ci"] == (0.0, 0.0) and g["best_score_gain"] == 0.0
            assert all(v == pytest.approx(0.0) for v in g["symptom_change"].values()) and g["matched_dollars_spread"] < 0.01
            if carrier != "none":
                assert c["gain"]["baseline_cell"] == base["cell_id"] and set(g) == (set(c["gain"]) - {"baseline_cell"}) | {"framing_cell"}
    # the analysis is a query over the ledger: recomputed, it agrees; the report draws the columns and names the reference
    again = analyze(led, rec["id"])
    assert {c["cell_id"]: c.get("gain_vs_framing") for c in again["cells"]} == {c["cell_id"]: c.get("gain_vs_framing") for c in an["cells"]}
    assert again["framing_rows"] == an["framing_rows"]
    md = Path(rec["files"]["markdown"]).read_text(encoding="utf-8")
    assert "| pass rate vs framing [95% CI] | gain in optimal rate [95% CI] | optimal rate vs framing [95% CI] | best score gain | best score vs framing |" in md
    assert "| framing: (the empty-notebook sentence) | unrelated | quad-square-d5 | framing | no | +0.000 [+0.000, +0.000] | reference |" in md
    assert "| archive: quad-d7 | lessons | quad-square-d5 | new_family_same_certificate | no | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |" in md
    assert "the reference every carrier's gain should be read against" in md and "holds one on quad-square-d5, quad-triangle-d2" in md
    # the rows of the 2026-11-07 matrix were written before the setting `framing` existed: they carry `no_sources` with an
    # empty source list and no flag, and the analysis reads them as the framing row all the same, so `--analyze` re-reads
    # that matrix with the new columns from the ledger alone
    fid = an["framing_rows"]["quad-square-d5"]
    led.conn.execute("UPDATE attempts SET context = json_set(json_remove(context, '$.transfer_flag'), '$.setting', 'no_sources') "
                     "WHERE json_extract(context, '$.cell_id') = ?", (fid,))
    old = analyze(led, rec["id"])
    oby = {(c["carrier"], c["target"]): c for c in old["cells"]}
    assert old["framing_rows"]["quad-square-d5"] == fid and oby[("unrelated", "quad-square-d5")]["setting"] == "framing"
    assert "framing control" in oby[("unrelated", "quad-square-d5")]["flag"]
    assert oby[("lessons", "quad-square-d5")]["gain_vs_framing"] == by[("lessons", "quad-square-d5")]["gain_vs_framing"]
    assert matrix_main(["--analyze", rec["id"], "--ledger", str(led.path)]) == 0
    out = capsys.readouterr().out
    assert "| framing: (the empty-notebook sentence) | unrelated | quad-square-d5 | framing | no |" in out and "| pass rate vs framing [95% CI] |" in out


# --- the loop: the base group mined, symptom rates per group, the query the diagnosis names ---------------------------------
def test_generation_mines_detectors_and_records_symptoms(tmp_path):
    led = Ledger(tmp_path / "ledger.sqlite")
    rec = _generation(tmp_path, led, targets=["quad-d7"], attempts=16, theorist_attempts=0, proposal_attempts=0, breaker_attempts=0)
    r = rec["rungs"][0]
    assert r["detectors"]["mining"] == "builtin" and r["detectors"]["proposed"] == 0 and "sandbox_crash" in r["detectors"]["kept"]
    assert r["detectors"]["attempts"] == 16 and r["detectors"]["failed"] == 9 and r["detectors"]["harness_attempt_id"] is None
    base = r["symptoms"]["base"]["sandbox_crash"]
    assert base["n"] == 16 and base["fires"] >= 2 and base["rate"] == base["fires"] / 16 and "raised or failed to run" in r["retrieval_query"]
    assert led.rows("detectors", "target = ? AND kept = 1", ("quad-d7",)) and rec["mine_detectors"] is True
    md = Path(rec["files"]["markdown"]).read_text(encoding="utf-8")
    assert "Detectors kept per rung" in md and "sandbox_crash" in md
    led2 = Ledger(tmp_path / "ledger2.sqlite")
    rec2 = _generation(tmp_path, led2, targets=["quad-d7"], attempts=16, theorist_attempts=0, proposal_attempts=0, breaker_attempts=0,
                       mine_detectors=False)
    assert rec2["rungs"][0]["detectors"]["mining"] == "off" and rec2["rungs"][0]["detectors"]["kept"] == [] and rec2["rungs"][0]["symptoms"] == {}
    assert rec2["rungs"][0]["retrieval_query"].endswith("quadrature")
    led.close()
    led2.close()
