"""The ablation harness end to end on recorded fixtures: three arms on quad-d7 (no notebook, notebook, notebook with
gate), every attempt in the ledger with its arm and lesson ids, the gate's check logged and never scored, the
report written. The local arms replay the synthetic fixtures (tests/make_fixtures.py writes the notebook arms'
sequences from the recorded lessons), so the three arms measure the machinery, not the model."""
import json
from pathlib import Path

import pytest

from lab.config import lab_config
from lab.harness.ablation import ARMS, run_ablation
from lab.harness.lessons import LessonBank, LessonHarness
from lab.ledger import Ledger

from make_fixtures import EXPECT  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def test_ablation_end_to_end(frontier, tmp_path):
    cfg = lab_config(env={"LAB_ROOT": str(tmp_path)})
    led = Ledger(tmp_path / "ledger.sqlite")
    harness = LessonHarness(frontier.provider, frontier.ledger, LessonBank(frontier.ledger), cfg=cfg)
    rep = run_ablation(target_id="quad-d7", attempts=16, seed=0, cfg=cfg, ledger_path=led.path, ledger=led, harness=harness,
                       provider_name="local", mode="offline", gpu="A100", concurrency=4, fixtures_root=ROOT, quiet=True,
                       cache_dir=tmp_path / "cache", timeout_s=30, out_dir=tmp_path / "ablations")
    assert set(rep["arms"]) == set(ARMS) and rep["harness"]["provider"] == frontier.name and rep["harness"]["mode"] == frontier.mode
    assert rep["harness"]["calls"] == 2 and rep["lessons_extracted"] and rep["gate"]["lesson_id"]
    a = rep["arms"]["no_notebook"]
    assert a["attempts"] == 16 and a["passed"] == EXPECT["valid"] and a["optimal"] == EXPECT["optimal"]
    rows = led.attempts(target="quad-d7")
    by_arm = {arm: [r for r in rows if r.context.get("arm") == arm] for arm in ARMS}
    assert all(len(v) == 16 for v in by_arm.values())
    assert all(r.context["lesson_ids"] == [] for r in by_arm["no_notebook"])
    assert all(r.context["lesson_ids"] == rep["notebook_ids"] and r.context["prompt_override"] for r in by_arm["notebook"])
    assert all(r.context["lesson_ids"] == [rep["gate"]["lesson_id"]] for r in by_arm["notebook_gate"])
    if not frontier.recorded:
        # record mode: the notebook arms' local fixtures do not exist until tests/make_fixtures.py has run
        assert all(r.api_dollars > 0 for r in frontier.ledger.attempts(role="harness:lessons"))
        return
    missing = [r for arm in ("notebook", "notebook_gate") for r in by_arm[arm] if r.output is None]
    assert not missing, ("the notebook arms have no local replay fixtures: run `python tests/make_fixtures.py` after recording "
                         f"the {frontier.name} fixtures ({missing[0].diagnosis})")
    for arm in ("notebook", "notebook_gate"):
        c = rep["arms"][arm]
        assert c["attempts"] == 16 and c["passed"] == EXPECT["valid"] and c["optimal"] == EXPECT["optimal"]   # same synthetic replies
        assert c["api_dollars"] == 0.0
    assert rep["gain_over_no_notebook"]["notebook"]["pass@1_valid"] == 0.0
    if rep["gate"]["check"] and rep["gate"]["check"]["valid"]:
        g = rep["arms"]["notebook_gate"]["gate_checks"]
        assert g["applied"] >= EXPECT["valid"] and all(r.meta["gate_check"] is not None for r in by_arm["notebook_gate"] if r.passed)
        assert sum(1 for r in by_arm["notebook_gate"] if r.passed) == EXPECT["valid"]   # the check never changes a score
    assert rep["matched_dollars"]["max_spread"] < 0.01 and all(r.api_dollars == 0 for r in rows)
    files = rep["files"]
    assert Path(files["json"]).exists() and json.loads(Path(files["json"]).read_text())["id"] == rep["id"]
    md = Path(files["markdown"]).read_text(encoding="utf-8")
    assert "| no_notebook |" in md and "| notebook_gate |" in md and "machinery check" in md
    assert frontier.ledger.api_spend(2026, 11) == 0.0
