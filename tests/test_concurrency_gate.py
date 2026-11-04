"""Concurrent attempts (a group as parallel requests gives the same rows as a sequential run), matched-dollar
stopping, prompt overrides with a gate check, and the gate runner's memos and verdicts, all on replayed fixtures."""
import json
import math
from pathlib import Path

import pytest

from lab.config import lab_config
from lab.demo import run_group
from lab.gate import BAND, gate_record, memo_markdown, run_gate, verdict
from lab.ledger import Ledger, prompt_hash
from lab.targets import SYSTEM_PROMPT, get_target

from make_fixtures import EXPECT, GAUSS4, SIMPSON  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def _group(tmp_path, **kw):
    cfg = lab_config(env={"LAB_ROOT": str(tmp_path)})
    args = dict(target_id="quad-d7", attempts=16, seed=0, provider_name="local", role_name="builder",
                ledger_path=tmp_path / "ledger.sqlite", gpu="A100", precision=None, units_per_hour=None, temperature=None,
                max_tokens=None, timeout_s=30, cfg=cfg, fixtures_root=ROOT, quiet=True, cache_dir=tmp_path / "cache",
                mode="offline")
    args.update(kw)
    return run_group(**args)


def test_concurrent_group_matches_the_sequential_run(tmp_path):
    s = _group(tmp_path, concurrency=8, judge_concurrency=4)
    assert s["attempts"] == 16 and s["passed"] == EXPECT["valid"] and s["optimal"] == EXPECT["optimal"]
    assert s["pass_at_k"]["pass@16"] == 1.0 and s["concurrency"] == 8 and s["stopped_at"] is None
    with Ledger(tmp_path / "ledger.sqlite") as led:
        rows = led.attempts(group_id=s["group_id"])
        assert sorted(r.seed for r in rows) == list(range(16))
        assert sorted(r.meta["attempt_index"] for r in rows) == list(range(16))
        assert all(r.meta["concurrency"] == 8 for r in rows)
        by_seed = {r.seed: r for r in rows}
        assert by_seed[0].certificate["optimal"] and not by_seed[9].certificate["passed"]


def test_matched_dollar_stop(tmp_path):
    s = _group(tmp_path, attempts=16, concurrency=4, stop_at_dollars=1e-9)
    assert s["stopped_at"] == 4 and s["attempts"] == 4 and s["requested_attempts"] == 16
    s = _group(tmp_path, attempts=16, concurrency=1, stop_at_dollars=1e-9)
    assert s["stopped_at"] == 1 and s["attempts"] == 1


def test_prompt_override_with_a_gate_check(tmp_path):
    target = get_target("quad-d7")
    prompt = target.prompt + "\n\nOne lesson chosen for this target: the weights of a rule on [-1, 1] sum to 2."
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    key = prompt_hash(messages).split(":")[1][:24]
    fx = tmp_path / "fx" / "tests" / "fixtures" / "providers" / "local"
    fx.mkdir(parents=True)
    (fx / f"{key}.seq.json").write_text(json.dumps([{"text": GAUSS4, "completion_tokens": 100}, {"text": SIMPSON, "completion_tokens": 50}]))
    seen = []

    def check(result, meta):
        nodes, weights = result
        ok = abs(sum(weights) - 2.0) < 1e-9
        seen.append(meta["seed"])
        return {"ok": ok, "value": sum(weights)}

    s = _group(tmp_path, attempts=2, fixtures_root=tmp_path / "fx", prompt=prompt,
               context_extra={"arm": "notebook_gate", "lesson_ids": ["les_x"]}, post_check=check)
    assert s["passed"] == 1 and s["prompt_override"] and s["gate_checks"] == {"applied": 2, "passed": 2, "failed": 0}
    assert sorted(seen) == [0, 1]
    with Ledger(tmp_path / "ledger.sqlite") as led:
        rows = led.attempts(group_id=s["group_id"])
        assert all(r.context["arm"] == "notebook_gate" and r.context["lesson_ids"] == ["les_x"] for r in rows)
        assert all(r.prompt_hash == prompt_hash(messages) for r in rows)
        assert all(r.meta["gate_check"]["ok"] is True for r in rows)   # Simpson's weights also sum to 2
        assert sum(1 for r in rows if r.passed) == 1                    # the check is logged, never scored


def test_thinking_budget_is_recorded_and_adds_to_max_tokens(tmp_path):
    s = _group(tmp_path, attempts=1, thinking_budget=512)
    assert s["thinking_budget"] == 512
    with Ledger(tmp_path / "ledger.sqlite") as led:
        r = led.attempts(group_id=s["group_id"])[0]
        assert r.context["thinking_budget"] == 512 and r.context["max_tokens"] == 4096 + 512


@pytest.mark.parametrize("p,expected", [(0.0, "below the band"), (0.049, "below the band"), (0.05, "in the band"),
                                        (0.5, "in the band"), (0.80, "in the band"), (0.81, "above the band"),
                                        (1.0, "above the band"), (float("nan"), "no attempts")])
def test_gate_verdicts(p, expected):
    v = verdict(p)
    assert v["verdict"] == expected and v["in_band"] == (expected == "in the band") and BAND == (0.05, 0.80)


def test_gate_runner_writes_memos_on_replay_fixtures(tmp_path):
    cfg = lab_config(env={"LAB_ROOT": str(tmp_path)})
    recs = run_gate(targets=["quad-d7"], attempts=16, thinking=[0], provider_name="local", mode="offline",
                    ledger_path=tmp_path / "ledger.sqlite", gpu="A100", precision=None, units_per_hour=None, concurrency=4,
                    out_dir=tmp_path / "gates", cfg=cfg, fixtures_root=ROOT, quiet=True, timeout_s=30, cache_dir=tmp_path / "cache")
    assert len(recs) == 1
    rec = recs[0]
    assert rec["target"] == "quad-d7" and rec["attempts"] == 16 and rec["pass_at_64_key"] == "pass@16"
    assert rec["verdict"] == "above the band" and rec["in_band"] is False     # quad-d7 is a verification target
    assert rec["pass_at_k_valid"]["pass@1"] == pytest.approx(7 / 16) and rec["pass_at_k_optimal"]["pass@1"] == pytest.approx(4 / 16)
    assert rec["mode"] == "offline" and rec["precision"] == "replay" and rec["api_dollars"] == 0.0
    memo = Path(rec["memo"])
    assert memo.exists() and memo.with_suffix(".json").exists()
    text = memo.read_text(encoding="utf-8")
    assert "above the band" in text and "pass@16" in text and "never enter a reported cell" in text
    assert json.loads(memo.with_suffix(".json").read_text())["group_id"] == rec["group_id"]


def test_gate_record_and_memo_from_a_summary():
    summary = {"target": "t", "best_known": 4.0, "attempts": 64, "pass_at_k": {"pass@1": 0.2, "pass@16": 0.6, "pass@64": 0.7},
               "optimal_at_k": {"pass@1": 0.1, "pass@16": 0.3, "pass@64": 0.4}, "provider": "local", "mode": "online",
               "model": "Qwen/Qwen3.6-27B", "model_version": "bf16:sha256:x", "precision": "bf16", "gpu": "G4", "passed": 13,
               "optimal": 6, "dollars": 1.25, "api_dollars": 0.0, "gpu_seconds": 3600.0, "cpu_seconds": 10.0, "units": 11.77,
               "wall_s": 400.0, "concurrency": 16, "group_id": "grp_x", "ledger": "/x/ledger.sqlite", "best_attempt_id": "att_y",
               "best_score": 4.0, "thinking_budget": 512, "requested_attempts": 64}
    rec = gate_record(summary)
    assert rec["verdict"] == "in the band" and rec["in_band"] and rec["pass_at_64_key"] == "pass@64"
    md = memo_markdown(rec)
    assert "in the band" in md and "may be reported" in md and "thinking budget 512" in md
    assert not math.isnan(rec["pass_at_k_valid"]["pass@64"])
