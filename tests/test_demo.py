"""The demo loop end to end on replayed fixtures: 16 attempts, judged, priced, written to the ledger,
pass@k reported, the best attempt printed; and the serving smoke test with a stub provider."""
import json
from pathlib import Path

import pytest

from lab.config import lab_config
from lab.demo import run_group
from lab.ledger import Ledger
from lab.smoke import smoke, throughput
from lab.targets import get_target, list_targets

from make_fixtures import EXPECT, RESPONSES  # noqa: E402  (tests/ is on sys.path under pytest)

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.timeout(600) if hasattr(pytest, "timeout") else (lambda f: f)
def test_demo_group_on_replay_fixtures(tmp_path):
    cfg = lab_config(env={"LAB_ROOT": str(tmp_path)})
    ledger = tmp_path / "ledger.sqlite"
    s = run_group(target_id="quad-d7", attempts=len(RESPONSES), seed=0, provider_name="local", role_name="builder",
                  ledger_path=ledger, gpu="A100", precision=None, units_per_hour=None, temperature=None,
                  max_tokens=None, timeout_s=30, cfg=cfg, fixtures_root=ROOT, quiet=True, cache_dir=tmp_path / "cache",
                  mode="offline")
    assert s["attempts"] == 16 and s["passed"] == EXPECT["valid"] and s["optimal"] == EXPECT["optimal"]
    assert s["best_score"] == 4.0 and s["best_known"] == 4.0
    assert s["pass_at_k"]["pass@1"] == pytest.approx(7 / 16) and s["pass_at_k"]["pass@16"] == 1.0
    assert s["optimal_at_k"]["pass@1"] == pytest.approx(4 / 16) and s["optimal_at_k"]["pass@16"] == 1.0
    assert s["verify_first"]["found"] == 4
    # replayed text costs no GPU time or tokens; the judge's CPU seconds are still priced
    assert s["units"] == 0.0 and s["gpu_seconds"] == 0.0 and s["api_dollars"] == 0.0 and 0.0 <= s["dollars"] < 0.01
    assert s["precision"] == "replay" and s["provider"] == "local" and s["provider_kind"] == "local" and s["mode"] == "offline"
    with Ledger(ledger) as led:
        rows = led.attempts(group_id=s["group_id"])
        assert len(rows) == 16 and all(r.target == "quad-d7" and r.role == "builder" for r in rows)
        assert all(r.provider == "local" and r.mode == "offline" and r.api_dollars == 0.0 for r in rows)
        assert all(r.context["mode"] == "offline" and r.meta["budget"] is None for r in rows)
        assert led.api_spend(2026, 11) == 0.0
        assert all(r.prompt_hash.startswith("sha256:") and r.seed == i for i, r in enumerate(rows))
        diag = {r.seed: r.diagnosis for r in rows}
        assert "ForbiddenImport" in diag[5] and "ForbiddenImport" in diag[6]
        assert "no code" in diag[8] and "syntax" in diag[9].lower() and "NameError" in diag[10]
        assert rows[0].certificate["optimal"] and rows[3].certificate["passed"] and not rows[3].certificate["optimal"]
        assert rows[7].certificate["score"] == 8.0 and rows[13].certificate["score"] == 5.0
        assert all(r.context["scoreboard"]["best_known"] == 4.0 for r in rows)
    group_file = ledger.parent / "groups" / f"{s['group_id']}.json"
    assert group_file.exists() and json.loads(group_file.read_text())["group_id"] == s["group_id"]


class _StubProvider:
    kind = "local"
    name = "local"
    mode = "online"
    model = "qwen3.6-27b"
    base_url = "http://stub/v1"

    def complete(self, messages, **kw):
        from lab.roles import Completion

        return Completion(text="A Gauss rule integrates polynomials exactly with optimally placed nodes.",
                          prompt_tokens=20, completion_tokens=30, latency_s=0.75, model=kw.get("model") or self.model,
                          provider=self.kind, finish_reason="stop")


def test_smoke_writes_serving_and_attempt_rows(tmp_path):
    cfg = lab_config(env={"LAB_ROOT": str(tmp_path)})
    ledger = tmp_path / "ledger.sqlite"
    row = smoke(gpu="H100", precision="bf16", ledger_path=str(ledger), cfg=cfg, provider=_StubProvider(), adapter="builder",
                units_per_hour=22.5)
    assert row["ok"] == 1 and row["tokens_per_second"] == pytest.approx(40.0) and row["units_per_hour"] == 22.5
    assert row["units_source"].startswith("entered") and row["adapter"] == "builder"
    assert row["dollars"] == pytest.approx(0.75 / 3600 * 22.5 * 0.09998, rel=1e-6)
    with Ledger(ledger) as led:
        srv = led.serving()
        assert len(srv) == 1 and srv[0]["precision"] == "bf16" and srv[0]["meta"]["served_model"] == "builder"
        att = led.attempts(target="smoke-test")
        assert len(att) == 1 and att[0].role == "smoke" and att[0].precision == "bf16"
        assert att[0].provider == "local" and att[0].mode == "online" and att[0].api_dollars == 0.0


def test_throughput_probe_writes_the_wave_as_one_serving_row(tmp_path):
    cfg = lab_config(env={"LAB_ROOT": str(tmp_path)})
    ledger = tmp_path / "ledger.sqlite"
    row = throughput(gpu="H100", precision="bf16", ledger_path=str(ledger), cfg=cfg, provider=_StubProvider(), concurrency=8,
                     units_per_hour=22.5)
    assert row["ok"] == 1 and row["concurrency"] == 8 and row["requests_ok"] == 8 and row["completion_tokens"] == 8 * 30
    assert row["tokens_per_second"] == pytest.approx(240.0 / row["latency_s"]) and row["meta"]["probe"] == "throughput"
    assert row["meta"]["per_request_tokens_per_second_mean"] == pytest.approx(40.0) and row["meta"]["errors"] == []
    with Ledger(ledger) as led:
        srv = led.serving()
        assert len(srv) == 1 and srv[0]["meta"]["concurrency"] == 8 and srv[0]["completion_tokens"] == 240
        att = led.attempts(target="throughput-probe")
        assert len(att) == 1 and att[0].role == "smoke" and att[0].context["concurrency"] == 8 and att[0].gpu_seconds == row["latency_s"]


def test_targets_resolve():
    ids = list_targets()
    assert "quad-d7" in ids and "pep-davis-yin-contraction-calibration" in ids
    t = get_target("quad-d7")
    assert t.best_known == 4 and t.status == "solved" and "degree at most 7" in t.prompt and t.entry_point == "rule"
    assert get_target("quad-square-d5").best_known == 7 and get_target("quad-triangle-d4").best_known == 6
    assert get_target("quad-d21").best_known == 11  # from the formula when not listed
    pi = get_target("quad-triangle-d3-pi")
    assert pi.params["require_positive"] and "positive weights" in pi.prompt
    with pytest.raises(KeyError):
        get_target("nope")
