import json
import math
import os

import pytest

from lab.config import expand, lab_config, model_config, prices_config
from lab.cost import Budget, BudgetGuard, Cost, CostMeter, load_budget
from lab.ledger import ATTEMPT_COLUMNS, Attempt, Ledger, pass_at_k, prompt_hash, summarize


# --- config -------------------------------------------------------------------------------------
def test_expand_defaults_and_nesting():
    env = {"A": "x"}
    assert expand("${A}", env) == "x"
    assert expand("${B:-fallback}", env) == "fallback"
    assert expand("${B:-${A}/ledger.sqlite}", env) == "x/ledger.sqlite"
    assert expand({"k": ["${A}", 3]}, env) == {"k": ["x", 3]}


def test_config_files_load():
    cfg = lab_config(env={"LAB_ROOT": "runs"})
    assert cfg["storage"]["ledger"].endswith("ledger.sqlite")
    assert {n: p["kind"] for n, p in cfg["providers"].items()} == {"local": "local", "openai": "openai", "anthropic": "anthropic"}
    assert cfg["providers"]["local"]["mode"] == "online" and cfg["providers"]["openai"]["mode"] == "offline"
    assert cfg["fixtures"] == "tests/fixtures/providers" and cfg["harness"]["provider"] == "openai"
    assert set(cfg["roles"]) >= {"builder", "theorist", "breaker"}
    assert all(r["provider"] == "local" for r in cfg["roles"].values())
    b = load_budget()
    assert b.monthly_cap_usd == 200.0 and b.warning_fraction == 0.8 and b.providers == ("openai", "anthropic")
    assert b.modes == ("online", "record") and b.path.endswith("budget.yaml")
    m = model_config()
    assert m["frozen_base"]["hf_repo"] == "Qwen/Qwen3.6-27B"
    assert m["serving"]["by_gpu"] == {"H100": "bf16", "G4": "bf16", "A100": "int4", "L4": "int4"}
    p = prices_config()
    assert p["colab"]["dollars_per_unit"] > 0


# --- cost ----------------------------------------------------------------------------------------
def test_gpu_cost_arithmetic():
    m = CostMeter(prices={"colab": {"dollars_per_unit": 0.1, "units_per_hour": {"A100": 12.0}}, "cpu": {"dollars_per_hour": 0.5}, "api": {}})
    c = m.gpu_cost("A100", 1800)
    assert c.gpu_seconds == 1800 and c.breakdown["units"] == pytest.approx(6.0) and c.dollars == pytest.approx(0.6)
    assert m.gpu_dollars_per_hour("A100") == pytest.approx(1.2)
    assert m.units_to_dollars(500) == pytest.approx(50.0)


def test_unknown_gpu_raises_and_override_is_recorded():
    m = CostMeter(prices={"colab": {"dollars_per_unit": 0.1, "units_per_hour": {}}, "api": {}})
    with pytest.raises(KeyError):
        m.rate("H100")
    m.set_units_per_hour("H100", 22.0)
    assert m.rate("H100") == (22.0, "entered from the Colab Resources panel")


def test_api_and_cpu_cost():
    m = CostMeter(prices={"colab": {}, "cpu": {"dollars_per_hour": 0.5}, "api": {"m": {"input": 1.0, "output": 10.0}}})
    assert m.api_cost("m", 1_000_000, 100_000).dollars == pytest.approx(2.0)
    with pytest.raises(KeyError):
        m.api_cost("other", 1, 1)
    assert m.cpu_cost(3600).dollars == pytest.approx(0.5)


def test_attempt_cost_by_provider_kind():
    m = CostMeter(prices={"colab": {"dollars_per_unit": 0.1, "units_per_hour": {"L4": 2.4}}, "cpu": {"dollars_per_hour": 0.5},
                          "api": {"gpt": {"input": 1.0, "output": 2.0}}})
    local = m.attempt_cost(provider_kind="local", gpu="L4", gpu_seconds=60, cpu_seconds=36, prompt_tokens=10, completion_tokens=5)
    assert local.dollars == pytest.approx(2.4 / 60 * 0.1 + 0.005) and local.api_dollars == 0.0
    api = m.attempt_cost(provider_kind="openai", model="gpt", prompt_tokens=1000, completion_tokens=1000)
    assert api.dollars == pytest.approx(0.003) and api.api_dollars == pytest.approx(0.003)
    rec = m.attempt_cost(provider_kind="anthropic", mode="record", model="gpt", prompt_tokens=1000, completion_tokens=0)
    assert rec.api_dollars == pytest.approx(0.001)
    # offline replays cost nothing: no GPU seconds, no tokens, whatever the kind; the judge's CPU is still priced
    off = m.attempt_cost(provider_kind="openai", mode="offline", model="gpt", prompt_tokens=1000, completion_tokens=1000, cpu_seconds=36)
    assert off.api_dollars == 0.0 and off.dollars == pytest.approx(0.005) and off.prompt_tokens == 1000
    assert m.attempt_cost(provider_kind="local", mode="offline", gpu_seconds=100).dollars == 0.0
    with pytest.raises(ValueError):
        m.attempt_cost(provider_kind="local", gpu=None)
    total = local + api
    assert total.dollars == pytest.approx(local.dollars + api.dollars) and total.prompt_tokens == 1010


def test_budget_guard_without_a_ledger():
    g = BudgetGuard(Budget(monthly_cap_usd=10.0, warning_fraction=0.5))
    assert g.month_spend() == 0.0 and g.check("openai", "online")["counted"] and not g.warned
    g.charge(5.0)
    assert g.check("anthropic", "record")["fraction"] == pytest.approx(0.5) and g.warned
    g.charge(5.0)
    import pytest as _pt

    with _pt.raises(Exception, match="budget exhausted"):
        g.check("openai", "online")
    assert g.check("openai", "offline")["counted"] is False


# --- ledger --------------------------------------------------------------------------------------
def test_attempt_schema_has_the_proposal_columns():
    names = [c for c, _ in ATTEMPT_COLUMNS]
    for required in ("id", "role", "model", "model_version", "precision", "target", "prompt_hash", "context", "output",
                     "tool_log", "certificate", "diagnosis", "gpu_seconds", "cpu_seconds", "dollars", "seed", "timestamp",
                     "provider", "mode", "api_dollars"):
        assert required in names


def test_api_spend_by_calendar_month(ledger_path):
    with Ledger(ledger_path) as led:
        def row(prov, mode, usd, ts):
            return Attempt(role="h", model="m", model_version="v", precision="api", target="t", prompt_hash="h",
                           provider=prov, mode=mode, api_dollars=usd, dollars=usd, timestamp=ts)
        led.add_attempt(row("openai", "online", 1.0, "2026-11-04T10:00:00+00:00"))
        led.add_attempt(row("anthropic", "record", 2.0, "2026-11-20T10:00:00+00:00"))
        led.add_attempt(row("openai", "offline", 4.0, "2026-11-21T10:00:00+00:00"))   # replays never count
        led.add_attempt(row("local", "online", 8.0, "2026-11-22T10:00:00+00:00"))     # the local endpoint is not frontier spend
        led.add_attempt(row("openai", "online", 16.0, "2026-10-31T23:59:59+00:00"))   # last month
        assert led.api_spend(2026, 11) == pytest.approx(3.0)
        assert led.api_spend(2026, 10) == pytest.approx(16.0)
        assert led.api_spend(2026, 11, kinds=("openai",)) == pytest.approx(1.0)
        assert led.api_spend(2026, 9) == 0.0


def test_v1_ledger_is_migrated_forward(tmp_path):
    import sqlite3

    p = tmp_path / "old.sqlite"
    con = sqlite3.connect(p)
    con.execute("CREATE TABLE attempts (id TEXT PRIMARY KEY, role TEXT NOT NULL, model TEXT NOT NULL, model_version TEXT NOT NULL, "
                "precision TEXT NOT NULL, target TEXT NOT NULL, prompt_hash TEXT NOT NULL, context TEXT NOT NULL, output TEXT, "
                "tool_log TEXT, certificate TEXT, diagnosis TEXT, gpu_seconds REAL, cpu_seconds REAL, dollars REAL, seed INTEGER, "
                "timestamp TEXT NOT NULL, group_id TEXT, meta TEXT)")
    con.execute("INSERT INTO attempts (id, role, model, model_version, precision, target, prompt_hash, context, timestamp) "
                "VALUES ('a1', 'builder', 'm', 'v', 'bf16', 't', 'h', '{}', '2026-11-01T00:00:00+00:00')")
    con.commit()
    con.close()
    with Ledger(p) as led:
        a = led.get_attempt("a1")
        assert a is not None and a.provider is None and a.mode is None and a.api_dollars in (None, 0.0)
        assert led.api_spend(2026, 11) == 0.0


def test_ledger_roundtrip_and_mirror(ledger_path):
    with Ledger(ledger_path) as led:
        a = Attempt(role="builder", model="m", model_version="bf16:abc", precision="bf16", target="quad-d7",
                    prompt_hash=prompt_hash([{"role": "user", "content": "hi"}]), context={"lesson_ids": [1, 2]},
                    output="```python\nx=1\n```", tool_log=[{"tool": "judge"}],
                    certificate={"passed": True, "score": 4.0, "optimal": True}, gpu_seconds=1.5, cpu_seconds=0.2,
                    dollars=0.001, seed=7, group_id="g1", meta={"k": "v"})
        led.add_attempt(a)
        b = led.get_attempt(a.id)
        assert b == a
        assert led.count() == 1
        assert led.attempts(target="quad-d7")[0].id == a.id and led.attempts(target="other") == []
        assert led.attempts(group_id="g1", role="builder")[0].passed and b.score == 4.0
    lines = ledger_path.with_suffix(".attempts.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["id"] == a.id


def test_serving_rows(ledger_path):
    with Ledger(ledger_path) as led:
        sid = led.add_serving(gpu="A100", precision="int4", model="m", model_version="v", tokens_per_second=35.2,
                              units_per_hour=11.77, units_source="config", dollars_per_hour=1.18, dollars=0.01, ok=1,
                              meta={"endpoint": "http://x"})
        rows = led.serving()
        assert rows[0]["id"] == sid and rows[0]["meta"]["endpoint"] == "http://x" and rows[0]["tokens_per_second"] == 35.2


def test_prompt_hash_is_stable():
    h1 = prompt_hash([{"role": "user", "content": "a"}])
    h2 = prompt_hash([{"content": "a", "role": "user"}])
    assert h1 == h2 and h1.startswith("sha256:")
    assert prompt_hash("a") != prompt_hash("b")


def test_pass_at_k_estimator():
    assert pass_at_k(16, 0, 1) == 0.0 and pass_at_k(16, 0, 16) == 0.0
    assert pass_at_k(16, 4, 1) == pytest.approx(0.25)
    assert pass_at_k(16, 4, 16) == 1.0
    assert pass_at_k(16, 1, 4) == pytest.approx(1 - math.comb(15, 4) / math.comb(16, 4))
    assert pass_at_k(16, 16, 64) == 1.0  # k capped at n
    assert math.isnan(pass_at_k(0, 0, 1))


def test_summarize_picks_lowest_score_and_counts_optimal():
    rows = [Attempt(role="b", model="m", model_version="v", precision="bf16", target="t", prompt_hash="h",
                    certificate={"passed": p, "score": s, "optimal": o, "lower_is_better": True}, dollars=0.01, gpu_seconds=1)
            for p, s, o in [(True, 5.0, False), (True, 4.0, True), (False, 3.0, False), (False, None, None)]]
    s = summarize(rows, ks=(1, 4))
    assert s["attempts"] == 4 and s["passed"] == 2 and s["optimal"] == 1
    assert s["best_score"] == 4.0 and s["best_attempt_id"] == rows[1].id
    assert s["pass_at_k"]["pass@1"] == pytest.approx(0.5) and s["optimal_at_k"]["pass@4"] == 1.0
    assert s["dollars"] == pytest.approx(0.04) and s["gpu_seconds"] == 4
