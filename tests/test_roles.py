"""The role interface: provider kinds (local, openai, anthropic) against local fake servers, the three
modes (offline replay, online, record), adapter addressing, the budget guard, and code extraction."""
import json
import logging
import os
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from lab.config import lab_config
from lab.cost import Budget, BudgetExceeded, BudgetGuard, CostMeter
from lab.ledger import Attempt, Ledger, prompt_hash
from lab.roles import (AnthropicProvider, BudgetedProvider, OpenAIChatProvider, RecordingProvider, ReplayProvider, Role,
                       extract_code, make_provider, make_role, request_key)

CONTENT = "```python\ndef rule():\n    return [0.0], [2.0]\n```"


class _Handler(BaseHTTPRequestHandler):
    requests: list = []

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        _Handler.requests.append((self.path, dict(self.headers), body))
        if self.path.endswith("/messages"):  # the Anthropic Messages API shape
            resp = {"id": "msg_1", "type": "message", "role": "assistant", "model": body.get("model"),
                    "content": [{"type": "text", "text": CONTENT}], "stop_reason": "end_turn",
                    "usage": {"input_tokens": 42, "output_tokens": 17}}
        else:  # the OpenAI chat completions shape
            resp = {"id": "chatcmpl-1", "object": "chat.completion", "created": 0, "model": body.get("model"),
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": CONTENT}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 42, "completion_tokens": 17, "total_tokens": 59}}
        data = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


@pytest.fixture(scope="module")
def fake_server():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


PRICES = {"colab": {"dollars_per_unit": 0.1, "units_per_hour": {"A100": 12.0}}, "cpu": {"dollars_per_hour": 0.5},
          "api": {"gpt-test": {"input": 1.0, "output": 10.0}, "claude-test": {"input": 3.0, "output": 15.0}}}


def test_request_key_is_stable_and_sensitive():
    msgs = [{"role": "user", "content": "hi"}]
    a = request_key("m", msgs, None, 0.8)
    assert a == request_key("m", [{"content": "hi", "role": "user"}], [], 0.8) and len(a) == 24
    assert a != request_key("m", msgs, None, 0.5) != request_key("m2", msgs, None, 0.5)
    assert a != request_key("m", msgs, None, 0.8, seed=1) != request_key("m", msgs, None, 0.8, seed=2)
    assert a != request_key("m", msgs, [{"name": "t"}], 0.8)


def test_local_provider_against_a_fake_endpoint(fake_server):
    p = OpenAIChatProvider(kind="local", base_url=fake_server + "/v1", api_key=None, model="qwen3.6-27b")
    c = p.complete([{"role": "user", "content": "hi"}], temperature=0.1, max_tokens=50, seed=3)
    assert "def rule" in c.text and c.prompt_tokens == 42 and c.completion_tokens == 17 and c.latency_s > 0
    assert c.provider == "local" and c.mode == "online" and c.finish_reason == "stop" and c.api_dollars == 0.0
    path, _, body = _Handler.requests[-1]
    assert path.endswith("/chat/completions") and body["seed"] == 3 and body["model"] == "qwen3.6-27b"


def test_openai_provider_is_priced_through_the_budget_guard(fake_server):
    live = OpenAIChatProvider(kind="openai", base_url=fake_server + "/v1", api_key="sk-test", model="gpt-test")
    guard = BudgetGuard(Budget(monthly_cap_usd=200.0, warning_fraction=0.8))
    p = BudgetedProvider(live, guard, CostMeter(prices=PRICES))
    c = p.complete([{"role": "user", "content": "hi"}], temperature=0.2)
    assert c.provider == "openai" and c.mode == "online"
    assert c.api_dollars == pytest.approx(42 * 1.0 / 1e6 + 17 * 10.0 / 1e6)
    assert guard.session_spend == pytest.approx(c.api_dollars) and c.raw["budget"]["counted"]
    _, headers, _ = _Handler.requests[-1]
    assert headers.get("Authorization", headers.get("authorization")) == "Bearer sk-test"


def test_anthropic_provider_against_a_fake_messages_api(fake_server):
    p = AnthropicProvider(api_key="test-key", model="claude-test", base_url=fake_server)
    c = p.complete([{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}], temperature=1.3, max_tokens=64)
    assert "def rule" in c.text and c.prompt_tokens == 42 and c.completion_tokens == 17
    assert c.provider == "anthropic" and c.mode == "online" and c.finish_reason == "end_turn"
    path, headers, body = _Handler.requests[-1]
    assert path.endswith("/v1/messages") and body["system"] == "be brief"
    assert "temperature" not in body and c.raw["temperature_requested"] == 1.3  # the Messages API has no sampling temperature
    assert body["messages"] == [{"role": "user", "content": "hi"}] and body["max_tokens"] == 64
    lower = {k.lower(): v for k, v in headers.items()}
    assert "test-key" in (lower.get("x-api-key"), lower.get("authorization", "").replace("Bearer ", "")), lower


def test_role_addresses_its_adapter_and_hashes_the_prompt(fake_server):
    p = OpenAIChatProvider(kind="local", base_url=fake_server + "/v1", api_key=None, model="qwen3.6-27b")
    role = Role("builder", p, system_prompt="sys", adapter="builder", temperature=0.5, max_tokens=64, thinking_budget=0)
    r = role.attempt("prompt text", seed=11)
    _, _, body = _Handler.requests[-1]
    assert body["model"] == "builder"  # the LoRA adapter is served under the role's name
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["temperature"] == 0.5 and body["max_tokens"] == 64 and body["seed"] == 11
    assert r.prompt_hash == prompt_hash([{"role": "system", "content": "sys"}, {"role": "user", "content": "prompt text"}])
    assert r.tool_log[0]["tool"] == "chat.completions" and r.tool_log[0]["mode"] == "online" and r.served_model == "builder"
    assert Role("builder", p, adapter=None).served_model == "qwen3.6-27b"


def test_frontier_role_does_not_send_qwen_template_kwargs(fake_server):
    p = OpenAIChatProvider(kind="openai", base_url=fake_server + "/v1", api_key="k", model="gpt-test")
    Role("librarian", p).attempt("summarise", seed=None)
    _, _, body = _Handler.requests[-1]
    assert "chat_template_kwargs" not in body and "seed" not in body


def test_offline_replay_never_calls_and_fails_on_a_missing_fixture(tmp_path):
    rep = ReplayProvider(tmp_path, kind="openai", model="gpt-test", name="openai")
    with pytest.raises(FileNotFoundError, match="offline mode"):
        rep.complete([{"role": "user", "content": "never recorded"}], temperature=0.8)


def test_record_then_replay_round_trip(tmp_path, fake_server):
    live = OpenAIChatProvider(kind="openai", base_url=fake_server + "/v1", api_key="k", model="gpt-test")
    rec = RecordingProvider(live, tmp_path)
    msgs = [{"role": "user", "content": "record me"}]
    c1 = rec.complete(msgs, temperature=0.3, seed=1)
    assert c1.mode == "record"
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1 and files[0].stem == request_key("gpt-test", msgs, None, 0.3, 1)
    rec_json = json.loads(files[0].read_text(encoding="utf-8"))
    assert rec_json["request"]["messages"] == msgs and rec_json["response"]["text"] == CONTENT and rec_json["provider"] == "openai"
    rep = ReplayProvider(tmp_path, kind="openai", model="gpt-test", name="openai")
    c2 = rep.complete(msgs, temperature=0.3, seed=1)
    assert c2.text == c1.text and c2.prompt_tokens == 42 and c2.mode == "offline" and c2.api_dollars == 0.0
    with pytest.raises(FileNotFoundError):
        rep.complete(msgs, temperature=0.9, seed=1)  # a different temperature is a different request


def test_replay_sequence_fixture(tmp_path):
    msgs = [{"role": "user", "content": "seq"}]
    key = prompt_hash(msgs).split(":")[1][:24]
    (tmp_path / f"{key}.seq.json").write_text(json.dumps([{"text": "a", "completion_tokens": 1}, {"text": "b", "completion_tokens": 2}]))
    rep = ReplayProvider(tmp_path)
    assert rep.complete(msgs, seed=0).text == "a" and rep.complete(msgs, seed=1).text == "b" and rep.complete(msgs, seed=2).text == "a"


def test_make_provider_modes_and_keys(tmp_path, fake_server, monkeypatch):
    cfg = lab_config(env={"LAB_ROOT": str(tmp_path), "LAB_ENDPOINT": fake_server + "/v1", "OPENAI_BASE_URL": fake_server + "/v1"})
    assert cfg["providers"]["openai"]["mode"] == "offline" and cfg["providers"]["anthropic"]["kind"] == "anthropic"
    off = make_provider("openai", cfg, fixtures_root=tmp_path)  # offline by default: no key needed
    assert isinstance(off, ReplayProvider) and off.mode == "offline" and off.dir == tmp_path / "tests" / "fixtures" / "providers" / "openai"
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        make_provider("openai", cfg, mode="online")
    with pytest.raises(ValueError):
        make_provider("openai", cfg, mode="sometimes")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    on = make_provider("openai", cfg, mode="online")
    assert isinstance(on, BudgetedProvider) and on.kind == "openai" and on.mode == "online"
    rec = make_provider("openai", cfg, mode="record", fixtures_root=tmp_path)
    assert isinstance(rec, RecordingProvider) and isinstance(rec.inner, BudgetedProvider)
    local = make_provider("local", cfg, mode="online")
    assert isinstance(local, OpenAIChatProvider) and local.kind == "local" and local.model == "qwen3.6-27b"
    cfg["roles"]["builder"]["provider"] = "local"
    role = make_role("builder", cfg, fixtures_root=tmp_path, mode="offline")
    assert isinstance(role.provider, ReplayProvider) and role.temperature == 0.8


class _Counting:
    kind = "openai"
    name = "openai"
    model = "gpt-test"
    mode = "online"
    calls = 0

    def complete(self, messages, **kw):
        from lab.roles import Completion

        self.calls += 1
        return Completion(text="x", prompt_tokens=1_000_000, completion_tokens=0, latency_s=0.1, model="gpt-test",
                          provider="openai")


def test_budget_guard_refuses_at_the_cap_and_warns_once(caplog):
    inner = _Counting()
    guard = BudgetGuard(Budget(monthly_cap_usd=2.5, warning_fraction=0.8))  # each call costs $1.00 at these prices
    p = BudgetedProvider(inner, guard, CostMeter(prices=PRICES))
    with caplog.at_level(logging.WARNING, logger="lab.budget"):
        p.complete([{"role": "user", "content": "1"}])   # spent 0 -> 1.0
        p.complete([{"role": "user", "content": "2"}])   # spent 1.0 -> 2.0: warns (>= 80% of 2.5 is 2.0? no: checked before, 1.0)
        p.complete([{"role": "user", "content": "3"}])   # checked at 2.0 >= 2.0: warning once; -> 3.0
        with pytest.raises(BudgetExceeded, match="No call was made"):
            p.complete([{"role": "user", "content": "4"}])  # 3.0 >= 2.5: refused
    assert inner.calls == 3
    assert sum("frontier budget" in r.message for r in caplog.records) == 1
    assert guard.status()["spent_usd"] == pytest.approx(3.0)


def test_budget_guard_reads_the_month_from_the_ledger(ledger_path):
    with Ledger(ledger_path) as led:
        now = datetime.now(timezone.utc)
        for prov, mode, usd, ts in [("openai", "online", 150.0, now), ("anthropic", "record", 30.0, now),
                                    ("openai", "offline", 999.0, now), ("local", "online", 5.0, now),
                                    ("openai", "online", 500.0, now.replace(year=now.year - 1))]:
            led.add_attempt(Attempt(role="h", model="m", model_version="v", precision="api", target="t", prompt_hash="h",
                                    provider=prov, mode=mode, api_dollars=usd, dollars=usd, timestamp=ts.isoformat()))
        guard = BudgetGuard(Budget(monthly_cap_usd=200.0, warning_fraction=0.8), ledger=led)
        assert guard.ledger_spend() == pytest.approx(180.0)
        assert guard.check("openai", "offline") == {"counted": False, "cap_usd": 200.0}  # offline never counts
        assert guard.check("local", "online")["counted"] is False
        st = guard.check("openai", "online")
        assert st["counted"] and st["spent_usd"] == pytest.approx(180.0) and guard.warned
        guard.charge(25.0)
        with pytest.raises(BudgetExceeded):
            guard.check("anthropic", "record")


@pytest.mark.online(key="OPENAI_API_KEY")
def test_openai_live_smoke(tmp_path):
    cfg = lab_config(env={"LAB_ROOT": str(tmp_path), **dict(os.environ)})
    p = make_provider("openai", cfg, mode="record", fixtures_root=tmp_path)
    c = p.complete([{"role": "user", "content": "Reply with the single word OK."}], temperature=0.0, max_tokens=5)
    assert c.text.strip() and c.mode == "record" and c.api_dollars > 0 and list(tmp_path.rglob("*.json"))


@pytest.mark.online(key="ANTHROPIC_API_KEY")
def test_anthropic_live_smoke(tmp_path):
    cfg = lab_config(env={"LAB_ROOT": str(tmp_path), **dict(os.environ)})
    p = make_provider("anthropic", cfg, mode="record", fixtures_root=tmp_path)
    c = p.complete([{"role": "user", "content": "Reply with the single word OK."}], temperature=0.0, max_tokens=5)
    assert c.text.strip() and c.mode == "record" and c.api_dollars > 0 and list(tmp_path.rglob("*.json"))


@pytest.mark.parametrize("text,expected", [
    ("Here:\n```python\nx = 1\n```\nbye", "x = 1\n"),
    ("```py\nx = 1\n```", "x = 1\n"),
    ("first\n```python\na = 1\n```\nthen\n```python\nb = 2\n```", "b = 2\n"),
    ("def rule():\n    return [0.0], [2.0]\n", "def rule():\n    return [0.0], [2.0]\n"),
    ("no code here, just words.", None),
    ("", None),
])
def test_extract_code(text, expected):
    assert extract_code(text) == expected
