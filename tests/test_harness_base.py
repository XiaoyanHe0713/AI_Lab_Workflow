"""The harness base: every call is a ledger row with role harness:<name>, provider, mode and API dollars; offline
replays cost nothing; live calls pass the budget guard and send the GPT-5 parameters; JSON is parsed robustly;
a refused or failed call is a row with the error."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from lab.cost import Budget, BudgetExceeded, BudgetGuard, CostMeter
from lab.harness import Harness, parse_json
from lab.ledger import Ledger
from lab.roles import BudgetedProvider, OpenAIChatProvider, ReplayProvider, request_key

PRICES = {"colab": {"dollars_per_unit": 0.1, "units_per_hour": {"A100": 12.0}}, "cpu": {"dollars_per_hour": 0.5},
          "api": {"gpt-test": {"input": 1.0, "output": 10.0}}}


class _Handler(BaseHTTPRequestHandler):
    requests: list = []

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        _Handler.requests.append(body)
        resp = {"id": "chatcmpl-1", "object": "chat.completion", "created": 0, "model": body.get("model"),
                "choices": [{"index": 0, "message": {"role": "assistant", "content": '{"answer": 42}'}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100}}
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


class Toy(Harness):
    name = "toy"


@pytest.mark.parametrize("text,expected", [
    ('{"a": 1}', {"a": 1}),
    ('Sure:\n```json\n{"a": [1, 2]}\n```\nthanks', {"a": [1, 2]}),
    ('prefix {"a": {"b": 2}} suffix', {"a": {"b": 2}}),
    ('[1, 2, 3]', [1, 2, 3]),
    ('```\n{"x": "y"}\n```', {"x": "y"}),
    # the week-4 defect: prose after the object with a brace in it, or a brace in the prose before it: the first balanced object
    ('{"lemmas": [1]}\n\nNote: the set {x, y} is empty.', {"lemmas": [1]}),
    ('the orbit {a, b} then {"structure": "seven nodes"} and {more}', {"structure": "seven nodes"}),
])
def test_parse_json(text, expected):
    assert parse_json(text) == expected


def test_parse_json_failure():
    with pytest.raises(ValueError):
        parse_json("no json here")
    with pytest.raises(ValueError):
        parse_json(None)


def _record(dirpath, messages, text, model="gpt-test"):
    key = request_key(model, messages, None, None, None)
    (dirpath / f"{key}.json").write_text(json.dumps({"provider": "openai", "request": {"model": model, "messages": messages},
                                                      "response": {"text": text, "prompt_tokens": 10, "completion_tokens": 5,
                                                                   "latency_s": 0.1, "model": model}}))


def test_offline_replay_is_a_free_row_with_the_harness_role(tmp_path):
    rep = ReplayProvider(tmp_path, kind="openai", model="gpt-test", name="openai")
    h = Toy(rep, Ledger(tmp_path / "l.sqlite"))
    messages = [{"role": "system", "content": Toy.system_prompt}, {"role": "user", "content": "say hi"}]
    _record(tmp_path, messages, '{"hi": true}')
    r = h.ask("greet", "say hi", target="t1", context={"k": "v"})
    assert r.ok and r.data == {"hi": True} and r.mode == "offline" and r.api_dollars == 0.0 and r.provider == "openai"
    rows = h.ledger.attempts(role="harness:toy")
    assert len(rows) == 1 and rows[0].target == "t1" and rows[0].provider == "openai" and rows[0].mode == "offline"
    assert rows[0].api_dollars == 0.0 and rows[0].dollars == 0.0 and rows[0].context["task"] == "greet" and rows[0].context["k"] == "v"
    assert rows[0].precision == "replay" and rows[0].meta["parsed"] is True and rows[0].diagnosis is None
    assert h.ledger.api_spend(2026, 11) == 0.0 and h.spent == 0.0


def test_missing_fixture_is_a_row_with_the_error(tmp_path):
    rep = ReplayProvider(tmp_path, kind="openai", model="gpt-test", name="openai")
    h = Toy(rep, Ledger(tmp_path / "l.sqlite"))
    r = h.ask("greet", "never recorded")
    assert not r.ok and r.data is None and "offline mode" in r.parse_error
    row = h.ledger.attempts(role="harness:toy")[0]
    assert "offline mode" in row.diagnosis and row.output is None


def test_live_call_is_budgeted_priced_and_sends_gpt5_parameters(tmp_path, fake_server):
    live = OpenAIChatProvider(kind="openai", base_url=fake_server + "/v1", api_key="sk-test", model="gpt-test",
                              extra_body={"reasoning_effort": "low"})
    guard = BudgetGuard(Budget(monthly_cap_usd=200.0, warning_fraction=0.8))
    p = BudgetedProvider(live, guard, CostMeter(prices=PRICES))
    h = Toy(p, Ledger(tmp_path / "l.sqlite"), meter=CostMeter(prices=PRICES), max_tokens=777)
    r = h.ask("compute", "what is the answer?", target="t2")
    assert r.ok and r.data == {"answer": 42} and r.mode == "online"
    assert r.api_dollars == pytest.approx(1000 * 1.0 / 1e6 + 100 * 10.0 / 1e6)
    body = _Handler.requests[-1]
    assert body["max_completion_tokens"] == 777 and "max_tokens" not in body and "temperature" not in body
    assert body["reasoning_effort"] == "low" and body["response_format"] == {"type": "json_object"} and "seed" not in body
    row = h.ledger.attempts(role="harness:toy")[0]
    assert row.mode == "online" and row.provider == "openai" and row.api_dollars == pytest.approx(r.api_dollars)
    assert row.precision == "api" and row.meta["budget"]["counted"] and h.ledger.api_spend(2026, 11) > 0
    assert guard.session_spend == pytest.approx(r.api_dollars)


def test_budget_refusal_is_a_row_and_raises(tmp_path, fake_server):
    live = OpenAIChatProvider(kind="openai", base_url=fake_server + "/v1", api_key="sk-test", model="gpt-test")
    guard = BudgetGuard(Budget(monthly_cap_usd=0.0, warning_fraction=0.8))
    p = BudgetedProvider(live, guard, CostMeter(prices=PRICES))
    h = Toy(p, Ledger(tmp_path / "l.sqlite"), meter=CostMeter(prices=PRICES))
    with pytest.raises(BudgetExceeded):
        h.ask("compute", "anything")
    row = h.ledger.attempts(role="harness:toy")[0]
    assert row.diagnosis.startswith("BudgetExceeded") and row.output is None and row.api_dollars == 0.0


def test_harness_prompt_hash_is_stable_across_calls(tmp_path):
    rep = ReplayProvider(tmp_path, kind="openai", model="gpt-test", name="openai")
    h = Toy(rep, Ledger(tmp_path / "l.sqlite"))
    messages = [{"role": "system", "content": Toy.system_prompt}, {"role": "user", "content": "q"}]
    _record(tmp_path, messages, "[1]")
    a = h.ask("t", "q")
    b = h.ask("t", "q")
    rows = h.ledger.attempts(role="harness:toy")
    assert a.data == b.data == [1] and rows[0].prompt_hash == rows[1].prompt_hash and rows[0].id != rows[1].id
