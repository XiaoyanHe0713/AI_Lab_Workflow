"""The thinking budget on the local endpoint: a reasoning request capped at the budget and stopped at the closing
tag, closed with the early-stop sentence when the budget runs out, then an answer request that continues the
assistant message; both calls logged and summed; a server that cannot continue leaves the thinking alone."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from lab.config import lab_config
from lab.demo import run_group
from lab.ledger import Ledger
from lab.roles import OpenAIChatProvider, Role, extract_code

CODE = "```python\ndef rule():\n    return [-0.5773502691896257, 0.5773502691896257], [1.0, 1.0]\n```"
THOUGHT = "<think>\nLet me think about Gauss rules with two nodes." + " More thinking." * 40


class _Handler(BaseHTTPRequestHandler):
    requests: list = []
    thinking_finish = "length"     # the server's finish reason for the thinking request
    refuse_continue = False        # make the answer request fail

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        _Handler.requests.append(body)
        if body.get("continue_final_message"):
            if _Handler.refuse_continue:
                data = json.dumps({"error": {"message": "continue_final_message is not supported", "type": "invalid_request_error"}}).encode()
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            content, finish, usage = "Here is the rule.\n\n" + CODE, "stop", {"prompt_tokens": 300, "completion_tokens": 60, "total_tokens": 360}
        else:
            content, finish, usage = THOUGHT, _Handler.thinking_finish, {"prompt_tokens": 200, "completion_tokens": int(body.get("max_tokens", 0)), "total_tokens": 200}
        resp = {"id": "chatcmpl-1", "object": "chat.completion", "created": 0, "model": body.get("model"),
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish}], "usage": usage}
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


@pytest.fixture
def provider(fake_server):
    _Handler.requests.clear()
    _Handler.thinking_finish = "length"
    _Handler.refuse_continue = False
    return OpenAIChatProvider(kind="local", base_url=fake_server + "/v1", api_key=None, model="qwen3.6-27b")


def test_budget_hit_closes_the_thinking_and_continues_with_the_answer(provider):
    role = Role("builder", provider, system_prompt="sys", thinking_budget=512, max_tokens=4096, temperature=0.8)
    r = role.attempt("make a rule", seed=3)
    think_req, answer_req = _Handler.requests[-2], _Handler.requests[-1]
    assert think_req["max_tokens"] == 512 and think_req["stop"] == ["</think>"] and think_req["chat_template_kwargs"]["enable_thinking"] is True
    assert answer_req["continue_final_message"] is True and answer_req["add_generation_prompt"] is False and answer_req["max_tokens"] == 4096
    last = answer_req["messages"][-1]
    assert last["role"] == "assistant" and last["content"].startswith("<think>\n") and last["content"].endswith("</think>\n\n")
    assert Role.THINK_CLOSE.strip() in last["content"] and "<think>" not in last["content"][7:]   # one opening tag only
    c = r.completion
    assert c.text.startswith("<think>") and c.text.endswith(CODE) and extract_code(c.text) is not None
    assert c.completion_tokens == 512 + 60 and c.prompt_tokens == 500 and c.finish_reason == "stop"
    assert c.raw["thinking"] == {"budget": 512, "budget_hit": True, "thinking_tokens": 512, "closed_by": "budget", "answer_tokens": 60}
    assert [t["tool"] for t in r.tool_log] == ["chat.completions:thinking", "chat.completions:answer"]
    assert r.tool_log[0]["budget_hit"] is True and r.tool_log[0]["completion_tokens"] == 512


def test_thinking_that_stops_by_itself_is_not_closed_by_the_budget(provider):
    _Handler.thinking_finish = "stop"
    role = Role("builder", provider, system_prompt="sys", thinking_budget=1024, max_tokens=2048)
    r = role.attempt("make a rule", seed=0)
    assert r.completion.raw["thinking"]["closed_by"] == "model" and r.completion.raw["thinking"]["budget_hit"] is False
    assert Role.THINK_CLOSE.strip() not in _Handler.requests[-1]["messages"][-1]["content"]


def test_a_server_that_cannot_continue_leaves_the_thinking_alone(provider):
    _Handler.refuse_continue = True
    role = Role("builder", provider, system_prompt="sys", thinking_budget=256, max_tokens=2048)
    r = role.attempt("make a rule", seed=0)
    c = r.completion
    assert c.text.startswith("<think>") and extract_code(c.text) is None
    assert "answer_error" in c.raw["thinking"] and r.tool_log[-1]["tool"] == "chat.completions:answer" and "error" in r.tool_log[-1]


def test_budget_zero_is_one_request_with_thinking_off(provider):
    role = Role("builder", provider, system_prompt="sys", thinking_budget=0, max_tokens=4096)
    role.attempt("make a rule", seed=0)
    body = _Handler.requests[-1]
    assert body["chat_template_kwargs"]["enable_thinking"] is False and "stop" not in body and body["max_tokens"] == 4096


def test_group_with_a_thinking_budget_records_both_calls(provider, tmp_path):
    cfg = lab_config(env={"LAB_ROOT": str(tmp_path)})
    led = Ledger(tmp_path / "ledger.sqlite")
    s = run_group(target_id="quad-d3", attempts=2, seed=0, provider_name="local", role_name="builder", ledger=led,
                  ledger_path=led.path, gpu="A100", precision="bf16", units_per_hour=None, temperature=None, max_tokens=None,
                  timeout_s=30, cfg=cfg, quiet=True, cache_dir=tmp_path / "cache", provider=provider, thinking_budget=512, concurrency=2)
    assert s["passed"] == 2 and s["thinking_budget"] == 512 and s["max_tokens"] == 4096 and s["capped"] == 0
    rows = led.attempts(group_id=s["group_id"])
    for r in rows:
        assert r.context["thinking_budget"] == 512 and r.context["max_tokens"] == 4096
        assert r.meta["thinking"]["budget_hit"] is True and r.meta["finish_reason"] == "stop"
        assert [t["tool"] for t in r.tool_log] == ["chat.completions:thinking", "chat.completions:answer", "judge"]
        assert r.meta["completion_tokens"] == 572
