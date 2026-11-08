"""The harness base: one place where a harness's language parts call a provider.

Every call goes through `make_provider` (so an online or record call to GPT or Claude passes the budget
guard and is priced by the meter) and is written to the ledger as an attempt row with role
"harness:<name>", the provider, the mode and the API dollars, so the monthly frontier spend is tracked
from the same table as everything else. Offline replays cost nothing and are rows too, marked offline.

Prompts must be deterministic functions of their inputs (no attempt ids, timestamps or paths), because
the recorded fixture is keyed by the request: a harness test records once online and replays forever.
A harness returns text and, when asked, the JSON it parsed from it; what it writes is information for a
person or for the next prompt, never a score.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import lab_config, repo_root
from ..cost import BudgetExceeded, CostMeter
from ..ledger import Attempt, Ledger, prompt_hash
from ..roles import FRONTIER_KINDS, Provider, make_provider

_FENCE = re.compile(r"```(?:json)?[ \t]*\r?\n(.*?)```", re.S)


def parse_json(text: str) -> Any:
    """The JSON object or array in a reply: a fenced block, the whole text, the outermost braces, the first balanced
    object in the text (a reply with prose after its JSON, or a brace in the prose before it), and only then an array
    the same way: a reply holding an object is that object, whatever arrays it contains."""
    if text is None:
        raise ValueError("empty reply")
    m = _FENCE.search(text)
    candidates = [m.group(1)] if m else []
    candidates.append(text)
    last = None

    def attempt(cands):
        nonlocal last
        for c in cands:
            try:
                return True, json.loads(c.strip())
            except ValueError as e:
                last = e
        return False, None

    def scan(opener):
        decoder = json.JSONDecoder()
        for pos, ch in enumerate(text):
            if ch == opener:
                try:
                    return True, decoder.raw_decode(text, pos)[0]
                except ValueError:
                    continue
        return False, None

    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        slices = [text[start:end + 1]] if start != -1 and end > start else []
        ok, value = attempt(candidates + slices)
        if ok:
            return value
        candidates = []
        ok, value = scan(opener)
        if ok:
            return value
    raise ValueError(f"no JSON in the reply: {last}")


@dataclass
class HarnessReply:
    text: str
    data: Any
    attempt_id: str
    provider: str
    mode: str
    model: str
    api_dollars: float
    prompt_tokens: int
    completion_tokens: int
    parse_error: str | None = None
    raw: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.parse_error is None


def harness_provider(cfg: dict | None = None, *, name: str | None = None, mode: str | None = None, ledger: Ledger | None = None,
                     fixtures_root: str | Path | None = None) -> Provider:
    """The provider for a harness's language parts: config harness.provider (GPT first) unless named."""
    cfg = cfg or lab_config()
    name = name or cfg.get("harness", {}).get("provider", "openai")
    return make_provider(name, cfg, mode=mode, fixtures_root=fixtures_root or repo_root(), ledger=ledger)


class Harness:
    """Subclasses set `name` and call `self.ask(...)` for their language parts."""

    name = "harness"
    system_prompt = ("You are a careful numerical analyst assisting an automated lab. Answer with the JSON the task "
                     "asks for and nothing else. Never invent documents, results or attempt labels that were not given.")

    def __init__(self, provider: Provider, ledger: Ledger, *, cfg: dict | None = None, meter: CostMeter | None = None,
                 temperature: float | None = None, max_tokens: int | None = None, seed: int | None = None):
        self.provider = provider
        self.ledger = ledger
        self.cfg = cfg or lab_config()
        self.meter = meter or CostMeter()
        self.temperature = temperature          # None: the provider's default (GPT-5 takes no other value)
        self.max_tokens = int(max_tokens or self.cfg.get("harness", {}).get("max_tokens", 4096))
        self.seed = seed
        self.calls: list[HarnessReply] = []

    # --- the one way a harness talks to a model ---------------------------------------------------
    def ask(self, task: str, user: str, *, system: str | None = None, target: str = "", context: dict | None = None,
            json_out: bool = True, max_tokens: int | None = None) -> HarnessReply:
        messages = [{"role": "system", "content": system or self.system_prompt}, {"role": "user", "content": user}]
        extra = {"response_format": {"type": "json_object"}} if (json_out and self.provider.kind == "openai") else None
        t0 = time.perf_counter()
        error = None
        comp = None
        try:
            comp = self._complete(messages, extra, max_tokens or self.max_tokens)
        except BudgetExceeded as e:
            error = f"BudgetExceeded: {e}"
        except Exception as e:  # the call failed: still a row
            error = f"{type(e).__name__}: {e}"
        api_dollars = float(comp.api_dollars) if comp else 0.0
        cost = self.meter.attempt_cost(provider_kind=self.provider.kind, mode=self.provider.mode, model=self.provider.model,
                                       prompt_tokens=comp.prompt_tokens if comp else 0,
                                       completion_tokens=comp.completion_tokens if comp else 0) \
            if comp and self.provider.kind in FRONTIER_KINDS else None
        dollars = cost.dollars if cost else 0.0
        if comp and comp.mode != "offline" and self.provider.kind in FRONTIER_KINDS:
            api_dollars = api_dollars or (cost.api_dollars if cost else 0.0)
        text = comp.text if comp else ""
        data, parse_error = None, None
        if json_out and comp:
            try:
                data = parse_json(text)
            except ValueError as e:
                parse_error = str(e)
        row = Attempt(
            role=f"harness:{self.name}", model=self.provider.model, model_version=f"api:{self.provider.model}",
            precision="replay" if self.provider.mode == "offline" else "api", target=target or f"harness:{self.name}:{task}",
            prompt_hash=prompt_hash(messages),
            context={"task": task, "provider": self.provider.name, "kind": self.provider.kind, "mode": self.provider.mode,
                     "json_out": json_out, **(context or {})},
            output=text or None,
            tool_log=[{"tool": "chat.completions", "provider": self.provider.name, "mode": self.provider.mode,
                       "model": self.provider.model, "latency_s": comp.latency_s if comp else None,
                       "prompt_tokens": comp.prompt_tokens if comp else 0, "completion_tokens": comp.completion_tokens if comp else 0}],
            certificate=None, diagnosis=error or parse_error, dollars=dollars, seed=self.seed,
            provider=self.provider.name, mode=self.provider.mode, api_dollars=api_dollars,
            meta={"harness": self.name, "wall_s": time.perf_counter() - t0, "fixture": (comp.raw or {}).get("fixture") if comp else None,
                  "budget": (comp.raw or {}).get("budget") if comp else None, "finish_reason": comp.finish_reason if comp else None,
                  "parsed": data is not None},
        )
        self.ledger.add_attempt(row)
        reply = HarnessReply(text=text, data=data, attempt_id=row.id, provider=self.provider.name, mode=self.provider.mode,
                             model=self.provider.model, api_dollars=api_dollars, prompt_tokens=comp.prompt_tokens if comp else 0,
                             completion_tokens=comp.completion_tokens if comp else 0, parse_error=error or parse_error,
                             raw=dict(comp.raw) if comp else {})
        self.calls.append(reply)
        if error and error.startswith("BudgetExceeded"):
            raise BudgetExceeded(error)
        return reply

    def _complete(self, messages, extra, max_tokens):
        try:
            return self.provider.complete(messages, temperature=self.temperature, max_tokens=max_tokens, seed=self.seed,
                                          extra_body=extra)
        except Exception as e:
            # an API that rejects the structured-output request still answers the plain prompt
            if extra and "response_format" in str(e):
                return self.provider.complete(messages, temperature=self.temperature, max_tokens=max_tokens, seed=self.seed)
            raise

    @property
    def spent(self) -> float:
        return sum(c.api_dollars for c in self.calls)
