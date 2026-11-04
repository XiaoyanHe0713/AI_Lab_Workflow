"""The role interface: a role is a prompt plus a provider; a provider is a kind and a mode, both chosen
by configuration.

Kinds
  local      the Qwen3.6-27B endpoint started by colab/serve.ipynb (vLLM or SGLang, OpenAI chat API);
             the provider of the three roles (theorist, builder, breaker). A role's LoRA adapter is
             addressed by its served name, so an untrained role is exactly the base.
  openai     GPT through the OpenAI API: the first frontier provider for the language parts of the
             harnesses (librarian, lesson extraction and gating, written diagnoses) from week 3 on.
  anthropic  Claude through the Anthropic API: the second frontier provider, behind the same interface.
  Nothing a frontier provider writes ever enters a reward.

Modes
  offline    replay recorded responses from tests/fixtures/providers/<provider>/, keyed by a hash of the
             request (model, messages, tools, temperature, and the seed when one is set). No network, no
             key. A missing fixture raises: a test fails instead of making a live call.
  online     live calls.
  record     live calls, each written as a fixture so the same test runs offline afterwards.
The pytest default is offline. Live tests carry the `online` marker and are skipped without a key.

Budget: every online or record call to a frontier provider passes through the budget guard
(config/budget.yaml, $200 a month): above the warning fraction it warns once per session, at the cap it
raises and makes no call. Offline replays cost nothing and never count.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..cost import BudgetGuard, CostMeter, load_budget
from ..ledger import prompt_hash

KINDS = ("local", "openai", "anthropic")
FRONTIER_KINDS = ("openai", "anthropic")
MODES = ("offline", "online", "record")


@dataclass
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    model: str
    provider: str                      # kind: local | openai | anthropic
    mode: str = "online"               # offline | online | record
    finish_reason: str | None = None
    api_dollars: float = 0.0           # priced by the budget guard for frontier calls (zero offline)
    raw: dict = field(default_factory=dict)

    @property
    def tokens_per_second(self) -> float:
        return self.completion_tokens / self.latency_s if self.latency_s > 0 else float("nan")


def request_key(model: str, messages: list[dict], tools: list | None = None, temperature: float | None = None,
                seed: int | None = None) -> str:
    """The fixture key: sha256 of the request fields that determine the response, first 24 hex digits."""
    req = {"model": model, "messages": messages, "tools": tools or [], "temperature": temperature}
    if seed is not None:
        req["seed"] = int(seed)
    blob = json.dumps(req, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:24]


class Provider:
    """Base class. `kind` prices the call (local by GPU time, frontier by tokens); `mode` says whether it is live."""

    kind: str = "local"
    name: str = "provider"
    model: str = ""
    mode: str = "online"

    def complete(self, messages: list[dict], *, temperature: float = 0.8, max_tokens: int = 2048,
                 seed: int | None = None, model: str | None = None, tools: list | None = None,
                 extra_body: dict | None = None) -> Completion:
        raise NotImplementedError


class OpenAIChatProvider(Provider):
    """The local endpoint (vLLM, SGLang) and the OpenAI API: both speak /v1/chat/completions."""

    def __init__(self, *, kind: str, base_url: str, api_key: str | None, model: str, name: str | None = None,
                 timeout_s: float = 600.0):
        from openai import OpenAI  # imported lazily so offline runs never need a client

        if kind not in ("local", "openai"):
            raise ValueError(kind)
        self.kind = kind
        self.name = name or kind
        self.base_url = base_url
        self.model = model
        self.mode = "online"
        self.client = OpenAI(base_url=base_url, api_key=api_key or "EMPTY", timeout=timeout_s, max_retries=1)

    def complete(self, messages, *, temperature=0.8, max_tokens=2048, seed=None, model=None, tools=None, extra_body=None):
        kwargs: dict[str, Any] = dict(model=model or self.model, messages=messages, temperature=temperature,
                                      max_tokens=max_tokens)
        if seed is not None:
            kwargs["seed"] = int(seed)
        if tools:
            kwargs["tools"] = tools
        if extra_body:
            kwargs["extra_body"] = extra_body
        t0 = time.perf_counter()
        resp = self.client.chat.completions.create(**kwargs)
        latency = time.perf_counter() - t0
        choice = resp.choices[0]
        usage = getattr(resp, "usage", None)
        return Completion(
            text=choice.message.content or "",
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            latency_s=latency, model=str(getattr(resp, "model", kwargs["model"])), provider=self.kind, mode="online",
            finish_reason=getattr(choice, "finish_reason", None),
            raw={"id": getattr(resp, "id", None), "created": getattr(resp, "created", None)},
        )


class AnthropicProvider(Provider):
    """Claude through the Anthropic Messages API, behind the same interface as GPT."""

    kind = "anthropic"

    def __init__(self, *, api_key: str | None, model: str, base_url: str | None = None, name: str = "anthropic",
                 timeout_s: float = 600.0):
        from anthropic import Anthropic

        self.name = name
        self.model = model
        self.mode = "online"
        self.base_url = base_url
        kw: dict[str, Any] = dict(api_key=api_key, timeout=timeout_s, max_retries=1)
        if base_url:
            kw["base_url"] = base_url
        self.client = Anthropic(**kw)

    def complete(self, messages, *, temperature=0.8, max_tokens=2048, seed=None, model=None, tools=None, extra_body=None):
        # The current Messages API takes no sampling temperature or seed (SDK 2.x dropped them); both are
        # kept in the fixture key and in `raw` so a recording still says what was asked for.
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        turns = [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"]
        kwargs: dict[str, Any] = dict(model=model or self.model, messages=turns, max_tokens=max_tokens)
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools
        if extra_body:
            kwargs["extra_body"] = extra_body
        t0 = time.perf_counter()
        resp = self.client.messages.create(**kwargs)
        latency = time.perf_counter() - t0
        text = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text")
        usage = getattr(resp, "usage", None)
        return Completion(
            text=text,
            prompt_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            latency_s=latency, model=str(getattr(resp, "model", kwargs["model"])), provider="anthropic", mode="online",
            finish_reason=getattr(resp, "stop_reason", None),
            raw={"id": getattr(resp, "id", None), "seed_requested": seed, "temperature_requested": temperature},
        )


class ReplayProvider(Provider):
    """Offline mode: recorded responses from a fixtures directory; never a live call.

    Lookup order: `<request key>.json` (one recorded call), then `<prompt hash>.seq.json` (a list of
    responses consumed in seed order, so a group of N attempts replays from one recording). Missing: raise.
    """

    mode = "offline"

    def __init__(self, fixtures_dir: str | Path, *, kind: str = "local", model: str = "replay", name: str = "replay"):
        self.dir = Path(fixtures_dir)
        self.kind = kind
        self.model = model
        self.name = name

    def complete(self, messages, *, temperature=0.8, max_tokens=2048, seed=None, model=None, tools=None, extra_body=None):
        model = model or self.model
        key = request_key(model, messages, tools, temperature, seed)
        path = self.dir / f"{key}.json"
        if path.exists():
            d = json.loads(path.read_text(encoding="utf-8"))
            d = d.get("response", d)
            source = path.name
        else:
            seq = self.dir / f"{prompt_hash(messages).split(':')[1][:24]}.seq.json"
            if not seq.exists():
                raise FileNotFoundError(
                    f"offline mode: no recorded response for provider {self.name!r} under {self.dir} "
                    f"(request key {key}, sequence {seq.name}); record it first with mode=record")
            items = json.loads(seq.read_text(encoding="utf-8"))
            if not items:
                raise FileNotFoundError(f"empty replay sequence {seq}")
            d = items[(seed or 0) % len(items)]
            source = seq.name
        return Completion(
            text=d["text"], prompt_tokens=int(d.get("prompt_tokens", 0)), completion_tokens=int(d.get("completion_tokens", 0)),
            latency_s=float(d.get("latency_s", 0.0)), model=str(d.get("model", model)), provider=self.kind, mode="offline",
            finish_reason=d.get("finish_reason"), api_dollars=0.0, raw={"fixture": source},
        )


class RecordingProvider(Provider):
    """Record mode: live calls through the wrapped provider, each written as `<request key>.json`."""

    mode = "record"

    def __init__(self, inner: Provider, fixtures_dir: str | Path):
        self.inner = inner
        self.kind = inner.kind
        self.name = inner.name
        self.model = inner.model
        self.dir = Path(fixtures_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def complete(self, messages, *, temperature=0.8, max_tokens=2048, seed=None, model=None, tools=None, extra_body=None):
        c = self.inner.complete(messages, temperature=temperature, max_tokens=max_tokens, seed=seed, model=model,
                                tools=tools, extra_body=extra_body)
        c.mode = "record"
        key = request_key(model or self.inner.model, messages, tools, temperature, seed)
        record = {"provider": self.kind, "name": self.name, "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  "request": {"model": model or self.inner.model, "messages": messages, "tools": tools or [],
                              "temperature": temperature, "seed": seed, "max_tokens": max_tokens},
                  "response": {k: v for k, v in asdict(c).items() if k != "raw"}}
        (self.dir / f"{key}.json").write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
        return c


class BudgetedProvider(Provider):
    """A frontier provider behind the budget guard: checked before every call, priced after it."""

    def __init__(self, inner: Provider, guard: BudgetGuard, meter: CostMeter | None = None):
        self.inner = inner
        self.guard = guard
        self.meter = meter or CostMeter()
        self.kind = inner.kind
        self.name = inner.name
        self.model = inner.model
        self.mode = inner.mode

    def complete(self, messages, *, temperature=0.8, max_tokens=2048, seed=None, model=None, tools=None, extra_body=None):
        status = self.guard.check(self.kind, self.mode)  # raises BudgetExceeded at the cap
        c = self.inner.complete(messages, temperature=temperature, max_tokens=max_tokens, seed=seed, model=model,
                                tools=tools, extra_body=extra_body)
        dollars = self.meter.api_cost(model or self.model, c.prompt_tokens, c.completion_tokens).dollars
        self.guard.charge(dollars)
        c.api_dollars = dollars
        c.raw["budget"] = {**status, "call_dollars": dollars}
        return c


def _fixtures_dir(cfg: dict, name: str, fixtures_root: str | Path | None) -> Path:
    d = Path(cfg.get("fixtures", "tests/fixtures/providers"))
    if not d.is_absolute():
        from ..config import repo_root

        d = Path(fixtures_root) / d if fixtures_root is not None else repo_root() / d
    return d / name


def make_provider(name: str, cfg: dict, *, mode: str | None = None, fixtures_root: str | Path | None = None,
                  ledger=None, budget=None, meter: CostMeter | None = None) -> Provider:
    """Build a provider from the `providers:` section of config/lab.yaml.

    `mode` overrides the configured mode. Offline returns a replay provider and touches no network.
    Online and record build the live client (a frontier provider needs its key in the environment and is
    wrapped by the budget guard, which reads this month's spend from `ledger` when one is given).
    """
    pcfg = cfg["providers"][name]
    kind = pcfg["kind"]
    if kind not in KINDS:
        raise ValueError(f"provider {name!r}: unknown kind {kind!r} (expected one of {KINDS})")
    mode = (mode or pcfg.get("mode") or "online").lower()
    if mode not in MODES:
        raise ValueError(f"provider {name!r}: unknown mode {mode!r} (expected one of {MODES})")
    fixtures = _fixtures_dir(cfg, name, fixtures_root)
    if mode == "offline":
        return ReplayProvider(fixtures, kind=kind, model=pcfg["model"], name=name)
    api_key = os.environ.get(pcfg["api_key_env"]) if pcfg.get("api_key_env") else None
    if kind in FRONTIER_KINDS and not api_key:
        raise RuntimeError(f"provider {name!r} ({kind}) in {mode} mode needs the API key in ${pcfg.get('api_key_env')}")
    if kind == "anthropic":
        live: Provider = AnthropicProvider(api_key=api_key, model=pcfg["model"], base_url=pcfg.get("base_url"), name=name,
                                           timeout_s=float(pcfg.get("timeout_s", 600)))
    else:
        live = OpenAIChatProvider(kind=kind, base_url=pcfg["base_url"], api_key=api_key, model=pcfg["model"], name=name,
                                  timeout_s=float(pcfg.get("timeout_s", 600)))
    if kind in FRONTIER_KINDS:
        guard = BudgetGuard(budget or load_budget(), ledger=ledger)
        live = BudgetedProvider(live, guard, meter)
    if mode == "record":
        return RecordingProvider(live, fixtures)
    return live


@dataclass
class RoleResult:
    role: str
    messages: list[dict]
    prompt_hash: str
    completion: Completion
    served_model: str
    adapter: str | None
    tool_log: list = field(default_factory=list)


class Role:
    """A role: name, system prompt, provider, optional adapter name, sampling settings."""

    def __init__(self, name: str, provider: Provider, *, system_prompt: str = "", adapter: str | None = None,
                 temperature: float = 0.8, max_tokens: int = 4096, thinking_budget: int = 0,
                 extra_body: dict | None = None):
        self.name = name
        self.provider = provider
        self.system_prompt = system_prompt
        self.adapter = adapter
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.thinking_budget = thinking_budget
        self.extra_body = dict(extra_body or {})

    @property
    def served_model(self) -> str:
        return self.adapter or self.provider.model  # a LoRA adapter is served under its own name

    def build_messages(self, user_prompt: str) -> list[dict]:
        msgs = []
        if self.system_prompt:
            msgs.append({"role": "system", "content": self.system_prompt})
        msgs.append({"role": "user", "content": user_prompt})
        return msgs

    def attempt(self, user_prompt: str, *, seed: int | None = None) -> RoleResult:
        messages = self.build_messages(user_prompt)
        extra = dict(self.extra_body)
        if self.provider.kind == "local":
            # Qwen3-family chat template: thinking off at budget 0 (vLLM/SGLang chat_template_kwargs)
            extra.setdefault("chat_template_kwargs", {})
            extra["chat_template_kwargs"].setdefault("enable_thinking", self.thinking_budget > 0)
        completion = self.provider.complete(messages, temperature=self.temperature, max_tokens=self.max_tokens,
                                            seed=seed, model=self.served_model, extra_body=extra or None)
        return RoleResult(role=self.name, messages=messages, prompt_hash=prompt_hash(messages), completion=completion,
                          served_model=self.served_model, adapter=self.adapter,
                          tool_log=[{"tool": "chat.completions", "provider": self.provider.name, "mode": completion.mode,
                                     "model": self.served_model, "seed": seed, "latency_s": completion.latency_s,
                                     "prompt_tokens": completion.prompt_tokens,
                                     "completion_tokens": completion.completion_tokens}])


def make_role(name: str, cfg: dict, provider: Provider | None = None, system_prompt: str = "",
              fixtures_root: str | Path | None = None, mode: str | None = None, ledger=None) -> Role:
    rcfg = cfg["roles"][name]
    provider = provider or make_provider(rcfg["provider"], cfg, mode=mode, fixtures_root=fixtures_root, ledger=ledger)
    return Role(name, provider, system_prompt=system_prompt, adapter=rcfg.get("adapter"),
                temperature=float(rcfg.get("temperature", 0.8)), max_tokens=int(rcfg.get("max_tokens", 4096)),
                thinking_budget=int(rcfg.get("thinking_budget", 0)), extra_body=rcfg.get("extra_body"))


_FENCE = re.compile(r"```(?:python|py)?[ \t]*\r?\n(.*?)```", re.S)


def extract_code(text: str) -> str | None:
    """The last fenced Python block in a completion, or the whole text if it parses as Python."""
    blocks = _FENCE.findall(text)
    if blocks:
        return blocks[-1].strip("\n") + "\n"
    import ast

    try:
        ast.parse(text)
        return text if text.strip() else None
    except SyntaxError:
        return None
