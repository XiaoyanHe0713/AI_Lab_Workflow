"""The episode runner (Work package 4): one episode is a role's attempt at a target, its tool rounds, the program
that scores it, and the ledger row that records it with its cost.

A role runner (lab/roles/builder.py, theorist.py, breaker.py) turns a seed into an `Outcome`: the model's
output after its fixed number of tool rounds, the tool log, every completion the attempt made, the certificate the
program wrote, and what the role adds to the row's context and meta. `run_episodes` runs a group of such attempts
as waves of parallel requests (the served model batches them), prices each attempt by its share of the wave's wall
time (lab/demo.py explains the attribution), writes one ledger row per attempt and returns the group's summary:
pass@k valid and optimal, the pass rate, the best attempt, the cost, and the verify-first check.

The row carries everything Work package 4 lists: role and adapter version, target, prompt hash (of the opening
prompt), context (lesson ids, skill ids, reading-list snapshot, citation ids, the proposal or candidate the episode
was derived from), output, tool log, certificate, the diagnosis when it failed (the certifier's reason and the
names of the kept detectors that fire), cost in GPU seconds, CPU seconds and dollars, seed, timestamp, provider,
mode and API dollars. Limits per attempt: the role's context and token caps, the thinking budget from the ladder,
the sandbox's 120 s wall-clock cap and the fixed number of tool calls, all recorded in the context.
"""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import __version__
from .certifiers.base import Certificate
from .cost import CostMeter
from .judge import SandboxResult
from .ledger import Attempt, Ledger, new_id, summarize
from .roles import Completion, FRONTIER_KINDS, Provider, Role


@dataclass
class Outcome:
    """What a role's attempt produced, before pricing and the ledger row."""
    seed: int
    index: int
    certificate: Certificate
    output: str | None = None
    tool_log: list = field(default_factory=list)
    completions: list = field(default_factory=list)      # every model call of the attempt, in order
    prompt_hash: str = ""
    call_error: str | None = None
    code: str | None = None
    sandbox: SandboxResult | None = None
    judge_cpu: float = 0.0
    check: dict | None = None
    context: dict = field(default_factory=dict)          # role-specific context fields
    meta: dict = field(default_factory=dict)             # role-specific meta fields
    wall_s: float = 0.0
    round_outputs: list = field(default_factory=list)

    @property
    def latency_s(self) -> float:
        return sum(c.latency_s for c in self.completions)

    @property
    def prompt_tokens(self) -> int:
        return sum(c.prompt_tokens for c in self.completions)

    @property
    def completion_tokens(self) -> int:
        return sum(c.completion_tokens for c in self.completions)

    @property
    def finish_reason(self) -> str | None:
        return self.completions[-1].finish_reason if self.completions else None

    @property
    def thinking(self) -> dict | None:
        return (self.completions[-1].raw or {}).get("thinking") if self.completions else None


def diagnosis_text(cert: Certificate, call_error: str | None) -> str | None:
    """The reason an attempt failed, from the certificate (and the call error when the model never answered)."""
    diagnosis = cert.error or call_error
    if diagnosis is None and not cert.passed:
        d = cert.details
        if d.get("max_monomial_error") is not None:
            diagnosis = f"not exact: max monomial error {d.get('max_monomial_error'):.3e} on {d.get('worst_monomial')}"
        elif d.get("checks"):
            failed = [k for k, v in d["checks"].items() if not v]
            diagnosis = "failed checks: " + ", ".join(failed) if failed else "failed certificate"
        elif d.get("reproduces") is False:
            diagnosis = "the witness does not reproduce"
        elif d.get("lemmas") is not None:
            diagnosis = f"no lemma check passed ({d.get('failed', 0)} failed, {d.get('rejected', 0)} rejected)"
        else:
            diagnosis = "failed certificate"
    return diagnosis


def wave_gpu_seconds(latencies: list[float]) -> list[float]:
    """GPU seconds of the attempts of one wave: the wave's wall time (its longest latency) shared in proportion
    to each attempt's latency, so the sum is the wall time. A single request keeps its own latency."""
    total = sum(latencies)
    if total <= 0:
        return [0.0 for _ in latencies]
    wall = max(latencies)
    return [wall * lat / total for lat in latencies]


def detector_diagnosis(ledger: Ledger, attempt: Attempt) -> list[str]:
    """The names of the kept detectors (checker harness) that fire on a failed attempt; [] without a detectors table."""
    try:
        rows = ledger.rows("detectors", "target = ? AND kept = 1", (attempt.target,), json_columns=("spec",))
    except Exception:
        return []
    if not rows:
        return []
    from .harness.checker import attempt_view, evaluate_detector

    v = attempt_view(attempt)
    out = []
    for d in rows:
        try:
            if evaluate_detector(d["spec"], v):
                out.append(d["name"])
        except Exception:
            pass
    return out


def run_episodes(attempt_fn: Callable[[int, int], Outcome], *, role_name: str, role: Role, provider: Provider, provider_name: str,
                 target, attempts: int, seed: int, ledger: Ledger, meter: CostMeter, model: str, model_version: str,
                 precision: str, gpu: str, concurrency: int = 1, context_base: dict | None = None, group_id: str | None = None,
                 stop_at_dollars: float | None = None, quiet: bool = True, label: str | None = None,
                 tool_calls: int = 0, best_known: float | None = None, prompt_override: bool = False,
                 context_extra: dict | None = None, post_check_used: bool = False) -> dict:
    """A group of attempts of one role on one target: waves of `concurrency` parallel attempts, each priced by its
    share of the wave's wall time, one ledger row each, and the group's summary. `attempt_fn(index, seed)` is the
    role runner's attempt."""
    from .demo import budget_status, print_summary  # the demo's printing; no cycle at import time

    group_id = group_id or new_id("grp")
    t_group = time.perf_counter()
    rows: list[Attempt] = []
    base = {"target": target.id, "scoreboard": target.scoreboard_context(), "lesson_ids": [], "skill_ids": [],
            "reading_list_snapshot": None, "citation_ids": [], "provider": provider_name, "mode": provider.mode,
            "adapter": role.adapter, "adapter_version": (role.adapter or "none") + ":frozen-base", "temperature": role.temperature,
            "max_tokens": role.max_tokens, "thinking_budget": role.thinking_budget, "tool_calls": int(tool_calls),
            "lab_version": __version__, "prompt_override": bool(prompt_override), "role": role_name,
            "transfer": target.transfer.to_dict() if target.transfer else None}
    if context_base:
        base.update(context_base)
    if context_extra:
        base.update(context_extra)

    def one(i: int) -> Outcome:
        s = seed + i
        t0 = time.perf_counter()
        try:
            out = attempt_fn(i, s)
        except Exception as e:  # the endpoint failed or the budget refused: still a row, with the error as the diagnosis
            out = Outcome(seed=s, index=i, certificate=Certificate(target.certifier_name, "", target.id, False,
                                                                   error=f"{type(e).__name__}: {e}", seed=s),
                          call_error=f"{type(e).__name__}: {e}")
        out.wall_s = time.perf_counter() - t0
        return out

    def record(a: Outcome, gpu_seconds: float) -> None:
        cert = a.certificate
        tool_log = list(a.tool_log) + [{"tool": "judge", "certifier": cert.certifier, "version": cert.version, "passed": cert.passed,
                                        "score": cert.score, "runtime_s": cert.runtime_s,
                                        "sandbox_wall_s": a.sandbox.wall_seconds if a.sandbox else None}]
        cost = meter.attempt_cost(provider_kind=provider.kind, mode=provider.mode, gpu=gpu if provider.kind == "local" else None,
                                  model=provider.model, gpu_seconds=gpu_seconds, cpu_seconds=a.judge_cpu,
                                  prompt_tokens=a.prompt_tokens, completion_tokens=a.completion_tokens)
        bs = budget_status(provider)
        context = dict(base)
        context.update(a.context)
        row = Attempt(
            role=role_name, model=model, model_version=model_version, precision=precision, target=target.id,
            prompt_hash=a.prompt_hash, context=context, output=a.output, tool_log=tool_log, certificate=cert.to_dict(),
            diagnosis=diagnosis_text(cert, a.call_error), gpu_seconds=cost.gpu_seconds, cpu_seconds=cost.cpu_seconds,
            dollars=cost.dollars, seed=a.seed, provider=provider_name, mode=provider.mode, api_dollars=cost.api_dollars,
            group_id=group_id,
            meta={"gpu": gpu, "prompt_tokens": cost.prompt_tokens, "completion_tokens": cost.completion_tokens,
                  "cost_breakdown": cost.breakdown, "code": a.code, "wall_s": a.wall_s, "latency_s": a.latency_s,
                  "gpu_attribution": "wave wall time shared by latency" if concurrency > 1 else "request latency",
                  "finish_reason": a.finish_reason, "thinking": a.thinking, "attempt_index": a.index, "concurrency": concurrency,
                  "gate_check": a.check, "rounds": len(a.completions), "round_outputs": a.round_outputs if len(a.round_outputs) > 1 else None,
                  "budget": {"cap_usd": bs["cap_usd"], "spent_usd_after": bs["spent_usd"]} if bs else None, **a.meta},
        )
        if not cert.passed:
            fired = detector_diagnosis(ledger, row)
            if fired:
                row.diagnosis = (row.diagnosis or "failed") + "; detectors: " + ", ".join(fired)
                row.meta["detectors_fired"] = fired
        ledger.add_attempt(row)
        rows.append(row)
        if not quiet:
            flag = "PASS" if cert.passed else "fail"
            opt = " optimal" if cert.optimal else ""
            sc = f"score={cert.score:g}" if cert.score is not None else "score=-"
            cap = " [cap]" if a.finish_reason == "length" else ""
            print(f"  [{a.index + 1:2d}/{attempts}] {role_name} seed={a.seed} {flag}{opt} {sc} ${cost.dollars:.4f} "
                  f"{a.completion_tokens} tok {a.latency_s:.1f}s{cap}  {row.diagnosis or ''}"[:170])

    def record_wave(results: list[Outcome]) -> None:
        results = sorted(results, key=lambda r: r.index)
        for a, g in zip(results, wave_gpu_seconds([r.latency_s for r in results])):
            record(a, g)

    stopped_at = None
    done = 0
    width = max(1, int(concurrency))
    if not quiet:
        print(f"group {group_id}: {role_name} on {target.id} ({label or target.title}); {attempts} attempts, concurrency {width}, "
              f"thinking {role.thinking_budget}, max tokens {role.max_tokens}, tool calls {tool_calls}")
    with ThreadPoolExecutor(max_workers=width) as pool:
        while done < attempts:
            wave = list(range(done, min(attempts, done + width)))
            if width == 1:
                results = [one(wave[0])]
            else:
                results = [f.result() for f in as_completed([pool.submit(one, i) for i in wave])]
            record_wave(results)
            done = wave[-1] + 1
            if stop_at_dollars is not None and sum(r.dollars for r in rows) >= stop_at_dollars:
                stopped_at = done
                break
    n = len(rows)
    ks = (1, 16, 64) if n in (16, 64) else (1, n, 16, 64)
    summary = summarize(rows, ks=ks)
    summary.update({"group_id": group_id, "target": target.id, "best_known": best_known if best_known is not None else target.best_known,
                    "model": model, "model_version": model_version, "precision": precision, "gpu": gpu, "provider": provider_name,
                    "provider_kind": provider.kind, "mode": provider.mode, "wall_s": time.perf_counter() - t_group,
                    "units": sum(r.meta.get("cost_breakdown", {}).get("units", 0.0) for r in rows),
                    "api_dollars": sum(r.api_dollars for r in rows), "concurrency": concurrency, "role": role_name,
                    "thinking_budget": role.thinking_budget, "max_tokens": role.max_tokens, "tool_calls": int(tool_calls),
                    "requested_attempts": attempts, "stopped_at": stopped_at, "stop_at_dollars": stop_at_dollars,
                    "prompt_override": bool(prompt_override), "context_extra": context_extra or {},
                    "pass_rate": (summary["passed"] / n) if n else float("nan"),
                    "optimal_rate": (summary["optimal"] / n) if n else float("nan"),
                    "capped": sum(1 for r in rows if r.meta.get("finish_reason") == "length"),
                    "no_code": sum(1 for r in rows if (r.certificate or {}).get("error", "") and "no code block" in r.certificate["error"]),
                    "request_latency_s": sum(r.meta.get("latency_s", 0.0) for r in rows),
                    "rounds": sum(r.meta.get("rounds", 1) for r in rows),
                    "ledger": str(Path(ledger.path).resolve()),
                    "verify_first": {"required": "best known value found at least once",
                                     "found": sum(1 for r in rows if r.passed and (r.certificate or {}).get("optimal") is True),
                                     "attempts": n}})
    if post_check_used:
        checks = [r.meta.get("gate_check") for r in rows]
        summary["gate_checks"] = {"applied": sum(1 for c in checks if c is not None),
                                  "passed": sum(1 for c in checks if c and c.get("ok") is True),
                                  "failed": sum(1 for c in checks if c and c.get("ok") is False)}
    best = ledger.get_attempt(summary["best_attempt_id"]) if summary["best_attempt_id"] else None
    if not quiet:
        print_summary(summary, best)
    out_dir = Path(ledger.path).resolve().parent / "groups"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{group_id}.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    return summary


def model_identity_for(provider: Provider, provider_name: str, cfg: dict, mcfg: dict, precision: str | None) -> tuple[str, str, str]:
    """(model, model_version, precision) for the rows of a group, by provider kind and mode."""
    from .demo import model_identity

    if provider.mode == "offline":
        return f"replay:{provider_name}", "replay:fixtures", precision or "replay"
    if provider.kind in FRONTIER_KINDS:
        return provider.model, f"api:{provider.model}", precision or "api"
    precision = precision or cfg["endpoint"].get("precision") or "unknown"
    model, version = model_identity(cfg, mcfg, precision)
    return model, version, precision


def judge_sandbox_cpu(sb: SandboxResult | None, cpu0: float) -> float:
    return time.process_time() - cpu0 + (sb.cpu_seconds if sb else 0.0)


class JudgeGate:
    """Bounds the sandboxes running at once (one semaphore shared by every attempt of a group)."""

    def __init__(self, n: int = 4):
        self.sem = threading.Semaphore(max(1, int(n)))

    def __enter__(self):
        self.sem.acquire()
        return self

    def __exit__(self, *exc):
        self.sem.release()


def completion_list(res) -> list[Completion]:
    return [res.completion]


def any_text(x: Any) -> str:
    return x if isinstance(x, str) else json.dumps(x, default=str)
