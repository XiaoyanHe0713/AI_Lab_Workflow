"""The demo loop: call the frozen model N times from the same prompt, judge every attempt, write the
ledger rows, print the group's scores, the best attempt, pass@1 and pass@16.

    python -m lab.demo --target quad-d7 --attempts 16                       # the local endpoint, live
    python -m lab.demo --target quad-d7 --attempts 16 --mode offline        # recorded fixtures, no key, no cost
    python -m lab.demo --target quad-d7 --attempts 4 --provider openai --mode record   # GPT, budgeted, recorded
    python -m lab.demo --target quad-d7 --attempts 16 --concurrency 16     # the group as parallel requests

The verify-first check for the quadrature family: the 4-node Gauss rule (the scoreboard's best known
value for degree 7) must be found at least once in a recorded number of attempts at a recorded cost.
Nothing here scores anything: the judge's certificate is the score, the cost meter prices the attempt,
and both go into the ledger with the prompt hash, the seed, the model version, the provider and the mode.

Concurrency: a group of attempts is sent as `concurrency` parallel requests (the served model batches
them; `--max-num-seqs 64` on the server bounds the number in flight), each judged as it returns, while
the ledger is written from one thread. The same seeds give the same rows whatever the concurrency.

GPU time under concurrency: requests in flight together share the GPU, so an attempt's own latency is
not its GPU time. A wave of requests costs the GPU its wall time (the longest latency in the wave), and
that time is attributed to the wave's attempts in proportion to their latencies; the sum over a group is
the group's GPU wall time. Sequential runs are the one-request wave, where the two coincide.

Frontier providers (openai, anthropic) are capped at the monthly budget in config/budget.yaml ($200):
the guard warns once per session above 80 percent and refuses calls at the cap. Raise it by editing the
file; every row records the cap that was in force.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from . import __version__
from .config import lab_config, model_config, repo_root
from .cost import CostMeter
from .judge import Judge
from .ledger import Attempt, Ledger, new_id, summarize
from .roles import FRONTIER_KINDS, KINDS, MODES, extract_code, make_provider, make_role
from .targets import SYSTEM_PROMPT, get_target, list_targets


def model_identity(cfg: dict, mcfg: dict, precision: str) -> tuple[str, str]:
    base = mcfg.get("frozen_base", {})
    model = str(base.get("hf_repo") or cfg["endpoint"]["model"])
    if precision in ("int4", "awq", "gptq", "bnb-nf4"):
        q = mcfg.get("quantized", {})
        version = f"{q.get('method') or 'int4'}:{q.get('checkpoint_hash') or 'unpinned'} from {base.get('checkpoint_hash') or 'unpinned'}"
    else:
        version = f"bf16:{base.get('checkpoint_hash') or 'unpinned'}"
    return model, version


def budget_status(provider) -> dict | None:
    """The budget guard's status for a (possibly wrapped) frontier provider, or None."""
    p = provider
    while p is not None:
        guard = getattr(p, "guard", None)
        if guard is not None:
            return guard.status()
        p = getattr(p, "inner", None)
    return None


def _diagnosis(cert, call_error) -> str | None:
    diagnosis = cert.error or call_error
    if diagnosis is None and not cert.passed:
        d = cert.details
        if d.get("max_monomial_error") is not None:
            diagnosis = f"not exact: max monomial error {d.get('max_monomial_error'):.3e} on {d.get('worst_monomial')}"
        elif d.get("checks"):
            failed = [k for k, v in d["checks"].items() if not v]
            diagnosis = "failed checks: " + ", ".join(failed) if failed else "failed certificate"
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


def run_group(*, target_id: str, attempts: int, seed: int, provider_name: str | None, role_name: str,
              ledger_path: str | Path, gpu: str | None, precision: str | None, units_per_hour: float | None,
              temperature: float | None, max_tokens: int | None, timeout_s: float | None, cfg: dict | None = None,
              fixtures_root: str | Path | None = None, quiet: bool = False, cache_dir: str | Path | None = None,
              mode: str | None = None, concurrency: int = 1, judge_concurrency: int = 4, prompt: str | None = None,
              context_extra: dict | None = None, post_check: Callable | None = None, stop_at_dollars: float | None = None,
              thinking_budget: int | None = None, provider=None, ledger: Ledger | None = None,
              group_id: str | None = None) -> dict:
    """One group of attempts on a target. `prompt` overrides the target's prompt (the ablation arms add lessons),
    `context_extra` is merged into every row's context, `post_check(result_json, row_meta) -> dict | None` runs the
    gate's check on a successful sandbox result and is recorded in meta (never scored), `stop_at_dollars` ends the
    group once the cumulative cost reaches that budget (matched-dollar arms), `thinking_budget` is the thinking
    ladder's rung (0 = off; tokens of reasoning before the answer, see Role.attempt)."""
    cfg = cfg or lab_config()
    mcfg = model_config()
    target = get_target(target_id)
    own_ledger = ledger is None
    ledger = ledger or Ledger(ledger_path)
    pname = provider_name or cfg["roles"][role_name]["provider"]
    provider = provider or make_provider(pname, cfg, mode=mode, fixtures_root=fixtures_root or repo_root(), ledger=ledger)
    role = make_role(role_name, cfg, provider=provider, system_prompt=SYSTEM_PROMPT)
    if temperature is not None:
        role.temperature = temperature
    if max_tokens is not None:
        role.max_tokens = max_tokens
    if thinking_budget is not None:
        role.thinking_budget = int(thinking_budget)
    meter = CostMeter()
    gpu = gpu or cfg["endpoint"].get("gpu") or "unknown"
    if provider.mode == "offline":
        precision = precision or "replay"
        model, model_version = f"replay:{pname}", "replay:fixtures"
    elif provider.kind in FRONTIER_KINDS:
        precision = precision or "api"
        model, model_version = provider.model, f"api:{provider.model}"
    else:
        precision = precision or cfg["endpoint"].get("precision") or "unknown"
        model, model_version = model_identity(cfg, mcfg, precision)
    if units_per_hour is not None:
        meter.set_units_per_hour(gpu, units_per_hour)
    jcfg = cfg.get("judge", {})
    judge = Judge(timeout_s=timeout_s or float(jcfg.get("timeout_s", 120)), memory_mb=int(jcfg.get("memory_mb", 2048)),
                  cache_dir=cache_dir or cfg["storage"].get("cache"), sdp_cap_s=float(jcfg.get("sdp_cap_s", 60)),
                  sdp_solver=str(jcfg.get("sdp_solver", "SCS")))
    group_id = group_id or new_id("grp")
    user_prompt = prompt if prompt is not None else target.prompt
    t_group = time.perf_counter()
    rows: list[Attempt] = []
    if not quiet:
        print(f"target {target.id}: {target.title}")
        print(f"  best known: {target.best_known} ({target.status}); provider {pname} ({provider.kind}, {provider.mode}); "
              f"model {model} [{precision}]; group {group_id}; concurrency {concurrency}; thinking {role.thinking_budget}; "
              f"max tokens {role.max_tokens}")
        bs = budget_status(provider)
        if bs is not None:
            print(f"  frontier budget: ${bs['spent_usd']:.2f} of ${bs['cap_usd']:.2f} spent in {bs['month']} "
                  f"(cap in {bs['budget_file']}; offline replays never count)")
    judge_gate = threading.Semaphore(max(1, int(judge_concurrency)))

    def one_attempt(i: int) -> dict:
        """The model call and the judge for attempt i; pricing and the ledger row happen on the main thread."""
        s = seed + i
        t0 = time.perf_counter()
        try:
            res = role.attempt(user_prompt, seed=s)
            output, tool_log, comp = res.completion.text, res.tool_log, res.completion
            phash = res.prompt_hash
            call_error = None
        except Exception as e:  # the endpoint failed or the budget refused: still a row, with the error as the diagnosis
            output, tool_log, comp, phash, call_error = None, [], None, "", f"{type(e).__name__}: {e}"
        code = extract_code(output) if output else None
        with judge_gate:
            cpu0 = time.process_time()
            cert, sb = judge.judge(code, target, seed=s)
            judge_cpu = time.process_time() - cpu0 + (sb.cpu_seconds if sb else 0.0)
        check = None
        if post_check is not None and sb is not None and sb.ok:
            try:
                check = post_check(sb.result, {"seed": s, "target": target.id})
            except Exception as e:  # a failing check is information, never a crash
                check = {"ok": None, "error": f"{type(e).__name__}: {e}"}
        return {"i": i, "seed": s, "output": output, "tool_log": list(tool_log), "comp": comp, "phash": phash,
                "call_error": call_error, "code": code, "cert": cert, "sb": sb, "judge_cpu": judge_cpu, "check": check,
                "wall_s": time.perf_counter() - t0, "latency_s": comp.latency_s if comp else 0.0}

    def record(a: dict, gpu_seconds: float) -> None:
        """Price the attempt with its attributed GPU seconds and write the row."""
        comp, cert, sb, i, s = a["comp"], a["cert"], a["sb"], a["i"], a["seed"]
        tool_log = a["tool_log"] + [{"tool": "judge", "certifier": cert.certifier, "version": cert.version,
                                     "passed": cert.passed, "score": cert.score, "runtime_s": cert.runtime_s,
                                     "sandbox_wall_s": sb.wall_seconds if sb else None}]
        cost = meter.attempt_cost(provider_kind=provider.kind, mode=provider.mode,
                                  gpu=gpu if provider.kind == "local" else None, model=provider.model,
                                  gpu_seconds=gpu_seconds, cpu_seconds=a["judge_cpu"],
                                  prompt_tokens=comp.prompt_tokens if comp else 0,
                                  completion_tokens=comp.completion_tokens if comp else 0)
        diagnosis = _diagnosis(cert, a["call_error"])
        bs = budget_status(provider)
        context = {"target": target.id, "scoreboard": target.scoreboard_context(), "lesson_ids": [], "skill_ids": [],
                   "reading_list_snapshot": None, "provider": pname, "mode": provider.mode, "adapter": role.adapter,
                   "temperature": role.temperature, "max_tokens": role.max_tokens,
                   "thinking_budget": role.thinking_budget, "lab_version": __version__, "prompt_override": prompt is not None}
        if context_extra:
            context.update(context_extra)
        thinking = (comp.raw or {}).get("thinking") if comp else None
        row = Attempt(
            role=role_name, model=model, model_version=model_version, precision=precision, target=target.id,
            prompt_hash=a["phash"], context=context,
            output=a["output"], tool_log=tool_log, certificate=cert.to_dict(), diagnosis=diagnosis,
            gpu_seconds=cost.gpu_seconds, cpu_seconds=cost.cpu_seconds, dollars=cost.dollars, seed=s,
            provider=pname, mode=provider.mode, api_dollars=cost.api_dollars, group_id=group_id,
            meta={"gpu": gpu, "prompt_tokens": cost.prompt_tokens, "completion_tokens": cost.completion_tokens,
                  "cost_breakdown": cost.breakdown, "code": a["code"], "wall_s": a["wall_s"], "latency_s": a["latency_s"],
                  "gpu_attribution": "wave wall time shared by latency" if concurrency > 1 else "request latency",
                  "finish_reason": comp.finish_reason if comp else None, "thinking": thinking,
                  "attempt_index": i, "concurrency": concurrency, "gate_check": a["check"],
                  "budget": {"cap_usd": bs["cap_usd"], "spent_usd_after": bs["spent_usd"]} if bs else None},
        )
        ledger.add_attempt(row)
        rows.append(row)
        if not quiet:
            flag = "PASS" if cert.passed else "fail"
            opt = " optimal" if cert.optimal else ""
            sc = f"score={cert.score:g}" if cert.score is not None else "score=-"
            cap = " [cap]" if comp and comp.finish_reason == "length" else ""
            print(f"  [{i + 1:2d}/{attempts}] seed={s} {flag}{opt} {sc} ${cost.dollars:.4f} "
                  f"{(comp.completion_tokens if comp else 0)} tok {(comp.latency_s if comp else 0):.1f}s{cap}  {diagnosis or ''}"[:170])

    def record_wave(results: list[dict]) -> None:
        results = sorted(results, key=lambda r: r["i"])
        for a, g in zip(results, wave_gpu_seconds([r["latency_s"] for r in results])):
            record(a, g)

    stopped_at = None
    done = 0
    width = max(1, int(concurrency))
    with ThreadPoolExecutor(max_workers=width) as pool:
        while done < attempts:
            wave = list(range(done, min(attempts, done + width)))
            if width == 1:
                results = [one_attempt(wave[0])]
            else:
                results = [f.result() for f in as_completed([pool.submit(one_attempt, i) for i in wave])]
            record_wave(results)
            done = wave[-1] + 1
            if stop_at_dollars is not None and sum(r.dollars for r in rows) >= stop_at_dollars:
                stopped_at = done
                break
    n = len(rows)
    ks = (1, 16, 64) if n in (16, 64) else (1, n, 16, 64)
    summary = summarize(rows, ks=ks)
    summary.update({"group_id": group_id, "target": target.id, "best_known": target.best_known, "model": model,
                    "model_version": model_version, "precision": precision, "gpu": gpu, "provider": pname,
                    "provider_kind": provider.kind, "mode": provider.mode, "wall_s": time.perf_counter() - t_group,
                    "units": sum(r.meta.get("cost_breakdown", {}).get("units", 0.0) for r in rows),
                    "api_dollars": sum(r.api_dollars for r in rows), "concurrency": concurrency,
                    "thinking_budget": role.thinking_budget, "max_tokens": role.max_tokens, "requested_attempts": attempts,
                    "stopped_at": stopped_at, "stop_at_dollars": stop_at_dollars, "prompt_override": prompt is not None,
                    "context_extra": context_extra or {}, "pass_rate": (summary["passed"] / n) if n else float("nan"),
                    "optimal_rate": (summary["optimal"] / n) if n else float("nan"),
                    "capped": sum(1 for r in rows if r.meta.get("finish_reason") == "length"),
                    "no_code": sum(1 for r in rows if (r.certificate or {}).get("error", "") and "no code block" in r.certificate["error"]),
                    "request_latency_s": sum(r.meta.get("latency_s", 0.0) for r in rows),
                    "ledger": str(Path(ledger.path).resolve()),
                    "verify_first": {"required": "best known value found at least once",
                                     "found": summary_optimal(rows), "attempts": n}})
    if post_check is not None:
        checks = [r.meta.get("gate_check") for r in rows]
        summary["gate_checks"] = {"applied": sum(1 for c in checks if c is not None),
                                  "passed": sum(1 for c in checks if c and c.get("ok") is True),
                                  "failed": sum(1 for c in checks if c and c.get("ok") is False)}
    best = ledger.get_attempt(summary["best_attempt_id"]) if summary["best_attempt_id"] else None
    if own_ledger:
        ledger.close()
    if not quiet:
        print_summary(summary, best)
    out_dir = Path(ledger_path).resolve().parent / "groups"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{group_id}.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    return summary


def summary_optimal(rows: list[Attempt]) -> int:
    return sum(1 for r in rows if r.passed and (r.certificate or {}).get("optimal") is True)


def print_summary(summary: dict, best: Attempt | None) -> None:
    print("\nscores:", [f"{s:g}" for s in summary["scores"]] or "none passed")
    print(f"passed {summary['passed']}/{summary['attempts']} valid; {summary['optimal']} matched the best known value "
          f"({summary['best_known']}); {summary['capped']} attempts ended at the token cap, {summary['no_code']} without a code block")
    pk, ok = summary["pass_at_k"], summary["optimal_at_k"]
    print("pass@k (valid):   " + "  ".join(f"{k}={v:.3f}" for k, v in pk.items()))
    print("pass@k (optimal): " + "  ".join(f"{k}={v:.3f}" for k, v in ok.items()))
    print(f"cost: ${summary['dollars']:.4f} (api ${summary['api_dollars']:.4f})  gpu {summary['gpu_seconds']:.1f}s "
          f"(request latencies summed {summary['request_latency_s']:.1f}s)  cpu {summary['cpu_seconds']:.1f}s  "
          f"units {summary['units']:.3f}  wall {summary['wall_s']:.1f}s  "
          f"[{summary['model']} {summary['precision']} on {summary['gpu']}; {summary['provider']} {summary['mode']}; "
          f"concurrency {summary.get('concurrency', 1)}; thinking {summary.get('thinking_budget', 0)}]")
    vf = summary["verify_first"]
    status = "PASS" if vf["found"] > 0 else "FAIL"
    print(f"verify-first: {status}: best known value found {vf['found']} time(s) in {vf['attempts']} attempts at ${summary['dollars']:.4f}")
    if summary.get("gate_checks"):
        g = summary["gate_checks"]
        print(f"gate check: applied {g['applied']}, passed {g['passed']}, failed {g['failed']} (logged, never scored)")
    if best is not None:
        print(f"\nbest attempt {best.id} (seed {best.seed}, score {best.score:g}):")
        print(best.meta.get("code") or best.output or "")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m lab.demo", description=__doc__.split("\n\n")[0],
                                 epilog="Frontier providers are capped by config/budget.yaml ($200 a month, warning at 80%%); "
                                        "edit that file by hand to raise the cap. Offline mode replays recorded fixtures and costs nothing.")
    ap.add_argument("--target", default="quad-d7", help="target id, e.g. quad-d7 (see --list)")
    ap.add_argument("--attempts", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--provider", default=None, choices=list(KINDS),
                    help="local | openai | anthropic (default: the role's provider in config/lab.yaml)")
    ap.add_argument("--mode", default=None, choices=list(MODES),
                    help="offline (recorded fixtures) | online | record (default: the provider's mode in config/lab.yaml)")
    ap.add_argument("--role", default="builder")
    ap.add_argument("--ledger", default=None, help="SQLite path (default: storage.ledger in config/lab.yaml)")
    ap.add_argument("--gpu", default=None, help="GPU tier for pricing: H100 | G4 | A100 | L4 (default: $LAB_GPU)")
    ap.add_argument("--precision", default=None, help="bf16 | int4 (default: $LAB_PRECISION; 'replay' offline, 'api' for frontier)")
    ap.add_argument("--units-per-hour", type=float, default=None, help="override the compute-unit rate for this GPU")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--max-tokens", type=int, default=None, help="answer tokens per attempt (default: the role's, 4096)")
    ap.add_argument("--thinking", type=int, default=None, help="thinking budget in tokens (0 = off): reasoning is capped at the budget, then the answer follows")
    ap.add_argument("--timeout", type=float, default=None, help="sandbox wall-clock cap in seconds")
    ap.add_argument("--concurrency", type=int, default=1, help="parallel requests per group (16 for a served model)")
    ap.add_argument("--judge-concurrency", type=int, default=4, help="sandboxes running at once")
    ap.add_argument("--fixtures", default=None, help="root for recorded fixtures (default: the repository)")
    ap.add_argument("--list", action="store_true", help="list target ids and exit")
    ap.add_argument("--json", action="store_true", help="print the summary as JSON")
    args = ap.parse_args(argv)
    if args.list:
        print("\n".join(list_targets()))
        return 0
    cfg = lab_config()
    ledger_path = args.ledger or cfg["storage"]["ledger"]
    summary = run_group(target_id=args.target, attempts=args.attempts, seed=args.seed, provider_name=args.provider,
                        role_name=args.role, ledger_path=ledger_path, gpu=args.gpu, precision=args.precision,
                        units_per_hour=args.units_per_hour, temperature=args.temperature, max_tokens=args.max_tokens,
                        timeout_s=args.timeout, cfg=cfg, fixtures_root=args.fixtures, quiet=args.json, mode=args.mode,
                        concurrency=args.concurrency, judge_concurrency=args.judge_concurrency, thinking_budget=args.thinking)
    if args.json:
        print(json.dumps(summary, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
