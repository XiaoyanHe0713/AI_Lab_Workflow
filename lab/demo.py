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
from pathlib import Path
from typing import Callable

from . import __version__
from .config import lab_config, model_config, repo_root
from .cost import CostMeter
from .episode import JudgeGate, diagnosis_text, model_identity_for, run_episodes, wave_gpu_seconds  # noqa: F401
from .judge import Judge
from .ledger import Attempt, Ledger, new_id, summarize  # noqa: F401
from .roles import FRONTIER_KINDS, KINDS, MODES, extract_code, make_provider, make_role  # noqa: F401
from .roles.builder import BuilderRunner
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
    return diagnosis_text(cert, call_error)


def prepare(*, cfg: dict, role_name: str, provider_name: str | None = None, mode: str | None = None,
            fixtures_root: str | Path | None = None, ledger: Ledger | None = None, provider=None, gpu: str | None = None,
            precision: str | None = None, units_per_hour: float | None = None, temperature: float | None = None,
            max_tokens: int | None = None, thinking_budget: int | None = None, timeout_s: float | None = None,
            cache_dir: str | Path | None = None, system_prompt: str | None = None) -> dict:
    """Everything a group of episodes shares: the provider, the role with its sampling settings, the judge, the cost
    meter, the model identity and precision of the rows. Used by run_group, the gate, the ablation and the loop."""
    mcfg = model_config()
    pname = provider_name or cfg["roles"][role_name]["provider"]
    provider = provider or make_provider(pname, cfg, mode=mode, fixtures_root=fixtures_root or repo_root(), ledger=ledger)
    role = make_role(role_name, cfg, provider=provider, system_prompt=SYSTEM_PROMPT if system_prompt is None else system_prompt)
    if temperature is not None:
        role.temperature = temperature
    if max_tokens is not None:
        role.max_tokens = max_tokens
    if thinking_budget is not None:
        role.thinking_budget = int(thinking_budget)
    meter = CostMeter()
    gpu = gpu or cfg["endpoint"].get("gpu") or "unknown"
    model, model_version, precision = model_identity_for(provider, pname, cfg, mcfg, precision)
    if units_per_hour is not None:
        meter.set_units_per_hour(gpu, units_per_hour)
    jcfg = cfg.get("judge", {})
    judge = Judge(timeout_s=timeout_s or float(jcfg.get("timeout_s", 120)), memory_mb=int(jcfg.get("memory_mb", 2048)),
                  cache_dir=cache_dir or cfg["storage"].get("cache"), sdp_cap_s=float(jcfg.get("sdp_cap_s", 60)),
                  sdp_solver=str(jcfg.get("sdp_solver", "SCS")))
    return {"provider": provider, "provider_name": pname, "role": role, "judge": judge, "meter": meter, "model": model,
            "model_version": model_version, "precision": precision, "gpu": gpu, "cfg": cfg}


def run_group(*, target_id: str, attempts: int, seed: int, provider_name: str | None, role_name: str,
              ledger_path: str | Path, gpu: str | None, precision: str | None, units_per_hour: float | None,
              temperature: float | None, max_tokens: int | None, timeout_s: float | None, cfg: dict | None = None,
              fixtures_root: str | Path | None = None, quiet: bool = False, cache_dir: str | Path | None = None,
              mode: str | None = None, concurrency: int = 1, judge_concurrency: int = 4, prompt: str | None = None,
              context_extra: dict | None = None, post_check: Callable | None = None, stop_at_dollars: float | None = None,
              thinking_budget: int | None = None, provider=None, ledger: Ledger | None = None,
              group_id: str | None = None, tool_calls: int = 0, proposal_for: Callable | None = None) -> dict:
    """One group of builder attempts on a target (the week-2 demo, the gate's groups, the ablation's arms, the loop's
    builder episodes). `prompt` overrides the target's prompt (the ablation arms add lessons), `context_extra` is
    merged into every row's context, `post_check(result_json, row_meta) -> dict | None` runs the gate's check on a
    successful sandbox result and is recorded in meta (never scored), `stop_at_dollars` ends the group once the
    cumulative cost reaches that budget (matched-dollar arms), `thinking_budget` is the thinking ladder's rung (0 = off;
    tokens of reasoning before the answer, see Role.attempt), `tool_calls` the number of certifier dry runs the builder
    gets before its final answer, `proposal_for(index)` the theorist proposal an attempt is derived from."""
    cfg = cfg or lab_config()
    target = get_target(target_id)
    own_ledger = ledger is None
    ledger = ledger or Ledger(ledger_path)
    s = prepare(cfg=cfg, role_name=role_name, provider_name=provider_name, mode=mode, fixtures_root=fixtures_root, ledger=ledger,
                provider=provider, gpu=gpu, precision=precision, units_per_hour=units_per_hour, temperature=temperature,
                max_tokens=max_tokens, thinking_budget=thinking_budget, timeout_s=timeout_s, cache_dir=cache_dir)
    provider, pname, role = s["provider"], s["provider_name"], s["role"]
    if not quiet:
        print(f"target {target.id}: {target.title}")
        print(f"  best known: {target.best_known} ({target.status}); provider {pname} ({provider.kind}, {provider.mode}); "
              f"model {s['model']} [{s['precision']}]; concurrency {concurrency}; thinking {role.thinking_budget}; "
              f"max tokens {role.max_tokens}; tool calls {tool_calls}")
        bs = budget_status(provider)
        if bs is not None:
            print(f"  frontier budget: ${bs['spent_usd']:.2f} of ${bs['cap_usd']:.2f} spent in {bs['month']} "
                  f"(cap in {bs['budget_file']}; offline replays never count)")
    runner = BuilderRunner(role, target, s["judge"], prompt=prompt, tool_calls=tool_calls, judge_gate=JudgeGate(judge_concurrency),
                           post_check=post_check, proposal_for=proposal_for)
    summary = run_episodes(runner.attempt, role_name=role_name, role=role, provider=provider, provider_name=pname, target=target,
                           attempts=attempts, seed=seed, ledger=ledger, meter=s["meter"], model=s["model"], model_version=s["model_version"],
                           precision=s["precision"], gpu=s["gpu"], concurrency=concurrency, group_id=group_id, stop_at_dollars=stop_at_dollars,
                           quiet=quiet, tool_calls=tool_calls, prompt_override=prompt is not None, context_extra=context_extra,
                           post_check_used=post_check is not None)
    if own_ledger:
        ledger.close()
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
