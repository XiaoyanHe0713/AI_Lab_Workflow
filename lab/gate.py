"""The training gate (Work package 5): before any training on a family, the base with no adapter runs 64
times on each of its targets; a family trains only if its pass rate lies between 5 and 80 percent.

    python -m lab.gate --targets quad-square-d5 quad-triangle-d4 --attempts 64 --thinking 0 512 --max-tokens 8192
    python -m lab.gate --targets quad-d7 --attempts 16 --mode offline          # the machinery on fixtures

For every target and thinking budget one group runs (as parallel requests), and a gate memo is written per
target: pass@1, pass@16 and pass@64 for valid and for optimal attempts, the verdict, the cost from the
meter, the group id and the ledger path.

The verdict is taken on the pass rate over the group (the fraction of certified attempts, which is the
pass@1 estimate from all 64 samples): that rate is the per-attempt probability a GRPO group of 16 needs to
vary, which is what the proposal's band protects. pass@64 itself is 1 whenever a single attempt of the 64
passes, so it cannot gate; it is reported beside pass@16 for the record. The memo also counts the attempts
that ended at the token cap and those without a code block, the two ways a long derivation fails before the
certifier sees anything. The memo reports; a person signs the gate. Nothing here is a score: the
certificates in the ledger are.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

from .config import lab_config
from .demo import run_group
from .roles import KINDS, MODES
from .targets import get_target

BAND = (0.05, 0.80)


def verdict(pass_rate: float, band: tuple[float, float] = BAND) -> dict:
    """The gate rule on the pass rate of valid attempts over the group."""
    lo, hi = band
    if pass_rate != pass_rate:  # nan: no attempts
        return {"in_band": False, "verdict": "no attempts", "action": "run the gate"}
    if pass_rate < lo:
        return {"in_band": False, "verdict": "below the band",
                "action": "a curriculum of easier instances or an imitation warm start, then re-measure"}
    if pass_rate > hi:
        return {"in_band": False, "verdict": "above the band", "action": "evaluation target only; nothing to learn"}
    return {"in_band": True, "verdict": "in the band", "action": "the family may train"}


def gate_record(summary: dict) -> dict:
    pk, ok = summary["pass_at_k"], summary["optimal_at_k"]
    n = summary["attempts"]
    rate = summary.get("pass_rate")
    if rate is None:
        rate = summary["passed"] / n if n else float("nan")
    v = verdict(rate)
    return {
        "target": summary["target"], "best_known": summary["best_known"], "attempts": n,
        "requested_attempts": summary.get("requested_attempts", n), "thinking_budget": summary.get("thinking_budget", 0),
        "max_tokens": summary.get("max_tokens"),
        "provider": summary["provider"], "mode": summary["mode"], "model": summary["model"],
        "model_version": summary["model_version"], "precision": summary["precision"], "gpu": summary["gpu"],
        "passed": summary["passed"], "optimal": summary["optimal"], "pass_rate": rate,
        "optimal_rate": summary.get("optimal_rate", (summary["optimal"] / n) if n else float("nan")),
        "capped": summary.get("capped", 0), "no_code": summary.get("no_code", 0),
        "pass_at_k_valid": pk, "pass_at_k_optimal": ok, "statistic": "pass rate over the group (pass@1 estimate from all samples)",
        "band": {"low": BAND[0], "high": BAND[1]}, **v,
        "verify_first_found": summary.get("verify_first", {}).get("found", 0),
        "dollars": summary["dollars"], "api_dollars": summary["api_dollars"], "gpu_seconds": summary["gpu_seconds"],
        "request_latency_s": summary.get("request_latency_s"),
        "cpu_seconds": summary["cpu_seconds"], "units": summary["units"], "wall_s": summary["wall_s"],
        "concurrency": summary.get("concurrency", 1), "group_id": summary["group_id"], "ledger": summary["ledger"],
        "best_attempt_id": summary["best_attempt_id"], "best_score": summary["best_score"],
        "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def memo_markdown(rec: dict) -> str:
    pk, ok = rec["pass_at_k_valid"], rec["pass_at_k_optimal"]
    n = rec["attempts"]
    vf = rec.get("verify_first_found", 0)
    lines = [
        f"# Gate memo: {rec['target']} (thinking budget {rec['thinking_budget']})",
        "",
        f"Written {rec['written']}. Model {rec['model']} ({rec['model_version']}), precision {rec['precision']}, "
        f"GPU {rec['gpu']}, provider {rec['provider']} in mode {rec['mode']}, {rec['concurrency']} parallel requests, "
        f"{rec.get('max_tokens') or 'the role default'} answer tokens.",
        "",
        f"Verdict: **{rec['verdict']}** (pass rate {rec['pass_rate']:.3f} = {rec['passed']} of {n} certified; "
        f"band {rec['band']['low']:.0%} to {rec['band']['high']:.0%}). Action: {rec['action']}.",
        "",
        f"Verify-first: {'PASS' if vf > 0 else 'FAIL'}: the best known value ({rec['best_known']}) was found {vf} time(s) in {n} attempts.",
        "",
        "| Quantity | Value |", "|---|---|",
        f"| attempts | {n} (requested {rec['requested_attempts']}) |",
        f"| valid, optimal | {rec['passed']}, {rec['optimal']} (pass rate {rec['pass_rate']:.3f}, optimal rate {rec['optimal_rate']:.3f}) |",
        "| pass@k (valid) | " + ", ".join(f"{k} = {v:.3f}" for k, v in pk.items()) + " |",
        "| pass@k (optimal) | " + ", ".join(f"{k} = {v:.3f}" for k, v in ok.items()) + " |",
        f"| ended at the token cap, no code block | {rec['capped']}, {rec['no_code']} |",
        f"| best score | {rec['best_score']} (attempt {rec['best_attempt_id']}) |",
        f"| cost | ${rec['dollars']:.4f} (api ${rec['api_dollars']:.4f}); GPU {rec['gpu_seconds']:.1f} s of wall time "
        f"(request latencies summed {rec.get('request_latency_s') or 0:.1f} s); CPU {rec['cpu_seconds']:.1f} s; {rec['units']:.3f} units |",
        f"| wall time | {rec['wall_s']:.1f} s |",
        f"| group, ledger | {rec['group_id']}, {rec['ledger']} |",
        "",
        f"Statistic: {rec['statistic']}. pass@{n} is one whenever a single attempt passes and does not gate.",
        "",
        "Offline replays and quantized builds never enter a reported cell; the precision and mode columns above say "
        "whether this record is one." if rec["mode"] == "offline" or rec["precision"] not in ("bf16",) else
        "This record is at the frozen base's precision and may be reported.",
    ]
    return "\n".join(lines) + "\n"


def run_gate(*, targets: list[str], attempts: int, thinking: list[int], provider_name: str | None, mode: str | None,
             ledger_path: str | Path, gpu: str | None, precision: str | None, units_per_hour: float | None,
             concurrency: int, out_dir: str | Path | None = None, cfg: dict | None = None, seed: int = 0,
             fixtures_root: str | Path | None = None, quiet: bool = False, timeout_s: float | None = None,
             cache_dir: str | Path | None = None, max_tokens: int | None = None) -> list[dict]:
    cfg = cfg or lab_config()
    out_dir = Path(out_dir) if out_dir else Path(ledger_path).resolve().parent / "gates"
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for tid in targets:
        get_target(tid)  # fail early on a bad id
        for tb in thinking:
            summary = run_group(target_id=tid, attempts=attempts, seed=seed, provider_name=provider_name, role_name="builder",
                                ledger_path=ledger_path, gpu=gpu, precision=precision, units_per_hour=units_per_hour,
                                temperature=None, max_tokens=max_tokens, timeout_s=timeout_s, cfg=cfg, fixtures_root=fixtures_root,
                                quiet=quiet, cache_dir=cache_dir, mode=mode, concurrency=concurrency, thinking_budget=tb)
            rec = gate_record(summary)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            base = out_dir / f"{tid}-t{tb}-{stamp}"
            base.with_suffix(".json").write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
            base.with_suffix(".md").write_text(memo_markdown(rec), encoding="utf-8")
            rec["memo"] = str(base.with_suffix(".md"))
            records.append(rec)
            if not quiet:
                print(f"\ngate memo written: {rec['memo']}\n  {tid} thinking {tb}: {rec['verdict']} "
                      f"(pass rate {rec['pass_rate']:.3f}, optimal rate {rec['optimal_rate']:.3f}; "
                      f"{rec['capped']} at the cap, {rec['no_code']} without code); ${rec['dollars']:.4f}")
    return records


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m lab.gate", description=__doc__.split("\n\n")[0])
    ap.add_argument("--targets", nargs="+", required=True)
    ap.add_argument("--attempts", type=int, default=64)
    ap.add_argument("--thinking", nargs="+", type=int, default=[0], help="thinking budgets, e.g. 0 512 1024")
    ap.add_argument("--max-tokens", type=int, default=None, help="answer tokens per attempt (default: the role's, 4096)")
    ap.add_argument("--provider", default=None, choices=list(KINDS))
    ap.add_argument("--mode", default=None, choices=list(MODES))
    ap.add_argument("--ledger", default=None)
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--precision", default=None)
    ap.add_argument("--units-per-hour", type=float, default=None)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=None)
    ap.add_argument("--out", default=None, help="directory for the memos (default: <ledger dir>/gates)")
    ap.add_argument("--fixtures", default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    cfg = lab_config()
    ledger_path = args.ledger or cfg["storage"]["ledger"]
    recs = run_gate(targets=args.targets, attempts=args.attempts, thinking=args.thinking, provider_name=args.provider,
                    mode=args.mode, ledger_path=ledger_path, gpu=args.gpu, precision=args.precision,
                    units_per_hour=args.units_per_hour, concurrency=args.concurrency, out_dir=args.out, cfg=cfg,
                    seed=args.seed, fixtures_root=args.fixtures, quiet=args.json, timeout_s=args.timeout, max_tokens=args.max_tokens)
    if args.json:
        print(json.dumps(recs, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
