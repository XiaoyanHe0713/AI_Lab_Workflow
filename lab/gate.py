"""The training gate (Work package 5): before any training on a family, the base with no adapter runs 64
times on each of its targets; a family trains only if its pass@64 lies between 5 and 80 percent.

    python -m lab.gate --targets quad-d7 quad-square-d5 --attempts 64 --thinking 0 512
    python -m lab.gate --targets quad-d7 --attempts 16 --mode offline          # the machinery on fixtures

For every target and thinking budget one group runs (as parallel requests), and a gate memo is written per
target: pass@1, pass@16 and pass@64 for valid and for optimal attempts, the in-band verdict against the
5 to 80 percent rule, the cost from the meter, the group id and the ledger path. The memo reports; a person
signs the gate. Nothing here is a score: the certificates in the ledger are.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from .config import lab_config
from .demo import run_group
from .roles import KINDS, MODES
from .targets import get_target

BAND = (0.05, 0.80)


def verdict(pass_at_64: float, band: tuple[float, float] = BAND) -> dict:
    """The gate rule on pass@64 of valid attempts."""
    lo, hi = band
    if pass_at_64 != pass_at_64:  # nan: no attempts
        return {"in_band": False, "verdict": "no attempts", "action": "run the gate"}
    if pass_at_64 < lo:
        return {"in_band": False, "verdict": "below the band",
                "action": "a curriculum of easier instances or an imitation warm start, then re-measure"}
    if pass_at_64 > hi:
        return {"in_band": False, "verdict": "above the band", "action": "evaluation target only; nothing to learn"}
    return {"in_band": True, "verdict": "in the band", "action": "the family may train"}


def gate_record(summary: dict) -> dict:
    pk, ok = summary["pass_at_k"], summary["optimal_at_k"]
    n = summary["attempts"]
    key64 = "pass@64" if "pass@64" in pk else f"pass@{n}"
    v = verdict(pk[key64])
    return {
        "target": summary["target"], "best_known": summary["best_known"], "attempts": n,
        "requested_attempts": summary.get("requested_attempts", n), "thinking_budget": summary.get("thinking_budget", 0),
        "provider": summary["provider"], "mode": summary["mode"], "model": summary["model"],
        "model_version": summary["model_version"], "precision": summary["precision"], "gpu": summary["gpu"],
        "passed": summary["passed"], "optimal": summary["optimal"],
        "pass_at_k_valid": pk, "pass_at_k_optimal": ok, "pass_at_64_key": key64,
        "band": {"low": BAND[0], "high": BAND[1]}, **v,
        "dollars": summary["dollars"], "api_dollars": summary["api_dollars"], "gpu_seconds": summary["gpu_seconds"],
        "cpu_seconds": summary["cpu_seconds"], "units": summary["units"], "wall_s": summary["wall_s"],
        "concurrency": summary.get("concurrency", 1), "group_id": summary["group_id"], "ledger": summary["ledger"],
        "best_attempt_id": summary["best_attempt_id"], "best_score": summary["best_score"],
        "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def memo_markdown(rec: dict) -> str:
    pk, ok = rec["pass_at_k_valid"], rec["pass_at_k_optimal"]
    lines = [
        f"# Gate memo: {rec['target']} (thinking budget {rec['thinking_budget']})",
        "",
        f"Written {rec['written']}. Model {rec['model']} ({rec['model_version']}), precision {rec['precision']}, "
        f"GPU {rec['gpu']}, provider {rec['provider']} in mode {rec['mode']}, {rec['concurrency']} parallel requests.",
        "",
        f"Verdict: **{rec['verdict']}** ({rec['pass_at_64_key']} valid = {pk[rec['pass_at_64_key']]:.3f}; "
        f"band {rec['band']['low']:.0%} to {rec['band']['high']:.0%}). Action: {rec['action']}.",
        "",
        "| Quantity | Value |", "|---|---|",
        f"| attempts | {rec['attempts']} (requested {rec['requested_attempts']}) |",
        f"| valid, optimal | {rec['passed']}, {rec['optimal']} (best known {rec['best_known']}) |",
        "| pass@k (valid) | " + ", ".join(f"{k} = {v:.3f}" for k, v in pk.items()) + " |",
        "| pass@k (optimal) | " + ", ".join(f"{k} = {v:.3f}" for k, v in ok.items()) + " |",
        f"| best score | {rec['best_score']} (attempt {rec['best_attempt_id']}) |",
        f"| cost | ${rec['dollars']:.4f} (api ${rec['api_dollars']:.4f}); GPU {rec['gpu_seconds']:.1f} s; "
        f"CPU {rec['cpu_seconds']:.1f} s; {rec['units']:.3f} units |",
        f"| wall time | {rec['wall_s']:.1f} s |",
        f"| group, ledger | {rec['group_id']}, {rec['ledger']} |",
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
             cache_dir: str | Path | None = None) -> list[dict]:
    cfg = cfg or lab_config()
    out_dir = Path(out_dir) if out_dir else Path(ledger_path).resolve().parent / "gates"
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for tid in targets:
        get_target(tid)  # fail early on a bad id
        for tb in thinking:
            summary = run_group(target_id=tid, attempts=attempts, seed=seed, provider_name=provider_name, role_name="builder",
                                ledger_path=ledger_path, gpu=gpu, precision=precision, units_per_hour=units_per_hour,
                                temperature=None, max_tokens=None, timeout_s=timeout_s, cfg=cfg, fixtures_root=fixtures_root,
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
                      f"({rec['pass_at_64_key']} valid {rec['pass_at_k_valid'][rec['pass_at_64_key']]:.3f}, "
                      f"optimal {rec['pass_at_k_optimal'][rec['pass_at_64_key']]:.3f}); ${rec['dollars']:.4f}")
    return records


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m lab.gate", description=__doc__.split("\n\n")[0])
    ap.add_argument("--targets", nargs="+", required=True)
    ap.add_argument("--attempts", type=int, default=64)
    ap.add_argument("--thinking", nargs="+", type=int, default=[0], help="thinking budgets, e.g. 0 512 1024")
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
                    seed=args.seed, fixtures_root=args.fixtures, quiet=args.json, timeout_s=args.timeout)
    if args.json:
        print(json.dumps(recs, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
