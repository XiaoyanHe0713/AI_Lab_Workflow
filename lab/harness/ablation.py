"""The ablation harness (Work package 3 acceptance): three arms on one verification family at matched dollars,
no notebook, notebook, notebook with gate; every attempt in the ledger.

    python -m lab.harness.ablation --target quad-d7 --attempts 16 --mode offline --harness-mode offline   # fixtures
    python -m lab.harness.ablation --target quad-d7 --attempts 16 --gpu G4 --precision bf16 --concurrency 16 \
        --harness-provider openai --harness-mode online                                                      # served model

Arm A (no notebook) runs the target's prompt. Its attempts are the experience: the lesson harness extracts
lessons from them (a frontier provider during the build) and the bank consolidates them under the token
budget. Arm B (notebook) adds every active lesson of the family to the prompt. Arm C (notebook with gate)
adds the one lesson the gate selects plus the concrete check it wrote, and runs that check on every result
(logged in the attempt's meta, never scored). Each arm runs the same number of attempts or stops at the same
dollar budget; the report gives pass@k valid and optimal per arm, the gain over arm A, and the harness spend.
Nothing a harness writes enters a reward: the certificate is the only score in every arm.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from ..config import lab_config
from ..demo import run_group
from ..ledger import Ledger, new_id
from ..roles import KINDS, MODES
from ..targets import get_target
from .base import harness_provider
from .lessons import LessonBank, LessonHarness, apply_check

ARMS = ("no_notebook", "notebook", "notebook_gate")


def notebook_prompt(target, text: str) -> str:
    if not text.strip():
        return target.prompt + "\n\n(The notebook holds no lessons for this family yet.)"
    return target.prompt + "\n\nLessons from earlier attempts on this family (the notebook):\n" + text


def gate_prompt(target, gate: dict) -> str:
    text = LessonHarness.gate_text(gate)
    if not text:
        return target.prompt + "\n\n(The gate selected no lesson for this target.)"
    return target.prompt + "\n\n" + text


def plan_ablation(*, target_id: str, attempts: int, seed: int, cfg: dict, ledger: Ledger, harness: LessonHarness,
                  provider_name: str = "local", mode: str | None = None, token_budget: int | None = None,
                  budget_usd: float | None = None, run_kw: dict | None = None, quiet: bool = True) -> dict:
    """Arm A, the lessons, the consolidation and the gate: everything the other two arms' prompts depend on."""
    target = get_target(target_id)
    bank = harness.bank
    token_budget = int(token_budget or cfg.get("harness", {}).get("lesson_token_budget", 2000))
    kw = dict(run_kw or {})
    arm_a = run_group(target_id=target_id, attempts=attempts, seed=seed, provider_name=provider_name, role_name="builder",
                      ledger=ledger, ledger_path=ledger.path, cfg=cfg, mode=mode, quiet=quiet, stop_at_dollars=budget_usd,
                      context_extra={"arm": "no_notebook", "lesson_ids": []}, **kw)
    rows_a = ledger.attempts(group_id=arm_a["group_id"])
    lessons = harness.extract(target, rows_a, group_id=arm_a["group_id"])
    actions = bank.consolidate(lessons, token_budget, family=target.family)
    text, ids = bank.notebook_text(target.family, token_budget)
    gate = harness.gate(target)
    return {"target": target, "arm_a": arm_a, "lessons": lessons, "consolidation": actions, "notebook_ids": ids,
            "prompt_notebook": notebook_prompt(target, text), "gate": gate, "prompt_gate": gate_prompt(target, gate),
            "token_budget": token_budget}


def run_ablation(*, target_id: str, attempts: int, seed: int = 0, cfg: dict | None = None, ledger_path: str | Path,
                 harness: LessonHarness, provider_name: str = "local", mode: str | None = None, token_budget: int | None = None,
                 budget_usd: float | None = None, gpu: str | None = None, precision: str | None = None,
                 units_per_hour: float | None = None, concurrency: int = 1, judge_concurrency: int = 4,
                 fixtures_root: str | Path | None = None, quiet: bool = True, cache_dir: str | Path | None = None,
                 timeout_s: float | None = None, out_dir: str | Path | None = None, ledger: Ledger | None = None) -> dict:
    cfg = cfg or lab_config()
    own = ledger is None
    ledger = ledger or Ledger(ledger_path)
    run_kw = dict(gpu=gpu, precision=precision, units_per_hour=units_per_hour, temperature=None, max_tokens=None, timeout_s=timeout_s,
                  fixtures_root=fixtures_root, cache_dir=cache_dir, concurrency=concurrency, judge_concurrency=judge_concurrency)
    plan = plan_ablation(target_id=target_id, attempts=attempts, seed=seed, cfg=cfg, ledger=ledger, harness=harness,
                         provider_name=provider_name, mode=mode, token_budget=token_budget, budget_usd=budget_usd, run_kw=run_kw, quiet=quiet)
    target, gate = plan["target"], plan["gate"]
    arm_b = run_group(target_id=target_id, attempts=attempts, seed=seed, provider_name=provider_name, role_name="builder",
                      ledger=ledger, ledger_path=ledger.path, cfg=cfg, mode=mode, quiet=quiet, stop_at_dollars=budget_usd,
                      prompt=plan["prompt_notebook"], context_extra={"arm": "notebook", "lesson_ids": plan["notebook_ids"]}, **run_kw)
    check = gate.get("check")
    gate_lesson = gate.get("lesson")
    arm_c = run_group(target_id=target_id, attempts=attempts, seed=seed, provider_name=provider_name, role_name="builder",
                      ledger=ledger, ledger_path=ledger.path, cfg=cfg, mode=mode, quiet=quiet, stop_at_dollars=budget_usd,
                      prompt=plan["prompt_gate"],
                      context_extra={"arm": "notebook_gate", "lesson_ids": [gate_lesson.id] if gate_lesson else [],
                                     "gate_check": check, "gate_harness_attempt_id": gate.get("harness_attempt_id")},
                      post_check=(lambda result, meta: apply_check(check, result)) if check and check.get("valid") else None, **run_kw)
    arms = {"no_notebook": plan["arm_a"], "notebook": arm_b, "notebook_gate": arm_c}
    base = arms["no_notebook"]

    def cell(s):
        return {"group_id": s["group_id"], "attempts": s["attempts"], "passed": s["passed"], "optimal": s["optimal"],
                "pass_at_k_valid": s["pass_at_k"], "pass_at_k_optimal": s["optimal_at_k"], "dollars": s["dollars"],
                "api_dollars": s["api_dollars"], "gpu_seconds": s["gpu_seconds"], "best_score": s["best_score"],
                "stopped_at": s.get("stopped_at"), "gate_checks": s.get("gate_checks"), "wall_s": s["wall_s"]}

    report = {
        "id": new_id("abl"), "target": target.id, "family": target.family, "best_known": target.best_known, "attempts_per_arm": attempts,
        "seed": seed, "budget_usd_per_arm": budget_usd, "provider": base["provider"], "mode": base["mode"], "model": base["model"],
        "model_version": base["model_version"], "precision": base["precision"], "gpu": base["gpu"],
        "harness": {"provider": harness.provider.name, "mode": harness.provider.mode, "model": harness.provider.model,
                    "api_dollars": harness.spent, "calls": len(harness.calls)},
        "lessons_extracted": [{"id": l.id, "level": l.level, "evidence": l.evidence_tag, "rule": l.rule} for l in plan["lessons"]],
        "consolidation": plan["consolidation"], "notebook_ids": plan["notebook_ids"], "token_budget": plan["token_budget"],
        "gate": {"lesson_id": gate_lesson.id if gate_lesson else None, "lesson": gate_lesson.text if gate_lesson else None,
                 "check": check, "reason": gate.get("reason"), "why": gate.get("why")},
        "arms": {name: cell(s) for name, s in arms.items()},
        "gain_over_no_notebook": {name: {"pass@1_valid": s["pass_at_k"]["pass@1"] - base["pass_at_k"]["pass@1"],
                                         "pass@1_optimal": s["optimal_at_k"]["pass@1"] - base["optimal_at_k"]["pass@1"],
                                         "passed": s["passed"] - base["passed"], "optimal": s["optimal"] - base["optimal"]}
                                  for name, s in arms.items() if name != "no_notebook"},
        "matched_dollars": {"max_spread": max(s["dollars"] for s in arms.values()) - min(s["dollars"] for s in arms.values()),
                            "note": "each arm ran the same number of attempts" if budget_usd is None else f"each arm stopped at ${budget_usd}"},
        "ledger": str(Path(ledger.path).resolve()), "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": "the certificate is the only score in every arm; lessons, the gate's choice and its check are logged, never scored",
    }
    if own:
        ledger.close()
    out_dir = Path(out_dir) if out_dir else Path(ledger_path).resolve().parent / "ablations"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{report['id']}.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    (out_dir / f"{report['id']}.md").write_text(report_markdown(report), encoding="utf-8")
    report["files"] = {"json": str(out_dir / f"{report['id']}.json"), "markdown": str(out_dir / f"{report['id']}.md")}
    if not quiet:
        print(report_markdown(report))
    return report


def report_markdown(r: dict) -> str:
    lines = [f"# Ablation {r['id']}: {r['target']} ({r['attempts_per_arm']} attempts per arm, seed {r['seed']})", "",
             f"Model {r['model']} ({r['model_version']}), precision {r['precision']}, provider {r['provider']} in mode {r['mode']}; "
             f"harness {r['harness']['provider']} ({r['harness']['model']}) in mode {r['harness']['mode']}, "
             f"{r['harness']['calls']} calls, ${r['harness']['api_dollars']:.4f}. Written {r['written']}.", "",
             "| Arm | attempts | valid | optimal | pass@1 valid | pass@1 optimal | pass@16 valid | $ | gate checks |", "|---|---|---|---|---|---|---|---|---|"]
    for name in ARMS:
        a = r["arms"][name]
        pk, ok = a["pass_at_k_valid"], a["pass_at_k_optimal"]
        p16 = pk.get("pass@16", pk.get(f"pass@{a['attempts']}", float("nan")))
        g = a.get("gate_checks")
        gtxt = f"{g['passed']}/{g['applied']}" if g else "-"
        lines.append(f"| {name} | {a['attempts']} | {a['passed']} | {a['optimal']} | {pk['pass@1']:.3f} | {ok['pass@1']:.3f} | {p16:.3f} | {a['dollars']:.4f} | {gtxt} |")
    lines += ["", f"Gain over no notebook (pass@1 valid, pass@1 optimal): " +
              "; ".join(f"{k}: {v['pass@1_valid']:+.3f}, {v['pass@1_optimal']:+.3f}" for k, v in r["gain_over_no_notebook"].items()),
              f"Matched dollars: spread ${r['matched_dollars']['max_spread']:.4f} ({r['matched_dollars']['note']}).", "",
              f"Lessons extracted: {len(r['lessons_extracted'])}; in the notebook: {len(r['notebook_ids'])} (token budget {r['token_budget']}).",
              f"Gate: {r['gate']['lesson'] or 'none'}" + (f"; check: {r['gate']['check']['description']} [{r['gate']['check']['expression']}]"
                                                          if r['gate']['check'] and r['gate']['check'].get('valid') else "; no valid check"),
              "", r["note"], f"Ledger: {r['ledger']}"]
    if r["mode"] == "offline":
        lines.append("All three arms replayed recorded fixtures: a machinery check, not a measurement of the model.")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m lab.harness.ablation", description=__doc__.split("\n\n")[0])
    ap.add_argument("--target", default="quad-d7")
    ap.add_argument("--attempts", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--provider", default="local", choices=list(KINDS))
    ap.add_argument("--mode", default=None, choices=list(MODES))
    ap.add_argument("--harness-provider", default=None, help="openai | anthropic (default: harness.provider in config)")
    ap.add_argument("--harness-mode", default=None, choices=list(MODES))
    ap.add_argument("--budget-usd", type=float, default=None, help="stop every arm at this spend (matched dollars)")
    ap.add_argument("--token-budget", type=int, default=None)
    ap.add_argument("--ledger", default=None)
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--precision", default=None)
    ap.add_argument("--units-per-hour", type=float, default=None)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=None)
    ap.add_argument("--fixtures", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    cfg = lab_config()
    ledger_path = args.ledger or cfg["storage"]["ledger"]
    ledger = Ledger(ledger_path)
    hp = harness_provider(cfg, name=args.harness_provider, mode=args.harness_mode, ledger=ledger, fixtures_root=args.fixtures)
    harness = LessonHarness(hp, ledger, LessonBank(ledger), cfg=cfg)
    report = run_ablation(target_id=args.target, attempts=args.attempts, seed=args.seed, cfg=cfg, ledger_path=ledger_path, ledger=ledger,
                          harness=harness, provider_name=args.provider, mode=args.mode, token_budget=args.token_budget,
                          budget_usd=args.budget_usd, gpu=args.gpu, precision=args.precision, units_per_hour=args.units_per_hour,
                          concurrency=args.concurrency, fixtures_root=args.fixtures, quiet=args.json, timeout_s=args.timeout, out_dir=args.out)
    ledger.close()
    if args.json:
        print(json.dumps(report, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
