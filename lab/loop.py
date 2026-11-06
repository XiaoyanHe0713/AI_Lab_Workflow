"""The loop (proposal, "The loop"), one generation on one family: search proposes candidates with the current model
and the archive; the judge certifies; the breaker attacks the survivors; survivors enter the archive and the lesson
bank; lessons are extracted; the generation ends with an archive snapshot and a ledger pointer. No training yet:
every role is the frozen base with no adapter, so this generation is the frozen-search control of E2 and the
experience the transfer matrix (lab/harness/matrix.py) draws on.

    python -m lab.loop --targets quad-d7 quad-square-d5 quad-triangle-d2 quad-triangle-d3 quad-triangle-d4 \\
        --attempts 64 --max-tokens 8192 --concurrency 16 --gpu G4 --precision bf16 \\
        --harness-provider openai --harness-mode online                      # Colab: the verification run
    python -m lab.loop --targets quad-d7 --attempts 16 --mode offline --no-harness    # the machinery on fixtures

Per target, in the order given (the curriculum rungs):
  1. the gate chooses lessons for the target when the bank holds any (a frontier call; without a harness the
     notebook text stands in);
  2. the builder runs its gate group from the plain prompt (the frozen base's record, from which the gate memo is
     written), and the diagnosis is mined on it (week 5): the checker's detectors when it runs, the built-in detectors
     alone otherwise, each kept only if it holds on the held-out half; their firing table names the failure;
  3. the theorist proposes, lemmas checked by computer algebra, structure, hypotheses, with the librarian's retrieval
     under the cutoff asked the question the diagnosis names;
  4. the builder runs a lesson group when lessons exist, and a proposal group where each attempt is derived from one of
     the theorist's checked proposals (the delayed credit); the symptom rate of every kept detector is recorded per group;
  5. the judge certifies every attempt inside its episode; the distinct certified candidates, best first, are the
     survivors;
  6. the breaker attacks each survivor with a group of attempts; a confirmed break voids the candidate and raises an
     exploit alarm (on quadrature a certificate covers the whole class, so a confirmed break means the certifier is wrong);
  7. the survivors enter the archive with their certificate, dossier, witness record, lesson ids, citations and
     reading-list snapshot; the lesson harness extracts lessons from the gate group, told which detectors fired on
     which attempts so that a lesson can name the failure it addresses, and the bank consolidates them under the
     token budget;
  8. the gate memo is written (lab/gate.py) and the delayed credit of every proposal computed.
Every attempt of every role is a ledger row with its role, adapter version, prompt hash, context (lesson ids, skill
ids, reading-list snapshot, citation ids, proposal or candidate), output, tool log, certificate, diagnosis, cost and
seed. Concurrency as in the demo: waves of parallel requests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .config import lab_config
from .demo import prepare
from .episode import JudgeGate, run_episodes
from .gate import gate_record, memo_markdown
from .harness.ablation import gate_prompt, notebook_prompt
from .harness.archive import Archive
from .harness.base import harness_provider
from .harness.checker import Checker, diagnosis_query, firing_table, kept_detectors, mine_builtin, symptom_rates
from .harness.lessons import LessonBank, LessonHarness, apply_check
from .harness.librarian import Librarian, ReadingList
from .ledger import Attempt, Ledger, new_id
from .roles import KINDS, MODES, make_provider
from .roles.breaker import BREAKER_SYSTEM, BreakerRunner, candidate_from_row
from .roles.builder import BuilderRunner
from .roles.theorist import THEORIST_SYSTEM, TheoristRunner, proposal_from_row, theorist_prompt
from .targets import get_target

CARRIERS = ("none", "lessons")


def template_dossier(row: Attempt, target) -> dict:
    """A dossier written by the program when no checker harness runs: the claim states only what the certificate shows."""
    cert = row.certificate or {}
    d = cert.get("details", {}) or {}
    if target.family == "quadrature":
        claim = (f"A {d.get('node_count')}-node rule on the {d.get('domain')} integrates every polynomial of degree at most "
                 f"{d.get('degree')} within {d.get('tol')} (certified by {cert.get('certifier')} {cert.get('version')}).")
        conds = [f"domain {d.get('domain')}", f"degree <= {d.get('degree')}",
                 "positive weights" if d.get("all_weights_positive") else "weights of either sign",
                 "nodes inside the domain" if d.get("all_nodes_interior") else "nodes possibly outside the domain"]
    else:
        claim = f"The candidate passed {cert.get('certifier')} {cert.get('version')} with score {row.score} on {target.id}."
        conds = [f"target {target.id}", f"family {target.family}"]
    code = (row.meta or {}).get("code") or ""
    return {"claim": claim, "conditions": conds, "written_by": {"harness": "template", "harness_attempt_id": None},
            "evidence": {"attempt_id": row.id, "target": row.target, "certifier": cert.get("certifier"), "certifier_version": cert.get("version"),
                         "score": row.score, "passed": bool(row.passed),
                         "certificate_sha256": hashlib.sha256(json.dumps(cert, sort_keys=True, default=str).encode()).hexdigest(),
                         "code_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(), "summary": "written by the program from the certificate"}}


def theorist_retrieval(librarian: Librarian | None, target, query: str | None = None) -> dict:
    """The theorist's retrieval tool: the librarian's lexical ranking under the target's cutoff (no model), every
    returned document a citation row; the snapshot and citation ids go into the episode's context. The query is the
    one the diagnosis names (lab/harness/checker.py, diagnosis_query) or, without a firing detector, the target's title."""
    if librarian is None:
        return {"snapshot": None, "citation_ids": [], "documents": [], "query": query}
    query = query or f"{target.title} {target.family}"
    ret = librarian.retrieve(query, target=target, use_model=False, k=3, purpose="theorist")
    return {"snapshot": ret["snapshot"], "citation_ids": ret["citation_ids"], "documents": ret["documents"], "query": query}


def survivors_of(rows: list[Attempt], k: int) -> list[Attempt]:
    """Distinct certified candidates (by code), best score first, at most k."""
    out: dict[str, Attempt] = {}
    for r in rows:
        if not r.passed:
            continue
        code = (r.meta or {}).get("code") or r.output or ""
        sha = hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]
        lower = (r.certificate or {}).get("lower_is_better", True)
        cur = out.get(sha)
        if cur is None or (r.score is not None and cur.score is not None and ((r.score < cur.score) if lower else (r.score > cur.score))):
            out[sha] = r
    rows_sorted = sorted(out.values(), key=lambda r: (r.score if (r.certificate or {}).get("lower_is_better", True) else -(r.score or 0.0), r.seed or 0, r.id))
    return rows_sorted[:k]


def delayed_credit(proposals: list[dict], proposal_rows: list[Attempt], base_summary: dict, lower_is_better: bool) -> list[dict]:
    """Per proposal: the best certificate score among the builds derived from it minus the frozen baseline's best, and
    the pass rate of its builds against the baseline's pass rate (positive = better in both)."""
    out = []
    base_best = base_summary.get("best_score")
    base_rate = base_summary.get("pass_rate")
    for p in proposals:
        rows = [r for r in proposal_rows if (r.context or {}).get("proposal_id") == p["id"]]
        passed = [r for r in rows if r.passed and r.score is not None]
        best = (min if lower_is_better else max)((r.score for r in passed), default=None)
        if best is None or base_best is None:
            credit = None
        else:
            credit = (base_best - best) if lower_is_better else (best - base_best)
        rate = (len(passed) / len(rows)) if rows else None
        out.append({"proposal_id": p["id"], "lemmas_passed": p.get("passed"), "lemmas_proposed": p.get("proposed"), "builds": len(rows),
                    "builds_passed": len(passed), "best_score": best, "baseline_best": base_best, "credit": credit,
                    "pass_rate": rate, "baseline_pass_rate": base_rate,
                    "pass_rate_gain": (rate - base_rate) if (rate is not None and base_rate is not None) else None})
    return out


def run_generation(*, targets: list[str], attempts: int, cfg: dict, ledger: Ledger, archive: Archive, bank: LessonBank,
                   harness: LessonHarness | None = None, checker: Checker | None = None, provider_name: str = "local",
                   mode: str | None = None, carrier: str = "lessons", lesson_attempts: int | None = None, theorist_attempts: int = 0,
                   proposal_attempts: int = 0, breaker_attempts: int = 0, breaker_candidates: int = 4, tool_calls: int = 0,
                   theorist_tool_calls: int = 0, breaker_tool_calls: int = 0, gpu: str | None = None, precision: str | None = None,
                   units_per_hour: float | None = None, max_tokens: int | None = 8192, thinking_budget: int | None = 0,
                   timeout_s: float | None = None, concurrency: int = 1, judge_concurrency: int = 4, fixtures_root=None,
                   cache_dir=None, seed: int = 0, token_budget: int | None = None, out_dir: str | Path | None = None,
                   generation_id: str | None = None, quiet: bool = True, reading_list: ReadingList | None = None,
                   mine_detectors: bool = True) -> dict:
    if carrier not in CARRIERS:
        raise ValueError(f"carrier must be one of {CARRIERS}")
    gen_id = generation_id or new_id("gen")
    t_gen = time.perf_counter()
    out_dir = Path(out_dir) if out_dir else Path(ledger.path).resolve().parent
    gates_dir, gens_dir = out_dir / "gates", out_dir / "generations"
    gates_dir.mkdir(parents=True, exist_ok=True)
    gens_dir.mkdir(parents=True, exist_ok=True)
    token_budget = int(token_budget or cfg.get("harness", {}).get("lesson_token_budget", 2000))
    lesson_attempts = attempts if lesson_attempts is None else int(lesson_attempts)
    provider = make_provider(provider_name, cfg, mode=mode, fixtures_root=fixtures_root, ledger=ledger)
    common = dict(cfg=cfg, provider_name=provider_name, provider=provider, gpu=gpu, precision=precision, units_per_hour=units_per_hour,
                  max_tokens=max_tokens, thinking_budget=thinking_budget, timeout_s=timeout_s, cache_dir=cache_dir, fixtures_root=fixtures_root)
    builder = prepare(role_name="builder", temperature=None, **common)
    theorist = prepare(role_name="theorist", temperature=None, system_prompt=THEORIST_SYSTEM, **common) if theorist_attempts > 0 else None
    breaker = prepare(role_name="breaker", temperature=None, system_prompt=BREAKER_SYSTEM, **common) if breaker_attempts > 0 else None
    librarian = None
    if theorist is not None:
        try:
            librarian = Librarian(provider, ledger, reading_list or ReadingList(cfg=cfg), cfg=cfg)
        except Exception as e:  # no reading list on this machine: the theorist works from the scoreboard entry alone
            if not quiet:
                print(f"  (no reading list: {type(e).__name__}: {e})")
    gate = JudgeGate(judge_concurrency)
    families = sorted({get_target(t).family for t in targets})
    record: dict = {"id": gen_id, "targets": list(targets), "families": families, "carrier": carrier, "attempts": attempts,
                    "lesson_attempts": lesson_attempts, "theorist_attempts": theorist_attempts, "proposal_attempts": proposal_attempts,
                    "breaker_attempts": breaker_attempts, "breaker_candidates": breaker_candidates,
                    "tool_calls": {"builder": tool_calls, "theorist": theorist_tool_calls, "breaker": breaker_tool_calls},
                    "provider": provider_name, "mode": provider.mode, "model": builder["model"], "model_version": builder["model_version"],
                    "precision": builder["precision"], "gpu": builder["gpu"], "adapters": "none (frozen base)",
                    "harness": ({"provider": harness.provider.name, "mode": harness.provider.mode, "model": harness.provider.model}
                                if harness else None),
                    "checker": bool(checker), "mine_detectors": bool(mine_detectors), "token_budget": token_budget, "seed": seed, "rungs": [],
                    "exploit_alarms": [], "started": datetime.now(timezone.utc).isoformat(timespec="seconds")}

    def run(role_setup: dict, runner_attempt, *, role_name: str, target, n: int, stage: str, label: str, extra: dict | None = None,
            prompt_override: bool = False, post_check_used: bool = False, tool_calls_n: int = 0) -> dict:
        ctx = {"generation_id": gen_id, "stage": stage, "carrier": carrier, "loop": True, **(extra or {})}
        return run_episodes(runner_attempt, role_name=role_name, role=role_setup["role"], provider=provider, provider_name=provider_name,
                            target=target, attempts=n, seed=seed, ledger=ledger, meter=role_setup["meter"], model=role_setup["model"],
                            model_version=role_setup["model_version"], precision=role_setup["precision"], gpu=role_setup["gpu"],
                            concurrency=concurrency, quiet=quiet, label=label, tool_calls=tool_calls_n, prompt_override=prompt_override,
                            context_extra=ctx, post_check_used=post_check_used)

    for rung_index, tid in enumerate(targets):
        target = get_target(tid)
        t_rung = time.perf_counter()
        rung: dict = {"target": tid, "rung": rung_index, "family": target.family, "best_known": target.best_known, "groups": {},
                      "transfer": target.transfer.to_dict() if target.transfer else None}
        if not quiet:
            print(f"\n=== generation {gen_id}, rung {rung_index}: {tid} ({target.title}) ===")
        # 1. lessons for this target (the gate when a harness runs; the notebook text otherwise), and the theorist
        lesson_prompt, lesson_ids, gate_info, post_check, lesson_text = None, [], None, None, ""
        if carrier == "lessons" and bank.active(target.family):
            if harness is not None:
                gate_info = harness.gate(target)
                lesson = gate_info.get("lesson")
                check = gate_info.get("check")
                lesson_prompt = gate_prompt(target, gate_info)
                lesson_ids = [lesson.id] if lesson else []
                lesson_text = LessonHarness.gate_text(gate_info)
                if check and check.get("valid"):
                    post_check = lambda result, meta, _c=check: apply_check(_c, result)  # noqa: E731
            else:
                text, ids = bank.notebook_text(target.family, token_budget)
                lesson_prompt, lesson_ids, lesson_text = notebook_prompt(target, text), ids, text
        rung["lessons"] = {"ids": lesson_ids, "gate": ({"lesson_id": gate_info["lesson"].id if gate_info.get("lesson") else None,
                                                        "check": gate_info.get("check"), "reason": gate_info.get("reason"),
                                                        "harness_attempt_id": gate_info.get("harness_attempt_id")} if gate_info else None),
                           "sources": sorted({t for lid in lesson_ids for t in ((bank.get(lid).targets if bank.get(lid) else []))})}
        # 2. the builder's gate group from the plain prompt (the frozen base's record), then the diagnosis (week 5): the
        #    detectors mined on this group, the checker's proposals when it runs and the built-in ones otherwise, each kept
        #    only if it holds on the held-out half, and their firing table, which names the failure before the theorist,
        #    the lesson writer and the matrix see it
        groups_rows: list[Attempt] = []
        base_runner = BuilderRunner(builder["role"], target, builder["judge"], tool_calls=tool_calls, judge_gate=gate)
        s_base = run(builder, base_runner.attempt, role_name="builder", target=target, n=attempts, stage="base", label="gate group, plain prompt",
                     tool_calls_n=tool_calls)
        base_rows = ledger.attempts(group_id=s_base["group_id"])
        groups_rows += base_rows
        rung["groups"]["base"] = {k: s_base[k] for k in ("group_id", "attempts", "passed", "optimal", "pass_rate", "optimal_rate", "best_score",
                                                           "dollars", "gpu_seconds", "capped", "no_code", "wall_s")}
        rung["groups"]["base"]["verify_first_found"] = s_base["verify_first"]["found"]
        mining = None
        if mine_detectors:
            try:
                mining = checker.mine(target, base_rows, seed=seed) if checker is not None else mine_builtin(ledger, target, base_rows, seed=seed)
            except Exception as e:
                rung["mining_error"] = f"{type(e).__name__}: {e}"
        kept = kept_detectors(ledger, tid)
        table = firing_table(kept, base_rows)
        rung["detectors"] = {"mining": ("checker" if checker is not None else "builtin") if mining is not None else "off",
                             "proposed": mining["proposed"] if mining else 0, "kept": [d["name"] for d in kept],
                             "origins": {d["name"]: d.get("origin") for d in kept},
                             "firing": [{k: d[k] for k in ("name", "fires", "failures_fired", "rate")} for d in table["detectors"]],
                             "attempts": table["n"], "failed": table["failed"],
                             "harness_attempt_id": mining.get("harness_attempt_id") if mining else None}
        # 3. the theorist: the librarian's retrieval under the cutoff, asked the question the diagnosis names, then the proposals
        proposals: list[dict] = []
        snapshot, citation_ids, docs = None, [], []
        query = diagnosis_query(target, table)
        rung["retrieval_query"] = query
        if librarian is not None:
            try:
                ret = theorist_retrieval(librarian, target, query=query)
                snapshot, citation_ids, docs = ret["snapshot"], ret["citation_ids"], ret["documents"]
            except Exception as e:
                rung["librarian_error"] = f"{type(e).__name__}: {e}"
        if theorist is not None:
            prompt = theorist_prompt(target, lessons_text=lesson_text, docs=docs)
            runner = TheoristRunner(theorist["role"], target, theorist["judge"], prompt=prompt, tool_calls=theorist_tool_calls, judge_gate=gate,
                                    context={"lesson_ids": lesson_ids, "citation_ids": citation_ids, "reading_list_snapshot": snapshot,
                                             "retrieval_query": query},
                                    timeout_s=timeout_s)
            s_th = run(theorist, runner.attempt, role_name="theorist", target=target, n=theorist_attempts, stage="theorist",
                       label="theorist proposals", tool_calls_n=theorist_tool_calls)
            th_rows = ledger.attempts(group_id=s_th["group_id"])
            proposals = [p for p in (proposal_from_row(r) for r in th_rows if r.passed) if p]
            rung["groups"]["theorist"] = {"group_id": s_th["group_id"], "attempts": s_th["attempts"], "passed": s_th["passed"],
                                          "pass_rate": s_th["pass_rate"], "proposals": len(proposals), "dollars": s_th["dollars"],
                                          "lemmas_proposed": sum((r.certificate or {}).get("details", {}).get("proposed", 0) for r in th_rows),
                                          "lemmas_passed": sum((r.certificate or {}).get("details", {}).get("passed", 0) for r in th_rows)}
        # 4. the builder's lesson group and proposal group
        if lesson_prompt is not None and lesson_attempts > 0:
            les_runner = BuilderRunner(builder["role"], target, builder["judge"], prompt=lesson_prompt, tool_calls=tool_calls, judge_gate=gate,
                                       post_check=post_check, context={"citation_ids": citation_ids, "reading_list_snapshot": snapshot})
            s_les = run(builder, les_runner.attempt, role_name="builder", target=target, n=lesson_attempts, stage="lessons", label="lesson group",
                        extra={"lesson_ids": lesson_ids, "gate_check": gate_info.get("check") if gate_info else None,
                               "lesson_sources": rung["lessons"]["sources"]}, prompt_override=True, post_check_used=post_check is not None,
                        tool_calls_n=tool_calls)
            groups_rows += ledger.attempts(group_id=s_les["group_id"])
            rung["groups"]["lessons"] = {k: s_les[k] for k in ("group_id", "attempts", "passed", "optimal", "pass_rate", "optimal_rate",
                                                                 "best_score", "dollars", "gpu_seconds", "capped", "no_code", "wall_s")}
            rung["groups"]["lessons"]["gate_checks"] = s_les.get("gate_checks")
        if proposals and proposal_attempts > 0:
            prop_runner = BuilderRunner(builder["role"], target, builder["judge"], prompt=lesson_prompt, tool_calls=tool_calls, judge_gate=gate,
                                        proposal_for=lambda i, _p=proposals: _p[i % len(_p)],
                                        context={"citation_ids": citation_ids, "reading_list_snapshot": snapshot, "lesson_ids": lesson_ids})
            s_prop = run(builder, prop_runner.attempt, role_name="builder", target=target, n=proposal_attempts, stage="proposals",
                         label="builds derived from proposals", extra={"proposal_ids": [p["id"] for p in proposals]}, prompt_override=True,
                         tool_calls_n=tool_calls)
            prop_rows = ledger.attempts(group_id=s_prop["group_id"])
            groups_rows += prop_rows
            rung["groups"]["proposals"] = {k: s_prop[k] for k in ("group_id", "attempts", "passed", "optimal", "pass_rate", "optimal_rate",
                                                                   "best_score", "dollars", "gpu_seconds", "wall_s")}
            lower = all((r.certificate or {}).get("lower_is_better", True) for r in prop_rows if r.passed) if prop_rows else True
            rung["delayed_credit"] = delayed_credit(proposals, prop_rows, s_base, lower)
        # the symptoms per group (week 5): how often each kept detector fires with the plain prompt, with lessons and with
        # proposals. The loop uses the family's own lessons by design, so this is recorded here and never reported as transfer;
        # the matrix (lab/harness/matrix.py) reports the same quantity across families.
        if kept:
            stage_rows = {"base": base_rows}
            for stage in ("lessons", "proposals"):
                if stage in rung["groups"]:
                    stage_rows[stage] = ledger.attempts(group_id=rung["groups"][stage]["group_id"])
            rung["symptoms"] = {stage: symptom_rates(kept, rows) for stage, rows in stage_rows.items()}
        else:
            rung["symptoms"] = {}
        # 5. survivors and 6. the breaker
        survivors = survivors_of(groups_rows, breaker_candidates)
        rung["survivors"] = [{"attempt_id": r.id, "score": r.score, "optimal": (r.certificate or {}).get("optimal"), "stage": (r.context or {}).get("stage")}
                             for r in survivors]
        voided: set[str] = set()
        rung["breaker"] = []
        if breaker is not None and survivors:
            for ci, cand_row in enumerate(survivors):
                dossier = None
                if checker is not None:
                    try:
                        dossier = checker.dossier(cand_row, target)
                    except Exception as e:
                        dossier = {"claim": "", "conditions": [], "error": f"{type(e).__name__}: {e}"}
                cand = candidate_from_row(cand_row, dossier, label=f"candidate {ci + 1}")
                runner = BreakerRunner(breaker["role"], target, breaker["judge"], cand, tool_calls=breaker_tool_calls, judge_gate=gate)
                s_br = run(breaker, runner.attempt, role_name="breaker", target=target, n=breaker_attempts, stage="breaker",
                           label=f"attacks on candidate {ci + 1} (score {cand_row.score})", extra={"candidate_index": ci}, tool_calls_n=breaker_tool_calls)
                br_rows = ledger.attempts(group_id=s_br["group_id"])
                confirmed = [r.id for r in br_rows if r.passed]
                outside = sum(1 for r in br_rows if "outside the class" in ((r.certificate or {}).get("error") or ""))
                no_witness = sum(1 for r in br_rows if "witness" in ((r.certificate or {}).get("error") or "") and "outside" not in ((r.certificate or {}).get("error") or ""))
                entry = {"candidate_attempt_id": cand_row.id, "candidate_score": cand_row.score, "group_id": s_br["group_id"], "attempts": s_br["attempts"],
                         "confirmed_breaks": len(confirmed), "break_attempt_ids": confirmed, "outside_class": outside, "no_witness": no_witness,
                         "reward_mean": (len(confirmed) / len(br_rows)) if br_rows else None, "dollars": s_br["dollars"],
                         "dossier_id": (dossier or {}).get("id"), "dossier_valid": (dossier or {}).get("valid")}
                rung["breaker"].append(entry)
                cand_row._dossier = dossier  # type: ignore[attr-defined]
                if confirmed:
                    voided.add(cand_row.id)
                    record["exploit_alarms"].append({"target": tid, "candidate_attempt_id": cand_row.id, "breaks": confirmed,
                                                     "note": "a certified candidate was broken inside its class: the certifier or the class definition needs review"})
        # 7. the archive and the lesson bank
        rung["archived"] = []
        for r in survivors:
            if r.id in voided:
                continue
            dossier = getattr(r, "_dossier", None)
            if dossier is None:
                if checker is not None and breaker is None:
                    try:
                        dossier = checker.dossier(r, target)
                    except Exception as e:
                        dossier = {"claim": "", "conditions": [], "error": f"{type(e).__name__}: {e}"}
                else:
                    dossier = template_dossier(r, target)
            attacks = [b for b in rung["breaker"] if b["candidate_attempt_id"] == r.id]
            witness = {"attacks": sum(b["attempts"] for b in attacks), "confirmed_breaks": 0, "breaker_group_ids": [b["group_id"] for b in attacks],
                       "certifier": "witness_check"} if attacks else None
            rec = archive.archive_method(r, dossier=dossier, witness=witness, lesson_ids=(r.context or {}).get("lesson_ids") or [],
                                         citation_ids=(r.context or {}).get("citation_ids") or [], reading_snapshot=snapshot)
            rung["archived"].append({"artifact_id": rec["artifact_id"], "attempt_id": r.id, "score": r.score})
        rung["voided"] = sorted(voided)
        if harness is not None:
            try:
                lessons = harness.extract(target, base_rows, group_id=s_base["group_id"], detectors=kept)
                actions = bank.consolidate(lessons, token_budget, family=target.family)
                rung["lessons_extracted"] = [{"id": l.id, "level": l.level, "evidence": l.evidence_tag, "rule": l.rule, "detectors": l.detectors}
                                             for l in lessons]
                rung["consolidation"] = actions
            except Exception as e:
                rung["lessons_extracted"], rung["lesson_error"] = [], f"{type(e).__name__}: {e}"
        # 8. the gate memo from the base group
        gate_rec = gate_record(s_base)
        gate_rec["generation_id"] = gen_id
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        base_path = gates_dir / f"{tid}-t{gate_rec['thinking_budget']}-{stamp}"
        base_path.with_suffix(".json").write_text(json.dumps(gate_rec, indent=1, default=str), encoding="utf-8")
        base_path.with_suffix(".md").write_text(memo_markdown(gate_rec), encoding="utf-8")
        rung["gate"] = {"verdict": gate_rec["verdict"], "pass_rate": gate_rec["pass_rate"], "memo": str(base_path.with_suffix(".md"))}
        rung["wall_s"] = time.perf_counter() - t_rung
        rung["dollars"] = sum(g.get("dollars", 0.0) for g in rung["groups"].values()) + sum(b["dollars"] for b in rung["breaker"])
        record["rungs"].append(rung)
        if not quiet:
            print(f"  rung {rung_index} {tid}: gate {gate_rec['verdict']} (pass rate {gate_rec['pass_rate']:.3f}); survivors {len(survivors)}, "
                  f"voided {len(voided)}, archived {len(rung['archived'])}; ${rung['dollars']:.4f}")
    # the generation's end: the archive snapshot and the ledger pointer
    methods = []
    for tid in targets:
        methods += [m["artifact_id"] for m in archive.methods(tid)]
    methods = sorted(set(methods))
    record["archive_snapshot"] = {"artifact_ids": methods, "count": len(methods),
                                  "sha256": hashlib.sha256("\n".join(methods).encode("utf-8")).hexdigest()[:16]}
    last = ledger.query("SELECT id, timestamp FROM attempts ORDER BY timestamp DESC, id DESC LIMIT 1")
    record["ledger_pointer"] = {"path": str(Path(ledger.path).resolve()), "rows": ledger.count(),
                                "last_attempt_id": last[0]["id"] if last else None, "last_timestamp": last[0]["timestamp"] if last else None}
    gen_rows = ledger.query("SELECT role, COUNT(*) AS n, COALESCE(SUM(dollars), 0) AS dollars, COALESCE(SUM(api_dollars), 0) AS api_dollars, "
                            "COALESCE(SUM(gpu_seconds), 0) AS gpu_seconds, COALESCE(SUM(cpu_seconds), 0) AS cpu_seconds FROM attempts "
                            "WHERE json_extract(context, '$.generation_id') = ? GROUP BY role ORDER BY role", (gen_id,))
    record["cost"] = {"by_role": gen_rows, "dollars": sum(r["dollars"] for r in gen_rows), "api_dollars": sum(r["api_dollars"] for r in gen_rows),
                      "gpu_seconds": sum(r["gpu_seconds"] for r in gen_rows), "harness_dollars": harness.spent if harness else 0.0,
                      "checker_dollars": checker.spent if checker else 0.0, "wall_s": time.perf_counter() - t_gen}
    record["lesson_bank"] = {"active": [{"id": l.id, "level": l.level, "evidence": l.evidence_tag, "family": l.family, "targets": l.targets, "rule": l.rule,
                                         "detectors": l.detectors} for fam in families for l in bank.active(fam)]}
    record["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (gens_dir / f"{gen_id}.json").write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
    (gens_dir / f"{gen_id}.md").write_text(generation_markdown(record), encoding="utf-8")
    record["files"] = {"json": str(gens_dir / f"{gen_id}.json"), "markdown": str(gens_dir / f"{gen_id}.md")}
    if not quiet:
        print("\n" + generation_markdown(record))
    return record


def generation_markdown(r: dict) -> str:
    lines = [f"# Generation {r['id']}: {', '.join(r['targets'])}", "",
             f"Model {r['model']} ({r['model_version']}), precision {r['precision']}, GPU {r['gpu']}, provider {r['provider']} in mode {r['mode']}, "
             f"adapters: {r['adapters']}. Carrier {r['carrier']}; harness {r['harness'] or 'none'}; checker {'yes' if r['checker'] else 'no'}. "
             f"Started {r['started']}, finished {r.get('finished')}.", "",
             "| rung | target | gate verdict | base pass rate | optimal | lesson group | proposal group | theorist lemmas | survivors | breaks | archived | $ |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for g in r["rungs"]:
        b = g["groups"]["base"]
        les = g["groups"].get("lessons")
        prop = g["groups"].get("proposals")
        th = g["groups"].get("theorist")
        breaks = sum(x["confirmed_breaks"] for x in g.get("breaker", []))
        lines.append(f"| {g['rung']} | {g['target']} | {g['gate']['verdict']} | {b['pass_rate']:.3f} ({b['passed']}/{b['attempts']}) | {b['optimal']} | "
                     f"{(f'{les['pass_rate']:.3f} ({les['passed']}/{les['attempts']})') if les else '-'} | "
                     f"{(f'{prop['pass_rate']:.3f} ({prop['passed']}/{prop['attempts']})') if prop else '-'} | "
                     f"{(f'{th['lemmas_passed']}/{th['lemmas_proposed']} in {th['attempts']} attempts') if th else '-'} | "
                     f"{len(g['survivors'])} | {breaks} | {len(g['archived'])} | {g['dollars']:.4f} |")
    c = r.get("cost", {})
    lines += ["", f"Cost: ${c.get('dollars', 0):.4f} (api ${c.get('api_dollars', 0):.4f}; harness ${c.get('harness_dollars', 0):.4f}, "
                  f"checker ${c.get('checker_dollars', 0):.4f}); GPU {c.get('gpu_seconds', 0):.1f} s; wall {c.get('wall_s', 0):.1f} s.",
              f"Archive snapshot: {r['archive_snapshot']['count']} methods, sha {r['archive_snapshot']['sha256']}. "
              f"Ledger: {r['ledger_pointer']['path']} ({r['ledger_pointer']['rows']} rows, last {r['ledger_pointer']['last_attempt_id']}).",
              f"Lesson bank: {len(r.get('lesson_bank', {}).get('active', []))} active lessons."]
    if r.get("exploit_alarms"):
        lines += ["", "Exploit alarms: " + "; ".join(f"{a['target']}: candidate {a['candidate_attempt_id']} broken by {a['breaks']}" for a in r["exploit_alarms"])]
    det_lines = []
    for g in r["rungs"]:
        d = g.get("detectors") or {}
        if not d.get("kept"):
            continue
        parts = []
        for f in d["firing"]:
            s = f"{f['name']} {f['fires']}/{d['attempts']}"
            les = (g.get("symptoms") or {}).get("lessons", {}).get(f["name"])
            if les:
                s += f" -> {les['fires']}/{les['n']} with lessons"
            parts.append(s)
        det_lines.append(f"- {g['target']} ({d['mining']} mining, {d['proposed']} proposed, query \"{g.get('retrieval_query', '')}\"): " + "; ".join(parts))
    if det_lines:
        lines += ["", "Detectors kept per rung (fires on the base group, then on the lesson group where one ran; the loop uses the family's "
                      "own lessons by design, so these symptom rates are recorded, never reported as transfer):"] + det_lines
    for g in r["rungs"]:
        if g.get("delayed_credit"):
            lines += ["", f"Delayed credit on {g['target']} (best score among builds derived from each proposal minus the base group's best; "
                          "positive is better):"]
            for d in g["delayed_credit"]:
                lines.append(f"- {d['proposal_id']}: {d['builds_passed']}/{d['builds']} builds certified, best {d['best_score']} vs baseline "
                             f"{d['baseline_best']}, credit {d['credit']}, pass-rate gain {d['pass_rate_gain']}")
    lines += ["", "Every score above is a certificate; lessons, dossiers and proposals are logged and never scored."]
    if r["mode"] == "offline":
        lines.append("Offline replay of recorded fixtures: a machinery check, not a measurement of the model.")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m lab.loop", description=__doc__.split("\n\n")[0])
    ap.add_argument("--targets", nargs="+", required=True, help="the curriculum rungs, in order")
    ap.add_argument("--attempts", type=int, default=64, help="builder attempts of the gate group per target")
    ap.add_argument("--lesson-attempts", type=int, default=None, help="builder attempts with lessons (default: --attempts; 0 skips)")
    ap.add_argument("--theorist-attempts", type=int, default=16)
    ap.add_argument("--proposal-attempts", type=int, default=16, help="builder attempts derived from the theorist's proposals")
    ap.add_argument("--breaker-attempts", type=int, default=8, help="breaker attempts per surviving candidate")
    ap.add_argument("--breaker-candidates", type=int, default=4)
    ap.add_argument("--carrier", default="lessons", choices=list(CARRIERS))
    ap.add_argument("--tool-calls", type=int, default=0, help="certifier dry runs the builder gets before its final answer")
    ap.add_argument("--theorist-tool-calls", type=int, default=1, help="revision rounds after the lemma checks")
    ap.add_argument("--breaker-tool-calls", type=int, default=1, help="probe rounds before the witness")
    ap.add_argument("--provider", default="local", choices=list(KINDS))
    ap.add_argument("--mode", default=None, choices=list(MODES))
    ap.add_argument("--harness-provider", default=None, help="openai | anthropic for lessons and dossiers (default: config)")
    ap.add_argument("--harness-mode", default=None, choices=list(MODES))
    ap.add_argument("--no-harness", action="store_true", help="no lesson extraction (search only; the notebook text stands in for the gate)")
    ap.add_argument("--no-checker", action="store_true", help="dossiers written by the program's template instead of the checker")
    ap.add_argument("--no-mine", action="store_true", help="no detector mining on the base groups (the diagnosis then names nothing)")
    ap.add_argument("--ledger", default=None)
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--precision", default=None)
    ap.add_argument("--units-per-hour", type=float, default=None)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--thinking", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--judge-concurrency", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--token-budget", type=int, default=None)
    ap.add_argument("--out", default=None, help="directory for gates/ and generations/ (default: the ledger's directory)")
    ap.add_argument("--fixtures", default=None)
    ap.add_argument("--generation", default=None, help="generation id (default: a new one)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    cfg = lab_config()
    ledger_path = args.ledger or cfg["storage"]["ledger"]
    ledger = Ledger(ledger_path)
    root = Path(ledger_path).resolve().parent
    archive, bank = Archive(ledger, root), LessonBank(ledger)
    harness = checker = None
    if not args.no_harness:
        hp = harness_provider(cfg, name=args.harness_provider, mode=args.harness_mode, ledger=ledger, fixtures_root=args.fixtures)
        harness = LessonHarness(hp, ledger, bank, cfg=cfg)
        if not args.no_checker:
            checker = Checker(hp, ledger, cfg=cfg)
    rec = run_generation(targets=args.targets, attempts=args.attempts, cfg=cfg, ledger=ledger, archive=archive, bank=bank, harness=harness,
                         checker=checker, provider_name=args.provider, mode=args.mode, carrier=args.carrier, lesson_attempts=args.lesson_attempts,
                         theorist_attempts=args.theorist_attempts, proposal_attempts=args.proposal_attempts, breaker_attempts=args.breaker_attempts,
                         breaker_candidates=args.breaker_candidates, tool_calls=args.tool_calls, theorist_tool_calls=args.theorist_tool_calls,
                         breaker_tool_calls=args.breaker_tool_calls, gpu=args.gpu, precision=args.precision, units_per_hour=args.units_per_hour,
                         max_tokens=args.max_tokens, thinking_budget=args.thinking, timeout_s=args.timeout, concurrency=args.concurrency,
                         judge_concurrency=args.judge_concurrency, fixtures_root=args.fixtures, seed=args.seed, token_budget=args.token_budget,
                         out_dir=args.out, generation_id=args.generation, quiet=args.json, mine_detectors=not args.no_mine)
    ledger.close()
    if args.json:
        print(json.dumps(rec, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
