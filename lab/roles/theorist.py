"""The theorist, the proposer (Work package 4): given the target, its scoreboard entry, the lessons the gate chose
and the reading-list excerpts the librarian returns under the target's cutoff, it proposes lemmas with
computer-algebra checks, a structure for the method and hypotheses for the builder. Tools: the lemma checker
(SymPy in the judge's sandbox, lab/certifiers/lemma.py), retrieval (the librarian's lexical ranker under the cutoff,
every document a citation row) and the scoreboard entry, which is in the prompt. The immediate reward is the
lemma-check certificate: pass when at least one valid check passes, the score the fraction of lemmas that pass.
The delayed credit, the best certificate among builds derived from the proposal minus the frozen baseline, is
computed by the loop at the end of a generation (lab/loop.py). With `tool_calls` >= 1 the check results go back to
the model for a revision under its own, smaller token cap (`revision_max_tokens`); the final proposal is the one scored.
The lemma file gets a time budget below the sandbox cap, shared among its checks, so one slow check cannot stall the rest.
"""
from __future__ import annotations

import json
import time
from typing import Any

from ..certifiers.lemma import (SANDBOX_ALLOW_ATTRS, SANDBOX_ALLOW_MODULES, LemmaCheckCertifier, check_budget_s, lemma_program,
                                validate_lemmas)
from ..episode import JudgeGate, Outcome, judge_sandbox_cpu
from ..harness.base import parse_json
from ..judge import Judge
from ..targets import Target
from .base import Role, strip_thinking

THEORIST_SYSTEM = ("You are a theorist in an automated numerical-analysis lab. You propose lemmas that a computer-algebra "
                   "program can check, the structure a method should have, and hypotheses for a builder who writes the code. "
                   "You never claim a check passed; the program decides. Answer with JSON only.")

CHECK_RULES = ("Each check is ONE SymPy expression (no statements, no imports, no assignments) over the real symbols x, y, z, t, u, v "
               "and the integer symbols n, k, m, i, j, using SymPy functions such as integrate, diff, Sum, summation, solve, roots, "
               "real_roots, legendre, Poly, Matrix, det, expand, factor, simplify, Rational, sqrt, pi, Eq, binomial, factorial, "
               "and the methods .simplify(), .expand(), .equals(), .subs(), .doit(), .evalf(). The check passes when it evaluates "
               "to True or simplifies to 0 (write an identity as lhs - rhs, or Eq(lhs, rhs)). A check must call at least one symbolic "
               "operation and mention a symbol: a bare constant is not a lemma. Example: {\"statement\": \"the integral of x^2 over "
               "[-1, 1] is 2/3\", \"check\": \"integrate(x**2, (x, -1, 1)) - Rational(2, 3)\"}.")

THEORIST_PROMPT = """Target {target_id}: {title}.
Certificate: {certifier}. Scoreboard entry: best known value {best_known} ({status}); the score is {score_desc}.
The task the builder will be given:
---
{task_prompt}
---
{lessons}{reading}
Propose, for this target:
1. up to {max_lemmas} lemmas that a computer-algebra check can verify and that would help a builder construct or verify a solution (moments of monomials, orthogonality, symmetry of orbits, counts of unknowns against equations, root locations, order conditions, ...);
2. the structure a good method should have (symmetry, orbits, node count, parameters);
3. hypotheses for the builder (at most 3).
{check_rules}
Reply with JSON: {{"lemmas": [{{"statement": "...", "check": "..."}}], "structure": "...", "hypotheses": ["..."]}}"""

REVISION_PROMPT = ("Lemma check results (by the program; a check is a pass or fail program, the statement's wording does not count):\n"
                   "{results}\n\nRevise: keep the lemmas whose checks passed, fix or drop the others (a failed check may mean the "
                   "lemma is false), and reply with the complete JSON again in the same format.")


def score_description(target: Target) -> str:
    return {"quadrature": "the number of nodes (lower is better)", "first_order": "the certified worst-case rate (lower is better)",
            "order": "the measured order of convergence (higher is better)", "opcount": "the operation count (lower is better)",
            "time_integrator": "the stability extent or SSP coefficient (higher is better)"}.get(target.family, "the certificate's score")


def reading_text(docs: list[dict]) -> str:
    if not docs:
        return ""
    cards = "\n".join(f"- [{d['id']}] {d.get('year')}: {d.get('title')}. {d.get('summary', '')}" for d in docs)
    return f"\nReading-list excerpts the librarian returned under the target's cutoff (cite ids in square brackets):\n{cards}\n"


def theorist_prompt(target: Target, *, lessons_text: str = "", docs: list[dict] | None = None, max_lemmas: int = 5) -> str:
    lessons = f"\nLessons chosen for this target:\n{lessons_text}\n" if lessons_text.strip() else ""
    return THEORIST_PROMPT.format(target_id=target.id, title=target.title, certifier=target.certifier_name,
                                  best_known=target.best_known, status=target.status or "unknown", score_desc=score_description(target),
                                  task_prompt=target.prompt, lessons=lessons, reading=reading_text(docs or []), max_lemmas=max_lemmas,
                                  check_rules=CHECK_RULES)


def parse_proposal(text: str | None) -> tuple[dict, str | None]:
    """The proposal object in a reply, with the lemma list validated by the program."""
    try:
        data = parse_json(text or "")
    except ValueError as e:
        return {"lemmas": [], "structure": "", "hypotheses": [], "raw_lemmas": []}, f"no JSON proposal: {e}"
    if not isinstance(data, dict):
        return {"lemmas": [], "structure": "", "hypotheses": [], "raw_lemmas": []}, "the proposal must be a JSON object"
    lemmas = validate_lemmas(data.get("lemmas"))
    hyps = [" ".join(str(h).split())[:300] for h in (data.get("hypotheses") or []) if str(h).strip()][:3] \
        if isinstance(data.get("hypotheses"), list) else []
    return {"lemmas": lemmas, "structure": " ".join(str(data.get("structure", "")).split())[:600], "hypotheses": hyps,
            "raw_lemmas": data.get("lemmas") if isinstance(data.get("lemmas"), list) else []}, None


def results_text(lemmas: list[dict]) -> str:
    lines = []
    for l in lemmas:
        status = l["status"]
        extra = f" ({l['error']})" if l.get("error") and status != "passed" else ""
        lines.append(f"- lemma {l['index'] + 1}: {status}{extra}: {l['statement'][:120]}")
    return "\n".join(lines) if lines else "- no lemmas were found in the reply"


class TheoristRunner:
    """One theorist attempt: the prompt, the proposal, the lemma checks in the sandbox, `tool_calls` revision rounds."""

    def __init__(self, role: Role, target: Target, judge: Judge, *, prompt: str | None = None, tool_calls: int = 0,
                 judge_gate: JudgeGate | None = None, context: dict | None = None, max_lemmas: int = 5, timeout_s: float | None = None,
                 revision_max_tokens: int | None = None):
        self.role = role
        self.target = target
        self.judge = judge
        self.prompt = prompt if prompt is not None else theorist_prompt(target, max_lemmas=max_lemmas)
        self.tool_calls = max(0, int(tool_calls))
        self.gate = judge_gate or JudgeGate(4)
        self.context = dict(context or {})
        self.certifier = LemmaCheckCertifier()
        self.timeout_s = timeout_s
        self.budget_s = check_budget_s(timeout_s or getattr(judge, "timeout_s", None))
        self.revision_max_tokens = int(revision_max_tokens) if revision_max_tokens else None

    def check(self, proposal: dict, seed: int) -> tuple[Any, Any, float]:
        """Run the valid lemma checks in the sandbox with SymPy allowed; returns (certificate, sandbox, cpu)."""
        lemmas = proposal["lemmas"]
        if not any(l["status"] == "valid" for l in lemmas):
            cert = self.certifier.certify([], target=self.target.id, seed=seed, lemmas=lemmas)
            return cert, None, 0.0
        src = lemma_program(lemmas, budget_s=self.budget_s)
        with self.gate:
            cpu0 = time.process_time()
            sb = self.judge.run_sandbox(src, "check", timeout_s=self.timeout_s, allow_modules=SANDBOX_ALLOW_MODULES,
                                        allow_attrs=SANDBOX_ALLOW_ATTRS)
            cpu = judge_sandbox_cpu(sb, cpu0)
        cert = self.certifier.certify(sb.result if sb.ok else None, target=self.target.id, seed=seed, lemmas=lemmas)
        if not sb.ok and sb.error:
            cert.details["sandbox_error"] = f"{sb.error.get('type')}: {str(sb.error.get('message', ''))[:300]}"
        return cert, sb, cpu

    def attempt(self, index: int, seed: int) -> Outcome:
        messages = self.role.build_messages(self.prompt)
        res = self.role.attempt_messages(messages, seed=seed)
        completions, tool_log, outputs = [res.completion], list(res.tool_log), [res.completion.text]
        proposal, perr = parse_proposal(res.completion.text)
        cert, sb, judge_cpu = self.check(proposal, seed)
        tool_log.append({"tool": "lemma_check", "round": 1, "proposed": cert.details.get("proposed"), "passed": cert.details.get("passed"),
                         "failed": cert.details.get("failed"), "rejected": cert.details.get("rejected"), "timed_out": cert.details.get("timed_out"),
                         "skipped": cert.details.get("skipped"), "runtime_s": cert.runtime_s})
        for r in range(self.tool_calls):
            feedback = REVISION_PROMPT.format(results=results_text(cert.details.get("lemmas", [])))
            messages = messages + [{"role": "assistant", "content": strip_thinking(res.completion.text)}, {"role": "user", "content": feedback}]
            res = self.role.attempt_messages(messages, seed=seed, max_tokens=self.revision_max_tokens)
            completions.append(res.completion)
            tool_log += res.tool_log
            outputs.append(res.completion.text)
            proposal, perr = parse_proposal(res.completion.text)
            cert, sb, cpu = self.check(proposal, seed)
            judge_cpu += cpu
            tool_log.append({"tool": "lemma_check", "round": r + 2, "proposed": cert.details.get("proposed"), "passed": cert.details.get("passed"),
                             "failed": cert.details.get("failed"), "rejected": cert.details.get("rejected"), "timed_out": cert.details.get("timed_out"),
                             "skipped": cert.details.get("skipped"), "runtime_s": cert.runtime_s})
        if perr:   # the reply held no proposal: that is the reason, not the empty lemma list it implies
            cert.error = perr
        final = {"lemmas": cert.details.get("lemmas", []), "structure": proposal["structure"], "hypotheses": proposal["hypotheses"],
                 "passed": cert.details.get("passed", 0), "proposed": cert.details.get("proposed", 0)}
        return Outcome(seed=seed, index=index, certificate=cert, output=res.completion.text, tool_log=tool_log, completions=completions,
                       prompt_hash=res.prompt_hash, code=None, sandbox=sb, judge_cpu=judge_cpu, context=dict(self.context),
                       meta={"proposal": final, "lemma_program": lemma_program(final["lemmas"], budget_s=self.budget_s) if final["lemmas"] else None},
                       round_outputs=outputs)


def proposal_from_row(row) -> dict | None:
    """The proposal a theorist row carries, with the row's id, for the builder's prompt and the delayed credit."""
    p = (row.meta or {}).get("proposal")
    if not p:
        return None
    return {"id": row.id, "target": row.target, "seed": row.seed, **p}
