"""The builder (Work package 4): turns a target, or a theorist's proposal, into a Python file; the judge's
certificate is its score. Prompt and context: the target with its certificate definition, the lessons and skills
the gate chose, and, when an episode is derived from a proposal, the theorist's checked lemmas and unverified
structure and hypotheses. Tools: the sandbox that runs its code and a dry run of the certifier for self-checks,
which is reported back to the model and never scored; the fixed number of tool calls is the number of dry runs
(0 means one request, as the demo and the gate make it). Skills from the archive come as reference
implementations in the prompt (the context carrier `skills`, lab/harness/carriers.py).
"""
from __future__ import annotations

import time
from typing import Callable

from ..episode import JudgeGate, Outcome, diagnosis_text, judge_sandbox_cpu
from ..judge import Judge
from ..targets import Target
from .base import Role, extract_code, strip_thinking


def proposal_text(proposal: dict) -> str:
    """A theorist's proposal as the builder sees it: the lemmas whose checks passed, marked as checked; the
    structure and hypotheses, marked as unverified."""
    lines = ["A proposal from the theorist. Lemmas marked [checked] passed a computer-algebra check; the structure and the "
             "hypotheses are unverified suggestions."]
    for l in proposal.get("lemmas", []):
        if l.get("status") == "passed":
            lines.append(f"- Lemma [checked]: {l['statement']} (check: {l['check']})")
    if proposal.get("structure"):
        lines.append(f"- Structure [unverified]: {proposal['structure']}")
    for h in proposal.get("hypotheses", []) or []:
        lines.append(f"- Hypothesis [unverified]: {h}")
    return "\n".join(lines)


def with_proposal(prompt: str, proposal: dict | None) -> str:
    return prompt if not proposal else prompt + "\n\n" + proposal_text(proposal)


def dry_run_feedback(cert, sb, target: Target) -> str:
    """What the dry run tells the model (deterministic given the certificate): the verdict, the reason and the
    facts a candidate can act on. Nothing here is a score."""
    if cert.passed:
        head = f"Dry run of the certifier {cert.certifier} on your file (unscored): PASS, score {cert.score:g}"
        if cert.optimal:
            head += " (matches the best known value)"
        elif target.best_known is not None:
            head += f" (best known {target.best_known:g})"
    else:
        head = f"Dry run of the certifier {cert.certifier} on your file (unscored): FAIL: {diagnosis_text(cert, None)}"
    d = cert.details or {}
    facts = []
    for k in ("node_count", "weight_sum", "volume", "measured_exactness_degree", "max_monomial_error", "worst_monomial",
              "all_weights_positive", "all_nodes_interior", "measured_order", "total_ops", "ssp_coefficient", "rate"):
        if k in d and d[k] is not None:
            v = d[k]
            facts.append(f"{k}={v:.3e}" if isinstance(v, float) and k == "max_monomial_error" else f"{k}={v}")
    body = ("; ".join(facts)) if facts else ""
    if sb is not None and not sb.ok and sb.error:
        body = (body + "; " if body else "") + f"sandbox: {sb.error.get('type')}: {str(sb.error.get('message', ''))[:300]}"
    tail = ("\n\nRevise the file if the dry run failed or the score can improve, or keep it; reply with the final complete "
            "file in exactly one ```python code block.")
    return head + ("\n" + body if body else "") + tail


class BuilderRunner:
    """One builder attempt: the prompt (plus lessons, a proposal), `tool_calls` dry runs with revisions, the judge."""

    def __init__(self, role: Role, target: Target, judge: Judge, *, prompt: str | None = None, tool_calls: int = 0,
                 judge_gate: JudgeGate | None = None, post_check: Callable | None = None,
                 proposal_for: Callable[[int], dict | None] | None = None, context: dict | None = None):
        self.role = role
        self.target = target
        self.judge = judge
        self.prompt = prompt if prompt is not None else target.prompt
        self.tool_calls = max(0, int(tool_calls))
        self.gate = judge_gate or JudgeGate(4)
        self.post_check = post_check
        self.proposal_for = proposal_for
        self.context = dict(context or {})

    def prompt_for(self, index: int) -> tuple[str, dict | None]:
        proposal = self.proposal_for(index) if self.proposal_for else None
        return with_proposal(self.prompt, proposal), proposal

    def judge_code(self, code: str | None, seed: int):
        with self.gate:
            cpu0 = time.process_time()
            cert, sb = self.judge.judge(code, self.target, seed=seed)
            return cert, sb, judge_sandbox_cpu(sb, cpu0)

    def attempt(self, index: int, seed: int) -> Outcome:
        prompt, proposal = self.prompt_for(index)
        messages = self.role.build_messages(prompt)
        res = self.role.attempt_messages(messages, seed=seed)
        completions, tool_log, outputs = [res.completion], list(res.tool_log), [res.completion.text]
        judge_cpu = 0.0
        dry_runs = []
        for r in range(self.tool_calls):
            code = extract_code(res.completion.text) if res.completion.text else None
            cert, sb, cpu = self.judge_code(code, seed)
            judge_cpu += cpu
            feedback = dry_run_feedback(cert, sb, self.target)
            dry_runs.append({"round": r + 1, "passed": cert.passed, "score": cert.score, "optimal": cert.optimal,
                             "diagnosis": diagnosis_text(cert, None), "had_code": code is not None})
            tool_log.append({"tool": "dry_run", "round": r + 1, "certifier": cert.certifier, "passed": cert.passed, "score": cert.score,
                             "runtime_s": cert.runtime_s, "sandbox_wall_s": sb.wall_seconds if sb else None, "feedback": feedback[:600]})
            messages = messages + [{"role": "assistant", "content": strip_thinking(res.completion.text)},
                                   {"role": "user", "content": feedback}]
            res = self.role.attempt_messages(messages, seed=seed)
            completions.append(res.completion)
            tool_log += res.tool_log
            outputs.append(res.completion.text)
        output = res.completion.text
        code = extract_code(output) if output else None
        cert, sb, cpu = self.judge_code(code, seed)
        judge_cpu += cpu
        check = None
        if self.post_check is not None and sb is not None and sb.ok:
            try:
                check = self.post_check(sb.result, {"seed": seed, "target": self.target.id})
            except Exception as e:  # a failing check is information, never a crash
                check = {"ok": None, "error": f"{type(e).__name__}: {e}"}
        context = dict(self.context)
        meta = {}
        if proposal is not None:
            context["proposal_id"] = proposal.get("id")
            context["derived_from_proposal"] = True
        if dry_runs:
            meta["dry_runs"] = dry_runs
            meta["revised_after_dry_run"] = outputs[-1] != outputs[0]
        return Outcome(seed=seed, index=index, certificate=cert, output=output, tool_log=tool_log, completions=completions,
                       prompt_hash=res.prompt_hash, code=code, sandbox=sb, judge_cpu=judge_cpu, check=check, context=context,
                       meta=meta, round_outputs=outputs)
