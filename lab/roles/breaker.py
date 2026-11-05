"""The breaker (Work package 4): given a certified candidate with its dossier and the class definition, it builds an
instance or input and submits a witness; the reward is one for a break the witness check confirms (the
`witness_check` certifier, lab/certifiers/witness.py) and zero for anything else, including an instance outside the
class definition. Tools: an instance generator (the class definition and the witness schema in the prompt), execution
of the candidate (the program runs the certified code once in the sandbox and, with `tool_calls` >= 1, evaluates the
rule on the monomials the model asks to probe), and witness submission (the final JSON). Nothing the breaker writes
is scored; the certifier re-runs the claimed break on exactly the submitted input.

On the quadrature family a certificate checks the whole monomial basis of the class, so a confirmed break against a
certified candidate means the certifier is wrong: the breaker's confirmed breaks are an exploit alarm (Work package 7),
and a candidate with a confirmed break never enters the archive.
"""
from __future__ import annotations

import json
import time
from typing import Any

from ..certifiers.quadrature import DIM, apply_rule, monomial_exact, normalize_rule
from ..certifiers.witness import WitnessCertifier
from ..episode import JudgeGate, Outcome, judge_sandbox_cpu
from ..harness.base import parse_json
from ..judge import Judge
from ..targets import Target
from .base import Role, strip_thinking

BREAKER_SYSTEM = ("You are the breaker in an automated numerical-analysis lab. Given a certified candidate and the class it is "
                  "claimed for, you construct an instance inside that class on which the candidate fails, and you submit it as a "
                  "witness. You are paid only for breaks the program confirms; an instance outside the class pays nothing. "
                  "Answer with JSON only.")

SCHEMAS = {
    "quadrature": ("the witness is {{\"kind\": \"exactness_error\", \"input\": {{\"exponents\": [a{dims}]}}}} for one monomial, or "
                   "{{\"kind\": \"exactness_error\", \"input\": {{\"coefficients\": {{\"a{dims_key}\": c, ...}}}}}} for a polynomial "
                   "(keys are comma-separated exponents, |c| <= 1), or {{\"kind\": \"exception\", \"input\": {{\"exponents\": [0{zeros}]}}}} "
                   "if evaluating the rule raises"),
    "first_order": "the witness is {{\"kind\": \"rate_violation\", \"curvature\": c}} with c in [mu, L]: the quadratic f(x) = c/2 x^2",
}

BREAKER_PROMPT = """Target {target_id}: {title}. Class definition: {class_def}.

The certified candidate (attempt {attempt_id}):
```python
{code}
```
Certificate {certifier} {version}: PASS, score {score}{facts}.{dossier}

The witness check re-runs the candidate on exactly the input you submit and confirms a break only if the candidate fails on it under the certifier's own criterion ({criterion}). Confirmed break: reward 1. Anything else, including an input outside the class, a malformed witness or no witness: reward 0.
{probe}
Witness format: {schema}. Reply with JSON: {{"witness": {{...}}, "reason": "one sentence"}}{probe_format}"""

PROBE_TEXT = ("\nYou may first probe the rule: reply {\"probe\": {\"exponents\": [[a, b], ...]}} with up to 8 monomials and the program "
              "returns the rule's error on each; then submit the witness in your next reply.")


def certificate_facts(cert: dict) -> str:
    d = (cert or {}).get("details", {}) or {}
    facts = []
    for k in ("node_count", "weight_sum", "volume", "measured_exactness_degree", "all_weights_positive", "all_nodes_interior",
              "max_monomial_error", "measured_order", "total_ops", "ssp_coefficient", "rate"):
        if k in d and d[k] is not None:
            v = d[k]
            if k == "max_monomial_error":
                # below the tolerance the value is this machine's rounding noise: naming it would make the prompt
                # (and the fixture key) platform-dependent
                facts.append("max_monomial_error<1e-9" if v <= 1e-9 else f"max_monomial_error={v:.3g}")
            else:
                facts.append(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}")
    return ("; " + ", ".join(facts)) if facts else ""


def criterion_text(target: Target, tol: float) -> str:
    if target.family == "quadrature":
        return f"|rule(p) - exact integral| > {tol:g} for a monomial, times the number of terms for a polynomial"
    if target.family == "first_order":
        return "the observed contraction ratio on the quadratic exceeds the certified rate"
    return "the certifier's own failure criterion"


def breaker_prompt(target: Target, candidate: dict, *, tool_calls: int = 0, tol: float = 1e-9) -> str:
    cert = candidate.get("certificate") or {}
    dossier = candidate.get("dossier") or {}
    dtext = ""
    if dossier.get("claim"):
        dtext = f"\nDossier claim: {dossier['claim']}" + (f" Conditions: {'; '.join(dossier.get('conditions', []))}." if dossier.get("conditions") else "")
    dims = DIM.get(target.params.get("domain", "interval"), 1) if target.family == "quadrature" else 1
    schema = SCHEMAS.get(target.family, "the witness is {\"kind\": \"...\", \"input\": {...}}")
    if target.family == "quadrature":
        schema = schema.format(dims=", b" if dims == 2 else (", b, c" if dims == 3 else ""), dims_key=",b" if dims == 2 else (",b,c" if dims == 3 else ""),
                               zeros=", 0" if dims == 2 else (", 0, 0" if dims == 3 else ""))
    probe = PROBE_TEXT if (tool_calls > 0 and target.family == "quadrature") else ""
    score = cert.get("score")
    return BREAKER_PROMPT.format(target_id=target.id, title=target.title, class_def=target.class_definition,
                                 attempt_id=candidate.get("label", candidate.get("attempt_id", "?")), code=(candidate.get("code") or "").strip(),
                                 certifier=cert.get("certifier", target.certifier_name), version=cert.get("version", ""),
                                 score=f"{score:g}" if isinstance(score, (int, float)) else score, facts=certificate_facts(cert),
                                 dossier=dtext, criterion=criterion_text(target, tol), probe=probe, schema=schema,
                                 probe_format=" (or the probe first)" if probe else "")


def parse_witness(text: str | None) -> tuple[dict | None, dict | None, str | None]:
    """(witness, probe, error) from a reply."""
    try:
        data = parse_json(text or "")
    except ValueError as e:
        return None, None, f"no JSON witness: {e}"
    if not isinstance(data, dict):
        return None, None, "the reply must be a JSON object"
    if isinstance(data.get("witness"), dict):
        return data["witness"], None, None
    if isinstance(data.get("probe"), dict):
        return None, data["probe"], None
    if "kind" in data and ("input" in data or "curvature" in data):
        return data, None, None
    return None, None, "the reply holds neither a witness nor a probe"


def complete_witness(target: Target, witness: dict, candidate: dict | None = None) -> dict:
    """The class is the target's: the program fills family, domain, degree and the claimed rate (the candidate's
    certified score); the model supplies the instance (kind and input, or the curvature)."""
    w = {"kind": str(witness.get("kind", "")).strip(), "input": witness.get("input") if isinstance(witness.get("input"), dict) else {}}
    if target.family == "quadrature":
        w.update({"family": "quadrature", "domain": target.params["domain"], "degree": int(target.params["degree"])})
    elif target.family == "first_order":
        claimed = ((candidate or {}).get("certificate") or {}).get("score")
        w.update({"family": "first_order", "curvature": witness.get("curvature"), "claimed_rate": claimed})
    else:
        w.update({"family": target.family})
    return w


def probe_text(target: Target, rule_result: Any, probe: dict, tol: float) -> str:
    """The execution tool: the rule's error on each monomial the model asked about (at most 8, inside the class)."""
    domain, degree = target.params["domain"], int(target.params["degree"])
    try:
        nodes, weights = normalize_rule(rule_result, domain)
    except Exception as e:
        return f"Probe: the rule could not be evaluated ({type(e).__name__}: {e}). Submit your witness now."
    lines = []
    for ex in (probe.get("exponents") or [])[:8]:
        try:
            a = tuple(int(v) for v in ex)
        except (TypeError, ValueError):
            lines.append(f"- {ex}: malformed exponents")
            continue
        if len(a) != DIM[domain] or any(v < 0 for v in a):
            lines.append(f"- {list(a)}: outside the class (dimension or sign)")
            continue
        if sum(a) > degree:
            lines.append(f"- {list(a)}: outside the class (degree {sum(a)} > {degree})")
            continue
        got, exact = apply_rule(nodes, weights, a), monomial_exact(domain, a)
        lines.append(f"- {list(a)}: rule {got:.12g}, exact {exact:.12g}, error {abs(got - exact):.3e} ({'above' if abs(got - exact) > tol else 'below'} tol {tol:g})")
    body = "\n".join(lines) if lines else "- no monomials given"
    return f"Probe results:\n{body}\n\nSubmit your witness now as JSON: {{\"witness\": {{...}}, \"reason\": \"...\"}}."


class BreakerRunner:
    """One breaker attempt against one certified candidate: the prompt, an optional probe round, the witness check."""

    def __init__(self, role: Role, target: Target, judge: Judge, candidate: dict, *, tool_calls: int = 0,
                 judge_gate: JudgeGate | None = None, context: dict | None = None, tol: float = 1e-9):
        self.role = role
        self.target = target
        self.judge = judge
        self.candidate = candidate
        self.tool_calls = max(0, int(tool_calls))
        self.gate = judge_gate or JudgeGate(4)
        self.context = dict(context or {})
        self.tol = tol
        self.certifier = WitnessCertifier(tol=tol)
        self.prompt = breaker_prompt(target, candidate, tool_calls=self.tool_calls, tol=tol)
        self._rule = None
        self._rule_cpu = 0.0
        self._rule_sb = None

    def candidate_result(self) -> tuple[Any, float]:
        """Execution of the candidate (once per candidate, in the sandbox): the value its entry point returns."""
        if self._rule is None and self._rule_sb is None:
            with self.gate:
                cpu0 = time.process_time()
                args = self.target.entry_args_for(0)
                sb = self.judge.run_sandbox(self.candidate.get("code") or "", self.target.entry_point, args=args,
                                            instrument=getattr(self.target, "instrument", None))
                self._rule_cpu = judge_sandbox_cpu(sb, cpu0)
            self._rule_sb = sb
            self._rule = sb.result if sb.ok else None
            if self.target.family == "first_order" and isinstance(self._rule, dict):
                self._rule = {**self._rule, "n": int((self.target.params.get("params") or {}).get("n", 1))}
            return self._rule, self._rule_cpu
        return self._rule, 0.0

    def attempt(self, index: int, seed: int) -> Outcome:
        messages = self.role.build_messages(self.prompt)
        res = self.role.attempt_messages(messages, seed=seed)
        completions, tool_log, outputs = [res.completion], list(res.tool_log), [res.completion.text]
        rule, judge_cpu = self.candidate_result()
        witness, probe, err = parse_witness(res.completion.text)
        rounds = 0
        while witness is None and probe is not None and rounds < self.tool_calls:
            rounds += 1
            feedback = probe_text(self.target, rule, probe, self.tol) if self.target.family == "quadrature" else \
                "Probing is not available for this family. Submit your witness now."
            tool_log.append({"tool": "execute_candidate", "round": rounds, "probe": probe, "feedback": feedback[:600]})
            messages = messages + [{"role": "assistant", "content": strip_thinking(res.completion.text)}, {"role": "user", "content": feedback}]
            res = self.role.attempt_messages(messages, seed=seed)
            completions.append(res.completion)
            tool_log += res.tool_log
            outputs.append(res.completion.text)
            witness, probe, err = parse_witness(res.completion.text)
        if witness is None:
            cert = self.certifier.certify(rule, target=self.target.id, seed=seed, witness=None)
            cert.error = err or ("a probe was submitted where a witness was due" if probe is not None else "no witness in the output")
            cert.score = 0.0
            full = None
        else:
            full = complete_witness(self.target, witness, self.candidate)
            cert = self.certifier.certify(rule, target=self.target.id, seed=seed, witness=full)
        tool_log.append({"tool": "witness_check", "reproduces": bool(cert.passed), "error": cert.error, "runtime_s": cert.runtime_s})
        context = {**self.context, "candidate_attempt_id": self.candidate.get("attempt_id"), "candidate_code_sha": self.candidate.get("code_sha"),
                   "candidate_artifact_id": self.candidate.get("artifact_id")}
        meta = {"witness": full, "witness_reward": 1.0 if cert.passed else 0.0, "candidate_ran": rule is not None,
                "probe_rounds": rounds}
        if self._rule_sb is not None and not self._rule_sb.ok:
            meta["candidate_error"] = self._rule_sb.error
        return Outcome(seed=seed, index=index, certificate=cert, output=res.completion.text, tool_log=tool_log, completions=completions,
                       prompt_hash=res.prompt_hash, code=None, sandbox=self._rule_sb, judge_cpu=judge_cpu, context=context, meta=meta,
                       round_outputs=outputs)


def candidate_from_row(row, dossier: dict | None = None, label: str | None = None) -> dict:
    """A certified attempt row as the breaker's input."""
    import hashlib

    code = (row.meta or {}).get("code") or row.output or ""
    return {"attempt_id": row.id, "target": row.target, "code": code, "certificate": row.certificate, "dossier": dossier,
            "code_sha": hashlib.sha256(code.encode("utf-8")).hexdigest()[:16], "label": label or row.id, "score": row.score}
