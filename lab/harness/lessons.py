"""The lesson bank (MUSE's notebook, as the proposal adopts it): lessons at three levels, each a rule with
conditions and an evidence tag; consolidation under a fixed token budget by add, replace and retire; a gate
that selects one lesson for a target and writes a concrete check to apply at use time.

Levels: approach (what method or structure to use), implementation (how to write it so the certificate
passes), process (how to work: verify, check, test). Evidence tags: observed (the cited attempts exist in the
ledger), derived (from the task contract, no attempts cited), proposed (unverified; also the tag a lesson
falls back to when its cited attempts do not exist).

The language parts, extracting lessons from a group of attempts and choosing one lesson plus a check for a
target, call the harness provider; everything else is a program. Prompts label attempts by their order
(attempt 1, 2, ...), never by their ids, so a recorded fixture replays. A check is a Python expression over the
candidate's result, validated against a whitelist before it is ever evaluated, and its outcome is logged in the
attempt's meta, never scored.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any

from ..ledger import Attempt, Ledger
from .base import Harness
from .tables import ensure_all

LEVELS = ("approach", "implementation", "process")
EVIDENCE_TAGS = ("observed", "derived", "proposed")
_TAG_RANK = {"observed": 2, "derived": 1, "proposed": 0}


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def lesson_id(rule: str, level: str, family: str) -> str:
    return "les_" + hashlib.sha256(f"{level}|{family}|{' '.join(rule.split()).lower()}".encode("utf-8")).hexdigest()[:12]


def fmt(x: Any) -> str:
    """Numbers in prompts rounded to three significant digits (platform-stable fixture keys)."""
    if isinstance(x, bool) or x is None:
        return str(x)
    if isinstance(x, (int,)):
        return str(x)
    if isinstance(x, float):
        if x == 0:
            return "0"
        if abs(x) < 1e-9:
            return "<1e-9"
        return f"{x:.3g}"
    return str(x)


@dataclass
class Lesson:
    id: str
    level: str
    rule: str
    conditions: list[str]
    evidence_tag: str
    support: list[str] = field(default_factory=list)          # attempt ids cited as evidence
    contradictions: list[str] = field(default_factory=list)
    family: str = ""
    targets: list[str] = field(default_factory=list)
    tokens: int = 0
    created: str = ""
    source_group: str | None = None
    retired: int = 0
    retired_at: str | None = None
    retired_reason: str | None = None
    replaced_by: str | None = None
    provenance: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        cond = ("; applies when: " + "; ".join(self.conditions)) if self.conditions else ""
        return f"[{self.level}] {self.rule}{cond}"

    def rank_key(self) -> tuple:
        return (_TAG_RANK.get(self.evidence_tag, 0), len(self.support), -self.tokens, self.id)

    @classmethod
    def from_row(cls, row: dict) -> "Lesson":
        d = {k: row.get(k) for k in cls.__dataclass_fields__}
        for k in ("conditions", "support", "contradictions", "targets"):
            d[k] = list(d.get(k) or [])
        d["provenance"] = dict(d.get("provenance") or {})
        d["retired"] = int(d.get("retired") or 0)
        d["tokens"] = int(d.get("tokens") or 0)
        return cls(**d)


class LessonBank:
    """The lessons table in the ledger plus the consolidation program."""

    def __init__(self, ledger: Ledger):
        self.ledger = ledger
        ensure_all(ledger)

    def add(self, lesson: Lesson) -> Lesson:
        if not lesson.created:
            lesson.created = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        lesson.tokens = lesson.tokens or estimate_tokens(lesson.text)
        self.ledger.insert("lessons", asdict(lesson), json_columns=("conditions", "support", "contradictions", "targets", "provenance"))
        return lesson

    def get(self, lid: str) -> Lesson | None:
        rows = self.ledger.rows("lessons", "id = ?", (lid,), json_columns=("conditions", "support", "contradictions", "targets", "provenance"))
        return Lesson.from_row(rows[0]) if rows else None

    def active(self, family: str | None = None, level: str | None = None) -> list[Lesson]:
        where, args = ["retired = 0"], []
        if family is not None:
            where.append("(family = ? OR level = 'process')")
            args.append(family)
        if level is not None:
            where.append("level = ?")
            args.append(level)
        rows = self.ledger.rows("lessons", " AND ".join(where), args,
                                json_columns=("conditions", "support", "contradictions", "targets", "provenance"), order="created, id")
        lessons = [Lesson.from_row(r) for r in rows]
        return sorted(lessons, key=lambda l: l.rank_key(), reverse=True)

    def all(self) -> list[Lesson]:
        return [Lesson.from_row(r) for r in self.ledger.rows("lessons", json_columns=("conditions", "support", "contradictions", "targets", "provenance"), order="created, id")]

    def retire(self, lid: str, reason: str, replaced_by: str | None = None) -> None:
        l = self.get(lid)
        if l is None:
            raise KeyError(lid)
        l.retired, l.retired_at, l.retired_reason, l.replaced_by = 1, datetime.now(timezone.utc).isoformat(timespec="milliseconds"), reason, replaced_by
        self.ledger.insert("lessons", asdict(l), json_columns=("conditions", "support", "contradictions", "targets", "provenance"))

    def total_tokens(self, family: str | None = None) -> int:
        return sum(l.tokens for l in self.active(family))

    @staticmethod
    def similarity(a: str, b: str) -> float:
        wa, wb = set(a.lower().split()), set(b.lower().split())
        return len(wa & wb) / max(1, len(wa | wb))

    def consolidate(self, candidates: list[Lesson], token_budget: int, *, family: str | None = None,
                    replace_threshold: float = 0.6) -> list[dict]:
        """Add, replace and retire under a token budget. A candidate that resembles an active lesson replaces it
        when its evidence is at least as good (the support is merged); otherwise it is added. Then the lowest-ranked
        active lessons are retired until the family's lessons fit the budget. Returns the log of actions."""
        log = []
        for cand in candidates:
            cand.tokens = cand.tokens or estimate_tokens(cand.text)
            active = self.active(family or cand.family or None)
            twin = None
            for l in active:
                if l.level == cand.level and self.similarity(l.rule, cand.rule) >= replace_threshold:
                    twin = l
                    break
            if twin is not None:
                if cand.rank_key()[:2] >= twin.rank_key()[:2]:
                    cand.support = sorted(set(cand.support) | set(twin.support))
                    cand.contradictions = sorted(set(cand.contradictions) | set(twin.contradictions))
                    if cand.id == twin.id:
                        self.add(cand)
                        log.append({"action": "replace", "lesson": cand.id, "replaced": twin.id, "reason": "same rule, merged support"})
                    else:
                        self.add(cand)
                        self.retire(twin.id, "replaced by a lesson with at least as much evidence", replaced_by=cand.id)
                        log.append({"action": "replace", "lesson": cand.id, "replaced": twin.id, "reason": f"similar rule (>= {replace_threshold:.0%} overlap)"})
                else:
                    log.append({"action": "skip", "lesson": cand.id, "kept": twin.id, "reason": "an active lesson with better evidence says the same"})
            else:
                self.add(cand)
                log.append({"action": "add", "lesson": cand.id, "reason": "new rule"})
        while True:
            active = self.active(family)
            total = sum(l.tokens for l in active)
            if total <= token_budget or not active:
                break
            victim = active[-1]  # lowest rank
            self.retire(victim.id, f"token budget {token_budget}: bank at {total} tokens")
            log.append({"action": "retire", "lesson": victim.id, "reason": f"over the token budget ({total} > {token_budget})"})
        return log

    def notebook_text(self, family: str, token_budget: int | None = None) -> tuple[str, list[str]]:
        """The lessons shown to a role (no gate): all active lessons of the family within the budget, in a stable order."""
        lessons = self.active(family)
        lines, ids, used = [], [], 0
        for l in sorted(lessons, key=lambda l: l.id):
            if token_budget is not None and used + l.tokens > token_budget:
                continue
            lines.append(f"- {l.text}")
            ids.append(l.id)
            used += l.tokens
        return "\n".join(lines), ids


# --- checks ------------------------------------------------------------------------------------------
_CHECK_NAMES = {"nodes", "weights", "n", "result", "abs", "sum", "min", "max", "len", "sqrt", "all", "any", "pi", "exp", "log",
                "cos", "sin", "round", "range", "float", "int", "True", "False", "None"}
_CHECK_NODES = (ast.Expression, ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.Compare, ast.Call, ast.Name, ast.Constant, ast.List,
                ast.Tuple, ast.Store, ast.ListComp, ast.GeneratorExp, ast.comprehension, ast.Subscript, ast.Slice, ast.IfExp, ast.Load,
                ast.And, ast.Or, ast.Not, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd, ast.Eq, ast.NotEq, ast.Lt,
                ast.LtE, ast.Gt, ast.GtE, ast.Mod, ast.FloorDiv)


def validate_expression(expr: str) -> tuple[bool, str | None]:
    """A check is a single expression over nodes, weights, n (or result) with arithmetic, comparisons and a few
    builtins; no attributes, no powers, no other names."""
    if not isinstance(expr, str) or not expr.strip() or len(expr) > 400:
        return False, "the check must be a non-empty expression of at most 400 characters"
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as e:
        return False, f"syntax error: {e}"
    bound = {t.id for node in ast.walk(tree) if isinstance(node, ast.comprehension)
             for t in ([node.target] if isinstance(node.target, ast.Name) else list(getattr(node.target, "elts", [])))
             if isinstance(t, ast.Name)}
    for node in ast.walk(tree):
        if not isinstance(node, _CHECK_NODES):
            return False, f"forbidden construct {type(node).__name__}"
        if isinstance(node, ast.Name) and node.id not in _CHECK_NAMES and node.id not in bound:
            return False, f"unknown name {node.id!r}"
        if isinstance(node, ast.Call) and not (isinstance(node.func, ast.Name) and node.func.id in _CHECK_NAMES):
            return False, "only the listed functions may be called"
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float, bool, type(None))):
            return False, "only numeric constants are allowed"
    return True, None


def apply_check(check: dict | None, result: Any) -> dict | None:
    """Evaluate a validated check on a candidate's result (nodes, weights for a quadrature rule; `result` otherwise)."""
    if not check or not check.get("valid"):
        return None
    ns: dict[str, Any] = {"abs": abs, "sum": sum, "min": min, "max": max, "len": len, "sqrt": math.sqrt, "all": all, "any": any,
                          "pi": math.pi, "exp": math.exp, "log": math.log, "cos": math.cos, "sin": math.sin, "round": round,
                          "range": range, "float": float, "int": int, "result": result, "nodes": None, "weights": None, "n": None}
    if isinstance(result, (list, tuple)) and len(result) == 2:
        ns["nodes"], ns["weights"] = result[0], result[1]
        try:
            ns["n"] = len(result[0])
        except TypeError:
            pass
    elif isinstance(result, dict):
        ns.update({k: v for k, v in result.items() if k in ("nodes", "weights", "n")})
    try:
        ok = bool(eval(compile(ast.parse(check["expression"].strip(), mode="eval"), "<check>", "eval"), {"__builtins__": {}}, ns))
        return {"ok": ok, "expression": check["expression"], "error": None}
    except Exception as e:
        return {"ok": None, "expression": check["expression"], "error": f"{type(e).__name__}: {e}"}


# --- the language parts ----------------------------------------------------------------------------------
def attempt_digest(label: str, a: Attempt, max_lines: int = 40) -> str:
    cert = a.certificate or {}
    d = cert.get("details", {}) or {}
    code = (a.meta or {}).get("code") or ""
    lines = code.strip("\n").splitlines()
    excerpt = "\n".join(lines[:max_lines]) + ("\n..." if len(lines) > max_lines else "")
    facts = []
    for k in ("node_count", "weight_sum", "volume", "all_weights_positive", "all_nodes_interior", "measured_exactness_degree",
              "measured_order", "total_ops", "ssp_coefficient", "rate"):
        if k in d:
            facts.append(f"{k}={fmt(d[k])}")
    if d.get("max_monomial_error") is not None:
        facts.append(f"max_monomial_error={fmt(d['max_monomial_error'])}")
    if d.get("worst_monomial") is not None:
        facts.append(f"worst_monomial={d['worst_monomial']}")
    status = "PASS" if a.passed else "FAIL"
    score = fmt(a.score) if a.score is not None else "-"
    diag = (a.diagnosis or "").replace("\n", " ")[:160]
    return (f"### {label}: {status}, score {score}" + (f", optimal" if cert.get("optimal") else "") +
            (f"; {' '.join(facts)}" if facts else "") + (f"\ndiagnosis: {diag}" if diag else "") +
            (f"\ncode:\n```python\n{excerpt}\n```" if excerpt else "\n(no code block in the output)"))


class LessonHarness(Harness):
    name = "lessons"
    system_prompt = ("You extract lessons for an automated numerical-analysis lab from its own attempts. A lesson is a short rule "
                     "with the conditions under which it applies. Never turn a detail of one attempt into a universal rule: cite "
                     "the attempts that support each lesson by their labels. Answer with JSON only.")

    def __init__(self, provider, ledger: Ledger, bank: LessonBank | None = None, *, cfg: dict | None = None, **kw):
        super().__init__(provider, ledger, cfg=cfg, **kw)
        self.bank = bank or LessonBank(ledger)

    def extract(self, target, attempts: list[Attempt], *, max_lessons: int = 5, group_id: str | None = None) -> list[Lesson]:
        """Lessons from a group of attempts on a target (the language part), validated and tagged by the program."""
        rows = sorted(attempts, key=lambda a: (a.seed if a.seed is not None else 0, a.id))
        labels = {f"attempt {i + 1}": a for i, a in enumerate(rows)}
        digest = "\n\n".join(attempt_digest(lab, a) for lab, a in labels.items())
        passed = sum(1 for a in rows if a.passed)
        user = (f"Target {target.id}: {target.title}.\nThe certificate: {target.certifier_name}; best known value {fmt(target.best_known)}.\n"
                f"Task prompt given to the model:\n---\n{target.prompt}\n---\n\n"
                f"{len(rows)} attempts, {passed} certified:\n\n{digest}\n\n"
                f"Write at most {max_lessons} lessons that would raise the certified rate next time, each at one level: "
                f"approach (what method or structure to use), implementation (how to write it so the certificate passes), or "
                f"process (how to work: verify, self-check). Each lesson: a rule of at most 40 words, one to three conditions under "
                f"which it applies, an evidence tag (observed: seen in the cited attempts; derived: follows from the task contract; "
                f"proposed: a guess), and the labels of the attempts that support it. Reply with JSON: "
                f"{{\"lessons\": [{{\"level\": \"...\", \"rule\": \"...\", \"conditions\": [\"...\"], \"evidence\": \"observed|derived|proposed\", "
                f"\"attempts\": [\"attempt 3\"]}}]}}")
        reply = self.ask("extract", user, target=target.id, context={"group_id": group_id, "n_attempts": len(rows), "labels": {k: v.id for k, v in labels.items()}})
        data = reply.data if isinstance(reply.data, dict) else {}
        out: list[Lesson] = []
        for item in (data.get("lessons") or [])[:max_lessons]:
            if not isinstance(item, dict):
                continue
            rule = " ".join(str(item.get("rule", "")).split())[:400]
            if not rule:
                continue
            level = str(item.get("level", "implementation")).lower().strip()
            if level not in LEVELS:
                level = "implementation"
            conds = [" ".join(str(c).split())[:200] for c in (item.get("conditions") or []) if str(c).strip()][:3]
            cited = [labels[str(x).strip().lower()].id for x in (item.get("attempts") or []) if str(x).strip().lower() in labels]
            claimed = str(item.get("evidence", "proposed")).lower().strip()
            if cited:
                tag = "observed"
            elif claimed == "derived":
                tag = "derived"
            else:
                tag = "proposed"
            lesson = Lesson(id=lesson_id(rule, level, target.family), level=level, rule=rule, conditions=conds, evidence_tag=tag,
                            support=cited, family=target.family, targets=[target.id], source_group=group_id,
                            provenance={"harness_attempt_id": reply.attempt_id, "provider": self.provider.name, "mode": self.provider.mode,
                                        "model": self.provider.model, "claimed_evidence": claimed})
            lesson.tokens = estimate_tokens(lesson.text)
            out.append(lesson)
        return out

    def gate(self, target, *, k_candidates: int = 6) -> dict:
        """Select one lesson for a target and write a concrete check for use time (the language part), with the
        program choosing the candidates, validating the choice and the check, and falling back to the top-ranked
        lesson when the model's choice is not a candidate."""
        candidates = self.bank.active(target.family)[:k_candidates]
        if not candidates:
            return {"lesson": None, "check": None, "candidates": [], "harness_attempt_id": None, "reason": "no active lessons"}
        listing = "\n".join(f"{i + 1}. id={l.id} level={l.level} evidence={l.evidence_tag} support={len(l.support)}: {l.text}"
                            for i, l in enumerate(candidates))
        variables = ("nodes (list of node coordinates), weights (list of floats), n (number of nodes)" if target.family == "quadrature"
                     else "result (the value the entry point returns)")
        user = (f"Target {target.id}: {target.title}. Certificate: {target.certifier_name}; entry point {target.entry_point}().\n"
                f"Task prompt:\n---\n{target.prompt}\n---\n\nActive lessons:\n{listing}\n\n"
                f"Choose the one lesson most likely to raise the certified rate on this target, say why in one sentence, and write a "
                f"concrete check to run on the candidate's returned value before it is submitted: a one-line description and a single "
                f"Python expression over {variables} using only arithmetic, comparisons, abs, sum, min, max, len, sqrt, all, any, pi "
                f"and list comprehensions (no powers, no imports, no attributes). Reply with JSON: "
                f"{{\"lesson_id\": \"les_...\", \"why\": \"...\", \"check\": {{\"description\": \"...\", \"expression\": \"...\"}}}}")
        reply = self.ask("gate", user, target=target.id, context={"candidates": [l.id for l in candidates]})
        data = reply.data if isinstance(reply.data, dict) else {}
        chosen_id = str(data.get("lesson_id", "")).strip()
        by_id = {l.id: l for l in candidates}
        fallback = chosen_id not in by_id
        lesson = by_id.get(chosen_id, candidates[0])
        chk = data.get("check") if isinstance(data.get("check"), dict) else {}
        expr = str(chk.get("expression", "")).strip()
        valid, err = validate_expression(expr) if expr else (False, "no expression")
        check = {"description": str(chk.get("description", ""))[:300], "expression": expr, "valid": bool(valid), "error": err,
                 "lesson_id": lesson.id}
        return {"lesson": lesson, "check": check, "why": str(data.get("why", ""))[:300], "candidates": [l.id for l in candidates],
                "harness_attempt_id": reply.attempt_id, "model_choice_valid": not fallback,
                "reason": "model choice" if not fallback else "fallback: the model named a lesson outside the candidates",
                "api_dollars": reply.api_dollars}

    @staticmethod
    def gate_text(gate: dict) -> str:
        lesson, check = gate.get("lesson"), gate.get("check") or {}
        if lesson is None:
            return ""
        text = f"One lesson chosen for this target: {lesson.text}"
        if check.get("valid") and check.get("description"):
            text += f"\nBefore returning, check: {check['description']} (in code: {check['expression']})."
        return text
