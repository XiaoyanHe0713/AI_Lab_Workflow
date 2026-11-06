"""Context carriers (proposal, "Transfer settings, carriers and controls"): how an experience set reaches a role's
prompt. `none` is the no-experience baseline; `lessons` puts the set's lessons in the prompt (the notebook);
`lessons_gate` lets the gate choose one lesson and write a check (a frontier call); `skills` offers the set's
archived, certified methods as reference implementations with their dossier claims; `process_only` is the control
that keeps the set's process-level lessons with their content redacted (digits and the family's content terms
replaced), isolating research craft from domain knowledge; `unrelated` applies another family's experience, which
separates content from the mere presence of experience. The ablation harness's three arms are `none`, `lessons`
and `lessons_gate` on one target; the matrix (lab/harness/matrix.py) runs any carrier from any experience set.
Two target-side controls (week 5) carry no experience at all: `diagnosis_only` puts the program's diagnosis of the
target's own baseline attempts in the prompt (the detectors that fired, with counts, and no remedy), which separates
knowing the fix from seeing the problem; `second_try` gives the builder one certifier dry run and a revision with no
lesson, the floor every lesson gain is read against (ADSD's self-improvement control).
Nothing a carrier adds to a prompt enters a reward: the certificate is the only score.
"""
from __future__ import annotations

import re
from typing import Any

from .lessons import Lesson, LessonBank, LessonHarness, apply_check, estimate_tokens

CARRIERS = ("none", "lessons", "lessons_gate", "skills", "process_only", "unrelated", "diagnosis_only", "second_try")
CONTENT_CARRIERS = ("lessons", "lessons_gate", "skills", "process_only", "unrelated")
CONTROL_CARRIERS = ("diagnosis_only", "second_try")      # target-side: no experience set, never refused by the transfer rule

_DIGITS = re.compile(r"\d+(?:\.\d+)?")


def redact(text: str, terms: list[str]) -> str:
    """Content redaction: digits and the family's content terms become [...]; adjacent gaps collapse."""
    out = _DIGITS.sub("[...]", text)
    if terms:
        pattern = re.compile(r"\b(" + "|".join(re.escape(t) for t in sorted(set(terms), key=len, reverse=True)) + r")\b", re.I)
        out = pattern.sub("[...]", out)
    out = re.sub(r"(\[\.\.\.\][\s-]*){2,}", "[...] ", out)   # "Gauss-Legendre rule" is one gap, not three
    return out.strip()


def process_only_lessons(lessons: list[Lesson], terms: list[str]) -> list[tuple[Lesson, str]]:
    """The process-level lessons of a set with their content redacted (rule and conditions)."""
    out = []
    for l in lessons:
        if l.level != "process":
            continue
        rule = redact(l.rule, terms)
        conds = [redact(c, terms) for c in l.conditions]
        text = f"[process] {rule}" + (("; applies when: " + "; ".join(conds)) if conds else "")
        out.append((l, text))
    return out


def lessons_block(entries: list[tuple[Lesson, str]], token_budget: int | None) -> tuple[str, list[str]]:
    """Lesson lines within the budget, in a stable order (by id), and the ids used."""
    lines, ids, used = [], [], 0
    for l, text in sorted(entries, key=lambda e: e[0].id):
        n = estimate_tokens(text)
        if token_budget is not None and used + n > token_budget:
            continue
        lines.append(f"- {text}")
        ids.append(l.id)
        used += n
    return "\n".join(lines), ids


def notebook_prompt(target, text: str) -> str:
    if not text.strip():
        return target.prompt + "\n\n(The notebook holds no lessons for this family yet.)"
    return target.prompt + "\n\nLessons from earlier attempts (the notebook):\n" + text


def gate_prompt(target, gate: dict) -> str:
    text = LessonHarness.gate_text(gate)
    if not text:
        return target.prompt + "\n\n(The gate selected no lesson for this target.)"
    return target.prompt + "\n\n" + text


def diagnosis_only_prompt(target, text: str) -> str:
    """The diagnosis-only control: the target's prompt plus the program's diagnosis of its baseline attempts, no remedy."""
    return target.prompt + "\n\n" + text


def skills_prompt(target, skills: list[dict]) -> str:
    if not skills:
        return target.prompt + "\n\n(No skills are available from the archive for this target.)"
    parts = []
    for s in skills:
        # labelled by the code's own hash: a method artifact id embeds the attempt id and would change the prompt per run
        head = f"Skill {s['code_sha'][:12]} (certified on {s['target']}, score {s['score']:g})"
        claim = f": {s['claim']}" if s.get("claim") else ""
        parts.append(f"{head}{claim}\n```python\n{s['code'].strip()}\n```")
    return (target.prompt + "\n\nSkills from the archive: certified methods from other targets, as reference implementations "
            "(their claims hold for their own targets, not for this one):\n\n" + "\n\n".join(parts))


def skills_from_archive(archive, artifact_ids: list[str], max_skills: int = 3, max_lines: int = 40) -> list[dict]:
    """The archived methods of an experience set as skills: code excerpt, target, score, dossier claim."""
    found = []
    for aid in artifact_ids:
        try:
            method = archive.get(aid)
            code = archive.get(method["code_sha"])["code"]
        except Exception:
            continue
        claim = None
        if method.get("dossier_sha"):
            try:
                claim = archive.get(method["dossier_sha"]).get("claim")
            except Exception:
                claim = None
        lines = code.strip("\n").splitlines()
        excerpt = "\n".join(lines[:max_lines]) + ("\n# ..." if len(lines) > max_lines else "")
        found.append({"artifact_id": aid, "code_sha": method["code_sha"], "target": method.get("target"), "score": float(method.get("score") or 0.0),
                      "claim": claim, "code": excerpt})
    # a stable order that does not depend on when a method was archived: by target, score, then the code hash
    found.sort(key=lambda s: (s["target"] or "", s["score"], s["code_sha"]))
    return found[:max_skills]


def apply_carrier(carrier: str, target, experience, *, bank: LessonBank, archive=None, harness: LessonHarness | None = None,
                  token_budget: int = 2000, diagnosis_table: dict | None = None) -> dict:
    """The prompt, the context fields, the post-check and the ids a carrier puts to work on a target.

    `experience` is an ExperienceSet (lab/harness/matrix.py) or None for `none` and the two target-side controls;
    `diagnosis_table` is the firing table of the target's own baseline attempts (lab/harness/checker.py) that the
    diagnosis-only control is written from. Returns a dict with `prompt`, `context` (carrier, lesson_ids, skill_ids,
    redacted, experience id), `post_check`, `gate`, and `tool_calls` (1 for the second-try control, 0 otherwise)."""
    if carrier not in CARRIERS:
        raise ValueError(f"carrier must be one of {CARRIERS}")
    ctx: dict[str, Any] = {"carrier": carrier, "lesson_ids": [], "skill_ids": [], "redacted": False,
                           "experience_id": getattr(experience, "id", None) if experience is not None else None}
    if carrier == "second_try":
        ctx["control"] = "second_try"
        return {"prompt": target.prompt, "context": ctx, "post_check": None, "gate": None, "prompt_override": False, "tool_calls": 1}
    if carrier == "diagnosis_only":
        from .checker import diagnosis_text   # the checker imports the lesson module: import here, not at the top

        fired = [d["name"] for d in (diagnosis_table or {}).get("detectors", []) if d.get("fires", 0) > 0]
        ctx.update({"control": "diagnosis_only", "detectors_named": fired, "diagnosis_attempts": int((diagnosis_table or {}).get("n", 0) or 0)})
        return {"prompt": diagnosis_only_prompt(target, diagnosis_text(target, diagnosis_table)), "context": ctx, "post_check": None, "gate": None,
                "prompt_override": True, "tool_calls": 0}
    if carrier == "none" or experience is None:
        return {"prompt": target.prompt, "context": ctx, "post_check": None, "gate": None, "prompt_override": carrier != "none", "tool_calls": 0}
    lessons = [l for l in (bank.get(lid) for lid in experience.lesson_ids) if l is not None and not l.retired]
    if carrier in ("lessons", "unrelated"):
        text, ids = lessons_block([(l, l.text) for l in lessons], token_budget)
        ctx["lesson_ids"] = ids
        return {"prompt": notebook_prompt(target, text), "context": ctx, "post_check": None, "gate": None, "prompt_override": True, "tool_calls": 0}
    if carrier == "process_only":
        terms = list(experience.content_terms)
        entries = process_only_lessons(lessons, terms)
        text, ids = lessons_block(entries, token_budget)
        ctx.update({"lesson_ids": ids, "redacted": True, "redaction_terms": len(terms)})
        return {"prompt": notebook_prompt(target, text), "context": ctx, "post_check": None, "gate": None, "prompt_override": True, "tool_calls": 0}
    if carrier == "lessons_gate":
        if harness is None:
            raise ValueError("the lessons_gate carrier needs the lesson harness (a frontier provider) to choose the lesson and write the check")
        gate = harness.gate(target, candidates=lessons) if lessons else {"lesson": None, "check": None, "reason": "no lessons in the experience set"}
        lesson, check = gate.get("lesson"), gate.get("check")
        ctx.update({"lesson_ids": [lesson.id] if lesson else [], "gate_check": check, "gate_harness_attempt_id": gate.get("harness_attempt_id")})
        post = (lambda result, meta, _c=check: apply_check(_c, result)) if check and check.get("valid") else None  # noqa: E731
        return {"prompt": gate_prompt(target, gate), "context": ctx, "post_check": post, "gate": gate, "prompt_override": True, "tool_calls": 0}
    if carrier == "skills":
        if archive is None:
            raise ValueError("the skills carrier needs the archive")
        skills = skills_from_archive(archive, list(experience.artifact_ids))
        ctx["skill_ids"] = [s["artifact_id"] for s in skills]
        return {"prompt": skills_prompt(target, skills), "context": ctx, "post_check": None, "gate": None, "prompt_override": True, "tool_calls": 0}
    raise ValueError(carrier)
