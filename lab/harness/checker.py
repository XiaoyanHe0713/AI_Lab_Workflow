"""The checker (ADSD's instruments, as the proposal adopts them): the frozen baseline record of the untrained
model, hypotheses turned into deterministic detectors that are kept only when right on more than nine of ten
held-out cases, and method dossiers with claim, conditions and evidence, schema-validated.

Detectors are programs: a spec in a small grammar (conjunctions of conditions over a flattened view of an
attempt: its certificate details, its diagnosis, its code) that the program evaluates. The language part
proposes hypotheses and specs from a mining pool of attempts it is shown; the program measures each spec on a
held-out pool it never sees (split by the attempt's seed, so the split and the prompt are stable) and keeps a
detector only if it fires on at least `min_fires` held-out cases, is right (fires only on failures) on more than
`min_right` of the held-out pool, and every firing is a failure. Built-in detectors written by hand pass the same
gate. A dossier's claim, conditions and evidence summary are language; its evidence fields (attempt id,
certificate hash, score, certifier) are filled and checked by the program. Nothing here enters a reward: the
diagnosis column and the dossiers are information for a person and for the next prompt.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any

from ..ledger import Attempt, Ledger, summarize
from .base import Harness
from .lessons import attempt_digest, fmt
from .tables import ensure_all

OPS = ("==", "!=", "<", "<=", ">", ">=", "abs_gt", "abs_lt", "contains", "not_contains", "is_true", "is_false", "is_null", "not_null")
FIELDS = ("passed", "score", "optimal", "error", "diagnosis", "has_code", "code", "code_lines", "node_count", "weight_sum", "volume",
          "weight_sum_error", "all_weights_positive", "all_nodes_interior", "max_monomial_error", "measured_exactness_degree",
          "worst_monomial_degree", "forbidden_import", "static_violations", "finish_reason", "completion_tokens", "uses_numpy",
          "uses_linalg", "uses_math", "measured_order", "total_ops", "ssp_coefficient", "rate", "sandbox_ok", "sandbox_error_type")


# --- the attempt view --------------------------------------------------------------------------------------
def attempt_view(a: Attempt) -> dict:
    cert = a.certificate or {}
    d = cert.get("details", {}) or {}
    sb = d.get("sandbox", {}) or {}
    code = (a.meta or {}).get("code") or ""
    err = cert.get("error") or ""
    worst = d.get("worst_monomial")
    v = {
        "passed": bool(a.passed), "score": a.score, "optimal": cert.get("optimal"), "error": err, "diagnosis": a.diagnosis or "",
        "has_code": bool(code.strip()), "code": code, "code_lines": len(code.strip("\n").splitlines()) if code.strip() else 0,
        "node_count": d.get("node_count"), "weight_sum": d.get("weight_sum"), "volume": d.get("volume"),
        "weight_sum_error": (abs(d["weight_sum"] - d["volume"]) if d.get("weight_sum") is not None and d.get("volume") is not None else None),
        "all_weights_positive": d.get("all_weights_positive"), "all_nodes_interior": d.get("all_nodes_interior"),
        "max_monomial_error": d.get("max_monomial_error"), "measured_exactness_degree": d.get("measured_exactness_degree"),
        "worst_monomial_degree": (sum(worst) if isinstance(worst, list) else None),
        "forbidden_import": (any(("forbidden import" in str(x) or "forbidden call" in str(x) or "forbidden attribute" in str(x))
                                 for x in (sb.get("static_violations") or [])) or "forbidden import at run time" in err),
        "static_violations": sb.get("static_violations") or [], "finish_reason": (a.meta or {}).get("finish_reason"),
        "completion_tokens": (a.meta or {}).get("completion_tokens"), "uses_numpy": "numpy" in code or "np." in code,
        "uses_linalg": "linalg" in code, "uses_math": "import math" in code, "measured_order": d.get("measured_order"),
        "total_ops": d.get("total_ops"), "ssp_coefficient": d.get("ssp_coefficient"), "rate": d.get("rate"),
        "sandbox_ok": sb.get("ok"), "sandbox_error_type": ((sb.get("error") or {}).get("type") if isinstance(sb.get("error"), dict) else None),
    }
    return v


def evaluate_condition(cond: dict, view: dict) -> bool:
    field_, op, value = cond.get("field"), cond.get("op"), cond.get("value")
    if field_ not in FIELDS or op not in OPS:
        raise ValueError(f"bad condition {cond}")
    x = view.get(field_)
    if op == "is_null":
        return x is None
    if op == "not_null":
        return x is not None
    if op == "is_true":
        return x is True
    if op == "is_false":
        return x is False
    if op in ("contains", "not_contains"):
        hit = str(value).lower() in str(x if x is not None else "").lower() if not isinstance(x, list) else any(str(value).lower() in str(i).lower() for i in x)
        return hit if op == "contains" else not hit
    if x is None:
        return False
    try:
        if op == "abs_gt":
            return abs(float(x)) > float(value)
        if op == "abs_lt":
            return abs(float(x)) < float(value)
        if isinstance(x, bool) or isinstance(value, bool):
            return (x == value) if op == "==" else (x != value) if op == "!=" else False
        xf, vf = float(x), float(value)
    except (TypeError, ValueError):
        return (str(x) == str(value)) if op == "==" else (str(x) != str(value)) if op == "!=" else False
    return {"==": xf == vf, "!=": xf != vf, "<": xf < vf, "<=": xf <= vf, ">": xf > vf, ">=": xf >= vf}[op]


def evaluate_detector(spec: dict, view: dict) -> bool:
    conds = spec.get("all") or []
    if not conds:
        return False
    return all(evaluate_condition(c, view) for c in conds)


def validate_spec(spec: Any) -> tuple[bool, str | None]:
    if not isinstance(spec, dict) or not isinstance(spec.get("all"), list) or not spec["all"]:
        return False, "a detector spec is {'all': [conditions]} with at least one condition"
    for c in spec["all"]:
        if not isinstance(c, dict) or c.get("field") not in FIELDS or c.get("op") not in OPS:
            return False, f"bad condition {c}: field in {FIELDS}, op in {OPS}"
        if c["field"] == "passed":
            return False, "a detector may not read the certificate's verdict itself"
    return True, None


BUILTIN_DETECTORS = [
    {"name": "no_code_block", "hypothesis": "the output has no code block", "spec": {"all": [{"field": "has_code", "op": "is_false"}]}},
    {"name": "forbidden_import", "hypothesis": "the code imports a forbidden library", "spec": {"all": [{"field": "forbidden_import", "op": "is_true"}]}},
    {"name": "weights_do_not_sum_to_volume", "hypothesis": "the weights do not sum to the domain's volume",
     "spec": {"all": [{"field": "weight_sum_error", "op": "abs_gt", "value": 1e-8}]}},
    {"name": "sandbox_crash", "hypothesis": "the candidate raised or failed to run",
     "spec": {"all": [{"field": "sandbox_ok", "op": "is_false"}]}},
    {"name": "too_low_degree", "hypothesis": "the rule is exact only to a lower degree than required",
     "spec": {"all": [{"field": "max_monomial_error", "op": "abs_gt", "value": 1e-9}, {"field": "weight_sum_error", "op": "abs_lt", "value": 1e-8}]}},
]


def split_pools(attempts: list[Attempt], seed: int = 0) -> tuple[list[Attempt], list[Attempt]]:
    """Mining pool (shown to the model) and held-out pool (never shown), by the attempt's seed parity (stable)."""
    rows = sorted(attempts, key=lambda a: (a.seed if a.seed is not None else 0, a.id))
    mining = [a for i, a in enumerate(rows) if ((a.seed if a.seed is not None else i) + seed) % 2 == 0]
    heldout = [a for i, a in enumerate(rows) if ((a.seed if a.seed is not None else i) + seed) % 2 == 1]
    return mining, heldout


def verify_detector(spec: dict, heldout: list[Attempt], *, min_fires: int = 2, min_right: float = 0.9) -> dict:
    """Right on a held-out case means: fires only if the case failed (the implication 'fires -> failed' holds).
    Kept iff it fires at least min_fires times, is right on more than min_right of the pool, and every firing is a failure."""
    n = len(heldout)
    fires = right = tp = 0
    for a in heldout:
        v = attempt_view(a)
        f = evaluate_detector(spec, v)
        if f:
            fires += 1
            if not a.passed:
                tp += 1
        if (not f) or (not a.passed):
            right += 1
    precision = tp / fires if fires else None
    accuracy = right / n if n else None
    kept = bool(n > 0 and fires >= min_fires and right > min_right * n and tp == fires)
    return {"heldout_n": n, "fires": fires, "right": right, "true_positives": tp, "precision": precision, "accuracy": accuracy, "kept": kept}


def detector_id(target: str, spec: dict) -> str:
    return "det_" + hashlib.sha256(json.dumps({"t": target, "s": spec}, sort_keys=True).encode()).hexdigest()[:12]


def store_detectors(ledger: Ledger, target, candidates: list[dict], heldout: list[Attempt], *, mining_n: int, proposed: int,
                    harness_attempt_id: str | None, provider: str, mode: str, min_fires: int = 2, min_right: float = 0.9) -> dict:
    """Verify every candidate on the held-out pool and store it (kept or not) in the detectors table; the report."""
    now = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    results = []
    for c in candidates:
        if not c["valid"]:
            results.append({**c, "kept": False, "verification": None})
            continue
        ver = verify_detector(c["spec"], heldout, min_fires=min_fires, min_right=min_right)
        did = detector_id(target.id, c["spec"])
        ledger.insert("detectors", {
            "id": did, "target": target.id, "family": target.family, "name": c["name"], "hypothesis": c["hypothesis"], "spec": c["spec"],
            "origin": c["origin"], "harness_attempt_id": harness_attempt_id if c["origin"] == "model" else None, "heldout_n": ver["heldout_n"],
            "fires": ver["fires"], "right": ver["right"], "precision": ver["precision"], "accuracy": ver["accuracy"],
            "kept": int(ver["kept"]), "created": now, "provider": provider, "mode": mode}, json_columns=("spec",))
        results.append({**c, "id": did, "kept": ver["kept"], "verification": ver})
    return {"target": target.id, "mining_n": mining_n, "heldout_n": len(heldout), "proposed": proposed,
            "kept": [r for r in results if r["kept"]], "rejected": [r for r in results if not r["kept"]],
            "harness_attempt_id": harness_attempt_id, "min_fires": min_fires, "min_right": min_right}


def mine_builtin(ledger: Ledger, target, attempts: list[Attempt], *, seed: int = 0, min_fires: int = 2, min_right: float = 0.9) -> dict:
    """Mining without a harness (week 5): the built-in detectors alone, verified on the held-out pool exactly as a model's
    proposals would be, and stored with provider `program`. A program end to end; the loop and the matrix use it when no
    checker runs, so the diagnosis of a base group never waits for a frontier call."""
    ensure_all(ledger)
    mining, heldout = split_pools(attempts, seed)
    candidates = [{**b, "valid": True, "error": None, "origin": "builtin"} for b in BUILTIN_DETECTORS]
    return store_detectors(ledger, target, candidates, heldout, mining_n=len(mining), proposed=0, harness_attempt_id=None,
                           provider="program", mode="program", min_fires=min_fires, min_right=min_right)


# --- the diagnosis as a program's object (week 5): kept detectors, the firing table, the query, the control text -----------
def kept_detectors(ledger: Ledger, target: str) -> list[dict]:
    """The detectors the program kept for a target (the loop's, the matrix's or the command's mining), oldest first."""
    try:
        return ledger.rows("detectors", "target = ? AND kept = 1", (target,), json_columns=("spec",), order="created, id")
    except Exception:
        return []


def firing_table(detectors: list[dict], attempts: list[Attempt]) -> dict:
    """Which kept detectors fire on which attempts of a group: per detector its firings (most frequent first), per attempt the
    names that fire. A program's measurement of the symptoms; nothing here is a score."""
    rows = sorted(attempts, key=lambda a: (a.seed if a.seed is not None else 0, a.id))
    views = [(a, attempt_view(a)) for a in rows]
    by_attempt: dict[str, list[str]] = {a.id: [] for a in rows}
    out = []
    for d in detectors:
        fired = []
        for a, v in views:
            try:
                hit = evaluate_detector(d["spec"], v)
            except Exception:
                hit = False
            if hit:
                fired.append(a.id)
                by_attempt[a.id].append(d["name"])
        fired_set = set(fired)
        out.append({"id": d.get("id"), "name": d["name"], "hypothesis": d.get("hypothesis") or "", "origin": d.get("origin"),
                    "fires": len(fired), "failures_fired": sum(1 for a in rows if a.id in fired_set and not a.passed),
                    "rate": (len(fired) / len(rows)) if rows else None, "attempt_ids": fired})
    out.sort(key=lambda x: (-x["fires"], x["name"]))
    return {"n": len(rows), "failed": sum(1 for a in rows if not a.passed), "detectors": out, "by_attempt": by_attempt}


def symptom_rates(detectors: list[dict], attempts: list[Attempt]) -> dict:
    """Per kept detector: how often it fires on a group (the symptom rate the matrix and the loop compare across groups)."""
    table = firing_table(detectors, attempts)
    return {d["name"]: {"fires": d["fires"], "n": table["n"], "rate": d["rate"]} for d in table["detectors"]}


def diagnosis_query(target, table: dict | None, max_detectors: int = 3) -> str:
    """The librarian's query named by the diagnosis: the target's title and family, then the hypotheses of the detectors that
    fired on the base group, most frequent first. Without a firing detector it is the plain query the week-4 loop used."""
    base = f"{target.title} {target.family}"
    fired = [d for d in (table or {}).get("detectors", []) if d["fires"] > 0][:max_detectors]
    if not fired:
        return base
    return base + " " + " ".join(d["hypothesis"] or d["name"].replace("_", " ") for d in fired)


def diagnosis_text(target, table: dict | None) -> str:
    """The diagnosis-only control's text (the matrix's target-side row): the failure modes a program measured on the target's
    own baseline attempts, with their counts, and no remedy. If this row matches the lessons row, the model held the
    knowledge and lacked the diagnosis; if it matches the baseline, knowledge is the barrier."""
    n = int((table or {}).get("n", 0) or 0)
    fired = [d for d in (table or {}).get("detectors", []) if d["fires"] > 0]
    head = ("What went wrong in earlier attempts on this target, as measured by a program's detectors (each line is a symptom "
            "seen on failed attempts; no remedy is given):")
    if not fired or not n:
        return head + "\n- no detector fired on the earlier attempts"
    return head + "\n" + "\n".join(f"- {d['fires']} of {n} attempts: {d['hypothesis'] or d['name'].replace('_', ' ')}" for d in fired)


def base_groups(ledger: Ledger, target: str, *, reportable_only: bool = False) -> list[tuple[str, list[Attempt]]]:
    """The plain-prompt builder groups of a target (no lessons, no prompt override, not derived from a proposal), oldest
    first: the groups a diagnosis is mined from."""
    rows = [r for r in ledger.attempts(target=target, role="builder")
            if not (r.context or {}).get("lesson_ids") and not (r.context or {}).get("prompt_override")
            and not (r.context or {}).get("derived_from_proposal")]
    if reportable_only:
        rows = [r for r in rows if r.mode not in ("offline", None) and r.precision == "bf16"]
    groups: dict[str, list[Attempt]] = {}
    for r in rows:
        groups.setdefault(r.group_id or "", []).append(r)
    return sorted(groups.items(), key=lambda kv: min(x.timestamp for x in kv[1]))


# --- dossiers -------------------------------------------------------------------------------------------------
DOSSIER_SCHEMA = {
    "claim": str, "conditions": list, "evidence": dict,
}
EVIDENCE_REQUIRED = ("attempt_id", "target", "certifier", "certifier_version", "score", "passed", "certificate_sha256", "code_sha256",
                     "summary")


def validate_dossier(d: Any, attempt: Attempt | None = None) -> list[str]:
    errors = []
    if not isinstance(d, dict):
        return ["a dossier is a dict"]
    for k, typ in DOSSIER_SCHEMA.items():
        if k not in d:
            errors.append(f"missing {k}")
        elif not isinstance(d[k], typ):
            errors.append(f"{k} must be {typ.__name__}")
    if isinstance(d.get("claim"), str) and len(d["claim"].split()) < 3:
        errors.append("claim must be a sentence")
    if isinstance(d.get("conditions"), list):
        if not d["conditions"]:
            errors.append("conditions must not be empty")
        if not all(isinstance(c, str) and c.strip() for c in d["conditions"]):
            errors.append("conditions must be non-empty strings")
    ev = d.get("evidence")
    if isinstance(ev, dict):
        for k in EVIDENCE_REQUIRED:
            if k not in ev:
                errors.append(f"evidence missing {k}")
        if attempt is not None:
            if ev.get("attempt_id") != attempt.id:
                errors.append("evidence.attempt_id must be the certified attempt")
            if ev.get("score") != attempt.score:
                errors.append("evidence.score must equal the certificate's score")
            if ev.get("passed") is not True or not attempt.passed:
                errors.append("a dossier needs a certified attempt")
    return errors


def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()


class Checker(Harness):
    name = "checker"
    system_prompt = ("You are the checker of an automated numerical-analysis lab. You turn observations about failed and certified "
                     "attempts into hypotheses that a program can test, and you write method dossiers that claim only what the "
                     "certificate shows. Answer with JSON only.")

    def __init__(self, provider, ledger: Ledger, *, cfg: dict | None = None, **kw):
        super().__init__(provider, ledger, cfg=cfg, **kw)
        ensure_all(self.ledger)
        h = self.cfg.get("harness", {})
        self.min_fires = int(h.get("detector_min_fires", 2))
        self.min_right = float(h.get("detector_min_right", 0.9))

    # --- the frozen baseline record (program only) -------------------------------------------------------------
    @staticmethod
    def baseline_record(ledger: Ledger, *, targets: list[str] | None = None, reportable_only: bool = False) -> dict:
        """Per target and group: the untrained model's record (builder rows without lessons), pass@k valid and optimal,
        cost, model version, precision, provider and mode. Rows from offline replays or quantized builds are flagged
        and, with reportable_only, left out."""
        rows = [r for r in ledger.attempts(role="builder") if not (r.context or {}).get("lesson_ids")
                and not (r.context or {}).get("prompt_override")]
        if targets:
            rows = [r for r in rows if r.target in targets]
        groups: dict[tuple, list[Attempt]] = {}
        for r in rows:
            groups.setdefault((r.target, r.model_version, r.precision, r.provider, r.mode, r.group_id), []).append(r)
        record = {"written": datetime.now(timezone.utc).isoformat(timespec="seconds"), "targets": {}}
        for (target, mv, prec, prov, mode, gid), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[1][0].timestamp)):
            reportable = mode not in ("offline", None) and prec in ("bf16",)
            if reportable_only and not reportable:
                continue
            s = summarize(rs)
            entry = {"group_id": gid, "model_version": mv, "precision": prec, "provider": prov, "mode": mode, "attempts": s["attempts"],
                     "passed": s["passed"], "optimal": s["optimal"], "pass_at_k_valid": s["pass_at_k"], "pass_at_k_optimal": s["optimal_at_k"],
                     "dollars": s["dollars"], "gpu_seconds": s["gpu_seconds"], "best_score": s["best_score"], "reportable": reportable,
                     "first": rs[0].timestamp, "adapter": (rs[0].context or {}).get("adapter")}
            record["targets"].setdefault(target, []).append(entry)
        return record

    # --- detectors ----------------------------------------------------------------------------------------------
    def propose_detectors(self, target, mining: list[Attempt], *, max_detectors: int = 6) -> tuple[list[dict], str | None]:
        rows = sorted(mining, key=lambda a: (a.seed if a.seed is not None else 0, a.id))
        digest = "\n\n".join(attempt_digest(f"attempt {i + 1}", a, max_lines=30) for i, a in enumerate(rows))
        views = "\n".join(f"attempt {i + 1}: " + ", ".join(f"{k}={fmt(v)}" for k, v in attempt_view(a).items()
                                                           if k in ("has_code", "node_count", "weight_sum", "volume", "weight_sum_error",
                                                                    "all_weights_positive", "all_nodes_interior", "max_monomial_error",
                                                                    "measured_exactness_degree", "forbidden_import", "sandbox_ok", "code_lines"))
                          for i, a in enumerate(rows))
        user = (f"Target {target.id}: {target.title} (certificate {target.certifier_name}).\n\n"
                f"Mining pool, {len(rows)} attempts:\n\n{digest}\n\nThe fields a detector may read, per attempt:\n{views}\n\n"
                f"Propose at most {max_detectors} hypotheses of the form 'attempts with property P fail the certificate', each as a "
                f"deterministic detector over the fields {', '.join(f for f in FIELDS if f not in ('passed', 'code', 'diagnosis', 'error'))} "
                f"with operators {', '.join(OPS)}. A detector fires when all its conditions hold; it is kept only if, on attempts you "
                f"have not seen, every firing is a failed attempt and it fires at least twice. Reply with JSON: "
                f"{{\"detectors\": [{{\"name\": \"snake_case\", \"hypothesis\": \"...\", \"spec\": {{\"all\": [{{\"field\": \"...\", \"op\": \"...\", \"value\": ...}}]}}}}]}}")
        reply = self.ask("propose_detectors", user, target=target.id, context={"n_mining": len(rows)})
        data = reply.data if isinstance(reply.data, dict) else {}
        out = []
        for item in (data.get("detectors") or [])[:max_detectors]:
            if not isinstance(item, dict):
                continue
            ok, err = validate_spec(item.get("spec"))
            name = re.sub(r"[^a-z0-9_]+", "_", str(item.get("name", "detector")).lower())[:60] or "detector"
            out.append({"name": name, "hypothesis": str(item.get("hypothesis", ""))[:300], "spec": item.get("spec"), "valid": ok,
                        "error": err, "origin": "model"})
        return out, reply.attempt_id

    def mine(self, target, attempts: list[Attempt], *, seed: int = 0, include_builtin: bool = True, max_detectors: int = 6) -> dict:
        """Split the attempts into a mining and a held-out pool, propose detectors from the mining pool, verify every
        candidate (and the built-in detectors) on the held-out pool, store the kept ones, and report."""
        mining, heldout = split_pools(attempts, seed)
        proposed, hid = self.propose_detectors(target, mining, max_detectors=max_detectors) if mining else ([], None)
        candidates = list(proposed)
        if include_builtin:
            candidates += [{**b, "valid": True, "error": None, "origin": "builtin"} for b in BUILTIN_DETECTORS]
        return store_detectors(self.ledger, target, candidates, heldout, mining_n=len(mining), proposed=len(proposed), harness_attempt_id=hid,
                               provider=self.provider.name, mode=self.provider.mode, min_fires=self.min_fires, min_right=self.min_right)

    def detectors(self, target: str | None = None, kept_only: bool = True) -> list[dict]:
        where, args = [], []
        if target:
            where.append("target = ?")
            args.append(target)
        if kept_only:
            where.append("kept = 1")
        return self.ledger.rows("detectors", " AND ".join(where), args, json_columns=("spec",), order="created")

    def diagnose(self, attempt: Attempt, target: str | None = None) -> list[str]:
        """The names of the kept detectors that fire on an attempt (a diagnosis a program can stand behind)."""
        v = attempt_view(attempt)
        return [d["name"] for d in self.detectors(target or attempt.target) if evaluate_detector(d["spec"], v)]

    # --- dossiers -----------------------------------------------------------------------------------------------
    def dossier(self, attempt: Attempt, target) -> dict:
        """A method dossier for a certified attempt: claim, conditions and an evidence summary from the model; the
        evidence fields from the program; validated against the schema; stored in the dossiers table."""
        if not attempt.passed:
            raise ValueError("a dossier is written for a certified attempt only")
        cert = attempt.certificate or {}
        d = cert.get("details", {}) or {}
        facts = {k: fmt(d[k]) for k in ("node_count", "degree", "domain", "weight_sum", "all_weights_positive", "all_nodes_interior",
                                       "measured_exactness_degree", "measured_order", "total_ops", "ssp_coefficient", "rate", "stages") if k in d}
        code = (attempt.meta or {}).get("code") or ""
        user = (f"Target {target.id}: {target.title}. Certificate {cert.get('certifier')} {cert.get('version')}: PASS, score {fmt(attempt.score)}, "
                f"best known {fmt(target.best_known)}, optimal={cert.get('optimal')}. Certified facts: {facts}.\n\n"
                f"Code:\n```python\n{code.strip()}\n```\n\nWrite a method dossier: a one-sentence claim that states only what the "
                f"certificate shows (the construction and the certified value), one to four conditions under which the method applies "
                f"(domain, degree, assumptions), and a two-sentence evidence summary of how the certificate established it. Reply with "
                f"JSON: {{\"claim\": \"...\", \"conditions\": [\"...\"], \"evidence_summary\": \"...\"}}")
        reply = self.ask("dossier", user, target=target.id)
        data = reply.data if isinstance(reply.data, dict) else {}
        dossier = {
            "claim": str(data.get("claim", "")).strip(),
            "conditions": [str(c).strip() for c in (data.get("conditions") or []) if str(c).strip()],
            "evidence": {"attempt_id": attempt.id, "target": attempt.target, "certifier": cert.get("certifier"),
                         "certifier_version": cert.get("version"), "score": attempt.score, "passed": bool(attempt.passed),
                         "certificate_sha256": _sha(cert), "code_sha256": _sha(code), "model": attempt.model,
                         "model_version": attempt.model_version, "precision": attempt.precision, "provider": attempt.provider,
                         "mode": attempt.mode, "summary": str(data.get("evidence_summary", "")).strip()},
            "written_by": {"harness": self.name, "harness_attempt_id": reply.attempt_id, "provider": self.provider.name,
                           "mode": self.provider.mode, "model": self.provider.model},
        }
        errors = validate_dossier(dossier, attempt)
        did = "dos_" + _sha({"a": attempt.id, "c": dossier["claim"]})[:12]
        self.ledger.insert("dossiers", {"id": did, "attempt_id": attempt.id, "target": attempt.target, "claim": dossier["claim"],
                                        "conditions": dossier["conditions"], "evidence": dossier["evidence"], "valid": int(not errors),
                                        "errors": errors, "harness_attempt_id": reply.attempt_id, "artifact_id": None,
                                        "created": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                                        "provider": self.provider.name, "mode": self.provider.mode},
                           json_columns=("conditions", "evidence", "errors"))
        dossier["id"] = did
        dossier["valid"] = not errors
        dossier["errors"] = errors
        return dossier


def mine_from_ledger(ledger: Ledger, target_ids: list[str], *, group_ids: list[str] | None = None, all_groups: bool = False,
                     reportable_only: bool = False, checker: "Checker | None" = None, seed: int = 0) -> list[dict]:
    """Mine detectors on the base groups a ledger already holds (week 5: the loop's plain-prompt groups on Drive, after the
    fact and with no GPU). Per target: the latest base group (or every one with `all_groups`, or the given group ids);
    the checker proposes detectors when one is given, the built-in detectors alone otherwise; every candidate is verified
    on the held-out half; the kept detectors' firing table over the group is reported."""
    from ..targets import get_target

    ensure_all(ledger)
    reports = []
    for tid in target_ids:
        target = get_target(tid)
        if group_ids:
            groups = [(g, ledger.attempts(group_id=g)) for g in group_ids]
            groups = [(g, [r for r in rows if r.target == tid]) for g, rows in groups]
            groups = [(g, rows) for g, rows in groups if rows]
        else:
            groups = base_groups(ledger, tid, reportable_only=reportable_only)
            if groups and not all_groups:
                groups = groups[-1:]
        for gid, rows in groups:
            rep = checker.mine(target, rows, seed=seed) if checker is not None else mine_builtin(ledger, target, rows, seed=seed)
            kept = kept_detectors(ledger, tid)
            table = firing_table(kept, rows)
            reports.append({"target": tid, "group_id": gid, "attempts": len(rows), "failed": table["failed"], "mining": "checker" if checker else "builtin",
                            "proposed": rep["proposed"], "kept": [{"name": d["name"], "origin": d["origin"], "heldout_fires": d["verification"]["fires"],
                                                                   "precision": d["verification"]["precision"]} for d in rep["kept"]],
                            "rejected": [d["name"] for d in rep["rejected"]], "harness_attempt_id": rep.get("harness_attempt_id"),
                            "firing": [{k: d[k] for k in ("name", "origin", "fires", "failures_fired", "rate")} for d in table["detectors"]],
                            "query": diagnosis_query(target, table), "diagnosis_text": diagnosis_text(target, table)})
    return reports


def mining_markdown(reports: list[dict]) -> str:
    lines = ["| target | group | attempts | failed | mining | kept detectors (fires on the group) | rejected |", "|---|---|---|---|---|---|---|"]
    for r in reports:
        fires = {f["name"]: f["fires"] for f in r["firing"]}
        kept = "; ".join(f"{k['name']} ({fires.get(k['name'], 0)} of {r['attempts']})" for k in r["kept"]) or "none"
        lines.append(f"| {r['target']} | {r['group_id']} | {r['attempts']} | {r['failed']} | {r['mining']} | {kept} | {len(r['rejected'])} |")
    for r in reports:
        lines += ["", f"{r['target']} ({r['group_id']}): librarian query \"{r['query']}\"", r["diagnosis_text"]]
    lines += ["", "A detector is kept only if, on attempts the proposer never saw, it fires at least twice and every firing is a failure. "
                  "The firing table is a program's measurement; nothing here is a score."]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    """python -m lab.harness.checker --baseline [--targets quad-d7 ...]: the frozen baseline record from the ledger.
    python -m lab.harness.checker --mine --targets quad-square-d5 ... [--provider openai --mode online]: detectors mined on
    the base groups the ledger holds (built-in detectors alone without a provider), stored, and their firing tables."""
    import argparse
    import json as _json
    from pathlib import Path

    from ..config import lab_config

    ap = argparse.ArgumentParser(prog="python -m lab.harness.checker")
    ap.add_argument("--baseline", action="store_true", help="write the baseline record from the ledger's groups")
    ap.add_argument("--mine", action="store_true", help="mine detectors on the base groups of --targets already in the ledger (no GPU)")
    ap.add_argument("--targets", nargs="*", default=None)
    ap.add_argument("--groups", nargs="*", default=None, help="with --mine: these group ids instead of the latest base group per target")
    ap.add_argument("--all-groups", action="store_true", help="with --mine: every base group of each target, not only the latest")
    ap.add_argument("--provider", default=None, help="with --mine: openai | anthropic proposes detectors (default: built-in detectors only)")
    ap.add_argument("--mode", default=None, choices=["offline", "online", "record"])
    ap.add_argument("--seed", type=int, default=0, help="with --mine: the pool split's seed")
    ap.add_argument("--reportable-only", action="store_true", help="leave out offline replays and quantized rows")
    ap.add_argument("--ledger", default=None)
    ap.add_argument("--fixtures", default=None)
    args = ap.parse_args(argv)
    cfg = lab_config()
    ledger = Ledger(args.ledger or cfg["storage"]["ledger"])
    if args.mine:
        if not args.targets:
            ap.error("--mine needs --targets")
        checker = None
        if args.provider:
            from .base import harness_provider

            checker = Checker(harness_provider(cfg, name=args.provider, mode=args.mode, ledger=ledger, fixtures_root=args.fixtures), ledger, cfg=cfg)
        reports = mine_from_ledger(ledger, args.targets, group_ids=args.groups, all_groups=args.all_groups, reportable_only=args.reportable_only,
                                   checker=checker, seed=args.seed)
        out = Path(ledger.path).resolve().parent / "detectors"
        out.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = out / f"mining-{stamp}"
        path.with_suffix(".json").write_text(_json.dumps(reports, indent=1, default=str), encoding="utf-8")
        path.with_suffix(".md").write_text(mining_markdown(reports), encoding="utf-8")
        print(mining_markdown(reports))
        print(f"mining report written to {path.with_suffix('.md')}" + (f" (harness spend ${checker.spent:.4f})" if checker else ""))
        ledger.close()
        return 0
    if args.baseline:
        rec = Checker.baseline_record(ledger, targets=args.targets, reportable_only=args.reportable_only)
        out = Path(ledger.path).resolve().parent / "baselines"
        out.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = out / f"baseline-{stamp}.json"
        path.write_text(_json.dumps(rec, indent=1, default=str), encoding="utf-8")
        print(f"baseline record written to {path}")
        print("| target | group | model version | precision | provider/mode | attempts | valid | optimal | pass@1 valid | reportable |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for target, entries in rec["targets"].items():
            for e in entries:
                print(f"| {target} | {e['group_id']} | {e['model_version'][:40]} | {e['precision']} | {e['provider']}/{e['mode']} | "
                      f"{e['attempts']} | {e['passed']} | {e['optimal']} | {e['pass_at_k_valid']['pass@1']:.3f} | {e['reportable']} |")
    ledger.close()
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
