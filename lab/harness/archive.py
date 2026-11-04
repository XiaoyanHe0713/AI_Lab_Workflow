"""The archive: a content-addressed store of certified methods with witnesses and dossiers, and the provenance
ledger that links each artifact to its attempt, reading-list snapshot, lessons, citations and cost, so that any
artifact is traceable to its full chain in one query.

Objects are canonical JSON under <root>/archive/objects/<aa>/<sha256>.json and rows in the ledger's
`artifacts` and `provenance` tables; `trace()` is one SQL statement joining artifacts, provenance, the attempt
row, the lessons (through json_each) and the citations. Nothing here calls a model.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..ledger import Attempt, Ledger
from .tables import ensure_all


def canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")


def sha256_of(obj: Any) -> str:
    return hashlib.sha256(canonical(obj)).hexdigest()


class Archive:
    def __init__(self, ledger: Ledger, root: str | Path):
        self.ledger = ledger
        self.root = Path(root) / "archive" / "objects"
        self.root.mkdir(parents=True, exist_ok=True)
        ensure_all(ledger)

    # --- objects -----------------------------------------------------------------------------------
    def put(self, kind: str, obj: Any, meta: dict | None = None) -> str:
        """Store an object by content; returns its id (the sha256 of its canonical JSON). Idempotent."""
        blob = canonical(obj)
        sha = hashlib.sha256(blob).hexdigest()
        path = self.root / sha[:2] / f"{sha}.json"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(blob)
        if not self.ledger.rows("artifacts", "id = ?", (sha,)):
            self.ledger.insert("artifacts", {"id": sha, "kind": kind, "sha256": sha, "path": str(path.relative_to(self.root.parent.parent)),
                                             "bytes": len(blob), "created": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                                             "meta": meta or {}}, json_columns=("meta",))
        return sha

    def get(self, artifact_id: str) -> Any:
        path = self.root / artifact_id[:2] / f"{artifact_id}.json"
        if not path.exists():
            raise KeyError(artifact_id)
        data = json.loads(path.read_bytes().decode("utf-8"))
        if hashlib.sha256(canonical(data)).hexdigest() != artifact_id:
            raise ValueError(f"artifact {artifact_id} does not match its content hash")
        return data

    def has(self, artifact_id: str) -> bool:
        return (self.root / artifact_id[:2] / f"{artifact_id}.json").exists()

    # --- certified methods ---------------------------------------------------------------------------
    def archive_method(self, attempt: Attempt, *, dossier: dict | None = None, witness: dict | None = None,
                       lesson_ids: list[str] | None = None, citation_ids: list[str] | None = None,
                       reading_snapshot: dict | None = None, require_certified: bool = True) -> dict:
        """Store a certified attempt's code, certificate, dossier and witness, and write its provenance row."""
        if require_certified and not attempt.passed:
            raise ValueError(f"attempt {attempt.id} is not certified; the archive holds certified methods only")
        code = attempt.meta.get("code") or attempt.output or ""
        code_sha = self.put("code", {"code": code, "target": attempt.target, "entry_point": None})
        cert_sha = self.put("certificate", attempt.certificate or {})
        dossier_sha = self.put("dossier", dossier) if dossier else None
        witness_sha = self.put("witness", witness) if witness else None
        snap_sha = self.put("reading_snapshot", reading_snapshot) if reading_snapshot else None
        method = {"kind": "method", "target": attempt.target, "attempt_id": attempt.id, "code_sha": code_sha,
                  "certificate_sha": cert_sha, "dossier_sha": dossier_sha, "witness_sha": witness_sha,
                  "reading_snapshot_sha": snap_sha, "lesson_ids": sorted(lesson_ids or []), "citation_ids": list(citation_ids or []),
                  "certifier": (attempt.certificate or {}).get("certifier"), "certifier_version": (attempt.certificate or {}).get("version"),
                  "score": attempt.score, "model": attempt.model, "model_version": attempt.model_version, "precision": attempt.precision,
                  "provider": attempt.provider, "mode": attempt.mode}
        method_sha = self.put("method", method, meta={"target": attempt.target, "score": attempt.score})
        self.ledger.insert("provenance", {
            "artifact_id": method_sha, "attempt_id": attempt.id, "target": attempt.target, "certifier": method["certifier"],
            "certifier_version": method["certifier_version"], "score": attempt.score, "code_sha": code_sha, "certificate_sha": cert_sha,
            "dossier_sha": dossier_sha, "witness_sha": witness_sha, "reading_snapshot_sha": snap_sha,
            "lesson_ids": sorted(lesson_ids or []), "citation_ids": list(citation_ids or []), "dollars": attempt.dollars,
            "api_dollars": attempt.api_dollars, "group_id": attempt.group_id,
            "created": datetime.now(timezone.utc).isoformat(timespec="milliseconds")}, json_columns=("lesson_ids", "citation_ids"))
        return {"artifact_id": method_sha, **method}

    TRACE_SQL = """
        SELECT a.id AS artifact_id, a.kind, a.sha256, a.path, a.created AS archived_at,
               p.attempt_id, p.target, p.certifier, p.certifier_version, p.score, p.code_sha, p.certificate_sha, p.dossier_sha,
               p.witness_sha, p.reading_snapshot_sha, p.lesson_ids, p.citation_ids, p.dollars, p.api_dollars, p.group_id,
               t.role, t.model, t.model_version, t.precision, t.provider, t.mode, t.seed, t.timestamp AS attempted_at,
               t.prompt_hash, t.context AS attempt_context,
               (SELECT group_concat(l.id || ' :: ' || l.rule, ' || ') FROM lessons l
                 WHERE l.id IN (SELECT value FROM json_each(p.lesson_ids))) AS lessons,
               (SELECT group_concat(c.doc_id || '@' || c.doc_date, ', ') FROM citations c
                 WHERE c.id IN (SELECT value FROM json_each(p.citation_ids))) AS cited_documents,
               (SELECT count(*) FROM attempts h WHERE h.role LIKE 'harness:%' AND h.target = p.target) AS harness_calls_on_target
        FROM artifacts a
        JOIN provenance p ON p.artifact_id = a.id
        LEFT JOIN attempts t ON t.id = p.attempt_id
        WHERE a.id = ?
    """

    def trace(self, artifact_id: str) -> dict:
        """The full chain of a method artifact in one query: the artifact, its provenance, the attempt that made it
        (model, precision, provider, mode, cost), the lessons it saw and the documents it cited."""
        rows = self.ledger.query(self.TRACE_SQL, (artifact_id,))
        if not rows:
            raise KeyError(f"no provenance for artifact {artifact_id}")
        r = rows[0]
        for k in ("lesson_ids", "citation_ids", "attempt_context"):
            if isinstance(r.get(k), str):
                try:
                    r[k] = json.loads(r[k])
                except ValueError:
                    pass
        return r

    def methods(self, target: str | None = None) -> list[dict]:
        return self.ledger.rows("provenance", "target = ?" if target else "", (target,) if target else (),
                                json_columns=("lesson_ids", "citation_ids"), order="created")
