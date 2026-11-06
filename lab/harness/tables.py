"""The harness tables in the ledger's SQLite file (proposal: SQLite tables for attempts, certificates, lessons,
citations and costs). Every row carries its provenance: the harness attempt row that produced it, the
provider and the mode, so a replayed response can never be mistaken for a live one."""
from __future__ import annotations

from ..ledger import Ledger

CITATIONS = [
    ("id", "TEXT PRIMARY KEY"), ("timestamp", "TEXT NOT NULL"), ("harness_attempt_id", "TEXT"), ("target", "TEXT"),
    ("query", "TEXT"), ("cutoff", "TEXT"), ("doc_id", "TEXT NOT NULL"), ("doc_date", "TEXT"), ("rank", "INTEGER"),
    ("selected_by", "TEXT"), ("allowed", "INTEGER"), ("provider", "TEXT"), ("mode", "TEXT"), ("purpose", "TEXT"),
]

LESSONS = [
    ("id", "TEXT PRIMARY KEY"), ("level", "TEXT NOT NULL"), ("rule", "TEXT NOT NULL"), ("conditions", "TEXT"),
    ("evidence_tag", "TEXT"), ("support", "TEXT"), ("contradictions", "TEXT"), ("family", "TEXT"), ("targets", "TEXT"),
    ("tokens", "INTEGER"), ("created", "TEXT"), ("source_group", "TEXT"), ("retired", "INTEGER"), ("retired_at", "TEXT"),
    ("retired_reason", "TEXT"), ("replaced_by", "TEXT"), ("provenance", "TEXT"),
    ("detectors", "TEXT"),      # week 5: names of the kept detectors the lesson addresses (an older ledger gains the column on open)
]

ARTIFACTS = [
    ("id", "TEXT PRIMARY KEY"), ("kind", "TEXT NOT NULL"), ("sha256", "TEXT NOT NULL"), ("path", "TEXT"),
    ("bytes", "INTEGER"), ("created", "TEXT"), ("meta", "TEXT"),
]

PROVENANCE = [
    ("artifact_id", "TEXT PRIMARY KEY"), ("attempt_id", "TEXT"), ("target", "TEXT"), ("certifier", "TEXT"),
    ("certifier_version", "TEXT"), ("score", "REAL"), ("code_sha", "TEXT"), ("certificate_sha", "TEXT"),
    ("dossier_sha", "TEXT"), ("witness_sha", "TEXT"), ("reading_snapshot_sha", "TEXT"), ("lesson_ids", "TEXT"),
    ("citation_ids", "TEXT"), ("dollars", "REAL"), ("api_dollars", "REAL"), ("group_id", "TEXT"), ("created", "TEXT"),
]

DETECTORS = [
    ("id", "TEXT PRIMARY KEY"), ("target", "TEXT"), ("family", "TEXT"), ("name", "TEXT"), ("hypothesis", "TEXT"),
    ("spec", "TEXT"), ("origin", "TEXT"), ("harness_attempt_id", "TEXT"), ("heldout_n", "INTEGER"), ("fires", "INTEGER"),
    ("right", "INTEGER"), ("precision", "REAL"), ("accuracy", "REAL"), ("kept", "INTEGER"), ("created", "TEXT"),
    ("provider", "TEXT"), ("mode", "TEXT"),
]

DOSSIERS = [
    ("id", "TEXT PRIMARY KEY"), ("attempt_id", "TEXT"), ("target", "TEXT"), ("claim", "TEXT"), ("conditions", "TEXT"),
    ("evidence", "TEXT"), ("valid", "INTEGER"), ("errors", "TEXT"), ("harness_attempt_id", "TEXT"), ("artifact_id", "TEXT"),
    ("created", "TEXT"), ("provider", "TEXT"), ("mode", "TEXT"),
]

TABLES = {"citations": CITATIONS, "lessons": LESSONS, "artifacts": ARTIFACTS, "provenance": PROVENANCE,
          "detectors": DETECTORS, "dossiers": DOSSIERS}

JSON_COLUMNS = {"lessons": ("conditions", "support", "contradictions", "targets", "provenance", "detectors"),
                "artifacts": ("meta",), "provenance": ("lesson_ids", "citation_ids"), "detectors": ("spec",),
                "dossiers": ("conditions", "evidence", "errors"), "citations": ()}


def ensure_all(ledger: Ledger) -> None:
    for name, cols in TABLES.items():
        ledger.ensure_table(name, cols)
