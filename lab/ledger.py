"""The ledger: one row per attempt, with provenance and cost (proposal, Work package 4).

Attempt record columns, in the order the proposal lists them:
    id, role, model + model_version (adapter version), precision, target, prompt_hash,
    context (lesson ids, skill ids, reading-list snapshot), output, tool_log,
    certificate (certifier version, score, pass or fail, witness), diagnosis,
    gpu_seconds, cpu_seconds, dollars, seed, timestamp.
Three columns say where the output came from, so a replayed response can never be mistaken for a live
one: provider (local, openai, anthropic), mode (offline, online, record) and api_dollars (the frontier
tokens priced, zero offline). Two more carry grouping and free metadata: group_id and meta.

Storage is SQLite (queryable; the transfer matrix is a query over this table) plus a JSONL mirror
next to it, because a FUSE-mounted Drive can lose SQLite journals. A claim that cannot be traced
back through the ledger is not reported.
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 2

ATTEMPT_COLUMNS: list[tuple[str, str]] = [
    ("id", "TEXT PRIMARY KEY"),
    ("role", "TEXT NOT NULL"),
    ("model", "TEXT NOT NULL"),
    ("model_version", "TEXT NOT NULL"),
    ("precision", "TEXT NOT NULL"),
    ("target", "TEXT NOT NULL"),
    ("prompt_hash", "TEXT NOT NULL"),
    ("context", "TEXT NOT NULL"),
    ("output", "TEXT"),
    ("tool_log", "TEXT"),
    ("certificate", "TEXT"),
    ("diagnosis", "TEXT"),
    ("gpu_seconds", "REAL"),
    ("cpu_seconds", "REAL"),
    ("dollars", "REAL"),
    ("seed", "INTEGER"),
    ("timestamp", "TEXT NOT NULL"),
    ("provider", "TEXT"),
    ("mode", "TEXT"),
    ("api_dollars", "REAL"),
    ("group_id", "TEXT"),
    ("meta", "TEXT"),
]

SERVING_COLUMNS: list[tuple[str, str]] = [
    ("id", "TEXT PRIMARY KEY"),
    ("timestamp", "TEXT NOT NULL"),
    ("gpu", "TEXT NOT NULL"),
    ("precision", "TEXT NOT NULL"),
    ("model", "TEXT NOT NULL"),
    ("model_version", "TEXT NOT NULL"),
    ("server", "TEXT"),
    ("adapter", "TEXT"),
    ("prompt_tokens", "INTEGER"),
    ("completion_tokens", "INTEGER"),
    ("latency_s", "REAL"),
    ("tokens_per_second", "REAL"),
    ("units_per_hour", "REAL"),
    ("units_source", "TEXT"),
    ("dollars_per_hour", "REAL"),
    ("dollars", "REAL"),
    ("ok", "INTEGER"),
    ("meta", "TEXT"),
]

_JSON_COLUMNS = {"context", "tool_log", "certificate", "meta"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def new_id(prefix: str = "att") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def prompt_hash(messages: Any) -> str:
    """Stable hash of a prompt (a string or a list of chat messages)."""
    blob = json.dumps(messages, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(blob).hexdigest()


def _json_default(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return str(o)
    return str(o)


@dataclass
class Attempt:
    role: str
    model: str
    model_version: str
    precision: str
    target: str
    prompt_hash: str
    context: dict = field(default_factory=dict)
    output: str | None = None
    tool_log: list = field(default_factory=list)
    certificate: dict | None = None
    diagnosis: str | None = None
    gpu_seconds: float = 0.0
    cpu_seconds: float = 0.0
    dollars: float = 0.0
    seed: int | None = None
    timestamp: str = field(default_factory=utc_now)
    provider: str | None = None        # local | openai | anthropic (the provider name in config/lab.yaml)
    mode: str | None = None            # offline | online | record
    api_dollars: float = 0.0           # frontier tokens priced; counts against the monthly budget
    group_id: str | None = None
    meta: dict = field(default_factory=dict)
    id: str = field(default_factory=new_id)

    @property
    def passed(self) -> bool:
        return bool(self.certificate and self.certificate.get("passed"))

    @property
    def score(self) -> float | None:
        if not self.certificate:
            return None
        return self.certificate.get("score")

    def to_row(self) -> dict:
        d = asdict(self)
        for k in _JSON_COLUMNS:
            d[k] = json.dumps(d[k], ensure_ascii=False, sort_keys=True, default=_json_default)
        return d

    @classmethod
    def from_row(cls, row: dict) -> "Attempt":
        d = dict(row)
        for k in _JSON_COLUMNS:
            v = d.get(k)
            d[k] = json.loads(v) if isinstance(v, str) else v
        if d.get("context") is None:
            d["context"] = {}
        if d.get("tool_log") is None:
            d["tool_log"] = []
        if d.get("meta") is None:
            d["meta"] = {}
        return cls(**d)


def _ddl(table: str, columns: list[tuple[str, str]]) -> str:
    cols = ", ".join(f"{name} {typ}" for name, typ in columns)
    return f"CREATE TABLE IF NOT EXISTS {table} ({cols})"


class Ledger:
    """SQLite ledger with a JSONL mirror. One writer at a time; journal_mode=DELETE works on Drive."""

    def __init__(self, path: str | Path, mirror: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.mirror_path = self.path.with_suffix(".attempts.jsonl") if mirror else None
        self.conn = sqlite3.connect(str(self.path), timeout=30, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=DELETE")
        self.conn.execute("PRAGMA synchronous=FULL")
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.execute(_ddl("attempts", ATTEMPT_COLUMNS))
        self.conn.execute(_ddl("serving", SERVING_COLUMNS))
        # forward migration: add columns a newer schema introduced (v2: provider, mode, api_dollars)
        existing = {r[1] for r in self.conn.execute("PRAGMA table_info(attempts)").fetchall()}
        for name, typ in ATTEMPT_COLUMNS:
            if name not in existing:
                self.conn.execute(f"ALTER TABLE attempts ADD COLUMN {name} {typ.replace('NOT NULL', '').replace('PRIMARY KEY', '')}")
        self.conn.execute("CREATE INDEX IF NOT EXISTS ix_attempts_target ON attempts(target)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS ix_attempts_group ON attempts(group_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS ix_attempts_provider ON attempts(provider, mode)")
        self.conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    def close(self) -> None:
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # --- attempts -------------------------------------------------------------------------
    def add_attempt(self, attempt: Attempt) -> str:
        row = attempt.to_row()
        names = [c for c, _ in ATTEMPT_COLUMNS]
        placeholders = ", ".join("?" for _ in names)
        self.conn.execute(
            f"INSERT INTO attempts ({', '.join(names)}) VALUES ({placeholders})",
            [row[n] for n in names],
        )
        if self.mirror_path is not None:
            with open(self.mirror_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"table": "attempts", **row}, ensure_ascii=False) + "\n")
        return attempt.id

    def get_attempt(self, attempt_id: str) -> Attempt | None:
        cur = self.conn.execute("SELECT * FROM attempts WHERE id=?", (attempt_id,))
        row = cur.fetchone()
        return Attempt.from_row(dict(row)) if row else None

    def attempts(self, target: str | None = None, group_id: str | None = None,
                 role: str | None = None) -> list[Attempt]:
        clauses, args = [], []
        if target is not None:
            clauses.append("target=?")
            args.append(target)
        if group_id is not None:
            clauses.append("group_id=?")
            args.append(group_id)
        if role is not None:
            clauses.append("role=?")
            args.append(role)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        cur = self.conn.execute(f"SELECT * FROM attempts{where} ORDER BY timestamp, id", args)
        return [Attempt.from_row(dict(r)) for r in cur.fetchall()]

    def count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM attempts").fetchone()[0])

    def api_spend(self, year: int, month: int, kinds: tuple[str, ...] = ("openai", "anthropic"),
                  modes: tuple[str, ...] = ("online", "record")) -> float:
        """This calendar month's frontier spend: the API dollars of live or recorded frontier rows."""
        ym = f"{int(year):04d}-{int(month):02d}"
        qk = ", ".join("?" for _ in kinds)
        qm = ", ".join("?" for _ in modes)
        cur = self.conn.execute(
            f"SELECT COALESCE(SUM(api_dollars), 0) FROM attempts WHERE provider IN ({qk}) AND mode IN ({qm}) "
            f"AND substr(timestamp, 1, 7) = ?", [*kinds, *modes, ym])
        return float(cur.fetchone()[0] or 0.0)

    # --- serving measurements ----------------------------------------------------------------
    def add_serving(self, **fields: Any) -> str:
        fields.setdefault("id", new_id("srv"))
        fields.setdefault("timestamp", utc_now())
        fields["meta"] = json.dumps(fields.get("meta") or {}, ensure_ascii=False, sort_keys=True,
                                    default=_json_default)
        names = [c for c, _ in SERVING_COLUMNS]
        values = [fields.get(n) for n in names]
        placeholders = ", ".join("?" for _ in names)
        self.conn.execute(f"INSERT INTO serving ({', '.join(names)}) VALUES ({placeholders})", values)
        if self.mirror_path is not None:
            with open(self.mirror_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"table": "serving", **dict(zip(names, values))}, ensure_ascii=False) + "\n")
        return fields["id"]

    def serving(self) -> list[dict]:
        cur = self.conn.execute("SELECT * FROM serving ORDER BY timestamp")
        out = []
        for r in cur.fetchall():
            d = dict(r)
            d["meta"] = json.loads(d["meta"]) if d.get("meta") else {}
            out.append(d)
        return out

    # --- harness tables (citations, lessons, artifacts, provenance, detectors, dossiers) ---------------
    def ensure_table(self, table: str, columns: list[tuple[str, str]]) -> None:
        """Create a harness table if missing and add any column a newer schema introduced."""
        if not table.isidentifier():
            raise ValueError(table)
        self.conn.execute(_ddl(table, columns))
        existing = {r[1] for r in self.conn.execute(f"PRAGMA table_info({table})").fetchall()}
        for name, typ in columns:
            if name not in existing:
                self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {typ.replace('NOT NULL', '').replace('PRIMARY KEY', '')}")

    def insert(self, table: str, row: dict, json_columns: tuple[str, ...] = ()) -> None:
        """Insert or replace one row (dicts and lists in json_columns are serialized); mirrored to the JSONL file."""
        if not table.isidentifier():
            raise ValueError(table)
        row = dict(row)
        for k in json_columns:
            if k in row and not isinstance(row[k], str):
                row[k] = json.dumps(row[k], ensure_ascii=False, sort_keys=True, default=_json_default)
        names = list(row)
        placeholders = ", ".join("?" for _ in names)
        self.conn.execute(f"INSERT OR REPLACE INTO {table} ({', '.join(names)}) VALUES ({placeholders})",
                          [row[n] for n in names])
        if self.mirror_path is not None:
            with open(self.mirror_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"table": table, **row}, ensure_ascii=False, default=_json_default) + "\\n")

    def rows(self, table: str, where: str = "", args: tuple | list = (), json_columns: tuple[str, ...] = (),
             order: str = "") -> list[dict]:
        if not table.isidentifier():
            raise ValueError(table)
        sql = f"SELECT * FROM {table}" + (f" WHERE {where}" if where else "") + (f" ORDER BY {order}" if order else "")
        out = []
        for r in self.conn.execute(sql, list(args)).fetchall():
            d = dict(r)
            for k in json_columns:
                if isinstance(d.get(k), str):
                    try:
                        d[k] = json.loads(d[k])
                    except ValueError:
                        pass
            out.append(d)
        return out

    def query(self, sql: str, args: tuple | list = ()) -> list[dict]:
        return [dict(r) for r in self.conn.execute(sql, list(args)).fetchall()]

    # --- summaries --------------------------------------------------------------------------
    def group_summary(self, group_id: str) -> dict:
        return summarize(self.attempts(group_id=group_id))


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al. 2021): 1 - C(n-c, k) / C(n, k), with k capped at n."""
    if n <= 0 or k <= 0:
        return float("nan")
    k = min(k, n)
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def summarize(rows: Iterable[Attempt], ks: tuple[int, ...] = (1, 16, 64)) -> dict:
    """Group statistics: pass counts, pass@k for valid and for optimal attempts, the best attempt, cost."""
    rows = list(rows)
    n = len(rows)
    passed = [r for r in rows if r.passed]
    c = len(passed)
    optimal = [r for r in passed if (r.certificate or {}).get("optimal") is True]
    co = len(optimal)
    scores = [r.score for r in passed if r.score is not None]
    best = None
    if passed:
        lower = all((r.certificate or {}).get("lower_is_better", True) for r in passed)

        def key(r):
            s = r.score if r.score is not None else (math.inf if lower else -math.inf)
            return s if lower else -s

        best = min(passed, key=key)
    ks_used = [k for k in ks if k <= max(n, 1)] or [1]
    return {
        "attempts": n,
        "passed": c,
        "optimal": co,
        "pass_at_k": {f"pass@{k}": pass_at_k(n, c, k) for k in ks_used},
        "optimal_at_k": {f"pass@{k}": pass_at_k(n, co, k) for k in ks_used},
        "best_attempt_id": best.id if best else None,
        "best_score": best.score if best else None,
        "scores": scores,
        "gpu_seconds": sum(r.gpu_seconds for r in rows),
        "cpu_seconds": sum(r.cpu_seconds for r in rows),
        "dollars": sum(r.dollars for r in rows),
    }
