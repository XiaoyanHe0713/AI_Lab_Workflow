"""The librarian: holds the reading list with dates and the scoreboards, answers retrieval queries under a
per-target date cutoff, and logs every citation to the ledger.

The cutoff is enforced by the program twice: a document dated on or after the cutoff is never a candidate
(so the model never sees it), and an id the model returns that is not among the candidates is dropped and
flagged. The lexical ranker (token overlap weighted by inverse document frequency) is a program and works
without any model; the language part only chooses among the allowed candidates and writes a one-line
reason per choice plus a summary, which is information for the next prompt, never a score.

The reading list lives in reading_list/*.yaml, one file per family, each document with an id, a date
(YYYY-MM-DD; year precision when only the year is known) and a summary written by us. A reading-list
snapshot (the ids visible under a cutoff and their hash) is what an attempt records in its context.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml

from ..config import lab_config, repo_root
from ..ledger import Ledger, new_id
from .base import Harness
from .tables import CITATIONS, ensure_all

_TOKEN = re.compile(r"[a-z0-9]+")
STOP = {"the", "a", "an", "of", "for", "and", "or", "in", "on", "to", "with", "by", "is", "are", "at", "as", "from", "that",
        "this", "its", "it", "be", "which", "than", "into", "over", "under", "not", "no", "any", "all", "one", "two"}


def tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOP and len(t) > 1]


@dataclass
class Document:
    id: str
    title: str
    date: str                      # YYYY-MM-DD
    year: int
    authors: list[str] = field(default_factory=list)
    kind: str = "paper"
    venue: str = ""
    families: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    summary: str = ""
    results: list[str] = field(default_factory=list)
    scoreboard_ids: list[str] = field(default_factory=list)
    date_precision: str = "year"
    source_file: str = ""

    @property
    def text(self) -> str:
        return " ".join([self.title, " ".join(self.authors), " ".join(self.topics), self.summary, " ".join(self.results)])

    def card(self) -> str:
        return f"[{self.id}] {self.year}: {self.title}. {self.summary}"

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        return d


def _date_of(entry: dict) -> tuple[str, int, str]:
    if entry.get("date"):
        d = str(entry["date"])
        return d, int(d[:4]), str(entry.get("date_precision", "day"))
    y = int(entry["year"])
    return f"{y:04d}-01-01", y, "year"


class ReadingList:
    def __init__(self, root: str | Path | None = None, cfg: dict | None = None):
        cfg = cfg or lab_config()
        rel = cfg.get("harness", {}).get("reading_list", "reading_list")
        self.root = Path(root) if root else (Path(rel) if Path(rel).is_absolute() else repo_root() / rel)
        self.docs: dict[str, Document] = {}
        for path in sorted(self.root.glob("*.yaml")):
            if path.name == "retrieval_tests.yaml":
                continue
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            for e in data.get("documents", []):
                date, year, prec = _date_of(e)
                strs = lambda key: [str(x) for x in (e.get(key) or [])]  # noqa: E731  (YAML may parse a topic as a number)
                doc = Document(id=str(e["id"]), title=str(e["title"]), date=date, year=year, authors=strs("authors"),
                               kind=str(e.get("kind", "paper")), venue=str(e.get("venue", "")), families=strs("families"),
                               topics=strs("topics"), summary=str(e.get("summary", "")).strip(),
                               results=strs("results"), scoreboard_ids=strs("scoreboard_ids"),
                               date_precision=prec, source_file=path.name)
                if doc.id in self.docs:
                    raise ValueError(f"duplicate document id {doc.id!r} in {path.name}")
                self.docs[doc.id] = doc
        self._df: dict[str, int] = {}
        for d in self.docs.values():
            for t in set(tokens(d.text)):
                self._df[t] = self._df.get(t, 0) + 1

    def __len__(self) -> int:
        return len(self.docs)

    def get(self, doc_id: str) -> Document:
        return self.docs[doc_id]

    def documents(self, cutoff: str | None = None, family: str | None = None) -> list[Document]:
        """Documents dated strictly before the cutoff (and of the family when given), oldest first."""
        out = []
        for d in self.docs.values():
            if cutoff is not None and d.date >= str(cutoff):
                continue
            if family is not None and family not in d.families:
                continue
            out.append(d)
        return sorted(out, key=lambda d: (d.date, d.id))

    def excluded(self, cutoff: str | None, family: str | None = None) -> list[Document]:
        if cutoff is None:
            return []
        return [d for d in self.docs.values() if d.date >= str(cutoff) and (family is None or family in d.families)]

    def snapshot(self, cutoff: str | None = None, family: str | None = None) -> dict:
        docs = self.documents(cutoff, family)
        blob = "\n".join(f"{d.id}:{d.date}" for d in docs)
        return {"cutoff": cutoff, "family": family, "count": len(docs), "ids": [d.id for d in docs],
                "sha256": hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]}

    def rank(self, query: str, docs: list[Document], k: int = 10) -> list[tuple[Document, float]]:
        """Lexical relevance: sum over query tokens of idf * (1 + log tf), normalized by document length."""
        q = tokens(query)
        n = max(1, len(self.docs))
        scored = []
        for d in docs:
            dt = tokens(d.text)
            if not dt:
                continue
            tf: dict[str, int] = {}
            for t in dt:
                tf[t] = tf.get(t, 0) + 1
            s = 0.0
            for t in q:
                if t in tf:
                    idf = math.log(1 + n / (1 + self._df.get(t, 0)))
                    s += idf * (1 + math.log(tf[t]))
            if s > 0:
                scored.append((d, s / math.sqrt(len(dt))))
        scored.sort(key=lambda x: (-x[1], x[0].date, x[0].id))
        return scored[:k]


class Librarian(Harness):
    name = "librarian"
    system_prompt = ("You are the librarian of an automated numerical-analysis lab. You only ever cite documents from the list "
                     "you are given, by their ids in square brackets; you never add documents, dates or results from memory. "
                     "Answer with JSON only.")

    def __init__(self, provider, ledger: Ledger, reading_list: ReadingList | None = None, *, cfg: dict | None = None, **kw):
        super().__init__(provider, ledger, cfg=cfg, **kw)
        self.reading_list = reading_list or ReadingList(cfg=self.cfg)
        ensure_all(self.ledger)

    # --- retrieval -------------------------------------------------------------------------------
    def retrieve(self, query: str, *, target=None, cutoff: str | None = None, family: str | None = None, k: int = 5,
                 n_candidates: int = 10, use_model: bool = True, purpose: str = "retrieval") -> dict:
        target_id = getattr(target, "id", None) if target is not None else None
        if target is not None and cutoff is None:
            cutoff = getattr(target, "cutoff", None)
        allowed = self.reading_list.documents(cutoff, family)
        excluded = self.reading_list.excluded(cutoff, family)
        ranked = self.reading_list.rank(query, allowed, n_candidates)
        candidates = [d for d, _ in ranked]
        lexical = [{"id": d.id, "score": round(s, 4)} for d, s in ranked]
        selected: list[dict] = []
        rejected: list[str] = []
        summary = ""
        reply = None
        if use_model and candidates:
            cards = "\n".join(f"{i + 1}. {d.card()}" for i, d in enumerate(candidates))
            user = (f"Query: {query}\n\nCandidate documents (the only ones you may cite):\n{cards}\n\n"
                    f"Choose up to {k} documents most relevant to the query, most relevant first, and give one short reason each. "
                    f"Then summarize in at most three sentences what these documents establish about the query, citing ids in "
                    f"square brackets. Reply with JSON: {{\"selected\": [{{\"id\": \"...\", \"why\": \"...\"}}], \"summary\": \"...\"}}")
            reply = self.ask("retrieve", user, target=target_id or f"query:{hashlib.sha256(query.encode()).hexdigest()[:8]}",
                             context={"query": query, "cutoff": cutoff, "family": family, "candidates": [d.id for d in candidates],
                                      "purpose": purpose})
            data = reply.data if isinstance(reply.data, dict) else {}
            allowed_ids = {d.id for d in candidates}
            for item in (data.get("selected") or [])[:k]:
                did = str(item.get("id", "")).strip("[] ") if isinstance(item, dict) else str(item).strip("[] ")
                if did in allowed_ids and did not in {s["id"] for s in selected}:
                    selected.append({"id": did, "why": str(item.get("why", "")) if isinstance(item, dict) else ""})
                else:
                    rejected.append(did)
            summary = str(data.get("summary", "")) if data else ""
            selected_by = "model"
            if not selected:  # the model failed or chose nothing usable: the lexical ranking stands
                selected = [{"id": d.id, "why": "lexical rank"} for d in candidates[:k]]
                selected_by = "lexical-fallback"
        else:
            selected = [{"id": d.id, "why": "lexical rank"} for d in candidates[:k]]
            selected_by = "lexical"
        # the cutoff, enforced again on what is returned
        selected = [s for s in selected if self.reading_list.get(s["id"]).date < str(cutoff)] if cutoff else selected
        citation_ids = []
        now = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        for rank, s in enumerate(selected, start=1):
            d = self.reading_list.get(s["id"])
            cid = new_id("cit")
            self.ledger.insert("citations", {
                "id": cid, "timestamp": now, "harness_attempt_id": reply.attempt_id if reply else None, "target": target_id,
                "query": query, "cutoff": cutoff, "doc_id": d.id, "doc_date": d.date, "rank": rank, "selected_by": selected_by,
                "allowed": 1, "provider": self.provider.name, "mode": self.provider.mode, "purpose": purpose})
            citation_ids.append(cid)
        return {"query": query, "target": target_id, "cutoff": cutoff, "family": family,
                "documents": [{**self.reading_list.get(s["id"]).to_dict(), "why": s["why"]} for s in selected],
                "ids": [s["id"] for s in selected], "selected_by": selected_by, "candidates": [d.id for d in candidates],
                "lexical": lexical, "rejected_ids": rejected, "excluded_after_cutoff": len(excluded),
                "excluded_ids": sorted(d.id for d in excluded), "summary": summary, "citation_ids": citation_ids,
                "snapshot": self.reading_list.snapshot(cutoff, family), "harness_attempt_id": reply.attempt_id if reply else None,
                "api_dollars": reply.api_dollars if reply else 0.0}

    def citations(self, target: str | None = None) -> list[dict]:
        return self.ledger.rows("citations", "target = ?" if target else "", (target,) if target else (), order="timestamp, rank")

    # --- the provisional-entry pass -----------------------------------------------------------------
    def review_provisional(self, board_name: str, *, k_docs: int = 6) -> list[dict]:
        """For every scoreboard entry marked provenance: provisional, gather the reading-list documents its source
        text points to, ask the model whether the entry is consistent with them, and return one review per entry
        (consistent | inconsistent | cannot verify, with the reason and the documents relied on). The scoreboard is
        never edited here: a person applies a reviewed diff."""
        path = repo_root() / "scoreboard" / f"{board_name}.yaml"
        with open(path, "r", encoding="utf-8") as f:
            board = yaml.safe_load(f)
        reviews = []
        for section, entry in _walk_entries(board):
            if entry.get("provenance") != "provisional":
                continue
            query = " ".join(str(entry.get(k, "")) for k in ("method", "source", "class", "domain", "rule_class"))
            ranked = self.reading_list.rank(query, list(self.reading_list.docs.values()), k_docs)
            docs = [d for d, _ in ranked]
            cards = "\n".join(f"{i + 1}. {d.card()} Results: " + "; ".join(d.results) for i, d in enumerate(docs))
            fields = {k: entry[k] for k in ("id", "degree", "best_known", "status", "method", "source", "stages", "order", "n", "axis")
                      if k in entry}
            user = (f"Scoreboard {board_name}, section {section}, entry:\n{json.dumps(fields, indent=1, default=str)}\n\n"
                    f"Documents from the reading list (summaries written by the lab; cite only these):\n{cards}\n\n"
                    f"Is the entry's best known value, its stated source and its solved/open status consistent with these "
                    f"documents? Reply with JSON: {{\"verdict\": \"consistent\" | \"inconsistent\" | \"cannot verify\", "
                    f"\"reason\": \"...\", \"documents\": [\"id\", ...], \"suggested_change\": \"...\" }}. Say 'cannot verify' when "
                    f"the documents do not state the value; never confirm from memory.")
            reply = self.ask("review_provisional", user, target=f"scoreboard:{board_name}:{entry['id']}",
                             context={"board": board_name, "entry": entry["id"], "candidates": [d.id for d in docs]})
            data = reply.data if isinstance(reply.data, dict) else {}
            verdict = str(data.get("verdict", "cannot verify")).lower()
            if verdict not in ("consistent", "inconsistent", "cannot verify"):
                verdict = "cannot verify"
            cited = [str(x).strip("[] ") for x in (data.get("documents") or []) if str(x).strip("[] ") in {d.id for d in docs}]
            now = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
            for rank, did in enumerate(cited, start=1):
                self.ledger.insert("citations", {"id": new_id("cit"), "timestamp": now, "harness_attempt_id": reply.attempt_id,
                                                 "target": f"scoreboard:{board_name}:{entry['id']}", "query": query[:200], "cutoff": None,
                                                 "doc_id": did, "doc_date": self.reading_list.get(did).date, "rank": rank,
                                                 "selected_by": "model", "allowed": 1, "provider": self.provider.name,
                                                 "mode": self.provider.mode, "purpose": "review_provisional"})
            reviews.append({"board": board_name, "section": section, "entry": entry["id"], "best_known": entry.get("best_known"),
                            "verdict": verdict, "reason": str(data.get("reason", "")), "documents": cited,
                            "suggested_change": str(data.get("suggested_change", "")), "harness_attempt_id": reply.attempt_id,
                            "provider": self.provider.name, "mode": self.provider.mode, "api_dollars": reply.api_dollars,
                            "note": "a model's reading of our own summaries; a person checks the source before any change"})
        return reviews


def _walk_entries(board: dict):
    for key, val in board.items():
        if isinstance(val, dict) and isinstance(val.get("entries"), list):
            for e in val["entries"]:
                yield key, e
        elif key == "entries" and isinstance(val, list):
            for e in val:
                yield "entries", e


def load_retrieval_tests(root: Path | None = None) -> list[dict]:
    rl = ReadingList(root)
    path = rl.root / "retrieval_tests.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["tests"]


def main(argv=None) -> int:
    import argparse

    from .base import harness_provider

    ap = argparse.ArgumentParser(prog="python -m lab.harness.librarian")
    ap.add_argument("--query", default=None)
    ap.add_argument("--target", default=None)
    ap.add_argument("--cutoff", default=None)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--review", default=None, help="scoreboard name whose provisional entries to review (e.g. quadrature)")
    ap.add_argument("--provider", default=None)
    ap.add_argument("--mode", default=None)
    ap.add_argument("--ledger", default=None)
    ap.add_argument("--no-model", action="store_true", help="lexical ranking only")
    args = ap.parse_args(argv)
    cfg = lab_config()
    ledger = Ledger(args.ledger or cfg["storage"]["ledger"])
    provider = harness_provider(cfg, name=args.provider, mode=args.mode, ledger=ledger)
    lib = Librarian(provider, ledger, cfg=cfg)
    if args.review:
        reviews = lib.review_provisional(args.review)
        out = repo_root() / "scoreboard" / "reviews"
        out.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
        p = out / f"{args.review}.{stamp}.{provider.name}.yaml"
        with open(p, "w", encoding="utf-8") as f:
            yaml.safe_dump({"reviews": reviews}, f, sort_keys=False, allow_unicode=True)
        print(f"{len(reviews)} reviews written to {p}")
        for r in reviews:
            print(f"  {r['entry']}: {r['verdict']} ({', '.join(r['documents']) or 'no documents'}) {r['reason'][:120]}")
    if args.query:
        from ..targets import get_target

        target = get_target(args.target) if args.target else None
        res = lib.retrieve(args.query, target=target, cutoff=args.cutoff, k=args.k, use_model=not args.no_model)
        print(json.dumps({k: res[k] for k in ("query", "cutoff", "ids", "selected_by", "excluded_after_cutoff", "summary", "api_dollars")},
                         indent=1))
        for d in res["documents"]:
            print(f"  [{d['id']}] {d['year']} {d['title']} :: {d['why']}")
    ledger.close()
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
