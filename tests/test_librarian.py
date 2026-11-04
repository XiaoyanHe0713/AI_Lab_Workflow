"""The librarian: the reading list loads with dates and unique ids, the lexical ranker passes the retrieval test
set, the cutoff never leaks a later document (neither to the model nor in the result), every citation is logged,
and the model-chosen retrieval and the provisional-entry pass replay from recorded fixtures per provider."""
import re
from pathlib import Path

import pytest
import yaml

from lab.harness.librarian import Librarian, ReadingList, load_retrieval_tests, tokens
from lab.ledger import Ledger
from lab.roles import ReplayProvider
from lab.targets import get_target

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def reading_list():
    return ReadingList()


def _scoreboard_ids():
    ids = set()
    for p in (ROOT / "scoreboard").glob("*.yaml"):
        board = yaml.safe_load(p.read_text(encoding="utf-8"))
        for key, val in board.items():
            if isinstance(val, dict) and isinstance(val.get("entries"), list):
                ids |= {e["id"] for e in val["entries"]}
            elif key == "entries":
                ids |= {e["id"] for e in val}
    return ids


def test_reading_list_loads_with_dates_unique_ids_and_known_scoreboard_links(reading_list):
    assert len(reading_list) >= 80
    boards = _scoreboard_ids()
    for d in reading_list.docs.values():
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", d.date), d.id
        assert d.year == int(d.date[:4]) and d.families and d.summary and d.title
        assert d.date_precision in ("day", "month", "year")
        for sid in d.scoreboard_ids:
            assert sid in boards, f"{d.id} links to an unknown scoreboard entry {sid}"
    assert sorted(reading_list.documents(cutoff="1800-01-01"), key=lambda d: d.date)[-1].date < "1800-01-01"


def test_lexical_retrieval_passes_the_test_set_and_never_leaks(tmp_path, reading_list):
    lib = Librarian(ReplayProvider(tmp_path, kind="openai", model="gpt-test", name="openai"), Ledger(tmp_path / "l.sqlite"), reading_list)
    for case in load_retrieval_tests():
        target = get_target(case["target"]) if case.get("target") else None
        res = lib.retrieve(case["query"], target=target, cutoff=case.get("cutoff"), k=5, use_model=False)
        assert any(i in res["ids"] for i in case["expect_any"]), (case["id"], res["ids"])
        assert not set(case["forbid"]) & set(res["ids"]) and not set(case["forbid"]) & set(res["candidates"]), case["id"]
        cutoff = res["cutoff"]
        if cutoff:
            assert all(reading_list.get(i).date < cutoff for i in res["candidates"])
            assert all(reading_list.get(i).date >= cutoff for i in res["excluded_ids"])
            assert res["snapshot"]["count"] == len(reading_list.documents(cutoff)) and set(res["snapshot"]["ids"]).isdisjoint(res["excluded_ids"])
        assert res["selected_by"] == "lexical" and len(res["citation_ids"]) == len(res["ids"])
    cits = lib.citations()
    assert len(cits) == sum(5 if True else 0 for _ in load_retrieval_tests()) or len(cits) >= len(load_retrieval_tests())
    for c in cits:
        assert c["doc_id"] in reading_list.docs and c["allowed"] == 1 and c["selected_by"] == "lexical"
        if c["cutoff"]:
            assert c["doc_date"] < c["cutoff"]


def test_snapshot_hash_changes_with_the_cutoff(reading_list):
    a, b = reading_list.snapshot("1814-01-01"), reading_list.snapshot("1815-01-01")
    assert a["sha256"] != b["sha256"] and a["count"] + 1 == b["count"] and "gauss-1814" in b["ids"] and "gauss-1814" not in a["ids"]
    assert reading_list.snapshot(None)["count"] == len(reading_list)


def test_tokens_drop_stop_words():
    assert tokens("The zeros of the Legendre polynomial") == ["zeros", "legendre", "polynomial"]


def test_model_retrieval_replays_and_logs_citations(frontier, reading_list):
    lib = Librarian(frontier.provider, frontier.ledger, reading_list)
    for case in load_retrieval_tests()[:4]:
        target = get_target(case["target"]) if case.get("target") else None
        res = lib.retrieve(case["query"], target=target, cutoff=case.get("cutoff"), k=4, use_model=True)
        assert res["ids"] and set(res["ids"]) <= set(res["candidates"]), (case["id"], res["ids"], res["rejected_ids"])
        assert not set(case["forbid"]) & set(res["ids"]) and not set(case["forbid"]) & set(res["candidates"])
        assert res["selected_by"] in ("model", "lexical-fallback") and len(res["citation_ids"]) == len(res["ids"])
        assert res["harness_attempt_id"]
    rows = frontier.ledger.attempts(role="harness:librarian")
    assert len(rows) == 4 and all(r.provider == frontier.name and r.mode == frontier.mode for r in rows)
    if frontier.recorded:
        assert all(r.api_dollars == 0.0 for r in rows) and frontier.ledger.api_spend(2026, 11) == 0.0
    else:
        assert all(r.api_dollars > 0 for r in rows)
    cits = [c for c in lib.citations() if c["purpose"] == "retrieval"]
    assert cits and all(c["harness_attempt_id"] in {r.id for r in rows} for c in cits)
    assert all(c["provider"] == frontier.name and c["mode"] == frontier.mode for c in cits)


def test_provisional_scoreboard_pass(frontier, reading_list):
    lib = Librarian(frontier.provider, frontier.ledger, reading_list)
    reviews = lib.review_provisional("quadrature")
    board = yaml.safe_load((ROOT / "scoreboard" / "quadrature.yaml").read_text(encoding="utf-8"))
    provisional = [e["id"] for sec in board.values() if isinstance(sec, dict) for e in sec.get("entries", []) if e.get("provenance") == "provisional"]
    assert [r["entry"] for r in reviews] == provisional and len(provisional) >= 5
    for r in reviews:
        assert r["verdict"] in ("consistent", "inconsistent", "cannot verify")
        assert set(r["documents"]) <= set(reading_list.docs) and r["harness_attempt_id"]
        assert r["provider"] == frontier.name and r["mode"] == frontier.mode
    cits = [c for c in lib.citations() if c["purpose"] == "review_provisional"]
    assert len(cits) == sum(len(r["documents"]) for r in reviews)
    # the scoreboard file itself is untouched: a person applies a reviewed diff
    assert yaml.safe_load((ROOT / "scoreboard" / "quadrature.yaml").read_text(encoding="utf-8")) == board
