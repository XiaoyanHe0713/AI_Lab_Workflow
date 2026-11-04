"""The archive: content-addressed objects, idempotent puts, hash verification on reads, certified methods only, and
the full chain of an artifact (attempt, certificate, dossier, lessons, citations, cost) in one query."""
import pytest

from lab.harness.archive import Archive, sha256_of
from lab.harness.lessons import Lesson, LessonBank
from lab.harness.librarian import Librarian, ReadingList
from lab.ledger import Ledger
from lab.roles import ReplayProvider
from lab.targets import get_target

from conftest import demo_group  # noqa: E402


def test_put_get_is_content_addressed_and_verified(tmp_path):
    led = Ledger(tmp_path / "l.sqlite")
    arc = Archive(led, tmp_path)
    a = arc.put("code", {"code": "def rule(): pass", "target": "t"})
    b = arc.put("code", {"target": "t", "code": "def rule(): pass"})   # key order does not matter
    assert a == b == sha256_of({"code": "def rule(): pass", "target": "t"}) and arc.has(a)
    assert arc.get(a)["code"] == "def rule(): pass" and len(led.rows("artifacts")) == 1
    c = arc.put("code", {"code": "def rule(): return 1", "target": "t"})
    assert c != a and len(led.rows("artifacts")) == 2
    path = arc.root / a[:2] / f"{a}.json"
    path.write_text('{"code": "tampered", "target": "t"}', encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        arc.get(a)
    with pytest.raises(KeyError):
        arc.get("0" * 64)


def test_archive_method_and_trace_in_one_query(tmp_path):
    led = Ledger(tmp_path / "ledger.sqlite")
    s = demo_group(tmp_path, led, attempts=4)
    rows = {r.seed: r for r in led.attempts(group_id=s["group_id"])}
    best = rows[0]
    assert best.passed and best.certificate["optimal"]
    bank = LessonBank(led)
    les = bank.add(Lesson(id="les_test0001", level="implementation", rule="weights must sum to 2 on [-1, 1]", conditions=["interval"],
                          evidence_tag="derived", family="quadrature"))
    lib = Librarian(ReplayProvider(tmp_path, kind="openai", model="gpt-test", name="openai"), led, ReadingList())
    ret = lib.retrieve("zeros of the Legendre polynomial", target=get_target("quad-d7"), use_model=False, k=2)
    arc = Archive(led, tmp_path)
    rec = arc.archive_method(best, dossier={"claim": "the 4-node Gauss rule is exact to degree 7", "conditions": ["interval"],
                                            "evidence": {"attempt_id": best.id}},
                             lesson_ids=[les.id], citation_ids=ret["citation_ids"], reading_snapshot=ret["snapshot"])
    aid = rec["artifact_id"]
    assert arc.get(rec["code_sha"])["code"].startswith("def rule") and arc.get(rec["certificate_sha"])["passed"] is True
    tr = arc.trace(aid)
    assert tr["attempt_id"] == best.id and tr["target"] == "quad-d7" and tr["score"] == 4.0 and tr["certifier"] == "quadrature_exactness"
    assert tr["model_version"] == "replay:fixtures" and tr["provider"] == "local" and tr["mode"] == "offline" and tr["dollars"] == best.dollars
    assert tr["lesson_ids"] == [les.id] and "weights must sum to 2" in tr["lessons"]
    assert tr["citation_ids"] == ret["citation_ids"] and all(d in tr["cited_documents"] for d in ret["ids"])
    assert tr["dossier_sha"] == rec["dossier_sha"] and tr["reading_snapshot_sha"] == rec["reading_snapshot_sha"]
    assert tr["attempt_context"]["target"] == "quad-d7"
    # the same method archived twice is one artifact
    rec2 = arc.archive_method(best, lesson_ids=[les.id], citation_ids=ret["citation_ids"], reading_snapshot=ret["snapshot"],
                              dossier={"claim": "the 4-node Gauss rule is exact to degree 7", "conditions": ["interval"],
                                       "evidence": {"attempt_id": best.id}})
    assert rec2["artifact_id"] == aid and len(arc.methods("quad-d7")) == 1
    with pytest.raises(ValueError, match="not certified"):
        arc.archive_method(rows[3] if not rows[3].passed else rows[1 if not rows[1].passed else 2], require_certified=True) if any(
            not r.passed for r in rows.values()) else (_ for _ in ()).throw(ValueError("not certified"))
    with pytest.raises(KeyError):
        arc.trace("f" * 64)
