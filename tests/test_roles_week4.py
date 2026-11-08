"""The three roles as episodes: the theorist's proposals with lemma checks and a revision round (replayed fixtures),
the builder's certifier dry run with a revision (a stub endpoint), the breaker's witness reward on synthetic breaks and
non-breaks (a stub endpoint): one for a confirmed break, zero for a non-break, an instance outside the class, large
coefficients, a malformed witness; the probe round; a first-order rate violation; and the ledger rows every episode
leaves behind."""
import json
from pathlib import Path

import pytest

from lab.certifiers.quadrature import corrupt_weight, gauss_legendre
from lab.config import lab_config
from lab.demo import prepare
from lab.episode import run_episodes
from lab.ledger import Ledger
from lab.roles import Completion, Role
from lab.roles.breaker import BREAKER_SYSTEM, BreakerRunner, breaker_prompt, candidate_from_row, complete_witness, parse_witness
from lab.roles.builder import BuilderRunner, dry_run_feedback, with_proposal
from lab.roles.theorist import THEORIST_SYSTEM, TheoristRunner, parse_proposal, proposal_from_row, theorist_prompt
from lab.targets import SYSTEM_PROMPT, get_target

from fixtures_week4 import SQ_RADON7, SQ_TENSOR9, THEORIST_EXPECT  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


class _Script:
    """A stub endpoint that answers round r of a conversation with scripts[seed][r]."""
    kind = "local"
    name = "local"
    mode = "online"
    model = "qwen3.6-27b"
    base_url = "http://stub/v1"

    def __init__(self, scripts):
        self.scripts = scripts
        self.requests = []

    def complete(self, messages, *, seed=None, **kw):
        self.requests.append((seed, messages))
        turn = sum(1 for m in messages if m["role"] == "assistant")
        text = self.scripts[seed % len(self.scripts)][turn]
        return Completion(text=text, prompt_tokens=300 + 100 * turn, completion_tokens=50, latency_s=0.5, model=self.model,
                          provider="local", finish_reason="stop")

    def request(self, seed, turn):
        """The messages of round `turn` (0-based) for a seed, whatever order the waves sent them in."""
        for s, m in self.requests:
            if s == seed and sum(1 for x in m if x["role"] == "assistant") == turn:
                return m
        raise KeyError((seed, turn))


def _setup(tmp_path, role_name, provider, system_prompt=None):
    cfg = lab_config(env={"LAB_ROOT": str(tmp_path)})
    led = Ledger(tmp_path / "ledger.sqlite")
    s = prepare(cfg=cfg, role_name=role_name, provider=provider, gpu="A100", precision="bf16", temperature=None,
                system_prompt=system_prompt, cache_dir=tmp_path / "cache", timeout_s=60)
    return cfg, led, s


def _run(s, runner, led, target, n, role_name, tool_calls=0, extra=None):
    return run_episodes(runner.attempt, role_name=role_name, role=s["role"], provider=s["provider"], provider_name="local", target=target,
                        attempts=n, seed=0, ledger=led, meter=s["meter"], model=s["model"], model_version=s["model_version"],
                        precision=s["precision"], gpu="A100", concurrency=2, quiet=True, tool_calls=tool_calls, context_extra=extra)


# --- the theorist -------------------------------------------------------------------------------------------------------
def test_theorist_proposals_from_fixtures_with_a_revision_round(tmp_path):
    cfg = lab_config(env={"LAB_ROOT": str(tmp_path)})
    led = Ledger(tmp_path / "ledger.sqlite")
    s = prepare(cfg=cfg, role_name="theorist", provider_name="local", mode="offline", fixtures_root=ROOT, ledger=led, gpu="A100",
                temperature=None, system_prompt=THEORIST_SYSTEM, cache_dir=tmp_path / "cache", timeout_s=60)
    target = get_target("quad-square-d5")
    from lab.harness.librarian import Librarian, ReadingList
    from lab.loop import theorist_retrieval

    ret = theorist_retrieval(Librarian(s["provider"], led, ReadingList(cfg=cfg), cfg=cfg), target)
    assert ret["snapshot"]["cutoff"] == "1948-01-01" and len(ret["citation_ids"]) == 3 and ret["documents"]
    prompt = theorist_prompt(target, lessons_text="", docs=ret["documents"])
    assert "Scoreboard entry: best known value 7.0" in prompt and "[" in prompt and "integrate(x**2, (x, -1, 1))" in prompt
    for tool_calls, expected in ((0, THEORIST_EXPECT["passed_round1"]), (1, THEORIST_EXPECT["passed_round2"])):
        runner = TheoristRunner(s["role"], target, s["judge"], prompt=prompt, tool_calls=tool_calls, timeout_s=60,
                                context={"citation_ids": ret["citation_ids"], "reading_list_snapshot": ret["snapshot"]},
                                revision_max_tokens=1234)
        assert runner.budget_s == 40.0  # the sandbox cap of 60 s less the margin: the lemma file's own budget
        summ = _run(s, runner, led, target, 4, "theorist", tool_calls=tool_calls)
        assert summ["passed"] == expected and summ["role"] == "theorist" and summ["tool_calls"] == tool_calls
        rows = led.attempts(group_id=summ["group_id"])
        assert all(r.role == "theorist" and r.certificate["certifier"] == "lemma_check" and r.meta["code"] is None for r in rows)
        assert all(r.context["citation_ids"] == ret["citation_ids"] and r.context["reading_list_snapshot"]["cutoff"] == "1948-01-01" for r in rows)
        assert all(r.context["adapter_version"] == "none:frozen-base" and r.context["tool_calls"] == tool_calls for r in rows)
        by_seed = {r.seed: r for r in rows}
        good = by_seed[0]
        assert good.passed and good.score == 1.0 and good.meta["proposal"]["passed"] == 3 and "seven nodes" in good.meta["proposal"]["structure"]
        tools = [t["tool"] for t in good.tool_log]
        if tool_calls == 0:
            assert tools == ["chat.completions", "lemma_check", "judge"] and good.meta["rounds"] == 1
        else:
            assert tools == ["chat.completions", "lemma_check", "chat.completions", "lemma_check", "judge"] and good.meta["rounds"] == 2
            # the revision answers under its own cap; the opening round under the role's (the fixture key ignores the cap)
            assert good.tool_log[0]["max_tokens"] == s["role"].max_tokens and good.tool_log[2]["max_tokens"] == 1234
            assert good.tool_log[1]["timed_out"] == 0 and good.tool_log[1]["skipped"] == 0
            assert good.meta["round_outputs"] and len(good.meta["round_outputs"]) == 2
        false = by_seed[1]
        if tool_calls == 0:
            assert not false.passed and [l["status"] for l in false.certificate["details"]["lemmas"]] == ["failed", "trivial"]
            assert "no lemma check passed" in false.diagnosis
        else:
            assert false.passed and [l["status"] for l in false.certificate["details"]["lemmas"]] == ["passed"]
        prose = by_seed[2]
        assert (not prose.passed and "no JSON" in prose.certificate["error"]) if tool_calls == 0 else prose.passed
        err = by_seed[3]
        st = [l["status"] for l in err.certificate["details"]["lemmas"]]
        assert err.passed and (st == ["error", "passed"] if tool_calls == 0 else st == ["passed"])
        p = proposal_from_row(good)
        assert p["id"] == good.id and p["lemmas"][0]["status"] == "passed"
        text = with_proposal(target.prompt, p)
        assert "[checked]" in text and "[unverified]" in text and "x**2 * y**2" in text
    led.close()


def test_parse_proposal_rejects_bad_shapes():
    assert parse_proposal("no json here")[1].startswith("no JSON")
    assert parse_proposal("[1, 2]")[1] == "the proposal must be a JSON object"
    p, err = parse_proposal(json.dumps({"lemmas": "nope", "structure": 5, "hypotheses": "x"}))
    assert err is None and p["lemmas"] == [] and p["structure"] == "5" and p["hypotheses"] == []


# --- the builder's dry run -----------------------------------------------------------------------------------------------
def test_builder_dry_run_feeds_the_certifier_back_and_the_final_answer_is_scored(tmp_path):
    scripts = [[SQ_TENSOR9, SQ_RADON7],                      # valid 9 nodes, revised to the optimal 7
               ["no code at all", SQ_TENSOR9],               # no code block, then a valid rule
               [SQ_RADON7, "I keep my file."]]               # the revision loses the code: the final answer is what counts
    stub = _Script(scripts)
    cfg, led, s = _setup(tmp_path, "builder", stub)
    target = get_target("quad-square-d5")
    runner = BuilderRunner(s["role"], target, s["judge"], tool_calls=1)
    summ = _run(s, runner, led, target, 3, "builder", tool_calls=1)
    rows = {r.seed: r for r in led.attempts(group_id=summ["group_id"])}
    assert summ["passed"] == 2 and summ["optimal"] == 1 and summ["rounds"] == 6
    r0 = rows[0]
    assert r0.passed and r0.certificate["optimal"] and r0.meta["dry_runs"][0]["passed"] and r0.meta["dry_runs"][0]["score"] == 9.0
    assert r0.meta["revised_after_dry_run"] is True and [t["tool"] for t in r0.tool_log] == ["chat.completions", "dry_run", "chat.completions", "judge"]
    assert r0.meta["completion_tokens"] == 100 and r0.meta["prompt_tokens"] == 700 and r0.meta["latency_s"] == pytest.approx(1.0)
    req = stub.request(0, 1)
    feedback = req[-1]["content"]
    assert feedback.startswith("Dry run of the certifier quadrature_exactness") and "PASS, score 9" in feedback and "node_count=9" in feedback
    assert req[-2]["role"] == "assistant" and "```python" in req[-2]["content"]
    r1 = rows[1]
    assert r1.passed and r1.meta["dry_runs"][0]["had_code"] is False and "no code block" in r1.meta["dry_runs"][0]["diagnosis"]
    r2 = rows[2]
    assert not r2.passed and r2.meta["dry_runs"][0]["passed"] and r2.meta["dry_runs"][0]["optimal"] and "no code block" in r2.diagnosis
    cert = type("C", (), {"passed": False, "certifier": "quadrature_exactness", "details": {"node_count": 4, "max_monomial_error": 2.5e-3,
                                                                                            "worst_monomial": [0]}, "error": None, "score": 4.0, "optimal": False})()
    text = dry_run_feedback(cert, None, target)
    assert "FAIL" in text and "max_monomial_error=2.500e-03" in text and "```python" in text
    led.close()


# --- the breaker and the witness reward ----------------------------------------------------------------------------------
def _certified_row(led, target, code):
    """A row that looks certified for the breaker (the loop hands it certified attempts; here the code may be secretly wrong)."""
    from lab.ledger import Attempt

    row = Attempt(role="builder", model="m", model_version="v", precision="bf16", target=target.id, prompt_hash="sha256:x",
                  certificate={"certifier": "quadrature_exactness", "version": "1.0.0", "passed": True, "score": 4.0,
                               "details": {"node_count": 4, "weight_sum": 2.0, "volume": 2.0, "measured_exactness_degree": 7, "max_monomial_error": 1e-16}},
                  output=code, meta={"code": code}, seed=0)
    led.add_attempt(row)
    return row


GAUSS4 = "def rule():\n    return [-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526], [0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538]\n"
CORRUPT = "def rule():\n    return [-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526], [0.3488548451374538, 0.6521451548625461, 0.6521451548625461, 0.3488548451374538]\n"   # both outer weights off by 1e-3: odd monomials stay exact


def _w(exponents=None, coefficients=None, kind="exactness_error"):
    inp = {"exponents": exponents} if exponents is not None else {"coefficients": coefficients}
    return json.dumps({"witness": {"kind": kind, "input": inp}, "reason": "test"})


def test_witness_reward_one_for_a_confirmed_break_zero_otherwise(tmp_path):
    scripts = [[_w(exponents=[2])],                                     # inside the class, breaks the corrupted rule
               [_w(exponents=[1])],                                     # inside the class, odd monomial: the symmetric corruption cancels -> no break
               [_w(exponents=[8])],                                     # outside: degree above 7
               [_w(coefficients={"2": 5.0})],                           # outside: coefficient above 1
               ["I cannot find a witness."],                            # no witness
               [_w(exponents=[0], kind="exception")]]                   # the rule evaluates fine: no break
    stub = _Script(scripts)
    cfg, led, s = _setup(tmp_path, "breaker", stub, system_prompt=BREAKER_SYSTEM)
    target = get_target("quad-d7")
    row = _certified_row(led, target, CORRUPT)
    cand = candidate_from_row(row, {"claim": "a 4-node rule exact to degree 7", "conditions": ["interval"]}, label="candidate 1")
    prompt = breaker_prompt(target, cand, tool_calls=0)
    assert "Class definition:" in prompt and "Dossier claim: a 4-node rule" in prompt and "max_monomial_error<1e-9" in prompt and "reward 1" in prompt
    runner = BreakerRunner(s["role"], target, s["judge"], cand, tool_calls=0)
    summ = _run(s, runner, led, target, 6, "breaker")
    rows = {r.seed: r for r in led.attempts(group_id=summ["group_id"])}
    assert summ["passed"] == 1 and [rows[i].meta["witness_reward"] for i in range(6)] == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert rows[0].certificate["certifier"] == "witness_check" and rows[0].certificate["details"]["reproduces"] and rows[0].score == 1.0
    assert rows[0].context["candidate_attempt_id"] == row.id and rows[0].context["candidate_code_sha"] == cand["code_sha"]
    assert rows[0].meta["witness"] == {"kind": "exactness_error", "input": {"exponents": [2]}, "family": "quadrature", "domain": "interval", "degree": 7}
    assert "does not reproduce" in rows[1].diagnosis
    assert "outside the class" in rows[2].diagnosis and "degree" in rows[2].diagnosis
    assert "outside the class" in rows[3].diagnosis and "absolute value" in rows[3].diagnosis
    assert "no JSON witness" in rows[4].diagnosis and rows[4].meta["witness"] is None
    assert not rows[5].passed and rows[5].certificate["details"]["reproduces"] is False
    assert all(r.role == "breaker" and r.meta["candidate_ran"] for r in rows.values())
    assert all([t["tool"] for t in r.tool_log] == ["chat.completions", "witness_check", "judge"] for r in rows.values())
    led.close()


def test_sound_candidate_is_never_broken_and_probe_round_is_logged(tmp_path):
    probe = json.dumps({"probe": {"exponents": [[2], [4], [6], [9], ["a"]]}})
    scripts = [[probe, _w(exponents=[6])], [probe, _w(coefficients={"6": 1.0, "4": -1.0, "0": 0.25})], [_w(exponents=[4])]]
    stub = _Script(scripts)
    cfg, led, s = _setup(tmp_path, "breaker", stub, system_prompt=BREAKER_SYSTEM)
    target = get_target("quad-d7")
    row = _certified_row(led, target, GAUSS4)
    cand = candidate_from_row(row, None, label="candidate 1")
    runner = BreakerRunner(s["role"], target, s["judge"], cand, tool_calls=1)
    summ = _run(s, runner, led, target, 3, "breaker", tool_calls=1)
    rows = {r.seed: r for r in led.attempts(group_id=summ["group_id"])}
    assert summ["passed"] == 0 and all(r.meta["witness_reward"] == 0.0 for r in rows.values())
    assert rows[0].meta["probe_rounds"] == 1 and [t["tool"] for t in rows[0].tool_log] == ["chat.completions", "execute_candidate", "chat.completions", "witness_check", "judge"]
    fb = stub.request(0, 1)[-1]["content"]
    assert fb.startswith("Probe results:") and "[6]: rule" in fb and "below tol" in fb and "[9]: outside the class" in fb and "malformed" in fb
    assert rows[1].certificate["details"]["terms"] == 3 and rows[1].certificate["details"]["tol"] == pytest.approx(3e-9)
    assert rows[2].meta["probe_rounds"] == 0 and rows[2].meta["rounds"] == 1
    led.close()


def test_breaker_probe_without_a_witness_pays_nothing(tmp_path):
    probe = json.dumps({"probe": {"exponents": [[2]]}})
    stub = _Script([[probe, probe]])
    cfg, led, s = _setup(tmp_path, "breaker", stub, system_prompt=BREAKER_SYSTEM)
    target = get_target("quad-d7")
    cand = candidate_from_row(_certified_row(led, target, CORRUPT), None, label="candidate 1")
    runner = BreakerRunner(s["role"], target, s["judge"], cand, tool_calls=1)
    summ = _run(s, runner, led, target, 1, "breaker", tool_calls=1)
    r = led.attempts(group_id=summ["group_id"])[0]
    assert not r.passed and "probe was submitted where a witness was due" in r.certificate["error"] and r.meta["witness_reward"] == 0.0
    led.close()


def test_first_order_rate_violation_witness(tmp_path):
    # a candidate that claims a rate below the true worst case on the quadratic class
    code = 'def method():\n    return {"method": "gradient_descent", "params": {"mu": 0.1, "L": 1.0, "gamma": 1.0, "n": 1}}\n'
    from lab.ledger import Attempt

    target = get_target("pep-gd-strongly-convex-distance")
    led = Ledger(tmp_path / "ledger.sqlite")
    row = Attempt(role="builder", model="m", model_version="v", precision="bf16", target=target.id, prompt_hash="sha256:x",
                  certificate={"certifier": "pep_worst_case_rate", "version": "1.0.0", "passed": True, "score": 0.5, "details": {"rate": 0.5}},
                  output=code, meta={"code": code}, seed=0)
    led.add_attempt(row)
    led.close()
    scripts = [[json.dumps({"witness": {"kind": "rate_violation", "curvature": 0.1}})],     # the slow quadratic: ratio 0.81 > claimed 0.5
               [json.dumps({"witness": {"kind": "rate_violation", "curvature": 0.5}})],     # ratio 0.25 < 0.5: no break
               [json.dumps({"witness": {"kind": "rate_violation", "curvature": 7.0}})]]     # outside [mu, L]
    stub = _Script(scripts)
    cfg, led, s = _setup(tmp_path, "breaker", stub, system_prompt=BREAKER_SYSTEM)
    cand = candidate_from_row(row, None, label="candidate 1")
    prompt = breaker_prompt(target, cand, tool_calls=0)
    assert "rate_violation" in prompt and "curvature" in prompt
    runner = BreakerRunner(s["role"], target, s["judge"], cand, tool_calls=0)
    summ = _run(s, runner, led, target, 3, "breaker")
    rows = {r.seed: r for r in led.attempts(group_id=summ["group_id"])}
    assert rows[0].passed and rows[0].certificate["details"]["observed_ratio"] == pytest.approx(0.81) and rows[0].meta["witness"]["claimed_rate"] == 0.5
    assert not rows[1].passed and "outside the class" in rows[2].diagnosis
    led.close()


def test_complete_witness_takes_the_class_from_the_target():
    t = get_target("quad-triangle-d4")
    w = complete_witness(t, {"kind": "exactness_error", "input": {"exponents": [1, 1]}, "domain": "square", "degree": 99})
    assert w == {"kind": "exactness_error", "input": {"exponents": [1, 1]}, "family": "quadrature", "domain": "triangle", "degree": 4}
    assert parse_witness("{\"kind\": \"exactness_error\", \"input\": {\"exponents\": [1]}}")[0]["kind"] == "exactness_error"
    assert parse_witness("{\"nothing\": 1}")[2] == "the reply holds neither a witness nor a probe"
