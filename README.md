# Discovery at 27B: the lab, weeks 1 to 4

A fully automated numerical lab on one open model, `Qwen/Qwen3.6-27B`, with three roles that will later be
LoRA adapters and program harnesses around them. This repository holds weeks 1 to 4 of the build plan:
the model served from Colab, the repository skeleton (ledger, cost meter, role interface), the certifier
library (seven certifiers), the judge sandbox, the scoreboard files, the demo loop with concurrent attempts,
the training gate, the three harnesses (librarian, archive and lesson bank, checker) with the ablation
harness around them, and, from week 4, the three roles (theorist, builder, breaker) with their prompts, tools
and the episode runner, the loop that runs them end to end on one family, and the transfer-matrix machinery
(experience sets, the two controls, cells at matched dollars with bootstrap intervals).

Rules that never bend: every score comes from a program (a certificate), never from a model's opinion;
verification of known results comes before any open problem; every attempt is logged with its cost; a
frontier model through an API may do the language parts of the harnesses during the build, but nothing a
harness writes enters a reward.

## Layout

| Path | What it is |
|---|---|
| `colab/serve.ipynb` | Drive, weights pinned by hash, 4-bit build, OpenAI-compatible server with LoRA serving, smoke test, demo (parallel requests), the gate runner and the online ablation (week 3 cells), the loop, the family baselines, the 256-attempt search baseline and the matrix cells (week 4 cells 13 to 16). Generated from `colab/build_notebook.py`. |
| `config/lab.yaml` | Storage path, endpoint, providers (`local`, `openai`, `anthropic`, each with a mode `offline`, `online` or `record`), the harness settings (provider, token budget of the lesson bank, detector thresholds, reading list), roles, judge limits. `${VAR:-default}` expands from the environment. |
| `config/model.yaml` | The frozen base (repo, revision, checkpoint hash) and the quantized build. |
| `config/prices.yaml`, `config/budget.yaml` | The price table; the frontier budget: $200 a month across GPT and Claude, warning at 80 percent, enforced in code. |
| `lab/ledger.py` | The attempt record (Work package 4 schema) in SQLite with a JSONL mirror, `provider`, `mode` and `api_dollars` columns, the `serving` table, pass@k, the month's frontier spend, and generic tables for the harnesses (`citations`, `lessons`, `artifacts`, `provenance`, `detectors`, `dossiers`). |
| `lab/cost.py` | The cost meter and the budget guard (locked for concurrent calls). |
| `lab/roles/base.py` | The role interface: provider kinds `local`, `openai` (GPT, first), `anthropic` (Claude, second); modes `offline`, `online`, `record`. The OpenAI provider sends `max_completion_tokens`, omits an unset temperature and carries a configured `reasoning_effort` (GPT-5). Week 4: `Role.attempt_messages` runs one round of a tool conversation; a replay sequence keyed by the opening prompt holds the reply of every round (`rounds`). |
| `lab/roles/theorist.py`, `builder.py`, `breaker.py` | Week 4: the three roles as episode runners (Work package 4). Theorist: the target, its scoreboard entry, the gate's lessons and the librarian's excerpts under the cutoff in the prompt; lemmas with SymPy checks run in the sandbox, structure and hypotheses for the builder; the lemma-check certificate is the immediate reward, a revision round after the checks. Builder: the demo's attempt plus a certifier dry run (unscored) with a revision, lessons and skills from the gate, and the theorist's checked lemmas when an episode is derived from a proposal. Breaker: a certified candidate with its dossier and the class definition; an optional probe of the rule on monomials (the execution tool); a witness checked by `witness_check`: reward one for a confirmed break, zero for anything else. |
| `lab/episode.py` | Week 4: the episode runner: a role's attempt with its fixed number of tool rounds, priced by its share of the wave's wall time, one ledger row with role, adapter version, prompt hash, context (lesson ids, skill ids, reading-list snapshot, citations, proposal or candidate), output, tool log, certificate, the diagnosis on failure (the certifier's reason plus the kept detectors that fire), cost and seed. `demo.run_group` is the builder's group on it. |
| `lab/certifiers/lemma.py` | Week 4: `lemma_check`, the theorist's reward: one SymPy expression per lemma over a whitelisted namespace, trivial and symbol-free checks refused, duplicates counted once, run in the sandbox with SymPy allowed for that run only; pass when at least one check passes, the score the fraction that pass. |
| `lab/loop.py` | Week 4: `python -m lab.loop --targets ...`: one generation on one family (the curriculum rungs in order): the gate's lessons, the theorist's proposals, the builder's gate group (plain prompt, the gate memo), lesson group and proposal group, the judge, the breaker on the distinct certified survivors (a confirmed break voids the candidate and raises an exploit alarm), the archive with dossiers and witness records, lesson extraction and consolidation, the delayed credit of every proposal, and the generation record with the archive snapshot and the ledger pointer. |
| `lab/harness/carriers.py`, `lab/harness/matrix.py` | Week 4: the context carriers (`none`, `lessons`, `lessons_gate`, `skills`, `process_only` with content redacted, `unrelated`) and the source-target matrix: experience sets from the archive per source family, the two controls, the cell runner (N seeds, matched attempts or dollars), the ledger query and the analysis with a percentile bootstrap interval per cell; the program refuses to measure lessons on the family they came from. The ablation harness is its single-cell version. |
| `lab/certifiers/quadrature.py`, `pep.py`, `witness.py` | Weeks 1 to 2: exactness, worst-case rates by performance estimation, the witness check. |
| `lab/certifiers/order.py` | Week 3: convergence order from a seeded refinement study (manufactured solutions for an ODE, a 1-D Poisson problem and a composite integral), slope with a t-interval and pairwise orders. |
| `lab/certifiers/opcount.py`, `counted.py` | Week 3: operation count by instrumented arithmetic (an expression DAG of counted scalars; only operations that reach an output count) with the outputs checked against the exact transform, and cost models checked against instrumented runs. |
| `lab/certifiers/stability.py` | Week 3: the stability polynomial, the real- and imaginary-axis extents of the region of absolute stability, the order conditions and the SSP coefficient (radius of absolute monotonicity) of an explicit Runge-Kutta method in Butcher or Shu-Osher form. |
| `lab/judge.py`, `lab/sandbox_runner.py` | The judge: subprocess sandbox plus the certifier; the runner can wrap entry arguments as counted scalars for the operation-count family. |
| `lab/targets.py` | Targets with prompts, scoreboard entries and reading-list cutoffs: `quad-*`, `pep-*`, `order-*`, `opcount-dft-n*`, `ssp-s*-p*`, `stab-{real,imag}-s*-p*`. Week 4: every target carries a transfer block (certificate family, instance, regime level, structure class, certifier, open flag) and a class definition for the breaker; `transfer_setting(sources, target)` names a target's setting relative to an experience set. |
| `lab/demo.py` | `python -m lab.demo`: N attempts from one prompt, judged, priced, in the ledger, pass@1 and pass@16; `--concurrency 16` sends a group as parallel requests (a wave's GPU time is its wall time, shared among its attempts by latency); `--thinking N` caps the reasoning at N tokens and then asks for the answer (two requests); prompt overrides and a logged post-check for the ablation arms; `tool_calls` dry runs for the builder. |
| `lab/gate.py` | `python -m lab.gate --targets ... --attempts 64 --thinking 0 512 --max-tokens 8192`: the training gate, a memo per target with the pass rate (the verdict's statistic), pass@1/16/64 valid and optimal, the attempts that hit the token cap or returned no code, the 5 to 80 percent verdict and the cost. |
| `lab/harness/base.py` | The harness base: every language call goes through the provider and the budget guard and into the ledger as role `harness:<name>` with provider, mode and API dollars. |
| `lab/harness/librarian.py`, `reading_list/*.yaml` | The librarian: the reading list with dates, retrieval under a per-target date cutoff (lexical ranking by a program, selection and summary by the provider), every citation logged, and the pass over `provenance: provisional` scoreboard entries. |
| `lab/harness/archive.py` | The archive: a content-addressed store of certified methods with witnesses and dossiers, linked to the attempt, reading-list snapshot, lessons, citations and cost; `trace()` is one SQL query. |
| `lab/harness/lessons.py` | The lesson bank: lessons at three levels (approach, implementation, process), each a rule with conditions and an evidence tag (observed, derived, proposed); consolidation under a token budget by add, replace and retire; the gate that selects one lesson and writes a concrete, validated check applied at use time. |
| `lab/harness/checker.py` | The checker: the frozen baseline record from ledger groups; hypotheses turned into deterministic detectors kept only when right on more than nine of ten held-out cases; method dossiers with claim, conditions and evidence, schema-validated. |
| `lab/harness/ablation.py` | `python -m lab.harness.ablation`: three arms on one verification family at matched dollars (no notebook, notebook, notebook with gate), every attempt in the ledger, a report per run. |
| `scoreboard/*.yaml` | Quadrature node counts, first-order rates, SSP coefficients and stability extents, DFT operation counts, convergence-order verification targets, with sources. |
| `tests/` | pytest suites: at least 20 known-answer cases per certifier, judge limits, ledger, cost, budget, roles, demo, concurrency, gate, harness base, the harness suites per frontier provider (offline from recorded fixtures), and the week-4 suites: the lemma check, the transfer blocks, the three roles (fixtures and stub endpoints), the loop end to end, the matrix. `tests/make_fixtures.py` writes the synthetic local sequences (`tests/fixtures_week4.py` holds the week-4 ones). Live tests carry the `online` marker. |
| `RESULTS.md` | pass@k, the calibration points, what the ablation harness measured, the precision used and the units spent. |

## Commands

Local (Python 3.11 or newer; no GPU needed):

```bash
pip install -e ".[dev]"
```

```bash
pytest
```

The demo loop offline, against the synthetic fixtures (no key, no cost), with the group sent as parallel requests:

```bash
python -m lab.demo --target quad-d7 --attempts 16 --provider local --mode offline --concurrency 8
```

The training gate on the training-family candidates (a memo per target under `<ledger dir>/gates/`):

```bash
python -m lab.gate --targets quad-square-d5 quad-triangle-d4 --attempts 64 --thinking 0 512 --max-tokens 8192 --gpu G4 --precision bf16
```

The ablation harness offline (local attempts and GPT lessons both from recorded fixtures):

```bash
python -m lab.harness.ablation --target quad-d7 --attempts 16 --mode offline --harness-mode offline
```

The librarian: a retrieval under a target's cutoff, and the pass over the provisional scoreboard entries (writes `scoreboard/reviews/`):

```bash
python -m lab.harness.librarian --query "fewest nodes for degree 7" --target quad-d7 --mode offline
```

```bash
python -m lab.harness.librarian --review quadrature --mode offline
```

The loop offline (one generation on three curriculum rungs with the theorist, the builder and the breaker, from the synthetic
fixtures; search-only, no frontier provider; a gate memo per rung and the generation record under the ledger's directory):

```bash
python -m lab.loop --targets quad-d7 quad-square-d5 quad-triangle-d2 --attempts 8 --theorist-attempts 4 --proposal-attempts 4 --breaker-attempts 5 --breaker-candidates 2 --mode offline --no-harness --gpu A100 --concurrency 4 --max-tokens 4096 --timeout 60
```

The matrix offline after the loop (the experience of `quad-d7` on the square and the triangle, with the two controls;
the same-family cells are refused by the program):

```bash
python -m lab.harness.matrix --sources quad-d7 --targets quad-square-d5 quad-triangle-d2 --carriers none lessons process_only unrelated skills --seeds 2 --attempts 8 --mode offline --no-harness --gpu A100 --concurrency 4 --max-tokens 4096
```

List targets:

```bash
python -m lab.demo --list
```

On Colab (Pro+, background execution): open `colab/serve.ipynb`, set the GPU runtime, add the secrets
`HF_TOKEN`, `GITHUB_TOKEN` (private repository) and, for the ablation, `OPENAI_API_KEY`, fill cell 1 and run all
cells. Cells 4 to 9 pin the frozen base, build or reuse the 4-bit copy, start vLLM with the zero adapter, run the
smoke test and the demo with `CONCURRENCY` parallel requests. Cells 10 and 11 (off by default: `RUN_GATE`,
`RUN_ABLATION`) run the gate on `GATE_TARGETS` with the thinking ladder and `GATE_MAX_TOKENS` answer tokens, and the ablation harness online; both write
to the ledger on Drive, the memos to `gates/` and the reports to `ablations/` next to it. Cell 12 (`RECORD_FIXTURES`,
no GPU needed) records the harness tests' GPT fixtures with the same secret (the loop's lesson-extraction fixtures included),
replays them once, and exports them with the librarian's reviews to `exports/recorded-fixtures/` on Drive for a commit from
your own machine (Colab pulls read-only). Week 4 (all off by default): cell 13 (`RUN_LOOP`) runs one generation of the loop
on `LOOP_TARGETS` (the quadrature curriculum: interval d7, square d5, triangle d2, d3, d4) with 64 attempts per gate group at
`GATE_MAX_TOKENS` answer tokens and GPT for lessons and dossiers; cell 14 (`RUN_BASELINES`) runs the gate on one target of
every other certifier family and writes the frozen baseline record; cell 15 (`RUN_SEARCH_BASELINE`) runs 256 attempts of the
frozen base on `quad-square-d5`; cell 16 (`RUN_MATRIX`) runs matrix cells from the loop's archive.

## Providers, modes, fixtures and the frontier budget

The three roles always run on `local`, the Qwen3.6-27B endpoint. GPT (`openai`, model `gpt-5-mini` by default) and
Claude (`anthropic`) are for the language parts of the harnesses (librarian selection and summaries, lesson
extraction and gating, detector proposals, dossiers), GPT first and Claude second behind the same interface;
nothing they write enters a reward. Every provider has a mode:

| Mode | What happens | Cost |
|---|---|---|
| `offline` | responses are replayed from `tests/fixtures/providers/<provider>/`, keyed by a hash of the request; a missing fixture raises, never a live call | nothing |
| `online` | live calls | GPU seconds for `local`, tokens for the frontier |
| `record` | live calls, each written as a fixture so the same test runs offline afterwards | as online |

The pytest default is offline. The harness tests are parametrized over the two frontier providers and replay the
recorded fixtures; a provider with no recorded fixtures is skipped with a message. To record them once (every call
passes the budget guard; the rows land in the repository's own ledger so the month's spend is tracked):

```bash
LAB_FRONTIER_MODE=record OPENAI_API_KEY=... pytest tests/test_librarian.py tests/test_lessons.py tests/test_checker.py tests/test_ablation.py -k openai
```

```bash
python tests/make_fixtures.py
```

The second command rebuilds the lessons offline from the recorded fixtures and writes the synthetic sequences the
ablation's notebook arms replay. Repeat both with `ANTHROPIC_API_KEY` and `-k anthropic` for Claude. With the key only on
Colab, cell 12 of the notebook runs the same two commands there and exports the files to Drive. Harness prompts
are deterministic functions of their inputs (attempts are labelled by order, numbers rounded to three digits), which
is what makes a recording replayable.

The frontier spend is capped at $200 a month (`config/budget.yaml`), enforced in code before every online or record
call; every harness row carries its `api_dollars`, and `Ledger.api_spend(year, month)` is the month's total.

## The judge's rules during measurement

The candidate is a Python file with a fixed entry point (`rule()` for quadrature, `method()` for first-order
targets, `solve(problem, hs)` for convergence order, `transform(re, im)` for the operation count, `tableau()` for
time integrators). It runs in a fresh `python -I` process with a scrubbed environment, a temporary working
directory, a wall-clock cap (120 s), a memory limit (2 GB), sockets disabled and network modules unimportable.
Library quadrature and optimisation solvers are forbidden: `numpy.polynomial`, `scipy.integrate`, `scipy.optimize`,
`scipy.special`, `scipy.interpolate`, `sympy`, `mpmath`, `cvxpy`; so are `os`, `sys`, `subprocess`, `importlib`,
`ctypes`, dynamic `__import__`, `exec`, `eval`, `open`. NumPy arithmetic and `numpy.linalg` are allowed. A forbidden
import, found by reading the code or at run time by the import guard, fails the attempt. The same attempt scores
the same twice. For the operation-count family the runner hands the candidate counted scalars and the count is
read off the outputs, never trusted.

## Certifiers

| Name | Version | Cap | Known-answer cases |
|---|---|---|---|
| `quadrature_exactness` | 1.0.0 | 30 s | Gauss-Legendre n = 1..10 passes at degree 2n-1 and fails at 2n; one corrupted weight fails; Clenshaw-Curtis, Newton-Cotes, tensor rules, six classical triangle rules, Duffy-collapsed rules; positivity and interiority flags. |
| `pep_worst_case_rate` | 1.0.0 | 60 s per SDP | Gradient descent and Douglas-Rachford match the tight published rates; Nesterov stays under its bound and matches PEPit's reference; Davis-Yin reproduces PEPit's reference (the calibration point); the dual reconstructs the bound. |
| `witness_check` | 1.1.0 | 30 s | Synthetic breaks reproduce, synthetic non-breaks do not, instances outside the class definition score zero; the quadrature class (1.1.0) is monomials up to the target degree or polynomials with coefficients of absolute value at most one, at the tolerance times the number of terms, so rounding error cannot be scaled into a break; malformed inputs score zero. |
| `lemma_check` | 1.0.0 | 120 s | Fifteen true identities (moments on the interval, the square and the triangle, Legendre orthogonality and root counts, sums, determinants) pass; six false claims fail; constants and symbol-free checks are trivial; forbidden names, attributes and statements are invalid; a duplicate counts once; a crashing check is a failed check; SymPy is allowed in the sandbox for this run only and `os` stays forbidden. |
| `convergence_order` | 1.0.0 | 60 s | Euler, Heun and RK4 measure orders 1, 2 and 4 on seeded manufactured ODE problems; central differences measure 2 on the Poisson problem and a deliberately wrong boundary term lowers the measured order to about 1; composite left, midpoint, trapezoid and Simpson rules measure 1, 2, 2 and 4; machine-precision levels are dropped from the fit; the interval and the pairwise orders are reported. |
| `operation_count` | 1.0.0 | 30 s | The radix-2 FFT counts exactly 2 N log2 N multiplications and 3 N log2 N additions for N = 2..64 and the naive transform 4 N^2 of each; shared subexpressions count once and dead code nothing; a wrong transform fails however cheap; an output that leaves counted arithmetic fails; cost models are checked against instrumented runs. |
| `stability_region_ssp` | 1.0.0 | 30 s | Euler, Heun, Kutta's third-order rule and classical RK4 reproduce the tabulated real-axis extents 2, 2, 2.5127, 2.7853 and imaginary extents 0, 0, sqrt 3, 2 sqrt 2; SSPRK(2,2), (3,3), (4,3) and Spiteri-Ruuth's (5,4) reproduce the coefficients 1, 1, 2, 1.508 from the Shu-Osher form and from the radius of absolute monotonicity; SSPRK(s,2) has s - 1 and SSPRK(n^2,3) has n^2 - n; RK4, Kutta's rule and the 3/8 rule get 0; the polynomial agrees with the method on the test equation at seeded points. |

## The harnesses

Every harness is a program; where it needs language it calls the harness provider through `lab/harness/base.py`,
and every such call is an attempt row with role `harness:<name>`, the provider, the mode and the API dollars.

* Librarian. The reading list under `reading_list/` (89 documents across quadrature, first-order methods, time
  integrators, fast transforms and verification, each with a date and a summary written by us). A target's cutoff
  (`lab/targets.py`, for example 1814 for the Gauss rediscovery, 2015 for Davis-Yin) is enforced by the program
  before the model sees any candidate and again on what is returned; the lexical ranker alone passes the retrieval
  test set in `reading_list/retrieval_tests.yaml`; every returned document is a `citations` row. The provisional
  pass asks the provider whether each `provenance: provisional` scoreboard entry is consistent with the documents
  its source names and writes reviews for a person; the scoreboard itself changes only by a reviewed diff.
* Archive and lesson bank. Methods are archived by content hash with their certificate, dossier, witness and
  reading-list snapshot, and `Archive.trace(id)` returns the whole chain (attempt, model, precision, cost, lessons,
  cited documents) in one query. Lessons are extracted from a group of attempts, tagged by the program (observed
  only when the cited attempts exist), consolidated under `harness.lesson_token_budget` by add, replace and retire,
  and the gate selects one lesson for a target and writes a check (a whitelisted Python expression over the
  candidate's result) whose outcome is logged in the attempt's meta and never scored.
* Checker. `Checker.baseline_record` summarizes the untrained model's groups per target and flags replays and
  quantized rows as not reportable; detectors proposed from a mining pool are verified on a held-out pool split by
  seed and kept only if they fire at least twice, every firing is a failure and they are right on more than 90
  percent of the pool (five built-in detectors pass the same gate); dossiers carry a claim, conditions and an
  evidence block whose attempt id, score and hashes the program fills and checks. The no-library-solvers rule is
  enforced by the judge.
* Ablation. Arm A (no notebook) is the experience; the lesson harness extracts and consolidates; arm B prompts
  with the notebook; arm C with the gate's lesson and check. The three arms run the same number of attempts or stop
  at the same dollar budget; the report gives pass@k per arm, the gain over arm A and the harness spend.

## The three roles, the episode and the loop (week 4)

Every role runs on `local`, the frozen base with no adapter (`adapter_version: none:frozen-base` in every row), and
every attempt is an episode: a prompt, a fixed number of tool rounds, a program that scores it, a ledger row.

* Theorist. Prompt: the target, its scoreboard entry, the lesson the gate chose, and up to three reading-list excerpts
  the librarian's lexical ranker returns under the target's cutoff (each a citation row; the snapshot is in the
  context). It replies with JSON: lemmas (a statement and one SymPy check each), a structure, hypotheses. The program
  validates every check against a whitelist (a check must call a symbolic operation and mention a symbol; constants
  and tautologies without symbols are refused), runs the valid ones in the judge's sandbox with SymPy allowed, and
  writes the `lemma_check` certificate: pass when at least one check passes, score the fraction that pass. With one
  tool call the results go back to the model for a revision. Proposals whose checks passed feed the builder; at the end
  of a generation the delayed credit of each proposal is the best certificate among the builds derived from it minus
  the gate group's best, with the pass-rate difference beside it.
* Builder. The demo's attempt, plus `tool_calls` dry runs of the certifier: the program runs the file in the sandbox,
  reports the verdict and the facts (node count, weight sum, worst monomial, the sandbox error) back to the model, and
  the model answers once more; the final file is what the judge scores, the dry runs are in the tool log and never
  scored. Lessons come from the gate (one lesson plus a check applied at use time) or the notebook; skills from the
  archive come as reference implementations; a proposal appends the theorist's checked lemmas and its unverified
  structure and hypotheses.
* Breaker. Prompt: the certified candidate's code, its certificate facts, its dossier claim and conditions, the class
  definition of the target and the witness schema. With one tool call it may first ask for the rule's error on up to
  eight monomials (the program evaluates the candidate; instances outside the class are named as such), then it submits
  the witness. The program fills in the class (family, domain, degree, or the claimed rate) and the `witness_check`
  certifier re-runs the candidate on exactly that input: reward one for a confirmed break, zero for anything else,
  including an instance outside the class, a malformed witness, or none. A certified quadrature rule covers the whole
  monomial basis of its class, so a confirmed break there means the certifier is wrong: the loop voids the candidate
  and records an exploit alarm.
* Limits per attempt, all in the row's context: the role's token cap (8192 answer tokens in the week-4 runs), the
  thinking budget from the ladder (0, 512, 1024 through the two-request mechanism), the sandbox's 120 s wall-clock
  cap, and the fixed number of tool calls.

The loop (`lab/loop.py`) runs one generation over the curriculum rungs in order. Per rung: the gate chooses lessons
when the bank holds any; the theorist proposes; the builder runs its gate group from the plain prompt (the frozen
base's record, from which the gate memo is written), a lesson group when lessons exist, and a proposal group; the
judge certifies inside the episodes; the distinct certified candidates, best first, are the survivors; the breaker
attacks each with a group of attempts; the survivors enter the archive with their certificate, a dossier (the checker's
when a frontier provider runs, the program's template otherwise), the witness record (attacks and confirmed breaks), the
lesson ids, citations and reading-list snapshot; the lesson harness extracts lessons from the gate group and the bank
consolidates them under the token budget. The generation ends with an archive snapshot (the sorted artifact ids and
their hash), a ledger pointer (path, row count, last row), the cost by role, the active lesson bank, and a record under
`generations/`. No training yet: this generation is the frozen-search control the later trained generations are
compared with at equal dollars.

## Transfer: the matrix, its rows and the rule (week 4)

Every target carries a transfer block: the certificate family, the instance, a regime level (larger is harder: the
exactness degree, the condition number, the order, the transform length), a structure class within the family (the
cubature domain, the method, the problem family, the transform, the objective), the certifier and the open flag. The
setting of a target relative to the targets an experience set came from is computed by the program: same instances,
new instances, harder or easier regime, a new family under the same certificate (another structure class), or a new
domain (another certificate); the open flag is recorded in every cell and never ranked.

An experience set is what the lab recorded on its source targets: the archived methods, the lessons extracted from
those targets' groups, the attempts behind them. Rows of the matrix are experience sets plus the two controls:
process-only (the set's process-level lessons with digits and the family's content terms redacted, which keeps the
craft and removes the knowledge) and unrelated (every archived method and lesson from targets under a different
certificate). Columns are targets in their settings. Carriers: `none` (the baseline, run once per target and seed),
`lessons` (the notebook), `lessons_gate` (the gate's one lesson and check, a frontier call), `skills` (archived methods as
reference implementations), `process_only`, `unrelated`. A cell runs N seeds of a group each, at matched attempts or
matched dollars (the baseline's spend per seed caps the carrier's group), and every row carries the matrix id, the cell
id, the experience, the carrier, the setting and the seed index. The analysis is a query over the ledger: the certified
gain (pass rate, optimal rate, best score) with the carrier minus the baseline, paired by seed, with a percentile
bootstrap interval over seeds (over attempts when there is one seed), rendered as one table per matrix.

The rule, enforced by the program: lessons are never measured on the family they came from. A content carrier is refused
when the target is one of the set's own instances, and refused when the target's family key (certificate family and
structure class, for example `quadrature/interval`) is one of the set's, unless `--allow-same-family` is passed, in which
case the cell is marked "same family (unreported)". So the experience of the interval rediscovery (`quad-d7`, whose
lessons name the answer) is measured on the square and the triangle (a new family under the same certificate) and on the
time integrators (a new domain), never on `quad-d9`.

## The ledger

One row per attempt: `id, role, model, model_version, precision, target, prompt_hash, context, output, tool_log,
certificate, diagnosis, gpu_seconds, cpu_seconds, dollars, seed, timestamp`, plus `provider`, `mode`,
`api_dollars`, `group_id` and `meta`. The `context` of a demo or ablation row carries the arm, the lesson ids and
the gate's check; the `meta` carries the concurrency and the check's outcome. Week-4 rows add the role's adapter
version, the tool-call limit, the transfer block, the stage of the loop (`theorist`, `base`, `lessons`, `proposals`,
`breaker`) and the generation id, the proposal a build was derived from or the candidate a breaker attacked, the citation
ids and reading-list snapshot, and, for the matrix, the matrix id, cell id, experience, carrier, setting and seed index;
the `meta` carries the round outputs, the dry runs, the proposal with its lemma statuses, the witness and its reward.
Harness rows share the table with role `harness:<name>`. The harness tables (`citations`, `lessons`, `artifacts`,
`provenance`, `detectors`, `dossiers`) live in the same SQLite file and the JSONL mirror.

## The repository and what never enters it

Committed: code, certifier tests, recorded provider fixtures, the reading list, scoreboard files, configuration.
Never committed: model weights, the SQLite ledger and its mirror, run outputs (gate memos, ablation reports), `.env`
and any API key. The ledger and the weights live on Drive; the repository holds nothing that cannot be rebuilt from
it plus Drive.

## What has run, and what has not

The package and its tests run on a CPU-only machine. The notebook ran end to end on a Colab G4 at bf16 on
2026-11-04 (weights pinned, vLLM serving, smoke test, the 16-attempt demo, see `RESULTS.md`). Week 3 adds the three
certifiers, the harnesses and the ablation harness, all exercised offline under pytest; the GPT fixtures were recorded
on Colab on 2026-11-05 (25 calls, $0.04) and the harness tests replay them; the gate and the ablation ran against the
served model on 2026-11-05 (cells 10 and 11, `RESULTS.md`). Week 4 adds the three roles, the episode runner, the loop
and the matrix, all exercised offline under pytest from synthetic fixtures; the loop ran one generation against the served
model on 2026-11-06 (cell 13: five quadrature rungs, gate memos, 20 archived methods, the theorist's proposals leading the
builder to Radon's 7-point rule, the lessons from the interval rung lowering the square's pass rate, see `RESULTS.md`), with
the 256-attempt search baseline on `quad-square-d5` (cell 15) and the ablation repeated with the fixed gate check (cell
11); the loop's GPT fixtures were recorded by cell 12 and wait on Drive for a commit. Not yet run: the frozen baseline for
every certifier family (cell 14) and the matrix cells (cell 16). Not yet exercised: the A100/L4 path, SGLang, the tunnel
cell, the thinking ladder with a real budget, the builder's dry run against the live server, and the Claude provider (no key
yet; its harness tests skip until recorded). Scoreboard entries marked `provenance: provisional` must be checked by a person
against the sources, with the librarian's reviews under `scoreboard/reviews/` as a starting point, before any novelty claim.
