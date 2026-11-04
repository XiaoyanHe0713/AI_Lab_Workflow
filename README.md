# Discovery at 27B: the lab, weeks 1 to 3

A fully automated numerical lab on one open model, `Qwen/Qwen3.6-27B`, with three roles that will later be
LoRA adapters and program harnesses around them. This repository holds weeks 1 to 3 of the build plan:
the model served from Colab, the repository skeleton (ledger, cost meter, role interface), the certifier
library (six certifiers), the judge sandbox, the scoreboard files, the demo loop with concurrent attempts,
the training gate, and the three harnesses (librarian, archive and lesson bank, checker) with the ablation
harness around them.

Rules that never bend: every score comes from a program (a certificate), never from a model's opinion;
verification of known results comes before any open problem; every attempt is logged with its cost; a
frontier model through an API may do the language parts of the harnesses during the build, but nothing a
harness writes enters a reward.

## Layout

| Path | What it is |
|---|---|
| `colab/serve.ipynb` | Drive, weights pinned by hash, 4-bit build, OpenAI-compatible server with LoRA serving, smoke test, demo (parallel requests), the gate runner and the online ablation (week 3 cells). Generated from `colab/build_notebook.py`. |
| `config/lab.yaml` | Storage path, endpoint, providers (`local`, `openai`, `anthropic`, each with a mode `offline`, `online` or `record`), the harness settings (provider, token budget of the lesson bank, detector thresholds, reading list), roles, judge limits. `${VAR:-default}` expands from the environment. |
| `config/model.yaml` | The frozen base (repo, revision, checkpoint hash) and the quantized build. |
| `config/prices.yaml`, `config/budget.yaml` | The price table; the frontier budget: $200 a month across GPT and Claude, warning at 80 percent, enforced in code. |
| `lab/ledger.py` | The attempt record (Work package 4 schema) in SQLite with a JSONL mirror, `provider`, `mode` and `api_dollars` columns, the `serving` table, pass@k, the month's frontier spend, and generic tables for the harnesses (`citations`, `lessons`, `artifacts`, `provenance`, `detectors`, `dossiers`). |
| `lab/cost.py` | The cost meter and the budget guard (locked for concurrent calls). |
| `lab/roles/base.py` | The role interface: provider kinds `local`, `openai` (GPT, first), `anthropic` (Claude, second); modes `offline`, `online`, `record`. The OpenAI provider sends `max_completion_tokens`, omits an unset temperature and carries a configured `reasoning_effort` (GPT-5). |
| `lab/certifiers/quadrature.py`, `pep.py`, `witness.py` | Weeks 1 to 2: exactness, worst-case rates by performance estimation, the witness check. |
| `lab/certifiers/order.py` | Week 3: convergence order from a seeded refinement study (manufactured solutions for an ODE, a 1-D Poisson problem and a composite integral), slope with a t-interval and pairwise orders. |
| `lab/certifiers/opcount.py`, `counted.py` | Week 3: operation count by instrumented arithmetic (an expression DAG of counted scalars; only operations that reach an output count) with the outputs checked against the exact transform, and cost models checked against instrumented runs. |
| `lab/certifiers/stability.py` | Week 3: the stability polynomial, the real- and imaginary-axis extents of the region of absolute stability, the order conditions and the SSP coefficient (radius of absolute monotonicity) of an explicit Runge-Kutta method in Butcher or Shu-Osher form. |
| `lab/judge.py`, `lab/sandbox_runner.py` | The judge: subprocess sandbox plus the certifier; the runner can wrap entry arguments as counted scalars for the operation-count family. |
| `lab/targets.py` | Targets with prompts, scoreboard entries and reading-list cutoffs: `quad-*`, `pep-*`, `order-*`, `opcount-dft-n*`, `ssp-s*-p*`, `stab-{real,imag}-s*-p*`. |
| `lab/demo.py` | `python -m lab.demo`: N attempts from one prompt, judged, priced, in the ledger, pass@1 and pass@16; `--concurrency 16` sends a group as parallel requests; prompt overrides and a logged post-check for the ablation arms. |
| `lab/gate.py` | `python -m lab.gate --targets ... --attempts 64 --thinking 0 512`: the training gate, a memo per target with pass@1/16/64 valid and optimal, the 5 to 80 percent verdict and the cost. |
| `lab/harness/base.py` | The harness base: every language call goes through the provider and the budget guard and into the ledger as role `harness:<name>` with provider, mode and API dollars. |
| `lab/harness/librarian.py`, `reading_list/*.yaml` | The librarian: the reading list with dates, retrieval under a per-target date cutoff (lexical ranking by a program, selection and summary by the provider), every citation logged, and the pass over `provenance: provisional` scoreboard entries. |
| `lab/harness/archive.py` | The archive: a content-addressed store of certified methods with witnesses and dossiers, linked to the attempt, reading-list snapshot, lessons, citations and cost; `trace()` is one SQL query. |
| `lab/harness/lessons.py` | The lesson bank: lessons at three levels (approach, implementation, process), each a rule with conditions and an evidence tag (observed, derived, proposed); consolidation under a token budget by add, replace and retire; the gate that selects one lesson and writes a concrete, validated check applied at use time. |
| `lab/harness/checker.py` | The checker: the frozen baseline record from ledger groups; hypotheses turned into deterministic detectors kept only when right on more than nine of ten held-out cases; method dossiers with claim, conditions and evidence, schema-validated. |
| `lab/harness/ablation.py` | `python -m lab.harness.ablation`: three arms on one verification family at matched dollars (no notebook, notebook, notebook with gate), every attempt in the ledger, a report per run. |
| `scoreboard/*.yaml` | Quadrature node counts, first-order rates, SSP coefficients and stability extents, DFT operation counts, convergence-order verification targets, with sources. |
| `tests/` | pytest suites: at least 20 known-answer cases per certifier, judge limits, ledger, cost, budget, roles, demo, concurrency, gate, harness base, and the harness suites per frontier provider (offline from recorded fixtures). Live tests carry the `online` marker. |
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
python -m lab.gate --targets quad-square-d5 quad-triangle-d4 --attempts 64 --thinking 0 512 --gpu G4 --precision bf16
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

List targets:

```bash
python -m lab.demo --list
```

On Colab (Pro+, background execution): open `colab/serve.ipynb`, set the GPU runtime, add the secrets
`HF_TOKEN`, `GITHUB_TOKEN` (private repository) and, for the ablation, `OPENAI_API_KEY`, fill cell 1 and run all
cells. Cells 4 to 9 pin the frozen base, build or reuse the 4-bit copy, start vLLM with the zero adapter, run the
smoke test and the demo with `CONCURRENCY` parallel requests. Cells 10 and 11 (off by default: `RUN_GATE`,
`RUN_ABLATION`) run the gate on `GATE_TARGETS` with the thinking ladder and the ablation harness online; both write
to the ledger on Drive, the memos to `gates/` and the reports to `ablations/` next to it.

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
ablation's notebook arms replay. Repeat both with `ANTHROPIC_API_KEY` and `-k anthropic` for Claude. Harness prompts
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
| `witness_check` | 1.0.0 | 30 s | Synthetic breaks reproduce, synthetic non-breaks do not, instances outside the class definition score zero. |
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

## The ledger

One row per attempt: `id, role, model, model_version, precision, target, prompt_hash, context, output, tool_log,
certificate, diagnosis, gpu_seconds, cpu_seconds, dollars, seed, timestamp`, plus `provider`, `mode`,
`api_dollars`, `group_id` and `meta`. The `context` of a demo or ablation row carries the arm, the lesson ids and
the gate's check; the `meta` carries the concurrency and the check's outcome. Harness rows share the table with
role `harness:<name>`. The harness tables (`citations`, `lessons`, `artifacts`, `provenance`, `detectors`,
`dossiers`) live in the same SQLite file and the JSONL mirror.

## The repository and what never enters it

Committed: code, certifier tests, recorded provider fixtures, the reading list, scoreboard files, configuration.
Never committed: model weights, the SQLite ledger and its mirror, run outputs (gate memos, ablation reports), `.env`
and any API key. The ledger and the weights live on Drive; the repository holds nothing that cannot be rebuilt from
it plus Drive.

## What has run, and what has not

The package and its tests run on a CPU-only machine. The notebook ran end to end on a Colab G4 at bf16 on
2026-11-04 (weights pinned, vLLM serving, smoke test, the 16-attempt demo, see `RESULTS.md`). Week 3 adds the three
certifiers, the harnesses and the ablation harness, all exercised offline under pytest; the harness tests replay
recorded GPT fixtures once those are recorded (see Providers). Not yet exercised: the A100/L4 path, SGLang, the
tunnel cell, the gate runner and the ablation harness against the served model (cells 10 and 11), and the Claude
provider (no key yet). Scoreboard entries marked `provenance: provisional` must be checked by a person against the
sources, with the librarian's reviews as a starting point, before any novelty claim.
