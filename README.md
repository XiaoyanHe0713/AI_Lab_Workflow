# Discovery at 27B: the lab, weeks 1 and 2

A fully automated numerical lab on one open model, `Qwen/Qwen3.6-27B`, with three roles that will later be
LoRA adapters and program harnesses around them. This repository holds weeks 1 and 2 of the build plan:
the model served from Colab, the repository skeleton (ledger, cost meter, role interface), the certifier
library, the judge sandbox, the scoreboard files and the demo loop.

Rules that never bend: every score comes from a program (a certificate), never from a model's opinion;
verification of known results comes before any open problem; every attempt is logged with its cost; a
frontier model through an API may do the language parts of the harnesses during the build, but nothing a
harness writes enters a reward.

## Layout

| Path | What it is |
|---|---|
| `colab/serve.ipynb` | Week 1: Drive, weights pinned by hash, 4-bit build, OpenAI-compatible server with LoRA serving, smoke test, demo. Generated from `colab/build_notebook.py`. |
| `config/lab.yaml` | Storage path, endpoint, providers (`local`, `openai`, `anthropic`, each with a mode `offline`, `online` or `record`), the harness provider, roles, judge limits. `${VAR:-default}` expands from the environment. |
| `config/model.yaml` | The frozen base (repo, revision, checkpoint hash) and the quantized build. The notebook writes the pin to `$LAB_ROOT/config/model.yaml` on Drive; commit it here. |
| `config/prices.yaml` | The price table: dollars per Colab unit, units per hour per GPU tier, API prices, CPU rate. |
| `config/budget.yaml` | The frontier budget: $200 a month across GPT and Claude, warning at 80 percent, enforced in code. |
| `lab/ledger.py` | The attempt record (Work package 4 schema) in SQLite with a JSONL mirror, with `provider`, `mode` and `api_dollars` columns; `serving` rows for smoke tests; pass@k; the month's frontier spend. |
| `lab/cost.py` | The cost meter (GPU seconds, API tokens and CPU seconds to dollars from `config/prices.yaml`; offline replays cost nothing) and the budget guard. |
| `lab/roles/base.py` | The role interface: provider kinds `local` (the Qwen endpoint, the roles' provider), `openai` (GPT, first) and `anthropic` (Claude, second) behind one interface; modes `offline` (replay recorded fixtures), `online`, `record` (write fixtures). |
| `tests/fixtures/providers/<provider>/` | Recorded responses keyed by a hash of the request (model, messages, tools, temperature, seed). The pytest default is offline: no network, no key; a missing fixture fails the test. |
| `lab/certifiers/quadrature.py` | Exactness on the interval, square, cube and triangle; node count; positivity and interiority. |
| `lab/certifiers/pep.py` | Worst-case rates by performance estimation (PEPit, CVXPY, SCS) with the dual certificate. |
| `lab/certifiers/witness.py` | The witness check: re-runs a claimed break and reports whether it reproduces. |
| `lab/judge.py`, `lab/sandbox_runner.py` | The judge: subprocess sandbox (timeout, memory, scrubbed environment, no network, import guard) plus the certifier. |
| `lab/targets.py` | Targets with prompts and their scoreboard entries (`quad-d7`, `quad-square-d5`, `quad-triangle-d4`, `pep-...`). |
| `lab/demo.py` | `python -m lab.demo`: N attempts from one prompt, judged, priced, in the ledger, pass@1 and pass@16. |
| `lab/smoke.py` | `python -m lab.smoke`: one completion, tokens per second, units per hour, into the ledger. |
| `lab/weights.py` | Manifest hash and pin, zero-initialized LoRA adapter, server command lines. |
| `scoreboard/*.yaml` | Minimal node counts by degree and domain; published rates of the four methods, with sources. |
| `tests/` | pytest suites: at least 20 known-answer cases per certifier, judge limits, ledger, cost, budget, roles (against local fake servers), demo, weights. Live tests carry the `online` marker and are skipped without the key. |
| `RESULTS.md` | pass@k, the Davis-Yin calibration, the precision used and the units spent. |

## Commands

Local (Python 3.11 or newer; no GPU needed):

```bash
pip install -e ".[dev]"
```

```bash
pytest
```

The demo loop offline, against the synthetic fixtures (no key, no cost):

```bash
python -m lab.demo --target quad-d7 --attempts 16 --provider local --mode offline
```

The same loop through a frontier provider, recording every response as a fixture (needs `OPENAI_API_KEY`
or `ANTHROPIC_API_KEY`; counts against the budget; a machinery check, never a baseline of the lab's model):

```bash
python -m lab.demo --target quad-d7 --attempts 4 --provider openai --mode record
```

Live provider smoke tests (skipped by default; they make one tiny call each and record it under a temporary directory):

```bash
OPENAI_API_KEY=... pytest -m online tests/test_roles.py
```

List targets:

```bash
python -m lab.demo --list
```

One certifier by hand, for example the Davis-Yin calibration point:

```bash
python -c "from lab.certifiers.pep import PEPRateCertifier as C; print(C(in_process=True).certify({'method':'davis_yin','params':{'mu1':0.1,'L1':10,'L3':1,'alpha':1,'theta':1,'n':4}}, target='dy').details['rate'])"
```

On Colab (Pro+, background execution): open `colab/serve.ipynb`, set the GPU runtime, add the secrets
`HF_TOKEN` and `GITHUB_TOKEN` if needed, fill cell 1 and run all cells. Cells 4 to 9 download and pin the
frozen base, build or reuse the 4-bit copy on an A100 or L4 (bf16 is served on an H100 or G4), start vLLM
with a zero-initialized rank-64 adapter named `builder`, run the smoke test at the session's precision and
the demo. Everything lands under `LAB_ROOT` on Drive:

| On Drive | Content |
|---|---|
| `models/Qwen3.6-27B-bf16/` | the frozen base; its hash is pinned in `config/model.yaml` |
| `models/Qwen3.6-27B-w4a16/` | the 4-bit development build (llm-compressor AWQ), or nothing when bitsandbytes serves in flight |
| `adapters/builder/` | the zero adapter |
| `ledger.sqlite`, `ledger.attempts.jsonl` | the ledger and its mirror |
| `groups/<group id>.json` | one summary per demo group |
| `config/model.yaml` | the pin to commit |
| `logs/vllm.log` | the server log |

To reach the endpoint from your own machine instead of running the orchestrator in the notebook, set
`OPEN_TUNNEL = True` in cell 1 and copy the three `export` lines it prints, then:

```bash
python -m lab.smoke --gpu A100 --precision int4 --server vllm
```

```bash
python -m lab.demo --target quad-d7 --attempts 16 --gpu A100 --precision int4
```

Inside the notebook the same two commands run in-process from cells 8 and 9.

## Providers, modes and the frontier budget

The three roles always run on `local`, the Qwen3.6-27B endpoint. GPT (`openai`) and Claude
(`anthropic`) are for the language parts of the harnesses from week 3 on (librarian summaries and
relevance, lesson extraction and gating, written diagnoses), GPT first and Claude second behind the same
interface; nothing they write enters a reward. Every provider has a mode:

| Mode | What happens | Cost |
|---|---|---|
| `offline` | responses are replayed from `tests/fixtures/providers/<provider>/`, keyed by a hash of the request; a missing fixture raises, never a live call | nothing |
| `online` | live calls | GPU seconds for `local`, tokens for the frontier |
| `record` | live calls, each written as a fixture so the same test runs offline afterwards | as online |

The pytest default is offline (`LAB_MODE=offline`, `LAB_FRONTIER_MODE=offline` in `tests/conftest.py`).
Every ledger row records the provider and the mode, so a replayed response can never be mistaken for a
live one.

The frontier spend is capped at $200 a month (`config/budget.yaml`), enforced in code: before every
online or record call to GPT or Claude the guard sums this month's frontier dollars from the ledger
(rows with provider `openai` or `anthropic` in mode `online` or `record`, by calendar month) plus the
session's own calls; above 80 percent it warns once per session, at or above the cap it raises and makes
no call. Offline replays never count. To raise the cap, edit `config/budget.yaml` by hand: every row
records the cap that was in force, so the change stays visible in the ledger.

## The judge's rules during measurement

The candidate is a Python file with a fixed entry point (`rule()` for quadrature, `method()` for first-order
targets). It runs in a fresh `python -I` process with a scrubbed environment, a temporary working directory,
a wall-clock cap (120 s), a memory limit (2 GB; RLIMIT_AS on Linux, polled and killed everywhere), sockets
disabled and network modules unimportable. Library quadrature and optimisation solvers are forbidden:
`numpy.polynomial`, `scipy.integrate`, `scipy.optimize`, `scipy.special`, `scipy.interpolate`, `sympy`,
`mpmath`, `cvxpy`; so are `os`, `sys`, `subprocess`, `importlib`, `ctypes`, dynamic `__import__`, `exec`,
`eval`, `open`. NumPy arithmetic and `numpy.linalg` are allowed. A forbidden import, found by reading the
code or at run time by the import guard, fails the attempt. The same attempt scores the same twice.

## Certifiers

| Name | Version | Cap | Known-answer cases |
|---|---|---|---|
| `quadrature_exactness` | 1.0.0 | 30 s | Gauss-Legendre n = 1..10 passes at degree 2n-1 and fails at 2n; one corrupted weight fails; Clenshaw-Curtis, Newton-Cotes, tensor rules on the square and cube, six classical triangle rules, Duffy-collapsed rules; positivity and interiority flags. |
| `pep_worst_case_rate` | 1.0.0 | 60 s per SDP, in a separate process, cached by method signature | Gradient descent (12 parameter sets) and Douglas-Rachford match the tight published rates; Nesterov stays under its published bound and matches PEPit's reference; Davis-Yin reproduces PEPit's reference value (the calibration point); the dual reconstructs the bound (PSD slack, function terms cancel, multipliers nonnegative). |
| `witness_check` | 1.0.0 | 30 s | Synthetic breaks reproduce, synthetic non-breaks do not, instances outside the class definition score zero; quadrature and first-order witnesses. |

## The ledger

One row per attempt, in the proposal's order: `id, role, model, model_version, precision, target,
prompt_hash, context, output, tool_log, certificate, diagnosis, gpu_seconds, cpu_seconds, dollars, seed,
timestamp`, plus `provider`, `mode`, `api_dollars`, `group_id` and `meta`. The `serving` table holds the smoke tests: GPU tier, precision,
tokens per second, units per hour and its source, dollars per hour. The transfer matrix will be a query
over these tables.

## The repository and what never enters it

The code lives in a private GitHub repository that Colab pulls at the start of every session with a
fine-grained read-only token kept in Colab secrets (`GITHUB_TOKEN`), never pasted into the notebook; the
same pattern will serve Delta jobs from the job environment. Committed: code, certifier tests, recorded
provider fixtures, scoreboard files, configuration. Never committed: model weights, the SQLite ledger and
its mirror, run outputs, `.env` and any API key (`.gitignore` covers them from the first commit). The
ledger and the weights live on Drive; the repository holds nothing that cannot be rebuilt from it plus
Drive.

## What is not verified here

This repository was built and tested on a CPU-only machine. The Colab notebook, the vLLM/SGLang server
command lines, the llm-compressor AWQ build and the LoRA adapter loading are written to the current
documentation but were not executed on a GPU; the first Colab session will tell. The compute-unit rates
in `config/prices.yaml` are estimates: confirm them against the Resources panel and set `UNITS_PER_HOUR`
in the notebook (the ledger records which source was used). Scoreboard entries marked `provenance:
provisional` were entered from memory of the cited compilations and must be checked by the librarian
before any novelty claim.
