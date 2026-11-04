# Results: weeks 1 and 2

Date: 2026-11-04. Two machines: a CPU-only Windows laptop (Python 3.12, NumPy 2.5, SciPy 1.18, CVXPY 1.9.3,
SCS 3.3.1, PEPit 0.5.1) for the tests and the offline demo, and one Colab Pro+ session on a G4 (NVIDIA RTX PRO
6000 Blackwell Server Edition, 96 GB; Python 3.13; vLLM 0.33.0) for serving, the smoke test and the demo
against the frozen model.

## Tests

`pytest`: 193 tests; 191 pass on the machine above and the 2 live-API tests (marker `online`) are skipped
without keys (about 4 minutes; the PEP suite solves 30 SDPs with SCS, the judge suite spawns a sandbox per
case). Every certifier ships with more than 20 known-answer cases (quadrature: 61, PEP: 33, witness: 31;
judge: 24, ledger, cost, budget and config: 15, roles and providers: 20, weights: 6, demo: 3),
the judge's limits are exercised (static and run-time forbidden imports, timeout, memory, network, crash,
determinism), and the demo loop runs end to end on replayed fixtures.

## The Davis-Yin calibration

Three-operator splitting with f1 L1-smooth mu1-strongly convex, f2 convex, f3 L3-smooth convex;
mu1 = 0.1, L1 = 10, L3 = 1, alpha = 1, theta = 1, n = 4; metric ||w_n - w_n'||^2 / ||w_0 - w_0'||^2.

| Quantity | Value |
|---|---|
| certified rate (SCS, eps 1e-9) | 0.475452321 |
| bound rebuilt from the dual multipliers | 0.475452322 |
| PEPit 0.5.1 reference (MOSEK, example docstring) | 0.4754523346 |
| difference | 1.4e-8 (tolerance 1e-4) |
| dual slack matrix, smallest eigenvalue | -9.6e-10 |
| function-value terms residual | 2.4e-11 |
| smallest inequality multiplier | -2e-18 |
| quadratic-instance lower bound (curvature L1) | 0.466507 |
| Gram size, constraints | 26, 170 |
| solve time | 11.6 s (cap 60 s) |

The other three methods, same solver and tolerance:

| Method | Parameters | Certified | Published | Status |
|---|---|---|---|---|
| gradient descent, distance | mu 0.1, L 1, gamma 1, n 1 | 0.810000000 | 0.81 (tight) | matches |
| gradient descent, distance | mu 0.1, L 1, gamma 1, n 3 | 0.531440999 | 0.531441 (tight) | matches |
| Nesterov, function value | mu 0.1, L 1, n 2 | 0.347602233 | 0.467544 (upper bound); PEPit reference 0.3476022263 | under the bound, matches the reference |
| Douglas-Rachford, contraction | mu 0.1, L 1, alpha 3, theta 1, n 2 | 0.350127796 | 0.3501277966 (tight) | matches |

No closed-form published rate is used for Davis-Yin: Davis & Yin (2017) give linear rates under extra
assumptions that are not tight, and the sharpest known rates (Lee, Yi & Ryu 2022; Yi & Ryu 2022) are for
operator classes that differ from this one. The calibration point is PEPit's documented value.

## The demo loop, offline (machinery check, not a baseline)

`python -m lab.demo --target quad-d7 --attempts 16 --provider local --mode offline`

The 16 responses are synthetic, hand-written to cover the kinds of output a model produces (see
`tests/make_fixtures.py`): three derivations of the 4-node Gauss rule and one hard-coded copy, a 5-node
Gauss rule, an 8-node Newton-Cotes rule, a 5-node Lobatto rule, Simpson, a 3-node Gauss rule, two forbidden
imports, a prose-only answer, a syntax error, a crash, mismatched lengths, and a weight perturbed by 1e-6.

| Quantity | Value |
|---|---|
| valid rules (exact to degree 7) | 7 / 16 |
| optimal (4 nodes, the scoreboard's best known value) | 4 / 16 |
| pass@1, pass@16 (valid) | 0.438, 1.000 |
| pass@1, pass@16 (optimal) | 0.250, 1.000 |
| verify-first check (best known value found at least once) | PASS, 4 times in 16 attempts |
| cost | $0.0008 (judge CPU time only; no GPU seconds, no tokens; offline rows never count against the budget) |
| ledger rows written | 16 attempts (provider `local`, mode `offline`) + 1 group summary |

These numbers test the loop, the judge and the ledger. They say nothing about Qwen3.6-27B.

## The frozen base

`Qwen/Qwen3.6-27B` at revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`: 29 files, 55.6 GB, checkpoint hash
`sha256:4286c4e5279e2dbaa43b0dd44c44a18c6202c34d2ecd1bfca769dd7129966778` (sha256 over the sorted
`path:sha256` lines of the Hub manifest), pinned on 2026-11-04 in `config/model.yaml`. vLLM identifies the
checkpoint as `Qwen3_5ForConditionalGeneration`: a hybrid Gated-DeltaNet architecture with a vision tower,
which is why serving needs `--max-num-seqs` well below vLLM's default (one state block per sequence).

## Serving (first Colab session, 2026-11-04)

| Tier | Precision | Build | Server | Start-up | tokens/s, one stream | units/h (source) | $/h |
|---|---|---|---|---|---|---|---|
| G4 | bf16 | frozen base | vLLM 0.33.0, LoRA on, 32k context, 64 sequences, CUDA graphs | 1120 s | 20.8 base, 18.2 through the zero adapter `builder` | 11.77 (config/prices.yaml, not yet confirmed against the Resources panel) | 1.18 |
| A100 or L4 | int4 | not built yet | | | | | |

Both smoke completions (43 tokens each) succeeded and are in the `serving` table of the ledger on Drive; the
zero-initialized adapter is served under its own name, so an untrained role is exactly the base.

## The frozen baseline, quad-d7 (first Colab session)

`python -m lab.demo --target quad-d7 --attempts 16 --gpu G4 --precision bf16`, temperature 0.8, thinking off,
seeds 0 to 15, group `grp_4ebed252ac564b60`.

| Quantity | Value |
|---|---|
| valid rules (exact to degree 7) | 14 / 16 |
| optimal (4 nodes, the scoreboard's best known value) | 14 / 16 (every valid rule was the 4-node Gauss rule) |
| pass@1, pass@16 (valid) | 0.875, 1.000 |
| pass@1, pass@16 (optimal) | 0.875, 1.000 |
| verify-first check | PASS: the 4-node Gauss rule found 14 times in 16 attempts at $0.32 |
| the two failures | 4-node rules whose weights do not sum to 2 (constant-monomial errors 38.6 and 6.0): derivations that lost a factor |
| completion tokens per attempt | 901 to 3111, about 26 tokens/s while generating |
| cost | $0.3163 for 967.6 GPU seconds (3.164 units at the unconfirmed rate); judge CPU 0.1 s |
| wall time | 971 s for 16 sequential attempts |

Two readings of this record. First, the machinery works end to end on the frozen model: prompt, endpoint,
sandbox, certificate, cost, ledger. Second, quad-d7 is a verification target, not a training family: with
pass@1 at 0.875 it sits above the 80 percent ceiling of the proposal's training gate (a family trains only
when the base's pass@64 lies between 5 and 80 percent), as a rediscovery target should.

Precision used so far: bf16 on the G4. Units spent so far: about 7.4 in that session (0.63 h at 11.77 units/h;
the rate is still the table's estimate). No run on the 4-bit build has happened; when one does, it is marked
`int4` in the ledger and never enters a reported cell.

# Results: week 3

Date: 2026-11-04, the same CPU-only Windows laptop (Python 3.12, NumPy 2.5, SciPy 1.18). Nothing in this section ran
on a GPU: the week-3 code is exercised offline under pytest and once by hand against the local fixtures. The online
runs (the gate on the training-family candidates and the ablation against the served model) wait for the next Colab
session (cells 10 and 11 of `colab/serve.ipynb`).

## Tests

`pytest`: 364 tests; 352 pass and 12 are skipped (the two live-API smoke tests without keys,
and the harness tests that replay recorded frontier fixtures, per provider, until those are recorded). Known-answer cases:
convergence order 27, operation count 34, stability and SSP 42 (plus 5 through the judge's sandbox); concurrency and the
gate 14; the harness base 11; the librarian 4 offline (the retrieval test set of 8 cases, the cutoff, the citations); the
archive 2; the lesson bank 16; the checker 7.

## The three new certifiers

| Certifier | Calibration points (all reproduced) |
|---|---|
| `convergence_order` | Euler 1.00, Heun 2.00, RK4 4.0 to 4.1 (pre-asymptotic, inside the interval) on seeded ODE problems; central differences 2.00 on the Poisson problem, 1.0 with a boundary value off by O(h); composite left 1, midpoint 2, trapezoid 2, Simpson 4. |
| `operation_count` | radix-2 FFT: 2 N log2 N multiplications and 3 N log2 N additions exactly for N = 2..64 (320 operations at N = 16); naive DFT: 4 N^2 of each (2048 at N = 16); the cost model `5*n*log2(n)` verified at N = 8, 16, 32. |
| `stability_region_ssp` | real-axis extents 2, 2, 2.5127453, 2.7852936 and imaginary extents 0, 0, 1.7320508, 2.8284271 for Euler, Heun, Kutta 3 and RK4 (Hairer, Norsett & Wanner); SSP coefficients 1, 1, 2, 1.508 for SSPRK(2,2), (3,3), (4,3), (5,4), s - 1 for SSPRK(s,2) with s = 2..8, n^2 - n for SSPRK(n^2,3) with n = 2, 3; 0 for RK4, Kutta 3 and the 3/8 rule. |

## The librarian's retrieval test set (lexical ranking, no model)

89 documents in the reading list. All 8 cases of `reading_list/retrieval_tests.yaml` pass: the expected document is in the
top five and no document dated on or after the cutoff appears among the candidates (for example 83 of 89 documents are
hidden from the Gauss rediscovery at cutoff 1814, 12 from the Davis-Yin target at cutoff 2015). Every returned document is
a citation row with the target, the cutoff and the document's date.

## The gate runner, offline (machinery check)

`python -m lab.gate --targets quad-d7 --attempts 16 --mode offline` on the synthetic fixtures: pass@16 valid 1.000,
pass@1 valid 0.438, pass@1 optimal 0.250; verdict "above the band" (as the real quad-d7 record is: a verification
target, not a training family). The memo records provider `local` in mode `offline` and precision `replay`, so it can
never enter a reported cell. The training-family candidates (`quad-square-d5`, `quad-triangle-d4`) are the notebook's
default `GATE_TARGETS`.

## What the ablation harness measured

Offline, under pytest, once the GPT fixtures are recorded: three arms of 16 attempts on quad-d7, the local attempts
replayed from the synthetic fixtures and the lessons and the gate replayed from the recorded GPT fixtures. Because the
three arms replay the same 16 synthetic responses, their certified rates are identical by construction (7 of 16 valid,
4 optimal): the offline run measures that the machinery works end to end (extraction, consolidation under the token
budget, the gate's lesson and check, the check's outcome logged in every notebook-gate row and never scored, every
attempt in the ledger with its arm and lesson ids, matched dollars), not the notebook's effect. The ablation against the
served model, which is the measurement, has not run yet: it is cell 11 of the notebook (needs the Colab secret
`OPENAI_API_KEY`).

Frontier spend this month: nothing yet from this machine (no key set during the build; the recording of the harness
fixtures is the first live spend and goes into the repository's own ledger).
