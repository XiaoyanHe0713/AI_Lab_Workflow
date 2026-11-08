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

`pytest`: 371 tests; 364 pass and 7 are skipped (the two live-API smoke tests without keys, and the
five harness tests parametrized over Claude until its fixtures are recorded; the five GPT ones replay the recorded fixtures). Known-answer cases:
concurrency, cost attribution and the gate 17, the thinking budget 5;
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

Offline, from recorded fixtures (`python -m lab.harness.ablation --target quad-d7 --attempts 16 --mode offline
--harness-mode offline`, run `abl_6b502ff5b0684c64`): three arms of 16 attempts on quad-d7, the local attempts replayed
from the synthetic fixtures and the lessons and the gate replayed from the GPT fixtures recorded on Colab on 2026-11-05.

| Arm | valid | optimal | pass@1 valid | pass@1 optimal | pass@16 valid | gate checks |
|---|---|---|---|---|---|---|
| no notebook | 7 / 16 | 4 | 0.438 | 0.250 | 1.000 | - |
| notebook (5 lessons, 2000-token budget) | 7 / 16 | 4 | 0.438 | 0.250 | 1.000 | - |
| notebook with gate | 7 / 16 | 4 | 0.438 | 0.250 | 1.000 | 9 of 11 passed |

The gate chose a process lesson ("self-test the rule on every monomial up to the required degree before submission") and
wrote the check `abs(sum(weights)-2) < 1e-9 and len(nodes) <= 8`; applied to the 11 attempts that returned a rule, it
failed exactly the two rules whose weights do not sum to 2 (the wrong-length rule and the perturbed weight), and the
outcome sat in each row's meta without touching a score. Because the three arms replay the same 16 synthetic responses,
their certified rates are identical by construction: this run shows the machinery working end to end (extraction,
consolidation, gating, the check at use time, every attempt in the ledger with its arm and lesson ids, matched dollars),
not the notebook's effect. The ablation against the served model, which is the measurement, is cell 11 of the notebook
and has not run yet.

## The librarian's pass over the provisional scoreboard entries

`scoreboard/reviews/*.20261105.openai.yaml` (16 entries, GPT reading the lab's own document summaries; a person
checks the sources before any change):

| Board | consistent | inconsistent | cannot verify |
|---|---|---|---|
| quadrature (7 provisional entries) | 6 | 1: triangle-d5, which credits the minimality of 7 nodes to Möller's bound; the review attributes it to Radon (1948) | 0 |
| time_integrators (5) | 5 | 0 | 0 |
| fast_transforms (4) | 0 | 1: opcount-dft-n8, where the literature's split-radix count (56) is below the certified radix-2 value (120); the entry already says the conventions differ | 3: the ranker handed the n = 16, 32, 64 entries an unrelated document (their source field says "as above"), so the review could not read them |

Left for a person: the attribution in triangle-d5 (a value-free wording change, still a reviewed diff). The three
`source: as above` fields were replaced by the full source text in the same commit, so the next pass can read them.

## Frontier spend

The recording on Colab made 25 GPT (gpt-5-mini) calls for $0.0365 in total: 4 retrievals, 16 reviews, 1 lesson
extraction, 2 gates, 1 detector proposal, 1 dossier. Every call is an attempt row with role `harness:<name>`, provider
`openai`, mode `record` and its dollars in the ledger on Drive, where the month's spend is read from. Two recorded
prompts (the extraction and the detector proposal) named a noise-level "worst monomial" for the passing rules, a detail
that differs between machines; the digests no longer include it, and those two fixtures were re-keyed under the stable
prompt with the original request text kept inside the file.


## The gate on the training-family candidates (second Colab session, 2026-11-05, G4, bf16)

Cell 10 of the notebook: `python -m lab.gate --targets quad-square-d5 quad-triangle-d4 --attempts 64 --thinking 0 512`,
16 parallel requests, 4096 answer tokens, temperature 0.8, seeds 0 to 63. The frozen base, no adapter. Read with the
three corrections below; the rows are in the ledger on Drive (groups grp_1a14d857f58245a0, grp_ff95ca660e71474f,
grp_539445a3f0434916, grp_0cdb1fc6c9154169).

| Target | thinking | valid | optimal | pass rate | pass@16 valid | no code (at the cap) | verdict (pass rate) | wall | GPU cost (corrected) |
|---|---|---|---|---|---|---|---|---|---|
| quad-square-d5 (best known 7) | 0 | 38 / 64 | 0 | 0.594 | 1.000 | 17 (17) | in the band | 678 s | $0.22 |
| quad-square-d5 | 512 | 4 / 64 | 0 | 0.062 | 0.694 | 59 (64 of 64 at the cap) | in the band, but see 3 | 798 s | $0.26 |
| quad-triangle-d4 (best known 6) | 0 | 0 / 64 | 0 | 0.000 | 0.000 | 53 (53) | below the band | 703 s | $0.23 |
| quad-triangle-d4 | 512 | 0 / 64 | 0 | 0.000 | 0.000 | 64 (64) | below the band | 798 s | $0.26 |

What the run says about the model: on the square the base finds a valid degree-5 rule in 59 percent of attempts, and
every valid rule is the 9-node 3 x 3 Gauss tensor product; Radon's 7-point rule never appears (verify-first FAIL, 0 of
128). On the triangle no attempt is valid at either budget: the few rules that return are off by O(1) on the constant
monomial (the area 1/2 is dropped or the barycentric map is wrong), and most attempts run out of tokens before any code.
With a 59 percent pass rate and a score spread between valid and invalid attempts, quad-square-d5 is a training
candidate; quad-triangle-d4 is below the band and needs a curriculum or an imitation warm start before it can train.

Three corrections, all fixed in the code after this run:

1. The printed costs ($2.34, $4.17, $3.47, $4.17; 43,280 GPU seconds) priced every attempt by its own request latency
   although 16 requests were in flight together. The GPU's time is the wall time of the waves: 2,977 s in all, 9.7 units,
   about $0.97 at the table rate ($0.22 to $0.26 per group as above). The ledger rows on Drive keep the inflated numbers;
   from now on a wave's wall time is shared among its attempts in proportion to their latencies (`meta.gpu_attribution`).
2. The memos said "above the band" for the square at thinking 0 because the verdict read pass@64, which is 1 whenever one
   attempt of the 64 passes. The verdict now reads the pass rate over the group (the per-attempt probability a GRPO
   group needs to vary), and the memo reports pass@16 and pass@64 beside it.
3. "Thinking 512" was thinking switched on with 512 extra completion tokens, not a budget: all 128 attempts at that rung
   ran to the 4,608-token cap inside their reasoning and 123 produced no code. A positive budget is now two requests, the
   reasoning capped at the budget and closed with Qwen's early-stop sentence, then the answer continued after `</think>`
   with the role's own token limit; both calls are in the tool log. The 512 rows above measure "thinking on, output
   capped", nothing else, and the rung has to be re-run. The gate also takes `--max-tokens` (the notebook's
   `GATE_MAX_TOKENS`, default 8192), because at thinking 0 most triangle attempts hit the 4,096-token cap before any code.

Session cost for the four groups: about 50 minutes of G4 time, 9.7 units at the unconfirmed 11.77 units per hour.

## The ablation against the served model (cell 11, second Colab session, 2026-11-05, G4, bf16)

`python -m lab.harness.ablation --target quad-d7 --attempts 16 --harness-provider openai --harness-mode online`,
16 parallel requests, 4096 answer tokens, thinking off, seeds 0 to 15 in every arm; GPT (gpt-5-mini) wrote the lessons
and the gate (2 calls, $0.0045). Run `abl_10601f5e6f104d3a`; groups grp_1643ddad076f4ccd, grp_fe7600b8ff714c16,
grp_6172f576c0974b49 in the ledger on Drive.

| Arm | valid | optimal | pass rate | failures | GPU cost (corrected) |
|---|---|---|---|---|---|
| no notebook | 13 / 16 | 13 | 0.812 | a derivation whose weights sum to 7.45, non-finite nodes, no code block | $0.03 |
| notebook (5 lessons) | 14 / 16 | 14 | 0.875 | two self-check snippets that compare a list with a number and crash | $0.03 |
| notebook with gate | 11 / 16 | 11 | 0.688 | three without a code block, a failed assert of the model's own, 4 nodes with 16 weights | $0.03 |

Read this as a machinery run, not as the notebook's effect, for three reasons. With 16 attempts per arm the standard
error of a pass rate near 0.8 is about 0.1, and the differences (+0.06, -0.12) sit inside it. quad-d7 is a verification
target: the lessons were extracted from the model's own certified attempts on it, and the gate chose the approach
lesson "use 4-point Gauss-Legendre", which names the known result; on a rediscovery target the notebook is a channel
for the answer, so a gain here would prove nothing. The measurement the proposal asks for puts experience from one
family to work on another (E1), which is week 4 and later. And the gate check never evaluated: "applied 12, passed 0,
failed 0" means all twelve returned an error, because the check namespace was passed as the locals of `eval`, which a
generator expression inside the check cannot see (the gate's own check calls `abs` inside `all(... for w in weights)`).
Fixed on 2026-11-05 with that exact expression as a test; the twelve outcomes of this run are lost (their rows hold the
error string in `meta.gate_check`).

What the run does show: the three-arm loop, the lesson extraction and consolidation, the gate and its check, and the
ledger rows with their arm and lesson ids all work against the served model; and the lessons changed the model's
behaviour in a visible way, as every valid notebook attempt carried self-verification code (asserts on the weight sum,
the node interval and the monomials), which is also what the two new failures in the notebook arm were.

Costs: the printed per-arm dollars ($0.32, $0.28, $0.25) are inflated by the concurrent-latency defect described above;
the arms' wall times were 88, 77 and 78 s, about 0.8 units or $0.08 in all at the table rate. Session total for cells
10 and 11: about 54 minutes of G4 time, 10.5 units.

# Results: week 4

Date: 2026-11-05, the same CPU-only Windows laptop (Python 3.12, NumPy 2.5, SciPy 1.18, SymPy 1.14). Nothing in this
section ran on a GPU: the week-4 code (the three roles, the episode runner, the loop, the transfer matrix) is exercised
offline under pytest from synthetic fixtures and once by hand. The verification run against the served model is cells
13 to 16 of `colab/serve.ipynb` (the loop on the quadrature curriculum, the frozen baseline for every certifier family,
the 256-attempt search baseline on `quad-square-d5`, the first matrix cells); its results are recorded below once it
has run.

## Tests

`pytest`: 435 tests; 426 pass and 9 are skipped (the two live-API smoke tests without keys, the six harness and loop
tests parametrized over Claude until its fixtures are recorded, and the loop's lesson-extraction test over GPT until the
notebook's recording cell has run on this branch); about 7 minutes. New known-answer cases: the lemma check 29 (15 true identities, 6 false claims, 7 trivial
checks, 9 invalid checks, duplicates, crashes, the sandbox allowance), the transfer blocks and settings 17, the roles 9
(theorist on fixtures with and without the revision round, the builder's dry run, five breaker cases), the loop 3, the
matrix 7, the witness certifier 6 more (the class bound, the tolerance per term, malformed inputs).

## The witness check, version 1.1.0

The quadrature class now bounds a polynomial witness's coefficients by one in absolute value and scales the tolerance
with the number of terms. Before, a witness with coefficients of 1e8 would have turned rounding error into a "break" of
any floating-point rule, which is a tolerance-edge exploit of the kind Work package 7 lists; now it is an instance
outside the class and pays nothing. The 31 known-answer cases of version 1.0.0 still pass, with the one test that used a
coefficient of 2 moved to the outside-the-class cases.

## The lemma check (the theorist's reward), calibration

Fifteen true identities pass through the sandbox with SymPy allowed (moments of monomials on the interval, the square
and the triangle, orthogonality of P2 and P3, the root count and location of P4 and P5, a finite sum, a binomial
symmetry, a determinant, a derivative); six false claims fail; seven constant or symbol-free checks are refused as
trivial before anything runs (`1 - 1`, `x - x`, `sqrt(4) - 2`, ...); a check that raises is a failed check (`IndexError`
on `solve(x**2 + 1, x)[9]`); a proposal of five lemmas with two passing scores 0.4 and passes. The sandbox allowance is
per run: `import sympy` fails a candidate judged for a certificate and succeeds for a lemma file, and `os` stays
forbidden in both. Known weakness, in the exploit catalogue: a tautology with a symbol (`simplify(x - x)`) passes the
filter; the delayed credit and the person's reading of the ten best theorist attempts per family are the controls.

## The loop, offline (machinery check, not a measurement)

`python -m lab.loop --targets quad-d7 quad-square-d5 quad-triangle-d2 --attempts 8 --theorist-attempts 4
--proposal-attempts 4 --breaker-attempts 5 --breaker-candidates 2 --mode offline --no-harness --concurrency 4`, every
role replayed from the synthetic sequences of `tests/fixtures_week4.py` (hand-written: 8 builder files per rung of the
kinds a model produces, 4 theorist proposals with their revision, 5 breaker replies), lemma checks and certificates real.

| rung | target | gate verdict (pass rate) | optimal | proposal group | theorist lemmas | survivors | confirmed breaks | archived |
|---|---|---|---|---|---|---|---|---|
| 0 | quad-d7 | in the band (0.625, 5/8) | 3 | 1.000 (4/4) | 4/5 in 4 attempts | 2 | 0 | 2 |
| 1 | quad-square-d5 | in the band (0.375, 3/8) | 1 (Radon's 7-point rule, synthetic) | 0.500 (2/4) | 5/6 in 4 attempts | 2 | 0 | 2 |
| 2 | quad-triangle-d2 | in the band (0.500, 4/8) | 2 | 0.500 (2/4) | 4/5 in 4 attempts | 2 | 0 | 2 |

78 ledger rows (builder, theorist, breaker) with their stage, prompt hash, context, tool log, certificate, diagnosis and
cost; 6 methods in the archive snapshot, each with a template dossier and a witness record (5 attacks, 0 confirmed
breaks); a gate memo per rung; the generation record with the ledger pointer and the delayed credit of every proposal
(for example on the square: the proposal with the orbit structure led to 1 certified build of 2, best 7 against the
gate group's best 7, credit 0, pass-rate gain +0.125). The breaker's replies exercised the whole reward: a monomial inside
the class that does not reproduce (0), one above the target degree (outside the class, 0), a probe of four monomials
followed by a bounded polynomial (0), prose without a witness (0), coefficients of 1e8 (outside the class, 0). Because
every role replays the same synthetic texts, the numbers test the loop, the lemma checks, the witness check, the archive
and the ledger; they say nothing about Qwen3.6-27B.

## The matrix, offline (machinery check)

The experience of `quad-d7` (3 archived methods, 4 program-written lessons: approach, implementation, two process) applied
to `quad-square-d5` and `quad-triangle-d2` (new family under the same certificate) and refused on `quad-d9` (same family,
harder regime) and on `quad-d7` itself (same instances); the two controls: process-only lessons with content redacted
("Self-test the rule on every monomial up to the required degree and check that the weights sum to the volume 2" becomes
"Self-test the [...] on every [...] up to the required [...] and check that the [...] sum to the volume [...]") and
unrelated experience (two lessons from `ssp-s3-p3`, a new domain). Two seeds of 8 attempts per cell; every cell row in the
ledger with its matrix id, cell id, experience, carrier, setting and seed index; the analysis a query over the ledger.
Every gain is 0.000 [0.000, 0.000] by construction (the carrier cells replay the same synthetic files as the baseline),
which is what the check wants: the machinery adds nothing of its own to a score.

## The verification run on Colab (cells 11 to 13 and 15; 2026-11-06 to 07, G4, bf16)

Branch `build/week-4-loop` at 574aeaa; vLLM ready after 785 s with the zero adapter; smoke 23.9 tokens/s base, 21.7
through the adapter. Session: 6.04 h of G4, 71.1 units, $7.11 at the table rate (unconfirmed). The notebook ran with
`RUN_ABLATION`, `RECORD_FIXTURES`, `RUN_LOOP` and `RUN_SEARCH_BASELINE` on; `RUN_BASELINES` (the frozen baseline for
every certifier family) and `RUN_MATRIX` were off and have not run yet.

### The loop, one generation on the quadrature curriculum (cell 13)

`python -m lab.loop --targets quad-d7 quad-square-d5 quad-triangle-d2 quad-triangle-d3 quad-triangle-d4 --attempts 64
--theorist-attempts 16 --proposal-attempts 16 --breaker-attempts 8 --breaker-candidates 4 --tool-calls 0 --carrier lessons
--max-tokens 8192 --thinking 0 --concurrency 16 --harness-provider openai --harness-mode online`, generation
`gen_bcb53faacaa14087`, 4 h 26 min wall, GPU 15,704 s (4.36 h), $5.24; GPT (gpt-5-mini) for the lessons, the gate and
the dossiers, $0.08 in all; 1,715 rows in the ledger on Drive, 20 methods in the archive snapshot, 28 active lessons.
The theorist had one revision round after its lemma checks, the breaker one probe round, the builder no dry run.

| rung | target | gate verdict (pass rate, valid / 64) | optimal | lesson group | proposal group (16) | theorist: lemmas passed, proposals certified | breaker: confirmed breaks / attacks | archived |
|---|---|---|---|---|---|---|---|---|
| 0 | quad-d7 | above the band (0.859, 55) | 55 | 0.734 (47) | 0.812 (13) | 57/77, 16/16 | 0 / 32 | 4 |
| 1 | quad-square-d5 | above the band (0.812, 52) | 0 | 0.391 (25) | 0.188 (3), 2 optimal | 51/74, 14/16 | 0 / 32 | 4 |
| 2 | quad-triangle-d2 | above the band (1.000, 64) | 64 | 0.797 (51) | 1.000 (16) | 42/82, 11/16 | 0 / 32 | 4 |
| 3 | quad-triangle-d3 | in the band (0.203, 13) | 12 | 0.266 (17) | 0.375 (6) | 24/66, 5/16 | 0 / 32 | 4 |
| 4 | quad-triangle-d4 | in the band (0.062, 4) | 2 | 0.000 (0) | 0.000 (0) | 13/68, 5/16 | 0 / 32 | 4 |

What the run says, rung by rung, with the standard error of a rate at n = 64 around 0.06:

1. The gate at 8,192 answer tokens. quad-square-d5 passes 52 of 64 (0.812), above the band, where it passed 38 of 64
   (0.594) at 4,096 tokens on 2026-11-05: the verdict on this target depends on the token cap, and the family sits at the
   band's upper edge. quad-triangle-d3 (0.203, 25 attempts at the cap) and quad-triangle-d4 (0.062, 4 valid of which 2
   are the optimal 6-point rule; 27 at the cap, 21 without a code block) are in the band, where d4 was below it at
   4,096 tokens. quad-d7 (0.859) and quad-triangle-d2 (1.000) are rediscovery targets above the band.
2. The square. The frozen base's best plain attempt is an 8-node degree-5 rule (four axis points at +-sqrt(7/15), weight
   40/49, four diagonal points at +-sqrt(7)/3, weight 9/49), one node better than the 9-node tensor product that was the
   only valid rule on 2026-11-05. Radon's 7-point rule appeared only in the proposal group: 2 of the 16 builds derived from
   the theorist's proposals wrote exactly it (centre weight 8/7, (0, +-sqrt(14/15)) weight 20/63, (+-sqrt(3/5), +-1/sqrt 3)
   weight 5/9), solved from the even moment equations in the file, against 0 of 64 in the gate group, 0 of 64 in the lesson
   group and 0 of 256 in the search baseline. The delayed credit of those two proposals is +1.0 each (best 7 against the gate
   group's best 8). This is the theorist-to-builder channel working, not a discovery: the proposals name the orbit
   structure, which the base knows from its training data on a rediscovery target.
3. The lessons hurt. The lesson group ran with the gate's one lesson and check; on the square, where the bank held only
   lessons from the interval rung (a new family under the same certificate, the first E1 cell of this kind), the pass rate
   fell from 0.812 to 0.391, far outside noise. The diagnoses say why: at least 20 of the 39 failures are the model's own
   assertions on the monomial x^0 y^6 (degree 6, outside the degree-5 class) or on a weight sum of 2.086 against the area 4,
   so the lesson's self-test prescription, applied beyond the class, killed valid files. On the triangle rungs the lesson
   groups wrote longer outputs (13, 32 and 43 attempts without a code block against 0, 25 and 21) and d4 fell to 0 of 64,
   inside noise; d3 rose from 13 to 17, inside noise. Rung 0's lesson group is not a transfer cell: the bank already held the
   week-3 ablation's lessons on quad-d7 itself, so it measured same-instance lessons (0.734 against 0.859), as the ablation
   did. The lesson texts and the gate's checks are in the generation record on Drive
   (`generations/gen_bcb53faacaa14087.json`), not in the notebook output.
4. The theorist. Lemma checks passed at 74, 69, 51, 36 and 19 percent down the curriculum; 16, 14, 11, 5 and 5 proposals
   per rung carried at least one checked lemma. On the triangle, 3, 6 and 6 proposals' checks "did not run": the sandbox
   failed, consistent with the 120 s cap and the judge CPU of 136, 348 and 232 s per theorist group (symbolic double
   integrals over the triangle are slow in SymPy; one slow lemma stalls the whole file). Four replies held no parsable JSON
   (trailing text after the object), and the revision round let two attempts run to 16,384 tokens.
5. The breaker. 160 attacks on 20 certified candidates, 0 confirmed breaks: 80 witnesses inside the class did not reproduce,
   17 were outside the class (a monomial above the target degree, or coefficients above one), 63 replies held no witness
   (prose, or a probe where the witness was due). The certificates held; the reward was zero throughout, as a class-covering
   certificate predicts, and no exploit alarm was raised.

### The search-only baseline, 256 attempts on quad-square-d5 (cell 15)

Group `grp_65174bd48bb1407a`, seeds 0 to 255, 8,192 answer tokens, thinking 0, 16 parallel requests: 197 of 256 valid
(pass rate 0.770, in the band), 0 optimal; scores 9 on 185 attempts, 8 on 10, 12 and 16 once each; 5 at the token cap,
5 without a code block; GPU 4,472 s (74.5 min), 14.6 units, $1.46. Verify-first FAIL: the 7-point rule never appears
without the theorist, and the best search-only rule is again the 8-node axis-and-diagonal rule (seed 11). Seeds 0 to 63
are the same seeds as the loop's gate group and seed 11 reproduced the same file, so the two runs share their first 64
samples; a baseline meant to be independent should start at `--seed 64`.

### The ablation repeated with the fixed gate check (cell 11, 2026-11-06)

`python -m lab.harness.ablation --target quad-d7 --attempts 16 --harness-provider openai --harness-mode online`, run
`abl_8bf88ffd40334802`, 4,096 answer tokens, GPT 2 calls $0.0048: no notebook 13 of 16, notebook 14 of 16, notebook with
gate 10 of 16; the gate's check (4 nodes, lengths, interval, finite weights, sum 2 within 1e-12) now evaluates and passed on
all 10 certified attempts, which closes the week-3 defect. The reading is unchanged: same-instance lessons on a
rediscovery target, differences inside the noise of n = 16.

### Fixtures recorded (cell 12)

Six harness and loop tests recorded against GPT and replayed offline on Colab (the loop's lesson extraction and gate among
them), the local sequences rebuilt, the provisional-entry reviews rewritten, all exported to
`exports/recorded-fixtures/` on Drive for a commit from this machine; until that copy lands, the loop's GPT test skips.

### Defects and follow-ups the run exposed

1. One slow lemma check stalls a whole proposal: give the lemma program a per-check time budget inside the file and mark
   the rest "skipped", so partial verdicts survive the 120 s cap.
2. A proposal with text after its JSON object fails to parse: take the first balanced object.
3. The revision round doubles the theorist's token cap: give the revision its own, smaller cap.
4. Search baselines share seeds with the gate groups: offset the seed.
5. The loop's lesson group at rung 0 drew on lessons already in the bank from earlier runs: record their source groups
   (done, in `lesson_sources`) and offer a fresh bank per generation.
6. The lessons' harm on the square is the first across-structure result and deserves the matrix treatment: the same
   experience as a row, with the process-only and unrelated controls, at five seeds (cell 16), after the family baselines
   of cell 14.

# Results: week 5

Date: 2026-11-06, the same CPU-only Windows laptop. Nothing in this section ran on a GPU: week 5 (the diagnosis as a program's
object in the loop and the matrix, and the two target-side control rows) is exercised offline under pytest from the synthetic
fixtures. Its first use against real attempts is the mining cell of `colab/serve.ipynb` on the ledger that cell 13 wrote, which
needs no GPU time, and then the matrix with the control rows on the next session (the run recorded below).

## Tests

`pytest`: 439 tests; 430 pass and 9 are skipped (the two live-API smoke tests without keys, the six harness and loop tests
parametrized over Claude until its fixtures are recorded, and the loop's lesson-extraction test over GPT until the fixtures that
cell 12 recorded on 2026-11-07 are copied from Drive into the repository). New known-answer cases (`tests/test_week5.py`): mining without a harness on the demo group,
the firing table, the query and the diagnosis-only text a diagnosis names, mining from the ledger and its command; the
extraction prompt with and without detectors and the detectors a lesson addresses through the bank; the two control rows
and the symptom-change column on the synthetic matrix; the loop mining its base group and recording symptom rates. The
week-4 loop test also checks that with 8 attempts per rung no detector is kept and the retrieval query is the week-4 one.

## What the offline checks show

The built-in detectors through the held-out gate on the synthetic groups (a detector is kept only if it fires at least twice
on the half the proposer never saw and every firing is a failure):

| group | attempts | valid | kept (fires on the whole group) | rejected |
|---|---|---|---|---|
| quad-d7, the week-2 demo group | 16 | 7 | sandbox_crash (4) | no_code_block, forbidden_import (one held-out firing), weights_do_not_sum_to_volume, too_low_degree |
| quad-square-d5, the matrix's `none` cell (2 seed blocks of 8) | 16 | 6 | sandbox_crash (4), weights_do_not_sum_to_volume (2) | no_code_block, forbidden_import, too_low_degree |
| quad-triangle-d2, the same | 16 | 8 | no_code_block (2), sandbox_crash (2), too_low_degree (2), weights_do_not_sum_to_volume (2) | forbidden_import |

* The loop with 8 attempts per rung keeps no detector (nothing fires twice on a held-out half of 4), so the theorist's
  query stays the target's title as in week 4 and the recorded theorist fixtures replay; with 16 attempts on `quad-d7` it
  keeps `sandbox_crash` and the query names "the candidate raised or failed to run".
* The diagnosis-only text on the square reads "4 of 16 attempts: the candidate raised or failed to run; 2 of 16 attempts:
  the weights do not sum to the domain's volume", and nothing else: no remedy.
* The control rows replay the same synthetic files as the baseline, so every gain and every symptom change is 0.000 by
  construction, which is what the check wants: the machinery adds nothing of its own to a score or a symptom rate. A second
  try replays as the model keeping its file (`revised_after_dry_run` false on every row).
* The extraction prompt without detectors is byte for byte the week-3 one, and the recorded GPT fixtures of the lesson
  harness and the ablation replay unchanged.
* The fixture set: 26 local sequences gained a second round (a second try keeps the file), 2 diagnosis-only sequences were
  added, the 2 breaker sequences for a fourth `quad-d7` candidate were re-keyed because the generator now mirrors the loop's
  8 attempts per rung, and 1 orphan from the week-3 re-keying was removed.

## The first run on Colab (cells 12, 16 and 17; 2026-11-07, G4, bf16)

Branch `build/week-5-diagnosis-controls` at 1b9f9c0 (week 5 with the week-4 run record merged in); an RTX PRO 6000
Blackwell reported as tier G4; vLLM ready after 1110 s with the zero adapter; smoke 19.6 tokens/s base, 18.1 through the
adapter. Session: 4.55 h, 53.5 units, $5.35 at the table rate (unconfirmed). The notebook ran with `RECORD_FIXTURES`,
`RUN_MATRIX` and `RUN_MINE` on and `RUN_DEMO` off. Cell 12 recorded six harness and loop tests against GPT (3 min 9 s),
replayed them offline and exported them to `exports/recorded-fixtures/` on Drive, where they wait for a copy into the
repository. Cell 17 ran after cell 16 and without GPT: its secret lookup returned nothing after the 4.5-hour matrix cell,
although cell 12 had read the key in the same runtime, so the built-in detectors stood alone.

### The matrix with the two control rows (cell 16)

`python -m lab.harness.matrix --sources quad-d7 --targets quad-square-d5 quad-triangle-d4 --carriers none second_try
diagnosis_only lessons process_only unrelated skills --seeds 3 --attempts 16 --max-tokens 8192 --thinking 0 --concurrency 16
--no-harness --tool-calls 0`, matrix `mtx_7db6a53d87d24a62`: 14 cells, 42 groups of 16 attempts, $4.21 of GPU time for
the twelve carrier cells at the table rate. The experience is the loop's quad-d7 archive and lessons. The unrelated
control found no other family in the archive, so its experience set was empty and its prompt is the plain prompt plus one
sentence, "(The notebook holds no lessons for this family yet.)": that row measures the sentence. The diagnosis-only row's
text came from the built-in detectors mined on each target's `none` cell (48 attempts): on the square "too low degree" at
0.10 and "weights do not sum to the volume" at 0.17; on the triangle "no code block" 0.33, "sandbox crash" 0.17, "too low
degree" 0.23 and "weight sum" 0.25. The second-try row had one certifier dry run and a revision per attempt and no lesson.

quad-square-d5 (best known 7 nodes). Failures of 48 attempts by the judge's reason; the symptom change is the kept
detector's firing rate with the carrier minus without:

| cell | valid / 48 | gain in pass rate [95% CI] | Radon's rule | best | not exact | crash | no code block | tokens per attempt | symptom change |
|---|---|---|---|---|---|---|---|---|---|
| none (baseline) | 34 (0.708) | | 0 | 8 | 13 | 0 | 1 | 2,743 | |
| second_try | 40 (0.833) | +0.125 [+0.000, +0.250] | 3 | 7 | 6 | 1 | 1 | 5,719 | weight sum -0.12, too low degree -0.02 |
| diagnosis_only | 34 (0.708) | +0.000 [-0.062, +0.125] | 0 | 8 | 13 | 1 | 0 | 3,171 | too low degree -0.06, weight sum +0.06 |
| lessons | 35 (0.729) | +0.021 [-0.250, +0.250] | 0 | 9 | 1 | 9 | 3 | 3,035 | weight sum -0.17, too low degree -0.08 |
| skills | 40 (0.833) | +0.125 [-0.188, +0.375] | 0 | 8 | 7 | 0 | 1 | 4,567 | too low degree -0.06, weight sum -0.06 |
| process_only | 27 (0.562) | -0.146 [-0.250, +0.000] | 0 | 9 | 0 | 19 | 2 | 1,918 | weight sum -0.17, too low degree -0.10 |
| unrelated (the empty notebook sentence) | 39 (0.812) | +0.104 [+0.000, +0.188] | 1 | 7 | 7 | 1 | 1 | 2,795 | too low degree -0.06, weight sum -0.06 |

quad-triangle-d4 (best known 6 nodes):

| cell | valid / 48 | gain in pass rate [95% CI] | best | not exact | crash | no code block | at the cap | tokens per attempt | symptom change |
|---|---|---|---|---|---|---|---|---|---|
| none (baseline) | 1 (0.021) | | 15 | 23 | 8 | 16 | 22 | 6,639 | |
| second_try | 2 (0.042) | +0.021 [+0.000, +0.062] | 15 | 34 | 6 | 6 | 6 | 10,149 | no code block -0.21, too low degree +0.13, weight sum +0.10 |
| diagnosis_only | 1 (0.021) | +0.000 [+0.000, +0.000] | 9 | 18 | 11 | 18 | 23 | 7,040 | weight sum -0.08, no code block +0.04, crash -0.02 |
| lessons | 1 (0.021) | +0.000 [-0.062, +0.062] | 9 | 8 | 13 | 26 | 28 | 6,978 | no code block +0.21, weight sum -0.17, too low degree -0.15 |
| skills | 0 | -0.021 [-0.062, +0.000] | - | 21 | 10 | 17 | 22 | 7,572 | too low degree -0.06, crash -0.04, no code block +0.02 |
| process_only | 0 | -0.021 [-0.062, +0.000] | - | 4 | 26 | 18 | 19 | 6,114 | crash +0.38, weight sum -0.23, too low degree -0.17 |
| unrelated (the empty notebook sentence) | 0 | -0.021 [-0.062, +0.000] | - | 9 | 9 | 30 | 33 | 7,431 | no code block +0.29, weight sum -0.19, too low degree -0.10 |

What the run says:

1. The noise floor first. At 3 seeds x 16 the standard error of a paired difference of pass rates on the square is about
   0.09, and the percentile bootstrap over three seed blocks gives intervals such as [+0.000, +0.000]. The empty-notebook
   row is the prompt-perturbation floor: one sentence added to the plain prompt gave +0.104 on the square and raised the
   triangle's no-code rate from 0.33 to 0.625 (three standard errors, 33 of 48 attempts at the token cap). Every pass-rate
   gain on the square, from -0.146 to +0.125, lies within what that sentence did; at three seeds no carrier is
   distinguishable from the floor by its pass rate.
2. The second try is the only row that moved the optimal count. Radon's 7-point rule appeared three times in 48 (seeds 12,
   1009 and 1013; two of the three are the rule with a^2 = 14/15, b^2 = 1/3, c^2 = 3/5 and weights 8/7, 20/63, 5/9), where
   the plain prompt gave it 0 times in 48 here and 0 in 64 and 0 in 256 on 2026-11-06 at the same settings; the
   empty-notebook row gave it once (seed 1006); the diagnosis, the lessons, the skills and the redacted lessons never. All
   four Radon rules of the session lie in the 288 attempts whose prompt differed from the plain one, against 0 in the 368
   plain attempts (p about 0.04 if attempts are exchangeable), and three of the four in the second-try cell (p about 0.015
   among the six variant cells): suggestive, not established. The mechanism is visible in the failures: the plain prompt's
   7-node attempts fail the certifier (seed 1 in both cells: score 7, not exact on x*y, and the revision did not save that
   one), the dry run returns exactly that verdict for the revision, and the weight-sum symptom fell from 0.17 to 0.05.
3. The diagnosis alone did nothing: 34 of 48 on the square, the baseline's 34 of 48, with no Radon rule; 1 of 48 on the
   triangle with symptom changes within 0.08. By the reading set down before the run, the square's barrier is not a
   missing diagnosis: told that its weights do not sum to 4 and that its rules fall short of the degree, the model wrote
   the same share of wrong rules. The lessons that name those failures removed them (weight sum 0.17 to 0.00, too low
   degree 0.10 to 0.02) and the pass rate did not move (0.729): the failures migrated to self-test crashes (9 of 48:
   assertion and value errors, a forbidden import) and missing code blocks (3). The symptom-change column shows what the
   pass rate hides: the lesson did what it said, and the attempt then failed for a new reason.
4. The redacted lessons hurt: process-only -0.146 on the square with 19 crashes in 48 (self-tests with their numbers
   redacted become assertions that fail), and sandbox crashes +0.38 on the triangle. The lessons' format without their
   content costs about 0.15 of pass rate, which the content pays back to parity with the baseline (0.729 against 0.562).
   The week-4 finding that the interval's lessons hurt the square (0.81 to 0.39) came with the gate's one lesson and its
   check; the lessons block without the check is level with the baseline here, and the harm sits in the format.
5. On the triangle the pass rate measures nothing (0.02); the symptoms do. The second try turned non-answers into wrong
   rules (no code block 0.33 to 0.125, at the cap 22 to 6, not exact 23 to 34, too low degree +0.13, weight sum +0.10) at
   10,149 tokens per attempt against 6,639: the certifier's verdict fixes the format, not the mathematics the target
   needs. Any prompt with the notebook framing lengthened the outputs (no code block 0.54 with the lessons, 0.625 with the
   empty sentence; 28 and 33 of 48 at the cap), which is the week-4 observation on the lesson groups again, now attributed
   to the framing rather than to the lessons' content. The two valid 9-node rules (diagnosis-only seed block 1, lessons
   seed block 2) against the baseline's 15-node rule are single attempts and say nothing about rank.
6. Mining without GPT (cell 17): three detectors kept on quad-triangle-d3's 64-attempt gate group (no code block 25, too low
   degree 14, weight sum 10, of 51 failures), none on the interval (9 failures; no built-in fires twice on the held-out
   half with every firing a failure) and none on the degree-2 triangle (no failures). For the square and the degree-4
   triangle the command took the latest base group, which was the matrix's last second-try block (16 attempts, one tool
   call), not the loop's 64-attempt gate group: the first defect below. The report is `detectors/mining-20261107T213824Z.md`
   on Drive.

### Mining on the ledger (cell 17, built-in detectors only)

| target | group mined | attempts | failed | kept (fires on the group) | rejected |
|---|---|---|---|---|---|
| quad-d7 | the loop's gate group | 64 | 9 | none | 5 |
| quad-square-d5 | the matrix's second-try block, seed 2 | 16 | 3 | none | 5 |
| quad-triangle-d2 | the loop's gate group | 64 | 0 | none | 5 |
| quad-triangle-d3 | the loop's gate group | 64 | 51 | no_code_block (25), too_low_degree (14), weights_do_not_sum_to_volume (10) | 2 |
| quad-triangle-d4 | the matrix's second-try block, seed 2 | 16 | 16 | too_low_degree (7), weights_do_not_sum_to_volume (4), no_code_block (3) | 2 |

The librarian query the diagnosis names on the degree-3 triangle is the target's title followed by "the output has no code
block", "the rule is exact only to a lower degree than required" and "the weights do not sum to the domain's volume"; the
diagnosis-only text lists the three with their counts and no remedy.

### Defects and follow-ups the run exposed

1. `base_groups` counts the second-try groups (one tool call) as base groups, and "latest" takes the newest: exclude groups
   with tool calls and mine the largest base group by default (`MINE_ALL_GROUPS = True` mines every group meanwhile).
2. Read the OpenAI secret once in cell 2 into the environment and let the later cells fall back to it: the lookup in cell 17
   returned nothing after the long matrix cell and the mining ran without GPT's proposals.
3. The unrelated control with no other family in the archive runs as the empty-notebook sentence: name the row so (a framing
   control) and keep it, since it measured the framing; a real unrelated row needs a loop run on another family.
4. The second-try summary should say how many attempts revised their file after the dry run and how many revisions passed;
   the ledger holds `revised_after_dry_run`, the printed summary does not.
5. Five seeds next time, as the default says, and the second try at 64 attempts on the square to settle the Radon count.
6. The recorded fixtures of cell 12 wait on Drive for a copy into the repository (`python tests/make_fixtures.py`, pytest,
   commit; grep for keys first).

Items 1 to 5 are fixed in the commit that follows this record: a group with a certifier dry run is not a base group and the
largest base group is mined by default; cell 2 reads the secret once, says why a read failed, and the later cells reuse the
copy; the empty unrelated row is reported with the setting `framing`; the second-try summary counts the revisions and the
revisions that passed; `MATRIX_SEEDS` defaults to 5. Item 6 waits for the copy from Drive. After that commit the mining cell
moved ahead of the matrix cell (cell 16 mines, cell 17 runs the matrix), so that one GPU session mines with GPT first and the
diagnosis row uses the detectors the program kept; the cell numbers in this record are the notebook's at the time of the run.
