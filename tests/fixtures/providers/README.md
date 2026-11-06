# Recorded provider responses

One directory per provider name in `config/lab.yaml` (`local`, `openai`, `anthropic`). Offline mode replays
from here and never makes a live call; record mode writes here; a missing fixture fails the test.

* `<request key>.json`: one recorded call, written by `RecordingProvider`, holding the request (model,
  messages, tools, temperature, seed) and the response. The key is the sha256 of those request fields.
  GPT and Claude share this format, so a harness test is the same file for both providers. The harness
  fixtures under `openai/` and `anthropic/` are recorded by running the harness tests with
  `LAB_FRONTIER_MODE=record` and the provider's key (see README, Providers).
* `<prompt hash>.seq.json`: a list of responses replayed in seed order, so a group of N attempts replays
  from one recording. The `local` sequences are synthetic (hand-written in `tests/make_fixtures.py` and
  `tests/fixtures_week4.py`), not model outputs: the week-2 quad-d7 sequence (16 responses: 7 valid rules, 4 of
  them optimal), one per ablation notebook arm per recorded provider, and the week-4 sequences for the loop
  (builder rungs on the square and the triangle; theorist proposals and breaker replies with a `rounds` list,
  one reply per tool round, keyed by the opening prompt; builders derived from proposals) and the matrix cells,
  the diagnosis-only control's prompts included (week 5). Every single-round builder item also carries
  `rounds: [text, text]`, so a second try (one certifier dry run and a revision) replays as the model keeping its file.
  Regenerate them with `python tests/make_fixtures.py` whenever a prompt text, the recorded lessons or the
  synthetic responses change; delete stale ones first (git clean on the local directory) so no orphan stays.
