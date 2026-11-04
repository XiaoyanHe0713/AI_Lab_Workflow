# Recorded provider responses

One directory per provider name in `config/lab.yaml` (`local`, `openai`, `anthropic`). Offline mode replays
from here and never makes a live call; record mode writes here; a missing fixture fails the test.

* `<request key>.json`: one recorded call, written by `RecordingProvider`, holding the request (model,
  messages, tools, temperature, seed) and the response. The key is the sha256 of those request fields.
  GPT and Claude share this format, so a harness test is the same file for both providers. The harness
  fixtures under `openai/` and `anthropic/` are recorded by running the harness tests with
  `LAB_FRONTIER_MODE=record` and the provider's key (see README, Providers).
* `<prompt hash>.seq.json`: a list of responses replayed in seed order, so a group of N attempts replays
  from one recording. The `local` sequences are synthetic (hand-written in `tests/make_fixtures.py`, 16
  responses: 7 valid rules, 4 of them optimal), not model outputs: one for the quad-d7 prompt and one per
  ablation notebook arm per recorded provider. Regenerate them with `python tests/make_fixtures.py` whenever
  the quad-d7 prompt text or the recorded lessons change.
