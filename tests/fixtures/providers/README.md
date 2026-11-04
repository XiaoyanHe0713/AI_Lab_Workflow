# Recorded provider responses

One directory per provider name in `config/lab.yaml` (`local`, `openai`, `anthropic`). Offline mode replays
from here and never makes a live call; record mode writes here; a missing fixture fails the test.

* `<request key>.json`: one recorded call, written by `RecordingProvider`, holding the request (model,
  messages, tools, temperature, seed) and the response. The key is the sha256 of those request fields.
  GPT and Claude share this format, so a harness test is the same file for both providers.
* `<prompt hash>.seq.json`: a list of responses replayed in seed order, so a group of N attempts replays
  from one recording. The `local` sequence for quad-d7 is synthetic (hand-written in
  `tests/make_fixtures.py`, 16 responses: 7 valid rules, 4 of them optimal), not a model output.
  Regenerate it with `python tests/make_fixtures.py` whenever the quad-d7 prompt text changes.
