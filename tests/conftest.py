import os
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("LAB_REPO", str(ROOT))
# The pytest default is offline: no network, no key. Tests that need a live provider pass mode="online"
# explicitly against a local fake server, or carry the `online` marker and are skipped without the key.
os.environ.setdefault("LAB_MODE", "offline")
os.environ.setdefault("LAB_FRONTIER_MODE", "offline")

FIXTURES = ROOT / "tests" / "fixtures" / "providers"


def pytest_configure(config):
    config.addinivalue_line("markers", "online(key): makes a live API call; skipped unless the environment variable `key` is set")


def pytest_runtest_setup(item):
    marker = item.get_closest_marker("online")
    if marker is not None:
        key = marker.kwargs.get("key") or (marker.args[0] if marker.args else None)
        if key and not os.environ.get(key):
            pytest.skip(f"online test: ${key} is not set")


@pytest.fixture
def ledger_path(tmp_path):
    return tmp_path / "ledger.sqlite"


@pytest.fixture(scope="session")
def repo_root():
    return ROOT


@dataclass
class Frontier:
    """A frontier provider for the harness tests, plus the ledger its rows go to."""
    name: str
    mode: str
    provider: object
    ledger: object
    ledger_path: Path
    cfg: dict

    @property
    def recorded(self) -> bool:
        return self.mode == "offline"


def _has_fixtures(name: str) -> bool:
    return any((FIXTURES / name).glob("*.json"))


@pytest.fixture(params=["openai", "anthropic"])
def frontier(request, tmp_path):
    """The harness tests run once per frontier provider. Offline (the default) replays the recorded fixtures under
    tests/fixtures/providers/<name>/ and skips a provider that has none yet. With LAB_FRONTIER_MODE=record and the
    provider's key set, the same tests make live calls, write the fixtures into the repository, and copy their rows
    (with the API dollars) into the repository's own ledger so the month's frontier spend is tracked."""
    from lab.config import lab_config
    from lab.ledger import Ledger
    from lab.roles import make_provider

    name = request.param
    mode = os.environ.get("LAB_FRONTIER_MODE", "offline")
    cfg = lab_config()
    pcfg = cfg["providers"][name]
    if mode == "offline" and not _has_fixtures(name):
        pytest.skip(f"no recorded {name} fixtures under {FIXTURES / name}: record them with LAB_FRONTIER_MODE=record and ${pcfg['api_key_env']}")
    if mode != "offline" and not os.environ.get(pcfg["api_key_env"]):
        pytest.skip(f"{mode} mode needs ${pcfg['api_key_env']}")
    ledger_path = tmp_path / "ledger.sqlite"
    ledger = Ledger(ledger_path)
    real = Ledger(cfg["storage"]["ledger"]) if mode == "record" else None   # the budget guard reads the month's spend from here
    provider = make_provider(name, cfg, mode=mode, fixtures_root=ROOT, ledger=real if real is not None else ledger)
    fr = Frontier(name=name, mode=mode, provider=provider, ledger=ledger, ledger_path=ledger_path, cfg=cfg)
    yield fr
    if real is not None:
        for row in ledger.attempts():
            if row.provider == name and row.mode == "record":
                real.add_attempt(row)
        real.close()
    ledger.close()


def demo_group(tmp_path, ledger, attempts=16, **kw):
    """The week-2 demo group on quad-d7 from the synthetic local fixtures: the experience the harness tests use."""
    from lab.config import lab_config
    from lab.demo import run_group

    cfg = lab_config(env={"LAB_ROOT": str(tmp_path)})
    args = dict(target_id="quad-d7", attempts=attempts, seed=0, provider_name="local", role_name="builder", ledger=ledger,
                ledger_path=ledger.path, gpu="A100", precision=None, units_per_hour=None, temperature=None, max_tokens=None,
                timeout_s=30, cfg=cfg, fixtures_root=ROOT, quiet=True, cache_dir=tmp_path / "cache", mode="offline", concurrency=4)
    args.update(kw)
    return run_group(**args)
