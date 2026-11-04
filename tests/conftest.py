import os
import sys
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
