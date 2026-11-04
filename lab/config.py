"""Configuration loading with ${VAR:-default} expansion.

The repository holds three YAML files under config/: lab.yaml (storage, endpoint, providers,
roles, judge limits), model.yaml (frozen base and quantized build) and prices.yaml (cost meter).
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-((?:[^{}]|\{[^{}]*\})*))?\}")


def repo_root() -> Path:
    """The repository root: the directory holding config/ and lab/."""
    env = os.environ.get("LAB_REPO")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parent.parent


def expand(value: Any, env: dict[str, str] | None = None) -> Any:
    """Expand ${VAR} and ${VAR:-default} recursively in strings, lists and dicts."""
    env = os.environ if env is None else env
    if isinstance(value, str):
        prev = None
        out = value
        # iterate so that a default that itself contains ${...} is expanded too
        while prev != out:
            prev = out
            out = _VAR.sub(lambda m: env.get(m.group(1)) or (m.group(2) if m.group(2) is not None else ""), out)
        return out
    if isinstance(value, list):
        return [expand(v, env) for v in value]
    if isinstance(value, dict):
        return {k: expand(v, env) for k, v in value.items()}
    return value


def load_yaml(path: str | Path, env: dict[str, str] | None = None) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return expand(data, env)


def lab_config(path: str | Path | None = None, env: dict[str, str] | None = None) -> dict:
    return load_yaml(path or repo_root() / "config" / "lab.yaml", env)


def model_config(path: str | Path | None = None) -> dict:
    """config/model.yaml, or the pinned copy the serving notebook writes under $LAB_ROOT/config/ if present
    (Colab pulls the repository read-only, so the pin lands on Drive first and is committed by hand)."""
    if path is None:
        root = os.environ.get("LAB_ROOT")
        if root and (Path(root) / "config" / "model.yaml").exists():
            path = Path(root) / "config" / "model.yaml"
    return load_yaml(path or repo_root() / "config" / "model.yaml")


def prices_config(path: str | Path | None = None) -> dict:
    return load_yaml(path or repo_root() / "config" / "prices.yaml")


def write_yaml(path: str | Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
