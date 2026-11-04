"""Certifier base: the certificate record, determinism under a seed, a runtime cap, a version.

A certifier is a program with its own test suite. It returns a Certificate: pass or fail, a score,
its own name and version, the seed, the runtime, and details enough to re-run it. No model's
judgment ever enters a certificate.
"""
from __future__ import annotations

import multiprocessing as mp
import time
import traceback
from dataclasses import dataclass, field, asdict
from typing import Any, Callable

import numpy as np


class CertifierTimeout(RuntimeError):
    pass


@dataclass
class Certificate:
    certifier: str
    version: str
    target: str
    passed: bool
    score: float | None = None
    lower_is_better: bool = True
    optimal: bool | None = None          # passed and score equals the scoreboard's best known value
    details: dict = field(default_factory=dict)
    witness: dict | None = None
    seed: int = 0
    runtime_s: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict:
        return _plain(asdict(self))


def _plain(o: Any) -> Any:
    """JSON-safe copy: numpy scalars and arrays become Python numbers and lists."""
    if isinstance(o, dict):
        return {str(k): _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


class Certifier:
    """Subclasses set name, version, runtime_cap_s and implement run(candidate, rng, **kw)."""

    name: str = "certifier"
    version: str = "0.0.0"
    runtime_cap_s: float = 60.0

    def certify(self, candidate: Any, *, target: str = "", seed: int = 0, **kw) -> Certificate:
        rng = np.random.default_rng(seed)
        t0 = time.perf_counter()
        try:
            cert = self.run(candidate, rng, target=target, seed=seed, **kw)
        except CertifierTimeout as e:
            cert = Certificate(self.name, self.version, target, False, error=f"timeout: {e}")
        except Exception as e:  # a crashing certifier is a failed certificate, never a crash of the lab
            cert = Certificate(self.name, self.version, target, False,
                               error=f"{type(e).__name__}: {e}",
                               details={"traceback": traceback.format_exc()[-2000:]})
        cert.seed = seed
        cert.runtime_s = time.perf_counter() - t0
        cert.certifier = self.name
        cert.version = self.version
        cert.target = target
        return cert

    def run(self, candidate: Any, rng: np.random.Generator, *, target: str, seed: int, **kw) -> Certificate:
        raise NotImplementedError


# --- a runtime cap for heavy certifiers -------------------------------------------------------
def _worker(conn, func, kwargs):
    try:
        conn.send(("ok", func(**kwargs)))
    except Exception as e:
        conn.send(("err", f"{type(e).__name__}: {e}", traceback.format_exc()[-2000:]))
    finally:
        conn.close()


def run_with_cap(func: Callable[..., Any], kwargs: dict, cap_s: float) -> Any:
    """Run func(**kwargs) in a fresh process and kill it after cap_s seconds.

    func must be importable (module-level) and kwargs picklable; the result must be picklable.
    """
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_worker, args=(child, func, kwargs), daemon=True)
    p.start()
    child.close()
    t0 = time.perf_counter()
    msg = None
    try:
        while True:
            if parent.poll(0.05):
                try:
                    msg = parent.recv()
                except EOFError:
                    msg = ("err", "the worker process ended without a result", "")
                break
            if not p.is_alive():
                if parent.poll(0.5):
                    try:
                        msg = parent.recv()
                    except EOFError:
                        msg = ("err", f"the worker process died (exit code {p.exitcode})", "")
                else:
                    msg = ("err", f"the worker process died (exit code {p.exitcode})", "")
                break
            if time.perf_counter() - t0 > cap_s:
                p.terminate()
                p.join(5)
                if p.is_alive():
                    p.kill()
                raise CertifierTimeout(f"exceeded the runtime cap of {cap_s:g} s")
    finally:
        parent.close()
    p.join(5)
    if msg[0] == "ok":
        return msg[1]
    raise RuntimeError(f"{msg[1]}\n{msg[2]}")
