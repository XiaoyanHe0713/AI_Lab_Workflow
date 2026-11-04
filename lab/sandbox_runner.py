"""The sandbox child process. Standalone: no lab imports, started as `python -I sandbox_runner.py spec.json`.

It applies resource limits, installs an import guard (allowlist of modules, denylist of submodule
prefixes such as numpy.polynomial and scipy.integrate), disables sockets, executes the candidate
file in a fresh namespace, calls its entry point, and writes a JSON result with the import log,
captured output, CPU time and any error. The parent (lab/judge.py) enforces the wall-clock
timeout, polls memory and kills the process tree.

With `instrument` in the spec ({"module": path of lab/certifiers/counted.py, "wrap_args": [...]}) the
listed arguments are wrapped as Counted scalars before the call and the result is replaced by
{"counts": ..., "outputs": ...}: the operation count is read off the outputs, never trusted.
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
import traceback

ALLOWED_TOP_LEVEL = {
    "math", "cmath", "itertools", "functools", "operator", "fractions", "decimal", "random", "statistics",
    "collections", "dataclasses", "typing", "numbers", "abc", "enum", "copy", "heapq", "bisect", "array",
    "string", "re", "json", "time", "warnings", "contextlib", "types", "builtins", "__future__",
    "numpy", "scipy",
}
DENIED_PREFIXES = (
    "numpy.polynomial",          # leggauss and friends: library quadrature nodes
    "scipy.integrate", "scipy.optimize", "scipy.special", "scipy.interpolate",
    "sympy", "mpmath", "quadpy", "cvxpy", "scs", "clarabel", "mosek", "PEPit",
    "socket", "http", "urllib", "requests", "ssl", "ftplib", "smtplib", "asyncio", "xmlrpc", "httpx", "aiohttp",
    "os", "sys", "subprocess", "ctypes", "importlib", "multiprocessing", "threading", "signal", "shutil",
    "pathlib", "pickle", "marshal", "inspect", "gc", "code", "codeop", "runpy", "pkgutil", "site", "sysconfig",
)

IMPORT_LOG: list[dict] = []


class _Guard:
    """A meta-path finder that refuses anything outside the allowlist and logs every attempt."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "resource":  # the runner's own accounting import; harmless to candidates
            return None
        top = fullname.split(".")[0]
        denied = any(fullname == p or fullname.startswith(p + ".") for p in DENIED_PREFIXES)
        allowed = (top in ALLOWED_TOP_LEVEL) and not denied
        IMPORT_LOG.append({"module": fullname, "allowed": allowed})
        if allowed:
            return None  # defer to the normal finders
        raise ImportError(f"import of {fullname!r} is forbidden inside the judge's sandbox")


def _limit_resources(memory_mb: int, cpu_s: int) -> dict:
    applied = {}
    try:
        import resource  # POSIX only

        if memory_mb:
            lim = int(memory_mb) * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
            applied["RLIMIT_AS_MB"] = memory_mb
        if cpu_s:
            resource.setrlimit(resource.RLIMIT_CPU, (int(cpu_s), int(cpu_s) + 5))
            applied["RLIMIT_CPU_S"] = cpu_s
        fsize = 64 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_FSIZE, (fsize, fsize))
        applied["RLIMIT_FSIZE_MB"] = 64
        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
            applied["RLIMIT_NPROC"] = 0
        except Exception:
            pass
    except ImportError:
        applied["note"] = "no resource module (Windows): the parent polls memory and kills"
    return applied


def _disable_network() -> None:
    try:
        import socket

        def _blocked(*a, **k):
            raise OSError("network access is disabled inside the judge's sandbox")

        socket.socket = _blocked  # type: ignore[assignment]
        socket.create_connection = _blocked  # type: ignore[assignment]
        socket.getaddrinfo = _blocked  # type: ignore[assignment]
    except Exception:
        pass


def _preimport_and_strip() -> None:
    """Import the allowed scientific packages first, then remove the denied submodules they loaded."""
    import numpy  # noqa: F401

    try:
        import scipy  # noqa: F401
        import scipy.linalg  # noqa: F401
    except Exception:
        pass
    for name in list(sys.modules):
        if any(name == p or name.startswith(p + ".") for p in DENIED_PREFIXES if p.startswith(("numpy.", "scipy."))):
            del sys.modules[name]
    try:
        del numpy.polynomial
    except Exception:
        pass
    try:
        import scipy as _sp

        for attr in ("integrate", "optimize", "special", "interpolate"):
            if hasattr(_sp, attr):
                delattr(_sp, attr)
    except Exception:
        pass


def _to_jsonable(obj):
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    if hasattr(obj, "item"):
        return obj.item()
    return repr(obj)


def main(spec_path: str) -> int:
    with open(spec_path, "r", encoding="utf-8") as f:
        spec = json.load(f)
    out = {"ok": False, "result": None, "error": None, "stdout": "", "stderr": "", "import_log": IMPORT_LOG,
           "cpu_seconds": 0.0, "wall_seconds": 0.0, "limits": {}}
    out["limits"] = _limit_resources(spec.get("memory_mb", 0), spec.get("cpu_s", 0))
    _disable_network()
    _preimport_and_strip()
    sys.meta_path.insert(0, _Guard())
    os.environ.clear()
    os.environ["PYTHONHASHSEED"] = "0"

    with open(spec["candidate_path"], "r", encoding="utf-8") as f:
        source = f.read()
    ns = {"__name__": "candidate", "__builtins__": __builtins__}
    buf_out, buf_err = io.StringIO(), io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = buf_out, buf_err
    t_wall = time.perf_counter()
    t_cpu = time.process_time()
    try:
        code = compile(source, "candidate.py", "exec")
        exec(code, ns)
        fn = ns.get(spec["entry_point"])
        if fn is None or not callable(fn):
            raise RuntimeError(f"candidate defines no callable {spec['entry_point']!r}")
        args = list(spec.get("args", []))
        instrument = spec.get("instrument")
        counted = None
        if instrument:
            counted = {"__name__": "counted"}
            with open(instrument["module"], "r", encoding="utf-8") as f:
                exec(compile(f.read(), "counted.py", "exec"), counted)
            wrap = set(instrument.get("wrap_args", range(len(args))))
            args = [counted["wrap_inputs"](a) if i in wrap else a for i, a in enumerate(args)]
        result = fn(*args, **spec.get("kwargs", {}))
        if counted is not None:
            result = {"counts": counted["count_ops"](result), "outputs": counted["values"](result)}
        out["result"] = _to_jsonable(result)
        out["ok"] = True
    except BaseException as e:  # noqa: BLE001  (MemoryError and SystemExit must be reported too)
        out["error"] = {"type": type(e).__name__, "message": str(e)[:2000],
                        "traceback": traceback.format_exc()[-4000:]}
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        out["cpu_seconds"] = time.process_time() - t_cpu
        out["wall_seconds"] = time.perf_counter() - t_wall
        out["stdout"] = buf_out.getvalue()[-4000:]
        out["stderr"] = buf_err.getvalue()[-4000:]
    try:
        import resource

        out["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        pass
    with open(spec["output_path"], "w", encoding="utf-8") as f:
        json.dump(out, f)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
