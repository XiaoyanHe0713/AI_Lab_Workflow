"""The judge: runs a candidate in a subprocess sandbox and applies the target's certifier.

Sandbox: a fresh `python -I` process (no user site, no PYTHONPATH, no env), a scrubbed environment,
a temporary working directory, a wall-clock timeout, a memory limit (RLIMIT_AS on POSIX; the parent
polls RSS everywhere and kills the process tree), and no network (sockets disabled, network modules
unimportable). Colab has no Docker, so this is the isolation the proposal asks for: a subprocess with
limits. The same attempt scores the same twice: the certifier is seeded and the candidate sees a
fixed PYTHONHASHSEED and single-threaded BLAS.

Rules during measurement: library quadrature and optimisation solvers are forbidden in candidate
code (numpy.polynomial, scipy.integrate, scipy.optimize, scipy.special, sympy, mpmath, cvxpy, ...).
NumPy arithmetic and numpy.linalg are allowed. A forbidden import, found statically or at run time,
fails the attempt.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .certifiers.base import Certificate
from .sandbox_runner import ALLOWED_TOP_LEVEL, DENIED_PREFIXES

RUNNER = Path(__file__).resolve().parent / "sandbox_runner.py"

FORBIDDEN_CALLS = {"__import__", "exec", "eval", "compile", "open", "globals", "vars", "breakpoint", "input"}
FORBIDDEN_ATTRS = {"polynomial", "integrate", "optimize", "special", "interpolate", "leggauss", "roots_legendre",
                   "fixed_quad", "quad", "quadrature", "gauss_legendre", "roots_jacobi"}


@dataclass
class SandboxResult:
    ok: bool
    result: object = None
    error: dict | None = None
    stdout: str = ""
    stderr: str = ""
    import_log: list = field(default_factory=list)
    cpu_seconds: float = 0.0
    wall_seconds: float = 0.0
    peak_rss_mb: float | None = None
    timed_out: bool = False
    killed: str | None = None
    returncode: int | None = None
    static_violations: list = field(default_factory=list)
    limits: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["import_log"] = [m for m in self.import_log if not m["allowed"]][:50] + \
                          [{"allowed_count": sum(1 for m in self.import_log if m["allowed"])}]
        return d


def static_check(source: str) -> list[str]:
    """Violations of the no-library-solvers rule found by reading the code, before it runs."""
    violations: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [f"syntax error: {e}"]
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            names = [base] + [f"{base}.{a.name}" for a in node.names]
        for name in names:
            top = name.split(".")[0]
            if top not in ALLOWED_TOP_LEVEL or any(name == p or name.startswith(p + ".") for p in DENIED_PREFIXES):
                violations.append(f"forbidden import {name!r} at line {node.lineno}")
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id in FORBIDDEN_CALLS:
                violations.append(f"forbidden call {fn.id}() at line {node.lineno}")
        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRS:
            violations.append(f"forbidden attribute .{node.attr} at line {node.lineno}")
    return violations


def _scrubbed_env() -> dict:
    env = {
        "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8",
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
    }
    if os.name == "nt":
        for k in ("SYSTEMROOT", "WINDIR", "TEMP", "TMP", "PATHEXT", "COMSPEC"):
            if k in os.environ:
                env[k] = os.environ[k]
        # the interpreter's own directory only, so DLLs resolve; nothing else from PATH
        env["PATH"] = str(Path(sys.executable).parent)
    else:
        env["PATH"] = "/usr/bin:/bin"
        env["HOME"] = tempfile.gettempdir()
    return env


class Judge:
    def __init__(self, *, timeout_s: float = 120.0, memory_mb: int = 2048, python: str | None = None,
                 cache_dir: str | Path | None = None, sdp_cap_s: float = 60.0, sdp_solver: str = "SCS",
                 network_isolation: str | None = None):
        self.timeout_s = float(timeout_s)
        self.memory_mb = int(memory_mb)
        self.python = python or sys.executable
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.sdp_cap_s = float(sdp_cap_s)
        self.sdp_solver = sdp_solver
        # "unshare" uses `unshare -n` (a network namespace) when available and permitted; otherwise the
        # python-level isolation of the runner applies alone. Recorded in every certificate.
        self.network_isolation = network_isolation or self._detect_unshare()

    @staticmethod
    def _detect_unshare() -> str:
        if os.name != "nt":
            try:
                r = subprocess.run(["unshare", "-n", "true"], capture_output=True, timeout=10)
                if r.returncode == 0:
                    return "unshare"
            except Exception:
                pass
        return "python-level"

    def run_sandbox(self, source: str, entry_point: str, args: list | None = None, kwargs: dict | None = None,
                    timeout_s: float | None = None) -> SandboxResult:
        violations = static_check(source)
        if violations:
            return SandboxResult(ok=False, error={"type": "ForbiddenImport", "message": "; ".join(violations)},
                                 static_violations=violations)
        timeout = timeout_s or self.timeout_s
        with tempfile.TemporaryDirectory(prefix="judge_") as tmp:
            tmp_path = Path(tmp)
            cand = tmp_path / "candidate.py"
            cand.write_text(source, encoding="utf-8")
            spec = {"candidate_path": str(cand), "entry_point": entry_point, "args": args or [],
                    "kwargs": kwargs or {}, "output_path": str(tmp_path / "out.json"),
                    "memory_mb": self.memory_mb, "cpu_s": int(timeout) + 1}
            spec_path = tmp_path / "spec.json"
            spec_path.write_text(json.dumps(spec), encoding="utf-8")
            cmd = [self.python, "-I", "-u", str(RUNNER), str(spec_path)]
            if self.network_isolation == "unshare":
                cmd = ["unshare", "-n"] + cmd
            t0 = time.perf_counter()
            proc = subprocess.Popen(cmd, cwd=tmp, env=_scrubbed_env(), stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, text=True)
            timed_out, killed, peak = self._wait(proc, timeout)
            wall = time.perf_counter() - t0
            out_path = tmp_path / "out.json"
            if out_path.exists() and not timed_out and killed is None:
                data = json.loads(out_path.read_text(encoding="utf-8"))
                res = SandboxResult(ok=bool(data.get("ok")), result=data.get("result"), error=data.get("error"),
                                    stdout=data.get("stdout", ""), stderr=data.get("stderr", ""),
                                    import_log=data.get("import_log", []), cpu_seconds=float(data.get("cpu_seconds", 0.0)),
                                    wall_seconds=wall, peak_rss_mb=data.get("peak_rss_mb", peak),
                                    returncode=proc.returncode, limits=data.get("limits", {}))
                if res.ok:
                    bad = [m["module"] for m in res.import_log if not m["allowed"]]
                    if bad:
                        res.ok = False
                        res.error = {"type": "ForbiddenImport", "message": f"forbidden import at run time: {bad}"}
                return res
            stderr_tail = ""
            try:
                stderr_tail = (proc.stderr.read() or "")[-2000:] if proc.stderr else ""
            except Exception:
                pass
            msg = "timeout" if timed_out else (killed or f"sandbox exited with code {proc.returncode}")
            return SandboxResult(ok=False, error={"type": "SandboxFailure", "message": msg, "stderr": stderr_tail},
                                 wall_seconds=wall, timed_out=timed_out, killed=killed, returncode=proc.returncode,
                                 peak_rss_mb=peak)

    def _wait(self, proc: subprocess.Popen, timeout: float) -> tuple[bool, str | None, float | None]:
        """Poll until exit, the timeout, or the memory limit; kill the whole tree on either."""
        try:
            import psutil

            ps = psutil.Process(proc.pid)
        except Exception:
            ps = None
        t0 = time.perf_counter()
        peak = 0.0
        limit_bytes = self.memory_mb * 1024 * 1024
        while True:
            if proc.poll() is not None:
                return False, None, (peak if peak else None)
            if ps is not None:
                try:
                    rss = ps.memory_info().rss
                    for ch in ps.children(recursive=True):
                        rss += ch.memory_info().rss
                    peak = max(peak, rss / 1024 / 1024)
                    if rss > limit_bytes:
                        self._kill_tree(proc, ps)
                        return False, f"memory limit of {self.memory_mb} MB exceeded", peak
                except Exception:
                    pass
            if time.perf_counter() - t0 > timeout:
                self._kill_tree(proc, ps)
                return True, None, (peak if peak else None)
            time.sleep(0.02)

    @staticmethod
    def _kill_tree(proc: subprocess.Popen, ps) -> None:
        try:
            if ps is not None:
                for ch in ps.children(recursive=True):
                    try:
                        ch.kill()
                    except Exception:
                        pass
            proc.kill()
        except Exception:
            pass
        try:
            proc.wait(timeout=10)
        except Exception:
            pass

    def judge(self, source: str | None, target, seed: int = 0) -> tuple[Certificate, SandboxResult | None]:
        """Run the candidate for `target` and certify it. A missing or crashing candidate fails."""
        cert_cls_name = target.certifier_name
        if source is None:
            cert = Certificate(cert_cls_name, "", target.id, False, error="no code block in the output",
                               seed=seed)
            return cert, None
        certifier = target.make_certifier(cache_dir=self.cache_dir, sdp_cap_s=self.sdp_cap_s,
                                          sdp_solver=self.sdp_solver)
        sb = self.run_sandbox(source, target.entry_point, args=target.entry_args, timeout_s=self.timeout_s)
        if not sb.ok:
            cert = Certificate(certifier.name, certifier.version, target.id, False, seed=seed,
                               error=f"{sb.error['type']}: {sb.error['message']}" if sb.error else "sandbox failed",
                               details={"sandbox": sb.to_dict(), "network_isolation": self.network_isolation})
            return cert, sb
        cert = certifier.certify(sb.result, target=target.id, seed=seed)
        cert.details["sandbox"] = sb.to_dict()
        cert.details["network_isolation"] = self.network_isolation
        return cert, sb
