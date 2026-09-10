"""Single-writer guard for GPU work.

NEXT_STEPS_ROUND_6_SAMEDAY_FINAL.md 0.5: "Never run two GPU arms concurrently." That rule was
violated once in this round by a shell-quoting accident -- a compound command was backgrounded as a
whole, so its `&&`-chained launch fired in addition to an explicit one, and two identical KL jobs
shared the card. Timings went from 32s to 1039s per checkpoint and neither run finished.

This makes the rule enforceable rather than a matter of care: a second GPU entry point aborts
immediately with a clear message instead of silently halving throughput.

Purely operational -- it changes no computation, no seed and no result.
"""
import ctypes
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
LOCK = REPO_ROOT / ".gpu.lock"

_SYNCHRONIZE = 0x00100000
_WAIT_TIMEOUT = 0x00000102          # still running
_ERROR_ACCESS_DENIED = 5


def _alive(pid: int) -> bool:
    """True if `pid` is a running process.

    Windows: OpenProcess(SYNCHRONIZE) + WaitForSingleObject(0). WAIT_TIMEOUT means the process
    object is not signalled, i.e. still running. This is used rather than GetExitCodeProcess
    (which needs query access SYNCHRONIZE does not grant, and whose STILL_ACTIVE sentinel is
    ambiguous with a genuine exit code of 259) and rather than os.kill (which on Windows does not
    treat signal 0 as a liveness probe and can terminate the target).
    """
    if pid <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True                      # exists, just not ours
    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(_SYNCHRONIZE, False, pid)
    if not h:
        return k32.GetLastError() == _ERROR_ACCESS_DENIED     # exists, not accessible
    try:
        return k32.WaitForSingleObject(h, 0) == _WAIT_TIMEOUT
    finally:
        k32.CloseHandle(h)


@contextmanager
def gpu_lock(tag: str, allow_override: bool = False):
    if LOCK.exists():
        try:
            info = json.loads(LOCK.read_text())
        except Exception:
            info = {}
        pid = int(info.get("pid", -1))
        if pid > 0 and pid != os.getpid() and _alive(pid) and not allow_override:
            raise RuntimeError(
                f"GPU already in use by pid {pid} ({info.get('tag')}, started "
                f"{info.get('started')}). Refusing to run '{tag}' concurrently -- "
                f"0.5 forbids two GPU arms at once. Wait for it, or remove {LOCK} if that "
                f"process is genuinely dead.")
        LOCK.unlink(missing_ok=True)      # stale lock from a killed process
    LOCK.write_text(json.dumps({"pid": os.getpid(), "tag": tag,
                                "started": time.strftime("%Y-%m-%d %H:%M:%S")}))
    try:
        yield
    finally:
        try:
            if LOCK.exists() and json.loads(LOCK.read_text()).get("pid") == os.getpid():
                LOCK.unlink()
        except Exception:
            pass
