"""IPC boundaries for the memory observer; no allocation or identity state."""

import math
import sys
import time

import psutil


_IDENTITY_PSUTIL_VERSION = "7.1.3"


def _check_identity_support():
    # _ident is private. Package installs, the lockfile and native CI pin this
    # version; refuse an unverified override rather than reporting guessed RAM.
    if psutil.__version__ != _IDENTITY_PSUTIL_VERSION:
        raise RuntimeError(f"Memory identity requires psutil {_IDENTITY_PSUTIL_VERSION}; got {psutil.__version__}")
    if sys.platform not in ("linux", "darwin", "win32"):
        raise RuntimeError(f"Memory identity is unsupported on {sys.platform}")


def _check_identity_value(identity):
    if (not isinstance(identity, (tuple, list)) or len(identity) != 2
            or type(identity[0]) is not int or identity[0] < 0
            or type(identity[1]) not in (int, float)
            or not math.isfinite(identity[1]) or identity[1] < 0):
        raise ValueError("Missing or malformed stable process identity")


def validate_process_identity(process):
    """Require a verified psutil identity; return only its owning Process.

    In particular, psutil can leave _ident=(pid, None) on access failure.
    That must not enter a runtime set or silently become a zero contribution.
    """
    _check_identity_support()
    identity = getattr(process, "_ident", None)
    if not isinstance(identity, tuple):
        raise ValueError("Missing or unsupported psutil Process._ident")
    _check_identity_value(identity)
    if identity[0] != process.pid:
        raise ValueError("Stable process identity does not match its PID")
    return process


def export_process_identity(process):
    """One-way IPC adapter for psutil 7.1.3's equality/hash identity.

    Linux/Darwin values are psutil's unadjusted native start values; Windows
    retains psutil's native creation value. Never reconstruct these ourselves.
    Receivers validate/discard the token and keep Process objects at runtime.
    """
    validate_process_identity(process)
    return {"psutil_version": _IDENTITY_PSUTIL_VERSION, "platform": sys.platform,
            "identity": list(process._ident)}


def process_is_running(process):
    """Use psutil equality for liveness, rejecting an unreadable fresh identity.

    Process.is_running() treats a fresh (pid, None) as PID reuse and caches that
    conclusion. An observation must report unknown coverage instead of dropping
    that process. This performs the same fresh-wrapper comparison with validation.
    """
    validate_process_identity(process)
    try:
        current = validate_process_identity(psutil.Process(process.pid))
    except psutil.NoSuchProcess:
        return False
    return process == current


def _read_process_identity(token):
    _check_identity_support()
    if (not isinstance(token, dict)
            or set(token) != {"psutil_version", "platform", "identity"}
            or token["psutil_version"] != _IDENTITY_PSUTIL_VERSION
            or token["platform"] != sys.platform):
        raise ValueError("Missing or unverified process identity token")
    _check_identity_value(token["identity"])
    return tuple(token["identity"])


def matches_process_identity(process, token):
    """Validate an IPC owner against its authoritative Process, without caching."""
    validate_process_identity(process)
    identity = _read_process_identity(token)
    if identity[0] != process.pid:
        raise ValueError("Process identity token does not match its PID")
    return process._ident == identity


def process_from_identity(token):
    """Consume a startup token once; only the validated Process escapes."""
    identity = _read_process_identity(token)
    process = psutil.Process(identity[0])
    if not matches_process_identity(process, token):
        raise psutil.NoSuchProcess(process.pid, msg="Backend PID was reused")
    return process


def observation_file_io(operation):
    """Retry transient Windows sharing violations for at most half a second.

    Used by both observation manifests and collector control/result files.
    Persistent denial and other errors propagate to the owning caller.
    """
    deadline = None
    while True:
        try:
            return operation()
        except PermissionError:
            if sys.platform != "win32":
                raise
            if deadline is None:
                deadline = time.monotonic() + 0.5
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise
            time.sleep(min(0.01, remaining))
