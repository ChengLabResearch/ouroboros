"""File IPC for the memory observer; no allocation or reporting state."""

import sys
import time


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

