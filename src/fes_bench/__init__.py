"""Face Enhancement Service — Phase 3 benchmark harness (evaluation only).

Candidate adapters, the shared geometry contract, metrics and run bookkeeping for `fes-syn-v1`.
Nothing here is a production service; the SFace evaluator is test-only (P0 master C1).
"""

__version__ = "1.0.0"


def read_bytes_retry(path, tries: int = 6, wait_s: float = 2.0) -> bytes:
    """Read a file, retrying transient OSError (ENOMEM/EIO) seen on the Windows→Docker bind mount under
    VM memory pressure (P3-T4 log). Content is still verified by SHA-256 by every caller."""
    import time
    from pathlib import Path
    for i in range(tries):
        try:
            return Path(path).read_bytes()
        except OSError:
            if i == tries - 1:
                raise
            time.sleep(wait_s * (i + 1))
