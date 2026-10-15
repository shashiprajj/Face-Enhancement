"""Process-isolated model execution (a thread timeout cannot kill a hung model).

The model lives in ONE child process (multiprocessing 'spawn'). The parent sends an image, waits at most `timeout_s`
for the answer, and on timeout KILLS the child (it is restarted lazily on the next call). Failures surface as
exceptions that the pipeline maps to its fallbacks (fes_service.pipeline.fallback_code):
  TimeoutError                         → TIMEOUT          (concurrent.futures.TimeoutError is TimeoutError, py ≥ 3.11)
  MemoryError / child killed by signal → RESOURCE_FAILURE (e.g. kernel OOM kill)
  RuntimeError (model exception)       → MODEL_FAILURE
`factory` is "module:function" returning (fn(img, ctx) -> (out, meta), info dict); it runs inside the child only.
"""

from __future__ import annotations

import importlib
import multiprocessing as mp
import os
import threading
import time


def _resolve(spec: str):
    mod, fn = spec.split(":")
    return getattr(importlib.import_module(mod), fn)


def _serve(conn, factory: str) -> None:
    try:
        import cv2
        cv2.setNumThreads(1)
    except ImportError:
        pass
    try:
        fn, info = _resolve(factory)()
    except BaseException as e:  # noqa: BLE001
        conn.send(("load_error", f"{type(e).__name__}: {e}"[:500]))
        return
    conn.send(("ready", info))
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            return
        if msg[0] == "stop":
            return
        _, img, ctx = msg
        t0 = time.perf_counter()
        try:
            out, meta = fn(img, ctx)
            conn.send(("ok", out, {**meta, "infer_s": round(time.perf_counter() - t0, 3)}))
        except MemoryError as e:
            conn.send(("err", "MemoryError", str(e)[:300]))
        except BaseException as e:  # noqa: BLE001
            conn.send(("err", type(e).__name__, str(e)[:300]))


class WorkerDied(RuntimeError):
    pass


class IsolatedModel:
    """Callable(img, ctx) -> (out, meta) backed by one killable child process. Calls are serialised."""

    def __init__(self, factory: str, timeout_s: float = 120.0, load_timeout_s: float = 600.0):
        self.factory, self.timeout_s, self.load_timeout_s = factory, timeout_s, load_timeout_s
        self._ctx = mp.get_context("spawn")
        self._proc = None
        self._conn = None
        self._lock = threading.Lock()
        self.info: dict = {}
        self.restarts = 0
        self.last_error = ""

    # -------------------------------------------------------------- lifecycle
    def alive(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    def start(self) -> dict:
        self._kill()
        parent, child = self._ctx.Pipe()
        self._proc = self._ctx.Process(target=_serve, args=(child, self.factory), daemon=True)
        self._proc.start()
        child.close()
        self._conn = parent
        if not parent.poll(self.load_timeout_s):
            self._kill()
            raise TimeoutError(f"model load exceeded {self.load_timeout_s}s")
        try:
            kind, payload = parent.recv()
        except (EOFError, OSError):
            code = self._proc.exitcode
            self._kill()
            raise RuntimeError(f"model worker exited during load (exit {code})") from None
        if kind != "ready":
            self._kill()
            raise RuntimeError(f"model load failed: {payload}")
        self.info = payload
        return payload

    def _kill(self) -> None:
        if self._proc is not None:
            if self._proc.is_alive():
                self._proc.kill()
            self._proc.join(5)
        if self._conn is not None:
            self._conn.close()
        self._proc = self._conn = None

    def close(self) -> None:
        with self._lock:
            if self.alive():
                try:
                    self._conn.send(("stop",))
                    self._proc.join(5)
                except (OSError, BrokenPipeError):
                    pass
            self._kill()

    # -------------------------------------------------------------- call
    def __call__(self, img, ctx):
        with self._lock:
            if not self.alive():
                if self.info:                      # it ran before: this is a restart after a kill/crash
                    self.restarts += 1
                self.start()
            try:
                self._conn.send(("run", img, ctx))
                ready = self._conn.poll(self.timeout_s)
            except (OSError, BrokenPipeError, EOFError) as e:
                self._kill()
                raise WorkerDied(f"worker pipe failed: {e}") from e
            if not ready:
                self._kill()                      # the hung model is killed, not abandoned
                self.last_error = "timeout"
                raise TimeoutError(f"model exceeded {self.timeout_s}s; worker killed")
            try:
                msg = self._conn.recv()
            except (EOFError, OSError):
                code = self._proc.exitcode if self._proc is not None else None
                self._proc.join(5) if self._proc is not None else None
                code = self._proc.exitcode if self._proc is not None else code
                self._kill()
                self.last_error = f"worker died (exit {code})"
                if code is not None and code < 0:      # killed by a signal (SIGKILL = kernel OOM killer)
                    raise MemoryError(f"worker killed by signal {-code} (out of memory?)")
                raise WorkerDied(f"worker exited with code {code}")
            if msg[0] == "ok":
                return msg[1], msg[2]
            _, etype, text = msg
            self.last_error = f"{etype}: {text}"
            if etype == "MemoryError":
                raise MemoryError(text)
            raise RuntimeError(f"{etype}: {text}")
