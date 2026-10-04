"""R5-1 — FastAPI boundary of the standalone enhancement pipeline (CPU, offline).

    FES_IMAGE=fes-dev:cpu-p5 scripts/dock.sh uvicorn fes_service.app:app --host 0.0.0.0 --port 8080
    (publish the port with an extra `docker run -p`; see docs/phase_5_rapid/R5-1_Standalone_Pipeline.md)

Endpoints
  GET  /health        liveness (process up)
  GET  /ready         readiness: frozen seals verified, detector loaded, A4 worker loaded and warmed up
  POST /v1/enhance    multipart `image` (file), optional form `input_mode` = full_image | face_chip,
                      `return_image` = true | false.
                      200 ENHANCED / ORIGINAL (JSON metadata + base64 image), 422 REJECTED, 503 busy / not ready
Environment
  FES_A4_FACTORY   model factory (default fes_service.models:a4_factory)
  FES_A4_TIMEOUT_S hard per-inference timeout, worker killed beyond it (default 120)
  FES_A4_THREADS   torch threads in the worker (default 2)
  FES_MAX_INFLIGHT requests allowed to wait (default 4)
  FES_WARMUP       1 = run one tiny A4 inference before reporting ready (default 1)
  FES_GATE         rw = R5-3 real-world gate (default) · t6 = frozen Phase 4 T6/T7 policy (synthetic-calibrated)
No request image is stored or logged.
"""

from __future__ import annotations

import base64
import os
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from fes_service.pipeline import REJECTED, SERVICE_VERSION, Busy, EnhancementPipeline
from fes_service.worker import IsolatedModel

STATE: dict = {"ready": False, "error": ""}


def build_pipeline() -> EnhancementPipeline:
    timeout = float(os.environ.get("FES_A4_TIMEOUT_S", "120"))
    model = IsolatedModel(os.environ.get("FES_A4_FACTORY", "fes_service.models:a4_factory"), timeout_s=timeout,
                          threads=int(os.environ.get("FES_A4_THREADS", "2")))
    return EnhancementPipeline(model, timeout_s=timeout, max_inflight=int(os.environ.get("FES_MAX_INFLIGHT", "4")),
                               gate=os.environ.get("FES_GATE", "rw"))


def startup() -> None:
    STATE.update(ready=False, error="", model_info={})
    try:
        pipe = build_pipeline()
        STATE["pipeline"] = pipe
        STATE["model_info"] = pipe.a4.start()
        if os.environ.get("FES_WARMUP", "1") == "1":
            pipe.a4(np.full((16, 16, 3), 128, np.uint8), {"s": 16})
        STATE["ready"] = True
    except Exception as e:  # noqa: BLE001 — stay alive, report not-ready
        STATE["ready"], STATE["error"] = False, f"{type(e).__name__}: {e}"[:300]


@asynccontextmanager
async def lifespan(_app):
    await run_in_threadpool(startup)
    yield
    pipe = STATE.get("pipeline")
    if pipe is not None:
        pipe.a4.close()


app = FastAPI(title="Face Enhancement Service (research)", version=SERVICE_VERSION, lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "alive", "service_version": SERVICE_VERSION}


@app.get("/ready")
def ready():
    pipe = STATE.get("pipeline")
    body = {"ready": STATE["ready"], "service_version": SERVICE_VERSION, "error": STATE["error"],
            "seals": pipe.seals if pipe else {}, "routing_table": pipe.table if pipe else {},
            "gate": pipe.gate if pipe else "",
            "model": STATE.get("model_info", {}),
            "worker_alive": bool(pipe and pipe.a4.alive()), "worker_restarts": pipe.a4.restarts if pipe else 0,
            "validity": "SYNTHETIC-calibrated gate/router; research candidate, not production-approved"}
    return JSONResponse(body, status_code=200 if STATE["ready"] else 503)


@app.post("/v1/enhance")
async def enhance(image: UploadFile = File(...), input_mode: str = Form("full_image"),
                  return_image: bool = Form(True)):
    if not STATE["ready"]:
        return JSONResponse({"status": "NOT_READY", "error": STATE["error"]}, status_code=503)
    pipe: EnhancementPipeline = STATE["pipeline"]
    data = await image.read(pipe.limits.max_bytes + 1)
    try:
        res = await run_in_threadpool(pipe.process, data, input_mode)
    except Busy:
        return JSONResponse({"status": "BUSY", "error": "too many requests in flight"}, status_code=503,
                            headers={"Retry-After": "30"})
    body = res.metadata()
    body["image_format"] = res.image_format
    body["image_b64"] = base64.b64encode(res.image).decode() if (return_image and res.image) else None
    return JSONResponse(body, status_code=422 if res.status == REJECTED else 200)
