"""R5-1 — FastAPI boundary of the standalone enhancement pipeline (CPU, offline).

    FES_IMAGE=fes-dev:cpu-p5 scripts/dock.sh uvicorn fes_service.app:app --host 0.0.0.0 --port 8080
    (publish the port with an extra `docker run -p`; see docs/phase_5_rapid/R5-1_Standalone_Pipeline.md)

Endpoints
  GET  /health        liveness (process up)
  GET  /ready         readiness: frozen seals verified, detector loaded, A4 worker loaded and warmed up
                      (+ `limits`: the input limits /v1/enhance enforces, for clients such as the demo UI)
  GET  /ui/           R5-UI local demo page (static HTML/CSS/JS, same origin; it only calls /ready and /v1/enhance)
  GET  /ui/compare.html  R5-9 research comparison page: ORIGINAL vs NAFNet R5-6 vs NAFNet R5-8 (experimental)
  POST /v1/enhance    multipart `image` (file), optional form `input_mode` = full_image | face_chip,
                      `return_image` = true | false, `force_a4` = false (default) | true (research: run A4 even
                      when the gate does not route it; the gate's decision is still reported, `forced` = true),
                      `route_mode` = "" (service default) | a4 | nafnet | auto (R5-5, see fes_service.pipeline),
                      `profile` = "" (service default, unchanged) | original (input returned unchanged, no model) |
                      r5_6 | r5_8 (R5-9 comparison; r5_8 only when FES_EXPERIMENTAL_PROFILES=r5_8, otherwise 422).
                      200 ENHANCED / ORIGINAL (JSON metadata + base64 image), 422 REJECTED, 503 busy / not ready
Environment
  FES_A4_FACTORY   model factory (default fes_service.models:a4_factory)
  FES_A4_TIMEOUT_S hard per-inference timeout, worker killed beyond it (default 120)
  FES_A4_THREADS   torch threads in the worker (default 2)
  FES_MAX_INFLIGHT requests allowed to wait (default 4)
  FES_WARMUP       1 = run one tiny A4 inference before reporting ready (default 1)
  FES_GATE         rw = R5-3 real-world gate (default) · t6 = frozen Phase 4 T6/T7 policy (synthetic-calibrated)
  FES_ROUTE_MODE   auto (default, R5-5: guarded NAFNet blur route, A4 kept only for strong noise) · nafnet · a4 (the
                   R5-1…R5-UI behaviour). nafnet/auto need FES_GATE=rw.
  FES_NAFNET_PROFILE    r5_6 (default: the validated live NAFNet route) · r5_8 (R5-8 research candidate: candidate
                        route config + local-patch guard + out-of-image reflect fill + reflect-pad16 NAFNet factory)
  FES_NAFNET_FACTORY    NAFNet model factory (default fes_service.models:nafnet_factory; r5_8: nafnet_factory_pad16)
  FES_EXPERIMENTAL_PROFILES  "" (default: off) · r5_8 = also load the R5-8 candidate (a second NAFNet worker,
                        FES_NAFNET_R5_8_FACTORY, default models:nafnet_factory_pad16) so a request may ask for
                        profile=r5_8. Never changes requests without `profile`; R5-8 is NOT production-approved.
  FES_NAFNET_TIMEOUT_S  hard per-inference timeout, worker killed beyond it (default 60)
  FES_NAFNET_THREADS    torch threads in the NAFNet worker (default 4)
Models are loaded once per worker process at startup (the ones the default route mode can use; the other one lazily
on its first request). A model that fails to load or crashes never takes the API down: that request returns ORIGINAL.
No request image is stored or logged.
"""

from __future__ import annotations

import base64
import dataclasses
import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from fes_service.pipeline import PROFILE_ORIGINAL, REJECTED, SERVICE_VERSION, Busy, EnhancementPipeline, Limits
from fes_service.worker import IsolatedModel

STATE: dict = {"ready": False, "error": ""}
UI_DIR = Path(__file__).with_name("ui")
# the demo page loads nothing from other origins and may only talk to this service
UI_CSP = ("default-src 'self'; img-src 'self' blob: data:; style-src 'self'; script-src 'self'; connect-src 'self'; "
          "object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'")


def build_pipeline() -> EnhancementPipeline:
    timeout = float(os.environ.get("FES_A4_TIMEOUT_S", "120"))
    model = IsolatedModel(os.environ.get("FES_A4_FACTORY", "fes_service.models:a4_factory"), timeout_s=timeout,
                          threads=int(os.environ.get("FES_A4_THREADS", "2")))
    gate = os.environ.get("FES_GATE", "rw")
    nafnet, experimental = None, {}
    for name in filter(None, (x.strip() for x in os.environ.get("FES_EXPERIMENTAL_PROFILES", "").split(","))):
        if name != "r5_8":
            raise ValueError(f"FES_EXPERIMENTAL_PROFILES: unknown profile {name!r} (only r5_8)")
        experimental[name] = IsolatedModel(
            os.environ.get("FES_NAFNET_R5_8_FACTORY", "fes_service.models:nafnet_factory_pad16"),
            timeout_s=float(os.environ.get("FES_NAFNET_TIMEOUT_S", "60")),
            threads=int(os.environ.get("FES_NAFNET_THREADS", "4")))
    if gate == "rw":
        default_factory = ("fes_service.models:nafnet_factory_pad16" if os.environ.get("FES_NAFNET_PROFILE") == "r5_8"
                           else "fes_service.models:nafnet_factory")
        nafnet = IsolatedModel(os.environ.get("FES_NAFNET_FACTORY", default_factory),
                               timeout_s=float(os.environ.get("FES_NAFNET_TIMEOUT_S", "60")),
                               threads=int(os.environ.get("FES_NAFNET_THREADS", "4")))
    return EnhancementPipeline(model, timeout_s=timeout, max_inflight=int(os.environ.get("FES_MAX_INFLIGHT", "4")),
                               gate=gate, nafnet=nafnet,
                               route_mode=os.environ.get("FES_ROUTE_MODE", "auto" if gate == "rw" else "a4"),
                               nafnet_profile=os.environ.get("FES_NAFNET_PROFILE", "r5_6"),
                               experimental_nafnet=experimental or None)


def startup() -> None:
    STATE.update(ready=False, error="", model_info={}, nafnet_info={}, experimental_info={})
    try:
        pipe = build_pipeline()
        STATE["pipeline"] = pipe
        warm = os.environ.get("FES_WARMUP", "1") == "1"
        if pipe.route_mode in ("a4", "auto"):               # auto keeps A4 for strong noise
            STATE["model_info"] = pipe.a4.start()
            if warm:
                pipe.a4(np.full((16, 16, 3), 128, np.uint8), {"s": 16})
        if pipe.route_mode in ("nafnet", "auto"):
            STATE["nafnet_info"] = pipe.nafnet.start()
            if warm:
                pipe.nafnet(np.full((32, 32, 3), 128, np.uint8), {"s": 32})
            for name, prof in pipe.profiles.items():          # R5-9 experimental profiles (opt-in only)
                if prof.model is not pipe.nafnet:
                    STATE["experimental_info"][name] = prof.model.start()
                    if warm:
                        prof.model(np.full((32, 32, 3), 128, np.uint8), {"s": 32})
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
        if pipe.nafnet is not None:
            pipe.nafnet.close()
        for prof in pipe.profiles.values():
            if prof.model is not pipe.nafnet:
                prof.model.close()


app = FastAPI(title="Face Enhancement Service (research)", version=SERVICE_VERSION, lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "alive", "service_version": SERVICE_VERSION}


@app.get("/ready")
def ready():
    pipe = STATE.get("pipeline")
    body = {"ready": STATE["ready"], "service_version": SERVICE_VERSION, "error": STATE["error"],
            "seals": pipe.seals if pipe else {}, "routing_table": pipe.table if pipe else {},
            "gate": pipe.gate if pipe else "", "route_mode": pipe.route_mode if pipe else "",
            "nafnet_profile": pipe.nafnet_profile if pipe else "",
            "model": STATE.get("model_info", {}),
            "worker_alive": bool(pipe and pipe.a4.alive()), "worker_restarts": pipe.a4.restarts if pipe else 0,
            "nafnet_model": STATE.get("nafnet_info", {}),
            "nafnet_worker_alive": bool(pipe and pipe.nafnet is not None and pipe.nafnet.alive()),
            "nafnet_worker_restarts": pipe.nafnet.restarts if (pipe and pipe.nafnet is not None) else 0,
            "limits": dataclasses.asdict(pipe.limits if pipe else Limits()),
            # R5-9: the profiles a request may name (`profile`), and the one used when it names none
            "profiles": ([PROFILE_ORIGINAL, *pipe.profiles] if pipe else []),
            "default_profile": (pipe.nafnet_profile if pipe and pipe.profiles else ""),
            "experimental_models": STATE.get("experimental_info", {}),
            "validity": "SYNTHETIC-calibrated gate/router; research candidate, not production-approved"}
    return JSONResponse(body, status_code=200 if STATE["ready"] else 503)


@app.post("/v1/enhance")
async def enhance(image: UploadFile = File(...), input_mode: str = Form("full_image"),
                  return_image: bool = Form(True), force_a4: bool = Form(False), route_mode: str = Form(""),
                  profile: str = Form("")):
    if not STATE["ready"]:
        return JSONResponse({"status": "NOT_READY", "error": STATE["error"]}, status_code=503)
    pipe: EnhancementPipeline = STATE["pipeline"]
    data = await image.read(pipe.limits.max_bytes + 1)
    try:
        res = await run_in_threadpool(pipe.process, data, input_mode, force_a4, route_mode or None,
                                     profile or None)
    except Busy:
        return JSONResponse({"status": "BUSY", "error": "too many requests in flight"}, status_code=503,
                            headers={"Retry-After": "30"})
    body = res.metadata()
    body["image_format"] = res.image_format
    body["image_b64"] = base64.b64encode(res.image).decode() if (return_image and res.image) else None
    body["chip_b64"] = base64.b64encode(res.chip).decode() if (return_image and res.chip) else None
    return JSONResponse(body, status_code=422 if res.status == REJECTED else 200)


# ------------------------------------------------------------------ R5-UI demo page (served same-origin: no CORS needed)

class _UIFiles(StaticFiles):
    async def get_response(self, path, scope):
        resp = await super().get_response(path, scope)
        resp.headers.update({"Content-Security-Policy": UI_CSP, "X-Content-Type-Options": "nosniff",
                             "Referrer-Policy": "no-referrer", "Cache-Control": "no-store"})
        return resp


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/ui/")


app.mount("/ui", _UIFiles(directory=UI_DIR, html=True), name="ui")
