"""FastAPI boundary of the Face Enhancement Service (CPU, offline).

    docker run -d --name fes-ui -p 127.0.0.1:8080:8080 -v <repo>:/workspace -v <repo>/models:/workspace/models:ro \
        -w /workspace -e PYTHONPATH=src fes-dev:cpu-p5-onnx uvicorn fes_service.app:app --host 0.0.0.0 --port 8080

Endpoints
  GET  /health        liveness (process up)
  GET  /ready         readiness: sealed configs verified, detector loaded, both model workers loaded and warmed up;
                      reports the profiles, the models, the input `limits` and the admin `blur_settings`
  GET  /ui/best.html  demo page: one photo in, one enhanced face out (static HTML/CSS/JS, same origin)
  POST /v1/enhance    multipart `image` (file), optional form `input_mode` = full_image (default) | face_chip,
                      `return_image` = true | false, `profile` = best (default) | best_side | r6 | r6_ft (see
                      fes_service.pipeline), `enhance_from_pct` / `give_up_above_pct` (may only narrow the deployment
                      blur range). 200 ENHANCED / ORIGINAL (JSON metadata + base64 image), 422 REJECTED, 503 busy /
                      not ready
  POST /v2/enhance    the same request and responses, always profile r6_ft
  POST /v3/enhance    the same request and responses, always profile best (what the demo page calls)
Environment
  FES_MAX_INFLIGHT      requests allowed to wait (default 4)
  FES_WARMUP            1 = run one tiny inference per worker before reporting ready (default 1)
  FES_NAFNET_TIMEOUT_S  hard per-request model timeout, worker killed beyond it (default 120: 4 runs per request)
  FES_NAFNET_THREADS    ONNX Runtime intra-op threads per worker (default 4)
  FES_NAFNET_R6_FACTORY / FES_NAFNET_R6_FT_FACTORY   model factory override ("module:function", tests only)
  FES_SETTINGS          deployment settings file (default configs/deployment_settings.json; missing = defaults). Its
                        "blur" block sets the admin blur range on the 0-100 % blur scale (fes_service.blur_scale):
                        enhance_from_pct (null = validated limits; minimum in /ready) and give_up_above_pct (null =
                        off). An invalid value = the service does not become ready.
Models are loaded once per worker process at startup. A model that fails, hangs or crashes on a request never takes
the API down: that request returns ORIGINAL. No request image is stored or logged.
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

from fes_service import blur_scale as BS
from fes_service.pipeline import (BEST_CANDIDATES, DEFAULT_PROFILE, PROFILE_BEST, PROFILES, R6_FT, REJECTED,
                                  SERVICE_VERSION, Busy, EnhancementPipeline, Limits)
from fes_service.worker import IsolatedModel

STATE: dict = {"ready": False, "error": ""}
UI_DIR = Path(__file__).with_name("ui")
UI_HOME = "/ui/best.html"
# candidate → (factory override env var, factory)
FACTORIES = {"r6": ("FES_NAFNET_R6_FACTORY", "fes_service.models:nafnet_factory_r6_onnx"),
             "r6_ft": ("FES_NAFNET_R6_FT_FACTORY", "fes_service.models:nafnet_factory_r6_ft_onnx")}
# the demo page loads nothing from other origins and may only talk to this service
UI_CSP = ("default-src 'self'; img-src 'self' blob: data:; style-src 'self'; script-src 'self'; connect-src 'self'; "
          "object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'")


def build_pipeline() -> EnhancementPipeline:
    blur, blur_src = BS.load_settings()             # admin blur range (validated by the pipeline)
    timeout = float(os.environ.get("FES_NAFNET_TIMEOUT_S", "120"))
    models = {name: IsolatedModel(os.environ.get(env, factory), timeout_s=timeout)
              for name, (env, factory) in FACTORIES.items()}
    return EnhancementPipeline(models, max_inflight=int(os.environ.get("FES_MAX_INFLIGHT", "4")),
                               blur_settings=blur, blur_settings_source=blur_src)


def startup() -> None:
    STATE.update(ready=False, error="", models={})
    try:
        pipe = build_pipeline()
        STATE["pipeline"] = pipe
        warm = os.environ.get("FES_WARMUP", "1") == "1"
        for name in BEST_CANDIDATES:
            STATE["models"][name] = pipe.models[name].start()
            if warm:
                pipe.models[name](np.full((32, 32, 3), 128, np.uint8), {"s": 32})
        STATE["ready"] = True
    except Exception as e:  # noqa: BLE001 — stay alive, report not-ready
        STATE["ready"], STATE["error"] = False, f"{type(e).__name__}: {e}"[:300]


@asynccontextmanager
async def lifespan(_app):
    await run_in_threadpool(startup)
    yield
    pipe = STATE.get("pipeline")
    if pipe is not None:
        for m in pipe.models.values():
            m.close()


app = FastAPI(title="Face Enhancement Service (research)", version=SERVICE_VERSION, lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "alive", "service_version": SERVICE_VERSION}


@app.get("/ready")
def ready():
    pipe = STATE.get("pipeline")
    body = {"ready": STATE["ready"], "service_version": SERVICE_VERSION, "error": STATE["error"],
            "seals": pipe.seals if pipe else {},
            "profiles": list(PROFILES) if pipe else [], "default_profile": DEFAULT_PROFILE if pipe else "",
            "models": STATE.get("models", {}),
            "workers": {n: {"alive": m.alive(), "restarts": m.restarts} for n, m in pipe.models.items()} if pipe else {},
            "limits": dataclasses.asdict(pipe.limits if pipe else Limits()),
            "blur_settings": blur_settings(pipe),
            "validity": "SYNTHETIC-calibrated gate; research candidate, not production-approved"}
    return JSONResponse(body, status_code=200 if STATE["ready"] else 503)


def blur_settings(pipe) -> dict | None:
    """The deployment's admin blur range and its limits (None before the pipeline exists)."""
    if pipe is None:
        return None
    s = pipe.blur_settings
    return {"scale_version": BS.SCALE_VERSION, "source": pipe.blur_settings_source,
            "enhance_from_pct": s.enhance_from_pct, "give_up_above_pct": s.give_up_above_pct,
            "validated_limit_pct": {k: round(v, 1) for k, v in pipe.blur_scale.floor().items()},
            "min_enhance_from_pct": pipe.blur_scale.min_enhance_from()}


@app.post("/v1/enhance")
async def enhance(image: UploadFile = File(...), input_mode: str = Form("full_image"),
                  return_image: bool = Form(True), profile: str = Form(""), enhance_from_pct: str = Form(""),
                  give_up_above_pct: str = Form("")):
    return await _enhance(image, input_mode, return_image, profile or None, enhance_from_pct, give_up_above_pct)


@app.post("/v2/enhance")
async def enhance_v2(image: UploadFile = File(...), input_mode: str = Form("full_image"),
                     return_image: bool = Form(True), enhance_from_pct: str = Form(""),
                     give_up_above_pct: str = Form("")):
    """The same contract as /v1/enhance, always with profile r6_ft."""
    return await _enhance(image, input_mode, return_image, R6_FT, enhance_from_pct, give_up_above_pct)


@app.post("/v3/enhance")
async def enhance_v3(image: UploadFile = File(...), input_mode: str = Form("full_image"),
                     return_image: bool = Form(True), enhance_from_pct: str = Form(""),
                     give_up_above_pct: str = Form("")):
    """1 input → 1 output. The same contract as /v1/enhance, always with profile `best`: r6 and r6_ft both run their
    guarded routes and the preferred accepted output is returned (`selection` says which and why)."""
    return await _enhance(image, input_mode, return_image, PROFILE_BEST, enhance_from_pct, give_up_above_pct)


async def _enhance(image, input_mode, return_image, profile, enhance_from_pct="", give_up_above_pct=""):
    if not STATE["ready"]:
        return JSONResponse({"status": "NOT_READY", "error": STATE["error"]}, status_code=503)
    pipe: EnhancementPipeline = STATE["pipeline"]
    try:
        blur = BS.parse(enhance_from_pct, give_up_above_pct)
    except ValueError as e:
        return JSONResponse({"status": REJECTED, "error": str(e)}, status_code=422)
    data = await image.read(pipe.limits.max_bytes + 1)
    try:
        res = await run_in_threadpool(pipe.process, data, input_mode, profile, blur)
    except Busy:
        return JSONResponse({"status": "BUSY", "error": "too many requests in flight"}, status_code=503,
                            headers={"Retry-After": "30"})
    body = res.metadata()
    body["image_format"] = res.image_format
    body["image_b64"] = base64.b64encode(res.image).decode() if (return_image and res.image) else None
    body["chip_b64"] = base64.b64encode(res.chip).decode() if (return_image and res.chip) else None
    return JSONResponse(body, status_code=422 if res.status == REJECTED else 200)


# ------------------------------------------------------------------ demo page (served same-origin: no CORS needed)

class _UIFiles(StaticFiles):
    async def get_response(self, path, scope):
        resp = await super().get_response(path, scope)
        resp.headers.update({"Content-Security-Policy": UI_CSP, "X-Content-Type-Options": "nosniff",
                             "Referrer-Policy": "no-referrer", "Cache-Control": "no-store"})
        return resp


@app.get("/", include_in_schema=False)
@app.get("/ui", include_in_schema=False)
@app.get("/ui/", include_in_schema=False)
def root():
    return RedirectResponse(UI_HOME)


app.mount("/ui", _UIFiles(directory=UI_DIR), name="ui")
