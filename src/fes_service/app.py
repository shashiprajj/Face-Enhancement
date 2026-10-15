"""R5-1 — FastAPI boundary of the standalone enhancement pipeline (CPU, offline).

    FES_IMAGE=fes-dev:cpu-p5 scripts/dock.sh uvicorn fes_service.app:app --host 0.0.0.0 --port 8080
    (publish the port with an extra `docker run -p`; see docs/phase_5_rapid/R5-1_Standalone_Pipeline.md)

Endpoints
  GET  /health        liveness (process up)
  GET  /ready         readiness: frozen seals verified, detector loaded, A4 worker loaded and warmed up
                      (+ `limits`: the input limits /v1/enhance enforces, for clients such as the demo UI)
  GET  /ui/           R5-UI local demo page (static HTML/CSS/JS, same origin; it only calls /ready and /v1/enhance)
  GET  /ui/compare.html  R5-9 research comparison page: ORIGINAL vs NAFNet R5-6 vs NAFNet R5-8 / R5-10 (experimental)
  POST /v1/enhance    multipart `image` (file), optional form `input_mode` = full_image | face_chip,
                      `return_image` = true | false, `force_a4` = false (default) | true (research: run A4 even
                      when the gate does not route it; the gate's decision is still reported, `forced` = true),
                      `route_mode` = "" (service default) | a4 | nafnet | auto (R5-5, see fes_service.pipeline),
                      `profile` = "" (service default, unchanged) | original (input returned unchanged, no model) |
                      r5_6 | r5_8 | r5_10 | r6 | r6_ft (R5-9/R5-10/R6 comparison; r5_8 / r5_10 / r6 / r6_ft only when
                      listed in FES_EXPERIMENTAL_PROFILES, otherwise 422) | best (needs r6 and r6_ft; see /v3) |
                      best_side (SIDE-FACE research profile, same requirement: `best`, and where `best` stops because
                      the face is turned, the same route on a chip framed by the detection box; pipeline.PROFILE_BEST_SIDE).
                      200 ENHANCED / ORIGINAL (JSON metadata + base64 image), 422 REJECTED, 503 busy / not ready
  POST /v2/enhance    R6-FT research endpoint: multipart `image`, optional `input_mode`, `return_image`; the same
                      responses as /v1/enhance, always run with profile r6_ft (the r6 route with the face-blur
                      fine-tuned NAFNet weights). 422 unless FES_EXPERIMENTAL_PROFILES lists r6_ft. /v1 is unchanged.
  POST /v3/enhance    R6-BEST research endpoint, 1 input → 1 output: the same request and responses, always run with
                      profile best (r6 and r6_ft both run; the output nafnet_route.select_best prefers is returned and
                      `selection` reports the candidates and the choice). 422 unless FES_EXPERIMENTAL_PROFILES lists
                      r6 and r6_ft.
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
                        · r5_10 (R5-10 research candidate: r5_8 with the DEV-recalibrated blur boundary)
                        · r6 (R6 research candidate: r5_8 route + NAFNet on the chip and its 3 mirrors, ORIGINAL when
                        any run is unstable, else the median of the 4 runs with the input's colour; nafnet_factory_r6)
  FES_NAFNET_FACTORY    NAFNet model factory (default fes_service.models:nafnet_factory; r5_8/r5_10: nafnet_factory_pad16)
  FES_EXPERIMENTAL_PROFILES  "" (default: off) · comma list of r5_8, r5_10 = also load those candidates so a request
                        may ask for profile=r5_8 / r5_10. They share ONE extra NAFNet worker (FES_NAFNET_R5_8_FACTORY,
                        default models:nafnet_factory_pad16: both use the same model + padding; only the route config
                        differs). r6 = its own worker (FES_NAFNET_R6_FACTORY, default models:nafnet_factory_r6,
                        timeout FES_NAFNET_R6_TIMEOUT_S, default 120: 4 NAFNet runs per request). r6_ft = r6 with
                        the fine-tuned weights, its own worker too (FES_NAFNET_R6_FT_FACTORY, default
                        models:nafnet_factory_r6_ft; also what POST /v2/enhance runs). Never changes
                        requests without `profile`; none of them is production-approved.
  FES_NAFNET_BACKEND    torch (default: unchanged) · onnx (R6-ONNX, opt-in: the r6 / r6_ft workers run the ONNX exports
                        of the same weights on ONNX Runtime — models:nafnet_factory_r6_onnx / _r6_ft_onnx; needs the
                        fes-dev:cpu-p5-onnx image and models/M28_nafnet/onnx/). Only those two profiles have an ONNX
                        model: A4 and the r5_6 / r5_8 / r5_10 NAFNet workers stay on PyTorch. Any other value, onnx
                        without r6 / r6_ft enabled, or a missing model / onnxruntime = the service does not become
                        ready (it never switches backend by itself). /ready reports `nafnet_backend`.
  FES_BEST_PARALLEL     0 (default: unchanged) · 1 (PERF Step 1: profile best / POST /v3/enhance starts the r6 and r6_ft
                        model inference at the same moment, each in its own worker, instead of one after the other;
                        same gate, checks, guards, selection and fallbacks). /ready reports `best_parallel`.
  FES_R6_MIRROR_BATCH   0 (default: unchanged) · 1 (PERF Step 2: the r6 / r6_ft workers run their four mirror passes
                        as one batch of 4 — models:nafnet_factory_r6_batched / _r6_ft_batched; PyTorch only, so it
                        cannot be combined with FES_NAFNET_BACKEND=onnx). /ready reports `r6_mirror_batch`.
                        Any other value of either setting = the service does not become ready.
  FES_NAFNET_TIMEOUT_S  hard per-inference timeout, worker killed beyond it (default 60)
  FES_NAFNET_THREADS    torch threads in the NAFNet worker (default 4)
  FES_SETTINGS          deployment settings file (default configs/deployment_settings.json; missing = defaults). Its
                        "blur" block sets the admin blur range on the 0-100 % blur scale (fes_service.blur_scale):
                        enhance_from_pct (null = validated limits; minimum in /ready) and give_up_above_pct (null =
                        off). An invalid value = the service does not become ready. /ready reports `blur_settings`.
                        /v1, /v2 and /v3 also take both as optional form fields; a request may only narrow the range.
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

from fes_service import blur_scale as BS
from fes_service.pipeline import (BEST_CANDIDATES, BEST_PROFILES, EXPERIMENTAL_PROFILES, PROFILE_BEST, PROFILE_ORIGINAL,
                                  REJECTED, SERVICE_VERSION, Busy, EnhancementPipeline, Limits)
from fes_service.worker import IsolatedModel

STATE: dict = {"ready": False, "error": ""}
UI_DIR = Path(__file__).with_name("ui")
# profiles with their own 4-run consensus worker: profile → (factory env var, default factory)
R6_FACTORIES = {"r6": ("FES_NAFNET_R6_FACTORY", "fes_service.models:nafnet_factory_r6"),
                "r6_ft": ("FES_NAFNET_R6_FT_FACTORY", "fes_service.models:nafnet_factory_r6_ft")}
# R6-ONNX: the same profiles on ONNX Runtime (FES_NAFNET_BACKEND=onnx); PyTorch stays the default backend
R6_ONNX_FACTORIES = {"r6": "fes_service.models:nafnet_factory_r6_onnx",
                     "r6_ft": "fes_service.models:nafnet_factory_r6_ft_onnx"}
NAFNET_BACKENDS = ("torch", "onnx")
# PERF Step 2: the same profiles with the four mirror passes as one batch (FES_R6_MIRROR_BATCH=1; PyTorch only)
R6_BATCH_FACTORIES = {"r6": "fes_service.models:nafnet_factory_r6_batched",
                      "r6_ft": "fes_service.models:nafnet_factory_r6_ft_batched"}
V2_PROFILE = "r6_ft"                                # POST /v2/enhance always runs this profile
# the demo page loads nothing from other origins and may only talk to this service
UI_CSP = ("default-src 'self'; img-src 'self' blob: data:; style-src 'self'; script-src 'self'; connect-src 'self'; "
          "object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'")


def nafnet_backend() -> str:
    """FES_NAFNET_BACKEND, validated: an unknown value is an error, never a silent default."""
    backend = os.environ.get("FES_NAFNET_BACKEND", "torch")
    if backend not in NAFNET_BACKENDS:
        raise ValueError(f"FES_NAFNET_BACKEND: unknown backend {backend!r} (only {', '.join(NAFNET_BACKENDS)})")
    return backend


def flag(name: str) -> bool:
    """A 0 / 1 setting, validated: anything else is an error, never a silent default."""
    v = os.environ.get(name, "0")
    if v not in ("0", "1"):
        raise ValueError(f"{name}: expected 0 or 1, got {v!r}")
    return v == "1"


def build_pipeline() -> EnhancementPipeline:
    timeout = float(os.environ.get("FES_A4_TIMEOUT_S", "120"))
    model = IsolatedModel(os.environ.get("FES_A4_FACTORY", "fes_service.models:a4_factory"), timeout_s=timeout,
                          threads=int(os.environ.get("FES_A4_THREADS", "2")))
    gate = os.environ.get("FES_GATE", "rw")
    blur, blur_src = BS.load_settings()             # admin blur range (validated by the pipeline)
    backend = nafnet_backend()
    parallel, batch = flag("FES_BEST_PARALLEL"), flag("FES_R6_MIRROR_BATCH")
    if batch and backend == "onnx":
        raise ValueError("FES_R6_MIRROR_BATCH=1 cannot be combined with FES_NAFNET_BACKEND=onnx (the ONNX models are "
                         "exported for one image per call)")
    r6_default = (R6_ONNX_FACTORIES if backend == "onnx" else R6_BATCH_FACTORIES if batch
                  else {k: v[1] for k, v in R6_FACTORIES.items()})
    nafnet, experimental = None, {}
    shared = None                                   # r5_8 and r5_10 run the same pad16 NAFNet: one worker for both
    for name in filter(None, (x.strip() for x in os.environ.get("FES_EXPERIMENTAL_PROFILES", "").split(","))):
        if name not in EXPERIMENTAL_PROFILES:
            raise ValueError(f"FES_EXPERIMENTAL_PROFILES: unknown profile {name!r} (only {', '.join(EXPERIMENTAL_PROFILES)})")
        if name in R6_FACTORIES:                    # R6 / R6-FT: each its own worker (4-run consensus factory)
            env = R6_FACTORIES[name][0]
            experimental[name] = IsolatedModel(
                os.environ.get(env, r6_default[name]),
                timeout_s=float(os.environ.get("FES_NAFNET_R6_TIMEOUT_S", "120")),
                threads=int(os.environ.get("FES_NAFNET_THREADS", "4")))
            continue
        if shared is None:
            shared = IsolatedModel(
                os.environ.get("FES_NAFNET_R5_8_FACTORY", "fes_service.models:nafnet_factory_pad16"),
                timeout_s=float(os.environ.get("FES_NAFNET_TIMEOUT_S", "60")),
                threads=int(os.environ.get("FES_NAFNET_THREADS", "4")))
        experimental[name] = shared
    if backend == "onnx" and not (set(experimental) | {os.environ.get("FES_NAFNET_PROFILE")}) & set(R6_ONNX_FACTORIES):
        raise ValueError("FES_NAFNET_BACKEND=onnx: no ONNX-backed profile is enabled (it applies to r6 / r6_ft only; "
                         "list them in FES_EXPERIMENTAL_PROFILES)")
    if gate == "rw":
        default_factory = (r6_default[os.environ["FES_NAFNET_PROFILE"]]
                           if os.environ.get("FES_NAFNET_PROFILE") in R6_FACTORIES
                           else "fes_service.models:nafnet_factory_pad16"
                           if os.environ.get("FES_NAFNET_PROFILE") in EXPERIMENTAL_PROFILES
                           else "fes_service.models:nafnet_factory")
        nafnet = IsolatedModel(os.environ.get("FES_NAFNET_FACTORY", default_factory),
                               timeout_s=float(os.environ.get("FES_NAFNET_TIMEOUT_S", "60")),
                               threads=int(os.environ.get("FES_NAFNET_THREADS", "4")))
    return EnhancementPipeline(model, timeout_s=timeout, max_inflight=int(os.environ.get("FES_MAX_INFLIGHT", "4")),
                               gate=gate, nafnet=nafnet,
                               route_mode=os.environ.get("FES_ROUTE_MODE", "auto" if gate == "rw" else "a4"),
                               nafnet_profile=os.environ.get("FES_NAFNET_PROFILE", "r5_6"),
                               experimental_nafnet=experimental or None, best_parallel=parallel,
                               blur_settings=blur, blur_settings_source=blur_src)


def startup() -> None:
    STATE.update(ready=False, error="", model_info={}, nafnet_info={}, experimental_info={}, nafnet_backend="")
    try:
        pipe = build_pipeline()
        STATE["nafnet_backend"] = nafnet_backend()
        STATE["r6_mirror_batch"] = flag("FES_R6_MIRROR_BATCH")
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
            started = {}                                       # model id → info (r5_8 / r5_10 share one worker)
            for name, prof in pipe.profiles.items():          # R5-9 / R5-10 experimental profiles (opt-in only)
                if prof.model is not pipe.nafnet:
                    if id(prof.model) not in started:
                        started[id(prof.model)] = prof.model.start()
                        if warm:
                            prof.model(np.full((32, 32, 3), 128, np.uint8), {"s": 32})
                    STATE["experimental_info"][name] = started[id(prof.model)]
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
        for m in {id(p.model): p.model for p in pipe.profiles.values() if p.model is not pipe.nafnet}.values():
            m.close()


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
            "profiles": ([PROFILE_ORIGINAL, *pipe.profiles,
                          *((BEST_PROFILES if pipe.side_cfg else [PROFILE_BEST])
                            if all(p in pipe.profiles for p in BEST_CANDIDATES) else [])]
                         if pipe else []),
            "default_profile": (pipe.nafnet_profile if pipe and pipe.profiles else ""),
            "experimental_models": STATE.get("experimental_info", {}),
            "nafnet_backend": STATE.get("nafnet_backend", ""),
            "best_parallel": bool(pipe and pipe.best_parallel),
            "r6_mirror_batch": bool(STATE.get("r6_mirror_batch", False)),
            "blur_settings": blur_settings(pipe),
            "validity": "SYNTHETIC-calibrated gate/router; research candidate, not production-approved"}
    return JSONResponse(body, status_code=200 if STATE["ready"] else 503)


def blur_settings(pipe) -> dict | None:
    """The deployment's admin blur range and its limits (None when the service has no blur scale)."""
    if pipe is None or pipe.blur_scale is None:
        return None
    s = pipe.blur_settings
    return {"scale_version": BS.SCALE_VERSION, "source": pipe.blur_settings_source,
            "enhance_from_pct": s.enhance_from_pct, "give_up_above_pct": s.give_up_above_pct,
            "validated_limit_pct": {k: round(v, 1) for k, v in pipe.blur_scale.floor().items()},
            "min_enhance_from_pct": pipe.blur_scale.min_enhance_from()}


@app.post("/v1/enhance")
async def enhance(image: UploadFile = File(...), input_mode: str = Form("full_image"),
                  return_image: bool = Form(True), force_a4: bool = Form(False), route_mode: str = Form(""),
                  profile: str = Form(""), enhance_from_pct: str = Form(""), give_up_above_pct: str = Form("")):
    return await _enhance(image, input_mode, return_image, force_a4, route_mode or None, profile or None,
                          enhance_from_pct, give_up_above_pct)


@app.post("/v2/enhance")
async def enhance_v2(image: UploadFile = File(...), input_mode: str = Form("full_image"),
                     return_image: bool = Form(True), enhance_from_pct: str = Form(""),
                     give_up_above_pct: str = Form("")):
    """R6-FT: the same contract as /v1/enhance, always with profile r6_ft (422 unless the service enables it)."""
    return await _enhance(image, input_mode, return_image, False, None, V2_PROFILE, enhance_from_pct,
                          give_up_above_pct)


@app.post("/v3/enhance")
async def enhance_v3(image: UploadFile = File(...), input_mode: str = Form("full_image"),
                     return_image: bool = Form(True), enhance_from_pct: str = Form(""),
                     give_up_above_pct: str = Form("")):
    """R6-BEST: 1 input → 1 output. The same contract as /v1/enhance, always with profile `best`: r6 and r6_ft both
    run their guarded routes and the preferred accepted output is returned (`selection` says which and why)."""
    return await _enhance(image, input_mode, return_image, False, None, PROFILE_BEST, enhance_from_pct,
                          give_up_above_pct)


async def _enhance(image, input_mode, return_image, force_a4, route_mode, profile, enhance_from_pct="",
                   give_up_above_pct=""):
    if not STATE["ready"]:
        return JSONResponse({"status": "NOT_READY", "error": STATE["error"]}, status_code=503)
    pipe: EnhancementPipeline = STATE["pipeline"]
    try:
        blur = BS.parse(enhance_from_pct, give_up_above_pct)
    except ValueError as e:
        return JSONResponse({"status": REJECTED, "error": str(e)}, status_code=422)
    data = await image.read(pipe.limits.max_bytes + 1)
    try:
        res = await run_in_threadpool(pipe.process, data, input_mode, force_a4, route_mode, profile, blur)
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
