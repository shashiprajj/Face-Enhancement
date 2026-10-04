# R5-UI — Interactive Face Enhancement Demo: Report

**Status: GREEN.** The demo UI works against the real service with the real A4 model on CPU.
**Date:** 2026-10-05 · **Base commit:** 7c15a66 (branch `main`) · **Gate:** p5-rwgate-1.0.0 (seal 78837a8d) · **Model:** A4 SwinIR-L

## 1. Objective

Build a simple local web page that does three things:

- lets a user upload a face image;
- sends it to the existing `/v1/enhance` endpoint at runtime;
- shows the original image next to the image the service returned, with the service's actual decision.

This is a demo for research only. The model, gate, detector and routing are unchanged. Feature-Match is not part of it.

## 2. Existing backend contract discovered

The service is `src/fes_service/app.py`, built in R5-1.

| Endpoint | Contract (unchanged) |
|---|---|
| `GET /health` | `{"status": "alive", "service_version"}`: the process is running |
| `GET /ready` | 200 when the frozen seals are verified, the detector is loaded and the A4 worker is warmed up; otherwise 503 with `ready: false, error` |
| `POST /v1/enhance` | Multipart fields: `image` (file), `input_mode` (`full_image` or `face_chip`), `return_image`. Responses: 200 `ENHANCED` or `ORIGINAL`, 422 `REJECTED`, 503 `NOT_READY` or `BUSY` |

**Response vocabulary (used as is by the UI):**

- `status`: ENHANCED, ORIGINAL, REJECTED, BUSY, NOT_READY.
- `decision`: ENHANCE_CANDIDATE, DO_NOT_ENHANCE, UNSAFE_OR_UNCERTAIN.
- Other fields: `reason`, `bucket`, `route`, `used`, `fallback` (TIMEOUT, MODEL_FAILURE, RESOURCE_FAILURE, INVALID_OUTPUT), `escalate`, `warnings`, `n_faces`, `chip_px`, `quality{…}`, `timings_ms`, `image_format` (`png` or `as_received`), `image_b64`.

**Output semantics:**

- ENHANCED returns the 224×224 A4 face crop as a PNG, not the full photo.
- ORIGINAL returns the input bytes unchanged.

**Limits** (`pipeline.Limits`): 20 MB, 40 MP, minimum side 8 px, aspect ratio ≤ 10. The backend decodes images with OpenCV.

**Existing frontend:** none. The only HTML file is `scripts/p4_review_tool.html`, an offline review tool that is unrelated.

## 3. UI architecture

```
Browser ──GET /ui/──────────► FastAPI StaticFiles (same process, same origin)
Browser ──GET /ready────────► existing readiness (online/offline badge + live input limits)
Browser ──POST /v1/enhance──► existing validation → YuNet detection → face chip → rw gate → A4 worker → output guard
        ◄── JSON (status, decision, reason, …, image_b64) ── displayed as received
```

- **Technology:** plain HTML, CSS and ES-module JavaScript. There is no framework, no build step and no new dependency. FastAPI serves the files at `/ui/`, and `/` redirects there.
- **Same origin:** the page and the API share one origin, so **no CORS configuration is needed**. CORS was not enabled, and a test checks that no `Access-Control-Allow-Origin` header is sent.
- **Locked to this service:** the UI responses carry a strict CSP: `default-src 'self'`, `connect-src 'self'`, `img-src 'self' blob: data:`, `frame-ancestors 'none'`. They also send `nosniff`, `no-referrer` and `no-store`. There are no inline scripts and no external URLs, and the page uses no browser storage.
- **Code split:**
  - `fes_ui.js` holds the pure logic: validation, request building, response interpretation and health. It has no DOM access and is unit-tested under Node.
  - `app.js` only wires the DOM.
- **No duplicated logic:** A4 loading, gate logic and detection all stay in the backend.
  - The browser checks only the file type, size and pixel limits. These mirror the backend's own `Limits`, which `/ready` now reports.
  - The backend still validates every input itself.
- **Timeout:** the browser waits 180 s (AbortController). The backend kills A4 at 120 s (`FES_A4_TIMEOUT_S`), and detection plus gate take under 1 s. A backend timeout therefore comes back as a normal `fallback: TIMEOUT` response before the browser gives up. Backend timeouts are unchanged.

## 4. Files changed / added

| File | Change |
|---|---|
| `src/fes_service/app.py` | Adds `limits` to `/ready`, serves `/ui/` as static files with security headers, and redirects `/` to `/ui/` |
| `src/fes_service/ui/index.html` | New: the page |
| `src/fes_service/ui/style.css` | New: styles (light and dark, responsive down to phone width) |
| `src/fes_service/ui/fes_ui.js` | New: the pure UI logic |
| `src/fes_service/ui/app.js` | New: DOM wiring |
| `tests/test_r5_ui.py` | New: tests for serving, CSP, the `/ready` extension and the contract; also runs the Node suite when `node` is available |
| `tests/ui/fes_ui.test.mjs` | New: Node unit tests for the UI logic |
| `docs/phase_5_rapid/R5_UI_Interactive_Demo_REPORT.md` | New: this report |

## 5. API / schema changes

There is one backward-compatible addition: `GET /ready` now includes

```json
"limits": {"max_bytes": 20971520, "max_pixels": 40000000, "min_side": 8, "max_aspect": 10.0}
```

The UI uses it to state and enforce the real upload limit. If the field is missing, as with an older backend, the UI falls back to the same defaults.

Nothing else changed:

- The `/v1/enhance` request and response are unchanged.
- `/health` is unchanged.
- No decision logic was touched.

**Pre-existing field, not changed:** `/ready` already exposed `model.weights`, a path relative to the repository. The UI does not read or display it. Changing it would be a contract change outside this task's scope, so it was left as is.

## 6. Tests

| Suite | Result |
|---|---|
| `node --test tests/ui/fes_ui.test.mjs` (host, Node 24) | **14/14 pass** |
| Docker `fes-dev:cpu-p5` running `pytest tests/test_r5_ui.py tests/test_r5_service.py tests/test_r5_rwgate.py tests/test_p4_gate.py tests/test_p4_routing.py` | **89 passed, 2 skipped** |

The 2 skipped tests are:

- the Node suite inside Docker, because the image has no Node; it runs on the host;
- the opt-in real-A4 test, which needs `FES_RUN_A4=1`.

**Node unit tests cover:**

- upload validation: format, empty file, missing file, oversized file, and the live limit taking precedence;
- dimension checks;
- the multipart request contract;
- the timeout value and that the loading text is truthful (no percentage);
- ENHANCED, ORIGINAL, UNSAFE and A4-fallback responses;
- REJECTED, BUSY, NOT_READY, malformed JSON, unknown status and a missing image;
- network and timeout errors;
- unknown reasons, which are shown raw with no invented explanation;
- MIME sniffing;
- health and limits from `/ready`.

**Python tests cover:**

- the `/` redirect and serving of the page and assets;
- the CSP and security headers, and the absence of CORS;
- no external URLs, inline scripts or browser storage in the UI files;
- the UI calling only `/ready` and `/v1/enhance`, with no SFace, dlib, Feature-Match or threshold code;
- the `/ready` `limits` field, with all R5-1 fields still present;
- the `/v1/enhance` contract;
- the UI's default limits matching `pipeline.Limits`.

**Reset** is a DOM behaviour and was checked in the manual smoke test (§7).

## 7. Manual smoke test (real runtime, real A4, CPU)

**Setup:**

- The service ran in `fes-dev:cpu-p5` on `127.0.0.1:8080` with real A4 and the rw gate. `/ready` reported `ready: true` with rw seal `78837a8d…`.
- The page was driven in Chrome.
- Test images came from the public CC0 photo set (`Dataset-and-code/dataset`) and from generated files.
- **No personal images were used**, and none were copied into the repository.

| # | Case | Input | UI result (from the backend response) | Expected (R5-3 manifest) |
|---|---|---|---|---|
| 1 | Poor quality, blur | img154.jpg (256×400, grey) | **ENHANCED** · ENHANCE_CANDIDATE · `strong_blur` · A4/A4 · 224×224 PNG crop · warning `grayscale_preserved` · server time 36.1 s | ENHANCED strong_blur, chip 166 ✔ |
| 2 | Poor quality, noise | img72.jpg | **ENHANCED** · `strong_noise` · noise σ 14.3 · 224×224 PNG · 38.8 s | ENHANCED strong_noise, chip 174 ✔ |
| 3 | Normal | img2.jpg | **ORIGINAL** · DO_NOT_ENHANCE · `acceptable` · original JPEG bytes returned (1200×629) · 45 ms | acceptable ✔ |
| 4 | Expected to stay original | img1.jpg | **ORIGINAL** · DO_NOT_ENHANCE · `degraded_no_a4_evidence:noise+lowres` · 73 ms | degraded_no_a4_evidence ✔ |
| 5 | Unsafe (real photo) | img6.jpg | **ORIGINAL** · UNSAFE_OR_UNCERTAIN · `no_valid_face:FACE_TOO_SMALL` · escalate yes · "Not processed (unsafe / uncertain)" | no_valid_face ✔ |
| 6 | No face | generated noise PNG | **ORIGINAL** · UNSAFE_OR_UNCERTAIN · `no_valid_face:NO_FACE_DETECTED` · PNG returned unchanged | — ✔ |
| 7 | Invalid file | text file named `.png` | Client error: "The browser could not read it…"; Enhance stays disabled | ✔ |
| 8 | Backend REJECTED | truncated PNG POSTed through the page's own request builder and interpreter | HTTP 422 REJECTED, "Image rejected: … not a decodable image", no image shown | ✔ |
| 9 | Oversized | 21 MB BMP | "File is too large (21.0 MB). The service accepts up to 20.0 MB." Nothing is sent | ✔ |
| 10 | Unsupported type | GIF | "Unsupported file type…" | ✔ |
| 11 | Backend unavailable | container stopped, then Enhance clicked | "Service unavailable. Could not reach the enhancement service…" The badge changes to **Enhancement Service Offline** and the button becomes usable again | ✔ |
| 12 | Duplicate submit | double-click on Enhance | the server log shows **exactly 1** `POST /v1/enhance` | ✔ |
| 13 | Reset | Try Another Image | result, preview, error, file input and metadata all cleared; Enhance disabled | ✔ |
| 14 | Loading state | during A4 | spinner, the message "Processing image… Enhancement may take up to a minute on CPU.", an honest elapsed-seconds counter (no percentage), and all buttons disabled | ✔ |

**Fix made during the smoke test:** case 5 first showed the explanation "No usable face was found" for `FACE_TOO_SMALL`. This was misleading, because 2 faces were detected. The text now reads "A face was found, but it is too small to process safely." The raw reason code was always shown.

**Fix made after hand-over (user report):** a tall photo was shown with its top and bottom cut off in both the ORIGINAL and PROCESSED boxes. This was a display bug in the CSS: the image's `max-height: 100%` did not resolve inside the grid box, and `overflow: hidden` clipped the rest. The service had returned the photo unchanged (ORIGINAL). The boxes now letterbox every image (`object-fit: contain` in a fixed 340 px box). Checked in Chrome with 600×1200, 1600×500, 224×224 and 100×3000 images, all fully visible, and covered by the regression test `test_compare_frames_never_crop_the_image`.

**Not exercised live:**

- the 180 s browser timeout and the BUSY response; these are covered by unit tests;
- A4 failure fallbacks; these are covered by the existing R5-1 worker tests.

## 8. UX description

- **Header:** title, the line "Upload → Enhance → Compare", and a live service badge (Online / Starting / Not Ready / Offline) that polls `/ready` every 15 s.
- **Upload card:**
  - A drop zone that also works as a file picker (click, or Enter/Space from the keyboard).
  - A preview with the file name, size and dimensions.
  - Replace and Remove buttons.
  - An "Enhance Image" button, enabled only when a valid image is selected.
  - Errors appear in a red alert.
- **Result card:**
  - The backend `status` badge: green ENHANCED, grey ORIGINAL, amber when UNSAFE_OR_UNCERTAIN.
  - The `decision` as a chip, a one-line title and message, and "Reason: `<code>` — `<plain-language note>`".
  - ORIGINAL and PROCESSED images side by side, stacked on phones.
  - A note under PROCESSED saying whether it is the enhanced 224×224 face crop or the unchanged original bytes.
  - Any service warnings.
  - A "Backend metadata" list showing only fields the backend returned, plus the round-trip time measured in the browser.
  - A fixed disclaimer: research only, gate not production-approved, enhanced output not verified for identity matching.
  - A "Try Another Image" button.
- **Not added:** a before/after slider. The two images have different framing (a full photo versus a 224-px face crop), so a slider would misalign them and mislead the viewer.

## 9. Run instructions

From Git Bash in the repository root, with Docker running and `fes-dev:cpu-p5` built (`docker build -f docker/Dockerfile.cpu-p5 -t fes-dev:cpu-p5 .`):

```bash
MSYS_NO_PATHCONV=1 docker run --rm --name fes-ui -p 127.0.0.1:8080:8080 \
  -v "$(pwd -W):/workspace" -v "$(pwd -W)/models:/workspace/models:ro" \
  -w /workspace -e PYTHONPATH=src fes-dev:cpu-p5 \
  uvicorn fes_service.app:app --host 0.0.0.0 --port 8080
```

1. Wait for the log line `Application startup complete` (about 20–40 s while A4 loads and warms up).
2. Open **http://localhost:8080/** in a browser. It redirects to `/ui/`.
3. Stop the service with `Ctrl+C` or `docker stop fes-ui`.

**Notes:**

- The UI needs no separate server and no extra environment variables.
- The existing service variables still apply: `FES_GATE` (default `rw`), `FES_A4_TIMEOUT_S` (120), `FES_A4_THREADS` (2) and `FES_MAX_INFLIGHT` (4).
- The port is bound to `127.0.0.1` only, so the demo is not exposed on the network.

**Tests:**

```bash
node --test tests/ui/fes_ui.test.mjs
FES_IMAGE=fes-dev:cpu-p5 scripts/dock.sh python -m pytest -q tests/test_r5_ui.py tests/test_r5_service.py
```

## 10. Limitations

- **ENHANCED output is a face crop.** The service returns the 224×224 A4 face chip; it does not paste the face back into the photo. The UI states this rather than hiding it.
- **The gate admits few real photos.** p5-rwgate admits only strongly blurred or noisy, recognisable faces. Most real photos therefore come back ORIGINAL, which is the designed behaviour (R5-3: 3 of 200 photos go to A4).
- **A4 takes about 35–40 s on this CPU**, and up to the 120 s backend limit.
- **One request at a time.** The pipeline processes requests one after another; when the queue is full, the UI shows the backend's BUSY message.
- **No identity verification.** Nothing in the UI claims that the enhanced output is identity-matchable. Feature-Match validation (R5-4) is still blocked.
- **Not production-ready.** There is no authentication and it is meant for local use only. The gate is research-grade, and the A4 and YuNet licences are still REVIEW_REQUIRED.

## 11. Model / gate behaviour unchanged

- `git diff` touches only `src/fes_service/app.py` among existing files, and only adds the `/ready` `limits` field and the static `/ui` mount.
- The following are byte-identical to the base commit: `pipeline.py`, `rwgate.py`, `models.py`, `worker.py`, `src/fes_bench/*`, `configs/*` and the model weights.
- The runtime `/ready` reports the same seals: t6 `b8ee2c43`, t7 `2f026f22`, rw `78837a8d`.
- The smoke-test decisions match the frozen R5-3 manifest (`data/manifests/phase5_rapid/rwgate/realphotos.csv.gz`) for every photo tested.
- No new model was added (no GFPGAN or CodeFormer), and no thresholds, Feature-Match, SFace or dlib appear in the UI.

## 12. Commit

The commit hash is given in the commit message and in the final task response. It is the commit that adds this report.
