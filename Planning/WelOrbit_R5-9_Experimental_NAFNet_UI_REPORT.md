# R5-9 — Experimental NAFNet UI + Manual Evaluation Harness

**Date:** 2026-10-08 · **Base commit:** `a09344a` (R5-8) · **Status:** DONE. All 20 acceptance criteria are met; see §16.

Neither NAFNet profile is production-approved. R5-8 is still RED, a rejected research candidate. This task adds a way
to look at the existing evidence. It changes no algorithm.

## 1. Objective

The owner needs to compare three outputs side by side on real images:

- **ORIGINAL**
- **NAFNet R5-6 Reference**
- **NAFNet R5-8 Experimental**

Each output shows its metadata and any fallback reason, and the owner can record a manual verdict. The result
decides whether NAFNet is worth a real Feature-Match validation.

## 2. Existing UI baseline

- `/ui/` (R5-UI, commit 459ab45) is a single-image demo. It sends one default request and has the **Force A4**
  research toggle. It is served same-origin by the FastAPI service under a strict CSP.
- At the owner's request this page is **kept as it is**. The only change is one header link to the new page.
- The new page is `/ui/compare.html` and has its own files: `compare.html`, `compare.css`, `compare.js` (DOM) and
  `fes_compare.js` (pure logic, tested with node).
- It reuses the existing `fes_ui.js` helpers. It calls only `/ready` and `/v1/enhance`.

## 3. Backend/profile integration

- **Saved candidate applied verbatim.** `Planning/R5-8_candidate/r5_8_candidate.patch` was applied with `git apply`
  and not edited. It adds:
  - `configs/phase5/r5_8_nafnet_candidate.json` (sealed, route 1.1.0)
  - the local-patch guard in `nafnet_route.py`
  - `models.nafnet_factory_pad16`
  - out-of-image reflect fill in `pipeline.py`
  - `tests/test_r5_8_candidate.py`
- The patch's process-level switch `FES_NAFNET_PROFILE` (default `r5_6`) is kept as it was.
- **Per-request profile (new).** `/v1/enhance` accepts a form field `profile`:

  | `profile` | Behaviour |
  |---|---|
  | `""` (omitted) | Service default, **unchanged**. No `profile` key in the response. |
  | `original` | The input is validated (limits/decode). The **input bytes** are returned. No detection, gate or model runs. Reason `profile_original`. |
  | `r5_6` | The default live route, explicitly named. |
  | `r5_8` | The saved candidate. Uses its own sealed config and its **own** NAFNet worker (`nafnet_factory_pad16`). |
  | anything else | **422 REJECTED** `unknown profile`. No model is called. |

- **R5-8 is opt-in at startup.**
  - `r5_8` is servable only when the service is started with `FES_EXPERIMENTAL_PROFILES=r5_8`. This loads a second
    NAFNet worker (≈1.2 GB).
  - Without the flag, `profile=r5_8` returns **422** ("not enabled…").
  - An unknown flag value makes the service report not-ready (it fails closed).
  - The pipeline refuses to add r5_8 unless the default profile is r5_6.
- **Other request rules:**
  - Combining `profile` with `force_a4` is rejected.
  - `route_mode` keeps the service default (`auto`).
- **`/ready` additions** (additive only):
  - `profiles` (for example `["original","r5_6","r5_8"]`)
  - `default_profile` (`r5_6`)
  - `experimental_models`
  - seal `nafnet_route_r5_8`. This appears only when r5_8 is enabled; `seals.nafnet_route` is still the live seal.
- **Implementation.** `EnhancementPipeline` holds a `profiles` map: name → (sealed config, model, oob_fill). The
  NAFNet path reads its config and model from the request's profile instead of `self`. With no profile it resolves to
  the same objects as before.

## 4. Original mode

- Returns the uploaded bytes exactly: a JPEG stays the same JPEG, with no re-encode. This was verified by byte
  comparison in the tests and over HTTP.
- `model = null`, `used = B0` (ORIGINAL), and no worker is called (spy counts are 0).
- The processing time is reported (about 1–40 ms).
- In the UI the Original panel is **always visible** as soon as an image is selected. It also shows the face crop the
  service analysed, so the NAFNet outputs (which are face crops) can be compared like for like.

## 5. R5-6 reference mode

- `profile=r5_6` is the live route:
  - config `r5_nafnet_route.json`, seal `eb6fe101…`, route `p5-nafnet-route-1.0.0`
  - the unmodified `nafnet_factory`
  - black-padded chip, and the A4 strong-noise branch of `auto`
- Tests assert that its image and metadata are **identical** to a default request. They also assert that a default
  request on a pipeline *with* r5_8 enabled is identical to one on a pipeline *without* it, in face_chip mode,
  full_image mode and the edge-chip case.
- UI label: **REFERENCE — NOT PRODUCTION-APPROVED**.

## 6. R5-8 experimental mode

`profile=r5_8` runs the saved candidate exactly as evaluated:

- route `p5-nafnet-route-1.1.0`
- the R5-7 guard values plus chroma p999
- the local "dottedness" detector (`l_dot_c_max ≤ 24.45`)
- out-of-image reflect fill
- reflect-pad16 NAFNet

A test asserts that the per-request `r5_8` result equals the process-level `FES_NAFNET_PROFILE=r5_8` pipeline (the
R5-8 evaluation setup) in pixels and metadata. The UI shows:

- route and version
- the gate decision
- the global guard's own verdict
- the local detector's verdict with `l_dot_c_max`
- the final output and fallback reason

UI label: **EXPERIMENTAL — NOT PRODUCTION-APPROVED**.

## 7. Side-by-side comparison

- Three columns (stacked on narrow screens): ORIGINAL | NAFNet R5-6 | NAFNet R5-8.
- **Compare all three** runs the requests one after another, because the service handles one request at a time.
- **Run selected mode** uses the Mode selector (Original / R5-6 / R5-8) to fill one column.
- Results stay per image. A new image or **Reset** clears them.
- A profile that returned the original gets:
  - a dashed frame
  - a stamp: **"ORIGINAL — enhancement not applied"**, red with the fallback code when a guard or model failure
    caused it
  - a grey or red panel border
- Only a real NAFNet output gets a green border. An A4 output from the strong-noise branch is amber and labelled
  "not NAFNet".

## 8. Metadata/fallback display

Each panel shows:

- Profile and model invoked
- Route
- Enhancement applied YES/NO and final output (NAFNet / A4 / ORIGINAL)
- Gate decision (ELIGIBLE / NOT ELIGIBLE with the reason code)
- Artifact guard verdict with reasons
- Local artifact detector (R5-8 only)
- Fallback code with a plain-language explanation and the guard reason codes
- Route version and NAFNet inference time
- Server time and round-trip time
- HTTP/API status and request ID

Errors are clean, one-line messages: 422 (with a hint when R5-8 is not enabled), 503 BUSY / NOT_READY, network
failure and timeout (180 s). No stack trace is shown; the worker error strings are already truncated by the service.

## 9. Manual evaluation mechanism

For each NAFNet profile the owner can record:

- a rating: **Better / Same / Worse / Unsafe / Artifact**
- tags: blur improved, identity appears preserved, rainbow/dotted artifact, grid/padding artifact, hallucinated
  detail, grayscale changed, face geometry changed, other
- notes

Per image there is also a category from the §12 list and an overall note.

- **Save evaluation** is enabled once at least one NAFNet profile has both a result and a rating.
- Records are kept **in the browser tab's memory only**. There is no database, no localStorage and no server storage.
- **Export JSON** and **Export CSV** produce local downloads. The CSV neutralises spreadsheet formulas typed into
  notes.
- If the log has not been exported, a `beforeunload` warning appears and **Clear log** asks twice.
- Exports contain no pixels. They do contain the **file name**, which may be personal, so keep exports out of git.
- Nothing computes or claims Feature-Match or any identity score.

## 10. Tests

| Suite | Result |
|---|---|
| `tests/test_r5_9_profiles.py` (new: default unchanged, r5_6 = default, r5_8 = saved candidate, original bytes/no model, invalid profile, profile+force_a4, not-enabled 422, fail-closed flag, API, page served/CSP/labels/no inline code, node hook) | pass |
| `tests/test_r5_8_candidate.py` (from the saved patch) | pass |
| Existing: `test_r5_ui.py`, `test_r5_service.py`, `test_r5_nafnet.py`, `test_r5_rwgate.py`, `test_r5_fm_harness.py` | pass |
| Combined (fes-dev:cpu-p5) | **163 passed, 7 skipped** (the skips are real-model and node tests, which need env/node) |
| Real NAFNet (`FES_RUN_NAFNET=1`, `-k real`, including the pad16 factory) | **5 passed** |
| Node: `tests/ui/fes_ui.test.mjs` (existing) + `tests/ui/fes_compare.test.mjs` (new) | **31/31 pass** |

The 19,290-image archive evaluation was **not** rerun.

## 11. Manual smoke-test results

Setup: live service on `127.0.0.1:8080` (container `fes-ui`, real A4 + NAFNet weights, `FES_EXPERIMENTAL_PROFILES=r5_8`).
Inputs are DEV and archive images only, with no personal photos. All requests use `input_mode=full_image`.

| Input | original | r5_6 | r5_8 | default |
|---|---|---|---|---|
| DEV strong blur on a 464 px canvas | bytes identical, 1–38 ms | **ENHANCED** NAFNet, guard PASS, 4.2 s | **ENHANCED** NAFNet, guard + local PASS, 3.2 s | = r5_6 |
| DEV strong blur, tight 224 px crop | ORIGINAL | ORIGINAL, NOT_BLUR (known black-padding framing issue) | **ENHANCED** (reflect fill) | = r5_6 |
| DEV clean reference | ORIGINAL | ORIGINAL, NOT_BLUR | ORIGINAL, NOT_BLUR | = r5_6 |
| archive test/image_82776 | ORIGINAL | **fallback ARTIFACT_GUARD** (chroma p99) | ENHANCED | — |
| archive test/image_83137 | ORIGINAL | **fallback ARTIFACT_GUARD** | **fallback LOCAL_ARTIFACT_GUARD** (l_dot_c_max) | — |
| archive test/image_84471 | ORIGINAL | ORIGINAL, NOT_BLUR | **fallback LOCAL_ARTIFACT_GUARD** | — |
| `profile=r5_9` | 422 `unknown profile` | | | |
| garbage bytes, `profile=r5_8` | 422 REJECTED | | | |

- The archive outcomes match the recorded R5-6/R5-8 evaluation rows (`data/manifests/phase5_rapid/r5_6/*`).
- **Browser check** (Chrome, `/ui/compare.html`) on image_83137:
  - the loading state, step counter, elapsed time and disabled buttons all appeared during the run
  - all three panels rendered, and both fallbacks carried the red "ORIGINAL — enhancement not applied" stamp
  - the evaluation was saved and the log updated
  - Reset cleared the image and panels and kept the log
  - `/ui/` (the A4 page) is unchanged apart from the link, with no console errors
- File export was not clicked in the automated check, because it would trigger a browser download.

## 12. Files changed

**Saved R5-8 candidate, applied unchanged from `Planning/R5-8_candidate/r5_8_candidate.patch`:**
- `configs/phase5/r5_8_nafnet_candidate.json` (new)
- `src/fes_service/models.py`
- `src/fes_service/nafnet_route.py`
- `src/fes_service/pipeline.py` (candidate parts)
- `src/fes_service/app.py` (candidate parts)
- `tests/test_r5_8_candidate.py` (new)

**R5-9:**
- `src/fes_service/pipeline.py`: `profile` per request, `original` profile, `experimental_nafnet`, profiles map, `profile` metadata key only when requested
- `src/fes_service/app.py`: `profile` form field, `FES_EXPERIMENTAL_PROFILES`, `FES_NAFNET_R5_8_FACTORY`, `/ready` profiles
- `src/fes_service/ui/compare.html`, `compare.css`, `compare.js`, `fes_compare.js` (new)
- `src/fes_service/ui/index.html` (one link)
- `tests/test_r5_9_profiles.py`, `tests/ui/fes_compare.test.mjs` (new)
- this report

**Not touched:**
- `configs/phase5/r5_nafnet_route.json`, `r5_rwgate.json`, `configs/phase4/*`
- `models/`
- `worker.py`, `rwgate.py`
- `Planning/R5-7_candidate`, `Planning/R5-8_candidate`
- all R5-6/R5-8 evidence manifests

## 13. Model/gate/routing behaviour NOT changed

The following were not edited: thresholds, sealed configs, the T6/T7 and RW gates, the NAFNet input gate, the global
artifact guard, the local-patch detector, weights, pre- and post-processing, the timeout policy and the fallback
semantics. The only code in R5-9 picks *which existing, sealed profile* serves a request.

## 14. Default behaviour NOT changed

- A request without `profile` returns the same image and the same metadata (no new keys) as before.
- `FES_NAFNET_PROFILE` still defaults to `r5_6`.
- `FES_EXPERIMENTAL_PROFILES` defaults to off, so r5_8 cannot be reached by default (422).
- The tests assert all of this. Note: the local demo container `fes-ui` was restarted **with**
  `FES_EXPERIMENTAL_PROFILES=r5_8` for the owner's evaluation. Default requests on it are still R5-6.

## 15. Known limitations

- **Face crop vs full photo.** NAFNet outputs are face crops, so the Original column shows the full photo plus the
  analysed face crop. Judge NAFNet against the crop.
- **Unenhanced columns show the full photo.** When a profile returns the original, that column shows the full
  uploaded photo, not the crop.
- **Memory.** The second NAFNet worker adds about 1.2 GB.
- **One request at a time.** Requests are serialised, so a full comparison takes one NAFNet run per profile (about
  1–5 s each on CPU, plus A4 for strong noise: up to about 60 s).
- **Records are lost on reload.** The evaluation log lives only in the tab until it is exported.
- **No API combination of `force_a4` + profile.** The A4 page still offers forced A4 on its own.
- **No real-image evidence yet.** The R5-6 tight-crop NOT_BLUR behaviour (§11) is the known R5-5/R5-6 framing
  limitation. It is shown as it is, not fixed.

## 16. Recommendation / next step

**For the owner:**
- Run the 30–50 real images through `/ui/compare.html` across the §12 categories, then export the CSV.
- Decide from that:
  - NAFNet promising → actual Feature-Match validation (R5-4 harness: needs access + ≥ 20 identities)
  - not useful → close enhancement
  - useful but artifacts unacceptable → decide on a future safety/integration project

**Not done in R5-9, and nothing started automatically:**
- Feature-Match work
- tuning
- promotion

**Start command** for the evaluation service:

```
docker run -d --rm --name fes-ui -p 127.0.0.1:8080:8080 -e PYTHONPATH=src -e FES_EXPERIMENTAL_PROFILES=r5_8 \
  -v "D:\ReBIT\Face-Enhancement:/workspace" -v "D:\ReBIT\Face-Enhancement\models:/workspace/models:ro" \
  -w /workspace fes-dev:cpu-p5 uvicorn fes_service.app:app --host 0.0.0.0 --port 8080
```

Then open http://127.0.0.1:8080/ui/compare.html. Omit `FES_EXPERIMENTAL_PROFILES` for the plain default service.

### Acceptance criteria

| # | Criterion | Met |
|---|---|---|
| 1–3 | original / R5-6 / R5-8 modes work | ✔ (§11) |
| 4 | R5-8 not reachable by default | ✔ (422; tests) |
| 5–8 | original always visible; direct comparison; fallbacks distinct; warnings visible | ✔ (§7, browser check) |
| 9 | loading state | ✔ |
| 10–11 | UI + backend tests | ✔ (§10) |
| 12 | real smoke for all three modes | ✔ (§11) |
| 13–15 | no weight, gate or threshold changes; routing only by explicit profile | ✔ (§13) |
| 16–18 | no Feature-Match, no large benchmark, no network/telemetry/storage | ✔ |
| 19 | default behaviour unchanged | ✔ (§14) |
| 20 | report | ✔ |
