# P0-T2 — Input / Output Specification

| Field | Value |
|---|---|
| Phase / Task | Phase 0 / P0-T2 |
| Status | Complete |
| Date | 2026-09-28 |
| Depends on | P0-T1 (reviewed) |
| Nature | **Conceptual contract only** — no API implementation, no model-specific dimensions fixed |

---

## 1. Inputs from P0-T1 that constrain this contract

- FES returns an *image plus metadata*; never identity decisions, embeddings, or match scores (P0-T1 §3, X1–X3).
- Face detection/cropping/alignment are *supporting* capabilities (S7–S9).
- No consumer-specific fields (X6, X7). Contract is generic and versioned.
- Still images only (A2).

## 2. Input Contract

### 2.1 Request content (conceptual)

| Field | Required | Description |
|---|---|---|
| `image` | Yes | One encoded image (binary upload or base64 — transport decided in the API phase). |
| `input_mode` | No | `auto` (default), `full_image`, or `face_crop`. Tells FES whether to run detection. |
| `face_selection` | No | Policy for multiple faces: `largest` (default), `all`, `index`. See §6. |
| `output_mode` | No | `face_crop` (default), `full_image` (paste-back), or `both`. |
| `output_format` | No | `png` (default) or `jpeg` / `webp`. |
| `return_metadata` | No | Default `true`. |
| `enhancement_options` | No | Reserved for later (e.g., strength/fidelity trade-off). Not fixed in Phase 0. |

### 2.2 Supported / initially targeted formats

| Format | Initial target | Notes |
|---|---|---|
| JPEG (`.jpg/.jpeg`) | **Yes** | Most common source; main carrier of compression artifacts. |
| PNG | **Yes** | Lossless; alpha channel dropped (composited on neutral background or rejected — decide in API phase). |
| WebP | **Yes** | Common from web/messaging sources. |
| BMP | Candidate | Cheap to support; low priority. |
| TIFF | Deferred | Multi-page/high-bit-depth complexity. |
| HEIC/HEIF | Deferred | Licensing/codec dependency; revisit if consumers need it. |
| GIF / animated formats | **No** | Video/animation out of scope. First frame only is *not* offered to avoid ambiguity. |
| RAW camera formats | **No** | |

Format is detected from **file content (magic bytes)**, not the extension or declared MIME type.

### 2.3 Color / encoding

- Decoded internally to **8-bit sRGB, 3-channel**.
- Grayscale input accepted; converted to 3-channel for processing. Output remains 3-channel; metadata records `input_color_mode: grayscale`. (Whether to re-convert to grayscale on output is deferred.)
- 16-bit input: accepted and down-converted to 8-bit (deferred: whether to preserve depth).
- CMYK: converted to sRGB if the decoder supports it; otherwise rejected as `UNSUPPORTED_COLOR_MODE`.
- Embedded ICC profiles: converted to sRGB where possible.
- **EXIF orientation is applied** before processing, so faces are upright as the user sees them.
- EXIF and other metadata are **stripped from the output** (privacy; P0-T5).

### 2.4 Size / resolution policy

Model-specific input sizes are **not** fixed here (the model is not chosen). The policy is expressed as limits and categories whose values are finalized in Phase 1/2.

| Aspect | Policy | Value |
|---|---|---|
| Max encoded file size | Hard limit; reject above | **TBD** (working assumption ≈ 10–20 MB; set in API phase from P0-T5 resource limits) |
| Max decoded pixel count | Hard limit to prevent decompression bombs | **TBD** (working assumption ≈ 25–50 MP) |
| Max side length | Hard limit | **TBD** |
| Min image side | Reject if too small to contain a usable face | **TBD** (working assumption ≈ 32 px) |
| Min face size (detected face box) | Below threshold → process but flag `face_too_small`, or reject — set by evaluation (P0-T4) | **TBD** |
| Large images | Accepted up to limits; detection may run on a downscaled copy; restoration runs on the original-resolution face region | — |

Face-size **categories** used for reporting and evaluation (thresholds TBD in P0-T4 experiments):

- `tiny` — face too small for reliable restoration; results flagged as low-confidence.
- `small` — primary super-resolution target.
- `medium` — mostly deblur/denoise/artifact removal.
- `large` — may need *downscaling* to the model's working size; enhancement may bring little benefit.

### 2.5 Full image vs face crop — **DECISION**

| Option | Pros | Cons |
|---|---|---|
| A) Crop only | Simplest; no detector; smallest attack surface. | Pushes detection/alignment onto every consumer; inconsistent crops across consumers degrade model quality; restoration models are usually trained on *aligned* faces. |
| B) Full image only | Consistent detection + alignment; supports paste-back. | Wasteful and less robust when the consumer already has a tight crop (detector may fail on tight crops). |
| **C) Both** | Serves both consumer types; FES controls alignment in both cases. | Slightly more complex contract and testing. |

**Recommendation: C — accept both, with `input_mode = auto` as the default.**

Rationale:

1. Restoration quality depends heavily on consistent alignment, so FES should own detection/alignment rather than trusting arbitrary consumer crops.
2. Many consumers already hold a crop; forcing them to send a full image is not possible.
3. In `face_crop` mode FES still runs landmark detection *on the crop* to align it; if no face is found, it can fall back to treating the whole crop as the face (flagged in metadata) rather than failing.
4. `auto` mode: run detection; if one face covers most of the frame, treat as crop. The exact heuristic is deferred to implementation.

### 2.6 Face-count policy

| Situation | Default behavior | Metadata / error |
|---|---|---|
| 0 faces (full_image / auto) | Reject | `NO_FACE_DETECTED` |
| 0 faces (face_crop) | Process whole crop without alignment, flag | `warnings: [no_landmarks_found]` |
| 1 face | Process | — |
| >1 faces, `face_selection=largest` (default) | Enhance the largest (by box area; ties → highest detector confidence) | `faces_detected: N`, `faces_processed: 1` |
| >1 faces, `face_selection=all` | Enhance each up to a max (**TBD**, e.g., 10) | Faces beyond the max → `warnings: [face_limit_reached]` |
| >1 faces, `face_selection=index` | Enhance the chosen detection index | `INVALID_FACE_INDEX` if out of range |

Detector confidence below threshold (**TBD**) counts as no face.

## 3. Conceptual Processing Flow

```
            ┌──────────────┐
 request ──▶│ 1. Validate  │── size/format/pixel limits ──✗──▶ INVALID_INPUT / FILE_TOO_LARGE / UNSUPPORTED_FORMAT
            └──────┬───────┘
                   ▼
            ┌──────────────┐
            │ 2. Decode &  │  magic-byte format check, EXIF orientation,
            │   normalize  │  → 8-bit sRGB RGB; strip metadata   ──✗──▶ DECODE_ERROR
            └──────┬───────┘
                   ▼
            ┌──────────────┐
            │ 3. Input QA  │  no-reference quality score, degradation estimate
            └──────┬───────┘
                   ▼
            ┌──────────────┐
            │ 4. Detect    │  (full_image / auto)            ──✗──▶ NO_FACE_DETECTED
            │  + landmarks │  (face_crop → landmarks only)
            └──────┬───────┘
                   ▼
            ┌──────────────┐
            │ 5. Select    │  face_selection policy          ──✗──▶ INVALID_FACE_INDEX
            │   face(s)    │
            └──────┬───────┘
                   ▼
            ┌──────────────┐
            │ 6. Align &   │  similarity transform to canonical frame;
            │   crop       │  keep inverse transform for paste-back
            └──────┬───────┘
                   ▼
            ┌──────────────┐
            │ 7. Restore   │  model inference (model TBD in Phase 1)
            │              │                                 ──✗──▶ PROCESSING_ERROR / TIMEOUT / RESOURCE_EXHAUSTED
            └──────┬───────┘
                   ▼
            ┌──────────────┐
            │ 8. Output QA │  output quality score, change magnitude,
            │  & guardrails│  flags (e.g. excessive_change)
            └──────┬───────┘
                   ▼
            ┌──────────────┐
            │ 9. Compose   │  face crop and/or paste-back into full image
            │   & encode   │  (blend mask), encode output format
            └──────┬───────┘
                   ▼
            response: enhanced image(s) + metadata
```

## 4. Output Contract

### 4.1 Enhanced image

| Output | Description |
|---|---|
| **Enhanced face crop** (default) | Aligned, restored face at a **standard output size** (value TBD, set by chosen model; typical candidates 512×512). Square, canonical alignment. |
| **Full image with paste-back** (optional) | Original image with the restored face(s) warped back and blended. Background is not an optimization target (P0-T1 X13) — passed through unchanged by default. |
| Format | PNG default (lossless, avoids re-introducing compression artifacts). JPEG/WebP optional with a documented quality setting. |
| Color | 8-bit sRGB, 3 channels. |
| Metadata in file | None (EXIF stripped). |

### 4.2 Guarantees

- Output is always a **derived, enhanced image** — metadata states `enhanced: true` and the model/version used.
- The same input with the same service version and options should produce the same output (determinism target; exceptions documented if the chosen model is stochastic).
- If FES decides enhancement is unsafe or not beneficial (e.g., extreme degradation), it returns a status that says so rather than silently returning a hallucinated face (P0-T3).

## 5. Metadata Candidates (future API response)

Grouped by priority. *No Feature-Match-specific fields.*

**Must have**

| Field | Meaning |
|---|---|
| `request_id` | Unique ID for tracing. |
| `status` | `success`, `success_with_warnings`, `rejected`, `error`. |
| `service_version`, `api_version` | Versioning. |
| `model_name`, `model_version` | Which restoration model produced the output. |
| `faces_detected`, `faces_processed` | Counts. |
| `faces[]` | Per face: `bbox` (in input coordinates), `detector_confidence`, `landmarks` (optional), `face_size_category`. |
| `warnings[]` | e.g. `face_too_small`, `extreme_degradation`, `excessive_change`, `occlusion_suspected`, `unusual_pose`, `face_limit_reached`, `no_landmarks_found`. |
| `processing_time_ms` | Total, optionally per stage. |

**Should have**

| Field | Meaning |
|---|---|
| `input_quality_score`, `output_quality_score` | No-reference quality estimates (method TBD, P0-T4). |
| `degradation_estimate` | Estimated dominant degradations (e.g., `{blur: high, noise: low, jpeg: medium}`). |
| `input_dimensions`, `output_dimensions` | Pixels. |
| `input_format`, `input_color_mode` | Detected format/color mode. |
| `enhancement_confidence` / `fidelity_risk` | Estimate of how much the model had to "invent"; used to warn consumers. Method TBD. |

**Could have / deferred**

| Field | Meaning |
|---|---|
| `per_stage_timing` | Breakdown for diagnostics. |
| `change_map` | Optional heatmap of where the image changed most. |
| `alignment_transform` | For consumers that want to reproduce the paste-back. |

**Explicitly excluded**: identity embeddings, identity-similarity scores to external references, match results, any consumer/business field.

## 6. Error Categories (conceptual)

| Code | Class | Meaning | Retryable |
|---|---|---|---|
| `INVALID_INPUT` | Client | Missing/malformed request. | No |
| `UNSUPPORTED_FORMAT` | Client | Format not in supported list. | No |
| `UNSUPPORTED_COLOR_MODE` | Client | Color mode cannot be converted. | No |
| `FILE_TOO_LARGE` | Client | Exceeds encoded size limit. | No |
| `IMAGE_TOO_LARGE` / `IMAGE_TOO_SMALL` | Client | Pixel/side limits exceeded / not met. | No |
| `DECODE_ERROR` | Client | Corrupt or truncated image. | No |
| `NO_FACE_DETECTED` | Content | No face above confidence threshold. | No |
| `INVALID_FACE_INDEX` | Client | `face_selection=index` out of range. | No |
| `FACE_TOO_SMALL` | Content | Only if policy is set to reject (vs. warn). | No |
| `ENHANCEMENT_NOT_RECOMMENDED` | Content | Degradation so extreme that output would be unreliable; FES declines (policy per P0-T3). | No |
| `PROCESSING_ERROR` | Server | Unexpected failure in pipeline. | Yes |
| `TIMEOUT` | Server | Exceeded processing time limit. | Yes |
| `RESOURCE_EXHAUSTED` | Server | Out of memory/VRAM or queue full. | Yes (with backoff) |
| `SERVICE_UNAVAILABLE` | Server | Model not loaded / maintenance. | Yes |

Transport-level mapping (HTTP status codes) is deferred to the API phase.

## 7. Deferred Decisions

| ID | Decision | Deferred to | Depends on |
|---|---|---|---|
| D2-1 | Exact file-size, pixel-count, side-length limits | API phase | P0-T5 resource profile, load testing |
| D2-2 | Standard output face size | Phase 1 | Chosen model |
| D2-3 | Min face size: reject vs. warn, and threshold | Phase 1 | P0-T4 evaluation results |
| D2-4 | Detector confidence threshold, `auto`-mode crop heuristic | Phase 1/2 | Chosen detector |
| D2-5 | Max faces per request in `all` mode | API phase | Performance results |
| D2-6 | Transport (multipart vs base64), sync vs async, batch endpoint | API phase | Consumer needs |
| D2-7 | HEIC/TIFF support | Later | Consumer demand, codec licensing |
| D2-8 | Alpha-channel handling, grayscale/16-bit output preservation | API phase | Consumer needs |
| D2-9 | Quality-score and fidelity-risk methods | Phase 1 | P0-T4 |
| D2-10 | Enhancement strength/fidelity option exposure | Phase 1/2 | Chosen model capabilities |

## 8. Acceptance Criteria

| # | Criterion | Status |
|---|---|---|
| AC1 | P0-T1 reviewed and respected | ✅ §1 |
| AC2 | Input contract: formats, size/resolution, file size, color/encoding, full vs crop, face count | ✅ §2 |
| AC3 | Full-image vs crop options evaluated with recommendation | ✅ §2.5 (C: both) |
| AC4 | Conceptual processing flow with ASCII diagram | ✅ §3 |
| AC5 | Output contract | ✅ §4 |
| AC6 | Metadata candidates | ✅ §5 |
| AC7 | Error categories | ✅ §6 |
| AC8 | No model-specific dimensions fixed as final requirements | ✅ (all marked TBD) |
| AC9 | No FastAPI implementation; no Feature-Match fields | ✅ |
| AC10 | Deferred decisions listed | ✅ §7 |
