// R5-UI — pure logic of the demo page (no DOM): input checks, request building, response interpretation.
// Every status, decision and reason shown comes from the backend (/v1/enhance, /ready); this file only
// adds plain-language explanations next to them. Unknown values are shown as received, never guessed.

export const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/webp", "image/bmp"];
export const ACCEPTED_EXT = [".jpg", ".jpeg", ".png", ".webp", ".bmp"];
// the pipeline's own defaults (fes_service.pipeline.Limits) — used only until /ready reports the live ones
export const DEFAULT_LIMITS = { max_bytes: 20 * 1024 * 1024, max_pixels: 40_000_000, min_side: 8, max_aspect: 10.0 };
// backend: A4 hard timeout 120 s (FES_A4_TIMEOUT_S) + detection/gate; the browser waits a little longer than that
export const REQUEST_TIMEOUT_MS = 180_000;
export const PROCESSING_MESSAGE = "Processing image… Enhancement may take up to a minute on CPU.";

export function formatBytes(n) {
  if (!Number.isFinite(n)) return "—";
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

export function formatMs(ms) {
  if (!Number.isFinite(ms)) return "—";
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`;
}

/** File-level checks before upload. `file` needs {name, size, type}. → {ok, error} */
export function validateFile(file, limits = DEFAULT_LIMITS) {
  if (!file) return { ok: false, error: "No file selected." };
  const name = String(file.name || "").toLowerCase();
  const typeOk = ACCEPTED_TYPES.includes(file.type);
  const extOk = ACCEPTED_EXT.some((e) => name.endsWith(e));
  if (!typeOk && !(extOk && !file.type)) {
    return { ok: false, error: "Unsupported file type. Please choose a JPEG, PNG, WebP or BMP image." };
  }
  if (!file.size) return { ok: false, error: "The selected file is empty." };
  const max = (limits && limits.max_bytes) || DEFAULT_LIMITS.max_bytes;
  if (file.size > max) {
    return { ok: false, error: `File is too large (${formatBytes(file.size)}). The service accepts up to ${formatBytes(max)}.` };
  }
  return { ok: true, error: "" };
}

/** Pixel-level checks once the browser has decoded the preview (mirrors the backend's limits). */
export function validateDimensions(w, h, limits = DEFAULT_LIMITS) {
  const L = { ...DEFAULT_LIMITS, ...(limits || {}) };
  if (!w || !h) return { ok: false, error: "The image could not be read. It may be corrupted." };
  if (Math.min(w, h) < L.min_side) return { ok: false, error: `Image is too small (${w}×${h}).` };
  if (w * h > L.max_pixels) return { ok: false, error: `Image has too many pixels (${w}×${h}). Limit: ${(L.max_pixels / 1e6).toFixed(0)} MP.` };
  if (Math.max(w, h) / Math.min(w, h) > L.max_aspect) return { ok: false, error: `Image aspect ratio is too extreme (${w}×${h}).` };
  return { ok: true, error: "" };
}

/** The multipart body for POST /v1/enhance (existing contract: image, input_mode, return_image). */
export function buildEnhanceForm(file, FormDataCtor = FormData) {
  const fd = new FormDataCtor();
  fd.append("image", file, file.name || "upload");
  fd.append("input_mode", "full_image");
  fd.append("return_image", "true");
  return fd;
}

/** MIME type from the first bytes of a base64 image (ORIGINAL returns the input bytes as received). */
export function sniffMime(b64) {
  let head = "";
  try {
    head = atob(String(b64).slice(0, 24));
  } catch {
    return "";
  }
  const c = (i) => head.charCodeAt(i);
  if (c(0) === 0x89 && head.slice(1, 4) === "PNG") return "image/png";
  if (c(0) === 0xff && c(1) === 0xd8) return "image/jpeg";
  if (head.slice(0, 4) === "RIFF" && head.slice(8, 12) === "WEBP") return "image/webp";
  if (head.slice(0, 2) === "BM") return "image/bmp";
  return "";
}

// plain-language notes for the backend's reason codes (the code itself is always displayed as well)
const REASONS = [
  [/^no_valid_face:FACE_TOO_SMALL/, "A face was found, but it is too small to process safely."],
  [/^no_valid_face:NO_FACE_DETECTED/, "No face was found in the image."],
  [/^no_valid_face/, "No usable face could be extracted from the image."],
  [/^alignment_unreliable/, "The face pose or landmarks were too uncertain to process safely."],
  [/^low_detection_confidence$/, "The face detector was not confident enough to process safely."],
  [/^quality_analysis_failed$/, "Quality analysis failed, so the image was not enhanced."],
  [/^insufficient_face_size$/, "The face is too small to be enhanced safely."],
  [/^beyond_evidence_extreme_blur$/, "The face is blurrier than anything the model was validated on."],
  [/^photometric_no_a4_evidence$/, "Lighting/contrast problem: the model has no proven benefit for this."],
  [/^compression_no_a4_evidence$/, "Compression artefacts: the model has no proven benefit for this."],
  [/^strong_noise$/, "Strong noise detected — inside the range the model was validated on."],
  [/^strong_blur$/, "Strong blur detected — inside the range the model was validated on."],
  [/^degraded_below_a4_size/, "Degraded, but the face is too small for the enhancement model."],
  [/^degraded_no_a4_evidence/, "Mildly degraded, outside the range where the model was shown to help."],
  [/^acceptable$/, "Quality already looks acceptable — no enhancement needed."],
];
const FALLBACKS = {
  TIMEOUT: "The enhancement model ran out of time; the original was returned.",
  MODEL_FAILURE: "The enhancement model failed; the original was returned.",
  RESOURCE_FAILURE: "The enhancement worker ran out of resources; the original was returned.",
  INVALID_OUTPUT: "The model output failed the safety check (no face found in it); the original was returned.",
};

export function explainReason(reason) {
  for (const [re, txt] of REASONS) if (re.test(reason || "")) return txt;
  return "";
}

const tone = (status, decision) =>
  status === "ENHANCED" ? "ok" : status === "ORIGINAL" ? (decision === "UNSAFE_OR_UNCERTAIN" ? "warn" : "neutral") : "bad";

/**
 * (HTTP status, parsed JSON body | null) → view model:
 *   {kind, tone, title, message, decision, reason, reasonText, fallback, imageSrc, outputNote, details:[[k,v]], warnings}
 * kind is the backend `status` (ENHANCED | ORIGINAL | REJECTED | BUSY | NOT_READY) or ERROR for anything else.
 */
export function interpretResponse(httpStatus, body) {
  const err = (title, message) => ({ kind: "ERROR", tone: "bad", title, message, details: [], warnings: [] });
  if (!body || typeof body !== "object" || typeof body.status !== "string") {
    if (httpStatus >= 500) return err("Service error", `The enhancement service returned an error (HTTP ${httpStatus}). Please try again.`);
    return err("Unexpected response", "The service returned a response this page does not understand.");
  }
  const s = body.status;
  if (s === "BUSY") return { ...err("Service busy", "The service is processing other images. Please wait a moment and try again."), kind: "BUSY", tone: "warn" };
  if (s === "NOT_READY") return { ...err("Service not ready", "The enhancement model is still loading or failed to load. Please try again shortly."), kind: "NOT_READY", tone: "warn" };
  if (s === "REJECTED") {
    return { ...err("Image rejected", `The service could not accept this image${body.error ? `: ${body.error}` : "."}`), kind: "REJECTED" };
  }
  if (s !== "ENHANCED" && s !== "ORIGINAL") return err("Unexpected response", `Unknown status "${s}" from the service.`);
  if (!body.image_b64) return err("Unexpected response", "The service did not return an image.");

  const mime = body.image_format === "png" ? "image/png" : sniffMime(body.image_b64) || "image/png";
  const decision = body.decision || "";
  const reason = body.reason || "";
  const fallback = body.fallback || "";
  let title, message;
  if (s === "ENHANCED") {
    title = "Enhanced";
    message = "The face was enhanced by the A4 model.";
  } else if (fallback) {
    title = "Original returned";
    message = FALLBACKS[fallback] || `Enhancement was attempted but fell back to the original (${fallback}).`;
  } else if (decision === "UNSAFE_OR_UNCERTAIN") {
    title = "Not processed (unsafe / uncertain)";
    message = "The service did not enhance this image because it could not do so safely. The original is returned.";
  } else {
    title = "Original returned (not enhanced)";
    message = "The quality gate decided enhancement is not needed or not proven to help for this image.";
  }
  const q = body.quality || {};
  const t = body.timings_ms || {};
  const num = (v, d = 3) => (Number.isFinite(v) ? Number(v).toFixed(d) : null);
  const details = [
    ["Status", s],
    ["Decision", decision],
    ["Reason", reason],
    ["Category", body.bucket],
    ["Route / used", body.route || body.used ? `${body.route || "—"} / ${body.used || "—"}` : ""],
    ["Fallback", fallback],
    ["Escalate", body.escalate === true ? "yes" : body.escalate === false ? "no" : ""],
    ["Faces detected", Number.isFinite(body.n_faces) ? String(body.n_faces) : ""],
    ["Face crop size", body.chip_px ? `${body.chip_px} px` : ""],
    ["Detector score", num(q.det_score)],
    ["Eye distance (IOD)", Number.isFinite(q.iod_px) ? `${num(q.iod_px, 1)} px` : null],
    ["Noise σ", num(q.noise_sigma)],
    ["Sharpness (lapd1_c)", num(q.lapd1_c, 4)],
    ["Grayscale input", q.is_gray === true || q.is_gray === 1 ? "yes" : q.is_gray === false || q.is_gray === 0 ? "no" : null],
    ["Server time", Number.isFinite(t.total) ? formatMs(t.total) : ""],
    ["Service version", body.service_version],
    ["Request ID", body.request_id],
  ].filter(([, v]) => v !== null && v !== undefined && v !== "");
  return {
    kind: s, tone: tone(s, decision), title, message, decision, reason, reasonText: explainReason(reason), fallback,
    imageSrc: `data:${mime};base64,${body.image_b64}`,
    outputNote: s === "ENHANCED"
      ? "Processed = the enhanced 224×224 face crop returned by the service (not the full photo)."
      : "Processed = the original image bytes returned unchanged by the service.",
    details, warnings: Array.isArray(body.warnings) ? body.warnings : [],
  };
}

/** fetch/abort failure → friendly view model (no stack traces). */
export function interpretNetworkError(e, timedOut = false) {
  if (timedOut) {
    return { kind: "ERROR", tone: "bad", title: "Request timed out",
      message: `No answer from the service within ${REQUEST_TIMEOUT_MS / 1000} s. It may be overloaded; please try again.`, details: [], warnings: [] };
  }
  return { kind: "ERROR", tone: "bad", title: "Service unavailable",
    message: "Could not reach the enhancement service. Check that it is running, then try again.", details: [], warnings: [] };
}

/** (HTTP status | 0 for network failure, /ready body | null) → {state: online|loading|offline, label, limits} */
export function interpretReady(httpStatus, body) {
  const limits = body && body.limits && Number.isFinite(body.limits.max_bytes) ? { ...DEFAULT_LIMITS, ...body.limits } : null;
  if (httpStatus === 200 && body && body.ready === true) return { state: "online", label: "Enhancement Service Online", limits };
  if (httpStatus === 503 && body && body.ready === false) {
    return { state: body.error ? "offline" : "loading", label: body.error ? "Enhancement Service Not Ready" : "Enhancement Service Starting…", limits };
  }
  return { state: "offline", label: "Enhancement Service Offline", limits: null };
}
