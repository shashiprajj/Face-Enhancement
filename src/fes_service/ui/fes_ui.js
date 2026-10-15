// Shared pure helpers of the demo page (no DOM): input checks, formatting, reason texts, /ready interpretation.
// Every status, decision and reason shown comes from the backend; this file only adds plain-language explanations
// next to them. Unknown values are shown as received, never guessed.

export const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/webp", "image/bmp"];
export const ACCEPTED_EXT = [".jpg", ".jpeg", ".png", ".webp", ".bmp"];
// the pipeline's own defaults (fes_service.pipeline.Limits) — used only until /ready reports the live ones
export const DEFAULT_LIMITS = { max_bytes: 20 * 1024 * 1024, max_pixels: 40_000_000, min_side: 8, max_aspect: 10.0 };
// backend: model hard timeout 120 s (FES_NAFNET_TIMEOUT_S) + detection/gate; the browser waits a little longer
export const REQUEST_TIMEOUT_MS = 180_000;

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
  [/^NAFNET_ELIGIBLE_STRONG_LOW_NOISE_BLUR$/, "Strong, low-noise blur — the guarded NAFNet deblurring route applies."],
  [/^NAFNET_REJECT_HIGH_NOISE$/, "Blurred but grainy/noisy: NAFNet is not safe on noise, so the original is kept."],
  [/^NAFNET_REJECT_PIXELATION$/, "Pixelated / block-structured image: NAFNet is not safe on it."],
  [/^NAFNET_REJECT_HALFTONE$/, "Printed / halftone pattern detected: NAFNet is not safe on it."],
  [/^NAFNET_REJECT_LOW_FACE_SIZE$/, "The face is too small for the NAFNet route."],
  [/^NAFNET_REJECT_UNSAFE_ALIGNMENT$/, "The face pose or landmarks were too uncertain to process safely."],
  [/^NAFNET_REJECT_LOW_DETECTION$/, "The face detector was not confident enough to process safely."],
  [/^NAFNET_REJECT_NO_FACE$/, "No usable face was found."],
  [/^NAFNET_REJECT_NOT_BLUR$/, "Not blurred (sharp, or a different kind of degradation) — nothing for the deblurring route to do."],
  [/^NAFNET_REJECT_MILD_BLUR$/, "Only mildly blurred — outside the strong-blur range the NAFNet route covers."],
  [/^NAFNET_REJECT_TOO_BLURRED$/, "Too blurred to fix — blurrier than the \"give up above\" setting, so it is returned unchanged."],
  [/^NAFNET_REJECT_PHOTOMETRIC$/, "Lighting/contrast problem: NAFNet does not address it."],
  [/^NAFNET_REJECT_COMPRESSION$/, "Compression artefacts: NAFNet is not safe on them."],
  [/^NAFNET_REJECT_MISSING_FEATURE/, "A quality measurement could not be computed, so the image was not enhanced."],
];

export function explainReason(reason) {
  for (const [re, txt] of REASONS) if (re.test(reason || "")) return txt;
  return "";
}

export function interpretReady(httpStatus, body) {
  const limits = body && body.limits && Number.isFinite(body.limits.max_bytes) ? { ...DEFAULT_LIMITS, ...body.limits } : null;
  if (httpStatus === 200 && body && body.ready === true) return { state: "online", label: "Enhancement Service Online", limits };
  if (httpStatus === 503 && body && body.ready === false) {
    return { state: body.error ? "offline" : "loading", label: body.error ? "Enhancement Service Not Ready" : "Enhancement Service Starting…", limits };
  }
  return { state: "offline", label: "Enhancement Service Offline", limits: null };
}
