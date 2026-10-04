// R5-UI logic tests (Node ≥ 18, no dependencies):  node --test tests/ui/fes_ui.test.mjs
// Also run from pytest by tests/test_r5_ui.py when `node` is on PATH.
import assert from "node:assert/strict";
import { test } from "node:test";

import {
  DEFAULT_LIMITS, PROCESSING_MESSAGE, REQUEST_TIMEOUT_MS, buildEnhanceForm, explainReason, interpretNetworkError,
  interpretReady, interpretResponse, sniffMime, validateDimensions, validateFile,
} from "../../src/fes_service/ui/fes_ui.js";

const PNG_B64 = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0, 0, 0, 13]).toString("base64");
const JPG_B64 = Buffer.from([0xff, 0xd8, 0xff, 0xe0, 0, 16, 0x4a, 0x46, 0x49, 0x46]).toString("base64");
const base = { service_version: "fes-service-r5-0.1.0", request_id: "abc", input_mode: "full_image", warnings: [],
  n_faces: 1, chip_px: 180, timings_ms: { total: 61234.5 }, quality: { det_score: 0.93, iod_px: 60.2, noise_sigma: 1.1,
    lapd1_c: 0.0031, is_gray: false } };

// ------------------------------------------------------------------ upload validation
test("accepts supported formats within the size limit", () => {
  for (const type of ["image/jpeg", "image/png", "image/webp", "image/bmp"]) {
    assert.equal(validateFile({ name: "a", size: 1000, type }).ok, true, type);
  }
  assert.equal(validateFile({ name: "face.JPG", size: 10, type: "" }).ok, true);   // OS gave no MIME type
});

test("rejects unsupported, empty, missing and oversized files", () => {
  assert.match(validateFile({ name: "a.gif", size: 10, type: "image/gif" }).error, /Unsupported/);
  assert.match(validateFile({ name: "a.pdf", size: 10, type: "application/pdf" }).error, /Unsupported/);
  assert.match(validateFile({ name: "a.png", size: 0, type: "image/png" }).error, /empty/);
  assert.match(validateFile(null).error, /No file/);
  const big = { name: "a.png", size: DEFAULT_LIMITS.max_bytes + 1, type: "image/png" };
  assert.match(validateFile(big).error, /too large.*20\.0 MB/);
  assert.equal(validateFile({ ...big, size: 2000 }, { max_bytes: 1000 }).ok, false);  // live /ready limit wins
});

test("dimension checks mirror the backend limits", () => {
  assert.equal(validateDimensions(640, 480).ok, true);
  assert.match(validateDimensions(4, 400).error, /too small/);
  assert.match(validateDimensions(100, 1100).error, /aspect/);
  assert.match(validateDimensions(8000, 6000).error, /too many pixels/);
  assert.match(validateDimensions(0, 0).error, /could not be read/);
});

// ------------------------------------------------------------------ API request
test("request uses the existing /v1/enhance multipart contract", () => {
  const file = new File([new Uint8Array([1, 2, 3])], "face.png", { type: "image/png" });
  const fd = buildEnhanceForm(file);
  assert.equal(fd.get("input_mode"), "full_image");
  assert.equal(fd.get("return_image"), "true");
  assert.equal(fd.get("image").name, "face.png");
  assert.equal(fd.get("image").size, 3);
  assert.deepEqual([...fd.keys()].sort(), ["image", "input_mode", "return_image"]);
});

test("timeout covers the backend A4 timeout and the loading text is truthful", () => {
  assert.ok(REQUEST_TIMEOUT_MS >= 150_000);
  assert.match(PROCESSING_MESSAGE, /up to a minute on CPU/);
  assert.doesNotMatch(PROCESSING_MESSAGE, /%/);
});

// ------------------------------------------------------------------ responses
test("ENHANCED response shows the returned PNG and the backend decision", () => {
  const v = interpretResponse(200, { ...base, status: "ENHANCED", enhanced: true, decision: "ENHANCE_CANDIDATE",
    reason: "strong_blur", bucket: "degraded_recognizable", route: "A4", used: "A4", fallback: "", escalate: false,
    image_format: "png", image_b64: PNG_B64 });
  assert.equal(v.kind, "ENHANCED");
  assert.equal(v.tone, "ok");
  assert.equal(v.imageSrc, `data:image/png;base64,${PNG_B64}`);
  assert.equal(v.decision, "ENHANCE_CANDIDATE");
  assert.equal(v.reason, "strong_blur");
  assert.match(v.reasonText, /Strong blur/);
  assert.match(v.outputNote, /224×224 face crop/);
  const d = Object.fromEntries(v.details);
  assert.equal(d.Status, "ENHANCED");
  assert.equal(d["Route / used"], "A4 / A4");
  assert.equal(d["Server time"], "61.2 s");
  assert.equal(d["Detector score"], "0.930");
});

test("ORIGINAL (not enhanced) returns the input bytes with the sniffed type", () => {
  const v = interpretResponse(200, { ...base, status: "ORIGINAL", decision: "DO_NOT_ENHANCE", reason: "acceptable",
    bucket: "acceptable", route: "B0", used: "B0", fallback: "", escalate: false, image_format: "as_received",
    image_b64: JPG_B64 });
  assert.equal(v.kind, "ORIGINAL");
  assert.equal(v.tone, "neutral");
  assert.match(v.title, /not enhanced/);
  assert.ok(v.imageSrc.startsWith("data:image/jpeg;base64,"));
  assert.match(v.outputNote, /unchanged/);
});

test("UNSAFE / no-face is shown as not processed with the backend reason", () => {
  const v = interpretResponse(200, { ...base, status: "ORIGINAL", decision: "UNSAFE_OR_UNCERTAIN",
    reason: "no_valid_face:NO_FACE_DETECTED", route: "B0", used: "B0", escalate: true, n_faces: 0,
    quality: {}, image_format: "as_received", image_b64: PNG_B64 });
  assert.equal(v.kind, "ORIGINAL");
  assert.equal(v.tone, "warn");
  assert.match(v.title, /unsafe/);
  assert.match(v.reasonText, /No face was found/);
  assert.equal(Object.fromEntries(v.details).Escalate, "yes");
});

test("A4 fallback (timeout / failure) is reported, not hidden", () => {
  const v = interpretResponse(200, { ...base, status: "ORIGINAL", decision: "ENHANCE_CANDIDATE", reason: "strong_noise",
    route: "A4", used: "B0", fallback: "TIMEOUT", image_format: "as_received", image_b64: PNG_B64 });
  assert.match(v.title, /Original returned/);
  assert.match(v.message, /ran out of time/);
  assert.equal(Object.fromEntries(v.details).Fallback, "TIMEOUT");
});

test("REJECTED, BUSY, NOT_READY and malformed responses become friendly errors", () => {
  const rej = interpretResponse(422, { status: "REJECTED", error: "not a decodable image", image_b64: null });
  assert.equal(rej.kind, "REJECTED");
  assert.match(rej.message, /not a decodable image/);
  assert.equal(rej.imageSrc, undefined);
  assert.equal(interpretResponse(503, { status: "BUSY", error: "x" }).kind, "BUSY");
  assert.equal(interpretResponse(503, { status: "NOT_READY", error: "x" }).kind, "NOT_READY");
  assert.match(interpretResponse(500, null).title, /Service error/);
  assert.match(interpretResponse(200, null).title, /Unexpected/);
  assert.match(interpretResponse(200, { foo: 1 }).title, /Unexpected/);
  assert.match(interpretResponse(200, { status: "WEIRD", image_b64: PNG_B64 }).message, /Unknown status "WEIRD"/);
  assert.match(interpretResponse(200, { status: "ENHANCED", image_b64: null }).message, /did not return an image/);
  for (const v of [rej, interpretResponse(500, null)]) assert.doesNotMatch(v.message, /Traceback|at .*\.js/);
});

test("network failure and timeout messages", () => {
  assert.match(interpretNetworkError(new TypeError("Failed to fetch")).title, /unavailable/);
  assert.match(interpretNetworkError(new Error("aborted"), true).title, /timed out/);
});

test("unknown reasons are shown raw without an invented explanation", () => {
  assert.equal(explainReason("something_new"), "");
  assert.match(explainReason("no_valid_face:FACE_TOO_SMALL"), /too small/);
  assert.match(explainReason("no_valid_face:LANDMARK_FAILURE"), /No usable face/);
  assert.match(explainReason("degraded_no_a4_evidence:blur+noise"), /Mildly degraded/);
});

test("sniffMime", () => {
  assert.equal(sniffMime(PNG_B64), "image/png");
  assert.equal(sniffMime(JPG_B64), "image/jpeg");
  assert.equal(sniffMime(Buffer.from("RIFF\0\0\0\0WEBPVP8 ").toString("base64")), "image/webp");
  assert.equal(sniffMime(Buffer.from("BM\0\0\0\0").toString("base64")), "image/bmp");
  assert.equal(sniffMime("!!!"), "");
});

// ------------------------------------------------------------------ health
test("health from /ready", () => {
  const lim = { max_bytes: 1000, max_pixels: 10, min_side: 8, max_aspect: 10 };
  const on = interpretReady(200, { ready: true, limits: lim });
  assert.equal(on.state, "online");
  assert.equal(on.label, "Enhancement Service Online");
  assert.equal(on.limits.max_bytes, 1000);
  assert.equal(interpretReady(503, { ready: false, error: "" }).state, "loading");
  assert.equal(interpretReady(503, { ready: false, error: "RuntimeError: load" }).state, "offline");
  const off = interpretReady(0, null);
  assert.equal(off.label, "Enhancement Service Offline");
  assert.equal(off.limits, null);
  assert.equal(interpretReady(200, { ready: true }).limits, null);        // older backend without `limits`
});
