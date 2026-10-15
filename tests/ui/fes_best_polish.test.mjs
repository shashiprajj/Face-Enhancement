// Single-output page, UI polish: (1) a technical model failure is not worded as a safety rejection, (2) Enhance is
// disabled unless the service is known to be online.  node --test tests/ui/fes_best_polish.test.mjs
import assert from "node:assert/strict";
import { test } from "node:test";

import { FAILED_TITLE, enhanceEnabled, interpretBestResult, serviceNote } from "../../src/fes_service/ui/fes_best.js";
import { interpretReady } from "../../src/fes_service/ui/fes_ui.js";

const PNG_B64 = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0, 0, 0, 13]).toString("base64");
const JPG_B64 = Buffer.from([0xff, 0xd8, 0xff, 0xe0, 0, 16, 0x4a, 0x46, 0x49, 0x46]).toString("base64");
const base = { request_id: "r1", warnings: [], chip_px: 180, timings_ms: { total: 900 }, profile: "best",
  image_b64: PNG_B64, chip_b64: PNG_B64, image_format: "png", quality: { lapd1_c: 0.0031 } };
const nafnet = (over = {}) => ({ eligible: true, reason: "NAFNET_ELIGIBLE_STRONG_LOW_NOISE_BLUR", ...over });
const cand = (status, score = null, fallback = "") => ({ status, fallback, score, model: "x" });
const sel = (chosen, candidates) => ({ rule: "sharper_lapd1_c", chosen, candidates });
const row = (v, k) => (v.rows.find(([key]) => key.startsWith(k)) || [])[1];
const original = (candidates) => interpretBestResult(200, { ...base, status: "ORIGINAL", image_b64: JPG_B64,
  image_format: "as_received", nafnet: nafnet(), selection: sel(null, candidates) }, 900);
const enhanced = (other) => interpretBestResult(200, { ...base, status: "ENHANCED", used: "NAFNET_GOPRO",
  nafnet: nafnet({ profile: "r6" }), selection: sel("r6", { r6: cand("ENHANCED", 0.0061), r6_ft: other }) }, 2000);

// ------------------------------------------------------------------ technical failure vs safety rejection

test("both models fail technically: failure wording, original shown, not a safety rejection", () => {
  for (const [a, b] of [["MODEL_FAILURE", "MODEL_FAILURE"], ["TIMEOUT", "RESOURCE_FAILURE"], ["RESOURCE_FAILURE", "MODEL_FAILURE"]]) {
    const v = original({ r6: cand("ORIGINAL", null, a), r6_ft: cand("ORIGINAL", null, b) });
    assert.equal(v.title, FAILED_TITLE);
    assert.equal(v.title, "Enhancement failed — showing original image");
    assert.match(v.message, /technical failure, not a quality rejection/);
    assert.doesNotMatch(v.message, /safety checks/);
    assert.equal(v.tone, "bad"); assert.equal(v.applied, false); assert.equal(v.ok, true);
    assert.equal(v.leftSrc, ""); assert.match(v.imageSrc, /^data:image\/jpeg/);            // the original, as returned
    assert.equal(v.rightLabel, "ORIGINAL — not enhanced"); assert.equal(row(v, "Output chosen"), "none");
  }
  const v = original({ r6: cand("ORIGINAL", null, "MODEL_FAILURE"), r6_ft: cand("ORIGINAL", null, "TIMEOUT") });
  assert.equal(row(v, "R6 —"), "no result — failed"); assert.equal(row(v, "R6-FT"), "no result — ran out of time");
});

test("both outputs rejected by safety checks: the existing wording is kept", () => {
  for (const [a, b] of [["STABILITY_GUARD", "STABILITY_GUARD"], ["STABILITY_GUARD", "ARTIFACT_GUARD"],
    ["LOCAL_ARTIFACT_GUARD", "INVALID_OUTPUT"]]) {
    const v = original({ r6: cand("ORIGINAL", null, a), r6_ft: cand("ORIGINAL", null, b) });
    assert.equal(v.title, "Not enhanced — original returned"); assert.equal(v.tone, "warn");
    assert.equal(v.message, "The models ran, but no output passed the safety checks, so the original image is returned unchanged.");
    assert.match(v.imageSrc, /^data:image\/jpeg/); assert.equal(v.applied, false);
  }
});

test("one model failed and the other was rejected: said as such, original shown", () => {
  const v = original({ r6: cand("ORIGINAL", null, "MODEL_FAILURE"), r6_ft: cand("ORIGINAL", null, "STABILITY_GUARD") });
  assert.equal(v.title, "Not enhanced — original returned"); assert.equal(v.tone, "warn");
  assert.match(v.message, /One model failed to run and the other's output did not pass the safety checks/);
  assert.equal(row(v, "R6 —"), "no result — failed"); assert.match(row(v, "R6-FT"), /unstable on this face/);
});

test("one model succeeds while the other fails: enhanced, and the failure is named as a failure", () => {
  const v = enhanced(cand("ORIGINAL", null, "MODEL_FAILURE"));
  assert.equal(v.applied, true); assert.equal(v.tone, "ok"); assert.equal(v.title, "Enhanced"); assert.equal(v.chosen, "r6");
  assert.equal(v.message, "Only one model produced a safe result; it was kept: R6 — general deblur weights (NAFNet-GoPro). "
    + "The other model failed to run.");
  assert.match(v.leftSrc, /^data:image\/png/); assert.match(v.imageSrc, /^data:image\/png/); assert.equal(v.rightLabel, "ENHANCED");
  assert.equal(row(v, "R6-FT"), "no result — failed"); assert.match(row(v, "R6 —"), /CHOSEN/);
  assert.match(enhanced(cand("ORIGINAL", null, "TIMEOUT")).message, /failed to run/);
  assert.match(enhanced(cand("ORIGINAL", null, "RESOURCE_FAILURE")).message, /failed to run/);
});

test("one model succeeds while the other is rejected: enhanced, and the rejection is named as a rejection", () => {
  const v = enhanced(cand("ORIGINAL", null, "STABILITY_GUARD"));
  assert.equal(v.applied, true); assert.equal(v.title, "Enhanced");
  assert.equal(v.message, "Only one model produced a safe result; it was kept: R6 — general deblur weights (NAFNet-GoPro). "
    + "The other model's output was rejected by a safety check.");
  assert.doesNotMatch(v.message, /failed to run/);
  assert.match(row(v, "R6-FT"), /rejected by the 4-run stability check/);
});

test("both models succeed: wording unchanged", () => {
  const v = interpretBestResult(200, { ...base, status: "ENHANCED", used: "NAFNET_GOPRO", nafnet: nafnet({ profile: "r6_ft" }),
    selection: sel("r6_ft", { r6: cand("ENHANCED", 0.0061), r6_ft: cand("ENHANCED", 0.0092) }) }, 2000);
  assert.equal(v.message, "Both models produced a safe result; the sharper one was kept: R6-FT — face-blur fine-tuned weights.");
});

test("a face the gate rejects is neither a failure nor a safety rejection", () => {
  const v = interpretBestResult(200, { ...base, status: "ORIGINAL", image_b64: JPG_B64, image_format: "as_received",
    nafnet: nafnet({ eligible: false, reason: "NAFNET_REJECT_NOT_BLUR" }), selection: sel(null, { r6: cand("ORIGINAL") }) }, 60);
  assert.equal(v.title, "Not enhanced — original returned"); assert.equal(v.tone, "neutral"); assert.match(v.message, /Not blurred/);
});

// ------------------------------------------------------------------ Enhance button vs service availability

const READY = { ready: true, profiles: ["original", "r5_6", "r6", "r6_ft", "best"], limits: { max_bytes: 1 } };
const ui = (over = {}) => ({ file: { name: "a.png" }, validDims: true, busy: false, available: true, service: "online", ...over });

test("Enhance needs a valid image, no request in flight, the mode enabled and a service known to be online", () => {
  assert.equal(enhanceEnabled(ui()), true);
  assert.equal(enhanceEnabled(ui({ file: null })), false);
  assert.equal(enhanceEnabled(ui({ validDims: false })), false);
  assert.equal(enhanceEnabled(ui({ available: false })), false);
  for (const service of ["offline", "loading", "unknown", "", undefined]) assert.equal(enhanceEnabled(ui({ service })), false);
});

test("the button stays disabled while a request is in progress, whatever the service state", () => {
  for (const service of ["online", "offline", "loading", "unknown"]) assert.equal(enhanceEnabled(ui({ busy: true, service })), false);
  assert.equal(enhanceEnabled(ui({ busy: false })), true);                  // and comes back when the request ends
});

test("service goes online, offline, and online again: Enhance follows the health check", () => {
  const s = ui({ service: "unknown" });
  assert.equal(enhanceEnabled(s), false); assert.match(serviceNote(s.service, s.available), /Checking/);
  s.service = interpretReady(200, READY).state;                             // first /ready answer
  assert.equal(s.service, "online"); assert.equal(enhanceEnabled(s), true); assert.equal(serviceNote(s.service, s.available), "");
  s.service = interpretReady(0, null).state;                                // service unreachable
  assert.equal(s.service, "offline"); assert.equal(enhanceEnabled(s), false);
  assert.match(serviceNote(s.service, s.available), /offline.*Enhance is disabled/);
  s.service = interpretReady(503, { ready: false, error: "" }).state;       // restarting: models loading
  assert.equal(s.service, "loading"); assert.equal(enhanceEnabled(s), false); assert.match(serviceNote(s.service, s.available), /starting/);
  s.service = interpretReady(503, { ready: false, error: "boom" }).state;   // failed to load
  assert.equal(s.service, "offline"); assert.equal(enhanceEnabled(s), false);
  s.service = interpretReady(200, READY).state;                             // back
  assert.equal(enhanceEnabled(s), true); assert.equal(serviceNote(s.service, s.available), "");
});

test("online but the single-output mode is not enabled: disabled, with the existing note", () => {
  assert.equal(enhanceEnabled(ui({ available: false })), false);
  assert.match(serviceNote("online", false), /Single-output mode is not enabled.*FES_EXPERIMENTAL_PROFILES=r6,r6_ft/);
  assert.match(serviceNote("offline", false), /offline/);                   // the more basic problem is reported first
});

test("a request that fails after the button was enabled still reports the service as unavailable", () => {
  const v = interpretBestResult(0, null, 10);
  assert.equal(v.ok, false); assert.equal(v.title, "Service unavailable");
  assert.equal(v.message, "Could not reach the local enhancement service.");
});
