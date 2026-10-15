// R6-BEST single-output page logic tests (Node ≥ 18, no dependencies):  node --test tests/ui/fes_best.test.mjs
import assert from "node:assert/strict";
import { test } from "node:test";

import { BEST_PROFILE, ENDPOINT, bestAvailable, buildBestForm, interpretBestResult } from "../../src/fes_service/ui/fes_best.js";

const PNG_B64 = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0, 0, 0, 13]).toString("base64");
const JPG_B64 = Buffer.from([0xff, 0xd8, 0xff, 0xe0, 0, 16, 0x4a, 0x46, 0x49, 0x46]).toString("base64");
const base = { request_id: "r1", warnings: [], chip_px: 180, timings_ms: { total: 4321 }, profile: "best",
  image_b64: PNG_B64, chip_b64: PNG_B64, image_format: "png", quality: { lapd1_c: 0.0031 } };
const nafnet = (over = {}) => ({ eligible: true, reason: "NAFNET_ELIGIBLE_STRONG_LOW_NOISE_BLUR", ...over });
const cand = (status, score = null, fallback = "") => ({ status, fallback, score, model: "x" });
const sel = (chosen, candidates) => ({ rule: "sharper_lapd1_c", chosen, candidates });
const row = (v, k) => (v.rows.find(([key]) => key.startsWith(k)) || [])[1];

class FakeFD { constructor() { this.e = []; } append(k, v) { this.e.push([k, v]); } }

test("request: /v3/enhance, full image, no profile field", () => {
  assert.equal(ENDPOINT, "/v3/enhance"); assert.equal(BEST_PROFILE, "best");
  const fd = buildBestForm({ name: "a.png" }, FakeFD);
  assert.deepEqual(fd.e.filter(([k]) => k !== "image"), [["input_mode", "full_image"], ["return_image", "true"]]);
});

test("availability comes from /ready", () => {
  assert.equal(bestAvailable({ profiles: ["original", "r5_6", "r6", "r6_ft", "best"] }), true);
  assert.equal(bestAvailable({ profiles: ["original", "r5_6", "r6"] }), false);
  assert.equal(bestAvailable(null), false);
});

test("both accepted: the chosen one is named, ONE image shown next to the face crop", () => {
  const v = interpretBestResult(200, { ...base, status: "ENHANCED", used: "NAFNET_GOPRO", nafnet: nafnet({ profile: "r6_ft" }),
    selection: sel("r6_ft", { r6: cand("ENHANCED", 0.0061), r6_ft: cand("ENHANCED", 0.0092) }) }, 4500);
  assert.equal(v.ok, true); assert.equal(v.applied, true); assert.equal(v.chosen, "r6_ft"); assert.equal(v.tone, "ok");
  assert.match(v.message, /Both models.*sharper one was kept.*R6-FT/);
  assert.match(v.leftSrc, /^data:image\/png;base64,/); assert.match(v.imageSrc, /^data:image\/png;base64,/);
  assert.equal(v.leftLabel, "ORIGINAL (face crop)"); assert.equal(v.rightLabel, "ENHANCED");
  assert.match(row(v, "R6-FT"), /safe result · sharpness 0.00920 · CHOSEN/);
  assert.match(row(v, "R6 —"), /safe result · sharpness 0.00610$/);
  assert.match(row(v, "Output chosen"), /R6-FT/);
});

test("one rejected by a guard: the other is kept and the rejection is explained", () => {
  const v = interpretBestResult(200, { ...base, status: "ENHANCED", used: "NAFNET_GOPRO", nafnet: nafnet({ profile: "r6" }),
    selection: sel("r6", { r6: cand("ENHANCED", 0.0061), r6_ft: cand("ORIGINAL", null, "STABILITY_GUARD") }) }, 4500);
  assert.equal(v.chosen, "r6"); assert.match(v.message, /Only one model/);
  assert.match(row(v, "R6-FT"), /no result — unstable on this face/);
});

test("none accepted: original, never shown as enhanced", () => {
  const v = interpretBestResult(200, { ...base, status: "ORIGINAL", image_b64: JPG_B64, image_format: "as_received",
    fallback: "STABILITY_GUARD", nafnet: nafnet(),
    selection: sel(null, { r6: cand("ORIGINAL", null, "STABILITY_GUARD"), r6_ft: cand("ORIGINAL", null, "ARTIFACT_GUARD") }) }, 4000);
  assert.equal(v.applied, false); assert.equal(v.tone, "warn"); assert.equal(v.leftSrc, "");
  assert.match(v.imageSrc, /^data:image\/jpeg/); assert.equal(v.rightLabel, "ORIGINAL — not enhanced");
  assert.match(v.message, /no output passed the safety checks/); assert.equal(row(v, "Output chosen"), "none");
});

test("not eligible: original with the gate's reason; the second model is reported as not run", () => {
  const v = interpretBestResult(200, { ...base, status: "ORIGINAL", image_b64: JPG_B64, image_format: "as_received",
    nafnet: nafnet({ eligible: false, reason: "NAFNET_REJECT_NOT_BLUR" }),
    selection: sel(null, { r6: cand("ORIGINAL") }) }, 300);
  assert.equal(v.applied, false); assert.equal(v.tone, "neutral");
  assert.match(row(v, "Gate decision"), /NOT ELIGIBLE — NAFNET_REJECT_NOT_BLUR/);
  assert.equal(row(v, "R6-FT"), "not run for this image"); assert.equal(row(v, "R6 —"), "not run for this image");
});

test("failures are never shown as a result", () => {
  assert.equal(interpretBestResult(0, null, 10).title, "Service unavailable");
  assert.equal(interpretBestResult(0, null, 10, { timedOut: true }).title, "Request timed out");
  assert.equal(interpretBestResult(503, { status: "BUSY" }, 10).kind, "BUSY");
  const ne = interpretBestResult(422, { status: "REJECTED", error: "profile 'best' is not enabled on this service" }, 5);
  assert.equal(ne.ok, false); assert.match(ne.message, /FES_EXPERIMENTAL_PROFILES=r6,r6_ft/);
  assert.equal(interpretBestResult(422, { status: "REJECTED", error: "not a decodable image" }, 5).message, "not a decodable image");
  assert.equal(interpretBestResult(200, { status: "ENHANCED" }, 5).ok, false);
});

test("blur range: request fields only when filled; panel text from /ready; result rows", async () => {
  const { blurSettingsView, blurRows } = await import("../../src/fes_service/ui/fes_best.js");
  const plain = buildBestForm({ name: "a.png" }, FakeFD, { enhanceFrom: "", giveUpAbove: "  " });
  assert.equal(plain.e.some(([k]) => k.endsWith("_pct")), false);
  const fd = buildBestForm({ name: "a.png" }, FakeFD, { enhanceFrom: "45", giveUpAbove: 90 });
  assert.deepEqual(fd.e.filter(([k]) => k.endsWith("_pct")), [["enhance_from_pct", "45"], ["give_up_above_pct", "90"]]);

  assert.equal(blurSettingsView(undefined).available, false);
  const off = blurSettingsView({ enhance_from_pct: null, give_up_above_pct: null, min_enhance_from_pct: 36,
    validated_limit_pct: { front: 35.9, side: 32.9 } });
  assert.equal(off.minFrom, 36);
  assert.match(off.text, /validated limit \(35.9 % front \/ 32.9 % side faces\) with no upper limit/);
  assert.match(blurSettingsView({ enhance_from_pct: 40, give_up_above_pct: 85, validated_limit_pct: {} }).text,
    /from 40 % up to 85 %/);

  assert.deepEqual(blurRows(null), []);
  const br = { framing: "side", blur_pct: 65.5, enhance_from_pct: 32.9, give_up_above_pct: 40, verdict: "TOO_BLURRED",
    source: { enhance_from: "validated_limit", give_up_above: "request" } };
  const v = interpretBestResult(200, { ...base, status: "ORIGINAL", reason: "NAFNET_REJECT_TOO_BLURRED",
    nafnet: nafnet({ eligible: false, reason: "NAFNET_REJECT_TOO_BLURRED", blur_range: br }),
    selection: sel(null, { r6: cand("ORIGINAL") }) }, 100);
  assert.equal(v.applied, false);
  assert.equal(v.message, 'Too blurred to fix: the face is 65.5 % blurred, above the "give up above" setting (40 %), so it is returned unchanged.');
  const mild = interpretBestResult(200, { ...base, status: "ORIGINAL", reason: "NAFNET_REJECT_MILD_BLUR",
    nafnet: nafnet({ eligible: false, reason: "NAFNET_REJECT_MILD_BLUR", blur_range: { ...br, blur_pct: 37.7,
      enhance_from_pct: 40, verdict: "NOT_BLURRY_ENOUGH", source: { enhance_from: "deployment", give_up_above: "off" } } }) }, 100);
  assert.match(mild.message, /^Not blurred enough: the face is 37.7 % blurred, below the "enhance from" setting \(40 %\)/);
  const sealed = interpretBestResult(200, { ...base, status: "ORIGINAL", reason: "NAFNET_REJECT_MILD_BLUR",
    nafnet: nafnet({ eligible: false, reason: "NAFNET_REJECT_MILD_BLUR", blur_range: { ...br, verdict: "NOT_BLURRY_ENOUGH",
      source: { enhance_from: "validated_limit", give_up_above: "off" } } }) }, 100);
  assert.match(sealed.message, /^Only mildly blurred/);                  // the validated limit keeps the old wording
  assert.equal(row(v, "Blur of the face"), "65.5 % (side-face scale) — too blurred");
  assert.equal(row(v, "Enhance from"), "32.9 % (validated limit)");
  assert.equal(row(v, "Give up above"), "40 % (this request)");
  const on = interpretBestResult(200, { ...base, status: "ENHANCED", used: "NAFNET_GOPRO", nafnet: nafnet({ blur_range:
    { ...br, framing: "front", blur_pct: 70, give_up_above_pct: null, verdict: "IN_RANGE", source: { enhance_from: "deployment", give_up_above: "off" } } }),
    selection: sel("r6", { r6: cand("ENHANCED", 0.005) }) }, 100);
  assert.equal(row(on, "Blur of the face"), "70 % (front-face scale) — inside the range");
  assert.match(row(on, "Give up above"), /^off/);
});
