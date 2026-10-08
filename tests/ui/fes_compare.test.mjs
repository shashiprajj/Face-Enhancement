// R5-9 comparison-page logic tests (Node ≥ 18, no dependencies):  node --test tests/ui/fes_compare.test.mjs
// Also run from pytest by tests/test_r5_ui.py when `node` is on PATH.
import assert from "node:assert/strict";
import { test } from "node:test";

import {
  PROFILES, PROFILE_IDS, RATINGS, TAGS, availableProfiles, buildProfileForm, canSave, interpretProfileResult,
  makeRecord, recordsToCSV, recordsToJSON,
} from "../../src/fes_service/ui/fes_compare.js";

const PNG_B64 = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0, 0, 0, 13]).toString("base64");
const JPG_B64 = Buffer.from([0xff, 0xd8, 0xff, 0xe0, 0, 16, 0x4a, 0x46, 0x49, 0x46]).toString("base64");
const base = { request_id: "r1", warnings: [], chip_px: 180, timings_ms: { total: 1234 }, route_mode: "auto",
  image_b64: JPG_B64, chip_b64: PNG_B64, image_format: "as_received" };
const nafnet = (over = {}) => ({ eligible: true, reason: "NAFNET_ELIGIBLE_STRONG_LOW_NOISE_BLUR",
  route_version: "p5-nafnet-route-1.0.0", infer_s: 0.9, artifact: { passed: true, reasons: [], metrics: {} }, ...over });
const row = (v, k) => (v.rows.find(([key]) => key === k) || [])[1];

class FakeFD { constructor() { this.e = []; } append(k, v) { this.e.push([k, v]); } }

test("profiles, labels and warning text", () => {
  assert.deepEqual(PROFILE_IDS, ["original", "r5_6", "r5_8"]);
  assert.equal(PROFILES[1].warning, "REFERENCE — NOT PRODUCTION-APPROVED");
  assert.equal(PROFILES[2].warning, "EXPERIMENTAL — NOT PRODUCTION-APPROVED");
  assert.deepEqual(RATINGS, ["Better", "Same", "Worse", "Unsafe / Artifact"]);
});

test("request always names an explicit profile; unknown profiles are refused client-side", () => {
  const fd = buildProfileForm({ name: "a.png" }, "r5_8", FakeFD);
  assert.deepEqual(fd.e.filter(([k]) => k !== "image"), [["input_mode", "full_image"], ["return_image", "true"], ["profile", "r5_8"]]);
  assert.throws(() => buildProfileForm({ name: "a" }, "default", FakeFD));
});

test("available profiles come from /ready", () => {
  assert.deepEqual(availableProfiles({ profiles: ["original", "r5_6"] }), ["original", "r5_6"]);
  assert.deepEqual(availableProfiles({ profiles: ["original", "r5_6", "r5_8"] }), PROFILE_IDS);
  assert.deepEqual(availableProfiles(null), []);
});

test("original profile: baseline, no model", () => {
  const v = interpretProfileResult("original", 200, { ...base, status: "ORIGINAL", model: null, used: "ORIGINAL",
    reason: "profile_original", profile: "original" }, 50);
  assert.equal(v.ok, true); assert.equal(v.applied, false); assert.equal(v.finalOutput, "ORIGINAL");
  assert.equal(row(v, "Model invoked"), "none"); assert.equal(row(v, "HTTP / API status"), "200 ORIGINAL");
  assert.match(v.imageSrc, /^data:image\/jpeg;base64,/);
});

test("NAFNet success is shown as applied with its output", () => {
  const v = interpretProfileResult("r5_6", 200, { ...base, status: "ENHANCED", image_format: "png", image_b64: PNG_B64,
    model: "NAFNET_GOPRO", used: "NAFNET_GOPRO", route: "NAFNET_GOPRO", artifact_check: "PASS", nafnet: nafnet() }, 2000);
  assert.equal(v.applied, true); assert.equal(v.finalOutput, "NAFNet"); assert.equal(v.tone, "ok");
  assert.equal(row(v, "Enhancement applied"), "YES"); assert.equal(row(v, "Artifact guard"), "PASS");
  assert.equal(row(v, "Local artifact detector"), "not part of this profile");
  assert.match(row(v, "Gate decision"), /^ELIGIBLE/);
});

test("fallback to ORIGINAL is never shown as an enhancement", () => {
  const v = interpretProfileResult("r5_8", 200, { ...base, status: "ORIGINAL", model: "NAFNET_GOPRO", used: "ORIGINAL",
    fallback: "LOCAL_ARTIFACT_GUARD", error: "ARTIFACT_LOCAL_PATCH:l_dot_c_max", artifact_check: "FAIL",
    nafnet: nafnet({ profile: "r5_8", local_artifact: { passed: false, reasons: ["ARTIFACT_LOCAL_PATCH:l_dot_c_max"],
      metrics: { l_dot_c_max: 31.2 } } }) }, 2000);
  assert.equal(v.applied, false); assert.equal(v.finalOutput, "ORIGINAL"); assert.equal(v.tone, "bad");
  assert.equal(v.label, "ORIGINAL — enhancement not applied");
  assert.match(v.headline, /Fallback to ORIGINAL — LOCAL_ARTIFACT_GUARD/);
  assert.equal(row(v, "Artifact guard"), "PASS");                       // the global guard itself passed
  assert.match(row(v, "Local artifact detector"), /^FAIL .*l_dot_c_max 31\.20/);
  assert.match(row(v, "Fallback / reason"), /^LOCAL_ARTIFACT_GUARD — .*local-patch/);
  assert.equal(row(v, "Model invoked"), "NAFNET_GOPRO");
});

test("not-eligible original is labelled as not applied, with the gate reason", () => {
  const v = interpretProfileResult("r5_6", 200, { ...base, status: "ORIGINAL", model: null, used: "ORIGINAL",
    reason: "NAFNET_REJECT_NOT_BLUR", nafnet: nafnet({ eligible: false, reason: "NAFNET_REJECT_NOT_BLUR", artifact: null }) }, 300);
  assert.equal(v.applied, false); assert.equal(v.label, "ORIGINAL — enhancement not applied");
  assert.match(row(v, "Gate decision"), /^NOT ELIGIBLE — NAFNET_REJECT_NOT_BLUR/);
  assert.equal(row(v, "Artifact guard"), "not run");
  assert.match(row(v, "Fallback / reason"), /^none — not eligible/);
});

test("A4 on the strong-noise route is not presented as NAFNet", () => {
  const v = interpretProfileResult("r5_6", 200, { ...base, status: "ENHANCED", image_format: "png", model: "A4_SWINIR_L",
    used: "A4", route: "A4", nafnet: nafnet({ eligible: false, reason: "NAFNET_REJECT_HIGH_NOISE", artifact: null }) }, 60000);
  assert.equal(v.finalOutput, "A4"); assert.equal(v.tone, "warn"); assert.match(v.label, /not NAFNet/);
});

test("errors are clean: not enabled, rejected, busy, network, timeout", () => {
  const ne = interpretProfileResult("r5_8", 422, { status: "REJECTED", error: "profile 'r5_8' is not enabled on this service" }, 5);
  assert.equal(ne.ok, false); assert.match(ne.message, /FES_EXPERIMENTAL_PROFILES=r5_8/);
  assert.equal(row(ne, "HTTP / API status"), "422 REJECTED");
  assert.equal(interpretProfileResult("r5_6", 503, { status: "BUSY" }, 1).kind, "BUSY");
  assert.match(interpretProfileResult("r5_6", 0, null, 1).headline, /unavailable/);
  assert.match(interpretProfileResult("r5_6", 0, null, 1, { timedOut: true }).headline, /timed out/);
  const s5 = interpretProfileResult("r5_6", 500, null, 1);
  assert.equal(s5.ok, false); assert.doesNotMatch(s5.message, /Traceback|at .*\.js/);
});

test("evaluation records: save rule, content, JSON and CSV export", () => {
  const ok = interpretProfileResult("r5_6", 200, { ...base, status: "ENHANCED", image_format: "png", model: "NAFNET_GOPRO",
    used: "NAFNET_GOPRO", nafnet: nafnet() }, 900);
  assert.equal(canSave({ r5_6: ok }, { r5_6: { rating: "" } }), false);
  assert.equal(canSave({}, { r5_6: { rating: "Better" } }), false);
  assert.equal(canSave({ r5_6: ok }, { r5_6: { rating: "Better" } }), true);
  const rec = makeRecord({ fileName: "x.jpg", fileSize: 10, dims: "10×10", category: "strong blur", overallNote: "=1+1",
    results: { r5_6: ok }, ratings: { r5_6: { rating: "Better", tags: [TAGS[0], "bogus"], notes: 'say "hi"' } } },
  new Date("2026-10-08T00:00:00Z"));
  assert.equal(rec.profiles.r5_6.rating, "Better"); assert.deepEqual(rec.profiles.r5_6.tags, [TAGS[0]]);
  assert.equal(rec.profiles.r5_6.final_output, "NAFNet"); assert.equal(rec.profiles.r5_8, undefined);
  assert.equal(JSON.parse(recordsToJSON([rec])).records.length, 1);
  const csv = recordsToCSV([rec]);
  const [head, line] = csv.trim().split("\r\n");
  assert.match(head, /"r5_6_rating"/); assert.match(head, /"r5_8_local_detector"/);
  assert.match(line, /"'=1\+1"/);                                       // formula injection neutralised
  assert.match(line, /"say ""hi"""/);
  assert.doesNotMatch(csv + recordsToJSON([rec]), /base64|data:image/);  // no pixels in exports
});
