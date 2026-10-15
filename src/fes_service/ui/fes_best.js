// R6-BEST — pure logic of the single-output page (no DOM): 1 image in → 1 image out through POST /v3/enhance.
// The service runs both NAFNet candidates (R6, R6-FT), applies every safety check and picks the output; this file only
// labels what it reports (`selection`). Nothing here is a measurement of identity or matching quality.
import { explainReason, formatMs, sniffMime } from "./fes_ui.js";

export const ENDPOINT = "/v3/enhance";
export const BEST_PROFILE = "best";
export const CANDIDATES = {
  r6: "R6 — general deblur weights (NAFNet-GoPro)",
  r6_ft: "R6-FT — face-blur fine-tuned weights",
};
const FALLBACK_TEXT = {
  TIMEOUT: "ran out of time",
  MODEL_FAILURE: "failed",
  RESOURCE_FAILURE: "ran out of resources",
  INVALID_OUTPUT: "output failed the output check",
  ARTIFACT_GUARD: "output failed the artifact guard",
  LOCAL_ARTIFACT_GUARD: "output rejected by the local-patch artifact detector",
  STABILITY_GUARD: "unstable on this face (rejected by the 4-run stability check)",
};

// A candidate that did not RUN to an output (as opposed to one whose output a safety check rejected).
const TECHNICAL_FAILURES = new Set(["TIMEOUT", "MODEL_FAILURE", "RESOURCE_FAILURE"]);
export const FAILED_TITLE = "Enhancement failed — showing original image";

/** Service state (fes_ui.interpretReady: "online" | "loading" | "offline"; "unknown" before the first check) and
 *  whether the single-output mode is enabled → the note shown under the upload, "" when Enhance can be used. */
export function serviceNote(service, available = true) {
  if (service === "unknown") return "Checking the enhancement service… Enhance is disabled until it answers.";
  if (service === "loading") return "The enhancement service is still starting. Enhance is disabled until it is ready.";
  if (service !== "online") return "The enhancement service is offline. Enhance is disabled until it is reachable again.";
  return available ? "" : "Single-output mode is not enabled on this service. Start it with FES_EXPERIMENTAL_PROFILES=r6,r6_ft.";
}

/** The Enhance button is usable only with a valid image, no request in flight, the mode enabled and the service
 *  known to be online (unknown / starting / offline = disabled). */
export const enhanceEnabled = ({ file, validDims, busy, available, service }) =>
  !!file && !!validDims && !busy && !!available && service === "online";

/** /ready body → true when this service can serve the single-output mode. */
export const bestAvailable = (readyBody) =>
  !!readyBody && Array.isArray(readyBody.profiles) && readyBody.profiles.includes(BEST_PROFILE);

/** Multipart body for POST /v3/enhance (the endpoint fixes the profile; none is sent). `blur` = this request's
 *  optional blur range {enhanceFrom, giveUpAbove} (percent strings / numbers; empty = the service setting). */
export function buildBestForm(file, FormDataCtor = FormData, blur = {}) {
  const fd = new FormDataCtor();
  fd.append("image", file, file.name || "upload");
  fd.append("input_mode", "full_image");
  fd.append("return_image", "true");
  const v = (x) => (x === null || x === undefined ? "" : String(x).trim());
  if (v(blur.enhanceFrom)) fd.append("enhance_from_pct", v(blur.enhanceFrom));
  if (v(blur.giveUpAbove)) fd.append("give_up_above_pct", v(blur.giveUpAbove));
  return fd;
}

const pct = (x) => `${Number(x).toFixed(Number.isInteger(Number(x)) ? 0 : 1)} %`;

/** /ready `blur_settings` → {available, text, minFrom} for the blur-range panel. The admin sets the range in the
 *  service's settings file; a request may only narrow it. */
export function blurSettingsView(bs) {
  if (!bs || typeof bs !== "object") return { available: false, text: "", minFrom: null };
  const lim = bs.validated_limit_pct || {};
  const from = Number.isFinite(bs.enhance_from_pct) ? `from ${pct(bs.enhance_from_pct)}`
    : `from the validated limit (${pct(lim.front)} front / ${pct(lim.side)} side faces)`;
  const upto = Number.isFinite(bs.give_up_above_pct) ? `up to ${pct(bs.give_up_above_pct)}` : "with no upper limit";
  return { available: true, minFrom: Number.isFinite(bs.min_enhance_from_pct) ? bs.min_enhance_from_pct : null,
    text: `Service setting: faces are enhanced ${from} ${upto} on the blur scale (0 % sharp – 100 % very heavy blur).` };
}

const VERDICT = { IN_RANGE: "inside the range", NOT_BLURRY_ENOUGH: "not blurred enough", TOO_BLURRED: "too blurred" };
const SOURCE = { request: "this request", deployment: "service setting", validated_limit: "validated limit", off: "off" };

/** The headline when the admin blur range (not the validated limit) decided the result; "" otherwise. */
export function blurRangeMessage(br) {
  if (!br || typeof br !== "object" || !Number.isFinite(br.blur_pct)) return "";
  const src = br.source || {};
  if (br.verdict === "TOO_BLURRED") {
    return `Too blurred to fix: the face is ${pct(br.blur_pct)} blurred, above the "give up above" setting `
      + `(${pct(br.give_up_above_pct)}), so it is returned unchanged.`;
  }
  if (br.verdict === "NOT_BLURRY_ENOUGH" && src.enhance_from !== "validated_limit") {
    return `Not blurred enough: the face is ${pct(br.blur_pct)} blurred, below the "enhance from" setting `
      + `(${pct(br.enhance_from_pct)}), so it is returned unchanged.`;
  }
  return "";
}

/** nafnet.blur_range → result rows (none when the service did not report one). */
export function blurRows(br) {
  if (!br || typeof br !== "object" || !Number.isFinite(br.blur_pct)) return [];
  const src = br.source || {};
  return [
    ["Blur of the face", `${pct(br.blur_pct)} (${br.framing === "side" ? "side-face" : "front-face"} scale) — ${VERDICT[br.verdict] || str(br.verdict)}`],
    ["Enhance from", `${pct(br.enhance_from_pct)} (${SOURCE[src.enhance_from] || str(src.enhance_from)})`],
    ["Give up above", Number.isFinite(br.give_up_above_pct)
      ? `${pct(br.give_up_above_pct)} (${SOURCE[src.give_up_above] || str(src.give_up_above)})` : "off — every heavily blurred face is attempted"],
  ];
}

const str = (v) => (v === null || v === undefined ? "" : String(v));
const candidateName = (id) => CANDIDATES[id] || str(id);

/**
 * (HTTP status | 0 = network failure, parsed body | null, round-trip ms, {timedOut}) → view:
 *   {ok, kind, tone, applied, chosen, title, message, leftLabel, rightLabel, leftSrc, imageSrc, rows: [[key, value]]}
 * `applied` is true ONLY when the service returned an enhanced image; `leftSrc` is the face crop the service analysed
 * (so both sides show the same framing) and "" when the upload itself should be shown.
 */
export function interpretBestResult(httpStatus, body, rttMs, { timedOut = false } = {}) {
  const fail = (kind, title, message) => ({ ok: false, kind, tone: kind === "BUSY" || kind === "NOT_READY" ? "warn" : "bad",
    applied: false, chosen: "", title, message, leftLabel: "", rightLabel: "", leftSrc: "", imageSrc: "", rows: [] });
  if (!httpStatus) {
    return timedOut ? fail("ERROR", "Request timed out", "No answer from the service in time. It may be overloaded.")
      : fail("ERROR", "Service unavailable", "Could not reach the local enhancement service.");
  }
  if (!body || typeof body !== "object" || typeof body.status !== "string") {
    return fail("ERROR", "Service error", `The service returned an unexpected response (HTTP ${httpStatus}).`);
  }
  const s = body.status;
  if (s === "BUSY") return fail("BUSY", "Service busy", "The service is processing other images. Try again in a moment.");
  if (s === "NOT_READY") return fail("NOT_READY", "Service not ready", "The models are still loading or failed to load.");
  if (s === "REJECTED") {
    return /not enabled/.test(str(body.error))
      ? fail("REJECTED", "Single-output mode is not enabled", "Start the service with FES_EXPERIMENTAL_PROFILES=r6,r6_ft.")
      : fail("REJECTED", "Image not accepted", str(body.error) || "The service could not accept this image.");
  }
  if ((s !== "ENHANCED" && s !== "ORIGINAL") || !body.image_b64) {
    return fail("ERROR", "Unexpected response", `Status "${s}" without an image.`);
  }

  const applied = s === "ENHANCED";
  const sel = body.selection && typeof body.selection === "object" ? body.selection : null;
  const cands = sel && sel.candidates && typeof sel.candidates === "object" ? sel.candidates : {};
  const ids = Object.keys(cands);
  const chosen = sel && typeof sel.chosen === "string" ? sel.chosen : "";
  const accepted = ids.filter((id) => cands[id] && cands[id].status === "ENHANCED");
  const nf = body.nafnet && typeof body.nafnet === "object" ? body.nafnet : null;
  const reason = nf && nf.reason ? str(nf.reason) : str(body.reason);
  const mime = body.image_format === "png" ? "image/png" : sniffMime(body.image_b64) || "image/png";
  const t = body.timings_ms || {};

  let title, message, tone;
  if (applied && chosen) {
    tone = "ok"; title = "Enhanced";
    const others = ids.filter((id) => id !== chosen && cands[id] && cands[id].fallback);
    const otherNote = !others.length ? ""
      : others.every((id) => TECHNICAL_FAILURES.has(cands[id].fallback))
        ? " The other model failed to run."
        : " The other model's output was rejected by a safety check.";
    message = accepted.length > 1
      ? `Both models produced a safe result; the sharper one was kept: ${candidateName(chosen)}.`
      : `Only one model produced a safe result; it was kept: ${candidateName(chosen)}.${otherNote}`;
  } else if (applied) {
    tone = "ok"; title = "Enhanced";
    message = `The face was enhanced by the ${str(body.used) || "other"} route (strong noise), not by NAFNet.`;
  } else if (ids.length && ids.every((id) => cands[id].fallback)) {
    const failed = ids.filter((id) => TECHNICAL_FAILURES.has(cands[id].fallback));
    if (failed.length === ids.length) {                 // nothing ran to an output: a technical failure, not a rejection
      tone = "bad"; title = FAILED_TITLE;
      message = "The models failed to run (a technical failure, not a quality rejection); the original is returned unchanged.";
    } else if (failed.length) {
      tone = "warn"; title = "Not enhanced — original returned";
      message = "One model failed to run and the other's output did not pass the safety checks; the original is returned unchanged.";
    } else {
      tone = "warn"; title = "Not enhanced — original returned";
      message = "The models ran, but no output passed the safety checks, so the original image is returned unchanged.";
    }
  } else {
    tone = "neutral"; title = "Not enhanced — original returned";
    message = blurRangeMessage(nf && nf.blur_range) || explainReason(reason)
      || "This image is not one the enhancement is meant for; it is returned unchanged.";
  }

  const candRow = (id) => {
    const c = cands[id] || {};
    if (c.status === "ENHANCED") {
      return `safe result${Number.isFinite(c.score) ? ` · sharpness ${Number(c.score).toFixed(5)}` : ""}${id === chosen ? " · CHOSEN" : ""}`;
    }
    return c.fallback ? `no result — ${FALLBACK_TEXT[c.fallback] || c.fallback}` : "not run for this image";
  };
  const rows = [
    ["Result", applied ? "ENHANCED" : "ORIGINAL (unchanged)"],
    ["Output chosen", applied ? (chosen ? candidateName(chosen) : str(body.used)) : "none"],
    ...Object.keys(CANDIDATES).map((id) => [candidateName(id), ids.includes(id) ? candRow(id) : "not run for this image"]),
    ["Gate decision", nf ? `${nf.eligible ? "ELIGIBLE" : "NOT ELIGIBLE"} — ${nf.reason || "—"}` : reason],
    ...blurRows(nf && nf.blur_range),
    ["Blur score of the input (lapd1_c)", body.quality && Number.isFinite(body.quality.lapd1_c) ? Number(body.quality.lapd1_c).toFixed(6) : ""],
    ["Selection rule", sel ? str(sel.rule) : ""],
    ["Processing time (server)", Number.isFinite(t.total) ? formatMs(t.total) : ""],
    ["Round-trip time", Number.isFinite(rttMs) ? formatMs(rttMs) : ""],
    ["Warnings", Array.isArray(body.warnings) ? body.warnings.join(", ") : ""],
    ["Request ID", str(body.request_id)],
  ].filter(([, v]) => v !== "");

  return {
    ok: true, kind: s, tone, applied, chosen, title, message, rows,
    leftLabel: applied ? "ORIGINAL (face crop)" : "ORIGINAL",
    rightLabel: applied ? "ENHANCED" : "ORIGINAL — not enhanced",
    leftSrc: applied && body.chip_b64 ? `data:image/png;base64,${body.chip_b64}` : "",
    imageSrc: `data:${mime};base64,${body.image_b64}`,
  };
}
