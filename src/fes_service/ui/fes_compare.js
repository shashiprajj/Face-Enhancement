// R5-9 — pure logic of the research comparison page (no DOM): profiles, requests, per-profile result views,
// manual-evaluation records and their local export. Every status, decision and reason shown comes from the backend
// (/ready, /v1/enhance); this file only labels it. Nothing here is a measurement of identity or matching quality.
import { explainReason, formatMs, sniffMime } from "./fes_ui.js";

export const PROFILES = [
  { id: "original", title: "ORIGINAL", warning: "", note: "Baseline — the uploaded image, returned unchanged. No model." },
  { id: "r5_6", title: "NAFNet R5-6 Reference", warning: "REFERENCE — NOT PRODUCTION-APPROVED",
    note: "The historical R5-6 route exactly as validated (NAFNet-GoPro, input gate, artifact guard)." },
  { id: "r5_8", title: "NAFNet R5-8 Experimental", warning: "EXPERIMENTAL — NOT PRODUCTION-APPROVED",
    note: "The saved R5-8 research candidate (rejected as RED): reflect framing + local-patch artifact detector." },
];
export const PROFILE_IDS = PROFILES.map((p) => p.id);
export const profileById = (id) => PROFILES.find((p) => p.id === id) || null;

export const RATINGS = ["Better", "Same", "Worse", "Unsafe / Artifact"];
export const TAGS = ["blur improved", "identity appears preserved", "rainbow/dotted artifact", "grid/padding artifact",
  "hallucinated detail", "grayscale changed", "face geometry changed", "other"];
export const CATEGORIES = ["strong blur", "moderate blur", "low resolution", "dark", "noisy", "JPEG/compression",
  "grayscale", "printed/photo-of-photo", "angled face", "glasses", "difficult real capture", "normal clean", "other"];

const FALLBACK_TEXT = {
  TIMEOUT: "the model ran out of time",
  MODEL_FAILURE: "the model failed",
  RESOURCE_FAILURE: "the model worker ran out of resources",
  INVALID_OUTPUT: "the output failed the output check",
  ARTIFACT_GUARD: "NAFNet ran, but its output failed the artifact guard",
  LOCAL_ARTIFACT_GUARD: "NAFNet ran, but the R5-8 local-patch artifact detector rejected its output",
};

/** /ready body → which profiles this service can serve right now (original is always local to the service). */
export function availableProfiles(readyBody) {
  const list = readyBody && Array.isArray(readyBody.profiles) ? readyBody.profiles : [];
  return PROFILE_IDS.filter((id) => list.includes(id));
}

/** Multipart body for POST /v1/enhance with an EXPLICIT profile (never the default request). */
export function buildProfileForm(file, profile, FormDataCtor = FormData) {
  if (!PROFILE_IDS.includes(profile)) throw new Error(`unknown profile ${profile}`);
  const fd = new FormDataCtor();
  fd.append("image", file, file.name || "upload");
  fd.append("input_mode", "full_image");
  fd.append("return_image", "true");
  fd.append("profile", profile);
  return fd;
}

const str = (v) => (v === null || v === undefined ? "" : String(v));
const num = (v, d = 2) => (Number.isFinite(v) ? Number(v).toFixed(d) : "");

/**
 * (profile id, HTTP status | 0 = network failure, parsed body | null, round-trip ms, {timedOut}) → panel view:
 *   {profile, ok, httpStatus, kind, applied, finalOutput, fallback, tone, headline, label, imageSrc, chipSrc,
 *    rows: [[key, value]], record: {...flat fields for the evaluation log}}
 * `applied` is true ONLY when the service returned an enhanced image. An unchanged fallback is never shown as one.
 */
export function interpretProfileResult(profile, httpStatus, body, rttMs, { timedOut = false } = {}) {
  const base = { profile, httpStatus, rttMs, applied: false, finalOutput: "—", fallback: "", imageSrc: "", chipSrc: "" };
  const fail = (kind, headline, message) => ({
    ...base, ok: false, kind, tone: kind === "BUSY" || kind === "NOT_READY" ? "warn" : "bad", headline, message,
    label: "", rows: [["Profile", profileById(profile)?.title || profile], ["HTTP / API status", httpStatus ? `${httpStatus} ${kind}` : kind]],
    record: { status: kind, http_status: httpStatus, error: message },
  });
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
    const notEnabled = /not enabled/.test(str(body.error));
    return fail("REJECTED", notEnabled ? "Profile not enabled on this service" : "Request rejected",
      notEnabled && profile === "r5_8"
        ? "R5-8 is off by default. Restart the service with FES_EXPERIMENTAL_PROFILES=r5_8 to compare it."
        : str(body.error) || "The service could not accept this image.");
  }
  if ((s !== "ENHANCED" && s !== "ORIGINAL") || !body.image_b64) {
    return fail("ERROR", "Unexpected response", `Status "${s}" without an image.`);
  }

  const nf = body.nafnet && typeof body.nafnet === "object" ? body.nafnet : null;
  const art = nf && nf.artifact && typeof nf.artifact === "object" ? nf.artifact : null;
  const loc = nf && nf.local_artifact && typeof nf.local_artifact === "object" ? nf.local_artifact : null;
  const applied = s === "ENHANCED";
  const model = typeof body.model === "string" && body.model ? body.model : "";
  const finalOutput = !applied ? "ORIGINAL" : body.used === "NAFNET_GOPRO" ? "NAFNet" : body.used === "A4" ? "A4" : str(body.used);
  const fallback = str(body.fallback);
  const reason = str(body.reason);
  const t = body.timings_ms || {};
  const mime = body.image_format === "png" ? "image/png" : sniffMime(body.image_b64) || "image/png";

  let headline, tone, label;
  if (profile === "original") {
    headline = "Original (baseline)"; tone = "neutral"; label = "ORIGINAL";
  } else if (applied && finalOutput === "NAFNet") {
    headline = "Enhancement applied — NAFNet output"; tone = "ok"; label = `NAFNet output (face crop, ${body.chip_px || "?"} px)`;
  } else if (applied) {
    headline = `Enhancement applied — ${finalOutput} (not NAFNet)`; tone = "warn";
    label = `${finalOutput} output (face crop) — strong-noise route, not NAFNet`;
  } else {
    headline = fallback ? `Fallback to ORIGINAL — ${fallback}` : "ORIGINAL — enhancement not applied";
    tone = fallback ? "bad" : "neutral";
    label = "ORIGINAL — enhancement not applied";
  }

  let gate = "";
  if (nf) gate = `${nf.eligible ? "ELIGIBLE" : "NOT ELIGIBLE"} — ${nf.reason || "—"}`;
  else if (profile !== "original") gate = [body.decision, reason].filter(Boolean).join(" — ");
  const gateText = nf ? explainReason(nf.reason) : explainReason(reason);
  // the guard's own verdict (artifact_check also turns FAIL when the later R5-8 local detector rejects)
  const guard = art
    ? `${art.passed ? "PASS" : "FAIL"}${Array.isArray(art.reasons) && art.reasons.length ? ` (${art.reasons.join(", ")})` : ""}`
    : "not run";
  let local;
  if (profile !== "r5_8") local = profile === "original" ? "" : "not part of this profile";
  else if (loc) {
    local = `${loc.passed ? "PASS" : "FAIL"}${Array.isArray(loc.reasons) && loc.reasons.length ? ` (${loc.reasons.join(", ")})` : ""}`;
    if (loc.metrics && Number.isFinite(loc.metrics.l_dot_c_max)) local += ` · l_dot_c_max ${num(loc.metrics.l_dot_c_max)}`;
  } else local = "not run (no NAFNet output reached it)";
  const fbText = fallback
    ? `${fallback} — ${FALLBACK_TEXT[fallback] || "enhancement fell back to the original"}${body.error ? ` [${str(body.error).slice(0, 160)}]` : ""}`
    : !applied && profile !== "original" ? `none — not eligible${gateText ? `: ${gateText}` : ""}` : "";

  const rows = [
    ["Profile", profileById(profile)?.title || profile],
    ["Model invoked", model || "none"],
    // B0 is the routing code of "keep the original"
    ["Route", [body.route_mode, body.route === "B0" ? "ORIGINAL" : body.route].filter(Boolean).join(" · ") || "—"],
    ["Enhancement applied", applied ? "YES" : "NO"],
    ["Final output", finalOutput],
    ["Gate decision", gate],
    ["Artifact guard", profile === "original" ? "" : guard],
    ["Local artifact detector", local],
    ["Fallback / reason", profile === "original" ? "" : fbText],
    ["Route version", nf ? str(nf.route_version) : ""],
    ["NAFNet inference", nf && Number.isFinite(nf.infer_s) ? formatMs(nf.infer_s * 1000) : ""],
    ["Processing time (server)", Number.isFinite(t.total) ? formatMs(t.total) : ""],
    ["Round-trip time", Number.isFinite(rttMs) ? formatMs(rttMs) : ""],
    ["HTTP / API status", `${httpStatus} ${s}`],
    ["Warnings", Array.isArray(body.warnings) ? body.warnings.join(", ") : ""],
    ["Request ID", str(body.request_id)],
  ].filter(([, v]) => v !== "");

  return {
    ...base, ok: true, kind: s, applied, finalOutput, fallback, tone, headline, label,
    message: gateText, rows,
    imageSrc: `data:${mime};base64,${body.image_b64}`,
    chipSrc: body.chip_b64 ? `data:image/png;base64,${body.chip_b64}` : "",
    record: {
      status: s, http_status: httpStatus, applied, final_output: finalOutput, model: model || "none",
      gate: gate, reason, fallback, artifact_guard: profile === "original" ? "" : guard,
      local_detector: local, server_ms: Number.isFinite(t.total) ? t.total : null,
      rtt_ms: Number.isFinite(rttMs) ? Math.round(rttMs) : null, request_id: str(body.request_id),
    },
  };
}

/** One manual-evaluation record (no pixels). `ratings[profile] = {rating, tags[], notes}` for r5_6 / r5_8. */
export function makeRecord({ fileName, fileSize, dims, category, overallNote, results, ratings }, now = new Date()) {
  const rec = { recorded_at: now.toISOString(), file_name: str(fileName), file_size: fileSize ?? null,
    dims: str(dims), category: str(category), overall_note: str(overallNote), profiles: {} };
  for (const id of PROFILE_IDS) {
    const r = results && results[id];
    const e = (ratings && ratings[id]) || {};
    if (!r && !e.rating) continue;
    rec.profiles[id] = { ...(r ? r.record : {}), rating: str(e.rating),
      tags: Array.isArray(e.tags) ? e.tags.filter((x) => TAGS.includes(x)) : [], notes: str(e.notes) };
  }
  return rec;
}

/** A record can be saved once at least one NAFNet profile has a result AND a rating. */
export function canSave(results, ratings) {
  return ["r5_6", "r5_8"].some((id) => results && results[id] && ratings && ratings[id] && RATINGS.includes(ratings[id].rating));
}

export function recordsToJSON(records) {
  return JSON.stringify({ kind: "fes-r5-9-manual-evaluation", version: 1,
    note: "Owner's visual judgement only. Not an identity or matching measurement.", records }, null, 2);
}

const CSV_PROFILE_FIELDS = ["rating", "tags", "notes", "applied", "final_output", "model", "gate", "fallback",
  "artifact_guard", "local_detector", "server_ms", "http_status"];
const csvCell = (v) => {
  const s = Array.isArray(v) ? v.join("; ") : v === null || v === undefined ? "" : String(v);
  // quote everything; neutralise spreadsheet formula injection from free-text notes
  return `"${(/^[=+\-@]/.test(s) ? `'${s}` : s).replace(/"/g, '""')}"`;
};

export function recordsToCSV(records) {
  const head = ["recorded_at", "file_name", "dims", "category", "overall_note"];
  const cols = [...head];
  for (const id of ["r5_6", "r5_8"]) for (const f of CSV_PROFILE_FIELDS) cols.push(`${id}_${f}`);
  const lines = [cols.map(csvCell).join(",")];
  for (const r of records) {
    const row = head.map((k) => r[k]);
    for (const id of ["r5_6", "r5_8"]) for (const f of CSV_PROFILE_FIELDS) row.push((r.profiles[id] || {})[f]);
    lines.push(row.map(csvCell).join(","));
  }
  return lines.join("\r\n") + "\r\n";
}
