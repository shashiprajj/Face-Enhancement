// R5-UI — DOM wiring of the demo page. All decisions are made by the service; see fes_ui.js for the logic.
import {
  ACCEPTED_TYPES, DEFAULT_LIMITS, PROCESSING_MESSAGE, REQUEST_TIMEOUT_MS, buildEnhanceForm, formatBytes, formatMs,
  interpretNetworkError, interpretReady, interpretResponse, validateDimensions, validateFile,
} from "./fes_ui.js";

const $ = (id) => document.getElementById(id);
const el = {
  health: $("health"), healthLabel: $("health-label"), drop: $("dropzone"), input: $("file-input"),
  dropEmpty: $("drop-empty"), preview: $("preview"), fileRow: $("file-row"), fileName: $("file-name"),
  replace: $("replace-btn"), remove: $("remove-btn"), enhance: $("enhance-btn"), loading: $("loading"),
  loadingText: $("loading-text"), elapsed: $("elapsed"), error: $("error"), formatHint: $("format-hint"),
  result: $("result"), badge: $("status-badge"), chip: $("decision-chip"), title: $("result-title"),
  message: $("result-message"), reasonText: $("reason-text"), orig: $("orig-img"), proc: $("proc-img"),
  outputNote: $("output-note"), warnings: $("warnings"), details: $("details"), reset: $("reset-btn"),
  force: $("force-a4"), forceBtn: $("force-btn"), leftLabel: $("left-label"), rightLabel: $("right-label"),
};

const state = { file: null, url: "", limits: { ...DEFAULT_LIMITS }, busy: false, validDims: false };

// ------------------------------------------------------------------ health (/ready)
async function checkHealth() {
  let status = 0, body = null;
  try {
    const r = await fetch("/ready", { cache: "no-store" });
    status = r.status;
    body = await r.json().catch(() => null);
  } catch { /* offline */ }
  const h = interpretReady(status, body);
  el.health.dataset.state = h.state;
  el.healthLabel.textContent = h.label;
  if (h.limits) {
    state.limits = h.limits;
    el.formatHint.textContent = `JPEG, PNG, WebP or BMP · up to ${formatBytes(h.limits.max_bytes)}`;
  }
}

// ------------------------------------------------------------------ upload
function showError(title, message) {
  el.error.replaceChildren();
  const t = document.createElement("strong");
  t.textContent = title;
  el.error.append(t, document.createTextNode(message));
  el.error.hidden = false;
}
const clearError = () => { el.error.hidden = true; el.error.replaceChildren(); };

function refreshButtons() {
  el.enhance.disabled = !state.file || !state.validDims || state.busy;
  el.replace.disabled = el.remove.disabled = el.force.disabled = el.forceBtn.disabled = state.busy;
  el.drop.classList.toggle("disabled", state.busy);
}

function clearFile() {
  if (state.url) URL.revokeObjectURL(state.url);
  state.file = null; state.url = ""; state.validDims = false;
  el.input.value = "";
  el.preview.hidden = true; el.preview.removeAttribute("src");
  el.dropEmpty.hidden = false; el.fileRow.hidden = true;
  el.drop.classList.remove("has-file");
  refreshButtons();
}

function selectFile(file) {
  if (state.busy) return;
  clearError();
  el.result.hidden = true;
  const v = validateFile(file, state.limits);
  if (!v.ok) { clearFile(); showError("Cannot use this file. ", v.error); return; }
  clearFile();
  state.file = file;
  state.url = URL.createObjectURL(file);
  el.preview.onload = () => {
    const d = validateDimensions(el.preview.naturalWidth, el.preview.naturalHeight, state.limits);
    state.validDims = d.ok;
    if (!d.ok) showError("Cannot use this image. ", d.error);
    el.fileName.textContent = `${file.name} · ${formatBytes(file.size)} · ${el.preview.naturalWidth}×${el.preview.naturalHeight}`;
    refreshButtons();
  };
  el.preview.onerror = () => {
    state.validDims = false;
    showError("Cannot use this image. ", "The browser could not read it — the file may be corrupted.");
    refreshButtons();
  };
  el.preview.src = state.url;
  el.preview.hidden = false; el.dropEmpty.hidden = true; el.fileRow.hidden = false;
  el.fileName.textContent = `${file.name} · ${formatBytes(file.size)}`;
  el.drop.classList.add("has-file");
  refreshButtons();
}

const browse = () => { if (!state.busy) el.input.click(); };
el.drop.addEventListener("click", () => { if (!state.file) browse(); });
el.drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); browse(); } });
el.input.addEventListener("change", () => { if (el.input.files[0]) selectFile(el.input.files[0]); });
el.replace.addEventListener("click", browse);
el.remove.addEventListener("click", () => { clearFile(); clearError(); el.result.hidden = true; });
["dragenter", "dragover"].forEach((t) => el.drop.addEventListener(t, (e) => {
  e.preventDefault(); if (!state.busy) el.drop.classList.add("drag");
}));
["dragleave", "drop"].forEach((t) => el.drop.addEventListener(t, (e) => { e.preventDefault(); el.drop.classList.remove("drag"); }));
el.drop.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) selectFile(f); });
// a file dropped outside the zone must not navigate away from the page
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", (e) => e.preventDefault());

// ------------------------------------------------------------------ enhance (/v1/enhance)
async function enhance(force = el.force.checked) {
  if (!state.file || state.busy) return;                  // duplicate-submission guard
  state.busy = true; refreshButtons(); clearError(); el.result.hidden = true;
  el.loadingText.textContent = PROCESSING_MESSAGE;
  el.loading.hidden = false;
  const t0 = performance.now();
  const tick = setInterval(() => { el.elapsed.textContent = `${Math.floor((performance.now() - t0) / 1000)} s elapsed`; }, 500);
  el.elapsed.textContent = "0 s elapsed";
  const ctrl = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => { timedOut = true; ctrl.abort(); }, REQUEST_TIMEOUT_MS);
  let view;
  try {
    const r = await fetch("/v1/enhance", { method: "POST", body: buildEnhanceForm(state.file, { force }), signal: ctrl.signal });
    const body = await r.json().catch(() => null);
    view = interpretResponse(r.status, body);
  } catch (e) {
    view = interpretNetworkError(e, timedOut);
  } finally {
    clearTimeout(timer); clearInterval(tick);
    el.loading.hidden = true; state.busy = false; refreshButtons();
  }
  const rtt = performance.now() - t0;
  if (view.imageSrc) renderResult(view, rtt);
  else { showError(`${view.title}. `, view.message); checkHealth(); }
}
el.enhance.addEventListener("click", () => enhance());
el.forceBtn.addEventListener("click", () => enhance(true));

function renderResult(v, rtt) {
  el.badge.textContent = v.forced ? `${v.kind} · FORCED` : v.kind;
  el.badge.dataset.tone = v.tone;
  el.chip.hidden = !v.decision; el.chip.textContent = v.decision;
  el.title.textContent = v.title;
  el.message.textContent = v.message;
  el.reasonText.hidden = !v.reason;
  el.reasonText.textContent = v.reason ? `Reason: ${v.reason}${v.reasonText ? ` — ${v.reasonText}` : ""}` : "";
  el.orig.src = v.leftSrc || state.url;
  el.proc.src = v.imageSrc;
  el.leftLabel.textContent = v.leftLabel;
  el.rightLabel.textContent = v.rightLabel;
  el.forceBtn.hidden = !v.canForce;
  el.outputNote.textContent = v.outputNote;
  el.warnings.hidden = !v.warnings.length;
  el.warnings.textContent = v.warnings.length ? `Service warnings: ${v.warnings.join(", ")}` : "";
  el.details.replaceChildren();
  for (const [k, val] of [...v.details, ["Round-trip time", formatMs(rtt)]]) {
    const dt = document.createElement("dt"); dt.textContent = k;
    const dd = document.createElement("dd"); dd.textContent = val;
    el.details.append(dt, dd);
  }
  el.result.hidden = false;
  el.result.scrollIntoView({ behavior: "smooth", block: "start" });
}

// ------------------------------------------------------------------ reset
el.reset.addEventListener("click", () => {
  if (state.busy) return;
  clearFile(); clearError();
  el.result.hidden = true; el.force.checked = false;
  el.orig.removeAttribute("src"); el.proc.removeAttribute("src"); el.details.replaceChildren();
  window.scrollTo({ top: 0, behavior: "smooth" });
});

el.input.accept = ACCEPTED_TYPES.join(",");
checkHealth();
setInterval(checkHealth, 15000);
