// R6-BEST — DOM wiring of the single-output page. The service makes every decision; see fes_best.js for the logic.
import { ACCEPTED_TYPES, DEFAULT_LIMITS, REQUEST_TIMEOUT_MS, formatBytes, interpretReady, validateDimensions,
  validateFile } from "./fes_ui.js";
import { ENDPOINT, bestAvailable, blurSettingsView, buildBestForm, enhanceEnabled, interpretBestResult,
  serviceNote } from "./fes_best.js";

const $ = (id) => document.getElementById(id);
const el = {
  health: $("health"), healthLabel: $("health-label"), drop: $("dropzone"), input: $("file-input"),
  dropEmpty: $("drop-empty"), preview: $("preview"), fileRow: $("file-row"), fileName: $("file-name"),
  replace: $("replace-btn"), remove: $("remove-btn"), enhance: $("enhance-btn"), loading: $("loading"),
  elapsed: $("elapsed"), error: $("error"), formatHint: $("format-hint"), modeNote: $("mode-note"),
  result: $("result"), badge: $("status-badge"), title: $("result-title"), message: $("result-message"),
  orig: $("orig-img"), proc: $("proc-img"), leftLabel: $("left-label"), rightLabel: $("right-label"),
  details: $("details"), reset: $("reset-btn"), blurPanel: $("blur-panel"), blurService: $("blur-service"),
  blurFrom: $("blur-from"), blurUpto: $("blur-upto"), blurMin: $("blur-min"),
};
// service: "unknown" until the first /ready answer, then interpretReady's state (online | loading | offline)
const state = { file: null, url: "", limits: { ...DEFAULT_LIMITS }, busy: false, validDims: false, available: true,
  service: "unknown" };

async function checkHealth() {
  let status = 0, body = null;
  try {
    const r = await fetch("/ready", { cache: "no-store" });
    status = r.status; body = await r.json().catch(() => null);
  } catch { /* offline */ }
  const h = interpretReady(status, body);
  el.health.dataset.state = h.state;
  el.healthLabel.textContent = h.label;
  if (h.limits) {
    state.limits = h.limits;
    el.formatHint.textContent = `JPEG, PNG, WebP or BMP · up to ${formatBytes(h.limits.max_bytes)}`;
  }
  state.service = h.state;
  if (body && Array.isArray(body.profiles)) state.available = bestAvailable(body);
  if (body) {
    const b = blurSettingsView(body.blur_settings);
    el.blurPanel.hidden = !b.available;
    el.blurService.textContent = b.text;
    el.blurMin.textContent = b.minFrom !== null ? `Lowest allowed "enhance from": ${b.minFrom} %.` : "";
    if (b.minFrom !== null) el.blurFrom.min = String(b.minFrom);
  }
  refresh();
}

function showError(title, message) {
  const t = document.createElement("strong");
  t.textContent = title;
  el.error.replaceChildren(t, document.createTextNode(message));
  el.error.hidden = false;
}
const clearError = () => { el.error.hidden = true; el.error.replaceChildren(); };

function refresh() {
  el.enhance.disabled = !enhanceEnabled(state);
  const note = serviceNote(state.service, state.available);     // why Enhance is disabled, when the service is the reason
  el.modeNote.textContent = note;
  el.modeNote.hidden = !note;
  el.replace.disabled = el.remove.disabled = el.blurFrom.disabled = el.blurUpto.disabled = state.busy;
  el.drop.classList.toggle("disabled", state.busy);
}

function clearFile() {
  if (state.url) URL.revokeObjectURL(state.url);
  state.file = null; state.url = ""; state.validDims = false;
  el.input.value = "";
  el.preview.hidden = true; el.preview.removeAttribute("src");
  el.dropEmpty.hidden = false; el.fileRow.hidden = true;
  el.drop.classList.remove("has-file");
  refresh();
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
    refresh();
  };
  el.preview.onerror = () => { state.validDims = false; showError("Cannot use this image. ", "The browser could not read it."); refresh(); };
  el.preview.src = state.url;
  el.preview.hidden = false; el.dropEmpty.hidden = true; el.fileRow.hidden = false;
  el.fileName.textContent = `${file.name} · ${formatBytes(file.size)}`;
  el.drop.classList.add("has-file");
  refresh();
}

const browse = () => { if (!state.busy) el.input.click(); };
el.drop.addEventListener("click", () => { if (!state.file) browse(); });
el.drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); browse(); } });
el.input.addEventListener("change", () => { if (el.input.files[0]) selectFile(el.input.files[0]); });
el.replace.addEventListener("click", browse);
el.remove.addEventListener("click", () => { clearFile(); clearError(); el.result.hidden = true; });
["dragenter", "dragover"].forEach((t) => el.drop.addEventListener(t, (e) => { e.preventDefault(); if (!state.busy) el.drop.classList.add("drag"); }));
["dragleave", "drop"].forEach((t) => el.drop.addEventListener(t, (e) => { e.preventDefault(); el.drop.classList.remove("drag"); }));
el.drop.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) selectFile(f); });
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", (e) => e.preventDefault());

async function enhance() {
  if (!state.file || state.busy) return;                  // duplicate-submission guard
  state.busy = true; refresh(); clearError(); el.result.hidden = true;
  el.loading.hidden = false;
  const t0 = performance.now();
  const tick = setInterval(() => { el.elapsed.textContent = `${Math.floor((performance.now() - t0) / 1000)} s elapsed`; }, 500);
  el.elapsed.textContent = "0 s elapsed";
  const ctrl = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => { timedOut = true; ctrl.abort(); }, REQUEST_TIMEOUT_MS);
  let status = 0, body = null;
  try {
    const r = await fetch(ENDPOINT, { method: "POST", body: buildBestForm(state.file, FormData,
      { enhanceFrom: el.blurFrom.value, giveUpAbove: el.blurUpto.value }), signal: ctrl.signal });
    status = r.status; body = await r.json().catch(() => null);
  } catch { /* network error / timeout → status 0 */ } finally {
    clearTimeout(timer); clearInterval(tick);
    el.loading.hidden = true; state.busy = false; refresh();
  }
  const view = interpretBestResult(status, body, performance.now() - t0, { timedOut });
  if (view.ok) renderResult(view);
  else { showError(`${view.title}. `, view.message); checkHealth(); }
}
el.enhance.addEventListener("click", enhance);

function renderResult(v) {
  el.badge.textContent = v.applied ? "ENHANCED" : "ORIGINAL";
  el.badge.dataset.tone = v.tone;
  el.title.textContent = v.title;
  el.message.textContent = v.message;
  el.orig.src = v.leftSrc || state.url;
  el.proc.src = v.imageSrc;
  el.leftLabel.textContent = v.leftLabel;
  el.rightLabel.textContent = v.rightLabel;
  el.details.replaceChildren();
  for (const [k, val] of v.rows) {
    const dt = document.createElement("dt"); dt.textContent = k;
    const dd = document.createElement("dd"); dd.textContent = val;
    el.details.append(dt, dd);
  }
  el.result.hidden = false;
  el.result.scrollIntoView({ behavior: "smooth", block: "start" });
}

el.reset.addEventListener("click", () => {
  if (state.busy) return;
  clearFile(); clearError();
  el.result.hidden = true;
  el.orig.removeAttribute("src"); el.proc.removeAttribute("src"); el.details.replaceChildren();
  window.scrollTo({ top: 0, behavior: "smooth" });
});

el.input.accept = ACCEPTED_TYPES.join(",");
refresh();
checkHealth();
setInterval(checkHealth, 15000);
