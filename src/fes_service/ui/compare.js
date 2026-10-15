// R5-9 / R5-10 — DOM wiring of the comparison page. The service makes every decision; see fes_compare.js for the logic.
import { ACCEPTED_TYPES, DEFAULT_LIMITS, REQUEST_TIMEOUT_MS, formatBytes, interpretReady, validateDimensions,
  validateFile } from "./fes_ui.js";
import { CATEGORIES, EXPERIMENTAL_IDS, NAFNET_IDS, PROFILES, PROFILE_IDS, RATINGS, TAGS, availableProfiles,
  buildProfileForm, canSave, interpretProfileResult, makeRecord, profileById, recordsToCSV, recordsToJSON } from "./fes_compare.js";

const $ = (id) => document.getElementById(id);
const el = {
  health: $("health"), healthLabel: $("health-label"), drop: $("dropzone"), input: $("file-input"),
  dropEmpty: $("drop-empty"), preview: $("preview"), fileRow: $("file-row"), fileName: $("file-name"),
  replace: $("replace-btn"), runOne: $("run-one-btn"), runAll: $("run-all-btn"), reset: $("reset-btn"),
  loading: $("loading"), loadingText: $("loading-text"), elapsed: $("elapsed"), error: $("error"),
  formatHint: $("format-hint"), panels: $("panels"), profilesNote: $("profiles-note"),
  eval: $("eval"), category: $("category"), overall: $("overall-note"), evalGrid: $("eval-grid"), save: $("save-btn"),
  saveHint: $("save-hint"), recCount: $("rec-count"), exportJson: $("export-json"), exportCsv: $("export-csv"),
  clearLog: $("clear-log"), logBody: document.querySelector("#log tbody"),
};
const state = { file: null, url: "", dims: "", validDims: false, limits: { ...DEFAULT_LIMITS }, busy: false,
  results: {}, pending: new Set(), records: [], unexported: false, available: null };

const make = (tag, props = {}, ...kids) => {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "class") n.className = v; else if (k === "dataset") Object.assign(n.dataset, v); else n[k] = v;
  }
  n.append(...kids.filter((k) => k !== null && k !== undefined && k !== false));
  return n;
};

// ------------------------------------------------------------------ health
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
  if (body && Array.isArray(body.profiles)) {
    state.available = availableProfiles(body);
    const off = EXPERIMENTAL_IDS.filter((id) => !state.available.includes(id));
    for (const id of EXPERIMENTAL_IDS) $(`mode-${id}`).classList.toggle("unavailable", off.includes(id));
    $("mode-r5_10").hidden = off.includes("r5_10");                  // R5-10 is listed only when enabled
    $("mode-r6").hidden = off.includes("r6");                        // R6 is listed only when enabled
    $("mode-r6_ft").hidden = off.includes("r6_ft");                  // R6-FT is listed only when enabled
    const offShown = off.filter((id) => id === "r5_8");             // a disabled R5-10 / R6 / R6-FT is simply not listed
    el.profilesNote.hidden = !offShown.length;
    el.profilesNote.textContent = !offShown.length ? "" : `${offShown.map((id) => profileById(id).title).join(" and ")} `
      + `${offShown.length > 1 ? "are" : "is"} not enabled on this service (off by default). Start the service with `
      + `FES_EXPERIMENTAL_PROFILES=${EXPERIMENTAL_IDS.join(",")} to compare; requests for it are refused until then.`;
  }
}

// ------------------------------------------------------------------ upload
function showError(title, message) {
  el.error.replaceChildren(make("strong", { textContent: title }), document.createTextNode(message));
  el.error.hidden = false;
}
const clearError = () => { el.error.hidden = true; el.error.replaceChildren(); };

function refresh() {
  const ready = !!state.file && state.validDims && !state.busy;
  el.runOne.disabled = el.runAll.disabled = !ready;
  el.replace.disabled = el.reset.disabled = state.busy;
  el.drop.classList.toggle("disabled", state.busy);
  document.querySelectorAll('input[name="mode"]').forEach((r) => { r.disabled = state.busy; });
  el.save.disabled = state.busy || !canSave(state.results, readRatings());
  el.saveHint.textContent = el.save.disabled && !state.busy && Object.keys(state.results).length
    ? "Rate at least one NAFNet profile to save." : "";
  const n = state.records.length;
  el.recCount.textContent = String(n);
  el.exportJson.disabled = el.exportCsv.disabled = el.clearLog.disabled = !n || state.busy;
}

function clearImage() {
  if (state.url) URL.revokeObjectURL(state.url);
  Object.assign(state, { file: null, url: "", dims: "", validDims: false, results: {} });
  state.pending.clear();
  el.input.value = "";
  el.preview.hidden = true; el.preview.removeAttribute("src");
  el.dropEmpty.hidden = false; el.fileRow.hidden = true;
  el.drop.classList.remove("has-file");
  resetEvalForm();
  renderPanels();
}

function selectFile(file) {
  if (state.busy) return;
  clearError();
  const v = validateFile(file, state.limits);
  if (!v.ok) { clearImage(); showError("Cannot use this file. ", v.error); return; }
  clearImage();
  state.file = file;
  state.url = URL.createObjectURL(file);
  el.preview.onload = () => {
    const w = el.preview.naturalWidth, h = el.preview.naturalHeight;
    const d = validateDimensions(w, h, state.limits);
    state.validDims = d.ok; state.dims = `${w}×${h}`;
    if (!d.ok) showError("Cannot use this image. ", d.error);
    el.fileName.textContent = `${file.name} · ${formatBytes(file.size)} · ${state.dims}`;
    renderPanels(); refresh();
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
["dragenter", "dragover"].forEach((t) => el.drop.addEventListener(t, (e) => { e.preventDefault(); if (!state.busy) el.drop.classList.add("drag"); }));
["dragleave", "drop"].forEach((t) => el.drop.addEventListener(t, (e) => { e.preventDefault(); el.drop.classList.remove("drag"); }));
el.drop.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) selectFile(f); });
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", (e) => e.preventDefault());

// ------------------------------------------------------------------ requests
async function runProfile(profile) {
  const t0 = performance.now();
  const ctrl = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => { timedOut = true; ctrl.abort(); }, REQUEST_TIMEOUT_MS);
  let status = 0, body = null;
  try {
    const r = await fetch("/v1/enhance", { method: "POST", body: buildProfileForm(state.file, profile), signal: ctrl.signal });
    status = r.status; body = await r.json().catch(() => null);
  } catch { /* network error / timeout → status 0 */ } finally { clearTimeout(timer); }
  return interpretProfileResult(profile, status, body, performance.now() - t0, { timedOut });
}

async function run(profiles) {
  if (!state.file || !state.validDims || state.busy) return;     // duplicate-submission guard
  state.busy = true; clearError();
  const file = state.file;
  profiles.forEach((p) => { delete state.results[p]; state.pending.add(p); });
  renderPanels(); refresh();
  el.loading.hidden = false;
  const t0 = performance.now();
  const tick = setInterval(() => { el.elapsed.textContent = `${Math.floor((performance.now() - t0) / 1000)} s elapsed`; }, 500);
  el.elapsed.textContent = "0 s elapsed";
  try {
    for (const [i, p] of profiles.entries()) {
      el.loadingText.textContent = `Running ${profileById(p).title} (${i + 1}/${profiles.length})… NAFNet on CPU can take several seconds.`;
      const view = await runProfile(p);
      if (state.file !== file) return;                             // image replaced meanwhile (not possible while busy)
      state.results[p] = view; state.pending.delete(p);
      renderPanels();
    }
  } finally {
    clearInterval(tick);
    el.loading.hidden = true; state.busy = false; state.pending.clear();
    renderPanels(); renderEvalForm(); refresh();
  }
}
const selectedMode = () => (document.querySelector('input[name="mode"]:checked') || {}).value || "original";
el.runOne.addEventListener("click", () => run([selectedMode()]));
// R5-10: an experimental profile the service has not enabled is neither run nor shown (R5-9 behaviour when it is off)
const shownIds = () => PROFILE_IDS.filter((id) => !EXPERIMENTAL_IDS.includes(id) || id === "r5_8"
  || (state.available || []).includes(id));
el.runAll.addEventListener("click", () => run(shownIds()));

// ------------------------------------------------------------------ panels
function dl(rows) {
  const d = make("dl");
  for (const [k, v] of rows) d.append(make("dt", { textContent: k }), make("dd", { textContent: v }));
  return d;
}

function renderPanels() {
  el.panels.replaceChildren();
  if (!state.file) return;
  const chip = PROFILE_IDS.map((p) => state.results[p]?.chipSrc).find(Boolean);
  for (const prof of PROFILES.filter((p) => shownIds().includes(p.id))) {
    const v = state.results[prof.id];
    const head = [make("h3", { textContent: prof.title }),
      prof.warning ? make("p", { class: `pwarn${EXPERIMENTAL_IDS.includes(prof.id) ? " exp" : ""}`, textContent: prof.warning }) : null];
    const panel = make("article", { class: "panel", dataset: { tone: v ? v.tone : "neutral", profile: prof.id } }, ...head);
    if (prof.id === "original") {
      // the original is ALWAYS visible, whether or not its request has run
      panel.append(make("p", { class: "headline", textContent: "ORIGINAL (uploaded image)" }),
        make("div", { class: "frame" }, make("img", { src: state.url, alt: "Original uploaded image" })));
      if (chip) panel.append(make("div", { class: "sub-frame" }, make("img", { src: chip, alt: "Face crop the service analysed" }),
        make("span", { textContent: "Face crop the service analysed (what the NAFNet outputs below correspond to)." })));
      if (v) panel.append(v.ok ? dl(v.rows) : make("p", { class: "note", textContent: `${v.headline}: ${v.message}` }));
    } else if (state.pending.has(prof.id)) {
      panel.append(make("div", { class: "placeholder" }, make("span", { class: "spinner", ariaHidden: "true" }),
        make("span", { textContent: state.busy ? "Processing…" : "Waiting…" })));
    } else if (!v) {
      panel.append(make("div", { class: "placeholder", textContent: "Not run yet for this image." }));
    } else if (!v.ok) {
      panel.append(make("p", { class: "headline", textContent: v.headline }),
        make("div", { class: "placeholder", textContent: v.message }), dl(v.rows));
    } else {
      const frame = make("div", { class: `frame${v.applied ? "" : " unchanged"}` },
        make("img", { src: v.imageSrc, alt: `${prof.title}: ${v.label}` }));
      if (!v.applied) frame.append(make("div", { class: `stamp${v.fallback ? " bad" : ""}`,
        textContent: v.fallback ? `ORIGINAL — enhancement not applied (fallback: ${v.fallback})` : "ORIGINAL — enhancement not applied" }));
      panel.append(make("p", { class: "headline", textContent: v.headline }), frame,
        make("p", { class: "note", textContent: v.label }), dl(v.rows));
    }
    el.panels.append(panel);
  }
}

// ------------------------------------------------------------------ manual evaluation
function renderEvalForm() {
  const have = NAFNET_IDS.filter((p) => state.results[p]?.ok);
  el.eval.hidden = !have.length;
  if (el.evalGrid.childElementCount) {                                 // keep what the owner already entered
    for (const col of el.evalGrid.children) col.classList.toggle("disabled", !have.includes(col.dataset.profile));
    return;
  }
  for (const id of NAFNET_IDS) {
    const col = make("div", { class: "eval-col", dataset: { profile: id } }, make("h3", { textContent: profileById(id).title }));
    const ratings = make("div", { class: "ratings", role: "radiogroup", ariaLabel: `${id} rating` });
    for (const r of RATINGS) ratings.append(make("label", {}, make("input", { type: "radio", name: `rate-${id}`, value: r }), ` ${r}`));
    const tags = make("div", { class: "tags" });
    for (const t of TAGS) tags.append(make("label", {}, make("input", { type: "checkbox", name: `tag-${id}`, value: t }), ` ${t}`));
    col.append(ratings, tags, make("textarea", { name: `notes-${id}`, maxLength: 1000, placeholder: "Notes (optional)" }));
    col.classList.toggle("disabled", !have.includes(id));
    el.evalGrid.append(col);
  }
}

function readRatings() {
  const out = {};
  for (const id of NAFNET_IDS) {
    out[id] = {
      rating: (document.querySelector(`input[name="rate-${id}"]:checked`) || {}).value || "",
      tags: [...document.querySelectorAll(`input[name="tag-${id}"]:checked`)].map((x) => x.value),
      notes: (document.querySelector(`textarea[name="notes-${id}"]`) || {}).value || "",
    };
  }
  return out;
}

function resetEvalForm() {
  el.evalGrid.replaceChildren(); el.eval.hidden = true;
  el.category.value = ""; el.overall.value = "";
}

el.eval.addEventListener("change", refresh);
el.save.addEventListener("click", () => {
  const ratings = readRatings();
  if (!canSave(state.results, ratings)) return;
  state.records.push(makeRecord({ fileName: state.file?.name, fileSize: state.file?.size, dims: state.dims,
    category: el.category.value, overallNote: el.overall.value, results: state.results, ratings }));
  state.unexported = true;
  renderLog();
  el.save.textContent = "Saved ✓";
  setTimeout(() => { el.save.textContent = "Save evaluation"; }, 1500);
  refresh();
});

function renderLog() {
  el.logBody.replaceChildren();
  state.records.forEach((r, i) => {
    const p = NAFNET_IDS.map((id) => r.profiles[id] || {});
    el.logBody.append(make("tr", {}, ...[i + 1, r.file_name, r.category, ...p.map((x) => x.rating),
      ...p.map((x) => x.final_output)].map((x) => make("td", { textContent: x ?? "" }))));
  });
}

function download(text, type, ext) {
  const stamp = new Date().toISOString().replace(/[:T]/g, "-").slice(0, 16);
  const a = make("a", { href: URL.createObjectURL(new Blob([text], { type })), download: `r5_9_manual_eval_${stamp}.${ext}` });
  document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
  state.unexported = false;
}
el.exportJson.addEventListener("click", () => download(recordsToJSON(state.records), "application/json", "json"));
el.exportCsv.addEventListener("click", () => download(recordsToCSV(state.records), "text/csv", "csv"));
el.clearLog.addEventListener("click", () => {
  if (state.unexported) { showError("Log not exported. ", "Export it first, or press Clear log again to discard it."); state.unexported = false; return; }
  state.records = []; renderLog(); refresh();
});
window.addEventListener("beforeunload", (e) => { if (state.unexported) { e.preventDefault(); e.returnValue = ""; } });

// ------------------------------------------------------------------ reset
el.reset.addEventListener("click", () => {
  if (state.busy) return;
  clearImage(); clearError();
  document.querySelector('input[name="mode"][value="original"]').checked = true;
  refresh();
  window.scrollTo({ top: 0, behavior: "smooth" });
});

for (const c of CATEGORIES) el.category.append(make("option", { value: c, textContent: c }));
el.input.accept = ACCEPTED_TYPES.join(",");
refresh();
checkHealth();
setInterval(checkHealth, 15000);
