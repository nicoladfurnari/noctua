// Noctua (package astroforge) wizard. Plain JS, no build step.
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
// an <img> shown from a blob: the previous object URL is released (every preview, zoom and viewer frame made a
// new one and the browser kept them all until the page was closed: hundreds of MB in a long session)
const showBlob = (img, blob) => { if (img.src.startsWith("blob:")) URL.revokeObjectURL(img.src); img.src = URL.createObjectURL(blob); };
const apiResp = async (url, body) => {
  const r = await fetch(url, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!r.ok) {
    let msg = await r.text();
    try { msg = JSON.parse(msg).detail || msg; } catch (e) {}
    throw new Error(r.status === 401 || r.status === 403 ? msg : `${url}: ${r.status} ${msg}`);
  }
  return r;
};
const api = async (url, body) => {
  const r = await apiResp(url, body);
  return r.headers.get("content-type")?.includes("json") ? r.json() : r.blob();
};
const imgUrl = (path, side = 1600) => `/api/image?path=${encodeURIComponent(path)}&max_side=${side}`;
// Lucide icon from the bundled sprite (static/icons.svg)
const ico = (name, cls = "") => `<svg class="ic${cls ? " " + cls : ""}" aria-hidden="true"><use href="/static/icons.svg#i-${name}"/></svg>`;

const state = JSON.parse(localStorage.getItem("af_state") || "null") || {
  lights: [], calibration: [], groupIndex: 0, exclude: [], input: null, steps: [], preset: "nebula",
};
if (!Array.isArray(state.panels)) state.panels = [[], []];
if (!Array.isArray(state.masterDirs)) state.masterDirs = [];
state.mosaic = !!state.mosaic;
if (!["single", "mosaic", "multi", "hybrid"].includes(state.mode)) state.mode = state.mosaic ? "mosaic" : "single";
if (!Array.isArray(state.sources)) state.sources = [];
if (!state.groupSettings || typeof state.groupSettings !== "object") state.groupSettings = {};
const kwList = v => String(v || "").split(/[,;\s]+/).map(x => x.trim().toUpperCase()).filter(Boolean);
const save = () => { try { localStorage.setItem("af_state", JSON.stringify(state)); } catch (e) {} };
const multiMode = () => state.mode === "multi" || state.mode === "hybrid";
// light folders actually used: the single field, every panel of the mosaic, or every source
const lightRoots = () => state.mosaic ? state.panels.flat() : multiMode() ? state.sources.flatMap(x => x.file ? [] : x.lights) : state.lights;
let scanResult = null, plugins = [], presets = [], stackResult = null;

// ---------------------------------------------------------------- navigation
function go(n) {
  n = +n;
  if (n === 2) setTimeout(() => showPlan(), 50);
  $$(".page").forEach(p => p.classList.toggle("on", +p.dataset.page === n));
  $$("#steps li").forEach(li => {
    li.classList.toggle("on", +li.dataset.step === n);
    li.classList.toggle("done", +li.dataset.step < n);
  });
  if (n === 5) enterPost();
  if (n === 6) enterExport();
  window.scrollTo(0, 0);
}
$$("#steps li").forEach(li => li.onclick = () => go(li.dataset.step));
document.addEventListener("click", e => { const b = e.target.closest("[data-go]"); if (b) go(b.dataset.go); });

// ---------------------------------------------------------------- folder browser
let browseMode = null, browsePath = "", browseFile = null;
async function openBrowser(mode, start) {
  browseMode = mode; browseFile = null;
  $("#bOk").textContent = mode === "post" || mode.endsWith(":file") ? "Usa il file selezionato"
    : mode === "ref" ? "Crea il profilo da questa immagine" : "Scegli questa cartella";
  $("#browser").hidden = false;
  await loadDir(start || localStorage.getItem("af_lastdir") || "");
}
async function loadDir(path) {
  let d;
  try {
    d = await api(`/api/browse?path=${encodeURIComponent(path)}${browseMode === "ref" || browseMode.endsWith(":file") ? "&images=true" : ""}`);
  } catch (e) {
    if (path) return loadDir("");             // the last folder no longer exists (or a drive was removed)
    browsePath = ""; $("#bPath").textContent = "Risorse del computer";
    $("#bList").innerHTML = `<li class="muted">Il server di Noctua non risponde: riavvialo con start-gui.cmd, poi ricarica la pagina. (${esc(e.message.slice(0, 120))})</li>`;
    return;
  }
  browsePath = d.path;
  $("#bPath").textContent = d.path || "Risorse del computer";
  $("#bUp").dataset.parent = d.parent ?? "";
  const ul = $("#bList"); ul.innerHTML = "";
  d.dirs.forEach(name => {
    const li = document.createElement("li");
    li.innerHTML = ico("folder") + `<span>${esc(name)}</span>`;
    li.onclick = () => loadDir(d.path ? `${d.path}\\${name}`.replace(/\\\\/g, "\\") : name);
    ul.appendChild(li);
  });
  if (browseMode === "post" || browseMode === "ref" || browseMode.endsWith(":file")) d.files.forEach(name => {
    const li = document.createElement("li");
    li.className = "file"; li.innerHTML = ico("image") + `<span>${esc(name)}</span>`;
    li.onclick = () => { $$("#bList li").forEach(x => x.classList.remove("sel")); li.classList.add("sel");
      browseFile = `${d.path}\\${name}`; };
    ul.appendChild(li);
  });
  else if (d.fits) {
    const li = document.createElement("li"); li.className = "muted"; li.textContent = `${d.fits} file FITS in questa cartella`;
    ul.appendChild(li);
  }
  if (d.path) try { localStorage.setItem("af_lastdir", d.path); } catch (e) {}
}
$("#bUp").onclick = () => loadDir($("#bUp").dataset.parent);
$("#bCancel").onclick = () => $("#browser").hidden = true;
$("#bOk").onclick = () => {
  if (browseMode === "ref") {
    if (!browseFile) return;
    $("#browser").hidden = true;
    addRoots([browseFile]).then(() => presetFromImage(browseFile)); return;
  }
  if (browseMode === "param:file") {          // a file parameter of a post-production step
    if (!browseFile || !browseTarget) return;
    browseTarget.step.params[browseTarget.key] = browseFile; addRoots([browseFile]);
    $("#browser").hidden = true; save(); renderSteps(openStep); refresh(); return;
  }
  if (browseMode.startsWith("src:")) {        // src:<i>:lights|calib|masters|file
    const [, i, what] = browseMode.split(":"), src = state.sources[+i];
    if (what === "file") {
      if (!browseFile) return; src.file = browseFile; addRoots([browseFile]);
      if (state.mode === "hybrid") src.role = guessRole(browseFile) || src.role;
    }
    else { if (!browsePath) return; src[what === "lights" ? "lights" : what === "calib" ? "calibration" : "masterDirs"].push(browsePath); addRoots([browsePath]); }
    save(); renderPaths(); $("#browser").hidden = true; return;
  }
  if (browseMode === "post") {
    if (!browseFile) return;
    setInput(browseFile); $("#browser").hidden = true;
    addRoots([browseFile]).then(() => go(5)); return;
  }
  if (!browsePath) return;
  addRoots([browsePath]);
  if (browseMode.startsWith("panel:")) state.panels[+browseMode.slice(6)].push(browsePath);
  else (browseMode === "light" ? state.lights : browseMode === "masters" ? state.masterDirs : state.calibration).push(browsePath);
  save(); renderPaths(); $("#browser").hidden = true;
};
$$("[data-browse]").forEach(b => b.onclick = () => openBrowser(b.dataset.browse));

function fillPathList(ul, list) {
  ul.innerHTML = "";
  list.forEach((p, i) => {
    const li = document.createElement("li");
    li.innerHTML = `<span></span><button title="Rimuovi">${ico("x")}</button>`;
    li.firstChild.textContent = p;
    li.lastChild.onclick = () => { list.splice(i, 1); save(); renderPaths(); };
    ul.appendChild(li);
  });
}
function renderPaths() {
  setTimeout(() => scanFrames(), 0);          // frames list follows the folders
  fillPathList($("#lightPaths"), state.lights);
  fillPathList($("#calibPaths"), state.calibration);
  fillPathList($("#masterPaths"), state.masterDirs);
  $$("#lightMode button").forEach(b => b.classList.toggle("on", b.dataset.v === state.mode));
  $("#singleBox").hidden = state.mode !== "single"; $("#mosaicBox").hidden = state.mode !== "mosaic";
  $("#sourcesBox").hidden = !multiMode();
  $("#toSettings").innerHTML = (alignOnly() ? "Allinea e apri in post-produzione" : "Avanti") + ico("arrow-right");
  $("#calibInput").hidden = $("#framesBox").hidden = alignOnly();     // finished files: nothing to calibrate
  renderSources();
  const box = $("#panelList"); box.innerHTML = "";
  state.panels.forEach((folders, k) => {
    const d = document.createElement("div"); d.className = "panel";
    d.innerHTML = `<div class="phead"><b>Pannello ${k + 1}</b><span><button class="btn small">+ Cartella</button>
      <button class="btn small" title="Rimuovi il pannello">${ico("x")}</button></span></div><ul class="paths"></ul>`;
    const [add, del] = $$("button", d);
    add.onclick = () => openBrowser(`panel:${k}`);
    del.onclick = () => { state.panels.splice(k, 1); save(); renderPaths(); };
    fillPathList($("ul", d), folders);
    if (!folders.length) $("ul", d).innerHTML = '<li class="muted">nessuna cartella</li>';
    box.appendChild(d);
  });
}
$$("#lightMode button").forEach(b => b.onclick = () => {
  state.mode = b.dataset.v; state.mosaic = state.mode === "mosaic";
  if (multiMode() && state.sources.length < 2) {
    const roles = state.mode === "hybrid" ? ["rgb", "ha"] : ["", ""];
    while (state.sources.length < 2) state.sources.push(newSource(roles[state.sources.length]));
  }
  if (state.mode === "hybrid") state.sources.forEach((x, i) => { if (!x.role) x.role = i ? "ha" : "rgb"; });
  save(); renderPaths();
});
// ---------------------------------------------------------------- sources (più camere / ibrida)
const ROLES = [["rgb", "Colore (RGB)"], ["ha", "Ha"], ["oiii", "OIII"], ["sii", "SII"]];
// role from the file name: "Rosetta_Ha_7nm.fits" -> ha; nothing recognisable -> null (the choice stays)
function guessRole(path) {
  const n = path.split(/[\\/]/).pop().toLowerCase().replace(/\.[^.]+$/, "");
  if (/(^|[^a-z])(ha|h-?alpha|halpha)([^a-z]|$)/.test(n)) return "ha";
  if (/(^|[^a-z])(oiii|o3|o-iii)([^a-z]|$)/.test(n)) return "oiii";
  if (/(^|[^a-z])(sii|s2|s-ii)([^a-z]|$)/.test(n)) return "sii";
  if (/(^|[^a-z])(rgb|lrgb|osc|colou?r)([^a-z]|$)/.test(n)) return "rgb";
  return null;
}
function newSource(role = "") {
  return { role, lights: [], calibration: [], masterDirs: [], file: "", useFile: false, filters: false };
}
function renderSources() {
  const hybrid = state.mode === "hybrid";
  $("#sourcesHint").innerHTML = hybrid
    ? "La prima sorgente è la foto a colori, le altre Ha / OIII / SII. Ognuna si integra con i suoi bias, dark e flat (o usa un file già integrato); poi la banda stretta viene allineata sul colore e in post-produzione trovi il passo <b>NBRGBCombination</b> con un cursore per ogni canale."
    : "Lo stesso oggetto ripreso con camere (o sessioni) diverse: ognuna si integra con i suoi master, poi tutte vengono allineate sulla prima (anche con pixel e rotazione diversi) e unite pesando il rumore.";
  const box = $("#sourceList"); box.innerHTML = "";
  state.sources.forEach((src, i) => {
    const d = document.createElement("div"); d.className = "panel source";
    const roleSel = hybrid ? `<select class="srole" title="Cosa contiene questa sorgente">${ROLES.map(([v, n]) =>
      `<option value="${v}" ${v === src.role ? "selected" : ""}>${n}</option>`).join("")}</select>` : `<b>Camera ${i + 1}</b>`;
    d.innerHTML = `<div class="phead"><span class="row">${roleSel}</span>
        <button class="btn small sdel" title="Rimuovi la sorgente">${ico("x")}</button></div>
      <div class="seg small"><button data-u="0" class="${src.useFile ? "" : "on"}">Da integrare</button><button data-u="1" class="${src.useFile ? "on" : ""}">File già integrato</button></div>
      <div class="sraw" ${src.useFile ? "hidden" : ""}>
        <div class="muted small">Light</div><ul class="paths sl"></ul>
        <div class="muted small">Calibrazione (file grezzi; vuoto = quella comune a destra)</div><ul class="paths sc"></ul>
        <div class="muted small">Master già pronti</div><ul class="paths sm"></ul>
        <div class="row"><button class="btn small" data-b="lights">+ Light</button><button class="btn small" data-b="calib">+ File per i master</button>
          <button class="btn small" data-b="masters">+ Master pronti</button>
          <label class="small" title="Camera mono con più filtri nella stessa sorgente (es. L, R, G, B): integra ogni filtro e li combina">
            <input type="checkbox" class="sfilt" ${src.filters ? "checked" : ""}> mono con filtri</label></div>
      </div>
      <div class="sfile" ${src.useFile ? "" : "hidden"}><div class="row"><span class="muted small sfpath">${esc(src.file || "nessun file")}</span>
        <button class="btn small" data-b="file">Scegli FITS / XISF / TIFF…</button></div>
        <div class="muted small">Immagine già integrata, lineare o già lavorata (anche da PixInsight o da un altro programma).</div></div>`;
    if (hybrid) $(".srole", d).onchange = e => { src.role = e.target.value; save(); };
    $(".sdel", d).onclick = () => { state.sources.splice(i, 1); save(); renderPaths(); };
    $$(".seg button", d).forEach(b => b.onclick = () => { src.useFile = b.dataset.u === "1"; save(); renderPaths(); });
    $$("[data-b]", d).forEach(b => b.onclick = () => openBrowser(`src:${i}:${b.dataset.b}`));
    $(".sfilt", d).onchange = e => { src.filters = e.target.checked; save(); };
    fillPathList($(".sl", d), src.lights); fillPathList($(".sc", d), src.calibration); fillPathList($(".sm", d), src.masterDirs);
    box.appendChild(d);
  });
}
// hybrid with finished files only: nothing to stack, the files are just registered on the colour one and the
// post-production opens on them (the narrowband stays a separate, adjustable ingredient)
const alignOnly = () => state.mode === "hybrid" && state.sources.length >= 2 && state.sources.every(x => x.useFile && x.file);
async function startAlignOnly() {
  const rgb = (state.sources.find(x => x.role === "rgb") || state.sources[0]).file, dir = rgb.replace(/[\\/][^\\/]*$/, ""), stem = rgb.split(/[\\/]/).pop().replace(/\.[^.]+$/, "");
  go(3); $("#log").textContent = ""; $("#bar").style.width = "0";
  try {
    const { job } = await api("/api/stack", { ...settings(), work_dir: dir, output: `${stem}_ibrida` });
    pollQueue();
    follow(job, $("#log"), $("#bar"), openStackResult, $("#stages"));
  } catch (e) { $("#log").textContent = e.message; }
}
$("#addSource").onclick = () => { state.sources.push(newSource(state.mode === "hybrid" ? "oiii" : "")); save(); renderPaths(); };
function sourcesForStack() {
  return state.sources.map(x => ({ role: state.mode === "hybrid" ? x.role : "",
    ...(x.useFile ? { file: x.file } : { lights: x.lights, calibration: x.calibration, master_dirs: x.masterDirs, filters: x.filters }) }))
    .filter(x => x.file || (x.lights && x.lights.length));
}
$("#addPanel").onclick = () => { state.panels.push([]); save(); renderPaths(); };

// ---------------------------------------------------------------- 2: scan
// the frames are analysed on the same page as the folders: again every time a folder is added or removed
let scanToken = 0, scanKey = "";
async function scanFrames(force) {
  const key = JSON.stringify([lightRoots(), state.calibration, state.masterDirs, state.mosaic, state.groupKw, state.calKw]);
  if (!force && key === scanKey && scanResult) return;
  if (!lightRoots().length) {
    scanResult = null; scanKey = ""; $("#lightGroups").innerHTML = ""; $("#calibCard").hidden = true;
    $("#scanInfo").textContent = "aggiungi una cartella di light"; return;
  }
  const token = ++scanToken;
  $("#scanInfo").textContent = "analisi dei frame in corso…";
  try {
    const r = await api("/api/scan", { lights: lightRoots(), calibration: state.calibration, master_dirs: state.masterDirs,
      grouping_keywords: kwList(state.groupKw), calibration_keywords: kwList(state.calKw) });
    if (token !== scanToken) return;
    scanResult = r; scanKey = key;
    if (state.groupIndex >= r.light_groups.length) { state.groupIndex = 0; save(); }   // fewer groups than before
    renderScan(); $("#calibCard").hidden = false;
    showSession();
    const n = r.light_groups.reduce((a, g) => a + (g.count || (g.frames || []).length || 0), 0);
    const ng = r.light_groups.length;
    $("#scanInfo").textContent = `${ng} ${ng === 1 ? "gruppo" : "gruppi"} di light` + (n ? `, ${n} frame` : "");
  } catch (e) { if (token === scanToken) $("#scanInfo").textContent = "errore: " + e.message.slice(0, 160); }
}
$("#toScan").onclick = () => scanFrames(true);
$("#toSettings").onclick = async () => {
  if (state.mosaic && state.panels.filter(p => p.length).length < 2) { alert("Un mosaico richiede almeno 2 pannelli con una cartella di light."); return; }
  if (alignOnly()) { startAlignOnly(); return; }
  if (multiMode()) {
    const ok = sourcesForStack();
    if (ok.length < 2) { alert("Servono almeno 2 sorgenti, ognuna con una cartella di light o un file già integrato."); return; }
    if (!lightRoots().length) { go(2); return; }        // only integrated files: nothing to scan
  }
  if (!lightRoots().length) { alert("Aggiungi almeno una cartella di light."); return; }
  await scanFrames();
  if (scanResult) go(2);
};

// per-group overrides (WBPP lets every group have its own integration settings); "" = as in Impostazioni
function groupSettingsForm(label) {
  const gs = state.groupSettings[label] || {};
  const sel = (k, name, opts) => `<label>${name} <select data-gs="${k}"><option value="">come nelle impostazioni</option>${opts.map(([v, n]) =>
    `<option value="${v}" ${String(gs[k]) === String(v) ? "selected" : ""}>${n}</option>`).join("")}</select></label>`;
  const num = (k, name, step) => `<label>${name} <input type="number" data-gs="${k}" step="${step}" value="${gs[k] ?? ""}" placeholder="="></label>`;
  return sel("rejection", "Rigetto", [["auto", "automatico"], ["winsorized", "Winsorized"], ["linear_fit", "Linear fit"], ["esd", "ESD"], ["sigma", "Sigma"], ["percentile", "Percentile"]])
    + num("reject_low", "σ basso", 0.5) + num("reject_high", "σ alto", 0.5)
    + sel("drizzle", "Drizzle", [[1, "off"], [2, "2×"], [3, "3×"]])
    + sel("local_norm", "Local normalization", [["true", "sì"], ["false", "no"]])
    + num("pedestal", "Pedestal ADU", 10);
}
$("#g_kw").value = state.groupKw || ""; $("#g_calkw").value = state.calKw || ""; $("#g_all").checked = !!state.allGroups;
$("#g_kw").onchange = e => { state.groupKw = e.target.value; save(); scanFrames(true); };
$("#g_calkw").onchange = e => { state.calKw = e.target.value; save(); scanFrames(true); };
$("#g_all").onchange = e => { state.allGroups = e.target.checked; save(); };
// ---------------------------------------------------------------- what the session shows (plate solving)
// the first light is identified as soon as the folders are scanned: header OBJECT, then ASTAP and the catalogue.
// The name goes above the frames and becomes the default name of the exported files.
const identified = {};
function identify(path) {
  if (!path) return Promise.resolve(null);
  return identified[path] = identified[path] || api("/api/identify", { path }).catch(() => { delete identified[path]; return null; });
}
const dms = (v, hours) => {
  const s = v < 0 ? "−" : hours ? "" : "+", a = Math.abs(hours ? v / 15 : v), d = Math.floor(a), m = Math.floor((a - d) * 60);
  return `${s}${d}${hours ? "h" : "°"}${String(m).padStart(2, "0")}${hours ? "m" : "′"}`;
};
async function showSession() {
  const box = $("#sessionBox"), g = scanResult && scanResult.light_groups[state.groupIndex] || scanResult?.light_groups[0];
  const first = g && g.frames.find(f => !isOut(f.path));
  if (!first) { box.hidden = true; return; }
  box.hidden = false;
  box.innerHTML = `${ico("crosshair")}<span class="muted">Identificazione di ${esc(fname(first.path))}…
    <span class="small">(plate solving con ASTAP, pochi secondi)</span></span>`;
  const r = await identify(first.path);
  if (g !== (scanResult && (scanResult.light_groups[state.groupIndex] || scanResult.light_groups[0]))) return;
  if (!r) { box.innerHTML = `${ico("crosshair")}<span class="muted">identificazione non disponibile (plate solving fallito)</span>`; return; }
  const name = r.name || "oggetto sconosciuto";
  const facts = r.solved ? [`RA ${dms(r.ra, true)} Dec ${dms(r.dec)}`, `campo ${r.fov[0].toFixed(2)}° × ${r.fov[1].toFixed(2)}°`,
    `${r.scale.toFixed(2)}″/px`, `rotazione ${Math.round(r.rotation)}°`] : [];
  box.innerHTML = `${ico("crosshair")}<div><div><span class="muted">Sessione</span> <b>${esc(name)}</b>
      ${r.object && r.object !== name ? `<span class="muted">· OBJECT nell'header: ${esc(r.object)}</span>` : ""}</div>
    <div class="muted small">${r.solved ? esc(facts.join(" · ")) + (r.objects.length > 1 ? " · nel campo anche: " + esc(r.objects.filter(o => o !== name).slice(0, 4).join(", ")) : "")
      : esc(r.note || "")}</div></div>`;
  const out = $("#outName"), auto = name.replace(/[^\w\- ]+/g, "").trim().replace(/\s+/g, "_");
  if (r.name && out && (out.value === "final" || out.value === state.sessionName)) { out.value = auto; state.sessionName = auto; save(); }
}

// the frames found are shown, not only counted: a list (name, exposure, gain, temperature, date) or a contact sheet
// of thumbnails; unticking / clicking excludes a frame (lights and calibration alike, state.exclude)
const thumbUrl = (path, side = 220) => `/api/thumb?path=${encodeURIComponent(path)}&side=${side}`;
const fname = p => p.split(/[\\/]/).pop();
const isOut = p => state.exclude.includes(p);
function setOut(p, out) {
  state.exclude = out ? [...new Set([...state.exclude, p])] : state.exclude.filter(x => x !== p); save();
}
function frameTable(frames, auto = []) {
  return `<div class="tablewrap frametable"><table>
    <tr><th></th><th>File</th><th>Posa</th><th>${frames.some(f => f.iso != null) ? "ISO" : "Gain"}</th><th>Offset</th><th>Bin</th><th>Temp</th><th>Filtro</th><th>Data</th></tr>
    ${frames.map(f => { const a = auto.includes(f.path), off = a || isOut(f.path);
      return `<tr class="${off ? "rej" : ""}" title="${esc(f.path)}"><td><input type="checkbox" data-path="${encodeURIComponent(f.path)}"
        ${off ? "" : "checked"} ${a ? "disabled" : ""}></td><td>${esc(fname(f.path))}${a ? ' <span class="muted">ignorato in automatico: posa diversa</span>' : ""}</td>
        <td>${f.exptime ?? ""} s</td><td>${f.iso ?? f.gain ?? ""}</td><td>${f.offset ?? ""}</td><td>${f.binning || 1}×${f.binning || 1}</td>
        <td>${f.temp != null ? f.temp + "°" : ""}</td><td>${esc(f.filter || "")}</td><td>${esc((f.date || "").replace("T", " ").slice(0, 16))}</td></tr>`; }).join("")}
    </table></div>`;
}
function frameSheet(frames) {
  return `<div class="sheet">${frames.map(f => `<figure class="shot ${isOut(f.path) ? "off" : ""}" data-path="${encodeURIComponent(f.path)}"
      title="${esc(f.path)} — clic per escludere / includere"><img loading="lazy" alt="" src="${thumbUrl(f.path)}">
      <figcaption><span>${esc(fname(f.path))}</span><span>${f.exptime ?? ""} s${f.iso != null ? ` · ISO ${f.iso}` : f.gain != null ? ` · G${f.gain}` : ""}${f.temp != null && f.temp !== "" ? ` · ${f.temp}°` : ""}</span></figcaption></figure>`).join("")}</div>`;
}
// ---------------------------------------------------------------- quick viewer of the frames (Sorgenti page)
// hover a file name (list) or a thumbnail (contact sheet): a colour preview with auto-stretch; click the name
// (double click a thumbnail) for the full-screen viewer: STF linked / unlinked / boost / linear, Bayer colour,
// wheel zoom + drag, previous / next frame of the list, X to exclude or include the frame shown
// default "unlinked": a raw colour frame with one curve per channel looks neutral (linked keeps the sensor cast)
const viewUrl = (path, side, o = {}) => `/api/view?path=${encodeURIComponent(path)}&side=${side}` +
  `&stretch_mode=${o.mode || "unlinked"}&color=${o.color === false ? "false" : "true"}&boost=${o.boost ? "true" : "false"}`;
let hoverTimer = null, hoverEl = null;
function hoverPreview(target, path) {
  clearTimeout(hoverTimer);
  hoverTimer = setTimeout(() => {
    if (!hoverEl) { hoverEl = document.createElement("div"); hoverEl.className = "fhover"; document.body.appendChild(hoverEl); }
    const o = state.viewer || {};
    hoverEl.innerHTML = `<img alt="" src="${viewUrl(path, 520, o)}"><div class="cap">${esc(fname(path))}</div>`;
    hoverEl.hidden = false;
    // beside the name when there is room, else above / below its row (never on top of what is hovered)
    const r = target.getBoundingClientRect(), W = Math.min(540, innerWidth * 0.7 + 8), H = W * 0.78;
    let x = r.right + 12, y = r.top - 40;
    if (x + W > innerWidth) x = r.left - W - 12;
    if (x < 8) { x = Math.min(Math.max(8, r.left), innerWidth - W - 8); y = r.top - H - 8 >= 8 ? r.top - H - 8 : r.bottom + 8; }
    else if (y + H > innerHeight) y = Math.max(8, innerHeight - H - 8);
    hoverEl.style.left = `${x}px`; hoverEl.style.top = `${Math.max(8, y)}px`;
  }, 280);
}
function hoverHide() { clearTimeout(hoverTimer); if (hoverEl) hoverEl.hidden = true; }

const fv = { list: [], i: 0, z: 1, tx: 0, ty: 0, pan: null, onExclude: null };
function openViewer(list, i, onExclude) {
  hoverHide();
  fv.list = list; fv.i = i; fv.onExclude = onExclude;
  let el = $("#fview");
  if (!el) {
    el = document.createElement("div"); el.id = "fview"; el.className = "fviewer";
    el.innerHTML = `<div class="fvbar">
        <b class="fvname"></b>
        <div class="seg small fvmode"><button data-m="linked" title="una curva per tutti i canali">Collegato</button>
          <button data-m="unlinked" title="una curva per canale: toglie la dominante dei frame a colori">Separato</button>
          <button data-m="linear" title="i valori come sono, senza stretch">Lineare</button></div>
        <label title="stretch più forte (Boost di PixInsight)"><input type="checkbox" class="fvboost"> Boost</label>
        <label title="colore ricostruito dalla matrice di Bayer (2×2)"><input type="checkbox" class="fvcolor"> Colore</label>
        <span class="fvsp"></span>
        <button class="btn small fvprev" title="frame precedente (←)">${ico("chevron-left")}</button>
        <span class="fvpos muted"></span>
        <button class="btn small fvnext" title="frame successivo (→)">${ico("chevron-right")}</button>
        <button class="btn small fvout" title="escludi / includi questo frame (X)"></button>
        <button class="btn small fvfit" title="immagine intera (doppio clic)">${ico("maximize")}Adatta</button>
        <button class="btn small fvclose" title="chiudi (Esc)">${ico("x")}</button></div>
      <div class="fvstage"><img alt=""></div><div class="fvinfo muted"></div>`;
    document.body.appendChild(el);
    const img = $(".fvstage img", el), stage = $(".fvstage", el);
    $$(".fvmode button", el).forEach(b => b.onclick = () => { state.viewer = { ...(state.viewer || {}), mode: b.dataset.m }; save(); showFrameView(); });
    $(".fvboost", el).onchange = e => { state.viewer = { ...(state.viewer || {}), boost: e.target.checked }; save(); showFrameView(); };
    $(".fvcolor", el).onchange = e => { state.viewer = { ...(state.viewer || {}), color: e.target.checked }; save(); showFrameView(); };
    $(".fvprev", el).onclick = () => stepViewer(-1);
    $(".fvnext", el).onclick = () => stepViewer(1);
    $(".fvout", el).onclick = () => toggleViewerOut();
    $(".fvfit", el).onclick = () => { fv.z = 1; fv.tx = fv.ty = 0; fvApply(); };
    $(".fvclose", el).onclick = () => { el.hidden = true; };
    stage.addEventListener("wheel", e => {
      e.preventDefault();
      const r = stage.getBoundingClientRect(), cx = e.clientX - r.left, cy = e.clientY - r.top;
      const z1 = Math.min(32, Math.max(1, fv.z * (e.deltaY < 0 ? 1.25 : 0.8)));
      fv.tx = cx - (cx - fv.tx) * z1 / fv.z; fv.ty = cy - (cy - fv.ty) * z1 / fv.z; fv.z = z1; fvApply();
    }, { passive: false });
    stage.addEventListener("pointerdown", e => {
      if (fv.z <= 1) return;
      e.preventDefault(); fv.pan = { x: e.clientX, y: e.clientY, tx: fv.tx, ty: fv.ty }; stage.setPointerCapture(e.pointerId);
    });
    stage.addEventListener("pointermove", e => {
      if (!fv.pan) return;
      fv.tx = fv.pan.tx + e.clientX - fv.pan.x; fv.ty = fv.pan.ty + e.clientY - fv.pan.y; fvApply();
    });
    stage.addEventListener("pointerup", () => { fv.pan = null; });
    stage.addEventListener("dblclick", () => { fv.z = 1; fv.tx = fv.ty = 0; fvApply(); });
    img.addEventListener("load", () => { $(".fvstage", el).classList.remove("loading"); });
    document.addEventListener("keydown", e => {
      if (el.hidden || e.target.closest?.("input, textarea, select")) return;
      if (e.key === "Escape") el.hidden = true;
      else if (e.key === "ArrowLeft") stepViewer(-1);
      else if (e.key === "ArrowRight") stepViewer(1);
      else if (e.key === "x" || e.key === "X") toggleViewerOut();
      else return;
      e.preventDefault();
    });
  }
  el.hidden = false; fv.z = 1; fv.tx = fv.ty = 0; fvApply(); showFrameView();
}
function fvApply() {
  const el = $("#fview"); if (!el) return;
  const stage = $(".fvstage", el), W = stage.clientWidth, H = stage.clientHeight;
  fv.z = Math.min(32, Math.max(1, fv.z));
  fv.tx = Math.min(0, Math.max(W - W * fv.z, fv.tx)); fv.ty = Math.min(0, Math.max(H - H * fv.z, fv.ty));
  $(".fvstage img", el).style.transform = `translate(${fv.tx}px, ${fv.ty}px) scale(${fv.z})`;
  stage.classList.toggle("zoomed", fv.z > 1);
}
let fvToken = 0;
async function showFrameView() {
  const el = $("#fview"), path = fv.list[fv.i]; if (!el || !path) return;
  const o = { mode: "unlinked", color: true, boost: false, ...(state.viewer || {}) };
  $$(".fvmode button", el).forEach(b => b.classList.toggle("on", b.dataset.m === o.mode));
  $(".fvboost", el).checked = !!o.boost; $(".fvcolor", el).checked = o.color !== false;
  $(".fvname", el).textContent = fname(path);
  $(".fvpos", el).textContent = `${fv.i + 1} / ${fv.list.length}`;
  const out = isOut(path);
  $(".fvout", el).textContent = out ? "Includi (X)" : "Escludi (X)"; $(".fvout", el).classList.toggle("danger", !out);
  el.classList.toggle("excluded", out);
  const side = Math.min(3000, Math.round(Math.max(innerWidth, innerHeight) * (window.devicePixelRatio || 1)));
  const token = ++fvToken;
  $(".fvstage", el).classList.add("loading");
  try {
    const r = await fetch(viewUrl(path, side, o));
    if (!r.ok) { let m = await r.text(); try { m = JSON.parse(m).detail || m; } catch (e) {} throw new Error(m); }
    const blob = await r.blob(), info = JSON.parse(r.headers.get("X-Info") || "{}");
    if (token !== fvToken) return;
    showBlob($(".fvstage img", el), blob);
    const parts = [`${info.width}×${info.height} px`, info.bayer ? `Bayer ${info.bayer}` : (info.channels === 3 ? "colore" : "mono"),
      info.type, info.exptime != null ? `${info.exptime} s` : "", info.iso != null ? `ISO ${info.iso}` : info.gain != null ? `gain ${info.gain}` : "",
      info.offset != null ? `offset ${info.offset}` : "", info.temp != null ? `${info.temp}°C` : "", info.filter ? `filtro ${info.filter}` : "",
      info.date ? String(info.date).replace("T", " ").slice(0, 19) : "", info.median ? `mediana ${info.median.map(v => v.toFixed(4)).join(" / ")}` : ""];
    $(".fvinfo", el).textContent = parts.filter(Boolean).join(" · ") + (out ? " · ESCLUSO" : "");
  } catch (e) {
    if (token === fvToken) { $(".fvstage", el).classList.remove("loading"); $(".fvinfo", el).textContent = "errore: " + e.message.slice(0, 160); }
  }
  [fv.i - 1, fv.i + 1].forEach(j => { if (fv.list[j]) fetch(viewUrl(fv.list[j], side, o)).catch(() => {}); });   // warm the neighbours
}
function stepViewer(d) {
  const j = fv.i + d; if (j < 0 || j >= fv.list.length) return;
  fv.i = j; showFrameView();
}
function toggleViewerOut() {
  const p = fv.list[fv.i]; if (!p) return;
  const out = !isOut(p); setOut(p, out);
  if (fv.onExclude) fv.onExclude(p, out);
  showFrameView();
}
function wireFrames(el, refresh) {
  $$("input[type=checkbox][data-path]", el).forEach(cb => cb.onchange = () => {
    setOut(decodeURIComponent(cb.dataset.path), !cb.checked); cb.closest("tr").classList.toggle("rej", !cb.checked); refresh();
  });
  $$(".shot", el).forEach(s => s.onclick = () => {
    const p = decodeURIComponent(s.dataset.path), out = !isOut(p); setOut(p, out); s.classList.toggle("off", out); refresh();
  });
  // quick viewer: the frames of this list in order; excluding from the viewer updates the list too
  const items = [...$$("input[type=checkbox][data-path], .shot", el)];
  const list = items.map(x => decodeURIComponent(x.dataset.path));
  const onExclude = (p, out) => {
    items.filter(x => decodeURIComponent(x.dataset.path) === p).forEach(x => {
      if (x.tagName === "INPUT") { x.checked = !out; x.closest("tr").classList.toggle("rej", out); } else x.classList.toggle("off", out);
    });
    refresh();
  };
  items.forEach((x, i) => {
    const target = x.tagName === "INPUT" ? x.closest("tr").children[1] : x;
    target.classList.add("fviewable");
    target.addEventListener("mouseenter", () => hoverPreview(target, list[i]));
    target.addEventListener("mouseleave", hoverHide);
    if (x.tagName === "INPUT") target.addEventListener("click", () => openViewer(list, i, onExclude));
    else target.addEventListener("dblclick", e => { e.preventDefault(); openViewer(list, i, onExclude); });
  });
}
function renderScan() {
  const box = $("#lightGroups"); box.innerHTML = "";
  if (!["list", "sheet"].includes(state.frameView)) state.frameView = "list";
  scanResult.light_groups.forEach((g, gi) => {
    const d = document.createElement("div"), sel = gi === state.groupIndex;
    d.className = "group" + (sel ? " sel" : "");
    const nOut = () => g.frames.filter(f => isOut(f.path)).length;
    d.innerHTML = `<label class="row ghead"><input type="radio" name="lg" ${sel ? "checked" : ""}>
        <span><b>${g.count} light</b> — ${g.label}</span><span class="muted gout"></span></label>
      <details class="gframes" ${sel ? "open" : ""}><summary class="muted">i frame di questo gruppo</summary>
        <div class="row framebar"><div class="seg small fview"><button data-v="list" class="${state.frameView === "list" ? "on" : ""}">${ico("list")}Elenco</button>
          <button data-v="sheet" class="${state.frameView === "sheet" ? "on" : ""}">${ico("images")}Provini</button></div>
          <button class="btn small" data-all="1">Includi tutti</button><button class="btn small" data-all="0">Escludi tutti</button></div>
        <div class="fbody">${state.frameView === "sheet" ? frameSheet(g.frames) : frameTable(g.frames)}</div></details>
      <details class="gset"><summary class="muted">${ico("settings")}impostazioni di questo gruppo${state.groupSettings[g.label] && Object.keys(state.groupSettings[g.label]).length ? " (modificate)" : ""}</summary>
        <div class="row">${groupSettingsForm(g.label)}</div></details>`;
    const count = () => { const n = nOut(); $(".gout", d).textContent = n ? `· ${n} esclusi, ne restano ${g.count - n}` : "· tutti inclusi"; };
    count();
    $("input[type=radio]", d).onchange = () => { state.groupIndex = gi; save(); renderScan(); showSession(); };
    $$(".fview button", d).forEach(b => b.onclick = e => { e.preventDefault(); state.frameView = b.dataset.v; save(); renderScan(); });
    $$("[data-all]", d).forEach(b => b.onclick = e => {
      e.preventDefault(); g.frames.forEach(f => setOut(f.path, b.dataset.all === "0")); renderScan();
    });
    wireFrames(d, count);
    $$("[data-gs]", d).forEach(el => el.onchange = () => {
      const gs = state.groupSettings[g.label] = state.groupSettings[g.label] || {};
      const k = el.dataset.gs, v = el.value;
      if (v === "") delete gs[k];
      else gs[k] = el.type === "number" ? +v : v === "true" ? true : v === "false" ? false : isNaN(+v) || k === "rejection" ? v : +v;
      if (!Object.keys(gs).length) delete state.groupSettings[g.label];
      save();
    });
    box.appendChild(d);
  });
  const names = { bias: "Bias", dark: "Dark", flat: "Flat", darkflat: "Dark flat" };
  const cs = $("#calibSummary"); cs.innerHTML = "";
  Object.entries(scanResult.calibration).forEach(([k, gs]) => {
    const row = document.createElement("div"); row.className = "calrow";
    row.innerHTML = `<b>${names[k]}</b><div>${gs.length ? "" : '<span class="muted">nessuno</span>'}</div>`;
    gs.forEach(g => {
      const frames = g.frames || [], odd = g.odd_paths || [];
      const det = document.createElement("details");
      const head = () => { const n = frames.filter(f => isOut(f.path) && !odd.includes(f.path)).length;
        return `${g.count} × ${g.label}` + (odd.length ? ` <span class="warn">· ${odd.length} con posa diversa, ignorati in automatico</span>` : "")
          + (n ? ` <span class="muted">· ${n} esclusi da te</span>` : ""); };
      det.innerHTML = `<summary>${head()}</summary>${frameTable(frames, odd)}`;
      wireFrames(det, () => { $("summary", det).innerHTML = head(); });
      $("div", row).appendChild(det);
    });
    cs.appendChild(row);
  });
  if (scanResult.masters && scanResult.masters.length) cs.insertAdjacentHTML("beforeend", `<div class="calrow"><b>Master pronti</b><div>` +
      scanResult.masters.map(m => `${names[m.kind] || m.kind}: ${esc(m.file)}` +
        (m.filter ? ` (${esc(m.filter)})` : "") + (m.exptime ? ` ${m.exptime} s` : "") +
        (m.flags.length ? ` <span class="warn">${esc(m.flags.join("; "))}</span>` : "")).join("<br>") +
      `<br><span class="muted small">usati così come sono quando per quel tipo non ci sono file grezzi</span></div></div>`);
  if (scanResult.unknown && scanResult.unknown.length) cs.insertAdjacentHTML("beforeend", `<div class="calrow"><b>Non riconosciuti</b><div>
      <details><summary class="muted">${scanResult.unknown.length} file non usati (tipo non capito da header e nome)</summary>
      <ul class="paths">${scanResult.unknown.map(u => `<li><span>${esc(u)}</span></li>`).join("")}</ul></details></div></div>`);
  if (!$("#s_work").value) $("#s_work").value = lightRoots()[0] + "\\astroforge";
  if (state.mosaic) box.insertAdjacentHTML("afterbegin", `<p class="hint">Mosaico di ${state.panels.filter(p => p.length).length} pannelli:
    ogni pannello usa il proprio gruppo di light più numeroso (la scelta qui sotto non si applica). Le esclusioni sì.</p>`);
}

// ---------------------------------------------------------------- 3: settings
let drizzle = 1;
$$("#s_drizzle button").forEach(b => b.onclick = () => {
  drizzle = +b.dataset.v; $$("#s_drizzle button").forEach(x => x.classList.toggle("on", x === b)); });
$("#s_pixfrac").oninput = e => $("#pixfracV").textContent = (+e.target.value).toFixed(2);

function settings() {
  return {
    lights: lightRoots(), calibration: state.calibration, work_dir: $("#s_work").value, output: "stack",
    mosaic: state.mosaic ? state.panels.filter(p => p.length) : null,
    mode: multiMode() ? state.mode : "single", sources: multiMode() ? sourcesForStack() : [],
    light_group: state.groupIndex, exclude: state.exclude,
    gpu: $("#s_gpu").checked, cosmetic: $("#s_cosmetic").checked, bayer_pattern: $("#s_bayer").value || null, debayer: $("#s_debayer").value,
    auto_reject: $("#s_autoreject").checked, fwhm_sigma: +$("#s_fwhm").value, ecc_max: +$("#s_ecc").value,
    rejection: $("#s_rej").value, reject_low: +$("#s_low").value, reject_high: +$("#s_high").value,
    weights: $("#s_weights").checked, drizzle, pixfrac: +$("#s_pixfrac").value,
    bayer_drizzle: $("#s_bayerdz").checked, drizzle_reject: $("#s_dzrej").checked,
    filters: $("#s_filters").checked, palette: $("#s_palette").value,
    dark_optimize: { auto: "auto", true: true, false: false }[$("#s_darkopt").value], flat_per_session: $("#s_flatsession").checked,
    exposure_tolerance: +$("#s_exptol").value || 0, weighting: $("#s_weighting").value,
    master_reuse: $("#s_libreuse").checked, banding: $("#s_banding").checked,
    master_dirs: state.masterDirs, defect_map: state.defectMap || null,
    master_options: state.masterForStack ? masterOptions() : {},
    grouping_keywords: kwList(state.groupKw), calibration_keywords: kwList(state.calKw), all_groups: !!state.allGroups,
    group_settings: state.groupSettings,
    pedestal: { "0": 0, auto: "auto", adu: +$("#s_pedestal").value || 0 }[$("#s_pedmode").value],
    large_scale_rejection: $("#s_lsr").checked ? { high: +$("#s_lsrhigh").value, low: +$("#s_lsrlow").value, growth: +$("#s_lsrgrow").value } : null,
    plate_solve: { auto: "auto", true: true, false: false }[$("#s_solve").value],
    local_norm_reference: $("#s_lnref").value, local_norm_ref_frames: +$("#s_lnrefn").value || 8,
    approval: $("#s_approval").value !== "" ? +$("#s_approval").value : null, drizzle_kernel: $("#s_dzkernel").value,
    dualband: $("#s_dualband").checked, local_norm: $("#s_localnorm").checked,
    satellites: { auto: "auto", true: true, false: false }[$("#s_sat").value], distortion: $("#s_distortion").checked,
    comet: $("#s_comet").checked && $("#s_cx1").value !== "",
    comet_first_xy: [+$("#s_cx1").value, +$("#s_cy1").value],
    comet_last_xy: $("#s_cx2").value !== "" ? [+$("#s_cx2").value, +$("#s_cy2").value] : null,
  };
}

// ---------------------------------------------------------------- 4: run
const STAGES = ["scan", "masters", "calibrate", "register", "normalize", "integrate", "drizzle", "solve", "done"];
const STAGE_IT = { scan: "Scansione", masters: "Master", calibrate: "Calibrazione", register: "Allineamento",
  normalize: "Normalizzazione", integrate: "Integrazione", drizzle: "Drizzle", solve: "Astrometria", done: "Fatto" };
// ---------------------------------------------------------------- plan (WBPP-like): what will run and how long
const fmtTime = s => s < 90 ? `${Math.round(s)} s` : s < 5400 ? `${Math.round(s / 60)} min` : `${(s / 3600).toFixed(1)} h`;
const HOW_IT = { crea: "da creare", cache: "già creato (cache)", libreria: "dalla libreria", manca: "manca" };
async function showPlan() {
  if (multiMode() || state.mosaic) { $("#planBox").textContent = "Piano disponibile per lo stacking di un campo (non per mosaico / più sorgenti)."; return; }
  if (!lightRoots().length) { $("#planBox").textContent = "Aggiungi le cartelle dei light."; return; }
  $("#planInfo").textContent = "calcolo…";
  try {
    const p = await api("/api/plan", settings());
    if (p.error) { $("#planBox").textContent = p.error; $("#planInfo").textContent = ""; return; }
    $("#planInfo").textContent = `totale stimato ${fmtTime(p.total_s)} · ${p.estimate_from}`;
    $("#planBox").innerHTML = p.groups.map(g => `<div class="plang"><b>${g.frames} light</b> — ${esc(g.label)} (${g.mpx} Mpx)
      ${Object.keys(g.overrides || {}).length ? `<span class="warn">impostazioni del gruppo: ${esc(Object.entries(g.overrides).map(([k, v]) => `${k}=${v}`).join(", "))}</span>` : ""}
      <div class="muted small">Master: ${g.masters.map(m => `${m.kind} ${m.frames ? m.frames + " frame, " : ""}<span class="${m.how === "manca" ? "warn" : ""}">${HOW_IT[m.how] || m.how}</span>`).join(" · ")}</div>
      <div class="planstages">${g.stages.map(st => `<span class="pst">${STAGE_IT[st.stage] || st.stage} <u style="width:${Math.round(60 * st.s / Math.max(...g.stages.map(x => x.s || 0), 1e-9))}px"></u><b>${fmtTime(st.s)}</b></span>`).join(ico("chevron-right", "sep"))}
        <span>= <b>${fmtTime(g.seconds)}</b></span></div></div>`).join("");
  } catch (e) { $("#planInfo").textContent = ""; $("#planBox").textContent = "errore: " + e.message.slice(0, 200); }
}
$("#showPlan").onclick = showPlan;
const pedToggle = () => { $("#s_pedestal").hidden = $("#s_pedmode").value !== "adu"; };
$("#s_pedmode").onchange = pedToggle; pedToggle();
let currentJob = null;

async function follow(jobId, logEl, barEl, onDone, stagesEl) {
  currentJob = jobId; let since = 0;
  logEl.textContent = "";
  for (;;) {
    let j;
    try { j = await api(`/api/jobs/${jobId}?since=${since}`); }
    catch (e) { logEl.textContent += `\n${e.message}\n`; currentJob = null; return; }
    if (j.status === "queued") {
      logEl.textContent = `In coda: posizione ${j.position}. Parte quando finiscono i lavori prima di questo.\n`;
      barEl.style.width = "0%";
      await new Promise(r => setTimeout(r, 1000));
      continue;
    }
    if (since === 0 && logEl.textContent.startsWith("In coda")) logEl.textContent = "";
    since = j.log_len;
    if (j.log.length) { logEl.textContent += j.log.join("\n") + "\n"; logEl.scrollTop = logEl.scrollHeight; }
    const si = STAGES.indexOf(j.stage);
    barEl.style.width = stagesEl ? `${Math.round(((Math.max(si, 0) + j.progress) / (STAGES.length - 1)) * 100)}%`
                                 : `${Math.round(j.progress * 100)}%`;
    if (stagesEl) stagesEl.innerHTML = STAGES.map((s, i) =>
      `<span class="${i < si ? "done" : i === si ? "on" : ""}">${STAGE_IT[s]}</span>`).join("");
    if (j.status !== "running") {
      currentJob = null;
      if (j.status === "done") { barEl.style.width = "100%"; onDone(j.result); }
      else logEl.textContent += `\n${j.status === "cancelled" ? "Annullato." : "Errore: vedi sopra."}\n`;
      return;
    }
    await new Promise(r => setTimeout(r, 800));
  }
}

// a new integrated image gets the automatic profile for its kind (SHO, HOO, mono, galaxy, colour nebula)
function setInput(path) {
  const changed = state.input !== path;
  state.input = path; save();
  if (changed) api(`/api/suggest_preset?path=${encodeURIComponent(path)}`).then(r => {
    state.preset = r.id; state.steps = []; state.autoReason = r.reason; save();
  }).catch(() => {});
}
const openStackResult = res => {
  stackResult = res;
  // hybrid: the post-production gets the narrowband step (registered files, sliders) right after the background
  state.hybridStep = res.hybrid_step || null; state.hybridFor = res.hybrid_step ? res.final_input : null;
  if (res.hybrid_step && (res.stretched || res.align_only)) {
    // integrated / finished files: only the alignment was asked, so no automatic profile, only the narrowband
    // step; the user adds the other processes (curves, saturation, LRGB, PixelMath...) from "+ Aggiungi passo"
    state.input = res.final_input; state.preset = EMPTY; openStep = 0;
    state.steps = [{ ...clone(res.hybrid_step), enabled: true }];
    state.autoReason = "file già integrati (banda stretta allineata: regola i cursori e aggiungi tu gli altri passi)";
    save(); go(5); return;
  }
  setInput(res.final_input); go(4); renderResults();
};
$("#startStack").onclick = async () => {
  go(3);
  try {
    const { job } = await api("/api/stack", settings());
    pollQueue();
    follow(job, $("#log"), $("#bar"), openStackResult, $("#stages"));
  } catch (e) { $("#log").textContent = e.message; }
};
// queue without leaving the page: the result is opened later from the job panel
async function enqueue(btn, url, body) {
  const label = btn.textContent;
  try {
    const r = await api(url, body);
    btn.innerHTML = `${ico("check")}In coda (#${r.position})`;
  } catch (e) { alert(e.message); }
  pollQueue();
  setTimeout(() => btn.textContent = label, 1800);
}
$("#queueStack").onclick = e => enqueue(e.currentTarget, "/api/stack", settings());
$("#cancelJob").onclick = () => currentJob && api(`/api/jobs/${currentJob}/cancel`, {});

// ---------------------------------------------------------------- 5: results
function renderResults() {
  const tabs = [];
  const rep = stackResult.report || {};
  if (stackResult.mosaic) {
    (stackResult.panels || []).forEach((p, i) => tabs.push([`Pannello ${i + 1}`, p]));
    tabs.push(["Mosaico", stackResult.mosaic]);
  } else {
    if (stackResult.filters) Object.entries(stackResult.filters).forEach(([f, p]) => tabs.push([f, p]));
    else if (stackResult.integration) tabs.push(["Integrazione", stackResult.integration]);
    if (stackResult.drizzle) tabs.push([`Drizzle`, stackResult.drizzle]);
    if (rep.comet) tabs.push(["Cometa", rep.comet.comet_stack]);
    if (rep.dualband) Object.entries(rep.dualband).forEach(([f, p]) => tabs.push([f, p]));
    if (stackResult.combined) tabs.push([rep.palette || "Combinata", stackResult.combined]);
  }
  const runs = Array.isArray(rep.runs) && rep.runs.length ? rep.runs : null;
  $("#qNote").hidden = !!runs || (!stackResult.mosaic && !stackResult.filters);
  $("#qNote").textContent = stackResult.mosaic ? "Mosaico: frame e blink dell'ultimo pannello." : "Filtri: frame e blink dell'ultimo filtro.";
  $("#resTabs").innerHTML = "";
  tabs.forEach(([name, path], i) => {
    const b = document.createElement("button"); b.textContent = name; b.dataset.name = name;
    b.onclick = () => { $$("#resTabs button").forEach(x => x.classList.toggle("on", x === b)); $("#resImg").src = imgUrl(path); };
    $("#resTabs").appendChild(b); if (i === tabs.length - 1) b.click();
  });
  // one filter / panel at a time: frames table, charts and blink follow the selector
  $("#runSel").hidden = !runs;
  $("#runSel").innerHTML = runs ? runs.map((r, k) => `<button data-k="${k}">${esc(r.name)}${r.frames ? ` <span class="muted">${r.frames.length}</span>` : ""}</button>`).join("") : "";
  const pick = k => {
    $$("#runSel button").forEach(b => b.classList.toggle("on", +b.dataset.k === k));
    renderFrames(runs[k].frames || []);
  };
  $$("#runSel button").forEach(b => b.onclick = () => {
    pick(+b.dataset.k);
    const t = $$("#resTabs button").find(x => x.dataset.name === runs[+b.dataset.k].name);
    if (t) t.click();
  });
  if (runs) pick(0); else renderFrames(rep.frames || []);
}
function renderFrames(fr) {
  const num = (v, d) => Number.isFinite(+v) && v !== null ? (+v).toFixed(d) : "";
  $("#qTable").innerHTML = `<tr><th>File</th><th>Stelle</th><th>FWHM</th><th>Ecc.</th><th>SNR</th><th>Peso</th><th>Punti</th><th></th></tr>` +
    fr.map((f, i) => `<tr data-i="${i}" class="${f.accepted ? "" : "rej"}"><td>${f.file.replace(/\.fits?$/i, "")}</td><td>${f.stars}</td>
      <td>${num(f.fwhm, 2)}</td><td>${num(f.eccentricity, 2)}</td><td>${num(f.snr, 0)}</td><td>${num(f.weight, 2)}</td>
      <td>${num(f.score, 0)}</td><td>${f.accepted ? "<svg class=\"mark\" viewBox=\"0 0 24 18\" aria-label=\"accettato\"><path d=\"M5 12C2 7 7 2 13 2.5c6 .5 9 4 8 8-1 4.5-8 6.5-13 5-3-1-4-2.5-3.5-4.5\"/></svg>" : f.reason}</td></tr>`).join("");
  $$("#qTable tr[data-i]").forEach(tr => tr.onclick = () => selectFrame(+tr.dataset.i, false));
  blinkFrames = fr; blinkAt = 0; bk.zoom = 0; bk.fw = bk.fh = 0;
  drawCharts(fr); showBlink();
}

// ---------------------------------------------------------------- frame charts (inline SVG)
const CHARTS = [["fwhm", "FWHM", 2], ["eccentricity", "Eccentricità", 2], ["snr", "SNR", 0], ["weight", "Peso", 2]];
const CW = 300, CH = 64, CP = 6;
function drawCharts(fr) {
  const box = $("#charts");
  if (!fr.length) { box.innerHTML = ""; return; }
  const X = i => fr.length === 1 ? CW / 2 : CP + i * (CW - 2 * CP) / (fr.length - 1);
  box.innerHTML = CHARTS.map(([k, name, dec]) => {
    const v = fr.map(f => Number(f[k]) || 0);
    const lo = Math.min(...v), hi = Math.max(...v), span = hi - lo || 1;
    const Y = x => CH - CP - (x - lo) / span * (CH - 2 * CP);
    const line = v.map((x, i) => `${X(i).toFixed(1)},${Y(x).toFixed(1)}`).join(" ");
    const dots = v.map((x, i) => `<circle data-i="${i}" cx="${X(i).toFixed(1)}" cy="${Y(x).toFixed(1)}" r="2.6"
      class="${fr[i].accepted ? "" : "rej"}"><title>${fr[i].file}: ${x.toFixed(dec)}${fr[i].accepted ? "" : " (scartato)"}</title></circle>`).join("");
    return `<div class="chart"><div class="lbl"><span>${name}</span><span class="muted">${lo.toFixed(dec)} – ${hi.toFixed(dec)}</span></div>
      <svg viewBox="0 0 ${CW} ${CH}"><polyline points="${line}"/>${dots}</svg></div>`;
  }).join("") + `<div class="chartlegend muted"><i style="background:var(--accent2)"></i>usato<i style="background:var(--bad)"></i>scartato
    <i style="background:var(--accent)"></i>selezionato · clic su un grafico per vedere il frame</div>`;
  $$("svg", box).forEach(svg => svg.onclick = e => {
    const r = svg.getBoundingClientRect(), fx = (e.clientX - r.left) / r.width * CW;
    const i = fr.length === 1 ? 0 : Math.round((fx - CP) / (CW - 2 * CP) * (fr.length - 1));
    selectFrame(Math.max(0, Math.min(fr.length - 1, i)), true);
  });
  highlightFrame();
}
function highlightFrame() {
  $$("#charts circle").forEach(c => {
    const on = +c.dataset.i === blinkAt;
    c.classList.toggle("sel", on); c.setAttribute("r", on ? "4.2" : "2.6");
  });
  $$("#qTable tr[data-i]").forEach(tr => tr.classList.toggle("sel", +tr.dataset.i === blinkAt));
}
function selectFrame(i, scrollRow) {
  blinkAt = i; showBlink();
  const tr = $(`#qTable tr[data-i="${i}"]`);
  if (scrollRow && tr) tr.scrollIntoView({ block: "nearest" });
}

// ---------------------------------------------------------------- blink
let blinkFrames = [], blinkAt = 0, blinkTimer = null;
const blinkCache = {};
// zoom: 0 = whole frame; otherwise screen pixels per sensor pixel (1 = 1:1). cx, cy = view center, sensor pixels
const ZOOMS = [0.5, 1, 2, 4, 8];
const bk = { zoom: 0, cx: 0, cy: 0, fw: 0, fh: 0, token: 0, url: null, drag: null };
const frameUrl = f => `/api/frame?path=${encodeURIComponent(f.calibrated)}&max_side=1000`;
function showBlink() {
  if (!blinkFrames.length) return;
  const f = blinkFrames[blinkAt];
  if (bk.zoom) loadCrop();
  else {
    $("#bkView").classList.remove("zoomed"); $("#bkView").style.height = "";
    $("#bkImg").style.transform = ""; $("#bkImg").src = frameUrl(f);
    // preload the next frame so playback is smooth
    const n = blinkFrames[(blinkAt + 1) % blinkFrames.length];
    if (n && !blinkCache[n.calibrated]) { blinkCache[n.calibrated] = new Image(); blinkCache[n.calibrated].src = frameUrl(n); }
  }
  $("#bkZoomLbl").textContent = bk.zoom ? `${bk.zoom >= 1 ? bk.zoom + ":1" : "1:" + 1 / bk.zoom} · centro ${Math.round(bk.cx)}, ${Math.round(bk.cy)}` : "intero";
  $("#bkName").textContent = `${blinkAt + 1}/${blinkFrames.length} · ${f.file} · FWHM ${(+f.fwhm).toFixed(2)} · ${f.accepted ? "ok" : "scartato: " + f.reason}`;
  $("#bkExclude").checked = state.exclude.includes(f.path);
  highlightFrame();
}
async function frameSize() {
  if (!bk.fw) {
    const i = await api(`/api/frame_info?path=${encodeURIComponent(blinkFrames[blinkAt].calibrated)}`);
    bk.fw = i.width; bk.fh = i.height;
  }
}
async function loadCrop() {
  const view = $("#bkView"), img = $("#bkImg"), f = blinkFrames[blinkAt];
  if (!view.classList.contains("zoomed")) {
    // keep the height the whole-frame view had, so the page does not jump
    view.style.height = Math.max(300, Math.round(img.getBoundingClientRect().height || 500)) + "px";
    view.classList.add("zoomed");
  }
  const w = Math.round(view.clientWidth), h = Math.round(view.clientHeight), token = ++bk.token;
  try {
    const r = await fetch(`/api/frame_crop?path=${encodeURIComponent(f.calibrated)}&x=${bk.cx.toFixed(1)}&y=${bk.cy.toFixed(1)}&zoom=${bk.zoom}&w=${w}&h=${h}`);
    if (!r.ok) throw new Error(`${r.status}`);
    const blob = await r.blob();
    if (token !== bk.token) return;            // a newer request is on its way
    bk.cx = +r.headers.get("X-Cx"); bk.cy = +r.headers.get("X-Cy");
    bk.fw = +r.headers.get("X-Frame-Width"); bk.fh = +r.headers.get("X-Frame-Height");
    if (bk.url) URL.revokeObjectURL(bk.url);
    bk.url = URL.createObjectURL(blob);
    img.onload = () => { img.style.transform = ""; img.onload = null; };
    img.src = bk.url;
    $("#bkZoomLbl").textContent = `${bk.zoom >= 1 ? bk.zoom + ":1" : "1:" + 1 / bk.zoom} · centro ${Math.round(bk.cx)}, ${Math.round(bk.cy)}`;
  } catch (e) { $("#bkZoomLbl").textContent = "errore zoom: " + e.message; }
}
// zoom to level z keeping the sensor point (sx, sy) where it is on screen
async function setZoom(z, sx, sy) {
  if (!blinkFrames.length) return;
  if (!z) { bk.zoom = 0; showBlink(); return; }
  await frameSize();
  if (!bk.zoom) { bk.cx = sx ?? bk.fw / 2; bk.cy = sy ?? bk.fh / 2; }
  else if (sx !== undefined) { bk.cx = sx + (bk.cx - sx) * bk.zoom / z; bk.cy = sy + (bk.cy - sy) * bk.zoom / z; }
  bk.zoom = z; showBlink();
}
function zoomStep(dir, sx, sy) {
  const i = ZOOMS.indexOf(bk.zoom);
  if (dir > 0) setZoom(bk.zoom ? ZOOMS[Math.min(ZOOMS.length - 1, i + 1)] : 1, sx, sy);
  else setZoom(i <= 0 ? 0 : ZOOMS[i - 1], sx, sy);
}
// screen point -> sensor pixel of the frame shown
function sensorAt(e) {
  const r = $("#bkImg").getBoundingClientRect();
  if (!r.width || !bk.fw) return [undefined, undefined];
  if (!bk.zoom) return [(e.clientX - r.left) / r.width * bk.fw, (e.clientY - r.top) / r.height * bk.fh];
  return [bk.cx + (e.clientX - (r.left + r.width / 2)) / bk.zoom, bk.cy + (e.clientY - (r.top + r.height / 2)) / bk.zoom];
}
$("#bkView").addEventListener("wheel", async e => {
  if (!blinkFrames.length) return;
  e.preventDefault();
  await frameSize();
  const [sx, sy] = sensorAt(e);
  zoomStep(e.deltaY < 0 ? 1 : -1, sx, sy);
}, { passive: false });
$("#bkView").addEventListener("mousedown", e => {
  if (!bk.zoom || e.button !== 0) return;
  e.preventDefault();
  bk.drag = { x: e.clientX, y: e.clientY, cx: bk.cx, cy: bk.cy };
  $("#bkView").classList.add("drag");
});
window.addEventListener("mousemove", e => {
  if (!bk.drag) return;
  $("#bkImg").style.transform = `translate(${e.clientX - bk.drag.x}px, ${e.clientY - bk.drag.y}px)`;
});
window.addEventListener("mouseup", e => {
  if (!bk.drag) return;
  const d = bk.drag; bk.drag = null;
  $("#bkView").classList.remove("drag");
  const dx = e.clientX - d.x, dy = e.clientY - d.y;
  if (Math.abs(dx) + Math.abs(dy) < 2) { $("#bkImg").style.transform = ""; return; }
  bk.cx = d.cx - dx / bk.zoom; bk.cy = d.cy - dy / bk.zoom;
  showBlink();
});
$("#bkZoomIn").onclick = () => zoomStep(1);
$("#bkZoomOut").onclick = () => zoomStep(-1);
$("#bkFit").onclick = () => setZoom(0);
$("#bkOne").onclick = () => setZoom(1);
document.addEventListener("keydown", e => {
  if ($(".page.on")?.dataset.page !== "4" || !blinkFrames.length) return;      // blink: Risultati
  if (e.target.closest?.("input, select, textarea") || e.ctrlKey || e.altKey || e.metaKey) return;
  if (e.key === "ArrowLeft") { e.preventDefault(); stepBlink(-1); }
  else if (e.key === "ArrowRight") { e.preventDefault(); stepBlink(1); }
  else if (e.key === "+" || e.key === "=") zoomStep(1);
  else if (e.key === "-") zoomStep(-1);
});
const stepBlink = d => { blinkAt = (blinkAt + d + blinkFrames.length) % blinkFrames.length; showBlink(); };
$("#bkPrev").onclick = () => stepBlink(-1);
$("#bkNext").onclick = () => stepBlink(1);
$("#bkPlay").onclick = () => {
  if (blinkTimer) { clearInterval(blinkTimer); blinkTimer = null; $("#bkPlay").innerHTML = ico("play") + "Play"; return; }
  blinkTimer = setInterval(() => stepBlink(1), 700); $("#bkPlay").innerHTML = ico("pause") + "Pausa";
};
$("#bkExclude").onchange = e => {
  const p = blinkFrames[blinkAt].path;
  state.exclude = e.target.checked ? [...new Set([...state.exclude, p])] : state.exclude.filter(x => x !== p);
  save();
};
$("#bkRerun").onclick = () => $("#startStack").click();

// ---------------------------------------------------------------- 6: post
const pluginById = id => plugins.find(p => p.id === id);
const CAT_IT = { background: "Sfondo", color: "Colore", noise: "Rumore", stretch: "Stretch", contrast: "Contrasto",
  sharpen: "Nitidezza", stars: "Stelle", geometry: "Geometria", cosmetic: "Cosmetica", export: "Esporta",
  analysis: "Analisi", tools: "Strumenti" };
const catName = c => CAT_IT[c] || c;
const clone = o => JSON.parse(JSON.stringify(o));
const presetName = p => p.label || p.name || p.id;
// "" = empty pipeline: the user picks every step with "+ Aggiungi passo…"
const EMPTY = "";
// only the names in the menu (long descriptions made the native menu as wide as the screen): the description of
// the chosen profile is shown under it
const fillPresets = () => {
  $("#presetSel").innerHTML = `<option value="${EMPTY}">— vuoto: scegli tu i passi —</option>` +
    presets.map(p => `<option value="${esc(p.id)}" title="${esc(p.description || "")}">${esc(presetName(p))}</option>`).join("");
  showPresetDesc();
};
function showPresetDesc() {
  const p = presets.find(x => x.id === state.preset);
  $("#presetDesc").textContent = p ? (p.description || "") : (state.preset === EMPTY ? "Pipeline vuota: scegli tu i passi." : "");
}
async function enterPost() {
  if (!state.input) { alert("Prima esegui lo stacking o apri un'immagine integrata."); go(1); return; }
  if (!plugins.length) {
    [plugins, presets] = await Promise.all([api("/api/plugins"), api("/api/presets")]);
    fillPresets();
  }
  $("#postInput").textContent = state.input ? state.input.split(/[\\/]/).pop() : "";
  identify(state.input).then(r => {          // what the image shows, next to its file name
    if (r && r.name && state.input) $("#postInput").textContent = `${state.input.split(/[\\/]/).pop()} · ${r.name}`;
  });
  if (!state.steps.length) loadPreset(state.preset);
  $("#presetSel").value = state.preset;
  $("#presetAuto").textContent = state.autoReason ? `scelto in automatico: ${state.autoReason}` : "";
  renderSteps(); refresh(); updateSplit();
}
// the whole step is kept (params, "mask", "enabled", "preview"): masks survive a preset round trip
function loadPreset(id) {
  if (id === EMPTY) { state.preset = EMPTY; state.steps = []; openStep = null; save(); showPresetDesc(); return; }
  const p = presets.find(x => x.id === id); if (!p) return;
  state.preset = id; showPresetDesc();
  state.steps = clone(p.steps).map(s => ({ ...s, params: s.params || {}, enabled: s.enabled !== false }));
  if (state.hybridStep && state.hybridFor === state.input && !state.steps.some(s => s.plugin === "color.nb_rgb")) {
    const firstStretch = state.steps.findIndex(s => s.plugin.startsWith("stretch."));
    const bgs = state.steps.map((s, i) => s.plugin.startsWith("background.") && (firstStretch < 0 || i < firstStretch) ? i : -1).filter(i => i >= 0);
    const at = bgs.length ? bgs[bgs.length - 1] + 1 : (state.steps[0] && state.steps[0].plugin.startsWith("crop.") ? 1 : 0);
    state.steps.splice(at, 0, { ...clone(state.hybridStep), enabled: true });
  }
  openStep = null; save();
}
$("#presetSel").onchange = e => { state.autoReason = ""; $("#presetAuto").textContent = "";
  loadPreset(e.target.value); renderSteps(); refresh(); updateMask(); };
$("#renamePreset").onclick = async () => {
  const cur = presets.find(x => x.id === state.preset); if (!cur) return;
  const label = prompt(`Nuovo nome per il profilo "${presetName(cur)}"`, presetName(cur)); if (!label) return;
  try {
    await api(`/api/presets/${encodeURIComponent(cur.id)}/rename`, { label });
    presets = await api("/api/presets"); fillPresets(); $("#presetSel").value = cur.id;
  } catch (e) { alert(e.message); }
};
$("#refPreset").onclick = () => openBrowser("ref");
async function presetFromImage(path) {
  const cur = presets.find(x => x.id === state.preset);
  const label = prompt("Nome del nuovo profilo (passi del profilo attuale + il look di questa immagine)",
    "come " + path.split(/[\\/]/).pop().replace(/\.[^.]+$/, "")); if (!label) return;
  try {
    const r = await api("/api/presets_from_image", { path, base: cur ? cur.id : "nebula_pro", label });
    presets = await api("/api/presets"); fillPresets();
    loadPreset(r.id); $("#presetSel").value = r.id; renderSteps(); refresh();
    alert(`Profilo "${r.label}" creato: cielo ${r.look.sky}, nebulosa ${r.look.nebula}, saturazione ${r.look.saturation}`);
  } catch (e) { alert(e.message); }
}
$("#savePreset").onclick = async () => {
  if (!state.steps.length) { alert("La pipeline è vuota: aggiungi almeno un passo."); return; }
  const name = prompt("Nome del profilo (stesso nome = sovrascrive)", state.preset ? state.preset + "_mio" : "mio_profilo"); if (!name) return;
  try {
    const cur = presets.find(x => x.id === state.preset);
    const r = await api(`/api/presets/${encodeURIComponent(name)}`, { name,
      description: cur && cur.id === name ? cur.description : "preset personale", steps: clone(state.steps) });
    presets = await api("/api/presets"); fillPresets();
    state.preset = r.id; $("#presetSel").value = r.id; save();
  } catch (e) { alert(e.message); }
};

// ---------------------------------------------------------------- plugin picker (categories + search)
// plugin picker: whole page, one column per section in workflow order, a short Italian line per plugin
// (the full English description stays in the tooltip)
const PK_SECTIONS = [
  ["geometry", "Geometria", "ritaglio, rotazione"], ["cosmetic", "Cosmetica", "pixel caldi, satelliti, ritocchi"],
  ["background", "Sfondo", "gradienti e inquinamento luminoso"], ["color", "Colore", "calibrazione e correzioni"],
  ["noise", "Rumore", "riduzione del rumore"], ["sharpen", "Nitidezza", "deconvoluzione e dettaglio"],
  ["stretch", "Stretch", "da lineare a visibile"], ["stars", "Stelle", "togliere, lavorare, rimettere"],
  ["contrast", "Contrasto e look", "curve, saturazione, look finale"], ["analysis", "Analisi", ""],
  ["annotate", "Testo", ""], ["tools", "Strumenti", ""], ["export", "Esporta", "annotazioni, web"]];
const PK_IT = {
  "crop.auto": "Taglia i bordi vuoti dello stacking", "crop.manual": "Ritaglio a mano", "geometry.rotate": "Ruota o capovolgi",
  "ext.cc_superres": "Ingrandisce 2-4x con l'AI (Cosmic Clarity)",
  "cosmetic.hot_pixels": "Toglie pixel caldi e freddi", "cosmetic.chroma_noise": "Toglie il rumore di colore",
  "cosmetic.heal": "Timbro clone su un difetto", "ext.cc_satellite": "Toglie le scie dei satelliti (Cosmic Clarity)",
  "background.auto": "Sceglie da solo DBE o superficie semplice", "background.dbe": "Estrazione del fondo con campioni (DBE)",
  "background.poly": "Superficie semplice (piano o grado 2)", "background.allsky": "Gradiente con immagine di riferimento DSS2",
  "ext.graxpert_background": "Gradiente con l'AI di GraXpert", "color.gradient_fit": "Toglie gradienti di colore",
  "color.white_balance_stars": "Bilancia il bianco sulle stelle", "color.neutralize_background": "Rende grigio il fondo cielo",
  "color.spcc": "Calibrazione colore con il catalogo Gaia (SPCC)", "color.pcc": "Calibrazione colore con APASS (PCC)",
  "color.linear_fit": "Allinea i livelli dei canali", "color.fix_saturated": "Ripara i nuclei saturi",
  "color.scnr": "Toglie la dominante verde", "color.selective": "Cambia saturazione o tinta di un colore",
  "color.fix_magenta": "Toglie il magenta (stelle SHO)", "color.nb_normalization": "Bilancia Ha/OIII/SII (banda stretta)",
  "color.sho_gold": "Palette Hubble oro/blu",
  "ext.deepsnr": "Riduzione rumore AI (DeepSNR), la migliore nelle prove", "noise.tgv": "Riduzione rumore TGV, molto buona",
  "ext.graxpert_denoise": "Riduzione rumore AI di GraXpert", "noise.mmt": "Riduzione rumore multiscala (MMT)",
  "noise.wavelet": "Riduzione rumore wavelet, leggera", "noise.first_layer_subtract": "Toglie il primo livello (grana fine)",
  "sharpen.deconvolution": "Deconvoluzione: stelle più piccole, più dettaglio", "sharpen.mlt": "MultiscaleLinearTransform a 5 livelli",
  "ext.cosmic_clarity": "Nitidezza o rumore con l'AI (Cosmic Clarity)", "ext.rc_astro": "BlurX / NoiseX / StarX (a pagamento)",
  "sharpen.wavelet": "Rinforza le scale medie", "sharpen.unsharp": "Maschera di contrasto classica",
  "stretch.auto": "Stretch automatico (STF)", "stretch.ghs": "Generalized Hyperbolic Stretch", "stretch.histogram": "Istogramma a mano",
  "stretch.linked_mtf": "Stretch uguale sui tre canali", "stretch.statistical": "Stretch a una mediana scelta",
  "stretch.arcsinh": "Stretch arcsinh (tiene i colori delle stelle)",
  "stars.starnet": "Toglie le stelle (StarNet), il più pulito", "stars.darkstar": "Toglie le stelle (Dark Star)",
  "stars.remove_basic": "Toglie le stelle senza programmi esterni", "stars.stretch": "Stira il livello stelle",
  "stars.morph": "Trasformazione morfologica: stelle più piccole, nucleo luminoso", "stars.size": "Stelle più grandi o più piccole",
  "stars.reduce": "Riduce le stelle con una maschera", "stars.halo_reduce": "Riduce gli aloni delle stelle",
  "stars.color_boost": "Più colore nelle stelle", "stars.halo_clean": "Toglie dischi rotondi (polvere dei flat)",
  "stars.recombine": "Rimette le stelle",
  "contrast.curves": "Curve (RGB, luminosità, saturazione…)", "contrast.saturation": "Saturazione",
  "contrast.local": "Contrasto locale (CLAHE)", "contrast.hdr": "Comprime le zone troppo luminose",
  "contrast.dark_structure": "Rinforza le strutture scure", "look.match": "Porta il look verso i finali di riferimento",
  "analysis.plate_solve": "Trova coordinate e campo (ASTAP)", "analysis.stats": "Statistiche dell'immagine",
  "annotate.text": "Scritta sull'immagine", "tools.pixelmath": "Espressioni sui pixel (PixelMath)",
  "tools.hdr_composition": "Unisce pose diverse (HDR)",
  "stretch.stf": "STF di PixInsight: canali collegati o separati",
  "stretch.masked": "MaskedStretch: fondo stirato, stelle protette", "stretch.adaptive": "AdaptiveStretch: curva calcolata dal segnale",
  "geometry.integer_resample": "Binning o ingrandimento intero (IntegerResample)",
  "sharpen.larson_sekanina": "Getti e gusci delle comete (LarsonSekanina)",
  "tools.convolution": "Sfocatura gaussiana, box o Moffat", "tools.invert": "Negativo dell'immagine",
  "tools.binarize": "Bianco sopra la soglia, nero sotto", "tools.grayscale": "Converte in scala di grigi",
  "tools.channel_extract": "Tiene un solo canale (R, G, B, L, Lab, HSV)",
  "color.nb_rgb": "Ha / OIII / SII dentro la foto a colori, con cursori (NBRGBCombination)",
  "color.lrgb": "Luminanza separata sotto il colore (LRGB)", "color.channel_combine": "Unisce file mono nei canali R, G, B",
"annotate.objects": "Nomi degli oggetti sull'immagine",
  "export.web": "Ridimensiona per web/social con firma" };
const PK_EXT = new Set(["stars.starnet", "stars.darkstar"]);
function openPicker() {
  $("#picker").hidden = false; $("#pkSearch").value = ""; renderPicker(); $("#pkSearch").focus();
}
function renderPicker() {
  const q = $("#pkSearch").value.trim().toLowerCase();
  const match = p => !q || `${p.name} ${p.id} ${PK_IT[p.id] || ""} ${p.description} ${catName(p.category)}`.toLowerCase().includes(q);
  const known = new Set(PK_SECTIONS.map(x => x[0]));
  const extra = [...new Set(plugins.map(p => p.category))].filter(c => !known.has(c)).map(c => [c, catName(c), ""]);
  const html = PK_SECTIONS.concat(extra).map(([cat, title, hint]) => {
    const ps = plugins.filter(p => p.category === cat && match(p)).sort((a, b) => a.name.localeCompare(b.name));
    if (!ps.length) return "";
    return `<section class="pksec"><h3>${esc(title)}${hint ? ` <span class="muted">${esc(hint)}</span>` : ""}</h3>` +
      ps.map(p => `<button class="pkitem" data-id="${esc(p.id)}" title="${esc(p.description || "")}">
        <span class="pkname">${esc(p.name)}${p.stage === "linear" ? ' <span class="pkb lin">L</span>' : ""}${p.id.startsWith("ext.") || PK_EXT.has(p.id) ? ' <span class="pkb ext">EXT</span>' : ""}</span>
        <span class="pkdesc">${esc(PK_IT[p.id] || (p.description || "").split(". ")[0])}</span></button>`).join("") + "</section>";
  }).join("");
  $("#pkList").innerHTML = html ? `<div class="pkcols">${html}</div>` : '<div class="muted">nessun plugin trovato</div>';
}
document.addEventListener("keydown", e => { if (e.key === "Escape" && !$("#picker").hidden) $("#picker").hidden = true; });
$("#addStep").onclick = openPicker;
$("#pkSearch").oninput = renderPicker;
$("#pkCancel").onclick = () => $("#picker").hidden = true;
$("#pkList").onclick = e => {
  const it = e.target.closest(".pkitem"); if (!it) return;
  // inserted after the open step, otherwise at the end
  const at = openStep !== null && openStep !== undefined && openStep < state.steps.length ? openStep + 1 : state.steps.length;
  state.steps.splice(at, 0, { plugin: it.dataset.id, params: {}, enabled: true });
  $("#picker").hidden = true; save(); renderSteps(at); refresh(); updateMask();
};

// ---------------------------------------------------------------- step list (drag & drop, up/down, duplicate)
let openStep = null, dragFrom = null;
function moveStep(from, to) {
  if (from === to || to < 0 || to >= state.steps.length) return;
  const open = openStep !== null && openStep !== undefined ? state.steps[openStep] : null;
  const [s] = state.steps.splice(from, 1);
  state.steps.splice(to, 0, s);
  openStep = open ? state.steps.indexOf(open) : null;
  save(); renderSteps(); refresh(); updateMask();
}
// "Mostra": opening a step applied to the stars shows the star layer; leaving it goes back to the result
let autoView = false;
function syncView() {
  const sel = $("#viewSel"); if (!sel) return;
  const stars = openStep !== null && openStep !== undefined && state.steps[openStep]?.target === "stars";
  if (stars && sel.value === "") { sel.value = "stars"; autoView = true; refreshSoon(); }
  else if (!stars && autoView && sel.value === "stars") { sel.value = ""; autoView = false; refreshSoon(); }
}
function renderSteps(open) {
  if (open !== undefined) openStep = open;
  const ol = $("#stepList"); ol.innerHTML = "";
  if (!state.steps.length) ol.innerHTML = '<li class="empty muted">Pipeline vuota: aggiungi i passi che vuoi con «+ Aggiungi passo…» (li riordini trascinandoli), poi «Salva» per farne un profilo.</li>';
  state.steps.forEach((s, i) => {
    const pl = pluginById(s.plugin);
    const li = document.createElement("li");
    li.className = (s.enabled === false ? "off " : "") + (openStep === i ? "open" : "");
    li.innerHTML = `<div class="head" draggable="true"><span class="grip" title="Trascina per riordinare">${ico("grip-vertical")}</span>
      <input type="checkbox" ${s.enabled === false ? "" : "checked"} title="Attiva / disattiva">
      <span class="name" title="${esc(s.plugin)}"><span class="nm"><span class="si">${String(i + 1).padStart(2, "0")}</span>${esc(pl ? pl.name : s.plugin)}</span><span class="cat">${esc(pl ? catName(pl.category) : "")}</span></span>${s.mask ? `<span class="mchip" title="Passo con maschera: ${esc(maskLabel(s.mask))}">M</span>` : ""}${s.target === "stars" ? `<span class="tchip" title="Applicato al livello stelle">${ico("star")}</span>` : ""}
      <span class="acts"><button title="Anteprima fino a qui">${ico("eye")}</button><button title="Su">${ico("arrow-up")}</button><button title="Giù">${ico("arrow-down")}</button>
      <button title="Duplica">${ico("copy")}</button><button title="Rimuovi">${ico("trash-2")}</button></span></div>
      <div class="body"></div>`;
    const head = $(".head", li), [cb] = $$("input", head), btns = $$("button", head);
    cb.onclick = e => { e.stopPropagation(); s.enabled = cb.checked; save(); renderSteps(); refresh(); };
    btns[0].onclick = e => { e.stopPropagation(); refresh(i + 1); };
    btns[1].onclick = e => { e.stopPropagation(); moveStep(i, i - 1); };
    btns[2].onclick = e => { e.stopPropagation(); moveStep(i, i + 1); };
    btns[3].onclick = e => { e.stopPropagation(); state.steps.splice(i + 1, 0, clone(s)); save(); renderSteps(i + 1); refresh(); };
    btns[4].onclick = e => { e.stopPropagation(); state.steps.splice(i, 1); save(); renderSteps(null); refresh(); updateMask(); updateDbe(); };
    head.onclick = () => { renderSteps(openStep === i ? null : i); updateDbe(); updateMask(); };
    // HTML5 drag and drop: the head is the handle (sliders inside the body keep working)
    head.ondragstart = e => { dragFrom = i; li.classList.add("dragging"); e.dataTransfer.effectAllowed = "move";
      try { e.dataTransfer.setData("text/plain", String(i)); } catch (err) {} };
    head.ondragend = () => { dragFrom = null; $$("#stepList li").forEach(x => x.classList.remove("dragging", "drop-before", "drop-after")); };
    li.ondragover = e => {
      if (dragFrom === null) return;
      e.preventDefault(); e.dataTransfer.dropEffect = "move";
      const r = li.getBoundingClientRect(), after = e.clientY > r.top + r.height / 2;
      li.classList.toggle("drop-before", !after); li.classList.toggle("drop-after", after);
    };
    li.ondragleave = () => li.classList.remove("drop-before", "drop-after");
    li.ondrop = e => {
      e.preventDefault();
      if (dragFrom === null) return;
      const r = li.getBoundingClientRect(), after = e.clientY > r.top + r.height / 2;
      let to = i + (after ? 1 : 0);
      if (dragFrom < to) to--;
      const from = dragFrom; dragFrom = null;
      moveStep(from, to);
    };
    const body = $(".body", li);
    if (pl) renderStepHelp(body, pl, s);
    if (pl && (afterStarRemoval(i) || s.target === "stars")) renderTarget(body, s, i);
    if (pl && s.plugin === "contrast.curves") renderCurveEditor(body, s, i);
    if (pl) renderParams(body, pl, s);
    else body.innerHTML = `<div class="help warn">plugin sconosciuto: ${esc(s.plugin)}</div>`;
    if (pl && s.plugin === "stretch.ghs" && pl.params.SP) renderSpPick(body, s, i);
    renderMaskEditor(body, s, i);
    ol.appendChild(li);
  });
  updateTools(); syncView();
}

// ---------------------------------------------------------------- star layer target ("Applica a: immagine / stelle")
// after a star removal (StarNet, Dark Star, RC-Astro StarXTerminator) a step can work on the star layer
// (step.target = "stars", pipeline.apply_step), like processing the separate star image in PixInsight
const isStarRemoval = s => s.enabled !== false && (["stars.starnet", "stars.darkstar"].includes(s.plugin) ||
  (s.plugin === "ext.rc_astro" && (s.params || {}).product === "sxt"));
const afterStarRemoval = i => state.steps.slice(0, i).some(isStarRemoval);
function renderTarget(body, step, i) {
  const on = step.target === "stars";
  const d = document.createElement("div"); d.className = "param target";
  d.innerHTML = `<div class="lbl"><span>Applica a</span></div>
    <div class="seg"><button data-v="image" class="${on ? "" : "on"}">immagine</button><button data-v="stars" class="${on ? "on" : ""}">stelle</button></div>
    ${on && !afterStarRemoval(i) ? '<div class="help warn">nessun passo di rimozione stelle prima di questo: il passo verrà saltato</div>' : ""}
    <div class="help">«stelle»: il passo lavora sul livello stelle separato dalla rimozione stelle; stars.recombine lo rimette poi nell'immagine.</div>`;
  $$("button", d).forEach(b => b.onclick = e => {
    e.stopPropagation();
    if (b.dataset.v === "stars") step.target = "stars"; else delete step.target;
    save(); renderSteps(); refresh(); updateMask();
  });
  body.appendChild(d);
}

// ---------------------------------------------------------------- GHS symmetry point from a click on the preview
let spPick = null;            // index of the GHS step waiting for a click
function renderSpPick(body, step, i) {
  const d = document.createElement("div"); d.className = "param";
  d.innerHTML = `<button class="btn small spbtn ${spPick === i ? "on" : ""}">${ico("crosshair")}Clicca per il punto di simmetria</button>
    <div class="help">poi clicca sull'anteprima (es. sul fondo cielo o sulla nebulosa): il valore dell'immagine che entra in
    questo passo in quel punto (mediana 5×5, luminanza o media RGB secondo «color») diventa SP.</div>`;
  $("button", d).onclick = e => {
    e.stopPropagation();
    spPick = spPick === i ? null : i;
    $("#pickInfo").textContent = spPick === null ? "" : "clicca sull'anteprima nel punto da usare come punto di simmetria (Esc = annulla)";
    renderSteps();
  };
  body.appendChild(d);
}
async function pickSP(p) {
  const i = spPick, step = state.steps[i];
  if (!step) return;
  $("#pickInfo").textContent = "lettura del pixel…";
  try {
    const d = await api("/api/pixel", { input: state.input, steps: state.steps, index: i, x: p.x, y: p.y });
    const pl = pluginById(step.plugin);
    const rgb = (step.params.color ?? pl?.params.color?.default) === "rgb";
    const v = Math.min(1, Math.max(0, +(rgb ? d.mean : d.value).toFixed(5)));
    step.params.SP = v; spPick = null; save(); renderSteps(); refresh();
    $("#pickInfo").textContent = `SP = ${v} dal punto ${Math.round(d.full[0])}, ${Math.round(d.full[1])}` +
      (d.channels.length === 3 ? ` (R ${d.channels[0].toFixed(4)} · G ${d.channels[1].toFixed(4)} · B ${d.channels[2].toFixed(4)})` : "");
  } catch (e) { $("#pickInfo").textContent = "valore non disponibile: " + e.message.slice(0, 100); }
}
// long lists (DBE samples) are summarised in the label; the full value stays in the text box
const paramText = v => v === null ? "auto" : Array.isArray(v) && v.length > 4 ? `${v.length} valori` :
  typeof v === "number" ? String(+v.toFixed(4)) : JSON.stringify(v);
// one control from the Param schema: choices -> select, bool -> checkbox, number with min+max -> slider,
// other numbers -> number box, strings -> text, lists / objects -> JSON text
function paramControl(spec, val) {
  const num = typeof spec.default === "number" || (spec.default === null && spec.min !== null);
  if (spec.choices) return `<select>${spec.choices.map(c => `<option ${c === val ? "selected" : ""} value="${esc(c)}">${esc(c)}</option>`).join("")}</select>`;
  if (typeof spec.default === "boolean") return `<input type="checkbox" ${val ? "checked" : ""}>`;
  if (num && spec.min !== null && spec.max !== null) {
    // whole numbers step by 1; decimals by the power of ten below a hundredth of the range (0-1 and 0-5: 0.01)
    const whole = spec.int ?? (Number.isInteger(spec.default ?? 0) && Number.isInteger(spec.min) && Number.isInteger(spec.max) && spec.default !== null);
    const st = whole ? 1 : 10 ** Math.floor(Math.log10((spec.max - spec.min) / 100));
    return `<input type="range" min="${spec.min}" max="${spec.max}" step="${st}" value="${val ?? spec.min}">`;
  }
  if (num) return `<input type="number" ${spec.min !== null ? `min="${spec.min}"` : ""} step="any" value="${val ?? ""}" placeholder="auto">`;
  if (typeof spec.default === "string") return `<input type="text" data-kind="str" value="${esc(val ?? "")}">`;
  return `<input type="text" data-kind="json" value="${esc(JSON.stringify(val))}">`;
}
function readControl(el) {
  if (el.type === "checkbox") return [true, el.checked];
  if (el.type === "range") return [true, +el.value];
  if (el.type === "number") return el.value === "" ? [true, null] : [Number.isFinite(+el.value), +el.value];
  if (el.tagName === "SELECT" || el.dataset.kind === "str") return [true, el.value];
  try { return [true, JSON.parse(el.value)]; } catch { return [false]; }
}
// what the step does: one short line, and the help of every parameter, only after "?" (the panels stay compact)
const helpOpen = new WeakSet();
function renderStepHelp(body, pl, step) {
  const sum = PK_IT[pl.id] || (pl.description || "").split(". ")[0];
  body.classList.toggle("showhelp", helpOpen.has(step));
  const d = document.createElement("div"); d.className = "stephelp";
  d.innerHTML = `<button class="hbtn" title="Cosa fa questo passo e i suoi parametri">${ico("circle-help")}</button><span class="hsum">${esc(sum)}</span>`;
  $("button", d).onclick = e => {
    e.stopPropagation();
    if (helpOpen.has(step)) helpOpen.delete(step); else helpOpen.add(step);
    body.classList.toggle("showhelp", helpOpen.has(step));
  };
  body.appendChild(d);
}
// parameters that are image files: a "Scegli…" button opens the file browser (narrowband, luminance, channels)
const FILE_PARAMS = { "color.nb_rgb": ["ha", "oiii", "sii"], "color.lrgb": ["luminance"], "color.channel_combine": ["R", "G", "B"] };
let browseTarget = null;           // {step, key} for the "param:file" browser mode
// ---------------------------------------------------------------- curves editor (Photoshop / PixInsight style)
// the same monotone spline as the server (scipy PchipInterpolator, enhance.curve_lut), drawn over the histogram of
// the image entering the step; drag points, click to add one, double click / right click to remove it
function pchip(pts) {
  const x = pts.map(p => p[0]), y = pts.map(p => p[1]), n = x.length;
  if (n < 2) return () => y[0] ?? 0;
  const h = [], d = [];
  for (let k = 0; k < n - 1; k++) { h.push(x[k + 1] - x[k]); d.push((y[k + 1] - y[k]) / (x[k + 1] - x[k])); }
  const m = new Array(n).fill(0);
  if (n === 2) { m[0] = m[1] = d[0]; }
  else {
    for (let k = 1; k < n - 1; k++) {
      if (d[k - 1] * d[k] > 0) {
        const w1 = 2 * h[k] + h[k - 1], w2 = h[k] + 2 * h[k - 1];
        m[k] = (w1 + w2) / (w1 / d[k - 1] + w2 / d[k]);
      }
    }
    const edge = (h0, h1, d0, d1) => {        // scipy's one-sided three-point estimate
      let e = ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1);
      if (Math.sign(e) !== Math.sign(d0)) e = 0;
      else if (Math.sign(d0) !== Math.sign(d1) && Math.abs(e) > Math.abs(3 * d0)) e = 3 * d0;
      return e;
    };
    m[0] = edge(h[0], h[1], d[0], d[1]);
    m[n - 1] = edge(h[n - 2], h[n - 3], d[n - 2], d[n - 3]);
  }
  return t => {
    if (t <= x[0]) return y[0];
    if (t >= x[n - 1]) return y[n - 1];
    let k = 0; while (k < n - 2 && t > x[k + 1]) k++;
    const hk = h[k], s = (t - x[k]) / hk, s2 = s * s, s3 = s2 * s;
    return (2 * s3 - 3 * s2 + 1) * y[k] + (s3 - 2 * s2 + s) * hk * m[k] + (-2 * s3 + 3 * s2) * y[k + 1] + (s3 - s2) * hk * m[k + 1];
  };
}
const CURVE_COLORS = { r: "#ff6b6b", g: "#6bdc7b", b: "#6ba8ff" };
// the channel tabs (contrast.CURVE_ORDER on the server: the curves are applied in this order)
const CURVE_TABS = [["rgb", "RGB"], ["r", "R"], ["g", "G"], ["b", "B"], ["l", "L*"], ["a", "a*"], ["b_star", "b*"],
  ["c", "Croma"], ["s", "Sat."], ["h", "Tinta"]];
const CURVE_PRESETS = [["Lineare", [[0, 0], [1, 1]]], ["Curva a S", [[0, 0], [0.25, 0.2], [0.75, 0.82], [1, 1]]],
  ["Schiarisci", [[0, 0], [0.35, 0.5], [1, 1]]]];
const CURVE_HINT = "clic per aggiungere un punto · doppio clic per toglierlo";
const histCache = new Map(), curveTab = new WeakMap();
const isIdentity = p => p.every(([x, y]) => Math.abs(x - y) < 1e-6);
// one curve per channel in step.params.curves; a step from an older profile (points on one channel) is moved there
function stepCurves(step) {
  if (!step.params.curves || typeof step.params.curves !== "object" || Array.isArray(step.params.curves) || !Object.keys(step.params.curves).length) {
    const def = pluginById("contrast.curves")?.params.points.default || [[0, 0], [1, 1]];
    step.params.curves = { [step.params.channel || "rgb"]: clone(step.params.points || def) };
    delete step.params.points; delete step.params.channel;
  }
  return step.params.curves;
}
function renderCurveEditor(body, step, index) {
  const S = 240, P = 8, W = S + 2 * P;                       // drawing area 0..1 -> P..P+S
  const curves = stepCurves(step);
  if (!curveTab.has(step)) curveTab.set(step, Object.keys(curves).find(k => !isIdentity(curves[k])) || "rgb");
  const box = document.createElement("div"); box.className = "curvebox";
  box.innerHTML = `<div class="ctabs">${CURVE_TABS.map(([k, n]) => `<button data-ch="${k}" title="curva del canale ${n}">${n}</button>`).join("")}</div>
    <svg viewBox="0 0 ${W} ${W}" aria-label="Curva"><g class="grid"></g><path class="hist"/><line class="diag"/>
      <g class="ghosts"></g><path class="curve"/><g class="pts"></g></svg>
    <div class="muted small coord">${CURVE_HINT}</div>
    <div class="row">${CURVE_PRESETS.map(([n], j) => `<button class="btn small" data-preset="${j}">${n}</button>`).join("")}
      <button class="btn small" data-reset-all title="Tutti i canali tornano lineari">Azzera tutti</button></div>`;
  const svg = $("svg", box), X = v => P + v * S, Y = v => P + (1 - v) * S;
  $(".grid", svg).innerHTML = [0.25, 0.5, 0.75].map(v => `<line x1="${X(v)}" y1="${P}" x2="${X(v)}" y2="${P + S}"/><line x1="${P}" y1="${Y(v)}" x2="${P + S}" y2="${Y(v)}"/>`).join("")
    + `<rect x="${P}" y="${P}" width="${S}" height="${S}"/>`;
  const diag = $(".diag", svg);
  diag.setAttribute("x1", X(0)); diag.setAttribute("y1", Y(0)); diag.setAttribute("x2", X(1)); diag.setAttribute("y2", Y(1));
  const ch = () => curveTab.get(step);
  const pts = () => {
    const c = stepCurves(step);
    if (!Array.isArray(c[ch()]) || c[ch()].length < 2) c[ch()] = [[0, 0], [1, 1]];
    return c[ch()];
  };
  const pathOf = p => {
    const f = pchip(p); let d = "";
    for (let k = 0; k <= 128; k++) { const t = k / 128; d += (k ? "L" : "M") + X(t).toFixed(1) + "," + Y(Math.max(0, Math.min(1, f(t)))).toFixed(1); }
    return d;
  };
  let hist = null, sel = -1, drag = -1, lastHit = -1, lastDown = 0;
  const draw = () => {
    const c = stepCurves(step), p = pts(), col = CURVE_COLORS[ch()] || "";
    $$(".ctabs button", box).forEach(b => {
      b.classList.toggle("on", b.dataset.ch === ch());
      b.classList.toggle("mod", Array.isArray(c[b.dataset.ch]) && !isIdentity(c[b.dataset.ch]));
    });
    // the other channels' curves, faint, in their colour
    $(".ghosts", svg).innerHTML = Object.entries(c).filter(([k, v]) => k !== ch() && Array.isArray(v) && v.length > 1 && !isIdentity(v))
      .map(([k, v]) => `<path d="${pathOf(v)}" style="stroke:${CURVE_COLORS[k] || "var(--muted)"}"/>`).join("");
    const cv = $(".curve", svg); cv.setAttribute("d", pathOf(p)); cv.style.stroke = col;
    $(".pts", svg).innerHTML = p.map(([a, b], j) => `<circle data-j="${j}" cx="${X(a)}" cy="${Y(b)}" r="${j === sel ? 5.5 : 4.5}" class="${j === sel ? "sel" : ""}"/>`).join("");
    const hp = $(".hist", svg), hv = hist && (hist[ch()] || hist.l);
    if (hv) {
      let hd = `M${X(0)},${Y(0)}`;
      hv.forEach((v, k) => { hd += `L${X(k / (hv.length - 1)).toFixed(1)},${(P + S - v * S * 0.9).toFixed(1)}`; });
      hp.setAttribute("d", hd + `L${X(1)},${Y(0)}Z`); hp.style.fill = col;
    } else hp.removeAttribute("d");
  };
  const coord = j => { $(".coord", box).textContent = j >= 0 ? `ingresso ${pts()[j][0].toFixed(3)} → uscita ${pts()[j][1].toFixed(3)}` : CURVE_HINT; };
  const changed = () => { save(); draw(); refreshSoon(); };
  // the point under the pointer: the nearest one within ~4% of the box (the dots are small to aim at)
  const nearest = (u, v) => {
    let best = -1, bd = 0.045;
    pts().forEach(([a, b], j) => { const d = Math.hypot(a - u, b - v); if (d < bd) { bd = d; best = j; } });
    return best;
  };
  const toUnit = e => {
    const r = svg.getBoundingClientRect(), k = W / r.width;
    return [Math.max(0, Math.min(1, ((e.clientX - r.left) * k - P) / S)), Math.max(0, Math.min(1, 1 - ((e.clientY - r.top) * k - P) / S))];
  };
  svg.addEventListener("pointerdown", e => {
    if (e.button !== 0) return;
    e.preventDefault(); e.stopPropagation();
    const [u, v] = toUnit(e), p = pts(), hit = nearest(u, v);
    // double click on a point removes it (pointerdown's preventDefault stops the browser's own dblclick)
    if (hit > 0 && hit < p.length - 1 && hit === lastHit && e.timeStamp - lastDown < 350) {   // never the end points
      p.splice(hit, 1); sel = -1; lastHit = -1; coord(-1); changed(); return;
    }
    lastHit = hit; lastDown = e.timeStamp;
    if (hit >= 0) drag = hit;
    else {                                         // a new point at the click, kept in x order
      const j = p.findIndex(q => q[0] > u);
      if (j <= 0 || p.some(q => Math.abs(q[0] - u) < 0.01)) return;
      p.splice(j, 0, [+u.toFixed(4), +v.toFixed(4)]); drag = j;
    }
    sel = drag; svg.setPointerCapture(e.pointerId); coord(drag); changed();
  });
  svg.addEventListener("pointermove", e => {
    if (drag < 0) return;
    const [u, v] = toUnit(e), p = pts();
    const lo = drag > 0 ? p[drag - 1][0] + 0.005 : 0, hi = drag < p.length - 1 ? p[drag + 1][0] - 0.005 : 1;
    p[drag] = [+Math.max(lo, Math.min(hi, u)).toFixed(4), +v.toFixed(4)];
    coord(drag); save(); draw(); refreshSoon();
  });
  const end = () => { drag = -1; };
  svg.addEventListener("pointerup", end); svg.addEventListener("pointercancel", end);
  svg.addEventListener("contextmenu", e => {
    e.preventDefault(); e.stopPropagation();
    const [u, v] = toUnit(e), hit = nearest(u, v), p = pts();
    if (hit <= 0 || hit >= p.length - 1) return;   // the end points stay
    p.splice(hit, 1); sel = -1; coord(-1); changed();
  });
  svg.addEventListener("click", e => e.stopPropagation());
  $$(".ctabs button", box).forEach(b => b.onclick = e => {
    e.stopPropagation(); curveTab.set(step, b.dataset.ch); sel = -1; coord(-1); draw();
  });
  $$("[data-preset]", box).forEach(b => b.onclick = e => {
    e.stopPropagation(); stepCurves(step)[ch()] = clone(CURVE_PRESETS[+b.dataset.preset][1]); sel = -1; coord(-1); changed();
  });
  $("[data-reset-all]", box).onclick = e => {
    e.stopPropagation(); step.params.curves = { rgb: [[0, 0], [1, 1]] }; curveTab.set(step, "rgb"); sel = -1; coord(-1); changed();
  };
  // histogram of the image entering this step (or of the star layer, for a step applied to the stars)
  const prev = state.steps.slice(0, index).filter(x => x.enabled !== false);
  const key = JSON.stringify([state.input, prev, step.target || ""]);
  const show = h => { hist = h; draw(); };
  if (histCache.has(key)) show(histCache.get(key));
  else if (state.input) api("/api/histogram", { input: state.input, steps: prev, target: step.target || "", max_side: 600 })
    .then(h => { if (histCache.size > 20) histCache.clear(); histCache.set(key, h); show(h); }).catch(() => {});
  body.curveRedraw = draw;
  draw(); body.appendChild(box);
}
function renderParams(body, pl, step) {
  Object.entries(pl.params).forEach(([k, spec]) => {
    if (pl.id === "contrast.curves" && ["points", "channel", "curves"].includes(k)) return;   // the curves editor
    const val = step.params[k] ?? spec.default;
    const d = document.createElement("div"); d.className = "param";
    d.innerHTML = `<div class="lbl"><span>${esc(k)}</span><span>${esc(paramText(val))}</span></div>${paramControl(spec, val)}
      ${spec.help ? `<div class="help">${esc(spec.help)}</div>` : ""}`;
    const el = $("input, select", d), shown = $(".lbl span:last-child", d);
    if (el && el.type === "text") shown.hidden = true;      // the box already shows it (long paths twice)
    if ((FILE_PARAMS[pl.id] || []).includes(k)) {
      const b = document.createElement("button"); b.className = "btn small pfile"; b.textContent = "Scegli…";
      b.title = "Scegli il file (FITS, XISF, TIFF)";
      b.onclick = e => {
        e.stopPropagation(); browseTarget = { step, key: k };
        const cur = String(step.params[k] || "");                      // start in the folder of the current file
        openBrowser("param:file", cur.includes("\\") ? cur.replace(/\\[^\\]*$/, "") : "");
      };
      el.insertAdjacentElement("afterend", b);
    }
    const update = () => {
      const [ok, v] = readControl(el);
      el.classList.toggle("bad", !ok);
      if (!ok) return;
      step.params[k] = v; shown.textContent = paramText(v); save(); refreshSoon();
      if (body.curveRedraw) body.curveRedraw();                   // channel changed: curve colour
    };
    el.oninput = update; el.onchange = update;
    body.appendChild(d);
  });
}

// ---------------------------------------------------------------- per-step masks (same JSON as the presets)
// [key, label, min, max, step, default]; the spec only stores the keys the user changed (+ type)
const MASK_TYPES = {
  range: ["Range (RangeSelection)", [["lower", "Limite inferiore", 0, 1, 0.01, 0.5], ["upper", "Limite superiore", 0, 1, 0.01, 1],
    ["fuzziness", "Sfumatura dei limiti", 0, 1, 0.01, 0], ["smoothness", "Morbidezza (px)", 0, 50, 0.5, 0]]],
  color: ["Colore (ColorMask)", [["width", "Ampiezza (gradi)", 10, 180, 1, 60], ["smooth", "Morbidezza (px)", 0, 50, 0.5, 0]]],
  luminance: ["Luminanza", [["black", "Nero (ombre)", 0, 1, 0.01, 0], ["mid", "Medi", 0.001, 0.999, 0.001, 0.5],
    ["white", "Bianco (luci)", 0, 1, 0.01, 1]]],
  channel: ["Canale (R, G, B, L)", [["black", "Nero (ombre)", 0, 1, 0.01, 0], ["mid", "Medi", 0.001, 0.999, 0.001, 0.5],
    ["white", "Bianco (luci)", 0, 1, 0.01, 1]]],
  stars: ["Stelle", [["grow", "Espansione (px)", 0, 20, 1, 2], ["soft", "Morbidezza stelle (px)", 0, 10, 0.1, 1.5]]],
};
const MASK_CHANNELS = [["R", "rosso (R)"], ["G", "verde (G)"], ["B", "blu (B)"], ["L", "luminanza (L)"]];
// painted mask: {type: "paint", base: <mask spec or null = black>, strokes: [[[x, y, r, value, softness], ...], ...]}
// (one stroke = the dabs of one drag, full-resolution pixels). The editor edits the base; the brush adds strokes.
const isPaint = m => !!m && m.type === "paint";
const baseOf = m => isPaint(m) ? m.base || null : m;
const maskLabel = m => isPaint(m) ? `pennello su ${m.base ? m.base.type : "nero"}` : m.type;
const MASK_COMMON = [["blur", "Sfocatura (px)", 0, 50, 0.5, 0], ["strength", "Intensità", 0, 1, 0.01, 1]];
const MASK_COLORS = [["red", "rosso"], ["yellow", "giallo"], ["green", "verde"], ["cyan", "ciano"], ["blue", "blu"], ["magenta", "magenta"]];
const MASK_OTHER = { lum: "Luminanza automatica", layer: "Livello salvato" };
function renderMaskEditor(body, step, index) {
  const outer = step.mask, base = baseOf(outer), t = isPaint(outer) && !base ? "paint" : base?.type || "";
  const m = base || outer;          // the spec the sliders edit (the base of a painted mask)
  const box = document.createElement("div"); box.className = "maskbox";
  const opts = [["", "nessuna"], ...Object.entries(MASK_TYPES).map(([k, v]) => [k, v[0]]),
    ["paint", "Solo pennello (parte da nero)"], ...(MASK_OTHER[t] ? [[t, MASK_OTHER[t]]] : [])];
  box.innerHTML = `<div class="lbl mhead"><span>Maschera</span>
      <select class="mtype">${opts.map(([k, n]) => `<option value="${k}" ${k === t ? "selected" : ""}>${n}</option>`).join("")}</select></div>`;
  const changed = () => { save(); refreshSoon(); updateMaskSoon(); };
  const slider = ([k, label, lo, hi, st, def]) => {
    const v = m[k] ?? def;
    const d = document.createElement("div"); d.className = "param";
    d.innerHTML = `<div class="lbl"><span>${label}</span><span>${+(+v).toFixed(3)}</span></div>
      <input type="range" min="${lo}" max="${hi}" step="${st}" value="${v}">`;
    const el = $("input", d);
    el.oninput = () => { m[k] = +el.value; $(".lbl span:last-child", d).textContent = +(+el.value).toFixed(3); changed(); };
    box.appendChild(d);
  };
  const check = (k, label, def) => {
    const d = document.createElement("label"); d.className = "param mcheck";
    d.innerHTML = `<input type="checkbox" ${(m[k] ?? def) ? "checked" : ""}> ${label}`;
    $("input", d).onchange = e => { m[k] = e.target.checked; changed(); };
    box.appendChild(d);
  };
  if (m && MASK_TYPES[t]) {
    if (t === "color") {
      const cur = Array.isArray(m.color) ? m.color[0] : m.color ?? "red";
      const d = document.createElement("div"); d.className = "param";
      d.innerHTML = `<div class="lbl"><span>Colore</span></div><select>${MASK_COLORS.map(([k, n]) =>
        `<option value="${k}" ${k === cur ? "selected" : ""}>${n}</option>`).join("")}${typeof cur === "number" ? `<option selected value="${cur}">${cur}°</option>` : ""}</select>`;
      $("select", d).onchange = e => { m.color = isNaN(+e.target.value) ? e.target.value : +e.target.value; changed(); };
      box.appendChild(d);
    }
    if (t === "channel") {
      const cur = String(m.channel ?? "G").toUpperCase();
      const d = document.createElement("div"); d.className = "param";
      d.innerHTML = `<div class="lbl"><span>Canale</span></div><select>${MASK_CHANNELS.map(([k, n]) =>
        `<option value="${k}" ${k === cur ? "selected" : ""}>${n}</option>`).join("")}</select>`;
      $("select", d).onchange = e => { m.channel = e.target.value; changed(); };
      box.appendChild(d);
    }
    MASK_TYPES[t][1].forEach(slider);
    if (t === "range") check("lightness", "Usa la luminosità L* (altrimenti per canale)", true);
  }
  if (m && MASK_OTHER[t]) box.insertAdjacentHTML("beforeend", `<div class="help">${t === "layer" ? `livello «${esc(m.name)}» creato da un passo precedente` : "luminanza della foto, stirata se lineare"}</div>`);
  if (m) {
    MASK_COMMON.forEach(slider);
    check("invert", "Inverti", false);
    box.insertAdjacentHTML("beforeend", `<div class="help">bianco = il passo si applica, nero = protetto. «Mostra maschera» (Ctrl+K) sull'anteprima: rosso = protetto, come PixInsight.</div>`);
    renderBrush(box, step, index);
  }
  $(".mtype", box).onchange = e => {
    const nt = e.target.value, strokes = isPaint(outer) ? outer.strokes || [] : [];
    if (!nt) {
      if (strokes.length && !confirm("Togliere la maschera e i tratti di pennello?")) { e.target.value = t; return; }
      delete step.mask;
    } else {
      const keep = {};
      ["invert", "blur", "strength"].forEach(k => { if (m && m[k] !== undefined) keep[k] = m[k]; });
      if (nt === "paint") step.mask = { type: "paint", base: null, strokes, ...keep };
      else step.mask = strokes.length ? { type: "paint", base: { type: nt, ...keep }, strokes } : { type: nt, ...keep };
    }
    save(); renderSteps(index); refresh();
    if (step.mask && !$("#showMask").checked) { $("#showMask").checked = true; }
    updateMask();
  };
  body.appendChild(box);
}

// ---------------------------------------------------------------- mask brush (Dynamic Paint Brush style)
// drag on the preview: soft round dabs in full-resolution pixels, base = the mask chosen above
if (!state.brush) state.brush = { radius: 80, soft: 0.5, value: 1 };
let brushOn = false, paintDrag = null;
const R_MIN = 2, R_MAX = 1500;     // brush radius slider (full-resolution pixels), logarithmic
const rToSlider = r => Math.round(100 * Math.log(r / R_MIN) / Math.log(R_MAX / R_MIN));
const sliderToR = v => Math.round(R_MIN * Math.pow(R_MAX / R_MIN, v / 100));
function renderBrush(box, step, index) {
  const strokes = isPaint(step.mask) ? step.mask.strokes || [] : [];
  const b = state.brush;
  const d = document.createElement("div");
  d.innerHTML = `<label class="param mcheck"><input type="checkbox" class="bon" ${brushOn ? "checked" : ""}> Pennello: dipingi la maschera sull'anteprima</label>
    <div class="brushctl" ${brushOn ? "" : "hidden"}>
      <div class="seg"><button data-v="1" class="${b.value >= 0.5 ? "on" : ""}">Dipingi (elabora)</button><button data-v="0" class="${b.value < 0.5 ? "on" : ""}">Cancella (proteggi)</button></div>
      <div class="param"><div class="lbl"><span>Dimensione (raggio, px)</span><span class="rv">${b.radius}</span></div>
        <input type="range" class="br" min="0" max="100" step="1" value="${rToSlider(b.radius)}"></div>
      <div class="param"><div class="lbl"><span>Morbidezza</span><span class="sv">${b.soft}</span></div>
        <input type="range" class="bs" min="0" max="1" step="0.05" value="${b.soft}"></div>
      <div class="row"><button class="btn small bundo" ${strokes.length ? "" : "disabled"}>${ico("undo-2")}Annulla tratto</button>
        <button class="btn small bclear" ${strokes.length ? "" : "disabled"}>Cancella tratti</button>
        <span class="muted">${strokes.length} tratt${strokes.length === 1 ? "o" : "i"}</span></div>
      <div class="help">trascina sull'anteprima: bianco = il passo si applica, «Cancella» = protegge. La maschera scelta sopra è la base da correggere.</div>
    </div>`;
  $(".bon", d).onchange = e => {
    brushOn = e.target.checked;
    if (brushOn) { spPick = null; if (!$("#showMask").checked) $("#showMask").checked = true; updateMask(); }
    renderSteps(index);
  };
  $$(".seg button", d).forEach(btn => btn.onclick = () => { b.value = +btn.dataset.v; save(); renderSteps(index); });
  $(".br", d).oninput = e => { b.radius = sliderToR(+e.target.value); $(".rv", d).textContent = b.radius; save(); };
  $(".bs", d).oninput = e => { b.soft = +e.target.value; $(".sv", d).textContent = b.soft; save(); };
  $(".bundo", d).onclick = () => { if (isPaint(step.mask)) { step.mask.strokes.pop(); unwrapPaint(step); maskEdited(index); } };
  $(".bclear", d).onclick = () => {
    if (!isPaint(step.mask) || !confirm("Cancellare tutti i tratti di pennello di questa maschera?")) return;
    step.mask.strokes = []; unwrapPaint(step); maskEdited(index);
  };
  box.appendChild(d);
}
// a painted mask without strokes goes back to its base (a black-only painted mask stays: it is what was chosen)
function unwrapPaint(step) {
  const m = step.mask;
  if (isPaint(m) && !(m.strokes || []).length && m.base) step.mask = m.base;
}
const maskEdited = index => { save(); renderSteps(index); refresh(); updateMask(); };
const openMaskStep = () => openStep !== null && openStep !== undefined ? state.steps[openStep] : null;
const brushActive = () => brushOn && !!openMaskStep()?.mask && !!state.input;

// tool layer over the preview: brush strokes or the GHS symmetry point click
let viewScale = 0;          // preview pixels / full-resolution pixels (X-Scale of /api/preview and /api/mask)
function updateTools() {
  const s = openMaskStep();
  if (spPick !== null && (spPick !== openStep || !s || s.plugin !== "stretch.ghs")) { spPick = null; $("#pickInfo").textContent = ""; }
  const brush = brushActive() && spPick === null;
  $("#toolLayer").hidden = !(brush || spPick !== null);
  $("#toolLayer").classList.toggle("brush", brush);
  if (!brush) $("#brushCur").hidden = true;
  if (!s || s.plugin !== "stretch.ghs") { if (spPick === null && /^SP =|^valore/.test($("#pickInfo").textContent)) $("#pickInfo").textContent = ""; }
}
function previewPoint(e) {
  const img = $("#postImg"), r = img.getBoundingClientRect(), W = img.naturalWidth, H = img.naturalHeight;
  if (!r.width || !r.height || !W || !H) return null;
  const x = (e.clientX - r.left) / r.width * W, y = (e.clientY - r.top) / r.height * H;
  if (!(x >= 0 && y >= 0 && x <= W && y <= H)) return null;
  return { x, y, W, H, k: r.width / W };
}
function moveBrushCursor(e) {
  const cur = $("#brushCur"), p = previewPoint(e);
  if (!brushActive() || !p || !viewScale) { cur.hidden = true; return; }
  const wr = $("#toolLayer").getBoundingClientRect(), rd = Math.max(2, state.brush.radius * viewScale * p.k);
  cur.hidden = false;
  cur.classList.toggle("erase", state.brush.value < 0.5);
  const z = zoom.z;                       // the cursor is inside the zoomed layer: its own px are screen px / zoom
  Object.assign(cur.style, { left: `${(e.clientX - wr.left - rd) / z}px`, top: `${(e.clientY - wr.top - rd) / z}px`,
    width: `${2 * rd / z}px`, height: `${2 * rd / z}px` });
}
// immediate feedback on the red overlay while dragging (the server mask replaces it at the end of the stroke)
function drawDab(px, py) {
  const cv = $("#maskCanvas");
  if (cv.hidden || !cv.width) return;
  const g = cv.getContext("2d"), k = cv.width / ($("#postImg").naturalWidth || cv.width);
  const r = Math.max(0.5, state.brush.radius * viewScale * k), core = Math.max(0, Math.min(r * (1 - state.brush.soft), r - 0.5));
  const x = px * k, y = py * k, paint = state.brush.value >= 0.5;
  const grad = g.createRadialGradient(x, y, core, x, y, r);
  grad.addColorStop(0, paint ? "rgba(0,0,0,1)" : "rgba(255,30,40,0.62)");
  grad.addColorStop(1, paint ? "rgba(0,0,0,0)" : "rgba(255,30,40,0)");
  g.save(); g.globalCompositeOperation = paint ? "destination-out" : "source-over"; g.fillStyle = grad;
  g.beginPath(); g.arc(x, y, r, 0, 2 * Math.PI); g.fill(); g.restore();
}
function addDab(p) {
  const d = paintDrag, b = state.brush, fx = p.x / viewScale, fy = p.y / viewScale;
  if (d.last) {
    const dist = Math.hypot(fx - d.last.fx, fy - d.last.fy), sp = Math.max(1, b.radius * 0.25);
    if (dist < sp) return;
    // fill the overlay along the segment so fast drags have no gaps (the server joins the dabs with segments)
    const n = Math.ceil(dist / sp);
    for (let j = 1; j < n; j++) drawDab(d.last.px + (p.x - d.last.px) * j / n, d.last.py + (p.y - d.last.py) * j / n);
  }
  d.stroke.push([+fx.toFixed(1), +fy.toFixed(1), b.radius, b.value, b.soft]);
  drawDab(p.x, p.y);
  d.last = { fx, fy, px: p.x, py: p.y };
}
const tl = $("#toolLayer");
tl.addEventListener("pointerdown", e => {
  if (e.button !== 0) return;
  const p = previewPoint(e);
  if (!p) return;
  e.preventDefault();
  if (spPick !== null) { pickSP(p); return; }
  const step = openMaskStep();
  if (!brushActive() || !viewScale) return;
  if (!isPaint(step.mask)) step.mask = { type: "paint", base: step.mask, strokes: [] };
  step.mask.strokes ||= [];
  const stroke = [];
  step.mask.strokes.push(stroke);
  paintDrag = { step, stroke, index: openStep, last: null };
  try { tl.setPointerCapture(e.pointerId); } catch (err) {}
  addDab(p);
});
tl.addEventListener("pointermove", e => {
  moveBrushCursor(e);
  if (!paintDrag) return;
  const p = previewPoint(e);
  if (p) addDab(p);
});
const endStroke = () => {
  if (!paintDrag) return;
  const d = paintDrag; paintDrag = null;
  if (!d.stroke.length) d.step.mask.strokes.pop();
  maskEdited(d.index);
};
tl.addEventListener("pointerup", endStroke);
tl.addEventListener("pointercancel", endStroke);
tl.addEventListener("pointerleave", () => { if (!paintDrag) $("#brushCur").hidden = true; });
document.addEventListener("keydown", e => {
  if ($(".page.on")?.dataset.page !== "5") return;                              // post-produzione
  if (e.key === "Escape" && spPick !== null) { spPick = null; $("#pickInfo").textContent = ""; renderSteps(); }
  if ((e.ctrlKey || e.metaKey) && (e.key === "z" || e.key === "Z") && brushActive() && !e.target.closest?.("input, select, textarea")) {
    const s = openMaskStep();
    if (isPaint(s.mask) && (s.mask.strokes || []).length) { e.preventDefault(); s.mask.strokes.pop(); unwrapPaint(s); maskEdited(openStep); }
  }
});

let refreshTimer = null, refreshing = false, pending = null;
const refreshSoon = () => { clearTimeout(refreshTimer); refreshTimer = setTimeout(() => refresh(), 350); };
async function refresh(upto) {
  if (refreshing) { pending = upto ?? null; return; }
  if (zoomAbort) { zoomAbort.abort(); zoomAbort = null; zoomToken++; }   // an old zoom render must not cover the new preview
  refreshing = true; $("#spin").classList.add("on");
  try {
    const t = performance.now();
    const r = await apiResp("/api/preview", { input: state.input, steps: state.steps, upto: upto ?? null, view: $("#viewSel").value });
    if (+r.headers.get("X-Scale")) viewScale = +r.headers.get("X-Scale");
    const blob = await r.blob();
    showBlob($("#postImg"), blob);
    hideZoomRender();
    $("#elapsed").textContent = `${((performance.now() - t) / 1000).toFixed(1)} s` + (upto ? ` — fino al passo ${upto}` : "");
  } catch (e) { $("#elapsed").textContent = "errore: " + e.message.slice(0, 120); }
  finally {
    refreshing = false; $("#spin").classList.remove("on");
    if (pending !== null) { const p = pending; pending = null; refresh(p ?? undefined); }
    else updateDbe();
  }
}

// ---------------------------------------------------------------- zoom / pan of the preview + full-quality render
// the layer with the preview and its overlays (mask, DBE samples, brush) is scaled with a CSS transform: clicks
// keep working because every handler maps through getBoundingClientRect, which includes the transform
const zoom = { z: 1, tx: 0, ty: 0, pan: null };
// "Adatta": the whole image inside the viewer, in width AND height (the viewer has the screen's height: a wide
// screen or a tall / drizzled image went past its bottom and needed the scrollbar). The box takes the fitted
// width, so the zoom / pan code, which measures #postWrap, keeps working unchanged.
function fitView() {
  const img = $("#postImg"), wrap = $("#postWrap"), viewer = wrap.closest(".viewer");
  if (!img.naturalWidth || !viewer) return;
  wrap.style.width = ""; wrap.style.marginLeft = wrap.style.marginRight = "";
  if (window.matchMedia("(max-width: 900px)").matches) return;      // phone layout: the page scrolls
  const full = wrap.clientWidth, cs = getComputedStyle(wrap);
  const rows = [...viewer.children].filter(c => c !== wrap && c.offsetParent !== null && getComputedStyle(c).position !== "absolute")
    .reduce((a, c) => a + c.offsetHeight, 0);
  // the viewer is sticky and as tall as its content: the room is the window below its sticky top
  const vs = getComputedStyle(viewer), top = isNaN(parseFloat(vs.top)) ? viewer.getBoundingClientRect().top : parseFloat(vs.top);
  const avail = innerHeight - top - rows - parseFloat(cs.marginTop) - parseFloat(cs.marginBottom)
    - parseFloat(vs.paddingTop) - parseFloat(vs.paddingBottom) - 2;
  if (avail < 120) return;
  const w = Math.floor(Math.min(full, avail * img.naturalWidth / img.naturalHeight));
  if (w < full - 1) { wrap.style.width = `${w}px`; wrap.style.marginLeft = wrap.style.marginRight = "auto"; }
}
function applyZoom() {
  const wrap = $("#postWrap"), W = wrap.clientWidth, H = $("#zoomLayer").offsetHeight;
  zoom.z = Math.min(64, Math.max(1, zoom.z));
  zoom.tx = Math.min(0, Math.max(W - W * zoom.z, zoom.tx));
  zoom.ty = Math.min(0, Math.max(H - H * zoom.z, zoom.ty));
  $("#zoomLayer").style.transform = `translate(${zoom.tx}px, ${zoom.ty}px) scale(${zoom.z})`;
  wrap.classList.toggle("zoomed", zoom.z > 1);
  $("#zoomVal").textContent = `${Math.round(zoom.z * 100)}%`;
  hideZoomRender();
}
function zoomAt(factor, cx, cy) {
  const z0 = zoom.z, z1 = Math.min(64, Math.max(1, z0 * factor));
  zoom.tx = cx - (cx - zoom.tx) * z1 / z0; zoom.ty = cy - (cy - zoom.ty) * z1 / z0; zoom.z = z1;
  applyZoom();
}
const wrapCenter = () => { const w = $("#postWrap"); return [w.clientWidth / 2, $("#zoomLayer").offsetHeight / 2]; };
$("#postWrap").addEventListener("wheel", e => {
  if (!$("#postImg").naturalWidth) return;
  e.preventDefault();
  const r = $("#postWrap").getBoundingClientRect();
  zoomAt(e.deltaY < 0 ? 1.25 : 0.8, e.clientX - r.left, e.clientY - r.top);
}, { passive: false });
// pan: left drag when zoomed and no tool (brush / SP pick / DBE samples) is active; middle button or Alt+drag always
$("#postWrap").addEventListener("pointerdown", e => {
  const tool = !$("#toolLayer").hidden || !$("#dbeSvg").hidden;
  if (!(e.button === 1 || e.altKey || (e.button === 0 && zoom.z > 1 && !tool))) return;
  e.preventDefault();
  zoom.pan = { x: e.clientX, y: e.clientY, tx: zoom.tx, ty: zoom.ty, id: e.pointerId };
  $("#postWrap").setPointerCapture(e.pointerId); $("#postWrap").classList.add("panning");
}, true);
$("#postWrap").addEventListener("pointermove", e => {
  if (!zoom.pan) return;
  zoom.tx = zoom.pan.tx + e.clientX - zoom.pan.x; zoom.ty = zoom.pan.ty + e.clientY - zoom.pan.y; applyZoom();
});
const endPan = () => { zoom.pan = null; $("#postWrap").classList.remove("panning"); };
$("#postWrap").addEventListener("pointerup", endPan);
$("#postWrap").addEventListener("pointercancel", endPan);
$("#zoomIn").onclick = () => zoomAt(1.5, ...wrapCenter());
$("#zoomOut").onclick = () => zoomAt(1 / 1.5, ...wrapCenter());
$("#zoomFit").onclick = () => { zoom.z = 1; zoom.tx = zoom.ty = 0; fitView(); applyZoom(); };
$("#postImg").addEventListener("load", () => { fitView(); applyZoom(); });
// 1:1 = one screen pixel per full-resolution image pixel (viewScale = preview width / full width)
$("#zoom11").onclick = () => {
  const img = $("#postImg"); if (!img.naturalWidth || !viewScale) return;
  const fullW = img.naturalWidth / viewScale, shown = $("#postWrap").clientWidth * (window.devicePixelRatio || 1);
  const want = fullW / shown;
  zoomAt(Math.max(1, want) / zoom.z, ...wrapCenter());
};
function zoomArea() {
  const W = $("#postWrap").clientWidth, H = $("#zoomLayer").offsetHeight;
  const x0 = -zoom.tx / (W * zoom.z), y0 = -zoom.ty / (H * zoom.z);
  return [x0, y0, x0 + 1 / zoom.z, y0 + 1 / zoom.z].map(v => Math.min(1, Math.max(0, v)));
}
function hideZoomRender() { const z = $("#zoomImg"); if (!z.hidden) { z.hidden = true; $("#zoomInfo").textContent = ""; } }
let zoomToken = 0, zoomAbort = null;
$("#zoomRender").onclick = async () => {
  if (!state.input) return;
  if (zoomAbort) {                    // a second click while rendering cancels it
    zoomAbort.abort(); zoomAbort = null; zoomToken++;
    $("#zoomInfo").textContent = "render annullato"; $("#spin").classList.remove("on"); return;
  }
  const token = ++zoomToken, crop = zoomArea(), ext = $("#zoomExt").checked;
  const width = Math.round($("#postWrap").clientWidth * (window.devicePixelRatio || 1));
  const t = performance.now(), what = ext ? "render con i programmi esterni" : "render";
  const tick = () => { if (token === zoomToken) $("#zoomInfo").textContent =
    `${what}… ${Math.round((performance.now() - t) / 1000)} s (clic su Renderizza zoom per annullare)`; };
  tick(); const timer = setInterval(tick, 1000);
  zoomAbort = new AbortController();
  $("#spin").classList.add("on");
  try {
    const r = await fetch("/api/preview_zoom", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ input: state.input, steps: state.steps, crop, width, external: ext }), signal: zoomAbort.signal });
    if (!r.ok) { let m = await r.text(); try { m = JSON.parse(m).detail || m; } catch (e) {} throw new Error(`${r.status} ${m}`); }
    const blob = await r.blob();
    if (token !== zoomToken) return;
    const z = $("#zoomImg"); showBlob(z, blob); z.hidden = false;
    const sc = +r.headers.get("X-Scale");
    $("#zoomInfo").textContent = `render ${r.headers.get("X-Size")} px a ${sc >= 0.999 ? "piena risoluzione" : Math.round(sc * 100) + "%"}`
      + ` in ${((performance.now() - t) / 1000).toFixed(1)} s` + (ext ? " (con programmi esterni)" : "") + " — muovi lo zoom per tornare all'anteprima";
  } catch (e) {
    if (token === zoomToken && e.name !== "AbortError") $("#zoomInfo").textContent = "errore nel render: " + e.message.slice(0, 160);
  } finally {
    clearInterval(timer);
    if (token === zoomToken) { $("#spin").classList.remove("on"); zoomAbort = null; }
  }
};
window.addEventListener("resize", () => { fitView(); applyZoom(); });

// ---------------------------------------------------------------- mask overlay (red = protected, like PixInsight)
let maskTimer = null, maskToken = 0;
const updateMaskSoon = () => { clearTimeout(maskTimer); maskTimer = setTimeout(updateMask, 300); };
async function updateMask() {
  const cv = $("#maskCanvas");
  const s = openStep !== null && openStep !== undefined ? state.steps[openStep] : null;
  const token = ++maskToken;
  if (!$("#showMask").checked || !s || !s.mask || !state.input) {
    cv.hidden = true;
    $("#maskInfo").textContent = $("#showMask").checked ? "apri un passo con una maschera" : "";
    return;
  }
  $("#maskInfo").textContent = "calcolo della maschera…";
  try {
    const r = await apiResp("/api/mask", { input: state.input, steps: state.steps, index: openStep });
    if (token !== maskToken) return;
    if (+r.headers.get("X-Scale")) viewScale = +r.headers.get("X-Scale");
    const blob = await r.blob();
    if (token !== maskToken || paintDrag) return;
    const bmp = await createImageBitmap(blob);
    cv.width = bmp.width; cv.height = bmp.height;
    const g = cv.getContext("2d");
    g.drawImage(bmp, 0, 0);
    const d = g.getImageData(0, 0, cv.width, cv.height), px = d.data;
    let sum = 0;
    for (let k = 0; k < px.length; k += 4) {
      const v = px[k]; sum += v;
      px[k] = 255; px[k + 1] = 30; px[k + 2] = 40; px[k + 3] = Math.round((255 - v) * 0.62);
    }
    g.putImageData(d, 0, 0);
    cv.hidden = false;
    $("#maskInfo").textContent = `maschera del passo ${openStep + 1}: ${Math.round(sum / (px.length / 4) / 2.55)}% elaborato · rosso = protetto`;
  } catch (e) {
    if (token === maskToken) { cv.hidden = true; $("#maskInfo").textContent = "maschera non disponibile: " + e.message.slice(0, 100); }
  }
}
$("#showMask").onchange = updateMask;
document.addEventListener("keydown", e => {
  if ($(".page.on")?.dataset.page !== "5") return;                              // post-produzione
  if ((e.ctrlKey || e.metaKey) && (e.key === "k" || e.key === "K")) {
    e.preventDefault();
    $("#showMask").checked = !$("#showMask").checked; updateMask();
  }
});

// ---------------------------------------------------------------- before / after split
function updateSplit() {
  const on = $("#before").checked, v = +$("#split").value;
  $("#split").hidden = !on; $("#beforeImg").hidden = !on; $("#splitLine").hidden = !on;
  if (!on) return;
  const src = imgUrl(state.input, 1200);
  if ($("#beforeImg").dataset.src !== src) { $("#beforeImg").src = src; $("#beforeImg").dataset.src = src; }
  // left of the line: input ("prima"), right: processed preview ("dopo")
  $("#beforeImg").style.clipPath = `inset(0 ${100 - v}% 0 0)`;
  $("#splitLine").style.left = `${v}%`;
}
$("#before").onchange = updateSplit;
$("#viewSel").onchange = () => { autoView = false; refresh(); };      // chosen by hand: kept until changed again
$("#split").oninput = updateSplit;

// ---------------------------------------------------------------- DBE samples on the preview
// When the open step is a background.dbe, its samples are drawn on the preview: click = add, click on a
// sample = remove. The step's `samples` param holds full-resolution pixels; [] = automatic.
let dbe = null, dbeToken = 0;
async function updateDbe() {
  const s = openStep !== null && openStep !== undefined ? state.steps[openStep] : null;
  if (!s || s.plugin !== "background.dbe" || !state.input) {
    dbe = null; dbeToken++; $("#dbeSvg").toggleAttribute("hidden", true); $("#dbeBar").hidden = true; return;
  }
  const token = ++dbeToken, index = openStep;
  $("#dbeBar").hidden = false;
  if (!dbe) $("#dbeInfo").textContent = "calcolo dei campioni…";
  try {
    const r = await api("/api/dbe_samples", { input: state.input, steps: state.steps, index });
    if (token !== dbeToken) return;
    dbe = { index, ...r }; drawDbe();
  } catch (e) {
    if (token === dbeToken) { $("#dbeInfo").textContent = "campioni non disponibili: " + e.message.slice(0, 100); $("#dbeSvg").toggleAttribute("hidden", true); }
  }
}
function drawDbe() {
  const svg = $("#dbeSvg"), b = Math.max(3, dbe.box);
  svg.setAttribute("viewBox", `0 0 ${dbe.width} ${dbe.height}`);
  svg.innerHTML = dbe.samples.map(([x, y], k) =>
    `<rect data-k="${k}" x="${(x - b / 2).toFixed(1)}" y="${(y - b / 2).toFixed(1)}" width="${b}" height="${b}"><title>campione ${k + 1}: ${Math.round(dbe.full[k][0])}, ${Math.round(dbe.full[k][1])}</title></rect>`).join("");
  svg.toggleAttribute("hidden", false);   // SVG elements have no .hidden property
  $("#dbeInfo").textContent = `${dbe.samples.length} campioni ${dbe.auto ? "automatici" : "manuali"} · clic = aggiungi, clic su un campione = togli`;
}
$("#dbeSvg").onclick = e => {
  if (!dbe) return;
  const step = state.steps[dbe.index];
  if (!step || step.plugin !== "background.dbe") return;
  let full = dbe.full.map(p => [p[0], p[1]]);      // the auto samples become manual on the first edit
  const hit = e.target.closest("rect[data-k]");
  if (hit) full.splice(+hit.dataset.k, 1);
  else {
    const r = $("#dbeSvg").getBoundingClientRect();
    const x = (e.clientX - r.left) / r.width * dbe.width, y = (e.clientY - r.top) / r.height * dbe.height;
    if (!Number.isFinite(x) || !Number.isFinite(y) || x < 0 || y < 0 || x > dbe.width || y > dbe.height) return;
    full.push([x / dbe.scale, y / dbe.scale]);
  }
  if (full.length < 3) { alert("Servono almeno 3 campioni."); return; }
  full = full.map(([x, y]) => [Math.round(x), Math.round(y)]);
  step.params.samples = full; save();
  dbe = { ...dbe, full, auto: false, samples: full.map(([x, y]) => [x * dbe.scale, y * dbe.scale]) };
  drawDbe(); renderSteps(); refresh();
};
$("#dbeAuto").onclick = () => {
  if (!dbe) return;
  const step = state.steps[dbe.index];
  if (step) { step.params.samples = []; save(); renderSteps(); refresh(); }
};

// ---------------------------------------------------------------- 7: export
// metadata written into the exported FITS / XISF header (kept in the browser between sessions)
const meta = JSON.parse(localStorage.getItem("af_meta") || "null") || {};
$$(".meta").forEach(el => {
  el.value = meta[el.dataset.key] || "";
  el.oninput = () => { meta[el.dataset.key] = el.value; try { localStorage.setItem("af_meta", JSON.stringify(meta)); } catch (e) {} };
});
const metadata = () => {
  const v = id => $(id).value.trim(), out = {};
  if (v("#metaAuthor")) out.OBSERVER = out.AUTHOR = v("#metaAuthor");
  if (v("#metaCopy")) out.COPYRIGHT = v("#metaCopy");
  if (v("#metaTel")) out.TELESCOP = v("#metaTel");
  if (v("#metaFilter")) out.FILTER = v("#metaFilter");
  if (v("#metaNotes")) out.COMMENT = v("#metaNotes");
  return out;
};
let headerCards = null, headerFor = null;
async function loadHeader() {
  if (headerFor !== state.input) { headerCards = (await api(`/api/fits_header?path=${encodeURIComponent(state.input)}`)).cards; headerFor = state.input; }
  return headerCards;
}
function enterExport() {
  if (!state.input) { go(1); return; }
  if (!$("#outDir").value) $("#outDir").value = state.input.replace(/\\[^\\]+$/, "");
  // empty Telescope / Filter fields keep the input file's values: show them as placeholders
  loadHeader().then(cards => {
    const get = k => cards.find(c => c[0] === k)?.[1];
    $("#metaTel").placeholder = get("TELESCOP") ? `dal file: ${get("TELESCOP")}` : "TELESCOP";
    $("#metaFilter").placeholder = get("FILTER") ? `dal file: ${get("FILTER")}` : "FILTER";
  }).catch(() => {});
}
$("#showHeader").onclick = async () => {
  if (!state.input) return;
  try {
    const cards = await loadHeader();
    $("#hdrPath").textContent = state.input;
    $("#hdrTable").innerHTML = cards.map(([k, v, c]) => k.startsWith("---") ? `<tr class="hdu"><td colspan="3">${esc(k)} ${esc(c)}</td></tr>`
      : `<tr><td>${esc(k)}</td><td>${esc(typeof v === "string" ? v : JSON.stringify(v))}</td><td>${esc(c)}</td></tr>`).join("")
      || '<tr><td class="muted">nessuna parola chiave</td></tr>';
    $("#hdrModal").hidden = false;
  } catch (e) { alert(e.message); }
};
$("#hdrClose").onclick = () => $("#hdrModal").hidden = true;
const exportBody = () => ({ input: state.input, steps: state.steps, name: $("#outName").value,
  out_dir: $("#outDir").value, formats: $$(".fmt").filter(c => c.checked).map(c => c.value), metadata: metadata() });
const showExport = res => {
  $("#finalImg").src = imgUrl(res.preview) + `&t=${Date.now()}`;
  $("#files").innerHTML = "";
  res.files.forEach(f => { const li = document.createElement("li"); li.textContent = f; $("#files").appendChild(li); });
};
$("#export").onclick = async () => {
  try {
    const { job } = await api("/api/post", exportBody());
    pollQueue();
    follow(job, $("#log2"), $("#bar2"), showExport);
  } catch (e) { $("#log2").textContent = e.message; }
};
$("#queueExport").onclick = e => enqueue(e.currentTarget, "/api/post", exportBody());

// ---------------------------------------------------------------- job queue panel
const STATUS_IT = { queued: "in coda", running: "in corso", done: "fatto", error: "errore", cancelled: "annullato" };
let queueTimer = null;
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const hhmm = t => t ? new Date(t * 1000).toLocaleTimeString("it-IT", { hour: "2-digit", minute: "2-digit" }) : "";
async function pollQueue() {
  clearTimeout(queueTimer);
  let q = null;
  try { q = await api("/api/jobs"); } catch (e) {}
  if (q) renderQueue(q);
  // poll fast while something is queued or running, slowly otherwise
  queueTimer = setTimeout(pollQueue, q && q.active.length ? 1500 : 6000);
}
function renderQueue(q) {
  $("#qList").innerHTML = q.active.length ? q.active.map(j => `<li title="${esc(j.label)}">
      <div class="qrow"><span class="qlabel">${esc(j.label)}</span><button data-cancel="${j.id}" title="Annulla">${ico("x")}</button></div>
      <div class="qrow"><span class="st-${j.status}">${STATUS_IT[j.status] || j.status}${j.status === "queued" ? ` #${j.position}` : ` · ${esc(j.stage)}`}</span>
        <span>${j.status === "running" ? Math.round(j.progress * 100) + "%" : ""}</span></div>
      ${j.status === "running" ? `<div class="bar"><div style="width:${Math.round(j.progress * 100)}%"></div></div>` : ""}</li>`).join("")
    : '<li class="muted">nessun lavoro</li>';
  $("#qHist").innerHTML = q.history.length ? q.history.map(j => `<li title="${esc(j.error || j.label)}">
      <div class="qrow"><span class="qlabel">${esc(j.label)}</span>${j.status === "done" ? `<button data-open="${j.id}" title="Apri il risultato">Apri</button>` : ""}</div>
      <div class="qrow"><span class="st-${j.status}">${STATUS_IT[j.status] || j.status}</span><span>${hhmm(j.finished)}</span></div></li>`).join("")
    : '<li class="muted">vuoto</li>';
}
$("#queue").onclick = async e => {
  const c = e.target.closest("[data-cancel]"), o = e.target.closest("[data-open]");
  if (c) { await api(`/api/jobs/${c.dataset.cancel}/cancel`, {}).catch(() => {}); pollQueue(); }
  if (o) {
    try {
      const j = await api(`/api/jobs/${o.dataset.open}`);
      if (j.kind === "stack") openStackResult(j.result);
      else if (j.kind === "post") { go(6); showExport(j.result); }
      else if (j.kind === "masters") { $("#mkModal").hidden = false; showMasters(j.result); }
    } catch (err) { alert(err.message); }
  }
};

// ---------------------------------------------------------------- reset
// the session (folders, open image, steps) is kept in the browser and survives a server restart: these start over
$("#newProject").onclick = async () => {
  if (!confirm("Ricominciare da zero? Si svuotano cartelle, immagine aperta, passi e anteprime (i file su disco restano).")) return;
  await api("/api/reset", {}).catch(() => {});
  try { localStorage.removeItem("af_state"); } catch (e) {}
  location.reload();
};
// ---------------------------------------------------------------- server: restart / stop
// a full-page note while the server is away; the session lives in the browser, so the page comes back as it was
function serverNote(html) {
  let d = $("#srvNote");
  if (!d) { d = document.createElement("div"); d.id = "srvNote"; d.className = "modal srvnote"; document.body.appendChild(d); }
  d.innerHTML = `<div class="modal-box">${html}</div>`; d.hidden = false;
}
async function jobsRunning() {
  try { return (await api("/api/jobs")).active.length; } catch (e) { return 0; }
}
$("#srvRestart").onclick = async () => {
  const n = await jobsRunning();
  if (n && !confirm(`${n} lavori in coda o in corso verranno interrotti. Riavviare il server?`)) return;
  let old;
  try { old = (await api("/api/server/restart", {})).pid; } catch (e) { alert(e.message); return; }
  serverNote(`<h2>${ico("refresh-cw")} Riavvio del server</h2><p class="muted">Controllo degli aggiornamenti e
    nuovo avvio: la pagina si ricarica da sola appena il server risponde.</p><div class="bar"><div class="srvwait"></div></div>`);
  const t0 = Date.now();
  const poll = async () => {
    try {
      const v = await (await fetch("/api/version", { cache: "no-store" })).json();
      if (v.pid && v.pid !== old) { location.reload(); return; }
    } catch (e) {}
    if (Date.now() - t0 > 120000) {
      serverNote(`<h2>Il server non è ripartito</h2><p class="muted">Avvialo con start-gui.cmd (o l'icona sul desktop)
        e ricarica questa pagina.</p><div class="actions"><button class="btn primary" onclick="location.reload()">Ricarica</button></div>`);
      return;
    }
    setTimeout(poll, 1000);
  };
  setTimeout(poll, 1500);
};
$("#srvStop").onclick = async () => {
  const n = await jobsRunning();
  if (!confirm(n ? `${n} lavori in coda o in corso verranno interrotti. Spegnere il server?` : "Spegnere il server di Noctua?")) return;
  try { await api("/api/server/stop", {}); } catch (e) { alert(e.message); return; }
  serverNote(`<h2>${ico("power")} Server spento</h2><p class="muted">Per riaprire Noctua lancia start-gui.cmd
    (o l'icona sul desktop), poi ricarica questa pagina: cartelle, immagine e passi restano.</p>
    <div class="actions"><button class="btn primary" onclick="location.reload()">Ricarica</button></div>`);
};
$("#closeImage").onclick = async () => {
  state.input = null; state.steps = []; state.autoReason = ""; stackResult = null; save();
  await api("/api/reset", {}).catch(() => {});
  $("#postImg").removeAttribute("src");
  go(1);
};

// ---------------------------------------------------------------- "Crea i master"
setTimeout(() => mkLoadDefaults().catch(() => {}), 0);
let mkJob = null, mkOut = "", mkReport = null;
// per-type settings: state.masterTypes = {bias: {...}, ...} (full values), defaults from the server
let mkDefaults = null, mkTab = "bias";
const MK_REJ = [["auto", "automatico (dal numero di frame)"], ["winsorized", "Winsorized sigma clipping"], ["linear_fit", "Linear fit clipping"],
  ["esd", "Generalized ESD"], ["sigma", "Sigma clipping"], ["percentile", "Percentile clipping"], ["none", "nessuno"]];
// [key, label, kind, extra] — kind: num [min, max, step] | sel [[v, label]] | chk
const MK_FIELDS = {
  _all: [["combination", "Combinazione", "sel", [["average", "media"], ["median", "mediana"]]],
    ["rejection", "Rigetto dei pixel", "sel", MK_REJ],
    ["low", "Sigma basso", "num", [0.5, 10, 0.25]], ["high", "Sigma alto", "num", [0.5, 10, 0.25]],
    ["iterations", "Iterazioni", "num", [1, 20, 1]],
    ["esd_outliers", "ESD: frazione massima di outlier", "num", [0.05, 0.5, 0.05]],
    ["esd_alpha", "ESD: significatività", "num", [0.01, 0.2, 0.01]],
    ["min_frames", "Minimo frame per creare il master", "num", [1, 200, 1]]],
  bias: [["superbias", "Superbias (bias senza rumore: colonne, righe e grande scala)", "chk"],
    ["superbias_scale", "Superbias: scala della grande struttura (px)", "num", [4, 256, 4]]],
  flat: [["calibrate_with", "Calibrazione dei flat", "sel", [["auto", "dark flat se c'è, altrimenti bias"], ["darkflat", "dark flat"], ["bias", "solo bias"], ["none", "nessuna"]]],
    ["normalize", "Normalizzazione", "sel", [["auto", "automatica (per sito Bayer sulle camere a colori)"], ["cfa", "per sito Bayer (colore)"], ["global", "unica (mono)"]]],
    ["per_night", "Un master flat per notte (la polvere si sposta)", "chk"],
    ["drop_odd_exposures", "Scarta i flat con esposizione diversa dal gruppo", "chk"]],
};
const MK_HINT = { bias: "PixInsight consiglia: media, Winsorized sigma clipping 3/3 con 20+ frame; Superbias con pochi frame.",
  dark: "Media senza normalizzazione; il rigetto toglie raggi cosmici e satelliti. Con 15+ frame Winsorized o Linear fit.",
  darkflat: "Come i dark, con l'esposizione dei flat. Servono se i flat sono lunghi (oltre ~1 s) o la camera ha amp glow.",
  flat: "Media con rigetto (Percentile con pochi frame, Winsorized con molti); i flat si calibrano con dark flat o bias e si normalizzano." };
async function mkLoadDefaults() {
  if (!mkDefaults) mkDefaults = await api("/api/masters/defaults");
  if (!state.masterTypes) state.masterTypes = JSON.parse(JSON.stringify(mkDefaults.types));
  for (const k of Object.keys(mkDefaults.types)) state.masterTypes[k] = { ...mkDefaults.types[k], ...(state.masterTypes[k] || {}) };
}
function mkRender() {
  $$("#mkTabs button").forEach(b => b.classList.toggle("on", b.dataset.k === mkTab));
  const t = state.masterTypes[mkTab], def = mkDefaults.types[mkTab], box = $("#mkType");
  const rows = MK_FIELDS._all.concat(MK_FIELDS[mkTab] || []).filter(([k]) => {
    if (k.startsWith("esd_")) return t.rejection === "esd" && t.combination !== "median";
    if (["low", "high", "iterations"].includes(k)) return t.combination !== "median" && !["none", "percentile", "esd"].includes(t.rejection);
    if (k === "rejection") return t.combination !== "median";
    if (k === "superbias_scale") return !!t.superbias;
    return true;
  });
  box.innerHTML = `<p class="muted small">${MK_HINT[mkTab]}</p>` + rows.map(([k, label, kind, x]) => {
    const changed = JSON.stringify(t[k]) !== JSON.stringify(def[k]) ? ' class="changed"' : "";
    if (kind === "chk") return `<label class="row"${changed}><input type="checkbox" data-k="${k}" ${t[k] ? "checked" : ""}> ${label}</label>`;
    if (kind === "sel") return `<label class="row"${changed}>${label} <select data-k="${k}">${x.map(([v, n]) =>
      `<option value="${v}" ${v === t[k] ? "selected" : ""}>${n}</option>`).join("")}</select></label>`;
    return `<label class="row"${changed}>${label} <input type="number" data-k="${k}" value="${t[k]}" min="${x[0]}" max="${x[1]}" step="${x[2]}"></label>`;
  }).join("");
  $$("[data-k]", box).forEach(el => el.onchange = () => {
    const k = el.dataset.k;
    t[k] = el.type === "checkbox" ? el.checked : el.tagName === "SELECT" ? el.value : +el.value;
    save(); mkRender();
  });
}
$$("#mkTabs button").forEach(b => b.onclick = () => { mkTab = b.dataset.k; mkRender(); });
$("#mkReset").onclick = () => { state.masterTypes = JSON.parse(JSON.stringify(mkDefaults.types)); save(); mkRender(); };
$("#mk_stack").onchange = e => { state.masterForStack = e.target.checked; save(); };
$("#openMasters").onclick = async () => {
  await mkLoadDefaults();
  if (!$("#mk_out").value && state.calibration.length) $("#mk_out").value = state.calibration[0].replace(/[\\/]+$/, "") + "\\AstroForge masters";
  $("#mk_stack").checked = !!state.masterForStack;
  $("#mkStatus").textContent = state.calibration.length ? `Cartelle: ${state.calibration.join(" · ")}`
    : "Nessun file grezzo: puoi impostare i parametri (anche per lo stacking), per creare i master aggiungi «+ File per creare i master».";
  $("#mkResult").innerHTML = ""; $("#mkUse").hidden = true; $("#mkStart").disabled = !state.calibration.length;
  mkRender();
  $("#mkModal").hidden = false;
};
// the settings that differ from the recommended ones (what the stacking and the masters job need)
function masterOptions() {
  if (!state.masterTypes || !mkDefaults) return {};
  const out = {};
  for (const [k, t] of Object.entries(state.masterTypes)) {
    const d = Object.fromEntries(Object.entries(t).filter(([n, v]) => JSON.stringify(v) !== JSON.stringify(mkDefaults.types[k][n])));
    if (Object.keys(d).length) out[k] = d;
  }
  return out;
}
$("#mkClose").onclick = () => $("#mkModal").hidden = true;
$("#mkStart").onclick = async () => {
  const options = { ...masterOptions(), defect_map: $("#mk_defects").checked, hot_sigma: +$("#mk_hot").value,
    cold_frac: +$("#mk_cold").value, columns_sigma: +$("#mk_cols").value, format: $("#mk_fmt").value, library: $("#mk_lib").checked };
  try {
    const r = await api("/api/masters", { folders: state.calibration, out_dir: $("#mk_out").value.trim(), options });
    mkJob = r.job; mkOut = r.out_dir; $("#mkStart").disabled = true;
    $("#mkStatus").textContent = "in coda…"; pollQueue(); watchMasters();
  } catch (e) { alert(e.message); }
};
async function watchMasters() {
  if (!mkJob) return;
  try {
    const j = await api(`/api/jobs/${mkJob}`);
    const last = j.log.slice(-1)[0] || "";
    $("#mkStatus").textContent = j.status === "running" ? `${Math.round(j.progress * 100)}% · ${last.replace(/^\[[^\]]*\]\s*/, "")}`
      : j.status === "queued" ? `in coda (#${j.position})` : (STATUS_IT[j.status] || j.status);
    if (j.status === "done") { showMasters(j.result); mkJob = null; return; }
    if (j.status === "error" || j.status === "cancelled") { $("#mkStatus").textContent += " — " + (j.error || ""); $("#mkStart").disabled = false; mkJob = null; return; }
  } catch (e) { /* keep trying */ }
  setTimeout(watchMasters, 1200);
}
function showMasters(rep) {
  mkReport = rep; mkOut = rep.out_dir;
  const names = { bias: "Bias", dark: "Dark", flat: "Flat", darkflat: "Dark flat" };
  const rows = rep.masters.map(e => {
    const st = e.stats || {};
    const extra = [e.filter, e.night, e.kind === "flat" && st.vignetting_pct != null ? `vignettatura ${st.vignetting_pct.toFixed(0)}%` : "",
      st.hot_pixels != null ? `${st.hot_pixels} pixel caldi` : ""].filter(Boolean).join(" · ");
    return `<tr><td>${names[e.kind] || e.kind}</td><td>${esc(e.file.split(/[\\/]/).pop())}</td><td>${e.frames}</td>
      <td>${e.kind === "flat" ? (100 * (st.noise || 0) / 65535).toFixed(2) + "%" : (st.noise || 0).toFixed(1)}</td><td>${esc(extra)}</td><td class="${e.flags.length ? "warn" : ""}">${esc(e.flags.join("; ") || "ok")}</td></tr>`;
  }).join("");
  const skipped = rep.skipped.map(x => `<li>${names[x.kind] || x.kind} ${esc(x.group)}: solo ${x.frames} frame</li>`).join("");
  const maps = rep.defect_maps.map(d => `<li>${esc(d.file.split(/[\\/]/).pop())}: ${d.pixels} pixel, ${d.columns} colonne intere</li>`).join("");
  $("#mkResult").innerHTML = `<table class="mktable"><tr><th>Tipo</th><th>File</th><th>Frame</th><th>Rumore (ADU, flat in %)</th><th></th><th>Controllo</th></tr>${rows}</table>`
    + (maps ? `<p><b>Mappa dei difetti</b></p><ul>${maps}</ul>` : "") + (skipped ? `<p class="muted">Saltati</p><ul class="muted">${skipped}</ul>` : "")
    + `<p class="muted small">Cartella: ${esc(rep.out_dir)} · rapporto in masters_report.txt · ${rep.seconds} s</p>`;
  $("#mkStatus").textContent = `${rep.masters.length} master creati`;
  $("#mkUse").hidden = !rep.masters.length; $("#mkStart").disabled = false;
}
$("#mkUse").onclick = () => {
  if (!mkOut) return;
  if (!state.masterDirs.includes(mkOut)) state.masterDirs.push(mkOut);
  state.calibration = [];
  state.defectMap = mkReport && mkReport.defect_maps.length ? mkReport.defect_maps[0].file : null;
  save(); renderPaths(); $("#mkModal").hidden = true;
};

// ---------------------------------------------------------------- system + external tools
api("/api/system").then(s => { $("#compute").textContent = `${s.compute} · ${s.cpus} thread CPU`; }).catch(() => {});
// version in use; after a git pull while the GUI kept running, a notice to restart it (new plugins are missing)
async function checkVersion() {
  try {
    const v = await api("/api/version");
    $("#version").textContent = `Noctua ${v.version}`;
    document.title = `Noctua ${v.version.split(" ")[0]}`;
    const n = $("#updateNotice");
    n.hidden = !v.restart_needed;
    if (v.restart_needed) n.textContent = `Codice aggiornato su disco (${v.disk}): premi «Riavvia server» in basso a sinistra per usarlo.`;
  } catch (e) { /* server restarting */ }
}
checkVersion();
setInterval(checkVersion, 60000);

// ---------------------------------------------------------------- live load bar (CPU, RAM, every GPU)
function meter(label, pct, detail, title) {
  const p = Math.max(0, Math.min(100, pct));
  const lvl = p >= 90 ? "hi" : p >= 65 ? "mid" : "lo";
  return `<div class="meter ${lvl}" title="${title || ""}"><span class="ml">${label}</span>` +
         `<span class="mb"><i style="width:${p.toFixed(0)}%"></i></span>` +
         `<span class="mv">${p.toFixed(0)}%</span>${detail ? `<span class="md">${detail}</span>` : ""}</div>`;
}
async function refreshLoad() {
  if (document.hidden) return;
  try {
    const s = await api("/api/system/load");
    const parts = [];
    if (s.cpu !== undefined) parts.push(meter("CPU", s.cpu));
    if (s.ram !== undefined) parts.push(meter("RAM", s.ram, `${s.ram_used_gb.toFixed(1)}/${s.ram_total_gb.toFixed(0)} GB`));
    for (const g of s.gpus) {
      const short = g.name.replace(/^NVIDIA\s+/i, "").replace(/GeForce\s+/i, "");
      parts.push(meter(`GPU${g.index}`, g.util,
        `${g.mem_used_gb.toFixed(1)}/${g.mem_total_gb.toFixed(0)} GB${g.temp != null ? ` · ${g.temp}°C` : ""}`,
        `${short}: carico ${g.util}%, memoria ${g.mem_used_gb.toFixed(1)} di ${g.mem_total_gb.toFixed(1)} GB`));
    }
    $("#loadbar").innerHTML = parts.join("");
  } catch (e) { /* server busy or restarting: keep the last values */ }
}
refreshLoad();
setInterval(refreshLoad, 1500);
const TOOL_NAMES = { starnet: "StarNet (starnet2.exe)", deepsnr: "DeepSNR (deepsnr.exe)", graxpert: "GraXpert", cosmic_clarity: "Cosmic Clarity (sharpen)",
  cosmic_clarity_denoise: "Cosmic Clarity (denoise)", astap: "ASTAP (plate solving)", "rc-astro": "RC-Astro CLI (rc-astro.exe)" };
async function renderTools(cfg) {
  cfg = cfg || await api("/api/config");
  $("#toolList").innerHTML = Object.entries(cfg.tools).map(([k, t]) =>
    `<label class="row">${TOOL_NAMES[k] || k} <span class="${t.found ? "" : "muted"}">${t.found ? ico("check") : "non trovato"}</span>
     <input type="text" class="wide" data-tool="${k}" value="${t.path.replace(/"/g, "&quot;")}" placeholder="C:\\...\\programma.exe"></label>`).join("");
}
// StarNet2 / DeepSNR versions: the installed one and the latest on starnetastro.com (checked once a day)
async function toolVersions(force = false) {
  try {
    const u = await api(`/api/tools/updates${force ? "?force=true" : ""}`);
    const list = Object.values(u);
    $("#toolVersions").innerHTML = list.map(t => `<div>${t.update ? ico("download") : ico("check")} ${esc(t.message)}${t.update ?
      ` — <a href="${t.url}" target="_blank" rel="noopener">scarica da starnetastro.com</a>` : ""}</div>`).join("");
    const up = list.filter(t => t.update);
    const n = $("#toolNotice");
    n.hidden = !up.length;
    if (up.length) n.innerHTML = `Nuova versione: ${up.map(t => esc(t.message)).join(" · ")} —
      <a href="${up[0].url}" target="_blank" rel="noopener">starnetastro.com</a>, poi indica il nuovo programma in ⚙ Programmi esterni.`;
  } catch (e) { $("#toolVersions").textContent = ""; }
}
setTimeout(() => toolVersions(), 3000);
$("#openTools").onclick = async () => { await renderTools(); $("#tools").hidden = false; toolVersions(); };
$("#tCancel").onclick = () => $("#tools").hidden = true;
$("#tFind").onclick = async () => {
  $("#tFind").disabled = true;
  try { renderTools(await api("/api/tools/find", {})); } catch (e) { alert(e.message); }
  $("#tFind").disabled = false;
};
$("#tSave").onclick = async () => {
  const tools = {}; $$("[data-tool]").forEach(i => tools[i.dataset.tool] = i.value.trim());
  try { renderTools(await api("/api/config", { tools })); } catch (e) { alert(e.message); }
};

// ---------------------------------------------------------------- start
// network mode: the server only opens files under folders chosen in the GUI ("session folders")
const addRoots = paths => api("/api/roots", { paths: paths.filter(Boolean) }).catch(() => {});
renderPaths();
pollQueue();
addRoots([...state.lights, ...state.calibration, ...state.panels.flat(), state.input,
  ...state.sources.flatMap(x => [...x.lights, ...x.calibration, ...x.masterDirs, x.file].filter(Boolean))])
  .finally(() => go(state.input && !lightRoots().length ? 5 : 1));
