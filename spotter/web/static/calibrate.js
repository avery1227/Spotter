/* Spotter calibration UI.
 *
 * Three kinds of evidence, one mode each:
 *   Points   -- click a landmark in the frame, the same spot on the map, add.
 *   Lines    -- trace a feature (a wall, a path edge) in the frame and on the
 *               map; the solver lines the two up without needing matching ends.
 *   Aircraft -- freeze the live stream with every plane's ADS-B position at
 *               that instant, click where each plane really is.
 * Then solve.
 *
 * The one subtlety worth knowing: the canvas is displayed at whatever zoom
 * fits, but every coordinate that leaves this file is in FULL-RESOLUTION image
 * pixels. Clicking a 1920px-wide frame inside a 900px-wide canvas costs about
 * two real pixels of precision per click, which is the same size as the
 * reprojection errors we are trying to measure -- hence the magnifier and the
 * arrow-key nudge before a point is committed.
 */

const $ = (id) => document.getElementById(id);

const state = {
  img: null,
  imgW: 0, imgH: 0,
  zoom: null,            // null = fit to container
  pending: null,         // {x, y} in full-resolution image pixels
  points: [],
  errors: {},            // name -> reprojection error px, after a solve
  reproj: null,
  selected: -1,
  map: null, marker: null, cameraMarker: null,
  camera: null,
  pickingCamera: false,   // next map click sets the camera, not a landmark
  mode: "points",         // points | lines | aircraft
  lines: [],
  lineErrors: {},         // name -> {rms_px, elev_m}, after a solve
  draft: { image: [], map: [] },
  mapLines: [],           // Leaflet layers for saved lines
  draftLayer: null,
  freeze: null,           // the capture currently shown, if any
  aircraftChoice: null,   // track id picked in the list
  aircraftDone: new Set(),// "freezeId:trackId" already added
  encoderDelay: null,
};

/* ── helpers ───────────────────────────────────────────── */

function toast(msg, kind = "") {
  const el = $("toast");
  el.textContent = msg;
  el.className = "show " + kind;
  clearTimeout(el._t);
  el._t = setTimeout(() => { el.className = ""; }, kind === "bad" ? 7000 : 2800);
}

async function api(path, options) {
  const res = await fetch(path, options);
  let body = null;
  try { body = await res.json(); } catch (e) { /* not JSON */ }
  if (!res.ok) throw new Error((body && body.error) || `${res.status} ${res.statusText}`);
  return body;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

/* ── frame rendering ───────────────────────────────────── */

function displayScale() {
  if (state.zoom) return state.zoom;
  const avail = Math.max(320, $("frameWrap").clientWidth - 2);
  return state.imgW ? Math.min(1, avail / state.imgW) : 1;
}

function drawFrame() {
  if (!state.img) return;
  const canvas = $("frameCanvas");
  const scale = displayScale();
  const w = Math.round(state.imgW * scale);
  const h = Math.round(state.imgH * scale);
  canvas.width = w; canvas.height = h;

  const ctx = canvas.getContext("2d");
  ctx.imageSmoothingEnabled = scale < 1;
  ctx.drawImage(state.img, 0, 0, w, h);
  $("zoomLabel").textContent = state.zoom ? `${Math.round(scale * 100)}%` : "fit";

  if ($("showHorizon").checked && state.reproj && state.reproj.horizon.length > 1) {
    ctx.strokeStyle = "rgba(255,213,79,.85)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    state.reproj.horizon.forEach(([x, y], i) => {
      i ? ctx.lineTo(x * scale, y * scale) : ctx.moveTo(x * scale, y * scale);
    });
    ctx.stroke();
  }

  if ($("showReproj").checked && state.reproj) {
    ctx.strokeStyle = "rgba(0,229,255,.9)";
    ctx.lineWidth = 1.4;
    state.reproj.points.forEach((p) => {
      if (!p.in_front) return;
      ctx.beginPath();
      ctx.arc(p.u * scale, p.v * scale, 7, 0, Math.PI * 2);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(p.px * scale, p.py * scale);
      ctx.lineTo(p.u * scale, p.v * scale);
      ctx.stroke();
    });
  }

  drawLinesOnFrame(ctx, scale);
  if (state.freeze && state.mode === "aircraft") drawAircraft(ctx, scale);

  state.points.forEach((p, i) => {
    const x = p.px * scale, y = p.py * scale;
    const colour = i === state.selected ? "#ffd54f" : (p.enabled ? "#7ed957" : "#6b7885");
    ctx.strokeStyle = colour;
    ctx.lineWidth = i === state.selected ? 2.4 : 1.8;
    ctx.beginPath();
    ctx.moveTo(x - 8, y); ctx.lineTo(x + 8, y);
    ctx.moveTo(x, y - 8); ctx.lineTo(x, y + 8);
    ctx.stroke();

    const label = `${i + 1} ${p.kind === "aircraft" ? "✈ " : ""}${p.name}`;
    ctx.font = "11px system-ui, sans-serif";
    ctx.lineWidth = 3;
    ctx.strokeStyle = "rgba(0,0,0,.85)";
    ctx.strokeText(label, x + 11, y - 6);
    ctx.fillStyle = colour;
    ctx.fillText(label, x + 11, y - 6);
  });

  if (state.pending) {
    const x = state.pending.x * scale, y = state.pending.y * scale;
    ctx.strokeStyle = "#ff9800"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(x, y, 10, 0, Math.PI * 2); ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x - 14, y); ctx.lineTo(x + 14, y);
    ctx.moveTo(x, y - 14); ctx.lineTo(x, y + 14);
    ctx.stroke();
  }
}

function strokePath(ctx, pts, scale) {
  ctx.beginPath();
  pts.forEach(([x, y], i) => {
    i ? ctx.lineTo(x * scale, y * scale) : ctx.moveTo(x * scale, y * scale);
  });
  ctx.stroke();
}

function drawLinesOnFrame(ctx, scale) {
  // Where the saved calibration puts each map line: this is what should sit
  // on top of the traced line if the calibration is right.
  if ($("showReproj").checked && state.reproj && state.reproj.lines) {
    ctx.save();
    ctx.setLineDash([6, 4]);
    ctx.strokeStyle = "rgba(0,229,255,.9)";
    ctx.lineWidth = 1.5;
    state.reproj.lines.forEach((l) => l.projected.forEach((run) => strokePath(ctx, run, scale)));
    ctx.restore();
  }
  state.lines.forEach((line) => {
    ctx.strokeStyle = line.enabled ? "rgba(255,152,0,.95)" : "rgba(107,120,133,.8)";
    ctx.lineWidth = 2;
    strokePath(ctx, line.image, scale);
    const [x, y] = line.image[0];
    ctx.font = "11px system-ui, sans-serif";
    ctx.lineWidth = 3; ctx.strokeStyle = "rgba(0,0,0,.85)";
    ctx.strokeText(line.name, x * scale + 6, y * scale - 6);
    ctx.fillStyle = "#ff9800";
    ctx.fillText(line.name, x * scale + 6, y * scale - 6);
  });
  if (state.mode === "lines" && state.draft.image.length) {
    ctx.strokeStyle = "#ffd54f"; ctx.lineWidth = 2;
    strokePath(ctx, state.draft.image, scale);
    ctx.fillStyle = "#ffd54f";
    state.draft.image.forEach(([x, y]) => {
      ctx.beginPath(); ctx.arc(x * scale, y * scale, 3.5, 0, Math.PI * 2); ctx.fill();
    });
  }
}

function drawAircraft(ctx, scale) {
  ctx.font = "12px system-ui, sans-serif";
  state.freeze.aircraft.forEach((a) => {
    if (!a.predicted) return;
    const [x, y] = [a.predicted[0] * scale, a.predicted[1] * scale];
    const chosen = a.id === state.aircraftChoice;
    const done = state.aircraftDone.has(`${state.freeze.id}:${a.id}`);
    ctx.strokeStyle = chosen ? "#ffd54f" : (done ? "rgba(126,217,87,.9)" : "rgba(0,229,255,.9)");
    ctx.lineWidth = chosen ? 2.2 : 1.4;
    ctx.beginPath(); ctx.arc(x, y, 9, 0, Math.PI * 2); ctx.stroke();
    const text = `${a.label} · tracker`;
    ctx.lineWidth = 3; ctx.strokeStyle = "rgba(0,0,0,.85)";
    ctx.strokeText(text, x + 12, y + 4);
    ctx.fillStyle = chosen ? "#ffd54f" : "#4fc3f7";
    ctx.fillText(text, x + 12, y + 4);
    // Tie the chosen plane's tracker position to the click, so the size of
    // the correction is visible.
    if (chosen && state.pending) {
      ctx.strokeStyle = "rgba(255,213,79,.8)"; ctx.lineWidth = 1.2;
      ctx.setLineDash([4, 3]);
      ctx.beginPath(); ctx.moveTo(x, y);
      ctx.lineTo(state.pending.x * scale, state.pending.y * scale); ctx.stroke();
      ctx.setLineDash([]);
    }
  });
}

function drawLoupe() {
  const canvas = $("loupe");
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#06090d";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  if (!state.img || !state.pending) return;

  const zoom = 8;
  const span = canvas.width / zoom;
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(state.img,
                state.pending.x - span / 2, state.pending.y - span / 2, span, span,
                0, 0, canvas.width, canvas.height);

  const mid = canvas.width / 2;
  ctx.strokeStyle = "rgba(255,152,0,.95)"; ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(mid, 0); ctx.lineTo(mid, canvas.height);
  ctx.moveTo(0, mid); ctx.lineTo(canvas.width, mid);
  ctx.stroke();
  ctx.strokeStyle = "rgba(255,152,0,.5)";
  ctx.strokeRect(mid - zoom / 2, mid - zoom / 2, zoom, zoom);
}

function setPending(x, y) {
  state.pending = {
    x: Math.max(0, Math.min(state.imgW - 1, x)),
    y: Math.max(0, Math.min(state.imgH - 1, y)),
  };
  const text = `${state.pending.x.toFixed(1)}, ${state.pending.y.toFixed(1)}`;
  $("pixelReadout").textContent = text;
  $("fPixel").value = text;
  if (state.mode === "aircraft") renderAircraftList(true);
  drawFrame();
  drawLoupe();
}

/* ── loading ───────────────────────────────────────────── */

function loadImage(src) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      state.img = img;
      state.imgW = img.naturalWidth;
      state.imgH = img.naturalHeight;
      $("framePlaceholder").style.display = "none";
      $("frameCanvas").style.display = "block";
      drawFrame(); drawLoupe();
      resolve(true);
    };
    img.onerror = () => resolve(false);
    img.src = src;
  });
}

async function loadFrame(bust) {
  const ok = await loadImage("/api/frame.png?t=" + (bust || Date.now()));
  if (!ok && !state.freeze) {
    $("framePlaceholder").style.display = "block";
    $("frameCanvas").style.display = "none";
  }
  return ok;
}

async function loadPoints() {
  state.points = (await api("/api/points")).points;
  renderTable();
  drawFrame();
}

async function loadReprojection() {
  try {
    state.reproj = await api("/api/reprojection");
    state.reproj.points.forEach((p) => {
      if (p.error_px !== undefined) state.errors[p.name] = p.error_px;
    });
    renderTable();
  } catch (e) {
    state.reproj = null;      // no calibration saved yet -- not an error
  }
  drawFrame();
}

/* ── points table ──────────────────────────────────────── */

function errClass(e) {
  if (e === undefined) return "";
  return e < 3 ? "err-ok" : (e < 8 ? "err-warn" : "err-bad");
}

function renderTable() {
  const tbody = document.querySelector("#pointsTable tbody");
  tbody.innerHTML = "";
  $("pointCount").textContent =
    `${state.points.filter((p) => p.enabled).length} enabled / ${state.points.length}`;

  state.points.forEach((point, index) => {
    const tr = document.createElement("tr");
    tr.className = (point.enabled ? "" : "disabled") +
                   (index === state.selected ? " selected" : "");

    const err = state.errors[point.name];
    tr.innerHTML = `
      <td><input type="checkbox" ${point.enabled ? "checked" : ""}></td>
      <td title="${escapeHtml(point.note || "")}">${point.kind === "aircraft" ? "✈ " : ""}${escapeHtml(point.name)}</td>
      <td class="num">${point.px.toFixed(0)}, ${point.py.toFixed(0)}</td>
      <td class="num">${point.lat.toFixed(5)}, ${point.lon.toFixed(5)}</td>
      <td class="num">${point.elev_m}</td>
      <td class="num ${errClass(err)}">${err === undefined ? "—" : err.toFixed(1)}</td>
      <td><button class="btn small danger">×</button></td>`;

    tr.querySelector("input[type=checkbox]").addEventListener("change", async (ev) => {
      ev.stopPropagation();
      await api(`/api/points/${index}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled: ev.target.checked }),
      });
      await loadPoints();
    });

    tr.querySelector("button").addEventListener("click", async (ev) => {
      ev.stopPropagation();
      if (!confirm(`Delete "${point.name}"?`)) return;
      try {
        // The name is sent so the server can refuse if this tab is stale and
        // index 3 is no longer the point shown on this row.
        const res = await api(
          `/api/points/${index}?name=${encodeURIComponent(point.name)}`,
          { method: "DELETE" });
        state.selected = -1;
        await loadPoints();
        offerUndo(res.removed);
      } catch (err) {
        toast(err.message, "bad");
        await loadPoints();
      }
    });

    tr.addEventListener("click", () => {
      state.selected = state.selected === index ? -1 : index;
      if (state.selected >= 0 && state.map) {
        state.map.setView([point.lat, point.lon], Math.max(state.map.getZoom(), 15));
        placeMarker(point.lat, point.lon);
      }
      renderTable();
      drawFrame();
    });

    tbody.appendChild(tr);
  });
}

/* Deleting a point that took real effort to place should never be a one-way
 * door, so the last deletion can be put straight back. */
function offerUndo(removed) {
  if (!removed) { toast("Point deleted"); return; }
  const el = $("toast");
  el.innerHTML = "";
  el.appendChild(document.createTextNode(`Deleted "${removed.name}" `));
  const btn = document.createElement("button");
  btn.className = "btn small";
  btn.textContent = "Undo";
  btn.addEventListener("click", async () => {
    try {
      await api("/api/points", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(removed),
      });
      await loadPoints();
      toast("Restored", "good");
    } catch (e) {
      toast(e.message, "bad");
    }
  });
  el.appendChild(btn);
  el.className = "show";
  clearTimeout(el._t);
  el._t = setTimeout(() => { el.className = ""; }, 9000);
}


/* ── map ───────────────────────────────────────────────── */

function placeMarker(lat, lon) {
  if (!state.map) return;
  if (state.marker) state.marker.setLatLng([lat, lon]);
  else state.marker = L.marker([lat, lon], { draggable: true }).addTo(state.map)
        .on("dragend", (e) => {
          const p = e.target.getLatLng();
          setLatLonFields(p.lat, p.lng);
        });
  setLatLonFields(lat, lon);
}

function setLatLonFields(lat, lon) {
  $("fLat").value = Number(lat).toFixed(7);
  $("fLon").value = Number(lon).toFixed(7);
}

function initMap(camera) {
  const centre = [camera.lat || 41.27, camera.lon || -72.5];
  state.map = L.map("map", { zoomControl: true }).setView(centre, 14);

  // Satellite imagery makes jetties, breakwaters and rocks far easier to
  // identify than a street map does.
  const sat = L.tileLayer(
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    { maxZoom: 21, attribution: "Imagery © Esri" });
  const street = L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    { maxZoom: 19, attribution: "© OpenStreetMap" });
  sat.addTo(state.map);
  L.control.layers({ "Satellite": sat, "Street": street }).addTo(state.map);

  state.cameraMarker = L.circleMarker(centre, {
    radius: 7, color: "#4fc3f7", fillColor: "#4fc3f7", fillOpacity: .5,
  }).addTo(state.map).bindTooltip("camera");

  state.map.on("click", (e) => {
    if (state.pickingCamera) {
      setCameraFields(e.latlng.lat, e.latlng.lng);
      state.cameraMarker.setLatLng(e.latlng);
      setPickingCamera(false);
      toast("Camera position set — press Save camera position");
    } else if (state.mode === "lines") {
      state.draft.map.push([e.latlng.lat, e.latlng.lng]);
      updateDraft();
    } else {
      placeMarker(e.latlng.lat, e.latlng.lng);
    }
  });
}

/* Accepts "41.276123, -72.459456", "41.276123 -72.459456", and the
 * degrees-minutes-seconds form Google Maps shows in its search box. */
function parseLatLon(text) {
  const clean = String(text).trim().replace(/[()]/g, "");
  const dec = clean.match(/^(-?\d+(?:\.\d+)?)\s*[, ]\s*(-?\d+(?:\.\d+)?)$/);
  if (dec) return { lat: parseFloat(dec[1]), lon: parseFloat(dec[2]) };

  const dms = clean.match(
    /(\d+)°\s*(\d+)'\s*([\d.]+)"?\s*([NS])[, ]+\s*(\d+)°\s*(\d+)'\s*([\d.]+)"?\s*([EW])/i);
  if (dms) {
    const toDeg = (d, m, s, hemi) => {
      const v = (+d) + (+m) / 60 + (+s) / 3600;
      return /[SW]/i.test(hemi) ? -v : v;
    };
    return {
      lat: toDeg(dms[1], dms[2], dms[3], dms[4]),
      lon: toDeg(dms[5], dms[6], dms[7], dms[8]),
    };
  }
  return null;
}

/* ── camera position ───────────────────────────────────── */

function setCameraFields(lat, lon) {
  $("camLat").value = Number(lat).toFixed(7);
  $("camLon").value = Number(lon).toFixed(7);
}

function setPickingCamera(on) {
  state.pickingCamera = on;
  document.querySelector(".camerabox").classList.toggle("picking", on);
  $("pickCameraBtn").textContent = on ? "Click the map…" : "Pick on map";
  $("camHint").textContent = on
    ? "Click where the camera is."
    : "";
}

function syncHeightUnknown() {
  const unknown = $("camHeightUnknown").checked;
  $("camHeight").disabled = unknown;
  $("camUnknownNote").style.display = unknown ? "" : "none";
  if (unknown && !$("camHeight").value) $("camHeight").value = 12;
}

async function saveCamera() {
  const lat = parseFloat($("camLat").value);
  const lon = parseFloat($("camLon").value);
  if (!isFinite(lat) || !isFinite(lon)) {
    toast("Set the camera latitude and longitude first", "bad");
    return;
  }
  const unknown = $("camHeightUnknown").checked;
  const btn = $("saveCameraBtn");
  btn.disabled = true;
  try {
    const res = await api("/api/camera", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        lat, lon,
        height_m: parseFloat($("camHeight").value) || 0,
        height_unknown: unknown,
      }),
    });
    state.camera = res.camera;
    $("camHeight").value = res.camera.height_m;
    $("camHint").textContent = unknown
      ? `saved · height free to move (±${res.height_sigma_m} m)`
      : `saved · height held near ${res.camera.height_m} m`;
    if (state.cameraMarker) state.cameraMarker.setLatLng([res.camera.lat, res.camera.lon]);
    if (state.map) state.map.setView([res.camera.lat, res.camera.lon],
                                     Math.max(state.map.getZoom(), 15));
    updateHeaderInfo();
    // The prior changed, so any earlier solve is stale.
    $("solveOut").innerHTML =
      `<p class="muted small">Camera position changed — solve again.</p>`;
    toast("Camera position saved to config.yaml", "good");
  } catch (e) {
    toast(e.message, "bad");
  } finally {
    btn.disabled = false;
  }
}

function updateHeaderInfo() {
  const c = state.camera || {};
  $("headerInfo").textContent =
    `camera ${Number(c.lat).toFixed(5)}, ${Number(c.lon).toFixed(5)}` +
    ` · ${c.height_m} m`;
}

/* ── modes ─────────────────────────────────────────────── */

const HINTS = {
  points: "Click a feature you can also find on the map: a jetty tip, a breakwater light, a chimney, a distinctive rock. Mix near and far — points strung along the horizon leave pitch, height and focal length trading off against each other.",
  lines: "Click along one edge: a wall top, a path, the waterline. A handful of clicks is enough; follow bends. Then trace the same edge on the map.",
  aircraft: "Click the centre of the plane (or its lights at night). Arrow keys nudge a pixel. The list below picks the nearest tracked aircraft; change it if that's the wrong one.",
};

const HEADINGS = {
  points: "1 · Click the landmark in the frame",
  lines: "1 · Trace a feature in the frame",
  aircraft: "1 · Click where the plane really is",
};

function setMode(mode) {
  state.mode = mode;
  document.querySelectorAll("[data-set-mode]").forEach((b) =>
    b.classList.toggle("active", b.dataset.setMode === mode));
  document.querySelectorAll("[data-mode]").forEach((el) =>
    el.classList.toggle("mode-hidden", !el.dataset.mode.split(" ").includes(mode)));
  $("frameHeading").textContent = HEADINGS[mode];
  $("modeHint").textContent = HINTS[mode];
  $("mapHeading").textContent = mode === "lines"
    ? "2 · Trace the same feature on the map" : "2 · Put the same spot on the map";
  if (mode !== "points" && state.marker) { state.map.removeLayer(state.marker); state.marker = null; }
  state.pending = null;
  $("pixelReadout").textContent = "—";
  if (state.map) setTimeout(() => state.map.invalidateSize(), 0);
  // Leaving aircraft mode goes back to the calibration frame.
  if (mode !== "aircraft" && state.freeze) unfreeze();
  updateDraft();
  drawFrame(); drawLoupe();
}

/* ── lines ─────────────────────────────────────────────── */

async function loadLines() {
  state.lines = (await api("/api/lines")).lines;
  renderLinesTable();
  drawMapLines();
  // Refresh the projected map lines too, so a new line shows how far the
  // current calibration is from it straight away.
  await loadReprojection();
}

function drawMapLines() {
  if (!state.map) return;
  state.mapLines.forEach((layer) => state.map.removeLayer(layer));
  state.mapLines = state.lines.map((line) =>
    L.polyline(line.map, { color: line.enabled ? "#ff9800" : "#6b7885", weight: 3 })
      .addTo(state.map).bindTooltip(line.name));
}

function updateDraft() {
  $("lineImgCount").textContent = state.draft.image.length;
  $("lineMapCount").textContent = state.draft.map.length;
  if (!state.map) return;
  if (state.draftLayer) { state.map.removeLayer(state.draftLayer); state.draftLayer = null; }
  if (state.mode === "lines" && state.draft.map.length) {
    state.draftLayer = L.layerGroup([
      L.polyline(state.draft.map, { color: "#ffd54f", weight: 3, dashArray: "6 4" }),
      ...state.draft.map.map((p) => L.circleMarker(p, { radius: 4, color: "#ffd54f" })),
    ]).addTo(state.map);
  }
}

function renderLinesTable() {
  const tbody = document.querySelector("#linesTable tbody");
  tbody.innerHTML = "";
  $("lineCount").textContent =
    `${state.lines.filter((l) => l.enabled).length} enabled / ${state.lines.length}`;
  state.lines.forEach((line, index) => {
    const tr = document.createElement("tr");
    tr.className = line.enabled ? "" : "disabled";
    const fit = state.lineErrors[line.name];
    const height = line.elev_sigma_m > 0
      ? `${line.elev_m} ±${line.elev_sigma_m}` + (fit && fit.elev_fitted ? ` → ${fit.elev_m}` : "")
      : `${line.elev_m} exact`;
    tr.innerHTML = `
      <td><input type="checkbox" ${line.enabled ? "checked" : ""}></td>
      <td>${escapeHtml(line.name)}</td>
      <td class="num">${line.image.length} / ${line.map.length}</td>
      <td class="num">${height}</td>
      <td class="num ${fit ? errClass(fit.rms_px) : ""}">${fit ? fit.rms_px.toFixed(1) : "—"}</td>
      <td><button class="btn small danger">×</button></td>`;
    const path = `/api/lines/${index}?name=${encodeURIComponent(line.name)}`;
    tr.querySelector("input").addEventListener("change", async (ev) => {
      try {
        await api(path, { method: "PATCH", headers: { "Content-Type": "application/json" },
                          body: JSON.stringify({ enabled: ev.target.checked }) });
      } catch (e) { toast(e.message, "bad"); }
      await loadLines();
    });
    tr.querySelector("button").addEventListener("click", async () => {
      if (!confirm(`Delete line "${line.name}"?`)) return;
      try {
        const res = await api(path, { method: "DELETE" });
        await loadLines();
        offerLineUndo(res.removed);
      } catch (e) { toast(e.message, "bad"); await loadLines(); }
    });
    tbody.appendChild(tr);
  });
}

function offerLineUndo(removed) {
  const el = $("toast");
  el.innerHTML = "";
  el.appendChild(document.createTextNode(`Deleted line "${removed.name}" `));
  const btn = document.createElement("button");
  btn.className = "btn small"; btn.textContent = "Undo";
  btn.addEventListener("click", async () => {
    try {
      await api("/api/lines", { method: "POST", headers: { "Content-Type": "application/json" },
                                body: JSON.stringify(removed) });
      await loadLines();
      toast("Restored", "good");
    } catch (e) { toast(e.message, "bad"); }
  });
  el.appendChild(btn);
  el.className = "show";
  clearTimeout(el._t);
  el._t = setTimeout(() => { el.className = ""; }, 9000);
}

async function addLine(e) {
  e.preventDefault();
  if (state.draft.image.length < 2) { toast("Click at least 2 points along the feature in the frame", "bad"); return; }
  if (state.draft.map.length < 2) { toast("Trace the same feature on the map (2+ clicks)", "bad"); return; }
  try {
    await api("/api/lines", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name: $("lName").value, image: state.draft.image, map: state.draft.map,
        elev_m: parseFloat($("lElev").value) || 0,
        elev_sigma_m: parseFloat($("lSigma").value),
      }),
    });
    toast(`Added line "${$("lName").value}"`, "good");
    state.draft = { image: [], map: [] };
    $("lName").value = "";
    updateDraft();
    await loadLines();
  } catch (err) {
    toast(err.message, "bad");
  }
}

/* ── aircraft ──────────────────────────────────────────── */

async function freezeFrame() {
  const btn = $("freezeBtn");
  btn.disabled = true; btn.textContent = "Freezing…";
  try {
    const capture = await api("/api/freeze", { method: "POST" });
    state.freeze = capture;
    state.aircraftChoice = null;
    state.pending = null;
    await loadImage(`/api/freeze/${capture.id}.png`);
    const when = new Date(capture.frame_time);
    $("freezeTime").textContent = when.toLocaleTimeString();
    const inView = capture.aircraft.filter((a) => a.predicted).length;
    $("freezeCount").textContent =
      `${capture.aircraft.length} aircraft tracked, ${inView} in front of the camera`;
    $("freezeBanner").style.display = "";
    $("freezeHint").textContent = capture.aircraft.length
      ? "Click the plane in the frame."
      : "No aircraft in range right now; try again when one is in view.";
    renderAircraftList(false);
    drawFrame(); drawLoupe();
  } catch (e) {
    toast(e.message, "bad");
  } finally {
    btn.disabled = false; btn.textContent = "Freeze frame + aircraft";
  }
}

async function unfreeze() {
  state.freeze = null;
  state.aircraftChoice = null;
  $("freezeBanner").style.display = "none";
  $("aircraftList").innerHTML = "";
  $("addAircraftBtn").disabled = true;
  await loadFrame();
}

function renderAircraftList(rerank) {
  const list = $("aircraftList");
  list.innerHTML = "";
  if (!state.freeze) return;
  let planes = state.freeze.aircraft.slice();
  // After a click, the nearest tracker position is the likeliest match; put
  // it first and pre-select it, but let the user overrule.
  if (state.pending) {
    const dist = (a) => a.predicted
      ? Math.hypot(a.predicted[0] - state.pending.x, a.predicted[1] - state.pending.y)
      : Infinity;
    planes.sort((a, b) => dist(a) - dist(b) || a.range_km - b.range_km);
    if (rerank || !state.aircraftChoice) state.aircraftChoice = planes.length ? planes[0].id : null;
  }
  planes.forEach((a) => {
    const done = state.aircraftDone.has(`${state.freeze.id}:${a.id}`);
    const label = document.createElement("label");
    label.className = (a.id === state.aircraftChoice ? "chosen " : "") + (done ? "done" : "");
    const feet = Math.round(a.alt_m / 0.3048 / 100) * 100;
    const offset = state.pending && a.predicted
      ? `${Math.round(Math.hypot(a.predicted[0] - state.pending.x, a.predicted[1] - state.pending.y))} px from click`
      : (a.predicted ? "" : "behind / off to the side");
    label.innerHTML = `
      <input type="radio" name="plane" ${a.id === state.aircraftChoice ? "checked" : ""}>
      <span><b>${escapeHtml(a.label)}</b>${done ? " ✓" : ""}
        <span class="meta">${feet.toLocaleString()} ft · ${a.range_km.toFixed(1)} km · ${a.bearing_deg.toFixed(0)}°</span>
        ${a.alt_source === "baro" ? '<span class="warn small" title="pressure altitude: can be 100 m+ off true height">baro alt</span>' : ""}</span>
      <span class="meta">${offset}</span>`;
    label.querySelector("input").addEventListener("change", () => {
      state.aircraftChoice = a.id;
      renderAircraftList(false);
      drawFrame();
    });
    list.appendChild(label);
  });
  $("addAircraftBtn").disabled = !(state.pending && state.aircraftChoice);
}

async function addAircraftPoint() {
  if (!state.freeze || !state.pending || !state.aircraftChoice) return;
  try {
    const res = await api(`/api/freeze/${state.freeze.id}/points`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ aircraft: state.aircraftChoice,
                             px: state.pending.x, py: state.pending.y }),
    });
    state.aircraftDone.add(`${state.freeze.id}:${state.aircraftChoice}`);
    toast(`Added ${res.added.name}`, "good");
    state.pending = null;
    state.aircraftChoice = null;
    $("pixelReadout").textContent = "—";
    await loadPoints();
    renderAircraftList(false);
    drawLoupe();
  } catch (e) {
    toast(e.message, "bad");
  }
}

/* ── solve ─────────────────────────────────────────────── */

function renderSolve(data) {
  const m = data.model;
  const s = data.summary;
  const rms = s.rms_px;
  const verdict = rms < 2 ? "okbox" : (rms < 6 ? "warnbox" : "errbox");

  let html = `<div class="${verdict}">
    <b>RMS ${rms.toFixed(2)} px</b> · worst ${s.max_px.toFixed(2)} px
    over ${s.n_points} points${s.n_lines ? ` and ${s.n_lines} lines` : ""}${data.saved ? " · saved to calibration.json" : ""}
  </div>
  <dl class="kv">
    <dt>pan / yaw</dt><dd>${m.yaw_deg.toFixed(3)}° from north</dd>
    <dt>tilt / pitch</dt><dd>${m.pitch_deg > 0 ? "+" : ""}${m.pitch_deg.toFixed(3)}°</dd>
    <dt>roll</dt><dd>${m.roll_deg > 0 ? "+" : ""}${m.roll_deg.toFixed(3)}°</dd>
    <dt>focal</dt><dd>${m.focal_px.toFixed(1)} px</dd>
    <dt>field of view</dt><dd>${m.hfov_deg.toFixed(2)}° × ${m.vfov_deg.toFixed(2)}°</dd>
    <dt>distortion</dt><dd>k1 ${m.k1.toFixed(5)} · k2 ${m.k2.toFixed(5)}</dd>
    <dt>height</dt><dd>${m.height_m.toFixed(2)} m above sea level</dd>
    <dt>position</dt><dd>${m.lat.toFixed(6)}, ${m.lon.toFixed(6)}</dd>
    <dt>offset</dt><dd>${m.offset_e_m.toFixed(1)} m E, ${m.offset_n_m.toFixed(1)} m N</dd>
  </dl>`;

  if (s.timing_offset_s !== null && s.timing_offset_s !== undefined) {
    const off = s.timing_offset_s;
    const later = off >= 0 ? "later" : "earlier";
    html += `<div class="${Math.abs(off) < 0.5 ? "okbox" : "warnbox"}">
      <b>Timing:</b> frames are ${Math.abs(off).toFixed(1)} s ${later} than the overlay assumed.`;
    if (s.suggested_encoder_delay_s !== null && s.suggested_encoder_delay_s !== undefined
        && Math.abs(off) >= 0.5) {
      html += ` Set <code>stream.encoder_delay_s</code> (the <code>ENCODER_DELAY_S</code>
        variable in Pelican) to <b>${s.suggested_encoder_delay_s.toFixed(1)} s</b>, then capture
        fresh aircraft points: the ones you have were positioned with the old delay.`;
    }
    html += `</div>`;
  }

  (data.warnings || []).forEach((w) => {
    html += `<div class="warnbox">${escapeHtml(w)}</div>`;
  });
  if (!data.warnings || !data.warnings.length) {
    html += `<div class="okbox">No warnings: point geometry and residuals look healthy.</div>`;
  }
  $("solveOut").innerHTML = html;

  state.errors = {};
  (data.points || []).forEach((p) => { state.errors[p.name] = p.error_px; });
  state.lineErrors = {};
  (data.lines || []).forEach((l) => { state.lineErrors[l.name] = l; });
  renderTable();
  renderLinesTable();
}

async function doSolve(save) {
  const body = {
    save,
    loo: true,
    lock_height: $("lockHeight").checked,
    lock_position: $("lockPosition").checked,
    lock_roll: $("lockRoll").checked,
    lock_distortion: $("lockDistortion").checked,
    fit_timing: $("fitTiming").checked,
  };
  $("solveBtn").disabled = $("saveBtn").disabled = true;
  $("solveOut").innerHTML = `<p class="muted">Solving…</p>`;
  try {
    const data = await api("/api/solve", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    renderSolve(data);
    if (save) {
      $("showReproj").checked = true;
      await loadReprojection();
      toast(data.pipeline_waiting
              ? "Calibration saved; the pipeline is starting now"
              : "Calibration saved", "good");
    }
  } catch (e) {
    $("solveOut").innerHTML = `<div class="errbox">${escapeHtml(e.message)}</div>`;
    toast(e.message, "bad");
  } finally {
    $("solveBtn").disabled = $("saveBtn").disabled = false;
  }
}

/* ── wiring ────────────────────────────────────────────── */

function wire() {
  const canvas = $("frameCanvas");

  canvas.addEventListener("click", (e) => {
    const rect = canvas.getBoundingClientRect();
    const scale = displayScale();
    const x = (e.clientX - rect.left) / scale, y = (e.clientY - rect.top) / scale;
    if (state.mode === "lines") {
      state.draft.image.push([Math.round(x * 10) / 10, Math.round(y * 10) / 10]);
      updateDraft();
      drawFrame();
      return;
    }
    if (state.mode === "aircraft" && !state.freeze) {
      toast("Press Freeze first, so the frame and the aircraft positions match", "bad");
      return;
    }
    setPending(x, y);
  });

  document.querySelectorAll("[data-set-mode]").forEach((b) =>
    b.addEventListener("click", () => setMode(b.dataset.setMode)));
  $("lineImgUndo").addEventListener("click", () => { state.draft.image.pop(); updateDraft(); drawFrame(); });
  $("lineMapUndo").addEventListener("click", () => { state.draft.map.pop(); updateDraft(); });
  $("lineClearBtn").addEventListener("click", () => {
    state.draft = { image: [], map: [] }; updateDraft(); drawFrame();
  });
  $("lineForm").addEventListener("submit", addLine);
  $("freezeBtn").addEventListener("click", freezeFrame);
  $("unfreezeBtn").addEventListener("click", unfreeze);
  $("addAircraftBtn").addEventListener("click", addAircraftPoint);

  // Arrow keys nudge the pending point one full-resolution pixel.
  document.addEventListener("keydown", (e) => {
    if (!state.pending) return;
    if (/^(INPUT|TEXTAREA)$/.test(document.activeElement.tagName)) return;
    const step = e.shiftKey ? 10 : 1;
    const moves = {
      ArrowLeft: [-step, 0], ArrowRight: [step, 0],
      ArrowUp: [0, -step], ArrowDown: [0, step],
    };
    if (!moves[e.key]) return;
    e.preventDefault();
    setPending(state.pending.x + moves[e.key][0], state.pending.y + moves[e.key][1]);
  });

  $("zoomIn").addEventListener("click", () => {
    state.zoom = Math.min(8, (state.zoom || displayScale()) * 1.5);
    drawFrame();
  });
  $("zoomOut").addEventListener("click", () => {
    const next = (state.zoom || displayScale()) / 1.5;
    state.zoom = next <= displayScale() * 1.01 ? null : next;
    drawFrame();
  });
  $("showReproj").addEventListener("change", drawFrame);
  $("showHorizon").addEventListener("change", drawFrame);
  window.addEventListener("resize", () => { if (!state.zoom) drawFrame(); });

  $("grabBtn").addEventListener("click", async () => {
    const btn = $("grabBtn");
    btn.disabled = true; btn.textContent = "Grabbing…";
    try {
      const info = await api("/api/grab", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ timeout: 150 }),
      });
      await loadFrame(Date.now());
      toast(`New frame · ${info.width}×${info.height}`, "good");
    } catch (e) {
      toast(e.message, "bad");
    } finally {
      btn.disabled = false; btn.textContent = "Grab fresh frame";
    }
  });

  const usePaste = () => {
    const parsed = parseLatLon($("pasteBox").value);
    if (!parsed) { toast("Could not read coordinates from that", "bad"); return; }
    placeMarker(parsed.lat, parsed.lon);
    state.map.setView([parsed.lat, parsed.lon], Math.max(state.map.getZoom(), 17));
    $("pasteBox").value = "";
  };
  $("pasteBtn").addEventListener("click", usePaste);
  $("pasteBox").addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); usePaste(); }
  });
  // Pasting straight into the box is the common case; act on it immediately.
  $("pasteBox").addEventListener("paste", () => setTimeout(usePaste, 0));

  $("pointForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!state.pending) { toast("Click a point in the frame first", "bad"); return; }
    const lat = parseFloat($("fLat").value), lon = parseFloat($("fLon").value);
    if (!isFinite(lat) || !isFinite(lon)) {
      toast("Set a position on the map, or paste coordinates", "bad"); return;
    }
    try {
      await api("/api/points", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: $("fName").value, px: state.pending.x, py: state.pending.y,
          lat, lon, elev_m: parseFloat($("fElev").value) || 0,
        }),
      });
      toast(`Added "${$("fName").value}"`, "good");
      state.pending = null;
      $("fName").value = ""; $("fPixel").value = ""; $("pixelReadout").textContent = "—";
      await loadPoints();
      drawLoupe();
      $("fName").focus();
    } catch (err) {
      toast(err.message, "bad");
    }
  });

  $("pickCameraBtn").addEventListener("click", () => setPickingCamera(!state.pickingCamera));
  $("saveCameraBtn").addEventListener("click", saveCamera);
  $("camHeightUnknown").addEventListener("change", syncHeightUnknown);

  $("solveBtn").addEventListener("click", () => doSolve(false));
  $("saveBtn").addEventListener("click", () => doSolve(true));
}

/* ── boot ──────────────────────────────────────────────── */

(async function main() {
  wire();
  let info;
  try {
    info = await api("/api/info");
  } catch (e) {
    toast("Cannot reach the server: " + e.message, "bad");
    return;
  }
  state.camera = info.camera;
  initMap(info.camera);

  setCameraFields(info.camera.lat, info.camera.lon);
  $("camHeight").value = info.camera.height_m;
  // A wide height prior in the config means a previous save said "unknown".
  $("camHeightUnknown").checked = Number(info.camera.height_sigma_m) >= 30;
  syncHeightUnknown();
  if (!info.camera.editable) {
    $("saveCameraBtn").disabled = true;
    $("camHint").textContent = "config was not loaded from a file";
  }
  updateHeaderInfo();

  state.encoderDelay = info.encoder_delay_s;
  if (!info.can_freeze) {
    $("freezeBtn").disabled = true;
    $("freezeHint").textContent =
      "Needs the live pipeline: start Spotter with 'run' (the Pelican default).";
  }
  setMode("points");

  await loadFrame();
  await loadPoints();
  await loadLines();
  await loadReprojection();

  if (state.reproj) {
    $("showReproj").checked = true;
    drawFrame();
  }
})();
