/* Spotter calibration UI.
 *
 * Workflow: click the landmark in the frame, click the same spot on the map
 * (or paste coordinates out of Google Maps), name it, add. Repeat six to
 * twelve times, then solve.
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

  state.points.forEach((p, i) => {
    const x = p.px * scale, y = p.py * scale;
    const colour = i === state.selected ? "#ffd54f" : (p.enabled ? "#7ed957" : "#6b7885");
    ctx.strokeStyle = colour;
    ctx.lineWidth = i === state.selected ? 2.4 : 1.8;
    ctx.beginPath();
    ctx.moveTo(x - 8, y); ctx.lineTo(x + 8, y);
    ctx.moveTo(x, y - 8); ctx.lineTo(x, y + 8);
    ctx.stroke();

    const label = `${i + 1} ${p.name}`;
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
  drawFrame();
  drawLoupe();
}

/* ── loading ───────────────────────────────────────────── */

function loadFrame(bust) {
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
    img.onerror = () => {
      $("framePlaceholder").style.display = "block";
      $("frameCanvas").style.display = "none";
      resolve(false);
    };
    img.src = "/api/frame.png?t=" + (bust || Date.now());
  });
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
      <td>${escapeHtml(point.name)}</td>
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

/* ── solve ─────────────────────────────────────────────── */

function renderSolve(data) {
  const m = data.model;
  const s = data.summary;
  const rms = s.rms_px;
  const verdict = rms < 2 ? "okbox" : (rms < 6 ? "warnbox" : "errbox");

  let html = `<div class="${verdict}">
    <b>RMS ${rms.toFixed(2)} px</b> · worst ${s.max_px.toFixed(2)} px
    over ${s.n_points} points${data.saved ? " · saved to calibration.json" : ""}
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

  (data.warnings || []).forEach((w) => {
    html += `<div class="warnbox">${escapeHtml(w)}</div>`;
  });
  if (!data.warnings || !data.warnings.length) {
    html += `<div class="okbox">No warnings: point geometry and residuals look healthy.</div>`;
  }
  $("solveOut").innerHTML = html;

  state.errors = {};
  (data.points || []).forEach((p) => { state.errors[p.name] = p.error_px; });
  renderTable();
}

async function doSolve(save) {
  const body = {
    save,
    loo: true,
    lock_height: $("lockHeight").checked,
    lock_position: $("lockPosition").checked,
    lock_roll: $("lockRoll").checked,
    lock_distortion: $("lockDistortion").checked,
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
    setPending((e.clientX - rect.left) / scale, (e.clientY - rect.top) / scale);
  });

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

  await loadFrame();
  await loadPoints();
  await loadReprojection();

  if (state.reproj) {
    $("showReproj").checked = true;
    drawFrame();
  }
})();
