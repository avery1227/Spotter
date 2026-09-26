/* Live view of the composited output, plus pipeline health.
 *
 * The preview is an MJPEG stream rather than a websocket or repeated polling:
 * an <img> handles it natively, it degrades gracefully when the tab is
 * backgrounded, and the server already rate-limits it well below the real
 * frame rate so the preview never competes with the H.264 encode for CPU.
 */

const $ = (id) => document.getElementById(id);

function toast(msg, kind = "") {
  const el = $("toast");
  el.textContent = msg;
  el.className = "show " + kind;
  clearTimeout(el._t);
  el._t = setTimeout(() => { el.className = ""; }, 3000);
}

function badge(ok, okText, badText, warn) {
  const cls = warn ? "warn" : (ok ? "ok" : "bad");
  return `<span class="badge ${cls}">${ok ? okText : badText}</span>`;
}

function row(label, value) {
  return `<dt>${label}</dt><dd>${value}</dd>`;
}

function setStreamSource() {
  const img = $("liveImg");
  const width = $("quality").value;
  if ($("liveToggle").checked) {
    img.src = `/api/live.mjpg?width=${width}&t=${Date.now()}`;
    $("liveHint").textContent = "Streaming. Uncheck 'live' to freeze.";
  } else {
    img.src = `/api/live.jpg?width=${width}&t=${Date.now()}`;
    $("liveHint").textContent = "Paused — showing a single snapshot.";
  }
}

async function poll() {
  let s;
  try {
    s = await (await fetch("/api/status")).json();
  } catch (e) {
    $("statusGrid").innerHTML = row("server", `<span class="badge bad">unreachable</span>`);
    return;
  }

  if (!s.live) {
    $("statusGrid").innerHTML =
      row("pipeline", `<span class="badge warn">not running</span>`) +
      row("hint", `start it with <code>spotter run --web</code>`);
    $("cullGrid").innerHTML = "";
    $("sources").innerHTML = "";
    $("liveHint").textContent =
      "No pipeline attached — run `spotter run --web` to see live output here.";
    return;
  }

  if (s.waiting_for_calibration) {
    $("statusGrid").innerHTML =
      row("pipeline", `<span class="badge warn">waiting for calibration</span>`) +
      row("next", `open <a href="/">Calibrate</a>; it starts once you save`);
    $("cullGrid").innerHTML = "";
    $("sources").innerHTML = "";
    $("liveHint").innerHTML = "No calibration yet. Open <a href='/'>Calibrate</a>, " +
      "then <b>Solve &amp; save</b>, and the pipeline starts by itself.";
    return;
  }

  const enc = s.output || {};
  const tracks = s.tracks || {};
  const drift = s.drift || {};
  const ingest = s.ingest || {};

  const lagWarn = s.lag_s !== undefined && s.lag_s > 90;
  const fpsOk = (s.pipeline_fps || 0) > (s.target_fps || 30) * 0.85;

  $("statusGrid").innerHTML =
    row("uptime", fmtDuration(s.uptime_s)) +
    row("pipeline fps", `${(s.pipeline_fps || 0).toFixed(1)} ` +
        badge(fpsOk, "ok", "behind")) +
    row("render", `${(s.render_ms || 0).toFixed(1)} ms / frame`) +
    row("frame lag", `${(s.lag_s || 0).toFixed(1)} s ` +
        (lagWarn ? `<span class="badge warn">high</span>` : "")) +
    row("frame time", s.frame_time || "—") +
    row("encoder", `${enc.encoder || "—"} ` +
        badge(enc.alive, "running", "dead")) +
    row("written", `${enc.written || 0} (${enc.repeated || 0} repeated, ` +
        `${enc.dropped || 0} dropped)`) +
    row("audio", enc.audio_kb ? `${enc.audio_kb} kB` : "—") +
    // In offline mode there is no HLS reader at all, so "reconnecting" would
    // be a false alarm rather than a status.
    (s.offline
      ? row("source", `<span class="badge ok">offline clip</span>`)
      : row("ingest", badge(ingest.connected, "connected", "reconnecting") +
            ` ${ingest.reconnects || 0} reconnects`) +
        row("segment age", ingest.last_segment_age_s !== null &&
            ingest.last_segment_age_s !== undefined
            ? `${ingest.last_segment_age_s} s` : "—") +
        row("url expires", ingest.playlist_expires_in_s
            ? fmtDuration(ingest.playlist_expires_in_s) : "—")) +
    row("tracks", `${tracks.aircraft || 0} aircraft · ${tracks.ships || 0} ships`) +
    row("labels drawn", s.visible !== undefined ? s.visible : "—") +
    row("drift", driftCell(drift)) +
    row("decode errors", s.decode_errors || 0);

  const cull = s.cull || {};
  $("cullGrid").innerHTML =
    row("in store", cull.total || 0) +
    row("drawn", cull.visible || 0) +
    row("behind camera", cull.behind || 0) +
    row("off frame", cull.out_of_frame || 0) +
    row("out of range", cull.out_of_range || 0) +
    row("below horizon", cull.below_horizon || 0);

  $("sources").innerHTML = (s.groups || []).map((g) =>
    `<div style="margin-bottom:6px">
       <b>${g.group}</b> → ${g.active || "none"}
       <span class="muted">(priority ${g.priority},
       ${(g.health && g.health.reports) || 0} reports${
         g.health && g.health.last_error
           ? `, last error: ${escapeHtml(String(g.health.last_error)).slice(0, 60)}`
           : ""})</span>
     </div>`).join("") || `<span class="muted">no sources</span>`;

  $("headerInfo").textContent = s.attributions ? s.attributions.join(" · ") : "";
}

/* Over the threshold but not yet confirmed is its own state: it is not "ok",
 * but it is not a flag either -- haze and rain produce transient misses, which
 * is exactly why the detector waits for several consecutive checks. */
function driftCell(drift) {
  if (!drift || !drift.enabled) return `<span class="badge warn">disabled</span>`;
  const px = drift.median_shift_px;
  const shown = px === null || px === undefined ? "—" : `${px} px`;
  if (drift.flagged) return `<span class="badge bad">FLAGGED</span> ${shown}`;
  if (drift.exceeded) {
    return `<span class="badge warn">over ${drift.threshold_px} px</span> ${shown}` +
           ` <span class="muted">(${drift.consecutive_exceeded}/` +
           `${drift.consecutive_needed} checks)</span>`;
  }
  return `<span class="badge ok">ok</span> ${shown}`;
}

function fmtDuration(seconds) {
  if (seconds === null || seconds === undefined) return "—";
  const s = Math.max(0, Math.round(seconds));
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${s % 60}s`;
  return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

$("liveToggle").addEventListener("change", setStreamSource);
$("quality").addEventListener("change", setStreamSource);
$("snapBtn").addEventListener("click", () => {
  const a = document.createElement("a");
  a.href = `/api/live.jpg?width=1920&quality=92&t=${Date.now()}`;
  a.download = `spotter_${new Date().toISOString().replace(/[:.]/g, "-")}.jpg`;
  a.click();
  toast("Snapshot saved");
});

$("liveImg").addEventListener("error", () => {
  $("liveHint").textContent = "No frames yet — is the pipeline running?";
});

setStreamSource();
poll();
setInterval(poll, 2000);
