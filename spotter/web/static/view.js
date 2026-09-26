/* Full-screen wall-display view.
 *
 * Just the composited output, filling the monitor, with chrome that fades away
 * when nobody is touching it. Meant to be left running for weeks, so the two
 * things that matter are: it must recover on its own when the stream drops,
 * and it must not let the display go to sleep.
 *
 * Query parameters (all optional):
 *   ?width=1920   preview width; defaults to the display's own pixel width
 *   ?fps=10       preview frame rate
 *   ?quality=80   JPEG quality
 *   ?stats=0      start with the stats panel hidden
 *   ?fill=1       crop to fill the screen instead of letter-boxing
 */

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);

const settings = {
  // A wall display is usually showing a 1080p source on a 1080p panel, so ask
  // for the panel's real pixel width rather than the 960 the monitor page uses.
  width: parseInt(params.get("width") || "", 10) ||
         Math.min(1920, Math.round(screen.width * (window.devicePixelRatio || 1))),
  fps: parseFloat(params.get("fps") || "") || 10,
  quality: parseInt(params.get("quality") || "", 10) || 80,
};

let retryDelay = 1000;
let lastSequence = -1;
let lastProgressAt = Date.now();

/* ── stream ────────────────────────────────────────────── */

function streamUrl() {
  return `/api/live.mjpg?width=${settings.width}&fps=${settings.fps}` +
         `&quality=${settings.quality}&t=${Date.now()}`;
}

function connect() {
  const img = $("video");
  img.src = streamUrl();
  lastProgressAt = Date.now();
}

function showBanner(msg, sub) {
  $("bannerMsg").textContent = msg;
  $("bannerSub").innerHTML = sub || "";
  $("banner").classList.add("show");
}

function hideBanner() {
  $("banner").classList.remove("show");
}

$("video").addEventListener("error", () => {
  showBanner("Reconnecting…", `retrying in ${Math.round(retryDelay / 1000)}s`);
  setTimeout(() => {
    connect();
    retryDelay = Math.min(retryDelay * 2, 15000);
  }, retryDelay);
});

$("video").addEventListener("load", () => {
  retryDelay = 1000;
  hideBanner();
});

/* ── status ────────────────────────────────────────────── */

function fmtLag(s) {
  if (s === null || s === undefined) return "—";
  return `${s.toFixed(0)}s`;
}

async function poll() {
  let s;
  try {
    s = await (await fetch("/api/status")).json();
  } catch (e) {
    $("sState").innerHTML = `<span class="dot bad"></span>offline`;
    showBanner("Server unreachable", "is the pipeline still running?");
    return;
  }

  if (!s.live) {
    $("sState").innerHTML = `<span class="dot warn"></span>no pipeline`;
    showBanner("Waiting for the pipeline…",
               "start it with <code>spotter run --web</code>");
    return;
  }

  if (s.waiting_for_calibration) {
    $("sState").innerHTML = `<span class="dot warn"></span>no calibration`;
    showBanner("Waiting for calibration",
               "open <a href='/' style='color:#4fc3f7'>Calibrate</a>; " +
               "the pipeline starts once you save");
    return;
  }

  // An MJPEG <img> gives no progress events, so watch the server's own frame
  // counter. If it is advancing but our picture is not, the stream socket died
  // quietly and only a reconnect will fix it.
  const seq = s.preview_sequence;
  if (seq !== lastSequence) {
    lastSequence = seq;
    lastProgressAt = Date.now();
    hideBanner();
  } else if (Date.now() - lastProgressAt > 12000) {
    showBanner("Stream stalled", "reconnecting");
    connect();
    lastProgressAt = Date.now();
  }

  const t = s.tracks || {};
  const stalled = (s.frame_age_s ?? 0) > 5;
  $("sState").innerHTML =
    `<span class="dot ${stalled ? "warn" : ""}"></span>` +
    (s.offline ? "clip" : "live");
  $("sFps").textContent = (s.pipeline_fps || 0).toFixed(1);
  $("sLag").textContent = fmtLag(s.lag_s);
  $("sAir").textContent = t.aircraft ?? "—";
  $("sShips").textContent = t.ships ?? "—";
  $("sLabels").textContent = s.visible ?? "—";
}

/* ── chrome auto-hide ──────────────────────────────────── */

let idleTimer = null;
function wake() {
  document.body.classList.remove("idle");
  clearTimeout(idleTimer);
  idleTimer = setTimeout(() => document.body.classList.add("idle"), 4000);
}
["mousemove", "mousedown", "keydown", "touchstart", "wheel"].forEach((e) =>
  document.addEventListener(e, wake, { passive: true }));

/* ── keep the display awake ────────────────────────────── */

let wakeLock = null;
async function requestWakeLock() {
  if (!("wakeLock" in navigator)) return;
  try {
    wakeLock = await navigator.wakeLock.request("screen");
    wakeLock.addEventListener("release", () => { wakeLock = null; });
  } catch (e) {
    /* denied, or the tab is not visible -- harmless */
  }
}
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible") {
    if (!wakeLock) requestWakeLock();
    // Browsers throttle background tabs hard; the stream is usually dead by
    // the time the tab comes back.
    connect();
  }
});

/* ── keys ──────────────────────────────────────────────── */

document.addEventListener("keydown", (e) => {
  const k = e.key.toLowerCase();
  if (k === "f") {
    if (document.fullscreenElement) document.exitFullscreen();
    else document.documentElement.requestFullscreen().then(requestWakeLock,
                                                           () => {});
  } else if (k === "s") {
    $("stats").classList.toggle("hidden");
  } else if (k === "c") {
    document.body.classList.toggle("fill");
  } else if (k === "r") {
    connect();
  }
});

// Clicking anywhere also goes fullscreen -- handy on a touch panel with no
// keyboard, and browsers only allow it from a user gesture anyway.
$("stage").addEventListener("dblclick", () => {
  if (!document.fullscreenElement) {
    document.documentElement.requestFullscreen().then(requestWakeLock, () => {});
  }
});

/* ── boot ──────────────────────────────────────────────── */

if (params.get("stats") === "0") $("stats").classList.add("hidden");
if (params.get("fill") === "1") document.body.classList.add("fill");

connect();
poll();
setInterval(poll, 2000);
wake();
