# Running Spotter on Pelican Panel

Works on Pelican and on Pterodactyl — Pelican reads the same `PTDL_v2` egg
format. Everything below is identical on both; only the menu names differ.

No GPU is needed. Give the server **3 CPU for 1080p**, or set
`STREAM_QUALITY` to `720p` and 2 CPU will do.

---

## Install

**1. Import the egg.**
Admin → Eggs → Import Egg, and upload [`egg-spotter.json`](egg-spotter.json).

**2. Point it at the image.**
The egg ships with a placeholder. Edit the egg's Docker image to:

```
ghcr.io/avery1227/spotter:latest-pelican
```

(The `publish` workflow stamps the real value into the egg artifact on every
release, so the copy attached to a release already has it.)

**3. Create a server** from the egg. One TCP allocation is enough; that port
becomes the web UI. Add a **UDP** allocation as well only if you intend to feed
NMEA AIS in from an AIS-catcher.

**4. Start it in `web` mode first.** Set the `STARTUP_MODE` variable to `web`.
There is no calibration yet, and the pipeline refuses to start without one —
this serves just the UI.

**5. Calibrate.** Open the server's address on its allocated port:

- set the camera position (map click, or paste from Google Maps)
- **Grab fresh frame**, then click 6–12 landmarks and give each a position
- **Solve & save** — this writes `calibration.json`

**6. Switch `STARTUP_MODE` back to `run` and restart.** You now get the
pipeline plus the live monitor at `/monitor` and a full-screen view at `/view`.

---

## What lives where

The application is inside the image at `/opt/spotter` and is read-only.
Everything under `/home/container` is yours and survives an image pull:

| path | |
| --- | --- |
| `config.yaml` | every setting. Written on first boot. |
| `points.csv` | calibration control points, written by the web UI |
| `calibration.json` | the solved camera model |
| `state/` | drift reference patches, and timestamped backups of `points.csv` |
| `data/offline/` | recorded clip and track data for offline test mode |

Upgrading is just pulling a newer image — your calibration is untouched.

---

## Variables

| variable | notes |
| --- | --- |
| `STARTUP_MODE` | `run` (pipeline + UI), `web` (UI only, for first-time calibration), `doctor` (config check, then exits) |
| `STREAM_URL` | the camera's live stream |
| `STREAM_QUALITY` | `best`, or `720p` to roughly halve CPU |
| `ENCODER_DELAY_S` | how far the camera's own pipeline lags reality. See "Tuning encoder_delay_s" in the main README |
| `OUTPUT_ENABLED` | `false` runs overlay-only — useful while calibrating |
| `RTMP_URL` | where to publish. **Contains a stream key; treat as a secret** |
| `VIDEO_ENCODER` | `auto` falls back to software `libx264`, which is what you get without a GPU passed through |
| `VIDEO_BITRATE` | `6000k` for 1080p, `3000k` for 720p |
| `AISSTREAM_API_KEY` | needed for ships unless you have a local AIS receiver |
| `LOG_FORMAT` | `console` reads well in the panel; `json` for log shipping |
| `EXTRA_ARGS` | any `--set key=value` from `config.yaml` |

---

## Things that will catch you out

**The pipeline will not start without `calibration.json`.** This is deliberate
— there is nothing to project through. Use `STARTUP_MODE=web` first. The
console says so on boot.

**No AIS source means no ships, ever.** The UDP listener binds happily and
hears nothing. You need either an AIS-catcher feeding this server's UDP
allocation, or an `AISSTREAM_API_KEY`. Aircraft work out of the box.

**Don't put the web UI on port 8080** if a readsb/tar1090 is on the same host —
that port is theirs by convention, and Spotter's default ADS-B source polls it.

**The web UI has no authentication.** Inside a panel it is exposed on whatever
allocation you gave it. If that is reachable from the internet, put it behind
the panel's proxy with auth, or firewall it.

**CPU.** 1080p is about 2.3 cores of software x264. At 2 cores it manages 20 of
30 fps and falls steadily behind. `STREAM_QUALITY=720p` is the fix.

---

## Checking a deployment

Set `STARTUP_MODE` to `doctor` and start the server. It prints a check of the
config, the calibration, ffmpeg and every track source, then exits — the
quickest way to see why something is not working.
