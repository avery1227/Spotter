"""Command line entry point: ``python -m spotter``."""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

from .config import ConfigError, load_config, parse_set_overrides
from .logging_setup import get_logger, setup_logging

log = get_logger("spotter.cli")


def cmd_run(args, cfg) -> int:
    from .pipeline import Pipeline, install_signal_handlers

    stop = threading.Event()
    install_signal_handlers(stop)

    hub = None
    if getattr(args, "web", False) or cfg.get("web.enabled", False):
        from .web.frames import FrameHub
        hub = FrameHub(max_width=int(cfg.get("web.preview_width", 960)),
                       quality=int(cfg.get("web.preview_quality", 70)))

    duration = float(getattr(args, "duration", 0.0) or 0.0)
    if duration > 0:
        # A self-imposed deadline, for smoke tests and for checking a config
        # change without having to interrupt the process by hand.
        threading.Timer(duration, stop.set).start()
        log.info("will stop automatically", extra={"after_s": duration})

    pipeline = Pipeline(cfg, stop_event=stop, frame_hub=hub)

    if hub is not None:
        from .web.server import serve
        serve(cfg, hub=hub, pipeline=pipeline,
              host=getattr(args, "web_host", None),
              port=getattr(args, "web_port", None), background=True)

    return pipeline.run()


def cmd_web(args, cfg) -> int:
    """Serve the calibration and monitor UI without running the pipeline."""
    from .web.server import serve

    host = args.host or str(cfg.get("web.host", "127.0.0.1"))
    port = int(args.port or cfg.get("web.port", 8090))
    shown = "localhost" if host in ("0.0.0.0", "::") else host
    print(f"\n  Spotter web UI:  http://{shown}:{port}/\n")
    print("  Calibrate:  click a landmark in the frame, click the same spot on")
    print("              the map (or paste coordinates), name it, add.")
    print("  Monitor:    /monitor  (live output appears once the pipeline runs")
    print("              with --web)\n")
    try:
        serve(cfg, host=host, port=port, background=False)
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


def cmd_doctor(args, cfg) -> int:
    """Check everything the pipeline needs before it is asked to run for real."""
    from .calib.model import CameraModel
    from .output.encoder import detect_encoders, ffmpeg_path

    problems: list[str] = []
    notes: list[str] = []

    def check(label: str, ok: bool, detail: str = "", fatal: bool = True) -> None:
        mark = "ok  " if ok else ("FAIL" if fatal else "warn")
        print(f"  [{mark}] {label}{('  -- ' + detail) if detail else ''}")
        if not ok:
            (problems if fatal else notes).append(f"{label}: {detail}")

    print("\n=== configuration " + "=" * 50)
    check("config loaded", True, str(cfg.root / "config.yaml"))
    check("camera position set",
          bool(cfg.get("camera.lat")) and bool(cfg.get("camera.lon")),
          f"{cfg.get('camera.lat')}, {cfg.get('camera.lon')}")

    print("\n=== calibration " + "=" * 52)
    calib_path = cfg.path("calibration.path", "./calibration.json")
    if calib_path and calib_path.is_file():
        try:
            model = CameraModel.load(calib_path)
            check("calibration.json", True,
                  f"{model.width}x{model.height}, yaw {model.yaw_deg:.1f}deg, "
                  f"hfov {model.hfov_deg:.1f}deg")
            rms = model.meta.get("rms_px")
            if rms is not None:
                check("reprojection RMS", float(rms) < 6.0, f"{rms} px", fatal=False)
            for warning in model.meta.get("warnings", []):
                print(f"         ! {warning}")
        except Exception as exc:
            check("calibration.json", False, str(exc))
    else:
        check("calibration.json", False, f"missing at {calib_path}; run calibrate.py")

    points_path = cfg.path("calibration.points", "./points.csv")
    check("points.csv", bool(points_path and points_path.is_file()),
          str(points_path), fatal=False)

    print("\n=== runtime dependencies " + "=" * 43)
    # Import the heavy native extensions explicitly. They are only pulled in
    # once the pipeline actually starts, so without this a missing shared
    # library (skia needs libEGL, for instance) shows up as a traceback
    # minutes into a run rather than here.
    for module, hint in (
        ("numpy", ""),
        ("scipy", ""),
        ("av", "PyAV: video decode"),
        ("cv2", "OpenCV: calibration window and drift matching"),
        ("skia", "skia-python: overlay rendering (needs libEGL, libGL)"),
        ("pymap3d", ""),
        ("flask", "web UI"),
    ):
        try:
            __import__(module)
            check(module, True, hint, fatal=False)
        except Exception as exc:
            check(module, False, f"{type(exc).__name__}: {exc}")

    print("\n=== ffmpeg " + "=" * 57)
    encoders = detect_encoders()
    check("ffmpeg binary", bool(encoders), ffmpeg_path())
    check("H.264 encoder", bool(encoders), ", ".join(encoders) or "none found")
    check("output.rtmp_url set", bool(cfg.get("output.rtmp_url")),
          "" if cfg.get("output.rtmp_url") else "nothing to publish to")

    print("\n=== drift references " + "=" * 47)
    patches_dir = cfg.path("drift.patches_dir", "./state/drift_patches")
    manifest = patches_dir / "patches.json" if patches_dir else None
    has_patches = bool(manifest and manifest.is_file())
    check("drift patches", has_patches,
          str(patches_dir) if has_patches else "run calibrate.py solve",
          fatal=False)

    print("\n=== track sources " + "=" * 50)
    for group in ("adsb", "ais"):
        if not cfg.get(f"tracks.{group}.enabled", True):
            print(f"  [skip] {group}: disabled")
            continue
        for entry in (cfg.get(f"tracks.{group}.sources") or []):
            name = entry.get("name", entry.get("type"))
            ok, detail = _probe_source(dict(entry), cfg)
            print(f"  [{'ok  ' if ok else 'warn'}] {group}/{name}  -- {detail}")

    print("\n=== summary " + "=" * 56)
    if problems:
        print(f"  {len(problems)} blocking problem(s):")
        for problem in problems:
            print(f"    - {problem}")
    if notes:
        print(f"  {len(notes)} warning(s):")
        for note in notes:
            print(f"    - {note}")
    if not problems and not notes:
        print("  Everything checks out.")
    print()
    return 1 if problems else 0


def _probe_source(entry: dict, cfg) -> tuple[bool, str]:
    """Try one track source once and report what happened."""
    import requests

    from .tracks.adsb import build_adsb_source
    from .tracks.ais import build_ais_source

    source_type = str(entry.get("type", ""))
    if source_type in ("nmea_udp", "udp", "aivdm_udp"):
        return True, f"UDP listener on {entry.get('bind_port')} (passive)"
    if source_type in ("aisstream", "aisstream.io", "websocket"):
        import os
        has_key = bool(entry.get("api_key") or os.environ.get("AISSTREAM_API_KEY"))
        return has_key, "API key present" if has_key else "no API key configured"

    entry.setdefault("timeout_s", 8.0)
    source = (build_adsb_source(entry, float(cfg.get("camera.lat", 0)),
                                float(cfg.get("camera.lon", 0)))
              if "adsb" in source_type or source_type in ("readsb", "dump1090",
                                                          "tar1090", "local",
                                                          "airplanes_live")
              else build_ais_source(entry))
    if source is None:
        return False, "unknown source type"
    try:
        reports = source.poll()
        return True, f"{len(reports)} report(s)"
    except requests.RequestException as exc:
        return False, f"{type(exc).__name__}: {str(exc)[:70]}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {str(exc)[:70]}"


def cmd_record(args, cfg) -> int:
    """Record live track reports to JSONL for offline mode."""
    from .offline import dump_report
    from .tracks.manager import TrackManager
    from .tracks.store import TrackStore

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen: set[tuple] = set()
    handle = out_path.open("w", encoding="utf-8")
    written = 0

    class RecordingStore(TrackStore):
        def add_report(self, report):
            nonlocal written
            if not super().add_report(report):
                return False
            # Feeds repeat the same report between polls; only write changes.
            key = (report.track_id, report.timestamp.timestamp(),
                   round(report.lat, 6), round(report.lon, 6))
            if key in seen:
                return True
            seen.add(key)
            handle.write(dump_report(report) + "\n")
            written += 1
            return True

    manager = TrackManager(cfg, store=RecordingStore(cfg)).start()
    print(f"Recording track data to {out_path} for {args.duration}s "
          f"(Ctrl-C to stop early) ...")
    deadline = time.monotonic() + args.duration
    try:
        while time.monotonic() < deadline:
            time.sleep(2.0)
            handle.flush()
            print(f"  {written} reports, {len(manager.store)} tracks", end="\r")
    except KeyboardInterrupt:
        print("\n  interrupted")
    finally:
        manager.stop()
        handle.flush()
        handle.close()

    print(f"\nWrote {written} reports to {out_path}")
    print("Set offline.tracks to this file and offline.enabled: true to replay it.")
    return 0


def cmd_clip(args, cfg) -> int:
    """Record a short clip of the live stream for offline mode."""
    import av

    from .ingest.decoder import SegmentDecoder
    from .ingest.hls import HLSReader

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    reader = HLSReader(cfg).start()
    decoder = SegmentDecoder(cfg, audio_sink=None)
    container = None
    stream = None
    frames = 0
    first_time = None
    deadline = time.monotonic() + args.duration + 120

    try:
        for payload in reader.segments(timeout=2.0):
            if payload is None:
                if time.monotonic() > deadline:
                    break
                continue
            for event in decoder.decode(payload):
                if container is None:
                    container = av.open(str(out_path), mode="w")
                    stream = container.add_stream("libx264",
                                                  rate=int(round(decoder.info.fps)))
                    stream.width = event.width
                    stream.height = event.height
                    stream.pix_fmt = "yuv420p"
                    stream.options = {"preset": "veryfast", "crf": "20"}
                    first_time = event.wall_time
                    print(f"Recording {event.width}x{event.height} @ "
                          f"{decoder.info.fps:.2f}fps ...")

                frame = av.VideoFrame.from_ndarray(event.image, format="bgra")
                for packet in stream.encode(frame):
                    container.mux(packet)
                frames += 1
                if frames % 30 == 0:
                    print(f"  {frames / decoder.info.fps:.1f}s", end="\r")
                if frames >= args.duration * decoder.info.fps:
                    raise StopIteration
    except StopIteration:
        pass
    except KeyboardInterrupt:
        print("\n  interrupted")
    finally:
        reader.stop()
        if container is not None:
            for packet in stream.encode():
                container.mux(packet)
            container.close()

    if frames == 0:
        print("ERROR: no frames captured", file=sys.stderr)
        return 1
    print(f"\nWrote {out_path}  ({frames} frames)")
    if first_time is not None:
        # The clip's own timestamps are relative; offline mode needs to know
        # what instant frame zero corresponds to in order to line up tracks.
        print(f"Set offline.start_time: \"{first_time.isoformat()}\"")
    return 0


def _add_global_options(parser: argparse.ArgumentParser, top_level: bool) -> None:
    """Attach the global options to a parser.

    They go on the top-level parser *and* on every subcommand, so both
    ``spotter --set x=1 run`` and ``spotter run --set x=1`` work. Argparse
    parses a subcommand into a fresh namespace and copies the result over the
    parent's, so the subcommand copies use distinct dests and :func:`main`
    merges them; otherwise flags given before the subcommand would be silently
    dropped when the same flag also appears after it.
    """
    suffix = "" if top_level else "_sub"
    parser.add_argument("-c", "--config", default=None, dest=f"config{suffix}",
                        metavar="PATH", help="path to config.yaml")
    parser.add_argument("--set", action="append", default=[], dest=f"set{suffix}",
                        metavar="KEY=VALUE",
                        help="override a config value, e.g. --set output.enabled=false")
    parser.add_argument("--log-level", default=None, dest=f"log_level{suffix}",
                        metavar="LEVEL", help="DEBUG | INFO | WARNING | ERROR")
    parser.add_argument("--log-format", default=None, dest=f"log_format{suffix}",
                        metavar="FORMAT", choices=["json", "console"],
                        help="json | console")


def merge_global_options(args: argparse.Namespace) -> argparse.Namespace:
    """Fold the subcommand's copies of the global options into the main ones."""
    args.set = list(getattr(args, "set", []) or []) +         list(getattr(args, "set_sub", []) or [])
    for name in ("config", "log_level", "log_format"):
        sub_value = getattr(args, f"{name}_sub", None)
        if sub_value is not None:
            setattr(args, name, sub_value)
    return args


def build_parser() -> argparse.ArgumentParser:
    # The global options are attached to the top-level parser *and* to every
    # subcommand, so both `spotter --set x=1 run` and `spotter run --set x=1`
    # work. Argparse otherwise only accepts them before the subcommand, which
    # is not where anyone types them.
    parser = argparse.ArgumentParser(
        prog="spotter",
        description="AR ship/aircraft overlay for a fixed shoreline camera")
    _add_global_options(parser, top_level=True)
    sub = parser.add_subparsers(dest="command")

    def add(name: str, help_text: str):
        child = sub.add_parser(name, help=help_text)
        _add_global_options(child, top_level=False)
        return child

    p_run = add("run", "run the overlay pipeline (default)")
    p_run.add_argument("-d", "--duration", type=float, default=0.0,
                       help="stop after this many seconds (0 = run forever)")
    p_run.add_argument("--web", action="store_true",
                       help="also serve the web UI, with a live monitor")
    p_run.add_argument("--web-host", default=None, metavar="HOST")
    p_run.add_argument("--web-port", type=int, default=None, metavar="PORT")
    p_run.set_defaults(func=cmd_run)

    p_web = add("web", "serve the calibration and monitor UI")
    p_web.add_argument("--host", default=None, metavar="HOST")
    p_web.add_argument("--port", type=int, default=None, metavar="PORT")
    p_web.set_defaults(func=cmd_web)
    add("doctor", "check configuration and dependencies").set_defaults(func=cmd_doctor)

    p_record = add("record-tracks", "record live track data for offline mode")
    p_record.add_argument("-o", "--output", default="./data/offline/tracks.jsonl")
    p_record.add_argument("-d", "--duration", type=float, default=120.0)
    p_record.set_defaults(func=cmd_record)

    p_clip = add("record-clip", "record a clip of the live stream for offline mode")
    p_clip.add_argument("-o", "--output", default="./data/offline/clip.mp4")
    p_clip.add_argument("-d", "--duration", type=float, default=60.0)
    p_clip.set_defaults(func=cmd_clip)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = merge_global_options(parser.parse_args(argv))
    if not getattr(args, "func", None):
        args.func = cmd_run

    try:
        cfg = load_config(args.config, overrides=parse_set_overrides(args.set))
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    setup_logging(cfg, level=args.log_level, fmt=args.log_format)

    from .compat import apply_all
    apply_all()

    return args.func(args, cfg)


if __name__ == "__main__":
    sys.exit(main())
