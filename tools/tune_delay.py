#!/usr/bin/env python3
"""Find the right ``stream.encoder_delay_s`` by trying several at once.

The manual method is to watch a plane cross the frame and nudge the delay until
its label sits on it. That works, but each guess costs a restart and another
plane. This does the whole sweep on a single frame:

1. Collect track history for a while, so every candidate delay has real reports
   bracketing it (no extrapolation, no guessing).
2. Grab one frame and note its PDT-derived capture time.
3. For each candidate delay, resolve every target's position at
   ``capture_time - delay`` and draw a marker, colour-coded by delay.

The delay whose markers land on the actual aircraft and ships in the image is
the one to put in ``config.yaml``. The markers form a track across the frame,
so you can read off the answer rather than bisect towards it.

    python tools/tune_delay.py --min 4 --max 20 --step 2 --collect 90
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spotter.calib.model import CameraModel                       # noqa: E402
from spotter.config import ConfigError, load_config, parse_set_overrides  # noqa: E402
from spotter.ingest.decoder import SegmentDecoder                 # noqa: E402
from spotter.ingest.grab import save_frame                        # noqa: E402
from spotter.ingest.hls import HLSReader                          # noqa: E402
from spotter.logging_setup import setup_logging                   # noqa: E402
from spotter.projection import TargetProjector                    # noqa: E402
from spotter.tracks.manager import TrackManager                   # noqa: E402
from spotter.tracks.model import TrackKind                        # noqa: E402
from spotter.util import utcnow                                   # noqa: E402

# Distinct hues so overlapping candidates stay tellable apart (BGR for OpenCV).
PALETTE = [
    (80, 80, 255), (60, 160, 255), (40, 220, 255), (80, 240, 160),
    (200, 220, 80), (255, 180, 60), (255, 110, 110), (230, 90, 220),
    (160, 120, 255), (120, 255, 120),
]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-c", "--config", default=None)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--min", type=float, default=4.0,
                        help="smallest delay to try, seconds")
    parser.add_argument("--max", type=float, default=20.0,
                        help="largest delay to try, seconds")
    parser.add_argument("--step", type=float, default=2.0)
    parser.add_argument("--collect", type=float, default=90.0,
                        help="seconds of track history to gather first")
    parser.add_argument("-o", "--output", default="delay_sweep.png")
    parser.add_argument("--kind", choices=["aircraft", "ship", "both"],
                        default="aircraft",
                        help="aircraft move fastest, so they resolve the delay best")
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config, overrides=parse_set_overrides(args.set))
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    setup_logging(cfg, level="WARNING", fmt="console")

    calib_path = cfg.path("calibration.path", "./calibration.json")
    if not calib_path or not calib_path.is_file():
        print(f"ERROR: no calibration at {calib_path}. Run calibrate.py first.",
              file=sys.stderr)
        return 1
    model = CameraModel.load(calib_path)

    delays = list(np.arange(args.min, args.max + 1e-9, args.step))
    if not delays:
        print("ERROR: empty delay range", file=sys.stderr)
        return 1

    # 1. Gather history. Every candidate must be able to interpolate, so we
    #    need reports spanning at least the widest delay plus a margin.
    print(f"Collecting track data for {args.collect:.0f}s ...")
    tracks = TrackManager(cfg).start()
    try:
        deadline = time.monotonic() + args.collect
        while time.monotonic() < deadline:
            time.sleep(2.0)
            counts = tracks.store.counts()
            print(f"  {counts['total']} tracks, {counts['reports']} reports",
                  end="\r")
        print()

        # 2. Grab a frame, with the delay removed so we hold the raw capture time.
        print("Grabbing a frame ...")
        raw_cfg = load_config(args.config, overrides={
            **parse_set_overrides(args.set), "stream": {"encoder_delay_s": 0.0}})
        reader = HLSReader(raw_cfg, queue_size=2).start()
        decoder = SegmentDecoder(raw_cfg, audio_sink=None)
        event = None
        try:
            for payload in reader.segments(timeout=2.0):
                if payload is None:
                    continue
                for event in decoder.decode(payload):
                    break
                if event is not None:
                    break
        finally:
            reader.stop()

        if event is None:
            print("ERROR: could not grab a frame", file=sys.stderr)
            return 1

        capture_time = event.wall_time
        print(f"Frame captured at {capture_time.isoformat()} "
              f"({(utcnow() - capture_time).total_seconds():.1f}s ago)")

        if (model.width, model.height) != (event.width, event.height):
            model_scaled = model.scaled_to(event.width, event.height)
        else:
            model_scaled = model

        # 3. Sweep.
        import cv2

        canvas = event.image[:, :, :3].copy()
        projector = TargetProjector(cfg, model_scaled)
        wanted = ({TrackKind.AIRCRAFT} if args.kind == "aircraft"
                  else {TrackKind.SHIP} if args.kind == "ship"
                  else {TrackKind.AIRCRAFT, TrackKind.SHIP})

        # The oldest report we hold bounds how far back a candidate can reach.
        # Without this check a delay that simply predates our history looks
        # identical to one that is wrong.
        oldest = None
        for track in list(tracks.store._tracks.values()):
            if track.first_seen is not None:
                oldest = track.first_seen if oldest is None else min(
                    oldest, track.first_seen)

        per_delay: dict[float, list] = {}
        unreachable: list[float] = []
        for index, delay in enumerate(delays):
            when = capture_time - timedelta(seconds=float(delay))
            if oldest is not None and when < oldest:
                unreachable.append(float(delay))
            states = [s for s in tracks.store.snapshot_at(when) if s.kind in wanted]
            targets = projector.project(states)
            per_delay[float(delay)] = targets

            color = PALETTE[index % len(PALETTE)]
            for target in targets:
                point = (int(round(target.u)), int(round(target.v)))
                cv2.circle(canvas, point, 6, color, 2, cv2.LINE_AA)
                cv2.putText(canvas, f"{delay:.0f}", (point[0] + 8, point[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(canvas, f"{delay:.0f}", (point[0] + 8, point[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

        # Join each target's markers so the sweep reads as a line per target.
        ids = {t.id for targets in per_delay.values() for t in targets}
        for track_id in ids:
            path = []
            for delay in delays:
                for target in per_delay[float(delay)]:
                    if target.id == track_id:
                        path.append((int(round(target.u)), int(round(target.v))))
            for a, b in zip(path, path[1:]):
                cv2.line(canvas, a, b, (255, 255, 255), 1, cv2.LINE_AA)
            if path:
                label = next(t for targets in per_delay.values()
                             for t in targets if t.id == track_id)
                name = (label.labels.get("callsign") or label.labels.get("name")
                        or track_id)
                cv2.putText(canvas, str(name), (path[0][0] + 10, path[0][1] + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(canvas, str(name), (path[0][0] + 10, path[0][1] + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                            cv2.LINE_AA)

        legend = ["encoder_delay_s sweep - pick the number sitting on the target",
                  f"frame captured {capture_time.strftime('%H:%M:%S')} UTC",
                  f"delays: {', '.join(f'{d:.0f}' for d in delays)} s"]
        cv2.rectangle(canvas, (0, 0), (760, 20 + 22 * len(legend)), (0, 0, 0), -1)
        for i, line in enumerate(legend):
            cv2.putText(canvas, line, (10, 26 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (240, 240, 240), 1, cv2.LINE_AA)

        out = Path(args.output)
        save_frame(np.dstack([canvas,
                              np.full(canvas.shape[:2], 255, np.uint8)]), out)
        print(f"\nWrote {out}")
        for delay in delays:
            note = ""
            if float(delay) in unreachable:
                note = "  (before our track history starts - collect for longer)"
            print(f"  {delay:5.1f}s -> {len(per_delay[float(delay)])} target(s) "
                  f"in frame{note}")
        if unreachable:
            need = max(unreachable) + (utcnow() - capture_time).total_seconds() + 30
            print(f"\nSome candidates reach further back than the data we "
                  f"collected.\nRe-run with --collect {need:.0f} or more to "
                  f"cover them.")
        print("\nOpen the image, find a moving target you can actually see, and "
              "read off\nthe delay whose circle sits on it. Put that number in "
              "config.yaml as\nstream.encoder_delay_s.")
        return 0
    finally:
        tracks.stop()


if __name__ == "__main__":
    sys.exit(main())
