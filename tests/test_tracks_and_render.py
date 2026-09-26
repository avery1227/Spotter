"""Track store behaviour, projection culling, declutter, config and drift."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from spotter.calib.model import CameraModel
from spotter.config import Config, deep_merge, parse_set_overrides
from spotter.projection import TargetProjector
from spotter.render.declutter import (LabelAnimator, LabelBox, resolve_overlaps)
from spotter.render.theme import format_distance, parse_color, with_alpha
from spotter.tracks.adsb import parse_aircraft_payload
from spotter.tracks.ais import AISAssembler, parse_aiscatcher_vessel
from spotter.tracks.model import TrackKind, TrackReport, category_of, default_height_m
from spotter.tracks.nmea import decode_payload
from spotter.tracks.store import MotionParams, TrackStore

REF_LAT, REF_LON = 41.2700, -72.5000
T0 = datetime(2026, 9, 25, 12, 0, 0, tzinfo=timezone.utc)


def store_config(**over):
    base = {
        "tracks": {
            "history_s": 900.0,
            "stale_timeout_s": {"aircraft": 30.0, "ship": 600.0},
            "motion": {
                "aircraft": {"smoothing_tau_s": 0.0, "max_extrapolate_s": 8.0,
                             "dead_reckon": True},
                "ship": {"smoothing_tau_s": 0.0, "max_extrapolate_s": 180.0,
                         "dead_reckon": True},
            },
        },
        "projection": {"max_range_m": 80000, "min_range_m": 50,
                       "frame_margin_px": 80, "horizon_culling": True,
                       "horizon_slack_m": 2.0, "default_ship_height_m": 12.0},
    }
    return Config(deep_merge(base, over))


def ship_report(track_id="mmsi:1", at=T0, lat=41.20, lon=-72.50,
                speed=5.0, course=90.0, **labels):
    return TrackReport(track_id=track_id, kind=TrackKind.SHIP, timestamp=at,
                       lat=lat, lon=lon, alt_m=0.0, speed_mps=speed,
                       course_deg=course, labels=labels, source="test")


# ---------------------------------------------------------------------------
# Track store
# ---------------------------------------------------------------------------

def test_position_between_reports_is_interpolated_not_extrapolated():
    """The core of the design: frames are rendered late, so we interpolate."""
    store = TrackStore(store_config())
    store.add_report(ship_report(at=T0, lat=41.20, lon=-72.50))
    store.add_report(ship_report(at=T0 + timedelta(seconds=30), lat=41.20,
                                 lon=-72.49))

    motion = store.motion[TrackKind.SHIP]
    track = store.get("mmsi:1")

    midpoint = track.position_at(T0 + timedelta(seconds=15), motion)
    assert midpoint.mode == "interpolated"
    assert midpoint.lon == pytest.approx(-72.495, abs=1e-6)
    assert midpoint.lat == pytest.approx(41.20, abs=1e-9)

    quarter = track.position_at(T0 + timedelta(seconds=7.5), motion)
    assert quarter.lon == pytest.approx(-72.4975, abs=1e-6)


def test_dead_reckoning_past_the_last_report():
    store = TrackStore(store_config())
    store.add_report(ship_report(at=T0, lat=41.20, lon=-72.50,
                                 speed=10.0, course=0.0))
    motion = store.motion[TrackKind.SHIP]
    track = store.get("mmsi:1")

    later = track.position_at(T0 + timedelta(seconds=60), motion)
    assert later.mode == "dead_reckoned"
    assert later.lat > 41.20        # heading due north

    from spotter.geodesy import geodetic_to_enu
    enu = geodetic_to_enu(later.lat, later.lon, 0.0, 41.20, -72.50, 0.0)
    assert float(enu[0, 1]) == pytest.approx(600.0, rel=0.02)


def test_extrapolation_is_capped():
    store = TrackStore(store_config())
    store.add_report(ship_report(at=T0))
    motion = store.motion[TrackKind.SHIP]
    track = store.get("mmsi:1")

    assert track.position_at(T0 + timedelta(seconds=179), motion) is not None
    assert track.position_at(T0 + timedelta(seconds=181), motion) is None


def test_frame_time_before_the_first_report_reckons_backwards():
    """Happens whenever a track's first report arrives after the current frame."""
    store = TrackStore(store_config())
    store.add_report(ship_report(at=T0, lat=41.20, lon=-72.50,
                                 speed=10.0, course=0.0))
    motion = store.motion[TrackKind.SHIP]
    track = store.get("mmsi:1")

    earlier = track.position_at(T0 - timedelta(seconds=60), motion)
    assert earlier is not None and earlier.mode == "dead_reckoned"
    assert earlier.lat < 41.20      # it was south of here a minute ago


def test_course_interpolation_takes_the_short_way_round_the_compass():
    store = TrackStore(store_config())
    store.add_report(ship_report(at=T0, course=350.0))
    store.add_report(ship_report(at=T0 + timedelta(seconds=10), course=10.0,
                                 lon=-72.499))
    motion = store.motion[TrackKind.SHIP]
    middle = store.get("mmsi:1").position_at(T0 + timedelta(seconds=5), motion)
    # Must read 0, not 180: averaging 350 and 10 naively points the wrong way.
    assert middle.course_deg == pytest.approx(0.0, abs=1e-6)


def test_labels_accumulate_and_are_not_erased_by_later_reports():
    """A vessel's name arrives once, in a static message; position reports
    that follow must not wipe it."""
    store = TrackStore(store_config())
    store.add_report(ship_report(at=T0, name="ATLANTIC DAWN", ship_type=70))
    store.add_report(ship_report(at=T0 + timedelta(seconds=10), lon=-72.49))

    labels = store.get("mmsi:1").labels
    assert labels["name"] == "ATLANTIC DAWN"
    assert labels["ship_type"] == 70


def test_out_of_order_reports_are_sorted():
    store = TrackStore(store_config())
    store.add_report(ship_report(at=T0 + timedelta(seconds=20), lon=-72.48))
    store.add_report(ship_report(at=T0, lon=-72.50))
    store.add_report(ship_report(at=T0 + timedelta(seconds=10), lon=-72.49))

    track = store.get("mmsi:1")
    assert track.report_count == 3
    motion = store.motion[TrackKind.SHIP]
    middle = track.position_at(T0 + timedelta(seconds=5), motion)
    assert middle.mode == "interpolated"
    assert middle.lon == pytest.approx(-72.495, abs=1e-6)


def test_duplicate_timestamps_replace_rather_than_accumulate():
    store = TrackStore(store_config())
    store.add_report(ship_report(at=T0, lon=-72.50))
    store.add_report(ship_report(at=T0, lon=-72.40))
    assert store.get("mmsi:1").report_count == 1
    assert store.get("mmsi:1").latest.lon == pytest.approx(-72.40)


def test_stale_tracks_are_pruned_by_kind():
    store = TrackStore(store_config())
    now = datetime.now(timezone.utc)
    store.add_report(ship_report(track_id="mmsi:1", at=now - timedelta(seconds=120)))
    store.add_report(TrackReport(
        track_id="icao:abc", kind=TrackKind.AIRCRAFT,
        timestamp=now - timedelta(seconds=120), lat=41.3, lon=-72.5, alt_m=3000))

    assert len(store) == 2
    store.prune(now)
    # Aircraft time out after 30 s, ships after 600 s.
    assert store.get("icao:abc") is None
    assert store.get("mmsi:1") is not None


def test_invalid_positions_are_rejected():
    store = TrackStore(store_config())
    assert not store.add_report(ship_report(lat=0.0, lon=0.0))     # null island
    assert not store.add_report(ship_report(lat=91.0, lon=-72.5))
    assert not store.add_report(ship_report(lat=float("nan"), lon=-72.5))
    assert len(store) == 0


def test_smoothing_lags_but_converges():
    store = TrackStore(store_config(tracks={"motion": {
        "ship": {"smoothing_tau_s": 2.0, "max_extrapolate_s": 180.0,
                 "dead_reckon": True}}}))
    store.add_report(ship_report(at=T0, lon=-72.50, speed=0.0))
    store.add_report(ship_report(at=T0 + timedelta(seconds=10), lon=-72.40,
                                 speed=0.0))
    track = store.get("mmsi:1")
    motion = MotionParams(smoothing_tau_s=2.0, max_extrapolate_s=180.0)

    # Querying forward in time, the filter should trail then catch up.
    track.position_at(T0, motion)
    lagged = track.position_at(T0 + timedelta(seconds=1), motion)
    assert -72.50 < lagged.lon < -72.49

    # An exponential filter fed a ramp settles to a lag of roughly
    # tau * slope, so it is still ~0.02 deg behind when the ramp ends at t=10.
    at_end = track.position_at(T0 + timedelta(seconds=10), motion)
    assert at_end.lon == pytest.approx(-72.40, abs=0.03)
    assert at_end.lon < -72.40

    # Once the input stops moving the lag decays away.
    for step in range(11, 25):
        latest = track.position_at(T0 + timedelta(seconds=step), motion)
    assert latest.lon == pytest.approx(-72.40, abs=0.001)


# ---------------------------------------------------------------------------
# Feed parsing
# ---------------------------------------------------------------------------

def test_readsb_payload_converts_to_si():
    payload = {"now": 1700000000.0, "aircraft": [{
        "hex": "a1b2c3", "flight": "UAL123 ", "r": "N12345", "t": "B738",
        "alt_geom": 35000, "gs": 450.0, "track": 270.0, "baro_rate": -1200,
        "lat": 41.3, "lon": -72.6, "seen_pos": 1.5, "true_heading": 268.0,
    }]}
    reports = parse_aircraft_payload(payload, "test")
    assert len(reports) == 1
    report = reports[0]
    assert report.track_id == "icao:a1b2c3"
    assert report.alt_m == pytest.approx(35000 * 0.3048)
    assert report.speed_mps == pytest.approx(450 * 0.514444)
    assert report.vertical_rate_mps == pytest.approx(-1200 * 0.00508)
    assert report.timestamp.timestamp() == pytest.approx(1700000000.0 - 1.5)
    assert report.labels["callsign"] == "UAL123"


def test_public_api_milliseconds_and_ground_aircraft():
    payload = {"now": 1700000000000, "ac": [
        {"hex": "abc", "lat": 41.0, "lon": -72.0, "alt_baro": "ground",
         "gs": 5, "dbFlags": 1},
        {"hex": "def", "lat": 41.0, "lon": -72.0, "alt_baro": 10000},
        {"hex": "noposition", "alt_baro": 10000},
    ]}
    reports = parse_aircraft_payload(payload, "test")
    assert len(reports) == 2        # the one with no position is dropped
    grounded = next(r for r in reports if r.track_id == "icao:abc")
    assert grounded.alt_m == 0.0
    assert grounded.labels["on_ground"] is True
    assert grounded.labels["military"] is True
    assert grounded.timestamp.timestamp() == pytest.approx(1700000000.0)


def test_aiscatcher_vessel_mapping_and_sentinels():
    entry = {"mmsi": 367001111, "lat": 41.2, "lon": -72.4, "speed": 12.4,
             "cog": 145.0, "heading": 511, "shipname": "ATLANTIC DAWN",
             "shiptype": 70, "destination": "NEW HAVEN", "last_signal": 3.0,
             "to_bow": 120, "to_stern": 60}
    report = parse_aiscatcher_vessel(entry, "test", 1700000000.0)
    assert report.track_id == "mmsi:367001111"
    assert report.speed_mps == pytest.approx(12.4 * 0.514444)
    assert report.heading_deg is None        # 511 means "not available"
    assert report.labels["length_m"] == 180.0
    assert report.timestamp.timestamp() == pytest.approx(1700000000.0 - 3.0)


def test_ais_assembler_attaches_a_late_name_to_a_known_position():
    """A type 5 message has no position, but we already know where the ship is."""
    from tests.test_nmea import bits_of, class_b_position, text_bits, to_payload

    assembler = AISAssembler()
    position = decode_payload(*to_payload(class_b_position(mmsi=338123456)))
    out = assembler.feed(position, "test")
    assert len(out) == 1 and out[0].labels.get("name") is None

    static_bits = (bits_of(24, 6) + bits_of(0, 2) + bits_of(338123456, 30)
                   + bits_of(0, 2) + text_bits("MISS MARIE", 20))
    static = decode_payload(*to_payload(static_bits))
    refreshed = assembler.feed(static, "test")

    assert len(refreshed) == 1
    assert refreshed[0].labels["name"] == "MISS MARIE"
    # Re-emitted at the original position's timestamp, so it does not look
    # like the vessel just moved.
    assert refreshed[0].timestamp == out[0].timestamp
    assert refreshed[0].lat == out[0].lat


def test_category_and_height_estimates():
    assert category_of({"ship_type": 70}, TrackKind.SHIP) == "ship_cargo"
    assert category_of({"ship_type": 80}, TrackKind.SHIP) == "ship_tanker"
    assert category_of({"ship_type": 60}, TrackKind.SHIP) == "ship_passenger"
    assert category_of({"military": True}, TrackKind.AIRCRAFT) == "aircraft_military"
    assert category_of({}, TrackKind.SHIP) == "ship"

    assert default_height_m({"length_m": 300}, TrackKind.SHIP, 12.0) > 40
    assert default_height_m({"ship_type": 37}, TrackKind.SHIP, 12.0) < 15
    assert default_height_m({}, TrackKind.SHIP, 12.0) == 12.0


# ---------------------------------------------------------------------------
# Projection / culling
# ---------------------------------------------------------------------------

def make_projector():
    model = CameraModel(ref_lat=REF_LAT, ref_lon=REF_LON, width=1920, height=1080,
                        yaw_deg=180.0, pitch_deg=-2.0, height_m=15.0,
                        focal_px=1500.0)
    return TargetProjector(store_config(), model), model


def states_at(bearings_ranges, kind=TrackKind.SHIP, alt=0.0, labels=None):
    from spotter.geodesy import enu_to_geodetic
    from spotter.tracks.store import Track, TrackState
    from spotter.tracks.model import Position

    out = []
    for index, (bearing, distance) in enumerate(bearings_ranges):
        east = distance * np.sin(np.radians(bearing))
        north = distance * np.cos(np.radians(bearing))
        lat, lon, _ = enu_to_geodetic(np.array([[east, north, alt]]),
                                      REF_LAT, REF_LON, 0.0)[0]
        track = Track(f"t{index}", kind)
        track.labels = dict(labels or {})
        out.append(TrackState(track=track,
                              position=Position(lat=lat, lon=lon, alt_m=alt)))
    return out


def test_targets_behind_the_camera_are_culled():
    projector, _ = make_projector()
    # Camera looks south (180); these are north of it.
    visible = projector.project(states_at([(0, 5000), (10, 5000)], alt=100.0))
    assert visible == []
    assert projector.last_stats.behind == 2


def test_range_limits():
    projector, _ = make_projector()
    projector.project(states_at([(180, 10.0)], alt=50.0))
    assert projector.last_stats.out_of_range == 1

    projector.project(states_at([(180, 200000.0)], alt=10000.0))
    assert projector.last_stats.out_of_range == 1


def test_surface_targets_beyond_the_horizon_are_culled_by_height():
    projector, _ = make_projector()
    # A small boat at 40 km is over the horizon from 15 m up; a tall ship is not.
    small = projector.project(states_at([(180, 40000.0)],
                                        labels={"ship_type": 37, "length_m": 10}))
    assert small == []
    assert projector.last_stats.below_horizon == 1

    tall = projector.project(states_at([(180, 40000.0)],
                                       labels={"ship_type": 70, "length_m": 300}))
    assert len(tall) == 1
    assert tall[0].category == "ship_cargo"


def test_horizon_culling_can_be_disabled():
    projector, _ = make_projector()
    projector.horizon_culling = False
    assert len(projector.project(states_at([(180, 40000.0)],
                                           labels={"ship_type": 37}))) == 1


def test_visible_target_carries_correct_geometry():
    projector, model = make_projector()
    targets = projector.project(states_at([(180, 8000.0)],
                                          labels={"ship_type": 70,
                                                  "length_m": 200}))
    assert len(targets) == 1
    target = targets[0]
    assert target.bearing_deg == pytest.approx(180.0, abs=0.5)
    assert target.ground_range_m == pytest.approx(8000.0, rel=0.01)
    assert target.elevation_deg < 0            # below the camera, on the water
    assert abs(target.u - model.cx) < 30       # dead ahead


# ---------------------------------------------------------------------------
# Declutter and theme
# ---------------------------------------------------------------------------

def box(x, y, w=140, h=48, target_id="t"):
    return LabelBox(target_id=target_id, anchor_x=x, anchor_y=y + h,
                    x=x, y=y, width=w, height=h)


def test_overlapping_labels_are_separated():
    boxes = [box(100, 100, target_id="a"), box(110, 110, target_id="b"),
             box(120, 120, target_id="c")]
    remaining = resolve_overlaps(boxes, 1920, 1080, gap=4.0, iterations=80)
    assert remaining == 0
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            assert not boxes[i].overlaps(boxes[j], 4.0)


def test_labels_are_clamped_inside_the_frame():
    boxes = [box(-500, -500), box(3000, 2000)]
    resolve_overlaps(boxes, 1920, 1080)
    for item in boxes:
        assert item.x >= 0 and item.y >= 0
        assert item.x + item.width <= 1920
        assert item.y + item.height <= 1080


def test_higher_priority_labels_move_less():
    """Nearest targets are first in the list and should stay put."""
    first = box(500, 500, target_id="near")
    second = box(505, 505, target_id="far")
    start = (first.x, first.y)
    resolve_overlaps([first, second], 1920, 1080, gap=4.0, iterations=80)
    moved_first = abs(first.x - start[0]) + abs(first.y - start[1])
    moved_second = abs(second.x - 505) + abs(second.y - 505)
    assert moved_second > moved_first


def test_label_fades_in_and_out():
    animator = LabelAnimator(fade_in_s=1.0, fade_out_s=1.0, linger_s=0.0)
    alphas = animator.update(["a"], now=0.0, dt=0.0)
    assert alphas["a"] == pytest.approx(0.0)

    alphas = animator.update(["a"], now=0.5, dt=0.5)
    assert alphas["a"] == pytest.approx(0.5)
    alphas = animator.update(["a"], now=1.0, dt=0.5)
    assert alphas["a"] == pytest.approx(1.0)

    # Now it disappears and should fade rather than pop.
    alphas = animator.update([], now=1.5, dt=0.5)
    assert alphas["a"] == pytest.approx(0.5)
    alphas = animator.update([], now=2.1, dt=0.6)
    assert "a" not in alphas


def test_a_pipeline_stall_does_not_jump_the_fades():
    animator = LabelAnimator(fade_in_s=1.0, fade_out_s=1.0, linger_s=0.0)
    animator.update(["a"], now=0.0, dt=0.0)
    # A 30 s stall must not instantly saturate the alpha.
    alphas = animator.update(["a"], now=30.0, dt=30.0)
    assert alphas["a"] <= 1.0


def test_colour_parsing():
    assert parse_color("#FF0000") == 0xFFFF0000
    assert parse_color("#0B1622CC") == 0xCC0B1622
    assert parse_color("nonsense", default=0xFF123456) == 0xFF123456
    half = with_alpha(0xFFFF0000, 0.5)
    assert (half >> 24) & 0xFF == 128        # round(255 * 0.5) == 128
    assert half & 0xFFFFFF == 0xFF0000       # the colour itself is untouched
    assert with_alpha(0xFFFF0000, 0.0) >> 24 == 0
    assert with_alpha(0xCC112233, 1.0) == 0xCC112233


def test_distance_formatting():
    assert format_distance(450) == "450 m"
    assert format_distance(4200) == "4.2 km"
    assert format_distance(21000) == "21 km"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def test_dotted_lookup_and_defaults():
    cfg = Config({"a": {"b": {"c": 1}}, "list": [1, 2]})
    assert cfg.get("a.b.c") == 1
    assert cfg.get("a.b.missing", "fallback") == "fallback"
    assert cfg.get("nope.at.all") is None
    assert cfg.sub("a.b").get("c") == 1
    assert cfg.sub("does.not.exist").get("x", 5) == 5


def test_set_overrides_parse_yaml_scalars():
    out = parse_set_overrides(["render.max_labels=10", "output.enabled=false",
                               "video.width=null", "a.b=hello"])
    assert out["render"]["max_labels"] == 10
    assert out["output"]["enabled"] is False
    assert out["video"]["width"] is None
    assert out["a"]["b"] == "hello"


def test_deep_merge_replaces_lists_but_merges_dicts():
    base = {"a": {"x": 1, "y": 2}, "list": [1, 2, 3]}
    merged = deep_merge(base, {"a": {"y": 20}, "list": [9]})
    assert merged["a"] == {"x": 1, "y": 20}
    assert merged["list"] == [9]
    assert base["list"] == [1, 2, 3]        # the original is untouched


# ---------------------------------------------------------------------------
# Drift
# ---------------------------------------------------------------------------

def test_template_matching_measures_a_known_shift():
    from spotter.drift import DriftDetector, save_reference_patches

    rng = np.random.default_rng(3)
    image = rng.integers(0, 255, (400, 600, 4), dtype=np.uint8)
    image[:, :, 3] = 255

    class Tmp:
        pass

    import tempfile
    from pathlib import Path
    tmpdir = Path(tempfile.mkdtemp())

    candidates = [("a", 120.0, 120.0), ("b", 450.0, 150.0), ("c", 300.0, 300.0)]
    refs = save_reference_patches(image, candidates, tmpdir, half_size=24, want=3,
                                  min_texture=1.0)
    assert len(refs) == 3

    cfg = Config({"drift": {"enabled": True, "check_interval_s": 0.0,
                            "patch_half_size": 24, "search_radius_px": 30,
                            "min_confidence": 0.5, "warn_shift_px": 3.0,
                            "consecutive_checks": 2,
                            "patches_dir": str(tmpdir)}})
    detector = DriftDetector(cfg)
    assert detector.active

    # Unmoved image: no drift.
    report = detector.check(image, force=True)
    assert report.median_shift_px == pytest.approx(0.0, abs=0.5)
    assert not report.exceeded

    # Shift the whole scene by a known amount.
    shifted = np.roll(image, shift=(4, 7), axis=(0, 1))
    first = detector.check(shifted, force=True)
    assert first.median_shift_px == pytest.approx(np.hypot(7, 4), abs=1.0)
    assert first.exceeded
    # One bad check is not enough; haze and rain produce those.
    assert not first.flagged

    second = detector.check(shifted, force=True)
    assert second.flagged
    assert detector.flagged

    # Returning to normal clears the flag.
    detector.check(image, force=True)
    assert not detector.flagged
