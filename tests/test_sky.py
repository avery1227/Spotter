"""Satellites and lightning: the geometry, the feed decoding, and the drawing.

Reference values for the satellite and Sun positions come from skyfield,
computed once and pasted in, so the test suite does not need it installed.
"""

import json
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pymap3d
import pytest

from spotter.calib.model import CameraModel
from spotter.config import Config
from spotter.render.overlay import OverlayRenderer, OverlayStatus
from spotter.sky import astro
from spotter.sky.lightning import (BoltEffect, LightningLayer, Strike,
                                   flash_intensity, haversine_km, lzw_decode,
                                   parse_strike)
from spotter.sky.satellites import SatelliteLayer, parse_tle_text
from spotter.tracks.model import TrackKind

ISS_TLE = (
    "ISS (ZARYA)\n"
    "1 25544U 98067A   26270.17419514  .00009528  00000+0  18291-3 0  9996\n"
    "2 25544  51.6315 155.3455 0007168 193.0560 167.0244 15.48664528587561\n"
)
#: A module docked to the ISS: catalogued separately, same orbit.
POISK_TLE = (
    "POISK\n"
    "1 36086U 09060A   26270.17419514  .00009528  00000+0  18291-3 0  9994\n"
    "2 36086  51.6315 155.3455 0007168 193.0560 167.0244 15.48664528587561\n"
)
#: ISS at 3.96 deg elevation, azimuth 177.59, 1967.5 km away (skyfield).
ISS_PASS = datetime(2026, 9, 29, 15, 10, tzinfo=timezone.utc)
CAMERA_LAT, CAMERA_LON = 41.26, -72.5


def camera() -> CameraModel:
    """Looking south over Long Island Sound, like the live deployment."""
    return CameraModel(ref_lat=CAMERA_LAT, ref_lon=CAMERA_LON, width=1920,
                       height=1080, yaw_deg=189.97, pitch_deg=-8.95,
                       height_m=27.0,
                       focal_px=1920 / (2 * math.tan(math.radians(77.4 / 2))))


def sky_config(tmp_path, **sky) -> Config:
    return Config({"camera": {"lat": CAMERA_LAT, "lon": CAMERA_LON},
                   "sky": sky}, root=tmp_path)


# ---------------------------------------------------------------------------
# Astronomy
# ---------------------------------------------------------------------------

def test_sgp4_position_matches_skyfield():
    from sgp4.api import Satrec
    name, line1, line2 = parse_tle_text(ISS_TLE)[0]
    jd, fraction = astro.split_julian_date(ISS_PASS)
    error, r_teme, _ = Satrec.twoline2rv(line1, line2).sgp4(jd, fraction)
    assert error == 0
    ecef = astro.teme_to_ecef(np.array(r_teme), jd + fraction)[0] * 1000.0
    e, n, u = pymap3d.ecef2enu(*ecef, CAMERA_LAT, CAMERA_LON, 27.0)
    elevation = math.degrees(math.atan2(u, math.hypot(e, n)))
    azimuth = math.degrees(math.atan2(e, n)) % 360.0
    assert elevation == pytest.approx(3.9587, abs=0.01)
    assert azimuth == pytest.approx(177.5945, abs=0.01)
    assert math.sqrt(e * e + n * n + u * u) / 1000 == pytest.approx(1967.5, abs=2)


@pytest.mark.parametrize("when, expected", [
    (datetime(2026, 9, 28, 22, 40, tzinfo=timezone.utc), -1.433),
    (datetime(2026, 6, 21, 16, 0, tzinfo=timezone.utc), 69.136),
])
def test_sun_elevation_matches_skyfield(when, expected):
    assert astro.sun_elevation_deg(when, CAMERA_LAT, CAMERA_LON) == pytest.approx(
        expected, abs=0.05)


def test_earth_shadow_is_behind_the_earth_only():
    sun = np.array([1.0, 0.0, 0.0])
    points = np.array([[7000.0, 0, 0],      # between Earth and Sun: lit
                       [-7000.0, 0, 0],     # directly behind: shadow
                       [-7000.0, 7000, 0],  # behind but off to the side: lit
                       [0.0, 7000, 0]])     # terminator, above the limb: lit
    assert list(astro.in_earth_shadow(points, sun)) == [False, True, False, False]


def test_refraction_is_about_half_a_degree_at_the_horizon():
    assert astro.refraction_deg(0.0) == pytest.approx(0.48, abs=0.03)
    assert astro.refraction_deg(5.0) == pytest.approx(0.16, abs=0.02)
    assert astro.refraction_deg(45.0) < 0.02


# ---------------------------------------------------------------------------
# Projection safety
# ---------------------------------------------------------------------------

def test_points_beyond_the_distortion_turnover_are_not_projected():
    """Barrel distortion folds far off-axis points back into the frame."""
    model = CameraModel(ref_lat=0, ref_lon=0, width=1920, height=1080,
                        focal_px=1500, k1=-0.0147, k2=-0.0362, height_m=0.0)
    limit_deg = math.degrees(math.atan(math.sqrt(model.max_valid_r2)))
    assert 50 < limit_deg < 60

    # Straight ahead is North at yaw 0; 70 degrees to the side is past it.
    ahead = [0.0, 1000.0, 0.0]
    side = [1000.0 * math.sin(math.radians(70)), 1000.0 * math.cos(math.radians(70)), 0]
    uv, valid, _ = model.project_enu(np.array([ahead, side]), refract=False)
    assert valid[0]
    assert not valid[1], "a point 70 degrees off-axis must not land in the frame"


# ---------------------------------------------------------------------------
# Satellites
# ---------------------------------------------------------------------------

def test_tle_parser_skips_junk():
    text = "garbage\n" + ISS_TLE + "\n\n" + POISK_TLE + "trailing\n"
    names = [entry[0] for entry in parse_tle_text(text)]
    assert names == ["ISS (ZARYA)", "POISK"]


def test_iss_pass_is_projected_into_the_frame(tmp_path):
    cache = tmp_path / "sats.tle"
    cache.write_text(ISS_TLE + POISK_TLE)
    layer = SatelliteLayer(sky_config(tmp_path, satellites={"cache": str(cache)}))
    assert layer.catalog.load_cache()

    targets = layer.project(ISS_PASS, camera())
    # POISK is docked to the ISS; it must not get a second, overlapping label.
    assert [t.labels["name"] for t in targets] == ["ISS"]
    iss = targets[0]
    assert iss.kind is TrackKind.SATELLITE
    assert iss.on_screen
    # Refraction lifts it by about 0.17 degrees above the geometric 3.96.
    assert iss.elevation_deg == pytest.approx(3.96 + 0.17, abs=0.05)
    assert iss.bearing_deg == pytest.approx(177.6, abs=0.05)
    assert iss.state.position.alt_m / 1000 == pytest.approx(426, abs=2)
    assert iss.labels["sunlit"] is True
    # Above the horizon line, below the top of the frame.
    assert 0 < iss.v < 540


def test_satellites_below_the_horizon_are_not_drawn(tmp_path):
    cache = tmp_path / "sats.tle"
    cache.write_text(ISS_TLE)
    layer = SatelliteLayer(sky_config(tmp_path, satellites={"cache": str(cache)}))
    layer.catalog.load_cache()
    # Half an orbit later it is on the far side of the planet.
    assert layer.project(ISS_PASS + timedelta(minutes=46), camera()) == []


def test_rocket_bodies_and_debris_are_skipped(tmp_path):
    cache = tmp_path / "sats.tle"
    cache.write_text(ISS_TLE.replace("ISS (ZARYA)", "SL-16 R/B")
                     + POISK_TLE.replace("POISK", "COSMOS 2251 DEB"))
    cfg = sky_config(tmp_path, satellites={"cache": str(cache),
                                           "show_in_daylight": True})
    layer = SatelliteLayer(cfg)
    layer.catalog.load_cache()
    assert layer.project(ISS_PASS, camera()) == []

    everything = SatelliteLayer(sky_config(tmp_path, satellites={
        "cache": str(cache), "show_in_daylight": True, "exclude_names": []}))
    everything.catalog.load_cache()
    assert len(everything.project(ISS_PASS, camera())) == 1


def test_satellites_in_shadow_are_hidden_unless_asked_for(tmp_path):
    cache = tmp_path / "sats.tle"
    cache.write_text(ISS_TLE.replace("ISS (ZARYA)", "SOME SATELLITE"))
    # 20:08 EDT: dark sky, and the orbit is in Earth's shadow by then.
    when = datetime(2026, 9, 29, 0, 8, 30, tzinfo=timezone.utc)
    shown = SatelliteLayer(sky_config(tmp_path, satellites={
        "cache": str(cache), "show_in_shadow": True}))
    shown.catalog.load_cache()
    targets = shown.project(when, camera())
    assert targets and targets[0].labels["sunlit"] is False

    hidden = SatelliteLayer(sky_config(tmp_path, satellites={"cache": str(cache)}))
    hidden.catalog.load_cache()
    assert hidden.project(when, camera()) == []


def test_daylight_hides_all_but_the_always_shown(tmp_path):
    cache = tmp_path / "sats.tle"
    cache.write_text(ISS_TLE.replace("ISS (ZARYA)", "SOME ROCKET BODY"))
    layer = SatelliteLayer(sky_config(tmp_path, satellites={"cache": str(cache)}))
    layer.catalog.load_cache()
    # 11:10 EDT: broad daylight.
    assert layer.project(ISS_PASS, camera()) == []
    layer.show_in_daylight = True
    assert len(layer.project(ISS_PASS, camera())) == 1


# ---------------------------------------------------------------------------
# Lightning
# ---------------------------------------------------------------------------

def lzw_encode(text: str) -> str:
    """The encoder matching Blitzortung's decoder, for round-trip tests."""
    table: dict[str, int] = {}
    out = []
    phrase = text[0]
    code = 256
    for char in text[1:]:
        if phrase + char in table:
            phrase += char
            continue
        out.append(phrase if len(phrase) == 1 else chr(table[phrase]))
        table[phrase + char] = code
        code += 1
        phrase = char
    out.append(phrase if len(phrase) == 1 else chr(table[phrase]))
    return "".join(out)


STRIKE_JSON = json.dumps({
    "time": 1790567279856720600, "lat": 40.93, "lon": -72.62, "alt": 0,
    "pol": 0, "mds": 8751, "status": 1, "region": 8, "delay": 13.9,
    "sig": [{"sta": 1, "time": 1, "lat": 45.1, "lon": -0.6, "alt": 0}] * 5,
})


def test_lzw_round_trip_on_a_realistic_message():
    encoded = lzw_encode(STRIKE_JSON)
    assert len(encoded) < len(STRIKE_JSON)
    assert lzw_decode(encoded) == STRIKE_JSON


def test_parse_strike_filters_by_range():
    message = lzw_encode(STRIKE_JSON)
    strike = parse_strike(message, CAMERA_LAT, CAMERA_LON, 300.0, received=0.0)
    assert strike is not None
    assert strike.distance_km == pytest.approx(
        haversine_km(CAMERA_LAT, CAMERA_LON, 40.93, -72.62), abs=0.01)
    assert strike.time == pytest.approx(1790567279.8567206)
    assert parse_strike(message, CAMERA_LAT, CAMERA_LON, 20.0, received=0.0) is None
    assert parse_strike("not json", CAMERA_LAT, CAMERA_LON, 300.0, 0.0) is None


def test_flash_flickers_then_ends():
    assert flash_intensity(0.0, 1.2) > 0.9
    assert flash_intensity(0.1, 1.2) < flash_intensity(0.12, 1.2)  # re-strike
    assert flash_intensity(1.2, 1.2) == 0.0
    assert flash_intensity(-0.1, 1.2) == 0.0


def strike_at(when: datetime, seconds_before: float, lat=40.93, lon=-72.62):
    stamp = when.timestamp() - seconds_before
    return Strike(time=stamp, lat=lat, lon=lon, received=stamp + 15.0,
                  distance_km=haversine_km(CAMERA_LAT, CAMERA_LON, lat, lon))


def test_strike_flashes_on_its_own_frame_and_labels_follow(tmp_path):
    layer = LightningLayer(sky_config(tmp_path))
    now = datetime(2026, 9, 29, 2, 0, tzinfo=timezone.utc)
    layer.add_strike(strike_at(now, 0.0))

    # Before it happened: nothing.
    assert layer.project(now - timedelta(seconds=1), camera()) == ([], [])

    targets, effects = layer.project(now, camera())
    assert len(targets) == 1 and len(effects) == 1
    target = targets[0]
    assert target.kind is TrackKind.LIGHTNING
    assert target.bearing_deg == pytest.approx(194, abs=2)
    # The channel runs upward from the ground point.
    assert effects[0].top_v < effects[0].ground_v
    assert layer.late == 0

    # After the flash the label stays, the bolt does not.
    targets, effects = layer.project(now + timedelta(seconds=5), camera())
    assert len(targets) == 1 and effects == []
    assert target.labels["age_s"] == pytest.approx(5.0)

    # And after label_s, everything is gone.
    assert layer.project(now + timedelta(seconds=25), camera()) == ([], [])


def test_late_strike_still_flashes_and_is_counted(tmp_path):
    layer = LightningLayer(sky_config(tmp_path))
    now = datetime(2026, 9, 29, 2, 0, tzinfo=timezone.utc)
    layer.add_strike(strike_at(now, 6.0))
    targets, effects = layer.project(now, camera())
    assert len(effects) == 1, "a strike we heard about late should still flash"
    assert layer.late == 1


def test_duplicates_are_dropped_and_labels_capped(tmp_path):
    layer = LightningLayer(sky_config(tmp_path, lightning={"max_labels": 2}))
    now = datetime(2026, 9, 29, 2, 0, tzinfo=timezone.utc)
    layer.add_strike(strike_at(now, 1.0))
    layer.add_strike(strike_at(now, 1.0, lat=40.931))   # re-published solution
    for i in range(4):
        layer.add_strike(strike_at(now, 2.0 + i, lon=-72.62 + 0.1 * i))
    assert layer.strikes_in_range == 5
    targets, _ = layer.project(now, camera())
    assert len(targets) == 2
    assert layer.in_view == 5


def test_strikes_behind_the_camera_are_ignored(tmp_path):
    layer = LightningLayer(sky_config(tmp_path))
    now = datetime(2026, 9, 29, 2, 0, tzinfo=timezone.utc)
    layer.add_strike(strike_at(now, 0.0, lat=42.0, lon=-72.5))   # due north
    assert layer.project(now, camera()) == ([], [])


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def test_renderer_draws_bolts_and_satellite_labels(tmp_path):
    cache = tmp_path / "sats.tle"
    cache.write_text(ISS_TLE)
    cfg = sky_config(tmp_path, satellites={"cache": str(cache)})
    layer = SatelliteLayer(cfg)
    layer.catalog.load_cache()
    model = camera()
    targets = layer.project(ISS_PASS, model)

    image = np.zeros((1080, 1920, 4), np.uint8)
    image[..., 3] = 255
    renderer = OverlayRenderer(cfg, model)
    bolt = BoltEffect(ground_u=1000, ground_v=355, top_u=1005, top_v=160,
                      intensity=1.0, seed=7)
    for step in range(20):   # let the label fade in
        frame = image.copy()
        renderer.render(frame, targets, ISS_PASS, OverlayStatus(),
                        now=100.0 + step * 0.1, effects=[bolt])

    # The bolt's core is drawn white along its channel...
    column = frame[165:350, 990:1016, :3]
    assert column.max() == 255
    # ...and the ISS label appears next to its marker.
    iss = targets[0]
    patch = frame[int(iss.v) - 80:int(iss.v) + 80,
                  int(iss.u) - 250:int(iss.u) + 250, :3]
    assert patch.max() > 150
