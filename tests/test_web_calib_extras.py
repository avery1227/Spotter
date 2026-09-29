"""Web API for traced lines and frozen-frame aircraft points."""

from datetime import datetime, timezone

import numpy as np
import pytest

from spotter.calib.aircraft import build_capture
from spotter.calib.model import CameraModel
from spotter.tracks.model import Position, TrackKind, TrackReport
from spotter.tracks.store import Track, TrackState
from spotter.web.server import WebState, create_app

from .conftest import REF_LAT, REF_LON

FRAME_TIME = datetime(2026, 9, 29, 20, 0, 0, tzinfo=timezone.utc)


def model() -> CameraModel:
    return CameraModel(ref_lat=REF_LAT, ref_lon=REF_LON, width=1920, height=1080,
                       yaw_deg=180.0, pitch_deg=-5.0, height_m=15.0, focal_px=1500.0)


def aircraft_state(icao, lat, lon, alt_m, speed=200.0, course=90.0, alt_source="geom"):
    track = Track(f"icao:{icao}", TrackKind.AIRCRAFT)
    track.add(TrackReport(track_id=track.id, kind=TrackKind.AIRCRAFT,
                          timestamp=FRAME_TIME, lat=lat, lon=lon, alt_m=alt_m,
                          speed_mps=speed, course_deg=course, vertical_rate_mps=-2.0,
                          labels={"callsign": f"TEST{icao}", "icao": icao.upper(),
                                  "alt_source": alt_source}))
    return TrackState(track=track, position=Position(
        lat=lat, lon=lon, alt_m=alt_m, speed_mps=speed, course_deg=course))


class FakePipeline:
    """Just enough pipeline for the freeze endpoint."""

    def __init__(self, fail=None):
        self.fail = fail
        self.calls = 0

    def capture_freeze(self, timeout_s=5.0):
        self.calls += 1
        if self.fail:
            raise self.fail
        image = np.full((1080, 1920, 4), 40, np.uint8)
        states = [
            aircraft_state("a1", REF_LAT - 0.25, REF_LON, 4900.0),              # south, in view
            aircraft_state("b2", REF_LAT + 0.35, REF_LON, 3000.0, course=270.0,
                           alt_source="baro"),                                    # north, behind
            aircraft_state("c3", REF_LAT - 3.0, REF_LON, 9000.0),                # 330 km: out
        ]
        return build_capture(image, FRAME_TIME, states, model(), delay_s=12.0)

    def web_status(self):
        return {}


@pytest.fixture
def live_client(workspace):
    state = WebState(workspace, pipeline=FakePipeline())
    app = create_app(state)
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture
def offline_client(workspace):
    app = create_app(WebState(workspace))
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


LINE = {"name": "seawall", "image": [[100, 700], [900, 720], [1700, 760]],
        "map": [[41.2690, -72.5010], [41.2688, -72.5000], [41.2685, -72.4990]],
        "elev_m": 3.0, "elev_sigma_m": 1.0}


# ---------------------------------------------------------------------------
# Lines
# ---------------------------------------------------------------------------

def test_lines_crud(offline_client, workspace):
    client = offline_client
    assert client.get("/api/lines").get_json()["lines"] == []
    res = client.post("/api/lines", json=LINE)
    assert res.status_code == 200
    assert res.get_json()["lines"][0]["name"] == "seawall"
    assert (workspace.root / "lines.json").is_file()

    # Edits and deletes must name the line they mean.
    assert client.patch("/api/lines/0", json={"enabled": False}).status_code == 400
    assert client.patch("/api/lines/0?name=other", json={"enabled": False}).status_code == 409
    res = client.patch("/api/lines/0?name=seawall", json={"enabled": False, "elev_m": 2.5})
    line = res.get_json()["lines"][0]
    assert line["enabled"] is False and line["elev_m"] == 2.5

    res = client.delete("/api/lines/0?name=seawall")
    assert res.get_json()["lines"] == []
    # Undo: the removed line posts straight back.
    assert client.post("/api/lines", json=res.get_json()["removed"]).status_code == 200
    assert client.delete("/api/lines/5?name=x").status_code == 404


def test_bad_lines_are_rejected(offline_client):
    bad = dict(LINE, image=[[1, 2]])
    assert offline_client.post("/api/lines", json=bad).status_code == 400
    bad = dict(LINE, map=[[141.0, 0.0], [41.0, 0.0]])
    assert offline_client.post("/api/lines", json=bad).status_code == 400


def test_a_line_counts_towards_the_minimum_to_solve(offline_client):
    client = offline_client
    client.post("/api/lines", json=LINE)
    res = client.post("/api/solve", json={})
    assert res.status_code == 400
    assert "a traced line counts as 2" in res.get_json()["error"]


# ---------------------------------------------------------------------------
# Freezing
# ---------------------------------------------------------------------------

def test_freeze_needs_the_pipeline(offline_client):
    res = offline_client.post("/api/freeze")
    assert res.status_code == 409
    assert "live pipeline" in res.get_json()["error"]
    assert offline_client.get("/api/info").get_json()["can_freeze"] is False


def test_freeze_returns_frame_and_aircraft(live_client):
    assert live_client.get("/api/info").get_json()["can_freeze"] is True
    res = live_client.post("/api/freeze")
    assert res.status_code == 200
    capture = res.get_json()
    assert capture["frame_time"] == FRAME_TIME.isoformat()
    assert capture["delay_s"] == 12.0
    ids = [a["id"] for a in capture["aircraft"]]
    # Nearest first; the one 330 km away is dropped.
    assert ids == ["icao:a1", "icao:b2"]
    south, north = capture["aircraft"]
    assert south["predicted"] is not None
    assert 0 < south["predicted"][0] < 1920
    assert north["predicted"] is None        # behind the camera
    assert north["alt_source"] == "baro"
    assert south["ve"] == pytest.approx(200.0, abs=0.01)   # course 090
    assert south["vn"] == pytest.approx(0.0, abs=0.01)
    assert south["vu"] == pytest.approx(-2.0)

    image = live_client.get(f"/api/freeze/{capture['id']}.png")
    assert image.status_code == 200 and image.data[:4] == b"\x89PNG"
    assert live_client.get("/api/freeze/nope.png").status_code == 404


def test_freeze_reports_pipeline_problems(workspace):
    for error, status in ((TimeoutError("no frame"), 504),
                          (RuntimeError("not started"), 409)):
        app = create_app(WebState(workspace, pipeline=FakePipeline(fail=error)))
        with app.test_client() as client:
            assert client.post("/api/freeze").status_code == status


def test_aircraft_point_is_positioned_from_the_capture(live_client):
    capture = live_client.post("/api/freeze").get_json()
    res = live_client.post(f"/api/freeze/{capture['id']}/points",
                           json={"aircraft": "icao:a1", "px": 960.0, "py": 300.0})
    assert res.status_code == 200
    point = res.get_json()["added"]
    assert point["kind"] == "aircraft"
    assert point["name"] == "TESTa1 20:00:00"
    assert point["lat"] == pytest.approx(REF_LAT - 0.25)
    assert point["elev_m"] == pytest.approx(4900.0)
    assert point["delay_s"] == 12.0
    assert point["ve"] == pytest.approx(200.0, abs=0.01)
    assert point["time"] == FRAME_TIME.isoformat()

    # Persisted with its velocity, so the solver can use it for timing.
    stored = live_client.get("/api/points").get_json()["points"][0]
    assert stored["kind"] == "aircraft" and stored["vu"] == pytest.approx(-2.0)

    # Unknown aircraft, missing pixel, expired capture.
    assert live_client.post(f"/api/freeze/{capture['id']}/points",
                            json={"aircraft": "icao:zz", "px": 1, "py": 1}).status_code == 400
    assert live_client.post(f"/api/freeze/{capture['id']}/points",
                            json={"aircraft": "icao:a1"}).status_code == 400
    assert live_client.post("/api/freeze/expired/points",
                            json={"aircraft": "icao:a1", "px": 1, "py": 1}).status_code == 404


def test_only_a_few_captures_are_kept(live_client):
    first = live_client.post("/api/freeze").get_json()["id"]
    for _ in range(4):
        live_client.post("/api/freeze")
    assert live_client.get(f"/api/freeze/{first}.png").status_code == 404


def test_pressure_altitude_is_flagged_in_the_note(live_client):
    capture = live_client.post("/api/freeze").get_json()
    point = live_client.post(f"/api/freeze/{capture['id']}/points",
                             json={"aircraft": "icao:b2", "px": 5, "py": 5}
                             ).get_json()["added"]
    assert "pressure altitude" in point["note"]


# ---------------------------------------------------------------------------
# Solving through the API
# ---------------------------------------------------------------------------

def test_solve_uses_lines_and_reports_timing(tmp_path):
    """Lines, far lights and aircraft posted through the API solve together."""
    from spotter.config import Config

    from . import test_calib_lines as scene

    truth = scene.true_camera()
    cfg = scene.solver_config()
    cfg = Config({**cfg.raw(), "calibration": {
        **cfg.raw()["calibration"], "path": "./calibration.json",
        "points": "./points.csv", "lines": "./lines.json"}}, root=tmp_path)
    app = create_app(WebState(cfg))
    app.config["TESTING"] = True
    with app.test_client() as client:
        for line in scene.near_field_lines(truth).lines:
            assert client.post("/api/lines", json=line.to_json()).status_code == 200
        points = scene.far_points(truth) + scene.aircraft_points(truth, dt_true=2.0)
        for point in points:
            body = {k: getattr(point, k) for k in (
                "name", "px", "py", "lat", "lon", "elev_m", "kind", "time",
                "delay_s", "ve", "vn", "vu")}
            assert client.post("/api/points", json=body).status_code == 200

        res = client.post("/api/solve", json={"save": True, "loo": False})
        assert res.status_code == 200, res.get_json()
        data = res.get_json()
        assert data["summary"]["n_lines"] == 4
        assert data["summary"]["timing_offset_s"] == pytest.approx(2.0, abs=0.3)
        assert data["summary"]["suggested_encoder_delay_s"] == pytest.approx(10.0, abs=0.3)
        assert {entry["name"] for entry in data["lines"]} == {
            "seawall", "waterline", "path", "divider"}
        assert data["model"]["yaw_deg"] == pytest.approx(truth.yaw_deg, abs=0.1)

        # The saved calibration projects every map line back onto the frame.
        reproj = client.get("/api/reprojection").get_json()
        assert len(reproj["lines"]) == 4
        assert all(line["projected"] for line in reproj["lines"])
