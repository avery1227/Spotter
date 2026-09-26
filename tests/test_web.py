"""Web API: points CRUD, the delete guards, solving, and the preview hub."""

import json

import numpy as np
import pytest

from spotter.config import Config
from spotter.web.frames import FrameHub
from spotter.web.server import WebState, create_app

REF_LAT, REF_LON = 41.2700, -72.5000


@pytest.fixture
def workspace(tmp_path):
    """A config rooted in a temp dir, with a frame and a points file."""
    import cv2

    rng = np.random.default_rng(5)
    frame = rng.integers(0, 255, (1080, 1920, 3), dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "calib_frame.png"), frame)

    cfg = Config({
        "camera": {"lat": REF_LAT, "lon": REF_LON, "height_m": 15.0,
                   "refraction_k": 7 / 6},
        "calibration": {
            "path": "./calibration.json",
            "points": "./points.csv",
            "solver": {"position_sigma_m": 75.0, "height_sigma_m": 5.0,
                       "loss": "soft_l1", "f_scale_px": 3.0, "max_nfev": 20000,
                       "warn": {"min_points": 6, "horizon_band_frac": 0.08,
                                "max_horizon_fraction": 0.85,
                                "min_spread_x_frac": 0.35,
                                "min_spread_y_frac": 0.12, "max_rms_px": 6.0,
                                "max_loo_delta_px": 12.0,
                                "outlier_rms_improvement_frac": 0.4}},
        },
        "web": {"calibration_frame": "./calib_frame.png"},
        "drift": {"patches_dir": "./state/drift_patches", "patch_half_size": 32},
    }, root=tmp_path)
    return cfg


@pytest.fixture
def client(workspace):
    state = WebState(workspace)
    app = create_app(state)
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def add(client, name, px, py, lat, lon, elev=0.0):
    return client.post("/api/points", json={
        "name": name, "px": px, "py": py, "lat": lat, "lon": lon, "elev_m": elev})


# ---------------------------------------------------------------------------
# Points
# ---------------------------------------------------------------------------

def test_points_start_empty_and_round_trip(client):
    assert client.get("/api/points").get_json()["points"] == []

    res = add(client, "jetty_tip", 774.1, 355.0, 41.2662, -72.4984, 2.0)
    assert res.status_code == 200
    points = res.get_json()["points"]
    assert len(points) == 1
    assert points[0]["name"] == "jetty_tip"
    assert points[0]["px"] == pytest.approx(774.1)
    assert points[0]["elev_m"] == pytest.approx(2.0)

    # Persisted, not just held in memory.
    assert len(client.get("/api/points").get_json()["points"]) == 1


def test_bad_coordinates_are_rejected(client):
    assert add(client, "bad", 10, 10, 91.0, -72.0).status_code == 400
    assert add(client, "bad", 10, 10, 41.0, 200.0).status_code == 400
    assert client.post("/api/points", json={"name": "x"}).status_code == 400


def test_delete_requires_the_expected_name(client):
    add(client, "alpha", 100, 100, 41.1, -72.1)
    add(client, "beta", 200, 200, 41.2, -72.2)

    # No name at all: refused, because a stale page must not be able to delete
    # whatever happens to be at that index now.
    assert client.delete("/api/points/0").status_code == 400
    assert len(client.get("/api/points").get_json()["points"]) == 2

    # Wrong name: conflict.
    res = client.delete("/api/points/0?name=beta")
    assert res.status_code == 409
    assert "not 'beta'" in res.get_json()["error"]
    assert len(client.get("/api/points").get_json()["points"]) == 2

    # Right name: gone, and the removed point comes back for an undo.
    res = client.delete("/api/points/0?name=alpha")
    assert res.status_code == 200
    body = res.get_json()
    assert body["removed"]["name"] == "alpha"
    assert [p["name"] for p in body["points"]] == ["beta"]


def test_deleting_backs_up_the_previous_file(client, workspace):
    add(client, "alpha", 100, 100, 41.1, -72.1)
    add(client, "beta", 200, 200, 41.2, -72.2)
    client.delete("/api/points/0?name=alpha")

    history = sorted((workspace.root / "state" / "points_history").glob("*.csv"))
    assert history, "a backup should exist after a destructive write"
    # The newest backup still contains what was there before the delete.
    assert "alpha" in history[-1].read_text(encoding="utf-8")


def test_undo_restores_a_deleted_point(client):
    add(client, "alpha", 100, 100, 41.1, -72.1)
    removed = client.delete("/api/points/0?name=alpha").get_json()["removed"]
    assert client.get("/api/points").get_json()["points"] == []

    client.post("/api/points", json=removed)
    restored = client.get("/api/points").get_json()["points"]
    assert len(restored) == 1
    assert restored[0]["name"] == "alpha"
    assert restored[0]["px"] == pytest.approx(100)


def test_patch_toggles_and_edits(client):
    add(client, "alpha", 100, 100, 41.1, -72.1)

    client.patch("/api/points/0", json={"enabled": False})
    assert client.get("/api/points").get_json()["points"][0]["enabled"] is False

    client.patch("/api/points/0", json={"name": "renamed", "elev_m": 12.5})
    point = client.get("/api/points").get_json()["points"][0]
    assert point["name"] == "renamed"
    assert point["elev_m"] == pytest.approx(12.5)


def test_patch_rejects_a_mismatched_expectation(client):
    add(client, "alpha", 100, 100, 41.1, -72.1)
    res = client.patch("/api/points/0?name=somethingelse", json={"enabled": False})
    assert res.status_code == 409
    assert client.get("/api/points").get_json()["points"][0]["enabled"] is True


def test_missing_index(client):
    assert client.delete("/api/points/7?name=x").status_code == 404
    assert client.patch("/api/points/7", json={}).status_code == 404


# ---------------------------------------------------------------------------
# Solving
# ---------------------------------------------------------------------------

def seed_solvable_points(client):
    """Points generated from a known camera, so the solve has a real answer."""
    from spotter.calib.model import CameraModel
    from spotter.geodesy import enu_to_geodetic

    truth = CameraModel(ref_lat=REF_LAT, ref_lon=REF_LON, width=1920, height=1080,
                        yaw_deg=168.0, pitch_deg=-2.4, roll_deg=0.9,
                        focal_px=1830.0, height_m=18.5)
    specs = [(140, 300, 6.0), (158, 850, 3.0), (150, 120, 12.0),
             (168, 21000, 8.0), (176, 24000, 14.0), (188, 19000, 5.0),
             (160, 15000, 2.0), (182, 2200, 4.0), (172, 600, 25.0)]
    added = 0
    for i, (bearing, distance, elevation) in enumerate(specs):
        east = distance * np.sin(np.radians(bearing))
        north = distance * np.cos(np.radians(bearing))
        lat, lon, _ = enu_to_geodetic(np.array([[east, north, elevation]]),
                                      REF_LAT, REF_LON, 0.0)[0]
        uv, in_front, _ = truth.project_geodetic([lat], [lon], [elevation])
        if not in_front[0] or not (0 <= uv[0, 0] <= 1920 and 0 <= uv[0, 1] <= 1080):
            continue
        add(client, f"p{i}", float(uv[0, 0]), float(uv[0, 1]), lat, lon, elevation)
        added += 1
    return truth, added


def test_solve_needs_enough_points(client):
    add(client, "alpha", 100, 100, 41.1, -72.1)
    res = client.post("/api/solve", json={})
    assert res.status_code == 400
    assert "at least 4" in res.get_json()["error"]


def test_solve_recovers_the_camera(client):
    truth, count = seed_solvable_points(client)
    assert count >= 6

    res = client.post("/api/solve", json={"save": False, "loo": False})
    assert res.status_code == 200
    body = res.get_json()
    model = body["model"]

    assert body["summary"]["rms_px"] < 2.0
    assert model["yaw_deg"] == pytest.approx(truth.yaw_deg, abs=0.3)
    assert model["pitch_deg"] == pytest.approx(truth.pitch_deg, abs=0.3)
    assert model["focal_px"] == pytest.approx(truth.focal_px, rel=0.03)
    # hfov is what a person reads off the page; make sure it is derived, not 0.
    assert 20 < model["hfov_deg"] < 120


def test_solve_without_save_writes_nothing(client, workspace):
    seed_solvable_points(client)
    client.post("/api/solve", json={"save": False})
    assert not (workspace.root / "calibration.json").exists()


def test_solve_with_save_writes_calibration_and_reprojection_works(client, workspace):
    seed_solvable_points(client)
    res = client.post("/api/solve", json={"save": True, "loo": False})
    assert res.get_json()["saved"] is True

    calibration = workspace.root / "calibration.json"
    assert calibration.exists()
    saved = json.loads(calibration.read_text(encoding="utf-8"))
    assert saved["meta"]["solved_via"] == "web"

    reproj = client.get("/api/reprojection")
    assert reproj.status_code == 200
    body = reproj.get_json()
    assert len(body["points"]) >= 6
    assert len(body["horizon"]) > 50
    for point in body["points"]:
        if point["in_front"]:
            assert point["error_px"] < 5.0


def test_reprojection_before_any_calibration(client):
    assert client.get("/api/reprojection").status_code == 404


def test_lock_flags_are_honoured(client):
    seed_solvable_points(client)
    res = client.post("/api/solve",
                      json={"save": False, "loo": False, "lock_roll": True})
    assert res.get_json()["model"]["roll_deg"] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Pages and preview
# ---------------------------------------------------------------------------

def test_pages_and_info(client):
    assert client.get("/").status_code == 200
    assert client.get("/monitor").status_code == 200
    assert client.get("/static/app.css").status_code == 200
    assert client.get("/api/frame.png").status_code == 200

    info = client.get("/api/info").get_json()
    assert info["frame"]["width"] == 1920
    assert info["frame"]["height"] == 1080
    assert info["live"] is False
    assert info["camera"]["lat"] == pytest.approx(REF_LAT)


def test_status_without_a_pipeline(client):
    body = client.get("/api/status").get_json()
    assert body["live"] is False


def test_live_endpoints_without_frames(client):
    assert client.get("/api/live.jpg").status_code == 404


def test_frame_hub_encodes_and_caches():
    hub = FrameHub(max_width=320, quality=60)
    assert hub.jpeg() is None

    frame = np.zeros((480, 640, 4), np.uint8)
    frame[:, :, 1] = 200
    frame[:, :, 3] = 255
    hub.publish(frame, {"frame_time": "now"})

    first = hub.jpeg()
    assert first is not None
    data, sequence = first
    assert data[:2] == b"\xff\xd8"          # JPEG SOI
    assert sequence == 1
    assert hub.snapshot_meta()["frame_time"] == "now"

    # Same frame and settings: the cached encode is reused.
    assert hub.jpeg()[0] is data

    # A new frame invalidates it.
    hub.publish(frame.copy(), {})
    assert hub.jpeg()[1] == 2


def test_frame_hub_wait_returns_on_new_frame():
    import threading

    hub = FrameHub()
    frame = np.zeros((16, 16, 4), np.uint8)

    result = {}

    def waiter():
        result["sequence"] = hub.wait_for_frame(0, timeout=3.0)

    thread = threading.Thread(target=waiter)
    thread.start()
    hub.publish(frame)
    thread.join(timeout=3.0)
    assert result.get("sequence") == 1


def test_frame_hub_wait_times_out_without_frames():
    hub = FrameHub()
    assert hub.wait_for_frame(0, timeout=0.2) == 0


# ---------------------------------------------------------------------------
# Camera position
# ---------------------------------------------------------------------------

CONFIG_SAMPLE = """# Site configuration
camera:
  # Rough position - the solver refines this.
  lat: 41.2700
  lon: -72.5000
  # Height of the lens above mean sea level, metres.
  height_m: 15.0
  refraction_k: 1.1666666667

calibration:
  path: ./calibration.json
  solver:
    position_sigma_m: 75.0
    height_sigma_m: 5.0         # vertical wander allowed (1 sigma)
    lock_height: false

drift:
  enabled: true

tracks:
  adsb:
    enabled: true
"""


def test_set_yaml_value_preserves_comments():
    from spotter.config import set_yaml_value

    out, ok = set_yaml_value(CONFIG_SAMPLE, "camera.lat", 41.2812345)
    assert ok
    assert "lat: 41.2812345" in out
    # Every comment survives; a yaml.dump round trip would lose all of them.
    assert out.count("#") == CONFIG_SAMPLE.count("#")
    assert "# Rough position - the solver refines this." in out
    assert len(out.split("\n")) == len(CONFIG_SAMPLE.split("\n"))


def test_set_yaml_value_keeps_trailing_comment():
    from spotter.config import set_yaml_value

    out, ok = set_yaml_value(CONFIG_SAMPLE, "calibration.solver.height_sigma_m", 60.0)
    assert ok
    line = [ln for ln in out.split("\n") if "height_sigma_m" in ln][0]
    assert "60" in line
    assert "vertical wander allowed" in line


def test_set_yaml_value_distinguishes_same_named_keys():
    """`drift.enabled` and `tracks.adsb.enabled` must not be confused."""
    from spotter.config import set_yaml_value
    import yaml

    out, ok = set_yaml_value(CONFIG_SAMPLE, "tracks.adsb.enabled", False)
    assert ok
    parsed = yaml.safe_load(out)
    assert parsed["tracks"]["adsb"]["enabled"] is False
    assert parsed["drift"]["enabled"] is True


def test_set_yaml_value_reports_missing_keys():
    from spotter.config import set_yaml_value

    assert set_yaml_value(CONFIG_SAMPLE, "camera.nonexistent", 1)[1] is False
    assert set_yaml_value(CONFIG_SAMPLE, "nope.at.all", 1)[1] is False


def test_update_config_file_round_trip(tmp_path):
    import yaml

    from spotter.config import update_config_file

    path = tmp_path / "config.yaml"
    path.write_text(CONFIG_SAMPLE, encoding="utf-8")

    update_config_file(path, {"camera.lat": 41.3, "camera.height_m": 22.5})
    parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert parsed["camera"]["lat"] == pytest.approx(41.3)
    assert parsed["camera"]["height_m"] == pytest.approx(22.5)
    # A backup of the original is kept.
    assert (tmp_path / "config.yaml.bak").read_text(encoding="utf-8") == CONFIG_SAMPLE


def test_update_config_file_rejects_unknown_keys(tmp_path):
    from spotter.config import ConfigError, update_config_file

    path = tmp_path / "config.yaml"
    path.write_text(CONFIG_SAMPLE, encoding="utf-8")
    with pytest.raises(ConfigError, match="not found"):
        update_config_file(path, {"camera.bogus": 1})
    # The file is untouched when any key is missing.
    assert path.read_text(encoding="utf-8") == CONFIG_SAMPLE


@pytest.fixture
def file_backed_client(tmp_path):
    """A client whose config really came from a file, so it can be written."""
    import cv2

    from spotter.config import load_config

    cv2.imwrite(str(tmp_path / "calib_frame.png"),
                np.zeros((1080, 1920, 3), np.uint8))
    config_text = CONFIG_SAMPLE + """
  ais:
    enabled: false
web:
  calibration_frame: ./calib_frame.png
  unknown_height_guess_m: 12.0
  unknown_height_sigma_m: 60.0
  known_height_sigma_m: 5.0
"""
    config_text = config_text.replace("  path: ./calibration.json",
                                      "  path: ./calibration.json\n  points: ./points.csv")
    path = tmp_path / "config.yaml"
    path.write_text(config_text, encoding="utf-8")

    state = WebState(load_config(path))
    app = create_app(state)
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c, path


def test_camera_position_is_written_to_config(file_backed_client):
    import yaml

    client, path = file_backed_client
    res = client.post("/api/camera", json={
        "lat": 41.2812345, "lon": -72.4567890, "height_m": 22.5})
    assert res.status_code == 200
    body = res.get_json()
    assert body["camera"]["height_m"] == pytest.approx(22.5)

    parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert parsed["camera"]["lat"] == pytest.approx(41.2812345)
    assert parsed["camera"]["lon"] == pytest.approx(-72.4567890)
    assert parsed["camera"]["height_m"] == pytest.approx(22.5)
    # A known height keeps the tight prior.
    assert parsed["calibration"]["solver"]["height_sigma_m"] == pytest.approx(5.0)

    # The running server picks the change up without a restart.
    assert client.get("/api/info").get_json()["camera"]["lat"] == pytest.approx(41.2812345)


def test_unknown_height_widens_the_prior(file_backed_client):
    import yaml

    client, path = file_backed_client
    res = client.post("/api/camera", json={
        "lat": 41.28, "lon": -72.45, "height_unknown": True})
    assert res.status_code == 200
    body = res.get_json()

    # It gets a starting guess, and the solver is set free to move off it.
    assert body["camera"]["height_m"] == pytest.approx(12.0)
    assert body["height_sigma_m"] == pytest.approx(60.0)

    parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert parsed["calibration"]["solver"]["height_sigma_m"] == pytest.approx(60.0)
    # Crucially NOT locked -- locking would freeze it at the guess.
    assert parsed["calibration"]["solver"]["lock_height"] is False


def test_camera_rejects_bad_input(file_backed_client):
    client, _ = file_backed_client
    assert client.post("/api/camera", json={"lon": -72.5}).status_code == 400
    assert client.post("/api/camera",
                       json={"lat": 91.0, "lon": -72.5}).status_code == 400
    assert client.post("/api/camera",
                       json={"lat": 41.0, "lon": -72.5,
                             "height_m": 0}).status_code == 400


def test_camera_not_editable_without_a_file(client):
    """The in-memory fixture config has no source file to write to."""
    assert client.get("/api/info").get_json()["camera"]["editable"] is False
    assert client.post("/api/camera",
                       json={"lat": 41.0, "lon": -72.5, "height_m": 10}).status_code == 409


def test_high_precision_coordinates_are_accepted(file_backed_client):
    """A lat/lon with more decimals than we store must not be rejected.

    Values are written at eight decimal places (about a millimetre). Verifying
    the write against the caller's unrounded input rather than against what was
    actually formatted made the endpoint reject its own correct write and roll
    the edit back.
    """
    import yaml

    client, path = file_backed_client
    res = client.post("/api/camera", json={
        "lat": 41.276123456789, "lon": -72.459456123456, "height_m": 15.0})
    assert res.status_code == 200, res.get_json()

    stored = yaml.safe_load(path.read_text(encoding="utf-8"))["camera"]
    # Stored to within a millimetre, which is far finer than any GPS fix.
    assert abs(stored["lat"] - 41.276123456789) * 111320 < 0.01
    assert abs(stored["lon"] - (-72.459456123456)) * 111320 < 0.01

    # The response reports what is really in the file, not what was submitted.
    assert res.get_json()["camera"]["lat"] == pytest.approx(stored["lat"])


@pytest.mark.parametrize("lat,lon", [
    (41.27, -72.5),
    (41.2761235, -72.4594561),
    (41.276123456789, -72.459456123456),
    (41.2812345678901234, -72.4567890123456789),
    (-41.999999999, 179.999999999),
])
def test_config_write_round_trips_at_any_precision(tmp_path, lat, lon):
    import yaml

    from spotter.config import update_config_file

    path = tmp_path / "config.yaml"
    path.write_text(CONFIG_SAMPLE, encoding="utf-8")
    applied = update_config_file(path, {"camera.lat": lat, "camera.lon": lon})

    stored = yaml.safe_load(path.read_text(encoding="utf-8"))["camera"]
    assert stored["lat"] == applied["camera.lat"]
    assert stored["lon"] == applied["camera.lon"]
    assert abs(stored["lat"] - lat) < 1e-7
    assert abs(stored["lon"] - lon) < 1e-7


# ---------------------------------------------------------------------------
# Packaging
# ---------------------------------------------------------------------------

def test_shell_scripts_use_unix_line_endings():
    """A CRLF shebang makes the container die on start.

    `exec /entrypoint.sh failed: No such file or directory` is the symptom,
    because the kernel looks for an interpreter literally named "/bin/bash\r".
    Editing these files from Windows silently reintroduces it, so it is worth
    a test rather than a comment.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    offenders = []
    for path in list(root.rglob("*.sh")) + [root / "Dockerfile"]:
        if ".venv" in path.parts or ".git" in path.parts or not path.is_file():
            continue
        if b"\r\n" in path.read_bytes():
            offenders.append(str(path.relative_to(root)))
    assert not offenders, f"CRLF line endings in: {offenders}"


def test_pelican_egg_is_valid():
    """The egg validator is part of CI; run it here too so a local edit fails fast."""
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    result = subprocess.run([sys.executable, str(root / "pelican" / "validate_egg.py")],
                            capture_output=True, text=True, cwd=str(root))
    assert result.returncode == 0, result.stdout + result.stderr
