"""Geodesy, the camera model's projection, and the calibration solver."""

import numpy as np
import pytest

from spotter.calib.model import CameraModel
from spotter.calib.points import ControlPoint, ControlPointSet
from spotter.calib.solver import solve_calibration
from spotter.config import Config
from spotter.geodesy import (DEFAULT_REFRACTION_K, EARTH_RADIUS_M, dead_reckon,
                             enu_to_geodetic, geodetic_to_enu,
                             horizon_distance_m, is_above_horizon,
                             max_visible_range_m, refraction_lift_m)

REF_LAT, REF_LON = 41.2700, -72.5000
W, H = 1920, 1080


def make_config(**camera):
    base = {
        "camera": {"lat": REF_LAT, "lon": REF_LON, "height_m": 15.0,
                   "refraction_k": DEFAULT_REFRACTION_K},
        "calibration": {"solver": {
            "position_sigma_m": 75.0, "height_sigma_m": 5.0,
            "loss": "soft_l1", "f_scale_px": 3.0, "max_nfev": 20000,
            "warn": {"min_points": 6, "horizon_band_frac": 0.08,
                     "max_horizon_fraction": 0.85, "min_spread_x_frac": 0.35,
                     "min_spread_y_frac": 0.12, "max_rms_px": 6.0,
                     "max_loo_delta_px": 12.0,
                     "outlier_rms_improvement_frac": 0.4}}},
    }
    base["camera"].update(camera)
    return Config(base)


# ---------------------------------------------------------------------------
# Geodesy
# ---------------------------------------------------------------------------

def test_enu_round_trip():
    enu = np.array([[1234.0, -5678.0, 42.0]])
    back = enu_to_geodetic(enu, REF_LAT, REF_LON, 0.0)
    again = geodetic_to_enu(back[:, 0], back[:, 1], back[:, 2],
                            REF_LAT, REF_LON, 0.0)
    assert np.allclose(enu, again, atol=1e-6)


def test_curvature_drop_matches_the_closed_form():
    """A sea-level point 30 km away sits about d^2/2R below the tangent plane."""
    distance = 30000.0
    enu = geodetic_to_enu(REF_LAT + distance / 111320.0, REF_LON, 0.0,
                          REF_LAT, REF_LON, 0.0)
    expected = -(distance ** 2) / (2 * EARTH_RADIUS_M)
    assert enu[0, 2] == pytest.approx(expected, rel=0.02)


def test_refraction_lift_is_a_seventh_of_the_drop_at_k_seven_sixths():
    distance = 30000.0
    drop = distance ** 2 / (2 * EARTH_RADIUS_M)
    lift = refraction_lift_m(distance, DEFAULT_REFRACTION_K)
    assert float(lift) == pytest.approx(drop / 7.0, rel=1e-6)
    # k = 1 means no refraction at all.
    assert float(refraction_lift_m(distance, 1.0)) == pytest.approx(0.0)


def test_horizon_distance():
    # A 15 m eye height gives ~13.8 km geometric, ~14.9 km with refraction.
    assert horizon_distance_m(15.0, 1.0) == pytest.approx(13820, rel=0.01)
    assert horizon_distance_m(15.0, DEFAULT_REFRACTION_K) == pytest.approx(14930,
                                                                           rel=0.01)
    assert horizon_distance_m(0.0) == 0.0


def test_tall_ships_stay_visible_past_the_horizon():
    """The whole point of height-aware culling."""
    observer = 15.0
    horizon = horizon_distance_m(observer)
    assert max_visible_range_m(observer, 30.0) > horizon * 1.5

    ranges = np.array([10_000.0, 20_000.0, 40_000.0])
    # A 30 m container ship is visible further out than a 2 m dinghy.
    big = is_above_horizon(ranges, 30.0, observer)
    small = is_above_horizon(ranges, 1.0, observer)
    assert big.tolist() == [True, True, False]
    assert small.tolist() == [True, False, False]


def test_dead_reckon_moves_the_right_way():
    lat, lon = dead_reckon(REF_LAT, REF_LON, course_deg=0.0,
                           speed_mps=10.0, dt_s=60.0)
    assert lat > REF_LAT and lon == pytest.approx(REF_LON, abs=1e-6)

    lat, lon = dead_reckon(REF_LAT, REF_LON, 90.0, 10.0, 60.0)
    assert lon > REF_LON and lat == pytest.approx(REF_LAT, abs=1e-5)

    # 600 m in 60 s at 10 m/s.
    enu = geodetic_to_enu(lat, lon, 0.0, REF_LAT, REF_LON, 0.0)
    assert float(np.hypot(enu[0, 0], enu[0, 1])) == pytest.approx(600.0, rel=0.01)

    # Reversing the clock retraces the path.
    assert dead_reckon(REF_LAT, REF_LON, 45.0, 0.0, 60.0) == (REF_LAT, REF_LON)


# ---------------------------------------------------------------------------
# Projection conventions
# ---------------------------------------------------------------------------

def base_model(**kwargs):
    defaults = dict(ref_lat=REF_LAT, ref_lon=REF_LON, width=W, height=H,
                    height_m=15.0, focal_px=1500.0)
    defaults.update(kwargs)
    return CameraModel(**defaults)


def project_point(model, e, n, u, refract=False):
    """Project one ENU point. Refraction defaults off here: these tests check
    the rotation/projection conventions, and refraction perturbs v by a
    fraction of a pixel, which would force a tolerance loose enough to let a
    genuine sign error through."""
    uv, in_front, depth = model.project_enu(np.array([[e, n, u]]), refract=refract)
    return uv[0], bool(in_front[0]), float(depth[0])


def test_yaw_zero_looks_north():
    model = base_model(yaw_deg=0.0)
    uv, in_front, _ = project_point(model, 0, 1000, 15)
    assert in_front
    assert uv == pytest.approx([model.cx, model.cy], abs=1e-6)


def test_east_is_right_and_up_is_up():
    model = base_model(yaw_deg=0.0)
    right, _, _ = project_point(model, 1000, 1000, 15)
    left, _, _ = project_point(model, -1000, 1000, 15)
    high, _, _ = project_point(model, 0, 1000, 115)
    assert right[0] > model.cx
    assert left[0] < model.cx
    assert high[1] < model.cy      # higher in the world is lower v in the image


def test_targets_behind_the_camera_are_flagged():
    model = base_model(yaw_deg=0.0)
    _, in_front, depth = project_point(model, 0, -1000, 15)
    assert not in_front and depth < 0


def test_pitching_up_moves_targets_down_the_frame():
    level, _, _ = project_point(base_model(pitch_deg=0.0), 0, 1000, 15)
    tilted, _, _ = project_point(base_model(pitch_deg=5.0), 0, 1000, 15)
    assert tilted[1] > level[1]


def test_yaw_rotates_the_world_the_right_way():
    model = base_model(yaw_deg=90.0)      # looking east
    centre, _, _ = project_point(model, 1000, 0, 15)
    north_east, _, _ = project_point(model, 1000, 1000, 15)
    assert centre == pytest.approx([model.cx, model.cy], abs=1e-6)
    assert north_east[0] < model.cx       # north of the axis is to the left


def test_fov_and_rescaling():
    model = base_model(focal_px=1500.0)
    assert model.hfov_deg == pytest.approx(
        np.degrees(2 * np.arctan(W / 3000.0)), rel=1e-9)

    scaled = model.scaled_to(W // 2, H // 2)
    assert scaled.focal_px == pytest.approx(750.0)
    assert scaled.hfov_deg == pytest.approx(model.hfov_deg, rel=1e-9)
    with pytest.raises(ValueError):
        model.scaled_to(1000, 1000)       # aspect ratio change must be refused


def test_horizon_polyline_stays_in_front_of_the_camera():
    model = base_model(yaw_deg=172.0, pitch_deg=-9.0)
    polyline = model.horizon_polyline()
    assert len(polyline) > 50
    # Sampling only the field of view keeps the polyline monotonic in x, which
    # is what stops the debug overlay drawing spikes back across the frame.
    assert np.all(np.diff(polyline[:, 0]) > 0) or np.all(np.diff(polyline[:, 0]) < 0)


def test_save_and_load_round_trip(tmp_path):
    model = base_model(yaw_deg=172.5, pitch_deg=-9.25, roll_deg=0.4,
                       k1=-0.03, k2=0.004, offset_e_m=25.0, offset_n_m=-18.0)
    path = tmp_path / "calibration.json"
    model.save(path)
    loaded = CameraModel.load(path)
    for field in ("yaw_deg", "pitch_deg", "roll_deg", "focal_px", "k1", "k2",
                  "offset_e_m", "offset_n_m", "height_m"):
        assert getattr(loaded, field) == pytest.approx(getattr(model, field))


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

TRUTH = dict(yaw_deg=168.0, pitch_deg=-2.4, roll_deg=0.9, focal_px=1830.0,
             k1=-0.085, k2=0.012, offset_e_m=38.0, offset_n_m=-52.0,
             height_m=18.5)

LANDMARKS = [(140, 300, 6.0), (158, 850, 3.0), (150, 120, 12.0),
             (168, 21000, 8.0), (176, 24000, 14.0), (188, 19000, 5.0),
             (196, 26000, 22.0), (160, 15000, 2.0), (182, 2200, 4.0),
             (145, 5000, 1.5), (172, 600, 25.0)]


def synthesise(truth_model, specs, noise_px=1.0, seed=7):
    rng = np.random.default_rng(seed)
    points = []
    for index, (bearing, distance, elevation) in enumerate(specs):
        east = truth_model.offset_e_m + distance * np.sin(np.radians(bearing))
        north = truth_model.offset_n_m + distance * np.cos(np.radians(bearing))
        lat, lon, _ = enu_to_geodetic(np.array([[east, north, elevation]]),
                                      REF_LAT, REF_LON, 0.0)[0]
        uv, in_front, _ = truth_model.project_geodetic([lat], [lon], [elevation])
        if not in_front[0] or not (0 <= uv[0, 0] <= W and 0 <= uv[0, 1] <= H):
            continue
        points.append(ControlPoint(
            f"p{index}",
            uv[0, 0] + rng.normal(0, noise_px),
            uv[0, 1] + rng.normal(0, noise_px),
            lat, lon, elevation))
    return ControlPointSet(points=points, image_width=W, image_height=H)


def test_solver_recovers_a_known_camera():
    truth = CameraModel(ref_lat=REF_LAT, ref_lon=REF_LON, width=W, height=H,
                        **TRUTH)
    points = synthesise(truth, LANDMARKS)
    assert len(points) >= 8

    result = solve_calibration(points, make_config(), W, H, run_loo=False)
    model = result.model

    assert result.converged
    assert result.rms_px < 2.0
    assert model.yaw_deg == pytest.approx(TRUTH["yaw_deg"], abs=0.2)
    assert model.pitch_deg == pytest.approx(TRUTH["pitch_deg"], abs=0.2)
    assert model.roll_deg == pytest.approx(TRUTH["roll_deg"], abs=0.2)
    assert model.focal_px == pytest.approx(TRUTH["focal_px"], rel=0.02)
    assert model.height_m == pytest.approx(TRUTH["height_m"], abs=1.0)


def test_solved_model_projects_unseen_targets_correctly():
    """The real test of a calibration is where it puts targets it never saw.

    k1 and k2 trade off against each other, so comparing them to the truth
    individually is meaningless; what matters is the combined mapping.
    """
    truth = CameraModel(ref_lat=REF_LAT, ref_lon=REF_LON, width=W, height=H,
                        **TRUTH)
    result = solve_calibration(synthesise(truth, LANDMARKS), make_config(), W, H,
                               run_loo=False)

    worst = 0.0
    for bearing, distance, elevation in [(165, 12000, 50.0), (185, 30000, 100.0),
                                         (150, 3000, 300.0), (175, 8000, 1500.0)]:
        east = truth.offset_e_m + distance * np.sin(np.radians(bearing))
        north = truth.offset_n_m + distance * np.cos(np.radians(bearing))
        lat, lon, _ = enu_to_geodetic(np.array([[east, north, elevation]]),
                                      REF_LAT, REF_LON, 0.0)[0]
        a, _, _ = truth.project_geodetic([lat], [lon], [elevation])
        b, _, _ = result.model.project_geodetic([lat], [lon], [elevation])
        worst = max(worst, float(np.hypot(*(a[0] - b[0]))))
    assert worst < 4.0


def test_all_points_on_the_horizon_is_warned_about():
    truth = CameraModel(ref_lat=REF_LAT, ref_lon=REF_LON, width=W, height=H,
                        yaw_deg=170, pitch_deg=-2.0, height_m=18.0, focal_px=1800.0)
    points = synthesise(truth, [(160 + i * 4, 20000 + i * 900, 2.0)
                                for i in range(8)])
    result = solve_calibration(points, make_config(), W, H, run_loo=False)
    assert any("same height" in w for w in result.warnings)


def test_leave_one_out_identifies_a_mistyped_coordinate():
    """A bad point shows up as "everything else improves without it".

    It does *not* show up as a large increase in its own error, because its
    in-fit error is already large -- which is why the report scores points by
    how much the remaining RMS improves.
    """
    truth = CameraModel(ref_lat=REF_LAT, ref_lon=REF_LON, width=W, height=H,
                        yaw_deg=170, pitch_deg=-2.0, height_m=18.0, focal_px=1800.0)
    points = synthesise(truth, LANDMARKS[:8])
    # synthesise() drops points that fall off-frame, so index != name suffix.
    corrupted = points.points[3]
    corrupted.lon += 0.02                 # about 1.7 km out

    result = solve_calibration(points, make_config(), W, H, run_loo=True)
    assert result.rms_px > 10.0

    culprit = max(result.errors, key=lambda e: e.rms_improvement_px or 0.0)
    assert culprit.name == corrupted.name
    assert culprit.loo_rest_rms_px < 3.0
    assert any(corrupted.name in w and "looks wrong" in w
               for w in result.warnings)


def test_too_few_points_is_refused():
    truth = CameraModel(ref_lat=REF_LAT, ref_lon=REF_LON, width=W, height=H,
                        **TRUTH)
    points = synthesise(truth, LANDMARKS[:3])
    with pytest.raises(ValueError, match="at least 4"):
        solve_calibration(points, make_config(), W, H)


def test_locking_height_holds_it_exactly():
    truth = CameraModel(ref_lat=REF_LAT, ref_lon=REF_LON, width=W, height=H,
                        **TRUTH)
    cfg = make_config()
    cfg_raw = cfg.raw()
    cfg_raw["calibration"]["solver"]["lock_height"] = True
    cfg_raw["camera"]["height_m"] = 15.0
    result = solve_calibration(synthesise(truth, LANDMARKS), Config(cfg_raw),
                               W, H, run_loo=False)
    assert result.model.height_m == pytest.approx(15.0)
