"""Fixtures shared across test modules."""

import numpy as np
import pytest

from spotter.config import Config

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
