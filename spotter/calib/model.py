"""The camera model: pose, intrinsics, distortion, and the projection itself.

Conventions, fixed once here so every other module can rely on them:

* **ENU** -- local East/North/Up metres, origin at sea level directly below the
  configured rough camera position. The camera itself sits at
  ``(offset_e, offset_n, height_m)``, which the solver is free to nudge.
* **Camera frame** -- computer-vision convention: ``x`` right, ``y`` down,
  ``z`` forward along the optical axis.
* **yaw** -- compass bearing of the optical axis, degrees clockwise from North.
* **pitch** -- degrees, positive when the camera is tilted up.
* **roll** -- degrees, positive rolls the horizon clockwise in the image.

The full chain is::

    v_cam = R_roll @ R_pitch @ B @ R_yaw @ (p_enu - c_enu)

where ``B`` maps ENU to a north-facing camera frame and ``R_yaw`` spins the
world about Up. Then a pinhole projection with two radial distortion terms.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ..geodesy import (DEFAULT_REFRACTION_K, geodetic_to_enu, refraction_lift_m)

#: Maps ENU to a camera frame looking due North with zero pitch and roll:
#: x_cam = East, y_cam = -Up, z_cam = North.
_ENU_TO_CAM_NORTH = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
    [0.0, 1.0, 0.0],
])


def _rz_enu(yaw_rad: float) -> np.ndarray:
    """Rotate an ENU vector about Up so that bearing ``yaw`` becomes straight ahead."""
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    return np.array([[c, -s, 0.0],
                     [s, c, 0.0],
                     [0.0, 0.0, 1.0]])


def _rx_cam(pitch_rad: float) -> np.ndarray:
    """Pitch about the camera's x (right) axis; positive tilts the camera up."""
    c, s = np.cos(pitch_rad), np.sin(pitch_rad)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, c, s],
                     [0.0, -s, c]])


def _rz_cam(roll_rad: float) -> np.ndarray:
    """Roll about the camera's z (forward) axis."""
    c, s = np.cos(roll_rad), np.sin(roll_rad)
    return np.array([[c, s, 0.0],
                     [-s, c, 0.0],
                     [0.0, 0.0, 1.0]])


@dataclass
class CameraModel:
    """A solved (or hand-specified) camera calibration."""

    # ENU origin: sea level below the rough camera position.
    ref_lat: float
    ref_lon: float

    # Image geometry the calibration was solved at.
    width: int
    height: int

    # Extrinsics
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    offset_e_m: float = 0.0
    offset_n_m: float = 0.0
    height_m: float = 10.0

    # Intrinsics
    focal_px: float = 1200.0
    cx: Optional[float] = None       # defaults to width / 2
    cy: Optional[float] = None       # defaults to height / 2
    k1: float = 0.0
    k2: float = 0.0

    # Atmosphere
    refraction_k: float = DEFAULT_REFRACTION_K

    # Provenance / diagnostics, carried through save/load for the README workflow.
    meta: dict = field(default_factory=dict)

    # -- derived ------------------------------------------------------------
    def __post_init__(self) -> None:
        if self.cx is None:
            self.cx = self.width / 2.0
        if self.cy is None:
            self.cy = self.height / 2.0

    @property
    def principal_point(self) -> tuple[float, float]:
        return float(self.cx), float(self.cy)

    @property
    def camera_enu(self) -> np.ndarray:
        """Camera position in the ENU frame, metres."""
        return np.array([self.offset_e_m, self.offset_n_m, self.height_m], dtype=float)

    @property
    def camera_latlon(self) -> tuple[float, float]:
        """Solved camera position as latitude/longitude."""
        from ..geodesy import enu_to_geodetic
        out = enu_to_geodetic(self.camera_enu.reshape(1, 3),
                              self.ref_lat, self.ref_lon, 0.0)
        return float(out[0, 0]), float(out[0, 1])

    @property
    def rotation(self) -> np.ndarray:
        """3x3 matrix taking an ENU displacement to camera coordinates."""
        return (_rz_cam(np.radians(self.roll_deg))
                @ _rx_cam(np.radians(self.pitch_deg))
                @ _ENU_TO_CAM_NORTH
                @ _rz_enu(np.radians(self.yaw_deg)))

    @property
    def hfov_deg(self) -> float:
        return float(np.degrees(2.0 * np.arctan(self.width / (2.0 * self.focal_px))))

    @property
    def vfov_deg(self) -> float:
        return float(np.degrees(2.0 * np.arctan(self.height / (2.0 * self.focal_px))))

    # -- projection ---------------------------------------------------------
    def relative_enu(self, target_enu: np.ndarray, refract: bool = True) -> np.ndarray:
        """Camera-to-target ENU displacement, optionally lifted for refraction."""
        target_enu = np.atleast_2d(np.asarray(target_enu, dtype=float))
        rel = target_enu - self.camera_enu
        if refract and self.refraction_k > 0:
            horizontal = np.hypot(rel[:, 0], rel[:, 1])
            rel = rel.copy()
            rel[:, 2] += refraction_lift_m(horizontal, self.refraction_k)
        return rel

    def project_enu(self, target_enu: np.ndarray, refract: bool = True):
        """Project ENU points to pixels.

        Returns ``(uv, in_front, depth)``: an ``(N, 2)`` pixel array, a boolean
        array that is False for anything at or behind the image plane, and the
        along-axis distance in metres. Pixels for points that are not in front
        are meaningless and must not be used.
        """
        rel = self.relative_enu(target_enu, refract=refract)
        cam = rel @ self.rotation.T

        depth = cam[:, 2]
        in_front = depth > 1e-6
        safe_depth = np.where(in_front, depth, 1.0)

        x_n = cam[:, 0] / safe_depth
        y_n = cam[:, 1] / safe_depth

        r2 = x_n * x_n + y_n * y_n
        radial = 1.0 + self.k1 * r2 + self.k2 * r2 * r2

        # Past the radius where the distortion polynomial turns over, a point
        # far outside the view folds back into the frame. Those pixels are as
        # meaningless as ones behind the camera, so report them the same way.
        in_front = in_front & (r2 <= self.max_valid_r2)

        u = self.cx + self.focal_px * x_n * radial
        v = self.cy + self.focal_px * y_n * radial
        return np.column_stack([u, v]), in_front, depth

    @property
    def max_valid_r2(self) -> float:
        """Largest squared undistorted radius at which distortion is monotonic.

        The distorted radius is ``r (1 + k1 r^2 + k2 r^4)``; its derivative
        ``1 + 3 k1 s + 5 k2 s^2`` (with ``s = r^2``) first reaches zero here.
        """
        a, b = 5.0 * self.k2, 3.0 * self.k1
        if abs(a) < 1e-15:
            return -1.0 / b if b < 0 else float("inf")
        disc = b * b - 4.0 * a
        if disc < 0:
            return float("inf")
        roots = [(-b - np.sqrt(disc)) / (2.0 * a), (-b + np.sqrt(disc)) / (2.0 * a)]
        positive = [r for r in roots if r > 0]
        return float(min(positive)) if positive else float("inf")

    def project_geodetic(self, lat, lon, alt_m, refract: bool = True):
        """Project geodetic positions straight to pixels."""
        enu = geodetic_to_enu(lat, lon, alt_m, self.ref_lat, self.ref_lon, 0.0)
        return self.project_enu(enu, refract=refract)

    def horizon_polyline(self, samples: int = 256,
                         margin_deg: float = 10.0) -> np.ndarray:
        """Pixel polyline of the true horizon, for the debug overlay.

        Sampled as sea-level points at the horizon distance and projected like
        anything else, so the line picks up roll and distortion.

        Only bearings inside the field of view are sampled. Sweeping the whole
        360-degree ring and sorting by x folds the half of the ring behind the
        camera back across the frame, which draws spurious spikes.
        """
        from ..geodesy import horizon_distance_m
        distance = horizon_distance_m(self.height_m, self.refraction_k)
        if distance <= 0:
            return np.empty((0, 2))

        half_span = min(89.0, self.hfov_deg / 2.0 + margin_deg)
        bearings = np.radians(np.linspace(self.yaw_deg - half_span,
                                          self.yaw_deg + half_span, samples))
        ring = np.column_stack([
            distance * np.sin(bearings) + self.offset_e_m,
            distance * np.cos(bearings) + self.offset_n_m,
            np.zeros(samples),
        ])
        uv, in_front, _ = self.project_enu(ring, refract=True)
        return uv[in_front]

    # -- scaling ------------------------------------------------------------
    def scaled_to(self, width: int, height: int) -> "CameraModel":
        """Return a copy rescaled to a different frame size.

        Calibration is often done on a still grabbed at one resolution while the
        live stream runs at another; distortion coefficients are defined on
        normalised coordinates so only the pixel-valued terms scale.
        """
        if width == self.width and height == self.height:
            return self
        sx = width / float(self.width)
        sy = height / float(self.height)
        if abs(sx - sy) > 1e-3:
            raise ValueError(
                f"cannot rescale calibration from {self.width}x{self.height} to "
                f"{width}x{height}: aspect ratio differs")
        clone = CameraModel(**{**asdict(self),
                               "width": width, "height": height,
                               "focal_px": self.focal_px * sx,
                               "cx": self.cx * sx, "cy": self.cy * sy})
        clone.meta = dict(self.meta)
        clone.meta["rescaled_from"] = [self.width, self.height]
        return clone

    # -- serialisation ------------------------------------------------------
    def to_dict(self) -> dict:
        data = asdict(self)
        data["derived"] = {
            "hfov_deg": round(self.hfov_deg, 4),
            "vfov_deg": round(self.vfov_deg, 4),
            "camera_lat": round(self.camera_latlon[0], 8),
            "camera_lon": round(self.camera_latlon[1], 8),
        }
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "CameraModel":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CameraModel":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_config(cls, cfg, width: int, height: int) -> "CameraModel":
        """An uncalibrated starting model built from the rough site config."""
        return cls(
            ref_lat=float(cfg.require("camera.lat")),
            ref_lon=float(cfg.require("camera.lon")),
            width=width,
            height=height,
            height_m=float(cfg.get("camera.height_m", 10.0)),
            refraction_k=float(cfg.get("camera.refraction_k", DEFAULT_REFRACTION_K)),
        )
