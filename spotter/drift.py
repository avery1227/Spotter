"""Calibration drift detection by template matching.

The camera "essentially never moves", but "essentially" is doing work: mounts
sag, housings get bumped, someone cleans the lens. A few pixels of shift is
enough to put a label on the wrong ship at 20 km.

At calibration time we save small image patches around well-textured landmarks.
Periodically we look for each patch in the live frame, near where it used to be.
The *median* shift across patches is the drift estimate -- median rather than
mean because one patch landing on a moving boat or a waving tree should not
move the answer.

A single bad check does not raise the flag: haze, rain on the dome and low sun
all produce transient mismatches, so we require several consecutive failures.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .logging_setup import get_logger

log = get_logger(__name__)

MANIFEST_NAME = "patches.json"


@dataclass
class PatchRef:
    """A reference patch: where it was, and what it looked like."""

    name: str
    x: int                      # centre in full-resolution image coordinates
    y: int
    half_size: int
    file: str
    #: Variance of the grayscale patch; low-texture patches match poorly.
    texture: float = 0.0
    image: Optional[np.ndarray] = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict:
        return {"name": self.name, "x": self.x, "y": self.y,
                "half_size": self.half_size, "file": self.file,
                "texture": round(self.texture, 2)}


@dataclass
class PatchMatch:
    name: str
    dx: float
    dy: float
    confidence: float
    matched: bool

    def to_dict(self) -> dict:
        return {"name": self.name, "dx": round(self.dx, 2), "dy": round(self.dy, 2),
                "confidence": round(self.confidence, 3), "matched": self.matched}


@dataclass
class DriftReport:
    """Outcome of one drift check."""

    checked_at: float
    median_shift_px: float
    matches: list[PatchMatch] = field(default_factory=list)
    matched_count: int = 0
    exceeded: bool = False
    #: True once enough consecutive checks have exceeded the threshold.
    flagged: bool = False
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "median_shift_px": round(self.median_shift_px, 2),
            "matched": self.matched_count,
            "exceeded": self.exceeded,
            "flagged": self.flagged,
            "note": self.note,
            "patches": [m.to_dict() for m in self.matches],
        }


def _to_gray(image: np.ndarray) -> np.ndarray:
    import cv2

    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def select_patch_landmarks(image: np.ndarray,
                           candidates: Sequence[tuple[str, float, float]],
                           half_size: int = 32,
                           want: int = 6,
                           min_texture: float = 25.0) -> list[tuple[str, int, int, float]]:
    """Choose which landmarks to use as drift references.

    Two criteria, in order: the patch must actually have texture to match on
    (a featureless patch of water or sky matches everywhere equally well), and
    the chosen set should be spread across the frame so a rotation shows up as
    disagreement between patches rather than a uniform translation.
    """
    gray = _to_gray(image)
    height, width = gray.shape[:2]

    scored: list[tuple[str, int, int, float]] = []
    for name, px, py in candidates:
        x, y = int(round(px)), int(round(py))
        if not (half_size <= x < width - half_size and
                half_size <= y < height - half_size):
            continue
        patch = gray[y - half_size:y + half_size + 1,
                     x - half_size:x + half_size + 1]
        texture = float(patch.astype(np.float32).var())
        if texture < min_texture:
            log.debug("skipping low-texture landmark",
                      extra={"name": name, "texture": round(texture, 1)})
            continue
        scored.append((name, x, y, texture))

    if not scored:
        return []

    # Greedy farthest-point selection, seeded with the most textured patch.
    scored.sort(key=lambda s: s[3], reverse=True)
    chosen = [scored[0]]
    remaining = scored[1:]
    while remaining and len(chosen) < want:
        best_index, best_score = 0, -1.0
        for i, cand in enumerate(remaining):
            distance = min(np.hypot(cand[1] - c[1], cand[2] - c[2]) for c in chosen)
            # Distance dominates; texture breaks ties between equally spread options.
            score = distance * (1.0 + 0.15 * np.log1p(cand[3]))
            if score > best_score:
                best_index, best_score = i, score
        chosen.append(remaining.pop(best_index))
    return chosen


def save_reference_patches(image: np.ndarray,
                           candidates: Sequence[tuple[str, float, float]],
                           out_dir: str | Path,
                           half_size: int = 32,
                           want: int = 6,
                           min_texture: float = 25.0) -> list[PatchRef]:
    """Extract and persist drift reference patches. Returns what was saved."""
    import cv2

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("patch_*.png"):
        stale.unlink()

    selected = select_patch_landmarks(image, candidates, half_size, want, min_texture)
    refs: list[PatchRef] = []
    for index, (name, x, y, texture) in enumerate(selected):
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)[:40]
        filename = f"patch_{index:02d}_{safe}.png"
        patch = image[y - half_size:y + half_size + 1,
                      x - half_size:x + half_size + 1]
        cv2.imwrite(str(out_dir / filename), patch[:, :, :3])
        refs.append(PatchRef(name=name, x=x, y=y, half_size=half_size,
                             file=filename, texture=texture))

    manifest = {
        "created_at": time.time(),
        "image_width": int(image.shape[1]),
        "image_height": int(image.shape[0]),
        "half_size": half_size,
        "patches": [r.to_dict() for r in refs],
    }
    (out_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2),
                                        encoding="utf-8")
    log.info("saved drift reference patches",
             extra={"count": len(refs), "dir": str(out_dir),
                    "names": [r.name for r in refs]})
    return refs


def load_reference_patches(out_dir: str | Path) -> tuple[list[PatchRef], dict]:
    import cv2

    out_dir = Path(out_dir)
    manifest_path = out_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        return [], {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    refs: list[PatchRef] = []
    for entry in manifest.get("patches", []):
        image = cv2.imread(str(out_dir / entry["file"]), cv2.IMREAD_COLOR)
        if image is None:
            log.warning("drift patch file missing", extra={"file": entry["file"]})
            continue
        refs.append(PatchRef(name=entry["name"], x=int(entry["x"]), y=int(entry["y"]),
                             half_size=int(entry["half_size"]), file=entry["file"],
                             texture=float(entry.get("texture", 0.0)),
                             image=image))
    return refs, manifest


class DriftDetector:
    """Periodically re-locates reference patches and reports camera movement."""

    def __init__(self, cfg):
        drift_cfg = cfg.sub("drift")
        self.enabled = bool(drift_cfg.get("enabled", True))
        self.interval_s = float(drift_cfg.get("check_interval_s", 60.0))
        self.search_radius = int(drift_cfg.get("search_radius_px", 48))
        self.min_confidence = float(drift_cfg.get("min_confidence", 0.55))
        self.warn_shift_px = float(drift_cfg.get("warn_shift_px", 6.0))
        self.consecutive_needed = int(drift_cfg.get("consecutive_checks", 3))
        self.hide_labels = bool(drift_cfg.get("hide_labels_on_drift", False))
        self.patches_dir = cfg.path("drift.patches_dir", "./state/drift_patches")

        self.refs: list[PatchRef] = []
        self.manifest: dict = {}
        self.scale: float = 1.0

        self._last_check: float = 0.0
        self._consecutive_exceeded = 0
        self.flagged = False
        self.last_report: Optional[DriftReport] = None

        if self.enabled:
            self._load()

    def _load(self) -> None:
        self.refs, self.manifest = load_reference_patches(self.patches_dir)
        if not self.refs:
            log.warning("drift detection enabled but no reference patches found; "
                        "run 'calibrate.py solve' to create them",
                        extra={"dir": str(self.patches_dir)})
            self.enabled = False
        else:
            log.info("drift detection armed", extra={
                "patches": len(self.refs), "interval_s": self.interval_s,
                "threshold_px": self.warn_shift_px})

    @property
    def active(self) -> bool:
        return self.enabled and bool(self.refs)

    def due(self, now: Optional[float] = None) -> bool:
        if not self.active:
            return False
        now = now if now is not None else time.monotonic()
        return (now - self._last_check) >= self.interval_s

    def check(self, image: np.ndarray, force: bool = False) -> Optional[DriftReport]:
        """Run a check if one is due. Returns None when it is not."""
        now = time.monotonic()
        if not force and not self.due(now):
            return None
        self._last_check = now

        # Calibration may have been done at a different resolution.
        ref_width = self.manifest.get("image_width") or image.shape[1]
        scale = image.shape[1] / float(ref_width)

        matches = [self._match_one(image, ref, scale) for ref in self.refs]
        good = [m for m in matches if m.matched]

        report = DriftReport(checked_at=now, median_shift_px=0.0, matches=matches,
                             matched_count=len(good))

        if len(good) < 2:
            report.note = ("too few patches matched to judge drift "
                           f"({len(good)}/{len(matches)})")
            # Not enough evidence either way: do not accumulate toward a flag.
            log.debug("drift check inconclusive", extra=report.to_dict())
            self.last_report = report
            return report

        shifts = np.array([np.hypot(m.dx, m.dy) for m in good])
        report.median_shift_px = float(np.median(shifts))
        report.exceeded = report.median_shift_px > self.warn_shift_px

        if report.exceeded:
            self._consecutive_exceeded += 1
        else:
            self._consecutive_exceeded = 0

        was_flagged = self.flagged
        self.flagged = self._consecutive_exceeded >= self.consecutive_needed
        report.flagged = self.flagged

        if self.flagged and not was_flagged:
            log.warning("CALIBRATION DRIFT detected", extra={
                "median_shift_px": round(report.median_shift_px, 2),
                "threshold_px": self.warn_shift_px,
                "consecutive": self._consecutive_exceeded,
                "per_patch": [m.to_dict() for m in good],
            })
        elif was_flagged and not self.flagged:
            log.info("calibration drift cleared",
                     extra={"median_shift_px": round(report.median_shift_px, 2)})
        else:
            log.debug("drift check", extra=report.to_dict())

        self.last_report = report
        return report

    def _match_one(self, image: np.ndarray, ref: PatchRef, scale: float) -> PatchMatch:
        import cv2

        gray = _to_gray(image)
        template = _to_gray(ref.image)
        if scale != 1.0:
            size = (max(3, int(round(template.shape[1] * scale))),
                    max(3, int(round(template.shape[0] * scale))))
            template = cv2.resize(template, size, interpolation=cv2.INTER_AREA)

        half_w, half_h = template.shape[1] // 2, template.shape[0] // 2
        cx, cy = int(round(ref.x * scale)), int(round(ref.y * scale))
        radius = int(round(self.search_radius * scale))

        x0 = max(0, cx - half_w - radius)
        y0 = max(0, cy - half_h - radius)
        x1 = min(gray.shape[1], cx + half_w + radius + 1)
        y1 = min(gray.shape[0], cy + half_h + radius + 1)
        window = gray[y0:y1, x0:x1]

        if (window.shape[0] < template.shape[0] or
                window.shape[1] < template.shape[1]):
            return PatchMatch(ref.name, 0.0, 0.0, 0.0, False)

        result = cv2.matchTemplate(window, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        found_x = x0 + max_loc[0] + half_w
        found_y = y0 + max_loc[1] + half_h
        dx = (found_x - cx) / scale
        dy = (found_y - cy) / scale

        matched = bool(max_val >= self.min_confidence)
        return PatchMatch(ref.name, float(dx), float(dy), float(max_val), matched)

    def status(self) -> dict:
        if not self.active:
            return {"enabled": False}
        return {
            "enabled": True,
            "flagged": self.flagged,
            "consecutive_exceeded": self._consecutive_exceeded,
            "consecutive_needed": self.consecutive_needed,
            "threshold_px": self.warn_shift_px,
            "exceeded": bool(self.last_report and self.last_report.exceeded),
            "median_shift_px": (round(self.last_report.median_shift_px, 2)
                                if self.last_report else None),
        }
