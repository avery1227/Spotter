"""Label placement: keep boxes off each other and inside the frame.

Aircraft cluster on approach paths and ships bunch up in channels, so raw
labels at a fixed offset overlap constantly. The approach here:

1. Cap the label count, keeping the nearest targets (they are the ones a viewer
   can see, and the ones whose position is most accurate).
2. Place each label on whichever side of its marker has more room.
3. Relax overlaps iteratively, pushing boxes apart mostly vertically -- a
   stacked column of labels reads far better than a horizontal spread, and it
   keeps each label near the x of the thing it names.
4. Clamp everything inside the frame.

Fading is handled here too: a label that appears or disappears mid-frame pops
distractingly, so alpha ramps over a configurable time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ..logging_setup import get_logger

log = get_logger(__name__)

@dataclass
class LabelBox:
    """A placed label, ready to draw."""

    target_id: str
    anchor_x: float           # the marker it belongs to
    anchor_y: float
    x: float                  # top-left of the box
    y: float
    width: float
    height: float
    lines: list = field(default_factory=list)
    color: int = 0xFFFFFFFF
    alpha: float = 1.0
    target: object = None
    #: True when this label is fading out after its track went away.
    ghost: bool = False

    @property
    def cx(self) -> float:
        return self.x + self.width / 2

    @property
    def cy(self) -> float:
        return self.y + self.height / 2

    def overlaps(self, other: "LabelBox", gap: float = 0.0) -> bool:
        return not (self.x + self.width + gap <= other.x
                    or other.x + other.width + gap <= self.x
                    or self.y + self.height + gap <= other.y
                    or other.y + other.height + gap <= self.y)

    def attach_point(self) -> tuple[float, float]:
        """Where the leader line should meet the box: the nearest edge midpoint."""
        if self.anchor_x < self.x:
            return self.x, self.cy
        if self.anchor_x > self.x + self.width:
            return self.x + self.width, self.cy
        return self.cx, (self.y + self.height if self.anchor_y > self.cy else self.y)

class LabelAnimator:
    """Per-track alpha, so labels fade in and out instead of popping."""

    def __init__(self, fade_in_s: float = 0.6, fade_out_s: float = 1.0,
                 linger_s: float = 2.0):
        self.fade_in_s = max(1e-3, fade_in_s)
        self.fade_out_s = max(1e-3, fade_out_s)
        self.linger_s = linger_s
        self._alpha: dict[str, float] = {}
        self._last_seen: dict[str, float] = {}

    def update(self, visible_ids: Sequence[str], now: float,
               dt: float) -> dict[str, float]:
        """Advance every alpha and return the ones still worth drawing."""
        dt = max(0.0, min(dt, 1.0))   # a stalled pipeline must not jump the fades
        present = set(visible_ids)

        for track_id in present:
            self._last_seen[track_id] = now
            current = self._alpha.get(track_id, 0.0)
            self._alpha[track_id] = min(1.0, current + dt / self.fade_in_s)

        for track_id in list(self._alpha):
            if track_id in present:
                continue
            if now - self._last_seen.get(track_id, now) < self.linger_s:
                continue   # hold at full alpha briefly before fading
            self._alpha[track_id] -= dt / self.fade_out_s
            if self._alpha[track_id] <= 0.0:
                del self._alpha[track_id]
                self._last_seen.pop(track_id, None)

        return dict(self._alpha)

    def alpha_for(self, track_id: str) -> float:
        return self._alpha.get(track_id, 0.0)

    def ghosts(self, visible_ids: Sequence[str]) -> list[str]:
        """Tracks that are fading out and no longer present."""
        present = set(visible_ids)
        return [t for t, a in self._alpha.items() if t not in present and a > 0.0]

def initial_placement(box: LabelBox, leader: float, width: int, height: int) -> None:
    """Put the label on whichever side of the marker has more room."""
    prefer_left = box.anchor_x > width * 0.62
    dx = -(leader + box.width) if prefer_left else leader
    # Sit slightly above the marker by default; that is where the eye expects a
    # callout, and it keeps labels off the water surface where the target is.
    dy = -box.height * 0.5 - leader * 0.35

    box.x = box.anchor_x + dx
    box.y = box.anchor_y + dy

    if box.y < 4:
        box.y = box.anchor_y + leader * 0.35

def resolve_overlaps(boxes: list[LabelBox], width: int, height: int,
                     gap: float = 4.0, iterations: int = 60) -> int:
    """Push overlapping boxes apart. Returns how many still overlap at the end.

    Boxes earlier in the list are higher priority and move less, so the nearest
    targets stay closest to where they want to be.
    """
    if len(boxes) < 2:
        _clamp_all(boxes, width, height)
        return 0

    count = len(boxes)
    # Priority weight: index 0 barely moves, the last box moves freely.
    weights = [0.15 + 0.85 * (i / max(1, count - 1)) for i in range(count)]

    for _ in range(iterations):
        moved = False
        for i in range(count):
            for j in range(i + 1, count):
                a, b = boxes[i], boxes[j]
                if not a.overlaps(b, gap):
                    continue

                overlap_y = min(a.y + a.height + gap - b.y,
                                b.y + b.height + gap - a.y)
                overlap_x = min(a.x + a.width + gap - b.x,
                                b.x + b.width + gap - a.x)

                wa, wb = weights[i], weights[j]
                total = wa + wb or 1.0

                # Prefer vertical separation unless the boxes are far more
                # overlapped vertically than horizontally, in which case sliding
                # sideways is the shorter move.
                if overlap_y <= overlap_x * 1.8:
                    shift = overlap_y / 2 + 0.5
                    direction = -1.0 if a.cy < b.cy else 1.0
                    a.y += direction * shift * (wa / total) * 2
                    b.y -= direction * shift * (wb / total) * 2
                else:
                    shift = overlap_x / 2 + 0.5
                    direction = -1.0 if a.cx < b.cx else 1.0
                    a.x += direction * shift * (wa / total) * 2
                    b.x -= direction * shift * (wb / total) * 2
                moved = True

        _clamp_all(boxes, width, height)
        if not moved:
            break

    remaining = sum(1 for i in range(count) for j in range(i + 1, count)
                    if boxes[i].overlaps(boxes[j], gap))
    return remaining

def _clamp_all(boxes: Sequence[LabelBox], width: int, height: int,
               margin: float = 4.0) -> None:
    for box in boxes:
        box.x = min(max(box.x, margin), max(margin, width - box.width - margin))
        box.y = min(max(box.y, margin), max(margin, height - box.height - margin))
