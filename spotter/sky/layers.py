"""Owns the sky layers and gives the pipeline one thing to call per frame."""

from __future__ import annotations

from datetime import datetime

from ..logging_setup import get_logger
from .lightning import BoltEffect, LightningLayer
from .satellites import SatelliteLayer

log = get_logger(__name__)


class SkyLayers:
    def __init__(self, cfg):
        self.satellites = SatelliteLayer(cfg)
        self.lightning = LightningLayer(cfg)

    def start(self) -> "SkyLayers":
        self.satellites.start()
        self.lightning.start()
        return self

    def stop(self) -> None:
        self.satellites.stop()
        self.lightning.stop()

    def project(self, when: datetime, model) -> tuple[list, list[BoltEffect]]:
        """Targets to label, and lightning flashes to draw, for one frame."""
        targets = []
        effects: list[BoltEffect] = []
        # A failure here costs one layer for one frame, never the frame.
        try:
            targets.extend(self.satellites.project(when, model))
        except Exception:
            log.exception("satellite projection failed")
        try:
            strikes, effects = self.lightning.project(when, model)
            targets.extend(strikes)
        except Exception:
            log.exception("lightning projection failed")
        return targets, effects

    def status(self) -> dict:
        return {"satellites": self.satellites.status(),
                "lightning": self.lightning.status()}

    def attributions(self) -> list[str]:
        return [text for text in (self.satellites.attribution(),
                                  self.lightning.attribution()) if text]
