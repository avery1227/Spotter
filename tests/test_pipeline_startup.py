"""Starting without a calibration.

A fresh deployment has no calibration.json. The web UI is how you make one, so
exiting in that state -- which the pipeline used to do -- killed the UI a few
seconds after it came up, and under a panel's restart policy became a crash
loop the operator could not get out of.
"""

import threading
import time

from spotter.config import Config
from spotter.pipeline import Pipeline
from spotter.web.frames import FrameHub


def make_pipeline(tmp_path, with_ui: bool) -> Pipeline:
    cfg = Config({
        "camera": {"lat": 41.27, "lon": -72.5, "height_m": 15.0},
        "calibration": {"path": str(tmp_path / "calibration.json")},
        "tracks": {"adsb": {"enabled": False}, "ais": {"enabled": False}},
        "drift": {"enabled": False},
    }, root=tmp_path)
    return Pipeline(cfg, stop_event=threading.Event(),
                    frame_hub=FrameHub() if with_ui else None)


def test_without_the_ui_a_missing_calibration_is_an_error(tmp_path):
    pipeline = make_pipeline(tmp_path, with_ui=False)
    started = time.monotonic()
    assert pipeline.run() == 2
    assert time.monotonic() - started < 5, "should fail fast, not wait"


def test_with_the_ui_it_waits_and_then_proceeds(tmp_path):
    pipeline = make_pipeline(tmp_path, with_ui=True)
    result = {}

    thread = threading.Thread(
        target=lambda: result.setdefault("ok", pipeline._wait_for_calibration()))
    thread.start()

    time.sleep(0.5)
    assert thread.is_alive(), "it exited instead of waiting for a calibration"
    assert pipeline.waiting_for_calibration is True
    assert pipeline.web_status()["waiting_for_calibration"] is True

    # Saving a calibration from the web UI is what ends the wait.
    (tmp_path / "calibration.json").write_text("{}", encoding="utf-8")
    thread.join(timeout=5)

    assert result.get("ok") is True
    assert pipeline.waiting_for_calibration is False


def test_stopping_while_waiting_is_a_clean_exit(tmp_path):
    pipeline = make_pipeline(tmp_path, with_ui=True)
    threading.Timer(0.5, pipeline.stop_event.set).start()
    assert pipeline.run() == 0


def test_an_existing_calibration_does_not_wait(tmp_path):
    (tmp_path / "calibration.json").write_text("{}", encoding="utf-8")
    pipeline = make_pipeline(tmp_path, with_ui=True)
    started = time.monotonic()
    assert pipeline._wait_for_calibration() is True
    assert time.monotonic() - started < 0.5
