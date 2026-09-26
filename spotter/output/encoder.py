"""Encode composited frames and push them to RTMP.

Frames are handed to ffmpeg as raw BGRA on stdin -- the same buffer Skia drew
into, so nothing is converted or copied on the way out.

Two things make this more than a pipe:

*   **Steady output cadence.** The input arrives in five-second bursts (one HLS
    segment at a time) and sometimes stops entirely. A writer thread meters
    frames out at exactly the target rate and repeats the last frame when the
    queue runs dry, so the RTMP endpoint never sees a stalled stream. The small
    frame queue also back-pressures decoding, which is what keeps memory flat:
    at 1080p a single BGRA frame is 8 MB, so buffering a whole segment would
    cost gigabytes.

*   **Audio passthrough.** Audio packets demuxed from the same segments are
    remuxed to an ADTS elementary stream and fed to ffmpeg on a second pipe, so
    they are copied rather than re-encoded and stay aligned with the video they
    came from.
"""

from __future__ import annotations

import os
import queue
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..logging_setup import get_logger

log = get_logger(__name__)

#: Preference order when `output.encoder` is `auto`. Hardware first.
HARDWARE_ENCODERS = ("h264_nvenc", "h264_qsv", "h264_vaapi")

#: How much audio ffmpeg is allowed to read while probing the second pipe, and
#: therefore how much the pipeline must buffer before starting the encoder.
#: Comfortably below a 64 KB pipe buffer so the prime never blocks.
AUDIO_PROBE_BYTES = 16384

#: Presets differ per encoder family; libx264's "veryfast" is not NVENC's.
ENCODER_PRESETS = {
    "h264_nvenc": {"veryfast": "p2", "fast": "p3", "medium": "p4", "slow": "p6"},
}


@dataclass
class EncoderCaps:
    available: list[str]
    chosen: str
    reason: str


def ffmpeg_path() -> str:
    return os.environ.get("FFMPEG_BINARY") or shutil.which("ffmpeg") or "ffmpeg"


def detect_encoders(timeout_s: float = 10.0) -> list[str]:
    """Ask ffmpeg which H.264 encoders this build actually has."""
    try:
        result = subprocess.run([ffmpeg_path(), "-hide_banner", "-encoders"],
                                capture_output=True, text=True, timeout=timeout_s)
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("could not query ffmpeg encoders", extra={"error": str(exc)})
        return []
    found = []
    for line in result.stdout.splitlines():
        for name in HARDWARE_ENCODERS + ("libx264",):
            if f" {name} " in line and name not in found:
                found.append(name)
    return found


def choose_encoder(requested: str) -> EncoderCaps:
    available = detect_encoders()
    if requested and requested != "auto":
        if available and requested not in available:
            log.warning("requested encoder is not in this ffmpeg build; "
                        "trying it anyway",
                        extra={"requested": requested, "available": available})
        return EncoderCaps(available, requested, "configured explicitly")

    for name in HARDWARE_ENCODERS:
        if name in available:
            return EncoderCaps(available, name, "hardware encoder detected")
    return EncoderCaps(available, "libx264", "no hardware encoder found")


class RTMPOutput:
    """Owns the ffmpeg child process and the frame-pacing writer thread."""

    def __init__(self, cfg, width: int, height: int, fps: float,
                 audio_codec: str = "", audio_rate: int = 0,
                 audio_channels: int = 0, audio_lead_s: float = 0.0):
        self.cfg = cfg
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps) if fps and fps > 0 else 30.0

        out = cfg.sub("output")
        self.enabled = bool(out.get("enabled", True))
        self.url = str(out.get("rtmp_url", ""))
        self.preset = str(out.get("preset", "veryfast"))
        self.bitrate = str(out.get("bitrate", "6000k"))
        self.maxrate = str(out.get("maxrate", self.bitrate))
        self.bufsize = str(out.get("bufsize", "12000k"))
        self.gop_seconds = float(out.get("gop_seconds", 2))
        self.pix_fmt = str(out.get("pix_fmt", "yuv420p"))
        self.extra_args = list(out.get("extra_args", []) or [])
        self.local_copy = cfg.path("output.local_copy")

        stall = cfg.sub("output.stall")
        self.repeat_last_frame = bool(stall.get("repeat_last_frame", True))
        self.badge_after_s = float(stall.get("badge_after_s", 3.0))

        self.caps = choose_encoder(str(out.get("encoder", "auto")))

        # Audio passthrough needs a second pipe, which needs fd inheritance.
        # That is POSIX-only; on Windows we run without audio and say so.
        self.audio_enabled = (
            str(cfg.get("video.audio", "passthrough")) == "passthrough"
            and bool(audio_codec) and os.name == "posix")
        if (str(cfg.get("video.audio", "passthrough")) == "passthrough"
                and audio_codec and os.name != "posix"):
            log.warning("audio passthrough needs a POSIX second pipe; "
                        "running video-only on this platform")
        self.audio_codec = audio_codec
        self.audio_rate = audio_rate
        self.audio_channels = audio_channels
        # Seconds of audio buffered before the first video frame was submitted.
        # Both inputs start at timestamp zero, so without correcting for it the
        # audio would run this far ahead of the picture for the whole stream.
        self.audio_lead_s = float(audio_lead_s)

        self._process: Optional[subprocess.Popen] = None
        self._audio_writer = None
        self._audio_fd: Optional[int] = None
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue(
            maxsize=int(out.get("frame_queue", 8)))
        self._stop = threading.Event()
        self._writer: Optional[threading.Thread] = None
        self._stderr_reader: Optional[threading.Thread] = None

        self.frames_written = 0
        self.frames_repeated = 0
        self.frames_dropped = 0
        self.audio_bytes = 0
        self.last_frame_at = 0.0
        self._last_frame: Optional[np.ndarray] = None
        self.starved_since: Optional[float] = None

    # -- command line -------------------------------------------------------
    def build_command(self, audio_fd: Optional[int] = None) -> list[str]:
        encoder = self.caps.chosen
        gop = max(1, int(round(self.fps * self.gop_seconds)))

        cmd = [ffmpeg_path(), "-hide_banner", "-loglevel", "warning", "-nostdin",
               "-y",
               "-f", "rawvideo",
               "-pixel_format", "bgra",
               "-video_size", f"{self.width}x{self.height}",
               "-framerate", f"{self.fps:.6f}",
               "-i", "pipe:0"]

        if audio_fd is not None:
            # The ADTS *demuxer* is called "aac"; "adts" is the muxer-only name
            # and ffmpeg rejects it as an input format.
            #
            # The probe size is a balancing act. ffmpeg opens its inputs in
            # order, so it reads this much audio before it will touch a single
            # video frame on stdin -- and the only thing producing audio is the
            # decode loop that is simultaneously blocked submitting video. Too
            # large and the two deadlock; too small and ffmpeg cannot parse even
            # one ADTS header ("Could not find codec parameters"). 16 KB is a
            # few dozen AAC frames, and the pipeline primes at least that much
            # audio before the encoder is started, so the probe is satisfied
            # from the pipe buffer without the video path having to advance.
            audio_input = ["-f", "aac", "-probesize", str(AUDIO_PROBE_BYTES),
                           "-analyzeduration", "500000"]
            if self.audio_lead_s > 0.01:
                # Shift the audio back by the lead. ffmpeg drops the packets
                # that land before zero, which discards exactly the priming
                # audio that has no matching video.
                audio_input = ["-itsoffset", f"-{self.audio_lead_s:.3f}"] + audio_input
            cmd += audio_input + ["-i", f"pipe:{audio_fd}"]

        cmd += ["-map", "0:v:0"]
        if audio_fd is not None:
            cmd += ["-map", "1:a:0", "-c:a", "copy"]

        cmd += ["-c:v", encoder, "-pix_fmt", self.pix_fmt,
                "-b:v", self.bitrate, "-maxrate", self.maxrate,
                "-bufsize", self.bufsize,
                "-g", str(gop), "-keyint_min", str(gop),
                # A live encoder must never wait for future frames.
                "-sc_threshold", "0"]

        if encoder == "libx264":
            cmd += ["-preset", self.preset, "-tune", "zerolatency",
                    "-profile:v", "high"]
        elif encoder == "h264_nvenc":
            cmd += ["-preset", ENCODER_PRESETS["h264_nvenc"].get(self.preset, "p4"),
                    "-rc", "cbr", "-tune", "ll"]
        elif encoder == "h264_qsv":
            cmd += ["-preset", self.preset, "-look_ahead", "0"]
        elif encoder == "h264_vaapi":
            # VAAPI needs frames uploaded to the GPU before it can encode them.
            cmd = cmd[:cmd.index("-c:v")] + [
                "-vaapi_device", str(self.cfg.get("output.vaapi_device",
                                                  "/dev/dri/renderD128")),
                "-vf", "format=nv12,hwupload"] + cmd[cmd.index("-c:v"):]

        cmd += self.extra_args

        muxer = self.output_format()
        if self.local_copy:
            self.local_copy.parent.mkdir(parents=True, exist_ok=True)
            cmd += ["-f", "tee", "-flags", "+global_header",
                    f"[f={muxer}]{self.url}|[f=mp4]{self.local_copy}"]
        else:
            cmd += ["-f", muxer, self.url]
        return cmd

    def output_format(self) -> str:
        """Pick a muxer from the destination.

        RTMP means FLV. Anything else is usually a local file during testing or
        a MediaMTX endpoint reached over another protocol, so honour the
        extension rather than forcing FLV onto it.
        """
        url = self.url.lower()
        if url.startswith(("rtmp://", "rtmps://")):
            return "flv"
        if url.startswith("srt://") or url.startswith("udp://"):
            return "mpegts"
        for extension, muxer in ((".mp4", "mp4"), (".mkv", "matroska"),
                                 (".flv", "flv"), (".ts", "mpegts"),
                                 (".m3u8", "hls")):
            if url.endswith(extension):
                return muxer
        return "flv"

    # -- lifecycle ----------------------------------------------------------
    def start(self) -> "RTMPOutput":
        if not self.enabled:
            # Still run the writer thread. It is what meters frames to the
            # target rate and back-pressures decoding; without it the decode
            # loop sprints through a whole segment and then waits for the next,
            # which looks like the preview playing fast and freezing. Disabling
            # output should mean "discard the frames", not "stop pacing".
            log.info("output disabled; pacing frames and discarding them")
            self._writer = threading.Thread(target=self._write_loop,
                                            name="frame-writer", daemon=True)
            self._writer.start()
            return self
        if not self.url:
            raise ValueError("output.rtmp_url is not set")

        self._spawn()

        # Audio is a nice-to-have; video is the product. If ffmpeg refuses the
        # audio input for any reason it exits immediately and takes the video
        # with it, so fall back to video-only rather than losing the stream.
        if self.audio_enabled and not self._survived_startup():
            log.error("encoder exited immediately with audio configured; "
                      "retrying without audio passthrough")
            self._teardown_process()
            self.audio_enabled = False
            self._spawn()
            if not self._survived_startup():
                raise RuntimeError(
                    "ffmpeg exited immediately; see the preceding 'ffmpeg' log "
                    "lines for the reason")

        self._writer = threading.Thread(target=self._write_loop,
                                        name="frame-writer", daemon=True)
        self._writer.start()
        return self

    def _survived_startup(self, grace_s: float = 2.0) -> bool:
        """True if ffmpeg is still alive a moment after being spawned."""
        deadline = time.monotonic() + grace_s
        while time.monotonic() < deadline:
            if self._process is None:
                return False
            if self._process.poll() is not None:
                return False
            time.sleep(0.1)
        return True

    def _teardown_process(self) -> None:
        if self._audio_writer is not None:
            try:
                self._audio_writer.close()
            except Exception:
                pass
            self._audio_writer = None
        if self._audio_fd is not None:
            try:
                os.close(self._audio_fd)
            except OSError:
                pass
            self._audio_fd = None
        if self._process is not None:
            try:
                self._process.kill()
                self._process.wait(timeout=5.0)
            except Exception:
                pass
            self._process = None

    def _spawn(self) -> None:
        pass_fds: tuple = ()
        audio_fd = None
        if self.audio_enabled:
            read_fd, write_fd = os.pipe()
            os.set_inheritable(read_fd, True)
            audio_fd = read_fd
            pass_fds = (read_fd,)
            self._audio_fd = write_fd

        cmd = self.build_command(audio_fd)
        log.info("starting encoder", extra={
            "encoder": self.caps.chosen, "reason": self.caps.reason,
            "available": self.caps.available,
            "size": f"{self.width}x{self.height}", "fps": round(self.fps, 3),
            "audio": self.audio_codec if self.audio_enabled else None,
            "url": _redact(self.url)})
        log.debug("ffmpeg command", extra={"cmd": " ".join(
            _redact(part) for part in cmd)})

        self._process = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE, pass_fds=pass_fds, bufsize=0)

        if audio_fd is not None:
            os.close(audio_fd)  # the child owns it now
            self._open_audio_writer()

        self._stderr_reader = threading.Thread(target=self._drain_stderr,
                                               name="ffmpeg-stderr", daemon=True)
        self._stderr_reader.start()

    def _open_audio_writer(self) -> None:
        """Wrap the audio pipe in a plain binary file object.

        The ADTS framing is done upstream, in the decoder, because it needs the
        source container's audio stream as a template. Here we only ferry bytes.
        """
        try:
            self._audio_writer = os.fdopen(self._audio_fd, "wb", buffering=0)
            self._audio_fd = None          # the file object owns it now
            log.info("audio passthrough armed",
                     extra={"codec": self.audio_codec, "rate": self.audio_rate})
        except Exception as exc:
            log.warning("could not open audio pipe; continuing without audio",
                        extra={"error": str(exc)})
            self._audio_writer = None
            self.audio_enabled = False

    def stop(self) -> None:
        self._stop.set()
        if self._writer is not None:
            self._writer.join(timeout=5.0)

        if self._audio_writer is not None:
            try:
                self._audio_writer.close()
            except Exception:
                pass
            self._audio_writer = None

        process = self._process
        if process is not None:
            try:
                if process.stdin:
                    process.stdin.close()
            except (OSError, BrokenPipeError):
                pass
            try:
                process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                log.warning("ffmpeg did not exit; terminating")
                process.terminate()
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    process.kill()
        log.info("encoder stopped", extra=self.status())

    # -- frame path ---------------------------------------------------------
    def submit(self, frame: np.ndarray, block: bool = True) -> bool:
        """Hand a composited frame to the writer.

        Blocking here is deliberate: it back-pressures decoding so we do not
        build an unbounded queue of 8 MB frames when the encoder falls behind.
        """
        try:
            self._queue.put(frame, block=block, timeout=5.0 if block else None)
            return True
        except queue.Full:
            self.frames_dropped += 1
            return False

    def write_audio(self, data: bytes) -> None:
        """Write already-framed ADTS bytes to ffmpeg's second pipe."""
        writer = self._audio_writer
        if writer is None or not data:
            return
        try:
            writer.write(data)
            self.audio_bytes += len(data)
        except (BrokenPipeError, OSError, ValueError) as exc:
            log.warning("audio passthrough failed; continuing video-only",
                        extra={"error": str(exc)})
            self._audio_writer = None
            self.audio_enabled = False

    def _write_loop(self) -> None:
        """Emit frames at a fixed cadence regardless of what the input is doing."""
        interval = 1.0 / self.fps
        # Absolute schedule rather than sleep(interval), so scheduling jitter
        # does not accumulate into a drifting output frame rate.
        next_deadline = time.monotonic()

        while not self._stop.is_set():
            # Wait for this frame's slot BEFORE taking one off the queue.
            # Using the deadline as the queue's timeout instead would not pace
            # anything: Queue.get returns the moment an item exists, so a full
            # queue would be drained at whatever speed the encoder accepts.
            remaining = next_deadline - time.monotonic()
            if remaining > 0 and self._stop.wait(remaining):
                return

            frame = None
            try:
                frame = self._queue.get_nowait()
            except queue.Empty:
                pass

            if frame is not None:
                self._last_frame = frame
                self.last_frame_at = time.monotonic()
                self.starved_since = None
            elif self.repeat_last_frame and self._last_frame is not None:
                frame = self._last_frame
                self.frames_repeated += 1
                if self.starved_since is None:
                    self.starved_since = time.monotonic()
            else:
                next_deadline += interval
                continue

            if not self._write_frame(frame):
                return

            next_deadline += interval
            # If we have fallen more than a second behind (a long encoder
            # stall), skip ahead rather than sprinting to catch up.
            now = time.monotonic()
            if next_deadline < now - 1.0:
                next_deadline = now

    def _write_frame(self, frame: np.ndarray) -> bool:
        if not self.enabled:
            self.frames_written += 1
            return True
        process = self._process
        if process is None or process.stdin is None:
            return False
        try:
            process.stdin.write(frame.tobytes())
            self.frames_written += 1
            return True
        except (BrokenPipeError, OSError) as exc:
            log.error("ffmpeg pipe closed", extra={
                "error": str(exc), "frames_written": self.frames_written,
                "returncode": process.poll()})
            self._stop.set()
            return False

    def _drain_stderr(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        for raw in iter(process.stderr.readline, b""):
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            lowered = line.lower()
            if "error" in lowered or "failed" in lowered:
                log.error("ffmpeg", extra={"line": line})
            else:
                log.warning("ffmpeg", extra={"line": line})

    # -- health -------------------------------------------------------------
    @property
    def alive(self) -> bool:
        if not self.enabled:
            return True
        return self._process is not None and self._process.poll() is None

    def starved_for_s(self) -> float:
        return 0.0 if self.starved_since is None else (
            time.monotonic() - self.starved_since)

    def should_show_badge(self) -> bool:
        return self.starved_for_s() >= self.badge_after_s

    def held_frame(self) -> Optional[np.ndarray]:
        """The frame being repeated while starved (the last one written)."""
        return self._last_frame

    def hold_frame(self, frame: np.ndarray) -> None:
        """Replace the frame repeated while starved, without ending the stall.

        submit() would count as fresh input and reset the stall clock; this only
        swaps what gets repeated, so the badge can be redrawn with its counter.
        """
        self._last_frame = frame

    def status(self) -> dict:
        return {
            "encoder": self.caps.chosen,
            "written": self.frames_written,
            "repeated": self.frames_repeated,
            "dropped": self.frames_dropped,
            "queue": self._queue.qsize(),
            "starved_s": round(self.starved_for_s(), 1),
            "audio_kb": round(self.audio_bytes / 1024, 1),
            "alive": self.alive,
        }


def _redact(text: str) -> str:
    """Keep stream keys out of the logs."""
    if not isinstance(text, str) or "://" not in text:
        return text
    head, _, tail = text.rpartition("/")
    if len(tail) >= 8 and ("rtmp" in head or "rtmps" in head):
        return f"{head}/{tail[:4]}...{tail[-2:]}"
    return text
