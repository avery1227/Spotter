# syntax=docker/dockerfile:1
#
# Spotter runtime image.
#
# Two stages: wheels are built once against the build toolchain, then copied
# into a slim runtime that carries only ffmpeg, the shared libraries Skia and
# OpenCV need, and fonts.

FROM python:3.11-slim-bookworm AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
# The container never opens a window, so build against the headless OpenCV.
# requirements.txt asks for the GUI build because `calibrate.py click` needs it
# on a desktop; swapping the one line here keeps a single source of truth for
# versions while keeping the image free of GUI toolkits.
RUN sed -i 's/^opencv-python>=/opencv-python-headless>=/' requirements.txt \
    && cp requirements.txt /requirements-runtime.txt \
    && python -m pip install --upgrade pip wheel \
    && python -m pip wheel --wheel-dir /wheels -r requirements.txt


# ---------------------------------------------------------------------------

FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    SPOTTER_CONFIG=/app/config.yaml

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        # Skia and OpenCV need these at run time even in headless builds.
        # skia-python's wheel links against libEGL/libGLES regardless of
        # whether it ends up using the GPU, so the raster-only path still
        # fails to import without them.
        libglib2.0-0 \
        libgl1 \
        libegl1 \
        libgles2 \
        libfontconfig1 \
        libfreetype6 \
        # DejaVu is the font the default config asks for; without it Skia
        # silently substitutes and the layout shifts.
        fonts-dejavu-core \
        ca-certificates \
        tzdata \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /wheels /wheels
COPY --from=builder /requirements-runtime.txt /tmp/requirements.txt
RUN python -m pip install --no-index --find-links=/wheels -r /tmp/requirements.txt \
    && rm -rf /wheels /tmp/requirements.txt

# Run as a non-root user. The GID below matches Debian's `video` group, which
# is what /dev/dri/renderD128 belongs to for VAAPI hardware encoding.
RUN groupadd --system --gid 1000 spotter \
    && useradd --system --uid 1000 --gid spotter --create-home spotter \
    && usermod -aG video spotter

WORKDIR /app
COPY --chown=spotter:spotter spotter/ ./spotter/
COPY --chown=spotter:spotter calibrate.py ./
# Shipped as the template; the entrypoint/compose supplies config.yaml.
COPY --chown=spotter:spotter config.example.yaml ./
RUN cp config.example.yaml config.yaml && chown spotter:spotter config.yaml
COPY --chown=spotter:spotter tools/ ./tools/

# Mount points for the things that are per-site rather than per-image.
RUN mkdir -p /app/state /app/data && chown -R spotter:spotter /app/state /app/data
VOLUME ["/app/state", "/app/data"]

USER spotter

# Web UI (calibration + live monitor), when started with `run --web`.
EXPOSE 8090

# `doctor` exits non-zero when calibration or ffmpeg is missing, which is
# exactly the condition that makes the container useless.
HEALTHCHECK --interval=60s --timeout=30s --start-period=30s --retries=3 \
    CMD python -m spotter doctor > /dev/null 2>&1 || exit 1

ENTRYPOINT ["python", "-m", "spotter"]
CMD ["run"]


# ---------------------------------------------------------------------------
# Pelican Panel / Pterodactyl target
# ---------------------------------------------------------------------------
# Built with:  docker build --target pelican -t spotter:pelican .
#
# Wings expects a specific shape, and it is not the same as the standalone
# image's:
#
#   * the server's data directory is a volume mounted at /home/container, and
#     everything the user edits from the panel's file manager lives there;
#   * the process runs as an unprivileged `container` user;
#   * the startup command comes in as $STARTUP with {{VAR}} placeholders that
#     the entrypoint expands from the environment;
#   * stdout is the console, and the panel decides the server is "up" by
#     matching a line in it.
#
# The application itself stays in /opt/spotter, read-only. Only configuration
# and state live under /home/container, so pulling a newer image upgrades the
# code without touching a user's calibration.

FROM python:3.11-slim-bookworm AS pelican

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    SPOTTER_CONFIG=/home/container/config.yaml \
    SPOTTER_HOME=/opt/spotter \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libglib2.0-0 libgl1 libegl1 libgles2 \
        libfontconfig1 libfreetype6 fonts-dejavu-core \
        ca-certificates tzdata curl iproute2 tini \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /wheels /wheels
COPY --from=builder /requirements-runtime.txt /tmp/requirements.txt
RUN python -m pip install --no-index --find-links=/wheels -r /tmp/requirements.txt \
    && rm -rf /wheels /tmp/requirements.txt

# Wings runs the container as uid 988 by default; matching it keeps files the
# panel's file manager creates writable by us and vice versa.
RUN useradd -m -d /home/container -u 988 -s /bin/bash container

COPY --chown=root:root spotter/ /opt/spotter/spotter/
COPY --chown=root:root calibrate.py config.example.yaml /opt/spotter/
COPY --chown=root:root tools/ /opt/spotter/tools/
COPY --chown=root:root pelican/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV PYTHONPATH=/opt/spotter

USER container
WORKDIR /home/container

# tini reaps the ffmpeg child and forwards the signals wings sends to stop us.
ENTRYPOINT ["/usr/bin/tini", "-g", "--", "/entrypoint.sh"]
