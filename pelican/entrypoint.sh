#!/bin/bash
# Entrypoint for the Pelican Panel / Pterodactyl image.
#
# Wings hands us a $STARTUP string containing {{VARIABLE}} placeholders and an
# environment holding their values. Our job is to make /home/container look
# like a working install, expand that string, and exec it.

set -o pipefail
cd /home/container || exit 1

APP=/opt/spotter

say() { echo -e "\033[0;36m[spotter]\033[0m $*"; }
warn() { echo -e "\033[1;33m[spotter]\033[0m $*"; }

# ---------------------------------------------------------------------------
# First run: seed the files the user is expected to edit from the file manager.
# The application itself lives in /opt/spotter and is replaced wholesale when
# the image is pulled, so nothing here is ever overwritten on an upgrade.
# ---------------------------------------------------------------------------
if [ ! -f /home/container/config.yaml ]; then
    say "first run: writing a default config.yaml"
    cp "${APP}/config.example.yaml" /home/container/config.yaml
fi

mkdir -p /home/container/state /home/container/data/offline

if [ ! -f /home/container/points.csv ]; then
    # An empty points file, so the web UI has somewhere to write and the panel's
    # file manager shows it. Calibration fills it in.
    printf 'name,px,py,lat,lon,elev_m,enabled,note\n' > /home/container/points.csv
fi

# ---------------------------------------------------------------------------
# Report the environment, because the panel console is the only place an
# operator will look when something is wrong.
# ---------------------------------------------------------------------------
say "$(python -c 'import spotter; print("spotter " + spotter.__version__)' 2>/dev/null || echo spotter)"
say "python $(python -V 2>&1 | cut -d' ' -f2) · ffmpeg $(ffmpeg -version 2>/dev/null | head -1 | cut -d' ' -f3)"
say "internal IP: $(ip addr show 2>/dev/null | awk '/inet /{print $2}' | grep -v '^127' | head -1)"

if [ ! -f /home/container/calibration.json ]; then
    warn "no calibration.json yet. The web UI will come up and wait:"
    warn "open it on this server's port, calibrate, and Solve & save."
    warn "The pipeline starts by itself once the calibration exists."
fi

# ---------------------------------------------------------------------------
# Expand wings' {{VAR}} placeholders into ${VAR} and evaluate.
# ---------------------------------------------------------------------------
MODIFIED_STARTUP=$(echo "${STARTUP}" | sed -e 's/{{/${/g' -e 's/}}/}/g')
MODIFIED_STARTUP=$(eval echo "\"${MODIFIED_STARTUP}\"")

say "starting: ${MODIFIED_STARTUP}"
echo

# exec so signals from wings reach python directly rather than this shell.
exec env PYTHONPATH=/opt/spotter ${MODIFIED_STARTUP}
