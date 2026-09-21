#!/usr/bin/env bash
# Install the cost guards as root: the idle stop and the GPU runtime cap. cloud-init runs
# this at first boot from the copy it wrote to /opt/ces-revisions/guards, and setup.sh
# reruns it from the checkout; rerunning it is harmless.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

install -d -m 0755 /etc/ces-revisions /usr/local/lib/ces-revisions
# infra/bin/vm refuses to run where this marker exists.
touch /etc/ces-revisions/vm
install -m 0755 "$here/idle_stop.py" /usr/local/lib/ces-revisions/idle_stop.py
# Installed once, so a local edit to the window or thresholds survives a rerun.
if [ ! -e /etc/ces-revisions/idle-stop.env ]; then
  install -m 0644 "$here/idle-stop.env" /etc/ces-revisions/idle-stop.env
fi
install -m 0644 "$here/ces-idle-stop.service" "$here/ces-idle-stop.timer" \
  "$here/ces-gpu-cap.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now ces-idle-stop.timer
# Enabled, not started: the cap is scheduled at boot, so a rerun never pushes it back.
systemctl enable ces-gpu-cap.service
