#!/usr/bin/env bash
# Installs Node Exporter on the host as a systemd user service. No root needed.
# It sees the real host CPU, memory, disks and sensors, which is why it is the one
# monitoring component outside Docker.
#
#   ./install.sh               install or upgrade, then start
#   ./install.sh --uninstall   stop and remove
#
# It listens on loopback and on the Docker bridge gateway, where the Prometheus
# container reaches it, so it is not exposed on your LAN.
# Koustav Kumar Mondal <erkoustavkumar@gmail.com>
set -euo pipefail

VERSION=1.12.1
SERVICE=qputwin-node-exporter
BASE="${XDG_DATA_HOME:-$HOME/.local/share}/qputwin-monitoring"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
UNIT="$UNIT_DIR/$SERVICE.service"

if [[ "${1:-}" == "--uninstall" ]]; then
  systemctl --user disable --now "$SERVICE" 2>/dev/null || true
  rm -f "$UNIT"
  rm -rf "$BASE"
  systemctl --user daemon-reload
  echo "Removed $SERVICE."
  exit 0
fi

case "$(uname -m)" in
  x86_64) ARCH=amd64 ;;
  aarch64) ARCH=arm64 ;;
  *) echo "Unsupported architecture: $(uname -m)" >&2; exit 1 ;;
esac

NAME="node_exporter-$VERSION.linux-$ARCH"
BIN="$BASE/$NAME/node_exporter"
if [[ ! -x "$BIN" ]]; then
  URL="https://github.com/prometheus/node_exporter/releases/download/v$VERSION"
  WORK="$(mktemp -d)"
  trap 'rm -rf "$WORK"' EXIT
  echo "Downloading Node Exporter $VERSION"
  curl -fsSL -o "$WORK/$NAME.tar.gz" "$URL/$NAME.tar.gz"
  curl -fsSL -o "$WORK/sha256sums.txt" "$URL/sha256sums.txt"
  (cd "$WORK" && grep " $NAME.tar.gz\$" sha256sums.txt | sha256sum -c --quiet -)
  mkdir -p "$BASE"
  tar -xzf "$WORK/$NAME.tar.gz" -C "$BASE"
fi

GATEWAY="$(docker network inspect bridge -f '{{(index .IPAM.Config 0).Gateway}}')"

mkdir -p "$UNIT_DIR"
cat > "$UNIT" <<EOF
[Unit]
Description=Node Exporter for QPUTwin monitoring
After=default.target

[Service]
# Restarts until Docker has created the bridge address after a reboot.
ExecStart=$BIN \\
  --web.listen-address=127.0.0.1:9100 \\
  --web.listen-address=$GATEWAY:9100 \\
  --collector.filesystem.mount-points-exclude=^/(dev|proc|run|sys|snap|var/lib/docker/.+)(\$|/)
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable "$SERVICE" >/dev/null 2>&1
systemctl --user restart "$SERVICE"

# Keeps the service running when you are logged out and across reboots.
loginctl enable-linger "$USER" 2>/dev/null || \
  echo "Note: could not enable linger, so Node Exporter runs only while you are logged in."

for _ in 1 2 3 4 5 6 7 8 9 10; do
  if curl -fsS -o /dev/null http://127.0.0.1:9100/metrics; then
    echo "Node Exporter $VERSION is running on 127.0.0.1:9100 and $GATEWAY:9100."
    exit 0
  fi
  sleep 0.5
done
echo "Node Exporter did not start. See: journalctl --user -u $SERVICE" >&2
exit 1
