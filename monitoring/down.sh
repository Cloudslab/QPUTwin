#!/usr/bin/env bash
# Stops the monitoring containers. Node Exporter keeps running on the host; use
# node-exporter/install.sh --uninstall to remove it.
#
#   ./down.sh             keep stored metrics, dashboards state and Portainer data
#   ./down.sh --volumes   delete them too
# Koustav Kumar Mondal <koustav@koustavmondal.com>
set -euo pipefail
cd "$(dirname "$0")"
docker compose down "$@"
