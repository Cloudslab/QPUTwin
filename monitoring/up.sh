#!/usr/bin/env bash
# Starts QPUTwin monitoring. Safe to run repeatedly.
# Koustav Kumar Mondal <erkoustavkumar@gmail.com>
set -euo pipefail
cd "$(dirname "$0")"

NETWORK="${QPUTWIN_NETWORK:-qputwin_qputwin-net}"
if ! docker network inspect "$NETWORK" >/dev/null 2>&1; then
  echo "QPUTwin is not running (network $NETWORK not found)." >&2
  echo "Start it first, from the project root: docker compose up -d --wait" >&2
  exit 1
fi

random_password() { LC_ALL=C tr -dc 'A-Za-z0-9' </dev/urandom | head -c 24 || true; }

if [[ ! -f .env ]]; then
  umask 077
  sed "s|^GRAFANA_ADMIN_PASSWORD=.*|GRAFANA_ADMIN_PASSWORD=$(random_password)|; \
       s|^QPUTWIN_UID=.*|QPUTWIN_UID=$(id -u)|; s|^QPUTWIN_GID=.*|QPUTWIN_GID=$(id -g)|" \
    .env.example > .env
fi
if [[ ! -f secrets/portainer_admin_password ]]; then
  umask 077
  mkdir -p secrets
  random_password > secrets/portainer_admin_password
fi

curl -fsS -o /dev/null -m 2 http://127.0.0.1:9100/metrics || ./node-exporter/install.sh

docker compose up -d --wait

GRAFANA_PASSWORD="$(sed -n 's/^GRAFANA_ADMIN_PASSWORD=//p' .env)"
PORTAINER_PASSWORD="$(cat secrets/portainer_admin_password)"
cat <<EOF

QPUTwin monitoring is up.
  Grafana       http://localhost:3000    admin / $GRAFANA_PASSWORD
  Portainer     https://localhost:9443   admin / $PORTAINER_PASSWORD
  Prometheus    http://localhost:9090
  Alertmanager  http://localhost:9093
  cAdvisor      http://localhost:8081
EOF
