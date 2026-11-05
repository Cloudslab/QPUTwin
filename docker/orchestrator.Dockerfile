# syntax=docker/dockerfile:1.7
# =============================================================================
# QPUTwin — central orchestrator image (FastAPI + benchmarking middleware).
# Installs only the `orchestrator` extra: no numpy/scipy (the QASM front-end used
# for validation is pure Python). Koustav Kumar Mondal <erkoustavkumar@gmail.com>
# =============================================================================
ARG PYTHON_IMAGE=python:3.13.16-slim-trixie
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.12.23

FROM ${UV_IMAGE} AS uv

# ----------------------------------------------------------------- builder ---
FROM ${PYTHON_IMAGE} AS builder
COPY --from=uv /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /app

# Layer 1 — third-party dependencies only: copy the manifest + lockfile and install
# everything *except* the project. Cached until pyproject.toml / uv.lock change.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev --extra orchestrator

# Layer 2 — the QPUTwin project itself (non-editable, so the runtime needs only .venv).
COPY README.md LICENSE NOTICE ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra orchestrator --no-editable

# ----------------------------------------------------------------- runtime ---
FROM ${PYTHON_IMAGE} AS runtime
LABEL org.opencontainers.image.title="QPUTwin orchestrator" \
      org.opencontainers.image.description="Routing API and benchmark telemetry for the QPUTwin QPU emulators" \
      org.opencontainers.image.authors="Koustav Kumar Mondal <erkoustavkumar@gmail.com>" \
      org.opencontainers.image.vendor="Koustav Kumar Mondal" \
      org.opencontainers.image.version="0.1.0" \
      org.opencontainers.image.licenses="Apache-2.0"
RUN useradd --system --uid 10001 --no-create-home qputwin \
 && mkdir -p /app/data && chown qputwin:qputwin /app/data
COPY --from=builder --chown=qputwin:qputwin /app/.venv /app/.venv
COPY --chown=qputwin:qputwin config /app/config
ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    QPUTWIN_BACKENDS_CONFIG=/app/config/backends.toml \
    QPUTWIN_PORT=8000
WORKDIR /app
USER qputwin
EXPOSE 8000
HEALTHCHECK --interval=5s --timeout=3s --start-period=10s --retries=12 \
  CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status == 200 else 1)"]
CMD ["python", "-m", "qputwin_orchestrator"]
