"""Exception classes used across QPUTwin.

Usage: from qputwin_core.errors import QPUTwinError

Koustav Kumar Mondal <koustav@koustavmondal.com>
"""

from __future__ import annotations

from typing import Any


class QPUTwinError(Exception):
    """Base class for every QPUTwin domain error."""

    http_status: int = 500

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = details or {}

    @property
    def type(self) -> str:
        return type(self).__name__

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "message": self.message, "details": self.details}


# --------------------------------------------------------------------------- QASM
class QASMError(QPUTwinError):
    http_status = 400


class QASMSyntaxError(QASMError):
    """The source text is not well-formed OpenQASM."""

    def __init__(self, message: str, *, line: int | None = None, col: int | None = None,
                 details: dict[str, Any] | None = None) -> None:
        d = dict(details or {})
        if line is not None:
            d["line"] = line
        if col is not None:
            d["col"] = col
        loc = f" (line {line}, col {col})" if line is not None else ""
        super().__init__(f"{message}{loc}", details=d)
        self.line = line
        self.col = col


class QASMSemanticError(QASMError):
    """Well-formed but meaningless: undeclared names, bad arity, out-of-range indices..."""

    def __init__(self, message: str, *, line: int | None = None,
                 details: dict[str, Any] | None = None) -> None:
        d = dict(details or {})
        if line is not None:
            d["line"] = line
        super().__init__(message if line is None else f"{message} (line {line})", details=d)
        self.line = line


class UnsupportedFeatureError(QASMSemanticError):
    """Valid OpenQASM 3 that lies outside the supported subset (see https://koustav.isroot.in/qputwin-docs/guide/openqasm/)."""


class InvalidRequestError(QPUTwinError):
    """The request is well-formed JSON but its options are inconsistent or out of range."""

    http_status = 400


# ---------------------------------------------------------------------- compilation
class CompilationError(QPUTwinError):
    http_status = 422


class TopologicalCompilationError(CompilationError):
    """A multi-qubit operation addresses a pair of physical qubits with no coupler."""


class NativeGateError(CompilationError):
    """``strict`` compilation mode received a gate outside the device's native set."""


# ------------------------------------------------------------------------ resources
class ResourceLimitError(QPUTwinError):
    """Request exceeds the device size, the configured cap, or the available memory."""

    http_status = 422


class CalibrationProvenanceError(QPUTwinError):
    """A noisy run would rely on calibration values not tagged as published (strict provenance mode)."""

    http_status = 422


# ---------------------------------------------------------------------- transport
class UnknownBackendError(QPUTwinError):
    http_status = 404


class BackendUnavailableError(QPUTwinError):
    http_status = 503


class BackendTimeoutError(QPUTwinError):
    http_status = 504


class BackendProtocolError(QPUTwinError):
    """The worker answered with something that is not a valid QPUTwin payload."""

    http_status = 502


class AuthenticationError(QPUTwinError):
    """Missing or invalid API key (only when the orchestrator is configured with API keys)."""

    http_status = 401


# ------------------------------------------------------------------ async jobs
class JobNotFoundError(QPUTwinError):
    """No asynchronous job with this id exists (never submitted, or expired from retention)."""

    http_status = 404


class JobStateError(QPUTwinError):
    """The requested transition is illegal in the job's current state (e.g. cancelling a running job)."""

    http_status = 409


class QueueFullError(QPUTwinError):
    """The orchestrator's asynchronous queue is at capacity; retry later."""

    http_status = 503


_REGISTRY: dict[str, type[QPUTwinError]] = {}


def _collect(cls: type[QPUTwinError]) -> None:
    _REGISTRY[cls.__name__] = cls
    for sub in cls.__subclasses__():
        _collect(sub)


_collect(QPUTwinError)


def error_from_dict(payload: dict[str, Any], *, status: int | None = None) -> QPUTwinError:
    """Rebuild a :class:`QPUTwinError` from its ``to_dict`` form (unknown types -> base)."""
    cls = _REGISTRY.get(str(payload.get("type")), QPUTwinError)
    err = cls.__new__(cls)
    QPUTwinError.__init__(err, str(payload.get("message", "")), details=payload.get("details") or {})
    if status is not None and cls is QPUTwinError:
        err.http_status = status
    return err
