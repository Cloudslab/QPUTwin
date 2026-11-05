"""Monotonic, nanosecond-resolution timing primitives used for all telemetry.

All QPUTwin latency measurements use :func:`time.perf_counter_ns`. Durations are
measured *within one process*; no timestamps are ever compared across containers,
which removes inter-container clock skew from the measurement error budget.

Koustav Kumar Mondal <erkoustavkumar@gmail.com>
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager

NS_PER_MS = 1_000_000


def now_ns() -> int:
    return time.perf_counter_ns()


def ns_to_ms(ns: int) -> float:
    return ns / NS_PER_MS


class SpanTimer:
    """Accumulates named, possibly repeated, spans in nanoseconds."""

    __slots__ = ("spans_ns",)

    def __init__(self) -> None:
        self.spans_ns: dict[str, int] = {}

    @contextmanager
    def span(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter_ns()
        try:
            yield
        finally:
            self.add(name, time.perf_counter_ns() - t0)

    def add(self, name: str, ns: int) -> None:
        self.spans_ns[name] = self.spans_ns.get(name, 0) + ns

    def ms(self, name: str) -> float:
        return ns_to_ms(self.spans_ns.get(name, 0))

    def as_ms(self) -> dict[str, float]:
        return {k: ns_to_ms(v) for k, v in self.spans_ns.items()}
