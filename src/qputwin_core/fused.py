"""Gate fusion and the multithreaded fused executor.

Usage: from qputwin_core.fused import set_engine_threads

Koustav Kumar Mondal <erkoustavkumar@gmail.com>
"""

from __future__ import annotations

import itertools
import math
import os
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import numpy as np

__all__ = ["Block", "FusedState", "engine_threads", "in_place_mode", "plan_blocks", "set_engine_threads",
           "thread_limit"]

# ------------------------------------------------------------------ threads
_threads: int | None = None
_pool: ThreadPoolExecutor | None = None
_MIN_CHUNK = 1 << 15                         # elements per task below which threading does not pay
_CHUNK_BITS = 15                             # gather / phase chunk: 2^15 amplitudes stay in cache
_CHUNK_BITS_MANY = 17                        # with many threads, larger chunks amortise per-chunk overhead
_MANY_THREADS = 8                            # (measured: +18–24 % on deep 24-qubit circuits at 16 threads)
# A block that grows the state writes into memory touched for the first time; page faults from many
# threads contend in the kernel, so growth uses at most this many threads (measured on 16 vCPUs: a
# 24-qubit GHZ chain takes 0.131 s with 16 growth threads and 0.086 s with 4).
_GROW_THREADS = int(os.environ.get("QPUTWIN_ENGINE_GROW_THREADS", "4")) or None
_LOW_BITS = 10                               # contiguous low axes kept below a gathered block
_MIN_RUN = 1 << 4                            # shortest contiguous run below a block worth a batched U @ X
_TABLE_LIMIT = 1 << 20                       # precomputed phase rows of a diagonal block, in entries
_CHOP = 1e-15                                # |entry| below this in a fused dense block is rounding noise
# Default executor mode for states created outside a simulator run (which sets it per job from its
# memory plan through :func:`in_place_mode`): two buffers unless QPUTWIN_ENGINE_IN_PLACE forces one.
_IN_PLACE = os.environ.get("QPUTWIN_ENGINE_IN_PLACE", "auto").strip().lower() in ("1", "true", "yes", "always")
_MODE: ContextVar[bool | None] = ContextVar("qputwin_engine_in_place", default=None)
_LIMIT: ContextVar[int | None] = ContextVar("qputwin_engine_thread_limit", default=None)
_MAX_DEFAULT_THREADS = 64                    # upper bound when the thread count comes from the CPU quota


def _default_threads() -> int:
    env = os.environ.get("QPUTWIN_ENGINE_THREADS")
    if env:
        return max(1, int(env))
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:                   # pragma: no cover - non-Linux
        n = os.cpu_count() or 1
    try:                                     # cgroup v2 CPU quota (Docker --cpus)
        with open("/sys/fs/cgroup/cpu.max") as f:
            quota, period = f.read().split()
        if quota != "max":
            n = min(n, max(1, math.floor(int(quota) / int(period))))
    except (OSError, ValueError):
        pass
    return max(1, min(n, _MAX_DEFAULT_THREADS))


def engine_threads() -> int:
    global _threads
    if _threads is None:
        _threads = _default_threads()
    limit = _LIMIT.get()
    return _threads if limit is None else max(1, min(_threads, limit))


@contextmanager
def thread_limit(n: int | None) -> Iterator[None]:
    """Cap the engine's threads in this context (e.g. ``1`` inside a task that is itself one of many)."""
    token = _LIMIT.set(n)
    try:
        yield
    finally:
        _LIMIT.reset(token)


def set_engine_threads(n: int) -> None:
    """Set the engine's thread count (1 disables threading)."""
    global _threads, _pool
    _threads = max(1, int(n))
    if _pool is not None:
        _pool.shutdown(wait=False)
        _pool = None


@contextmanager
def in_place_mode(flag: bool | None) -> Iterator[None]:
    """Run the fused executor in single-buffer mode (``True``), two-buffer mode (``False``) or the default."""
    token = _MODE.set(flag)
    try:
        yield
    finally:
        _MODE.reset(token)


def _chunk_bits() -> int:
    return _CHUNK_BITS_MANY if engine_threads() >= _MANY_THREADS else _CHUNK_BITS


def _executor() -> ThreadPoolExecutor:
    global _pool
    if _pool is None:
        _pool = ThreadPoolExecutor(max_workers=max(engine_threads(), _threads or 1),   # never the limited count
                                   thread_name_prefix="qputwin-engine")
    return _pool


def _parallel(tasks: list[Callable[[], None]]) -> None:
    if len(tasks) == 1 or engine_threads() == 1:
        for t in tasks:
            t()
        return
    for f in [_executor().submit(t) for t in tasks]:
        f.result()


def _ranges(n: int, work: int, max_parts: int | None = None) -> list[tuple[int, int]]:
    """Split ``range(n)`` into contiguous parts, one per thread when ``work`` elements justify it."""
    t = engine_threads() if max_parts is None else max(1, min(engine_threads(), max_parts))
    parts = 1 if t == 1 else max(1, min(t, n, work // _MIN_CHUNK))
    step = -(-n // parts)
    return [(s, min(n, s + step)) for s in range(0, n, step)]


def _split(n: int, work: int, fn: Callable[[int, int], None], max_parts: int | None = None) -> None:
    _parallel([lambda s=s, e=e: fn(s, e) for s, e in _ranges(n, work, max_parts)])


# ------------------------------------------------------------------ planner
@dataclass(slots=True)
class Block:
    axes: list[int]                  # index bit order: axes[0] is the most significant bit
    matrix: np.ndarray               # dense 2^k × 2^k matrix, or the length-2^k diagonal when ``diagonal``
    diagonal: bool


def _diag_embed(d: np.ndarray, pos: Sequence[int], k: int) -> np.ndarray:
    """Diagonal ``d`` (on ``len(pos)`` axes) lifted to the diagonal of a ``k``-axis operator."""
    g = len(pos)
    if g == k and list(pos) == list(range(k)):
        return d
    order = sorted(range(g), key=lambda i: pos[i])
    t = d.reshape((2,) * g).transpose(order) if g else d.reshape(())
    shape = [1] * k
    for i in order:
        shape[pos[i]] = 2
    return np.broadcast_to(t.reshape(shape), (2,) * k).reshape(-1)


def _bits_zero(k: int, positions: Sequence[int]) -> np.ndarray:
    """Mask of the ``2^k`` indices whose bits at ``positions`` (0 = most significant) are all 0."""
    idx = np.arange(1 << k)
    keep = np.ones(1 << k, dtype=bool)
    for p in positions:
        keep &= ((idx >> (k - 1 - p)) & 1) == 0
    return keep


def _kron_eye(m: np.ndarray, d: int) -> np.ndarray:
    """``m ⊗ I_d`` without ``np.kron``'s generic overhead."""
    k = m.shape[0]
    return (m[:, None, :, None] * np.eye(d, dtype=m.dtype)[None, :, None, :]).reshape(k * d, k * d)


def _left_apply(g: np.ndarray, pos: Sequence[int], m: np.ndarray) -> np.ndarray:
    """``G · M`` with gate ``g`` acting on row-index bits ``pos`` (0 = most significant) of ``M``.

    A direct contraction, O(2^|pos| · size(M)) — instead of embedding ``g`` into a full
    ``2^k × 2^k`` matrix and multiplying, O(8^k).
    """
    k, gn = m.shape[0].bit_length() - 1, len(pos)
    t = m.reshape((2,) * k + (m.shape[1],))
    r = np.tensordot(g.reshape((2,) * (2 * gn)), t, axes=(list(range(gn, 2 * gn)), list(pos)))
    return np.moveaxis(r, list(range(gn)), list(pos)).reshape(m.shape)


class _Open:
    __slots__ = ("axes", "d", "m", "seq")

    def __init__(self, axes: list[int], m: np.ndarray | None, d: np.ndarray | None, seq: int) -> None:
        self.axes, self.m, self.d, self.seq = axes, m, d, seq       # exactly one of m (dense) / d (diagonal)

    @property
    def diagonal(self) -> bool:
        return self.d is not None

    def dense(self) -> np.ndarray:
        if self.d is not None:
            self.m, self.d = np.diag(self.d), None
        assert self.m is not None
        return self.m

    def merge(self, other: _Open) -> None:
        """Tensor in a disjoint open block (they commute)."""
        if self.d is not None and other.d is not None:
            self.d = np.kron(self.d, other.d)
        else:
            self.m = np.kron(self.dense(), other.dense())     # rare: only when a gate bridges blocks
        self.axes = self.axes + other.axes

    def absorb(self, axes: Sequence[int], g: np.ndarray, gd: np.ndarray | None) -> None:
        """Multiply gate ``g`` (dense, or its diagonal ``gd``) onto the block from the left."""
        new = [a for a in axes if a not in self.axes]
        if new:
            if self.d is not None:
                self.d = np.repeat(self.d, 1 << len(new))
            else:
                self.m = _kron_eye(self.dense(), 1 << len(new))
            self.axes = self.axes + new
        pos = [self.axes.index(a) for a in axes]
        if gd is not None:
            phases = _diag_embed(gd, pos, len(self.axes))
            if self.d is not None:
                self.d = self.d * phases
            else:
                self.m = phases[:, None] * self.m
        else:
            self.m = _left_apply(g, pos, self.dense())

    def block(self) -> Block:
        if self.d is not None:
            return Block(self.axes, self.d, True)
        m = self.dense()
        small = np.abs(m) < _CHOP           # rounding noise (cos π/2 ≈ 6e-17) would decay into denormals
        if small.any():
            # a new array: a block of one gate is that gate's own (possibly shared, read-only) matrix
            m = np.where(small, 0, m)
        return Block(self.axes, m, False)


def _merge_diagonals(blocks: list[Block], dmax: int) -> list[Block]:
    """Merge runs of consecutive diagonal blocks (diagonals always commute) up to ``dmax`` axes."""
    out: list[Block] = []
    for b in blocks:
        prev = out[-1] if out else None
        if b.diagonal and prev is not None and prev.diagonal and len(set(prev.axes) | set(b.axes)) <= dmax:
            axes = prev.axes + [a for a in b.axes if a not in prev.axes]
            d = np.repeat(prev.matrix, 1 << (len(axes) - len(prev.axes)))
            out[-1] = Block(axes, d * _diag_embed(b.matrix, [axes.index(a) for a in b.axes], len(axes)), True)
        else:
            out.append(b)
    return out


def plan_blocks(ops: Sequence[tuple[Sequence[int], np.ndarray]], max_axes: int,
                max_diagonal_axes: int | None = None) -> list[Block]:
    """Fuse ``(axes, matrix)`` gates (matrix index MSB = ``axes[0]``) into blocks.

    A gate's matrix may be given as its diagonal (a 1-D array). Dense blocks span at
    most ``max_axes`` axes and diagonal blocks at most ``max_diagonal_axes`` (default
    ``max_axes``); a single gate wider than the limit forms its own block.
    """
    dmax = max(max_axes, max_diagonal_axes or max_axes)
    out: list[Block] = []
    open_: list[_Open] = []
    seq = 0
    for axes, m in ops:
        axes = list(axes)
        m = np.asarray(m, dtype=np.complex128)
        if m.ndim == 1:
            gd: np.ndarray | None = m
        else:
            gd = np.diag(m).copy() if not m[~np.eye(len(m), dtype=bool)].any() else None
        is_diag = gd is not None

        def limit(blocks: list[_Open], is_diag: bool = is_diag) -> int:
            return dmax if is_diag and all(b.diagonal for b in blocks) else max_axes

        touching = [b for b in open_ if not set(b.axes).isdisjoint(axes)]
        keep = touching
        if len(set(axes).union(*(b.axes for b in touching))) > limit(touching):
            # emit blocks until the gate fits with the rest (dense blocks first for a diagonal
            # gate, then the widest); the rest are disjoint from them, so the order stays valid
            keep = sorted(touching, key=lambda b: (is_diag and not b.diagonal, len(b.axes), -b.seq))
            while keep and len(set(axes).union(*(b.axes for b in keep))) > limit(keep):
                b = keep.pop()
                out.append(b.block())
                open_.remove(b)
        if keep:
            keep.sort(key=lambda b: b.seq)
            base = keep[0]
            for other in keep[1:]:
                base.merge(other)
                open_.remove(other)
            base.absorb(axes, m, gd)
        else:
            open_.append(_Open(axes, None if is_diag else m, gd, seq))
            seq += 1
    out.extend(b.block() for b in sorted(open_, key=lambda b: b.seq))
    return _merge_diagonals(out, dmax)


# ------------------------------------------------------------------ executor
def _reorder_cols(m: np.ndarray, axes: Sequence[int], order: Sequence[int]) -> np.ndarray:
    """``m`` with its column index bits (MSB = ``axes[0]``) relabelled to follow ``order``."""
    if list(axes) == list(order):
        return m
    k = len(axes)
    t = m.reshape((m.shape[0],) + (2,) * k)
    return t.transpose([0] + [1 + list(axes).index(a) for a in order]).reshape(m.shape[0], 1 << k)


def _reorder_both(m: np.ndarray, axes: Sequence[int], order: Sequence[int]) -> np.ndarray:
    """Square ``m`` with its row *and* column index bits relabelled to follow ``order``."""
    if list(axes) == list(order):
        return m
    k = len(axes)
    p = [list(axes).index(a) for a in order]
    return m.reshape((2,) * (2 * k)).transpose(p + [k + i for i in p]).reshape(1 << k, 1 << k)


class FusedState:
    """A ``2^N`` state stored as a tensor over its *active* axes, in a lazily permuted order.

    ``perm[i]`` is the logical axis stored at tensor position ``i`` (position 0 is the
    most significant). Logical axis ``a`` is bit ``a`` of the standard little-endian
    index. Axes not in ``perm`` are exactly ``|0⟩``.
    """

    def __init__(self, n_axes: int, dtype: np.dtype | type = np.complex128, *, in_place: bool | None = None) -> None:
        self.n = n_axes
        self.dtype = np.dtype(dtype)
        if in_place is None:
            in_place = _MODE.get()
        self.in_place = _IN_PLACE if in_place is None else bool(in_place)
        self._buf = [np.empty(1 << n_axes, dtype=self.dtype)]
        if not self.in_place:
            self._buf.append(np.empty(1 << n_axes, dtype=self.dtype))
        self._buf[0][0] = 1.0                     # |0…0⟩ with no active axes: the scalar 1
        self.perm: list[int] = []
        self._i = 0
        self.transposes = 0

    # ------------------------------------------------------------ helpers
    @property
    def cur(self) -> np.ndarray:
        return self._buf[self._i][: 1 << len(self.perm)]

    def _dst(self, m: int) -> np.ndarray:
        return self._buf[0 if self.in_place else 1 - self._i][: 1 << m]

    def _swap(self, perm: list[int]) -> None:
        if not self.in_place:
            self._i = 1 - self._i
        self.perm = perm

    @staticmethod
    def _tensor(a: np.ndarray, m: int) -> np.ndarray:
        return a.reshape((2,) * m) if m else a.reshape(())

    def _move(self, new: list[int]) -> None:
        """Transpose the stored tensor into axis order ``new``."""
        if new == self.perm:
            return
        m = len(self.perm)
        pos = {a: i for i, a in enumerate(self.perm)}
        src = self._tensor(self.cur, m).transpose([pos[a] for a in new])
        if self.in_place:                        # a full transpose cannot run in place: the result gets
            out = np.empty(1 << self.n, dtype=self.dtype)     # a new buffer and the old one is freed
            dst = self._tensor(out[: 1 << m], m)
        else:
            dst = self._tensor(self._dst(m), m)
        j = min(m, max(0, math.ceil(math.log2(engine_threads())) + 1)) if (1 << m) >= 2 * _MIN_CHUNK else 0
        if j == 0:
            np.copyto(dst, src)
        else:
            _parallel([lambda idx=idx: np.copyto(dst[idx], src[idx])
                       for idx in itertools.product((0, 1), repeat=j)])
        if self.in_place:
            self._buf[0] = out
        self._swap(new)
        self.transposes += 1

    # ------------------------------------------------------------- apply
    def apply(self, block: Block) -> None:
        if block.diagonal:
            self.apply_diagonal(block.axes, block.matrix)
        else:
            self.apply_dense(block.axes, block.matrix)

    def apply_dense(self, axes: Sequence[int], m: np.ndarray) -> None:
        axes = list(axes)
        active = set(self.perm)
        new = [i for i, a in enumerate(axes) if a not in active]
        if new:                                   # new axes are |0⟩: only those columns contribute
            m = np.asarray(m)[:, _bits_zero(len(axes), new)]
        self._apply([a for a in axes if a in active], axes, np.asarray(m))

    def _apply(self, in_axes: list[int], out_axes: list[int], m: np.ndarray) -> None:
        """``Y = M X`` with ``M`` of shape ``2^|out| × 2^|in|`` (``out_axes`` ⊇ ``in_axes``)."""
        p_all = len(self.perm)
        ki, ko = len(in_axes), len(out_axes)
        if self.in_place and ki == ko:
            self._apply_in_place(in_axes, m)
            return
        # the state grows (or the two-buffer executor is selected): read the current state, write a new one
        cur = self.cur.copy() if self.in_place else self.cur     # in place: at most half the buffer
        dst = self._dst(p_all - ki + ko)
        if ki == 0:                               # product with fresh |0…0⟩ axes: an outer product
            col = np.ascontiguousarray(m[:, 0], dtype=self.dtype)
            x, y = cur, dst.reshape(-1, 1 << ko)
            _split(len(x), len(dst), lambda s, e: np.multiply(x[s:e, None], col[None, :], out=y[s:e]),
                   _GROW_THREADS)
            self._swap(self.perm + out_axes)
            return
        pos = {a: i for i, a in enumerate(self.perm)}
        ps = sorted(pos[a] for a in in_axes)
        p0 = ps[0]
        b, r = 1 << p0, 1 << (p_all - p0 - ki)
        if ps[-1] - p0 + 1 == ki and r >= _MIN_RUN and (b >= engine_threads() or len(dst) < 2 * _MIN_CHUNK):
            # adjacent, with a contiguous run below and rows to share: one batched U @ X, no data movement
            mm = np.ascontiguousarray(_reorder_cols(m, in_axes, self.perm[p0:p0 + ki]), dtype=self.dtype)
            x, y = cur.reshape(b, 1 << ki, r), dst.reshape(b, 1 << ko, r)
            _split(b, len(dst), lambda s, e: np.matmul(mm, x[s:e], out=y[s:e]), _GROW_THREADS if ko > ki else None)
            self._swap(self.perm[:p0] + out_axes + self.perm[p0 + ki:])
            return
        # gather the block's axes next to the lowest contiguous axes, chunk by chunk, and multiply
        ins = sorted(in_axes, key=pos.__getitem__)
        mm = np.ascontiguousarray(_reorder_cols(m, in_axes, ins), dtype=self.dtype)
        others = [a for a in self.perm if a not in set(in_axes)]
        r = min(len(others), _LOW_BITS)
        hi, lo = others[:len(others) - r], others[len(others) - r:]
        src = self._tensor(cur, p_all).transpose([pos[a] for a in hi + ins + lo])
        j = min(len(hi), max(0, p_all - _chunk_bits()))
        y = dst.reshape(1 << j, -1)
        idxs = list(itertools.product((0, 1), repeat=j))

        def work(s: int, e: int) -> None:
            tmp = np.empty(src.shape[j:], dtype=self.dtype)
            tv = tmp.reshape(-1, 1 << ki, 1 << r)
            for c in range(s, e):
                np.copyto(tmp, src[idxs[c]])
                np.matmul(mm, tv, out=y[c].reshape(-1, 1 << ko, 1 << r))

        _split(len(idxs), len(dst), work, _GROW_THREADS if ko > ki else None)
        self._swap(hi + out_axes + lo)
        self.transposes += 1

    def _apply_in_place(self, axes: list[int], m: np.ndarray) -> None:
        """``X ← M X`` for a square block on active ``axes``, chunk by chunk; ``perm`` is unchanged."""
        p_all, k = len(self.perm), len(axes)
        pos = {a: i for i, a in enumerate(self.perm)}
        ins = sorted(axes, key=pos.__getitem__)  # the block's axes in stored order
        mm = np.ascontiguousarray(_reorder_both(np.asarray(m), axes, ins), dtype=self.dtype)
        kk = 1 << k
        ps = [pos[a] for a in ins]
        p0 = ps[0]
        x = self.cur
        if ps[-1] - p0 + 1 == k:
            # adjacent axes: X viewed as (b, 2^k, r); each unit is a run of rows, or one row and a run of columns
            b, r = 1 << p0, 1 << (p_all - p0 - k)
            xv = x.reshape(b, kk, r)
            chunk = 1 << _chunk_bits()
            if kk * r <= chunk:
                rows = max(1, chunk // (kk * r))
                units = [(s, min(b, s + rows), 0, r) for s in range(0, b, rows)]
            else:
                w = max(1, chunk // kk)
                units = [(i, i + 1, c, c + w) for i in range(b) for c in range(0, r, w)]

            def work_adj(s: int, e: int) -> None:
                tmp = None
                for rs, re_, cs, ce in units[s:e]:
                    view = xv[rs:re_, :, cs:ce]
                    if tmp is None or tmp.shape != view.shape:
                        tmp = np.empty(view.shape, dtype=self.dtype)
                    np.matmul(mm, view, out=tmp)
                    np.copyto(view, tmp)

            _split(len(units), len(x), work_adj)
            return
        # scattered axes: gather each cache-sized chunk next to the lowest contiguous axes, multiply, scatter back
        others = [a for a in self.perm if a not in set(axes)]
        r = min(len(others), _LOW_BITS)
        hi, lo = others[:len(others) - r], others[len(others) - r:]
        src = self._tensor(x, p_all).transpose([pos[a] for a in hi + ins + lo])
        j = min(len(hi), max(0, p_all - _chunk_bits()))
        idxs = list(itertools.product((0, 1), repeat=j))

        def work(s: int, e: int) -> None:
            tmp = np.empty(src.shape[j:], dtype=self.dtype)
            out = np.empty_like(tmp)
            tv, ov = tmp.reshape(-1, kk, 1 << r), out.reshape(-1, kk, 1 << r)
            for c in range(s, e):
                view = src[idxs[c]]
                np.copyto(tmp, view)
                np.matmul(mm, tv, out=ov)
                np.copyto(view, out)

        _split(len(idxs), len(x), work)

    def apply_diagonal(self, axes: Sequence[int], d: np.ndarray) -> None:
        """Multiply by the diagonal operator ``diag(d)`` (index MSB = ``axes[0]``)."""
        axes, d = list(axes), np.asarray(d)
        active = set(self.perm)
        new = [i for i, a in enumerate(axes) if a not in active]
        if new:                                   # phases on |0⟩ axes: keep the entries where they are 0
            d = d[_bits_zero(len(axes), new)]
            axes = [a for a in axes if a in active]
        d = np.asarray(d, dtype=self.dtype)
        if not axes:
            if d[0] != 1:
                x = self.cur
                _split(len(x), len(x), lambda s, e: np.multiply(x[s:e], d[0], out=x[s:e]))
            return
        k, p_all = len(axes), len(self.perm)
        pos = {a: i for i, a in enumerate(self.perm)}
        weight = {a: k - 1 - i for i, a in enumerate(axes)}
        cb = min(p_all, _chunk_bits())
        nh = p_all - cb
        low = np.arange(1 << cb)
        lo_map = np.zeros(1 << cb, dtype=np.intp)
        hi = [a for a in axes if pos[a] < nh]
        for a in axes:
            if pos[a] >= nh:
                lo_map |= ((low >> (p_all - 1 - pos[a])) & 1) << weight[a]
        x = self.cur.reshape(-1, 1 << cb)
        if not hi:
            ph = d[lo_map]
            _split(len(x), x.size, lambda s, e: np.multiply(x[s:e], ph, out=x[s:e]))
            return
        rows = np.arange(1 << nh)
        hi_map = np.zeros(1 << nh, dtype=np.intp)
        for a in hi:
            hi_map |= ((rows >> (nh - 1 - pos[a])) & 1) << weight[a]
        values, which = np.unique(hi_map, return_inverse=True)
        if len(values) << cb <= _TABLE_LIMIT:                   # few distinct rows: precompute them
            tables = d[values[:, None] | lo_map[None, :]]

            def work(s: int, e: int) -> None:
                for i in range(s, e):
                    np.multiply(x[i], tables[which[i]], out=x[i])
        else:

            def work(s: int, e: int) -> None:
                for i in range(s, e):
                    np.multiply(x[i], d[hi_map[i] | lo_map], out=x[i])

        _split(len(x), x.size, work)

    # ------------------------------------------------------------- read out
    def standard(self) -> np.ndarray:
        """The full ``2^N`` vector in standard little-endian order (bit ``a`` ↔ logical axis ``a``)."""
        missing = [a for a in range(self.n) if a not in set(self.perm)]
        if missing:
            e0 = np.zeros((1 << len(missing), 1), dtype=self.dtype)
            e0[0, 0] = 1.0
            self._apply([], missing, e0)
        self._move(list(range(self.n - 1, -1, -1)))
        return self.cur

    def marginal(self, axes: Sequence[int], *, consume: bool = False) -> tuple[np.ndarray, list[int]]:
        """Probabilities of the measured ``axes``, in stored order, without reordering the state.

        Returns ``(p, order)``: bit ``i`` (LSB = 0) of an index into ``p`` is axis
        ``order[len(order) - 1 - i]``. Measured axes that are inactive are always 0
        and are left out of ``order``. ``consume=True`` writes the probabilities into
        the state's own memory (destroying it) instead of a fresh array, which avoids
        faulting in a new state-sized allocation.
        """
        x = self.cur
        bits = _chunk_bits()
        if consume and len(x) > 1 << bits:
            # chunk c reads amplitudes [s, e) and writes floats [s, e). For complex64 those are the
            # chunk's own bytes, so all chunks run at once. For complex128 they are the bytes of
            # amplitudes [s/2, e/2), i.e. half of chunk c // 2: chunks therefore run in waves
            # {0}, {1}, {2, 3}, {4…7}, …, each after the wave holding the chunks it overwrites, and
            # a chunk's own memory is only written by the next wave.
            p = x.view(np.float64)[: len(x)]
            step = 1 << bits
            nchunks = len(x) // step

            def chunks(c0: int, c1: int) -> None:
                pc, qc = np.empty(step, dtype=np.float64), np.empty(step, dtype=np.float64)
                for c in range(c0, c1):
                    xc = x[c * step:(c + 1) * step]
                    np.square(xc.real, out=pc, dtype=np.float64)
                    np.square(xc.imag, out=qc, dtype=np.float64)
                    np.add(pc, qc, out=p[c * step:(c + 1) * step])

            if x.dtype == np.complex64:
                _split(nchunks, len(x), chunks)
            else:
                chunks(0, 1)
                lo = 1
                while lo < nchunks:
                    hi_ = min(nchunks, 2 * lo)
                    _split(hi_ - lo, (hi_ - lo) * step, lambda a, b, lo=lo: chunks(lo + a, lo + b))
                    lo = hi_
        else:
            p = np.empty(len(x), dtype=np.float64)

            def work(s: int, e: int) -> None:
                np.square(x.real[s:e], out=p[s:e], dtype=np.float64)
                p[s:e] += np.square(x.imag[s:e], dtype=np.float64)

            _split(len(x), len(x), work)
        keep = set(axes)
        drop = tuple(i for i, a in enumerate(self.perm) if a not in keep)
        if drop:
            p = p.reshape((2,) * len(self.perm)).sum(axis=drop).ravel()
        return p, [a for a in self.perm if a in keep]

    def release(self) -> None:
        """Drop the spare buffer once no more blocks will be applied (the in-place executor has none)."""
        if not self.in_place:
            self._buf[1 - self._i] = np.empty(0, dtype=self.dtype)
