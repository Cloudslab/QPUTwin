"""Benchmark circuit families (GHZ, QFT, QV and others) written as OpenQASM 2.0.

Usage: from circuits import FAMILIES

Koustav Kumar Mondal <erkoustavkumar@gmail.com>
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable

H2 = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'


def _wrap(n: int, body: list[str], ncl: int | None = None, measure: list[int] | None = None) -> str:
    ncl = n if ncl is None else ncl
    qs = list(range(n)) if measure is None else measure
    meas = [f"measure q[{q}] -> c[{k}];" for k, q in enumerate(qs)]
    return H2 + f"qreg q[{n}];\ncreg c[{ncl}];\n" + "\n".join(body + meas) + "\n"


def _a(x: float) -> str:
    return f"{x:.12g}"


def ghz(n: int, seed: int = 0) -> str:
    return _wrap(n, ["h q[0];"] + [f"cx q[{k}],q[{k + 1}];" for k in range(n - 1)])


def _qft_ops(n: int) -> list[str]:
    ops = []
    for j in range(n):
        ops.append(f"h q[{j}];")
        ops += [f"cu1({_a(math.pi / 2 ** (k - j))}) q[{k}],q[{j}];" for k in range(j + 1, n)]
    return ops + [f"swap q[{k}],q[{n - 1 - k}];" for k in range(n // 2)]


def _invert(op: str) -> str:
    return op.replace("cu1(", "cu1(-").replace("cu1(--", "cu1(") if op.startswith("cu1") else op


def qft(n: int, seed: int = 0) -> str:
    """Draper adder: QFT|x⟩, add a constant with one phase per qubit, QFT† → |x + a⟩ (deterministic).

    The transform takes q[0] as the most significant bit, so the output is the bit-reversed sum.
    """
    rng = random.Random(seed)
    x, a = rng.randrange(2 ** n), rng.randrange(1, 2 ** n)
    body = [f"x q[{k}];" for k in range(n) if x >> k & 1] + _qft_ops(n)
    body += [f"u1({_a(2 * math.pi * a * 2 ** (n - 1 - j) / 2 ** n)}) q[{j}];" for j in range(n)]
    body += [_invert(op) for op in reversed(_qft_ops(n))]
    return _wrap(n, body)


def _rand_u3(rng: random.Random, q: int) -> str:
    return f"u3({_a(rng.uniform(0, math.pi))},{_a(rng.uniform(-math.pi, math.pi))}," \
           f"{_a(rng.uniform(-math.pi, math.pi))}) q[{q}];"


def qv(n: int, seed: int = 0) -> str:
    rng = random.Random(seed)
    body = []
    for _ in range(n):
        perm = list(range(n))
        rng.shuffle(perm)
        for a, b in zip(perm[0::2], perm[1::2], strict=False):
            for _k in range(3):
                body += [_rand_u3(rng, a), _rand_u3(rng, b), f"cx q[{a}],q[{b}];"]
            body += [_rand_u3(rng, a), _rand_u3(rng, b)]
    return _wrap(n, body)


def _regular3(n: int, rng: random.Random) -> list[tuple[int, int]]:
    if n < 4 or n % 2:
        return [(k, (k + 1) % n) for k in range(n)] if n > 2 else [(0, 1)]
    for _ in range(1000):                        # configuration model with rejection
        stubs = [v for v in range(n) for _ in range(3)]
        rng.shuffle(stubs)
        edges = {tuple(sorted(p)) for p in zip(stubs[0::2], stubs[1::2], strict=False)}
        if len(edges) == 3 * n // 2 and all(a != b for a, b in edges):
            return sorted(edges)
    return [(k, (k + 1) % n) for k in range(n)]


def qaoa(n: int, seed: int = 0, p: int = 2) -> str:
    rng = random.Random(seed)
    edges = _regular3(n, rng)
    body = [f"h q[{k}];" for k in range(n)]
    for layer in range(p):
        gamma, beta = 0.4 + 0.2 * layer, 0.7 - 0.2 * layer
        body += [f"rzz({_a(2 * gamma)}) q[{a}],q[{b}];" for a, b in edges]
        body += [f"rx({_a(2 * beta)}) q[{k}];" for k in range(n)]
    return _wrap(n, body)


def hea(n: int, seed: int = 0, layers: int = 3) -> str:
    rng = random.Random(seed)
    body = []
    for _ in range(layers):
        for k in range(n):
            body += [f"ry({_a(rng.uniform(0, 2 * math.pi))}) q[{k}];",
                     f"rz({_a(rng.uniform(0, 2 * math.pi))}) q[{k}];"]
        body += [f"cx q[{k}],q[{k + 1}];" for k in range(n - 1)]
    body += [f"ry({_a(rng.uniform(0, 2 * math.pi))}) q[{k}];" for k in range(n)]
    return _wrap(n, body)


def ising(n: int, seed: int = 0, steps: int = 4, dt: float = 0.25) -> str:
    body = []
    for _ in range(steps):
        body += [f"rzz({_a(2 * dt)}) q[{k}],q[{k + 1}];" for k in range(n - 1)]
        body += [f"rx({_a(2 * dt * 1.2)}) q[{k}];" for k in range(n)]
    return _wrap(n, body)


def bv_secret(n: int, seed: int = 0) -> str:
    rng = random.Random(seed)
    s = [rng.randint(0, 1) for _ in range(n - 1)]
    if not any(s):
        s[0] = 1
    return "".join(map(str, reversed(s)))       # c[n-2] … c[0]


def bv(n: int, seed: int = 0) -> str:
    secret = bv_secret(n, seed)[::-1]           # secret[k] belongs to data qubit k
    anc = n - 1
    body = [f"x q[{anc}];", f"h q[{anc}];"] + [f"h q[{k}];" for k in range(n - 1)]
    body += [f"cx q[{k}],q[{anc}];" for k in range(n - 1) if secret[k] == "1"]
    body += [f"h q[{k}];" for k in range(n - 1)]
    return _wrap(n, body, ncl=n - 1, measure=list(range(n - 1)))


def adder(n: int, seed: int = 0) -> str:
    """Cuccaro adder on k = (n − 2)/2 bits: qubits cin, a0, b0, a1, b1, …, cout (n even ≥ 4)."""
    k = (n - 2) // 2
    rng = random.Random(seed)
    cin, cout = 0, n - 1
    a = [1 + 2 * j for j in range(k)]
    b = [2 + 2 * j for j in range(k)]
    body = [f"x q[{a[j]}];" for j in range(k) if rng.random() < 0.5]
    body += [f"x q[{b[j]}];" for j in range(k) if rng.random() < 0.5]

    def maj(c: int, y: int, x: int) -> list[str]:
        return [f"cx q[{x}],q[{y}];", f"cx q[{x}],q[{c}];", f"ccx q[{c}],q[{y}],q[{x}];"]

    def uma(c: int, y: int, x: int) -> list[str]:
        return [f"ccx q[{c}],q[{y}],q[{x}];", f"cx q[{x}],q[{c}];", f"cx q[{c}],q[{y}];"]

    carry = [cin] + a[:-1]
    for j in range(k):
        body += maj(carry[j], b[j], a[j])
    body.append(f"cx q[{a[-1]}],q[{cout}];")
    for j in reversed(range(k)):
        body += uma(carry[j], b[j], a[j])
    return _wrap(n, body)


def mirror(n: int, seed: int = 0, layers: int | None = None) -> str:
    rng = random.Random(seed)
    fwd: list[tuple[str, tuple]] = []
    for _ in range(layers or max(2, n // 2)):
        for q in range(n):
            fwd.append(("u3", (rng.uniform(0, math.pi), rng.uniform(-math.pi, math.pi),
                               rng.uniform(-math.pi, math.pi), q)))
        perm = list(range(n))
        rng.shuffle(perm)
        fwd += [("cx", (x, y)) for x, y in zip(perm[0::2], perm[1::2], strict=False)]

    def emit(op: str, args: tuple, inverse: bool) -> str:
        if op == "cx":
            return f"cx q[{args[0]}],q[{args[1]}];"
        t, p, lam, q = args
        if inverse:                              # u3(θ,φ,λ)† = u3(−θ,−λ,−φ)
            t, p, lam = -t, -lam, -p
        return f"u3({_a(t)},{_a(p)},{_a(lam)}) q[{q}];"

    body = [emit(o, a, False) for o, a in fwd] + ["barrier q;"] + [emit(o, a, True) for o, a in reversed(fwd)]
    return _wrap(n, body)


def random_gates(n: int, gates: int, seed: int = 0) -> str:
    """``gates`` random gates (≈ 70 % one-qubit, 30 % CX) for compile-throughput tests."""
    rng = random.Random(seed)
    body = []
    for _ in range(gates):
        if rng.random() < 0.3:
            a, b = rng.sample(range(n), 2)
            body.append(f"cx q[{a}],q[{b}];")
        else:
            body.append(f"{rng.choice(('h', 'x', 's', 't', 'sdg'))} q[{rng.randrange(n)}];")
    return _wrap(n, body)


FAMILIES: dict[str, Callable[..., str]] = {
    "ghz": ghz, "qft": qft, "qv": qv, "qaoa": qaoa, "hea": hea, "ising": ising, "bv": bv,
    "adder": adder, "mirror": mirror,
}

LABELS = {"ghz": "GHZ", "qft": "QFT adder", "qv": "QV-style", "qaoa": "QAOA (p=2)", "hea": "HEA (VQE)",
          "ising": "Ising (Trotter)", "bv": "Bernstein–Vazirani", "adder": "Cuccaro adder",
          "mirror": "Mirror"}


def heavy_outputs(probs: dict[str, float]) -> set[str]:
    """Outcomes whose ideal probability exceeds the median over all 2ⁿ outcomes (QV definition)."""
    n = len(next(iter(probs)))
    full = sorted([probs.get(format(i, f"0{n}b"), 0.0) for i in range(2 ** n)])
    med = (full[len(full) // 2 - 1] + full[len(full) // 2]) / 2
    return {k for k, p in probs.items() if p > med}


def hellinger_fidelity(p: dict[str, float], q: dict[str, float]) -> float:
    return sum(math.sqrt(p.get(k, 0.0) * q.get(k, 0.0)) for k in set(p) | set(q)) ** 2


def normalise(counts: dict[str, int]) -> dict[str, float]:
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}
