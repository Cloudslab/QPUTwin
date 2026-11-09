"""Embedded OpenQASM gate libraries.

``include`` statements never touch the filesystem: only the names below resolve,
to texts embedded in this module. This removes a whole class of path-traversal
issues from a network-facing service and makes every circuit's semantics fully
determined by the request payload plus this (version-controlled) file.

Libraries
---------
``qelib1.inc``
    The OpenQASM 2 standard header (Cross et al., arXiv:1707.03429), followed by
    the de-facto extensions in common use (``p``, ``sx``, ``swap``, ``cswap``,
    ``crx``/``cry``, ``cp``, ``cu``, ``csx``, ``rxx``, ``rzz``, ``rccx``). As in the
    specification, gates are defined only up to global phase.

``stdgates.inc``
    OpenQASM 3 standard gates, written so that every definition equals its
    canonical matrix **exactly** (including global phase, via ``gphase``). Exact
    phases matter in OpenQASM 3 because ``ctrl @`` turns a global phase into an
    observable relative phase.

Built-in library (always loaded, QPUTwin extension)
    The native gates of all seven emulated devices plus a few common two-qubit
    interactions, defined exactly in terms of ``U``/``CX``/``gphase``. Because
    every native gate has a portable definition, a circuit written in one
    vendor's native gates can still be compiled for another vendor. Helper gates
    are prefixed ``_qputwin_`` so user includes can never alter their meaning.

Library definitions are verified against the closed-form matrices of
:mod:`qputwin_core.gates` in ``tests/test_qasm_library.py``.

Koustav Kumar Mondal <erkoustavkumar@gmail.com>
"""

from __future__ import annotations

QELIB1_INC = r"""
// OpenQASM 2.0 standard header (qelib1.inc) + common extensions.
gate u3(theta,phi,lambda) q { U(theta,phi,lambda) q; }
gate u2(phi,lambda) q { U(pi/2,phi,lambda) q; }
gate u1(lambda) q { U(0,0,lambda) q; }
gate cx c,t { CX c,t; }
gate id a { U(0,0,0) a; }
gate u0(gamma) q { U(0,0,0) q; }
gate u(theta,phi,lambda) q { U(theta,phi,lambda) q; }
gate p(lambda) q { U(0,0,lambda) q; }
gate x a { u3(pi,0,pi) a; }
gate y a { u3(pi,pi/2,pi/2) a; }
gate z a { u1(pi) a; }
gate h a { u2(0,pi) a; }
gate s a { u1(pi/2) a; }
gate sdg a { u1(-pi/2) a; }
gate t a { u1(pi/4) a; }
gate tdg a { u1(-pi/4) a; }
gate rx(theta) a { u3(theta,-pi/2,pi/2) a; }
gate ry(theta) a { u3(theta,0,0) a; }
gate rz(phi) a { u1(phi) a; }
gate sx a { sdg a; h a; sdg a; }
gate sxdg a { s a; h a; s a; }
gate cz a,b { h b; cx a,b; h b; }
gate cy a,b { sdg b; cx a,b; s b; }
gate swap a,b { cx a,b; cx b,a; cx a,b; }
gate ch a,b { ry(-pi/4) b; cz a,b; ry(pi/4) b; }
gate ccx a,b,c
{
  h c;
  cx b,c; tdg c;
  cx a,c; t c;
  cx b,c; tdg c;
  cx a,c; t b; t c; h c;
  cx a,b; t a; tdg b;
  cx a,b;
}
gate cswap a,b,c { cx c,b; ccx a,b,c; cx c,b; }
gate crz(lambda) a,b { rz(lambda/2) b; cx a,b; rz(-lambda/2) b; cx a,b; }
gate cry(lambda) a,b { ry(lambda/2) b; cx a,b; ry(-lambda/2) b; cx a,b; }
gate crx(lambda) a,b { h b; crz(lambda) a,b; h b; }
gate cu1(lambda) a,b { u1(lambda/2) a; cx a,b; u1(-lambda/2) b; cx a,b; u1(lambda/2) b; }
gate cp(lambda) a,b { p(lambda/2) a; cx a,b; p(-lambda/2) b; cx a,b; p(lambda/2) b; }
gate cu3(theta,phi,lambda) c,t
{
  u1((lambda+phi)/2) c;
  u1((lambda-phi)/2) t;
  cx c,t;
  u3(-theta/2,0,-(phi+lambda)/2) t;
  cx c,t;
  u3(theta/2,phi,0) t;
}
gate cu(theta,phi,lambda,gamma) c,t
{
  p(gamma) c;
  p((lambda+phi)/2) c;
  p((lambda-phi)/2) t;
  cx c,t;
  u(-theta/2,0,-(phi+lambda)/2) t;
  cx c,t;
  u(theta/2,phi,0) t;
}
gate csx a,b { h b; cu1(pi/2) a,b; h b; }
gate rzz(theta) a,b { cx a,b; u1(theta) b; cx a,b; }
gate rxx(theta) a,b { h a; h b; cx a,b; u1(theta) b; cx a,b; h a; h b; }
gate rccx a,b,c
{
  u2(0,pi) c;
  u1(pi/4) c;
  cx b,c;
  u1(-pi/4) c;
  cx a,c;
  u1(pi/4) c;
  cx b,c;
  u1(-pi/4) c;
  u2(0,pi) c;
}
"""

STDGATES_INC = r"""
// OpenQASM 3 standard gates — phase-exact definitions (QPUTwin).
gate p(λ) a { U(0, 0, λ) a; }
gate phase(λ) a { U(0, 0, λ) a; }
gate x a { U(π, 0, π) a; }
gate y a { U(π, π/2, π/2) a; }
gate z a { U(0, 0, π) a; }
gate h a { U(π/2, 0, π) a; }
gate s a { U(0, 0, π/2) a; }
gate sdg a { U(0, 0, -π/2) a; }
gate t a { U(0, 0, π/4) a; }
gate tdg a { U(0, 0, -π/4) a; }
gate sx a { gphase(π/4); U(π/2, -π/2, π/2) a; }
gate sxdg a { gphase(-π/4); U(-π/2, -π/2, π/2) a; }
gate rx(θ) a { U(θ, -π/2, π/2) a; }
gate ry(θ) a { U(θ, 0, 0) a; }
gate rz(λ) a { gphase(-λ/2); U(0, 0, λ) a; }
gate id a { U(0, 0, 0) a; }
gate u1(λ) a { U(0, 0, λ) a; }
gate u2(φ, λ) a { U(π/2, φ, λ) a; }
gate u3(θ, φ, λ) a { U(θ, φ, λ) a; }
gate u(θ, φ, λ) a { U(θ, φ, λ) a; }
gate cx a, b { CX a, b; }
gate cy a, b { sdg b; cx a, b; s b; }
gate cz a, b { h b; cx a, b; h b; }
gate cp(λ) a, b { p(λ/2) a; cx a, b; p(-λ/2) b; cx a, b; p(λ/2) b; }
gate cphase(λ) a, b { cp(λ) a, b; }
gate cu1(λ) a, b { cp(λ) a, b; }
gate crz(θ) a, b { rz(θ/2) b; cx a, b; rz(-θ/2) b; cx a, b; }
gate cry(θ) a, b { ry(θ/2) b; cx a, b; ry(-θ/2) b; cx a, b; }
gate crx(θ) a, b { h b; crz(θ) a, b; h b; }
gate ch a, b { ry(-π/4) b; cz a, b; ry(π/4) b; }
gate swap a, b { cx a, b; cx b, a; cx a, b; }
gate cu3(θ, φ, λ) c, t
{
  p((λ+φ)/2) c;
  p((λ-φ)/2) t;
  cx c, t;
  U(-θ/2, 0, -(φ+λ)/2) t;
  cx c, t;
  U(θ/2, φ, 0) t;
}
gate cu(θ, φ, λ, γ) c, t { p(γ) c; cu3(θ, φ, λ) c, t; }
gate ccx a, b, c
{
  h c;
  cx b, c; tdg c;
  cx a, c; t c;
  cx b, c; tdg c;
  cx a, c; t b; t c; h c;
  cx a, b; t a; tdg b;
  cx a, b;
}
gate cswap a, b, c { cx c, b; ccx a, b, c; cx c, b; }
"""

BUILTIN_INC = r"""
// QPUTwin built-in library: vendor native gates + common interactions.
// Every definition is phase-exact with respect to qputwin_core.gates.
gate _qputwin_h a { U(pi/2, 0, pi) a; }
gate _qputwin_rz(t) a { gphase(-t/2); U(0, 0, t) a; }
gate _qputwin_rx(t) a { U(t, -pi/2, pi/2) a; }
gate _qputwin_rzz(t) a, b { CX a, b; _qputwin_rz(t) b; CX a, b; }

// --- universal virtual Z rotation (every device)
gate rz(t) a { _qputwin_rz(t) a; }

// --- Quantinuum (Helios / H-series)
gate u1q(theta, phi) a { U(theta, phi - pi/2, pi/2 - phi) a; }
gate rzz(t) a, b { _qputwin_rzz(t) a, b; }
gate zzmax a, b { _qputwin_rzz(pi/2) a, b; }

// --- IonQ (Forte)
gate gpi(phi) a { U(pi, phi, pi - phi) a; }
gate gpi2(phi) a { U(pi/2, phi - pi/2, pi/2 - phi) a; }
gate ms(phi0, phi1, t) a, b
{
  _qputwin_rz(-phi0) a; _qputwin_rz(-phi1) b;
  _qputwin_h a; _qputwin_h b;
  _qputwin_rzz(t) a, b;
  _qputwin_h a; _qputwin_h b;
  _qputwin_rz(phi0) a; _qputwin_rz(phi1) b;
}
gate zz(t) a, b { _qputwin_rzz(t) a, b; }

// --- Rigetti (Ankaa-3)
gate rx(t) a { _qputwin_rx(t) a; }
gate cz a, b { _qputwin_h b; CX a, b; _qputwin_h b; }
gate iswap a, b
{
  U(0, 0, pi/2) a; U(0, 0, pi/2) b;
  _qputwin_h a;
  CX a, b; CX b, a;
  _qputwin_h b;
}

// --- IQM (Emerald)
gate prx(theta, phi) a { U(theta, phi - pi/2, pi/2 - phi) a; }

// --- IBM (Heron, Nighthawk): sx = e^{i pi/4} rx(pi/2), x = e^{i pi/2} rx(pi); rz and cz above
gate sx a { gphase(pi/4); _qputwin_rx(pi/2) a; }
gate x a { gphase(pi/2); _qputwin_rx(pi) a; }

// --- Google (Willow): PhasedXZ in radians = rz(zeta) . prx(theta, phi); cz above
gate phased_xz(theta, phi, zeta) a { U(theta, phi - pi/2, pi/2 - phi) a; _qputwin_rz(zeta) a; }

// --- common two-qubit interactions
gate rxx(t) a, b { _qputwin_h a; _qputwin_h b; _qputwin_rzz(t) a, b; _qputwin_h a; _qputwin_h b; }
gate ryy(t) a, b
{
  _qputwin_rx(pi/2) a; _qputwin_rx(pi/2) b;
  _qputwin_rzz(t) a, b;
  _qputwin_rx(-pi/2) a; _qputwin_rx(-pi/2) b;
}
"""

LIBRARIES: dict[str, str] = {
    "qelib1.inc": QELIB1_INC,
    "stdgates.inc": STDGATES_INC,
}

__all__ = ["QELIB1_INC", "STDGATES_INC", "BUILTIN_INC", "LIBRARIES"]
