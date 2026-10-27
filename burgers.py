"""Inviscid Burgers on the periodic unit interval: WENO5-JS + SSP-RK3 finite volumes, and the
exact entropy solution's Fourier coefficients for u0 = m + sin(2 pi x).

Exact solution: v(x,t) for v0 = sin(2 pi x) has a stationary shock at x = 1/2; u(x,t) = v(x - m t, t).
Fourier coefficients (integer k) through the Lagrangian map x = xi + t sin(2 pi xi):
  v_hat(k) = int_{xi_R - 1}^{xi_L} sin(2 pi xi) exp(-2 pi i k (xi + t sin 2 pi xi)) (1 + 2 pi t cos 2 pi xi) d xi
where xi_L in (0, 1/2) solves xi + t sin(2 pi xi) = 1/2 (xi != 1/2), xi_R = 1 - xi_L.
Evaluated with composite Gauss-Legendre (panels refined with |k|), checked by doubling.
"""
import numpy as np
from scipy.optimize import brentq
from numpy.polynomial import legendre as npleg

TWO_PI = 2 * np.pi


# ---------------------------------------------------------------- WENO5 finite volumes
def _weno5_left(v):
    """Left-biased WENO5-JS reconstruction at i+1/2 from cell averages v (periodic)."""
    vm2, vm1, v0, vp1, vp2 = (np.roll(v, 2), np.roll(v, 1), v, np.roll(v, -1), np.roll(v, -2))
    q0 = (2 * vm2 - 7 * vm1 + 11 * v0) / 6
    q1 = (-vm1 + 5 * v0 + 2 * vp1) / 6
    q2 = (2 * v0 + 5 * vp1 - vp2) / 6
    b0 = 13 / 12 * (vm2 - 2 * vm1 + v0) ** 2 + 0.25 * (vm2 - 4 * vm1 + 3 * v0) ** 2
    b1 = 13 / 12 * (vm1 - 2 * v0 + vp1) ** 2 + 0.25 * (vm1 - vp1) ** 2
    b2 = 13 / 12 * (v0 - 2 * vp1 + vp2) ** 2 + 0.25 * (3 * v0 - 4 * vp1 + vp2) ** 2
    e = 1e-6
    a0, a1, a2 = 0.1 / (e + b0) ** 2, 0.6 / (e + b1) ** 2, 0.3 / (e + b2) ** 2
    return (a0 * q0 + a1 * q1 + a2 * q2) / (a0 + a1 + a2)


def _rhs(u, h):
    # Lax-Friedrichs flux splitting f^{+-} = (u^2/2 +- a u)/2
    a = np.max(np.abs(u))
    fp = 0.5 * (0.5 * u * u + a * u)
    fm = 0.5 * (0.5 * u * u - a * u)
    Fp = _weno5_left(fp)                                   # at i+1/2 from the left
    Fm = _weno5_left(fm[::-1])[::-1]                       # mirrored: at i-1/2 from the right
    Fm = np.roll(Fm, -1)                                   # -> at i+1/2
    F = Fp + Fm
    return -(F - np.roll(F, 1)) / h


def solve_fv(N, T, m=0.3, cfl=0.4):
    xe = np.linspace(0, 1, N + 1); h = 1.0 / N
    # exact cell averages of u0 = m + sin(2 pi x)
    u = m + (np.cos(TWO_PI * xe[:-1]) - np.cos(TWO_PI * xe[1:])) / (TWO_PI * h)
    t = 0.0
    while t < T - 1e-14:
        dt = min(cfl * h / np.max(np.abs(u)), T - t)
        u1 = u + dt * _rhs(u, h)
        u2 = 0.75 * u + 0.25 * (u1 + dt * _rhs(u1, h))
        u = u / 3 + 2 / 3 * (u2 + dt * _rhs(u2, h))
        t += dt
    return xe, u


# ---------------------------------------------------------------- exact spectrum
def shock_feet(t):
    g = lambda xi: xi + t * np.sin(TWO_PI * xi) - 0.5
    xl = brentq(g, 0.0, 0.5 - 1e-12) if t > 1 / TWO_PI else 0.5
    return xl, 1 - xl


def exact_coeffs(k, t, m=0.3, ppw=24, nmin=64):
    """u_hat(k) = int_0^1 u(x,t) e^{-2 pi i k x} dx for integer k (periodic)."""
    k = np.atleast_1d(np.asarray(k, float))
    xl, xr = shock_feet(t)
    a, b = xr - 1, xl
    out = np.empty(k.size, complex)
    tg, wg = npleg.leggauss(20)
    for i, kk in enumerate(k):
        nwav = abs(kk) * (b - a) * (1 + TWO_PI * t) + 1
        P = int(max(nmin, np.ceil(nwav * ppw / 20)))
        e = np.linspace(a, b, P + 1)
        xi = (0.5 * (e[:-1] + e[1:])[:, None] + 0.5 * np.diff(e)[:, None] * tg[None, :]).ravel()
        w = (0.5 * np.diff(e)[:, None] * wg[None, :]).ravel()
        s, c = np.sin(TWO_PI * xi), np.cos(TWO_PI * xi)
        out[i] = np.sum(w * s * np.exp(-1j * TWO_PI * kk * (xi + t * s)) * (1 + TWO_PI * t * c))
    # v_hat over a period centred at 0 equals the coefficient on [0,1); add mean, shift by m t
    out = out * np.exp(-1j * TWO_PI * k * m * t)
    out[k == 0] += m
    return out


def shock_position(t, m=0.3):
    return (0.5 + m * t) % 1.0


def exact_cell_averages(N, t, m=0.3, ng=24):
    """Exact cell averages of the entropy solution on N uniform cells of [0,1).

    v(x,t) (stationary shock at 1/2) has, on the period (-1/2, 1/2), the Lagrangian map
    x = xi + t sin(2 pi xi), xi in (xi_R - 1, xi_L), monotone there. Cell edges are mapped back
    to xi by root finding; each cell is integrated in xi with Gauss-Legendre (smooth integrand)."""
    from numpy.polynomial import legendre as npleg
    xl, xr = shock_feet(t)
    lo, hi = xr - 1, xl
    X = lambda xi: xi + t * np.sin(TWO_PI * xi)
    xe = np.linspace(0, 1, N + 1)
    # v-frame coordinates of the edges, unwrapped into (-1/2, 1/2]: y = x - m t (mod 1)
    tg, wg = npleg.leggauss(ng)
    ab = np.empty(N)
    for j in range(N):
        a, b = xe[j] - m * t, xe[j + 1] - m * t
        # split the cell at the shock (y = 1/2 mod 1) and at the period seam
        pieces = []
        ya = (a + 0.5) % 1.0 - 0.5
        yb = ya + (b - a)
        if yb <= 0.5 + 1e-15:
            pieces.append((ya, min(yb, 0.5)))
        else:
            pieces.append((ya, 0.5)); pieces.append((-0.5, yb - 1.0))
        tot = 0.0
        for p, q in pieces:
            if q - p <= 0:
                continue
            g = lambda xi, y: X(xi) - y
            xp = brentq(g, lo, hi, args=(p,)) if p > -0.5 + 1e-15 else lo
            xq = brentq(g, lo, hi, args=(q,)) if q < 0.5 - 1e-15 else hi
            xi = 0.5 * (xp + xq) + 0.5 * (xq - xp) * tg
            sv, cv = np.sin(TWO_PI * xi), np.cos(TWO_PI * xi)
            tot += 0.5 * (xq - xp) * np.sum(wg * sv * (1 + TWO_PI * t * cv))
        ab[j] = tot * N + m
    return xe, ab
