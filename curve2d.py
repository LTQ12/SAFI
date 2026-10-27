"""Star-shaped smooth closed curves on the pixel grid: geometry shared by the test generator
(bench2d_curve.py) and the curved-interface reconstruction (safi2d_curve.py).

A curve is r(t) = c + rho(t) (cos t, sin t), t in [0, 2 pi), traversed counter-clockwise.
Pixel q_{ij} = [i h, (i+1) h] x [j h, (j+1) h].  For integrals over q n Omega we use Green's theorem
with Q(x, y) = int_{x_i}^{x} p(s, y) ds, so that

  int_{q n Omega} p dA = sum_{curve pieces in q} int Q(r(t)) y'(t) dt + int_{right edge of q, inside Omega} Q dy,

since Q = 0 on the left edge and dy = 0 on the horizontal edges.  `pieces` returns the parameter
intervals of the curve inside each pixel; `right_edges` returns, for every pixel, the parts of its
right edge that lie inside Omega.
"""
import numpy as np
from scipy.optimize import brentq

TWO_PI = 2 * np.pi


class StarCurve:
    def __init__(self, center, rho, drho, d2rho=None):
        self.c = np.asarray(center, float)
        self.rho, self.drho, self.d2rho = rho, drho, d2rho

    # ------------------------------------------------------------ geometry
    def xy(self, t):
        r = self.rho(t)
        return self.c[0] + r * np.cos(t), self.c[1] + r * np.sin(t)

    def dxy(self, t):
        r, dr = self.rho(t), self.drho(t)
        return dr * np.cos(t) - r * np.sin(t), dr * np.sin(t) + r * np.cos(t)

    def curvature(self, t):
        r, dr, d2r = self.rho(t), self.drho(t), self.d2rho(t)
        return (r * r + 2 * dr * dr - r * d2r) / (r * r + dr * dr) ** 1.5

    def inside(self, x, y):
        a = np.arctan2(y - self.c[1], x - self.c[0])
        return np.hypot(x - self.c[0], y - self.c[1]) < self.rho(np.mod(a, TWO_PI))

    def perimeter(self, N=4096):
        t = np.arange(N) * TWO_PI / N
        dx, dy = self.dxy(t)
        return float(np.mean(np.hypot(dx, dy)) * TWO_PI)

    # ------------------------------------------------------------ grid crossings
    def crossings(self, h, ns_per_pixel=48):
        """All parameters where the curve crosses a grid line x = k h or y = k h, sorted.
        Returns (tc, kind, k): kind 0 = vertical line x = k h, 1 = horizontal line y = k h."""
        N = int(max(8192, ns_per_pixel * self.perimeter() / h))
        # start at an irrational offset: a curve whose extreme point lies exactly on a grid line
        # at t = 0 (e.g. an ellipse centred on a grid line) would otherwise lose that crossing to
        # the rounding of y(0) and y(2 pi)
        t = 0.6180339887498949 + np.arange(N + 1) * TWO_PI / N
        X, Y = self.xy(t)
        out = []
        for kind, V in ((0, X), (1, Y)):
            F = np.floor(V / h)
            idx = np.where(F[1:] != F[:-1])[0]
            for a in idx:
                lo, hi = F[a], F[a + 1]
                for k in range(int(min(lo, hi)) + 1, int(max(lo, hi)) + 1):
                    g = (lambda s, k=k: self.xy(s)[0] - k * h) if kind == 0 else (lambda s, k=k: self.xy(s)[1] - k * h)
                    tc = brentq(g, t[a], t[a + 1], xtol=1e-15, rtol=1e-15)
                    out.append((tc % TWO_PI, kind, k))
        out.sort()
        return out

    def pieces(self, h, cr=None):
        """List of (i, j, t0, t1) with t0 < t1 (t1 may exceed 2 pi): the curve inside pixel (i, j)."""
        cr = self.crossings(h) if cr is None else cr
        ts = [c[0] for c in cr]
        res = []
        for a in range(len(ts)):
            t0 = ts[a]
            t1 = ts[a + 1] if a + 1 < len(ts) else ts[0] + TWO_PI
            if t1 - t0 < 1e-15:
                continue
            xm, ym = self.xy(0.5 * (t0 + t1))
            res.append((int(np.floor(xm / h)), int(np.floor(ym / h)), t0, t1))
        return res

    def right_edges(self, h, n, cr=None):
        """dict (i, j) -> list of (y0, y1): parts of the right edge x = (i+1) h of pixel (i, j) inside Omega."""
        cr = self.crossings(h) if cr is None else cr
        lines = {}
        for tc, kind, k in cr:
            if kind == 0:
                lines.setdefault(k, []).append(self.xy(tc)[1])
        out = {}
        for k, ys in lines.items():
            ys = np.sort(ys)
            i = k - 1
            for a in range(0, len(ys) - 1, 2):
                y0, y1 = ys[a], ys[a + 1]
                for j in range(int(np.floor(y0 / h)), int(np.floor(y1 / h)) + 1):
                    lo, hi = max(y0, j * h), min(y1, (j + 1) * h)
                    if hi > lo and 0 <= i < n and 0 <= j < n:
                        out.setdefault((i, j), []).append((lo, hi))
        return out


def fourier_rho(coef):
    """rho(t) = a0 + sum_j (a_j cos jt + b_j sin jt); coef = [a0, a1, b1, a2, b2, ...]."""
    coef = np.asarray(coef, float)
    N = (coef.size - 1) // 2
    js = np.arange(1, N + 1)
    a, b = coef[1::2], coef[2::2]

    def rho(t):
        t = np.asarray(t, float)
        return coef[0] + np.cos(np.multiply.outer(t, js)) @ a + np.sin(np.multiply.outer(t, js)) @ b

    def drho(t):
        t = np.asarray(t, float)
        return -np.sin(np.multiply.outer(t, js)) @ (a * js) + np.cos(np.multiply.outer(t, js)) @ (b * js)

    def d2rho(t):
        t = np.asarray(t, float)
        return -np.cos(np.multiply.outer(t, js)) @ (a * js ** 2) - np.sin(np.multiply.outer(t, js)) @ (b * js ** 2)

    def basis(t):
        t = np.asarray(t, float)
        cols = [np.ones_like(t)]
        for j in js:
            cols += [np.cos(j * t), np.sin(j * t)]
        return np.stack(cols, -1)
    return rho, drho, d2rho, basis


def exp_trig_rho(R0, eps, phis):
    """rho(t) = R0 exp(sum_j eps_j cos(j t + phi_j)), j = 2, 3, ...: analytic, infinitely many Fourier modes."""
    js = np.arange(2, 2 + len(eps))
    eps, phis = np.asarray(eps, float), np.asarray(phis, float)

    def s(t):
        return np.cos(np.multiply.outer(t, js) + phis) @ eps

    def ds(t):
        return -np.sin(np.multiply.outer(t, js) + phis) @ (eps * js)

    def d2s(t):
        return -np.cos(np.multiply.outer(t, js) + phis) @ (eps * js ** 2)

    rho = lambda t: R0 * np.exp(s(np.asarray(t, float)))
    drho = lambda t: rho(t) * ds(np.asarray(t, float))
    d2rho = lambda t: rho(t) * (ds(np.asarray(t, float)) ** 2 + d2s(np.asarray(t, float)))
    return rho, drho, d2rho


def ellipse_curve(cx, cy, a, b, phi=0.0):
    """Ellipse with semi-axes a (along angle phi) and b, as a StarCurve about its centre."""
    d2 = a * a - b * b

    def D(t):
        s = t - phi
        return b * b * np.cos(s) ** 2 + a * a * np.sin(s) ** 2

    def rho(t):
        return a * b / np.sqrt(D(t))

    def drho(t):
        s = t - phi
        return -0.5 * a * b * D(t) ** -1.5 * d2 * np.sin(2 * s)

    def d2rho(t):
        s = t - phi
        Dp = d2 * np.sin(2 * s)
        Dpp = 2 * d2 * np.cos(2 * s)
        return a * b * (0.75 * D(t) ** -2.5 * Dp ** 2 - 0.5 * D(t) ** -1.5 * Dpp)

    return StarCurve((cx, cy), rho, drho, d2rho)
