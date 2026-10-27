"""Test images with a smooth curved inclusion: f = bg + (g - bg) 1_Omega, Omega star-shaped with
rho(t) = R0 exp(sum_j eps_j cos(j t + phi_j)) (analytic boundary, infinitely many Fourier modes).
Phases are bench2d.Phase (sums of complex exponentials), so all integrals reduce to boundary
integrals that are computed to near machine precision:
  pixel averages : Green's theorem on q n Omega (Gauss-Legendre on each curve piece, exact on edges);
  F(u, v)        : Green's theorem on Omega, periodic trapezoidal rule in t (spectrally accurate).
"""
import numpy as np
from numpy.polynomial import legendre as npleg
from bench2d import Phase
from curve2d import StarCurve, exp_trig_rho

TWO_PI = 2 * np.pi
_G, _W = npleg.leggauss(40)


class CurveImage:
    def __init__(self, bg, curve, phase):
        self.bg, self.curve, self.g = bg, curve, phase

    @classmethod
    def random(cls, rng, R0=(0.2, 0.28), eps=0.12, nmodes=3, osc=0.7):
        while True:
            c = rng.uniform(0.38, 0.62, 2)
            r0 = rng.uniform(*R0)
            e = rng.uniform(-eps, eps, nmodes)
            ph = rng.uniform(0, TWO_PI, nmodes)
            rho, drho, d2rho = exp_trig_rho(r0, e, ph)
            C = StarCurve(c, rho, drho, d2rho)
            t = np.linspace(0, TWO_PI, 4000, endpoint=False)
            x, y = C.xy(t)
            if C.curvature(t).min() > 0.5 and x.min() > 0.05 and y.min() > 0.05 and x.max() < 0.95 and y.max() < 0.95:
                break
        bg = Phase.random(rng, osc=osc, decay=1.0)
        g = Phase.random(np.random.default_rng(rng.integers(1 << 30)), osc=osc, decay=1.0,
                         offset=rng.choice([-1, 1]) * rng.uniform(1.5, 3.0))
        return cls(bg, C, g)

    def __call__(self, x, y):
        return np.where(self.curve.inside(x, y), self.g(x, y), self.bg(x, y))

    def _diff_terms(self):
        return self.g.terms() + [(-c, w) for c, w in self.bg.terms()]

    # ------------------------------------------------------------ data
    def pixel_averages(self, n):
        h = 1.0 / n
        xe = np.linspace(0, 1, n + 1)
        A = np.zeros((n, n))
        for c, w in self.bg.terms():
            ex = (np.exp(w[0] * xe[1:]) - np.exp(w[0] * xe[:-1])) / w[0] if abs(w[0]) > 1e-14 else np.full(n, h)
            ey = (np.exp(w[1] * xe[1:]) - np.exp(w[1] * xe[:-1])) / w[1] if abs(w[1]) > 1e-14 else np.full(n, h)
            A = A + np.real(c * np.outer(ex, ey))
        A = (A + np.real(self.region_integrals(n))) / h ** 2
        return A

    def region_integrals(self, n):
        """(n, n) array of int_{q n Omega} (g - bg) dA."""
        h = 1.0 / n
        C = self.curve
        cr = C.crossings(h)
        I = np.zeros((n, n), complex)
        terms = self._diff_terms()
        for (i, j, t0, t1) in C.pieces(h, cr):
            t = 0.5 * (t0 + t1) + 0.5 * (t1 - t0) * _G
            x, y = C.xy(t)
            _, dy = C.dxy(t)
            xa = i * h
            val = 0
            for c, w in terms:
                if abs(w[0]) > 1e-14:
                    Q = c * np.exp(w[1] * y) * (np.exp(w[0] * x) - np.exp(w[0] * xa)) / w[0]
                else:
                    Q = c * np.exp(w[1] * y) * (x - xa)
                val = val + Q
            I[i, j] += 0.5 * (t1 - t0) * np.sum(_W * val * dy)
        # pixels whose right edge is (partly) inside Omega; includes interior pixels
        for (i, j), segs in C.right_edges(h, n, cr).items():
            xa, xb = i * h, (i + 1) * h
            for y0, y1 in segs:
                for c, w in terms:
                    gx = (np.exp(w[0] * xb) - np.exp(w[0] * xa)) / w[0] if abs(w[0]) > 1e-14 else h
                    gy = (np.exp(w[1] * y1) - np.exp(w[1] * y0)) / w[1] if abs(w[1]) > 1e-14 else (y1 - y0)
                    I[i, j] += c * gx * gy
        return I

    # ------------------------------------------------------------ reference transform
    def ft(self, u, v, N=None):
        u = np.atleast_1d(np.asarray(u, float)); v = np.atleast_1d(np.asarray(v, float))
        sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
        F = self.bg.integral(sq, u, v).astype(complex)
        F = F + self.region_ft(u, v, N)
        return F

    def region_ft_fast(self, u, v, batch=128):
        """Vectorised region_ft: frequencies are processed in batches sorted by |k|, each batch with
        one trapezoid rule sized for its largest |k| (same accuracy as region_ft)."""
        u = np.atleast_1d(np.asarray(u, float)); v = np.atleast_1d(np.asarray(v, float))
        C = self.curve
        out = np.zeros(u.size, complex)
        kmax = np.hypot(u, v)
        perim = C.perimeter()
        order = np.argsort(kmax)
        terms = self._diff_terms()
        cache = {}
        for s in range(0, u.size, batch):
            idx = order[s:s + batch]
            NN = int(2 ** np.ceil(np.log2(4096 + 6 * kmax[idx].max() * perim)))
            if NN not in cache:
                t = np.arange(NN) * TWO_PI / NN
                cache = {NN: (C.xy(t), C.dxy(t))}
            (x, y), (dx, dy) = cache[NN]
            acc = np.zeros(idx.size, complex)
            E = np.exp(-1j * TWO_PI * (np.outer(u[idx], x) + np.outer(v[idx], y)))   # shared phase
            for c, w in terms:
                wx = w[0] - 1j * TWO_PI * u[idx]
                wy = w[1] - 1j * TWO_PI * v[idx]
                ew = np.exp(w[0] * x + w[1] * y)
                big_x = np.abs(wx) >= np.abs(wy)
                zero = (np.abs(wx) < 1e-14) & (np.abs(wy) < 1e-14)
                Ey, Ex = E @ (ew * dy), E @ (ew * dx)
                val = np.where(big_x, Ey / np.where(big_x, wx, 1), -Ex / np.where(big_x, 1, wy))
                val = np.where(zero, 0.5 * np.sum(x * dy - y * dx), val)
                acc += c * val
            out[idx] = acc * TWO_PI / NN
        return out

    def region_ft(self, u, v, N=None):
        """int_Omega (g - bg) exp(-2 pi i (u x + v y)) dA via oint Q dy (or -oint Q dx), trapezoid in t."""
        C = self.curve
        out = np.zeros(u.size, complex)
        kmax = np.hypot(u, v)
        perim = C.perimeter()
        for s in range(u.size):
            NN = N or int(2 ** np.ceil(np.log2(4096 + 6 * kmax[s] * perim)))
            t = np.arange(NN) * TWO_PI / NN
            x, y = C.xy(t)
            dx, dy = C.dxy(t)
            acc = 0
            for c, w in self._diff_terms():
                wx, wy = w[0] - 1j * TWO_PI * u[s], w[1] - 1j * TWO_PI * v[s]
                e = np.exp(wx * x + wy * y)
                if abs(wx) < 1e-14 and abs(wy) < 1e-14:
                    acc = acc + c * 0.5 * np.sum(x * dy - y * dx)      # area
                elif abs(wx) >= abs(wy):
                    acc = acc + c * np.sum(e * dy) / wx
                else:
                    acc = acc - c * np.sum(e * dx) / wy
            out[s] = acc * TWO_PI / NN
        return out
