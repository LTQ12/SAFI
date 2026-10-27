"""2-D test images with exact pixel averages and exact Fourier transforms.

Image on [0,1]^2: f = g_bg outside the polygons, f = g_k inside polygon k (disjoint polygons).
Each phase g = Re sum_r alpha_r exp(beta_r . x), beta_r in C^2.
Integrals of exp(w . x) over a polygon are exact via Green's theorem (edge sums of
E(z) = (e^z - 1)/z), with a Gauss rule on a triangle fan when |w| diam is small.
"""
import numpy as np
from numpy.polynomial import legendre as npleg

TWO_PI = 2 * np.pi


def _E(z):
    """(e^z - 1)/z, stable near 0."""
    z = np.asarray(z, complex)
    out = np.empty(z.shape, complex)
    sm = np.abs(z) < 1e-4
    out[sm] = 1 + z[sm] / 2 + z[sm] ** 2 / 6 + z[sm] ** 3 / 24
    out[~sm] = np.expm1(z[~sm]) / z[~sm]
    return out


_TG, _TW = npleg.leggauss(14)
_TG = 0.5 * (_TG + 1); _TW = 0.5 * _TW


def _tri_quad(P, fun):
    """Collapsed Gauss rule on a triangle fan of polygon P (k,2): integral of fun(x, y)."""
    tot = 0.0
    A = P[0]
    s, t = np.meshgrid(_TG, _TG, indexing='ij'); ws = np.outer(_TW, _TW)
    for i in range(1, len(P) - 1):
        B, C = P[i], P[i + 1]
        det = (B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0])
        x = A[0] + s * (B[0] - A[0]) + s * t * (C[0] - B[0])
        y = A[1] + s * (B[1] - A[1]) + s * t * (C[1] - B[1])
        tot = tot + np.sum(ws * s * fun(x, y)) * det
    return tot


def poly_exp_integral(P, wx, wy):
    """int_P exp(wx x + wy y) dA for polygon P (k,2, counter-clockwise), wx, wy arrays (same shape)."""
    P = np.asarray(P, float)
    wx = np.asarray(wx, complex); wy = np.asarray(wy, complex)
    shp = np.broadcast(wx, wy).shape
    wx = np.broadcast_to(wx, shp).ravel(); wy = np.broadcast_to(wy, shp).ravel()
    out = np.zeros(wx.size, complex)
    diam = np.max(np.ptp(P, axis=0)) + 1e-300
    big = np.maximum(np.abs(wx), np.abs(wy)) * diam > 0.5
    Pn = np.roll(P, -1, axis=0)
    D = Pn - P
    if big.any():
        ax = np.abs(wx[big]) >= np.abs(wy[big])
        wxb, wyb = wx[big], wy[big]
        c = wxb[:, None] * D[None, :, 0] + wyb[:, None] * D[None, :, 1]
        e0 = np.exp(wxb[:, None] * P[None, :, 0] + wyb[:, None] * P[None, :, 1])
        edge = e0 * _E(c)                                     # int_0^1 exp(w.(P + t D)) dt
        # dA-integral = (1/wx) oint e dy   or   -(1/wy) oint e dx
        res = np.where(ax, (edge * D[None, :, 1]).sum(1) / np.where(ax, wxb, 1),
                       -(edge * D[None, :, 0]).sum(1) / np.where(ax, 1, wyb))
        out[big] = res
    if (~big).any():
        idx = np.where(~big)[0]
        for k in idx:
            out[k] = _tri_quad(P, lambda x, y: np.exp(wx[k] * x + wy[k] * y))
    return out.reshape(shp)


def clip_square(P, x0, x1, y0, y1):
    """Sutherland-Hodgman clip of polygon P by the axis-aligned box."""
    def clip(poly, inside, inter):
        out = []
        n = len(poly)
        for i in range(n):
            cur, prv = poly[i], poly[i - 1]
            ci, pi = inside(cur), inside(prv)
            if ci:
                if not pi:
                    out.append(inter(prv, cur))
                out.append(cur)
            elif pi:
                out.append(inter(prv, cur))
        return out

    def ix(a, b, xv):
        t = (xv - a[0]) / (b[0] - a[0]); return (xv, a[1] + t * (b[1] - a[1]))

    def iy(a, b, yv):
        t = (yv - a[1]) / (b[1] - a[1]); return (a[0] + t * (b[0] - a[0]), yv)
    poly = [tuple(p) for p in P]
    for inside, inter in ((lambda p: p[0] >= x0, lambda a, b: ix(a, b, x0)),
                          (lambda p: p[0] <= x1, lambda a, b: ix(a, b, x1)),
                          (lambda p: p[1] >= y0, lambda a, b: iy(a, b, y0)),
                          (lambda p: p[1] <= y1, lambda a, b: iy(a, b, y1))):
        if not poly:
            break
        poly = clip(poly, inside, inter)
    return np.array(poly) if len(poly) >= 3 else None


def point_in_convex(P, x, y):
    P = np.asarray(P)
    ok = np.ones(np.shape(x), bool)
    for i in range(len(P)):
        a, b = P[i], P[(i + 1) % len(P)]
        ok &= (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0]) >= -1e-15
    return ok


class Phase:
    """g(x, y) = Re sum alpha exp(beta . (x, y))."""
    def __init__(self, alpha, beta):
        self.alpha = np.asarray(alpha, complex); self.beta = np.asarray(beta, complex)   # (R,), (R,2)

    @classmethod
    def random(cls, rng, R=3, osc=2.0, decay=1.5, scale=1.0, offset=0.0):
        a = scale * (rng.normal(size=R) + 1j * rng.normal(size=R)) / np.sqrt(R)
        b = rng.uniform(-decay, decay, (R, 2)) + 1j * TWO_PI * rng.uniform(-osc, osc, (R, 2))
        if offset:
            a = np.r_[a, offset]; b = np.vstack([b, [0, 0]])
        return cls(a, b)

    def terms(self):
        """Complex exponential terms (c, w) with g = sum c exp(w.x)."""
        out = []
        for al, be in zip(self.alpha, self.beta):
            out.append((al / 2, be)); out.append((np.conj(al) / 2, np.conj(be)))
        return out

    def __call__(self, x, y):
        v = 0
        for c, w in self.terms():
            v = v + c * np.exp(w[0] * x + w[1] * y)
        return np.real(v)

    def integral(self, P, u=0.0, v=0.0):
        """int_P g exp(-2 pi i (u x + v y)) over polygon P, vectorised in (u, v)."""
        u = np.asarray(u, float); v = np.asarray(v, float)
        tot = 0
        for c, w in self.terms():
            tot = tot + c * poly_exp_integral(P, w[0] - 1j * TWO_PI * u, w[1] - 1j * TWO_PI * v)
        return tot


class PolyImage:
    def __init__(self, bg, polys, phases):
        self.bg, self.polys, self.phases = bg, [np.asarray(p, float) for p in polys], phases

    @classmethod
    def random(cls, rng, npoly=1, nv=6, rmin=0.18, rmax=0.3, jump=1.0):
        bg = Phase.random(rng, offset=0.0)
        polys, phases = [], []
        centers = []
        while len(polys) < npoly:
            c = rng.uniform(0.3, 0.7, 2) if npoly == 1 else rng.uniform(0.25, 0.75, 2)
            if any(np.linalg.norm(c - cc) < 2 * rmax + 0.03 for cc in centers):
                continue
            ang = np.sort(rng.uniform(0, TWO_PI, nv))
            # keep angular gaps reasonable -> convex, non-degenerate
            if np.min(np.diff(np.r_[ang, ang[0] + TWO_PI])) < 0.4:
                continue
            r = rng.uniform(rmin, rmax)
            P = np.c_[c[0] + r * np.cos(ang), c[1] + r * np.sin(ang)]
            polys.append(P); centers.append(c)
            phases.append(Phase.random(rng, offset=jump * rng.choice([-1, 1]) * rng.uniform(1, 2)))
        return cls(bg, polys, phases)

    def __call__(self, x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        out = self.bg(x, y)
        for P, g in zip(self.polys, self.phases):
            ins = point_in_convex(P, x, y)
            out = np.where(ins, g(x, y), out)
        return out

    def ft(self, u, v):
        sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
        F = self.bg.integral(sq, u, v)
        for P, g in zip(self.polys, self.phases):
            F = F + g.integral(P, u, v) - self.bg.integral(P, u, v)
        return F

    def pixel_averages(self, n):
        """Exact pixel averages on an n x n grid (index [i, j] = x-cell i, y-cell j)."""
        h = 1.0 / n
        xe = np.linspace(0, 1, n + 1)
        # background: separable exact square integrals of each exp term
        A = np.zeros((n, n))
        for c, w in self.bg.terms():
            ex = (np.exp(w[0] * xe[1:]) - np.exp(w[0] * xe[:-1])) / w[0] if abs(w[0]) > 1e-14 else np.full(n, h)
            ey = (np.exp(w[1] * xe[1:]) - np.exp(w[1] * xe[:-1])) / w[1] if abs(w[1]) > 1e-14 else np.full(n, h)
            A = A + np.real(c * np.outer(ex, ey))
        for P, g in zip(self.polys, self.phases):
            i0, i1 = int(np.floor(P[:, 0].min() / h)), int(np.ceil(P[:, 0].max() / h))
            j0, j1 = int(np.floor(P[:, 1].min() / h)), int(np.ceil(P[:, 1].max() / h))
            for i in range(max(0, i0), min(n, i1)):
                for j in range(max(0, j0), min(n, j1)):
                    Q = clip_square(P, xe[i], xe[i + 1], xe[j], xe[j + 1])
                    if Q is None:
                        continue
                    Q = Q[::1]
                    A[i, j] += np.real(g.integral(Q) - self.bg.integral(Q))
        return A / (h * h)
