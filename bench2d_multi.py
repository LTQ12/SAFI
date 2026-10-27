"""Images with several smooth closed interfaces (disjoint or nested), exact pixel averages and
exact Fourier transform.

f = g_0 on the background; inside curve k (and outside the curves nested in it) f = g_k.
Curve k has a parent (the curve whose interior contains it, or None for the background), so
    f = g_0 + sum_k (g_k - g_parent(k)) 1_{Omega_k},
and every term is handled by bench2d_curve.CurveImage (Green's theorem, near machine precision).
Phases g are bench2d.Phase objects (real parts of sums of complex exponentials).
"""
import numpy as np
from bench2d import Phase
from bench2d_curve import CurveImage
from curve2d import StarCurve, exp_trig_rho, ellipse_curve, TWO_PI


def scaled_phase(g, s):
    """Phase s * g for a real scalar s."""
    return Phase(np.asarray(g.alpha) * s, g.beta)


def product_phase(coil, value):
    """value * coil (value real scalar): the region intensity times a smooth coil sensitivity."""
    return scaled_phase(coil, value)


class MultiCurveImage:
    def __init__(self, bg, curves, phases, parents):
        self.bg, self.curves, self.phases, self.parents = bg, curves, phases, parents
        self.sub = []
        for C, g, p in zip(curves, phases, parents):
            outer = bg if p is None else phases[p]
            self.sub.append(CurveImage(outer, C, g))

    def region_of(self, x, y):
        """Index of the innermost curve containing (x, y), or -1 (background)."""
        x = np.asarray(x, float); y = np.asarray(y, float)
        lab = -np.ones(np.broadcast(x, y).shape, int)
        depth = -np.ones(lab.shape, int)
        for k, C in enumerate(self.curves):
            d, p = 0, self.parents[k]
            while p is not None:
                d += 1; p = self.parents[p]
            ins = C.inside(x, y) & (d > depth)
            lab[ins] = k; depth[ins] = d
        return lab

    def __call__(self, x, y):
        lab = self.region_of(x, y)
        out = self.bg(x, y)
        for k, g in enumerate(self.phases):
            out = np.where(lab == k, g(x, y), out)
        return out

    def pixel_averages(self, n):
        h = 1.0 / n
        xe = np.linspace(0, 1, n + 1)
        A = np.zeros((n, n))
        for c, w in self.bg.terms():
            ex = (np.exp(w[0] * xe[1:]) - np.exp(w[0] * xe[:-1])) / w[0] if abs(w[0]) > 1e-14 else np.full(n, h)
            ey = (np.exp(w[1] * xe[1:]) - np.exp(w[1] * xe[:-1])) / w[1] if abs(w[1]) > 1e-14 else np.full(n, h)
            A = A + np.real(c * np.outer(ex, ey))
        for s in self.sub:
            A = A + np.real(s.region_integrals(n))
        return A / h ** 2

    def ft(self, u, v):
        u = np.atleast_1d(np.asarray(u, float)); v = np.atleast_1d(np.asarray(v, float))
        sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
        F = self.bg.integral(sq, u, v).astype(complex)
        for s in self.sub:
            F = F + s.region_ft_fast(u, v)
        return F

    # ------------------------------------------------------------ generators
    @classmethod
    def random(cls, rng, ninc=3, nested=1, osc=0.7, sep=0.06):
        """ninc disjoint star-shaped inclusions (exp-trig radius), `nested` of which contain a
        smaller inclusion; all interfaces at least `sep` apart."""
        curves, phases, parents = [], [], []
        bg = Phase.random(rng, osc=osc, decay=1.0)
        tries = 0
        outer = []
        while len(outer) < ninc and tries < 5000:
            tries += 1
            c = rng.uniform(0.2, 0.8, 2)
            r0 = rng.uniform(0.09, 0.16)
            e = rng.uniform(-0.12, 0.12, 3); ph = rng.uniform(0, TWO_PI, 3)
            rho, drho, d2rho = exp_trig_rho(r0, e, ph)
            C = StarCurve(c, rho, drho, d2rho)
            t = np.linspace(0, TWO_PI, 2000, endpoint=False)
            x, y = C.xy(t)
            if C.curvature(t).min() < 1.0 or x.min() < 0.05 or y.min() < 0.05 or x.max() > 0.95 or y.max() > 0.95:
                continue
            if any(_min_dist(C, D) < sep for D, _ in outer):
                continue
            outer.append((C, r0))
        for C, r0 in outer:
            curves.append(C); parents.append(None)
            phases.append(Phase.random(np.random.default_rng(rng.integers(1 << 30)), osc=osc, decay=1.0,
                                       offset=rng.choice([-1, 1]) * rng.uniform(1.5, 3.0)))
        for k in range(min(nested, len(outer))):
            C, r0 = outer[k]
            for _ in range(200):
                ri = rng.uniform(0.35, 0.5) * r0
                cc = C.c + rng.uniform(-0.15, 0.15, 2) * r0
                e = rng.uniform(-0.1, 0.1, 3); ph = rng.uniform(0, TWO_PI, 3)
                rho, drho, d2rho = exp_trig_rho(ri, e, ph)
                Ci = StarCurve(cc, rho, drho, d2rho)
                t = np.linspace(0, TWO_PI, 2000, endpoint=False)
                xi, yi = Ci.xy(t)
                if not np.all(C.inside(xi, yi)) or _min_dist(C, Ci) < sep:
                    continue
                curves.append(Ci); parents.append(k)
                phases.append(Phase.random(np.random.default_rng(rng.integers(1 << 30)), osc=osc, decay=1.0,
                                           offset=rng.choice([-1, 1]) * rng.uniform(1.5, 3.0)))
                break
        return cls(bg, curves, phases, parents)


def _min_dist(C, D, N=720):
    t = np.linspace(0, TWO_PI, N, endpoint=False)
    x1, y1 = C.xy(t); x2, y2 = D.xy(t)
    return float(np.min(np.hypot(x1[:, None] - x2[None, :], y1[:, None] - y2[None, :])))


# Shepp-Logan (Toft's modified version), ellipses in [-1, 1]^2: (intensity, a, b, x0, y0, phi deg)
SHEPP_LOGAN = [(1.0, .69, .92, 0, 0, 0), (-.8, .6624, .874, 0, -.0184, 0),
               (-.2, .11, .31, .22, 0, -18), (-.2, .16, .41, -.22, 0, 18),
               (.1, .21, .25, 0, .35, 0), (.1, .046, .046, 0, .1, 0), (.1, .046, .046, 0, -.1, 0),
               (.1, .046, .023, -.08, -.605, 0), (.1, .023, .023, 0, -.606, 0), (.1, .023, .046, .06, -.605, 0)]


def shepp_logan(coil=None, drop=(), move=None):
    """Shepp-Logan phantom on [0,1]^2 as nested ellipses (additive intensities), optionally times a
    smooth coil sensitivity (a Phase). drop: ellipse indices removed; move: {index: (x0, y0)}.
    Requires that no two ellipse boundaries cross (crossing interfaces create corners)."""
    E = [list(e) for k, e in enumerate(SHEPP_LOGAN) if k not in drop]
    idx = [k for k in range(len(SHEPP_LOGAN)) if k not in drop]
    for k, (x0, y0) in (move or {}).items():
        E[idx.index(k)][3:5] = [x0, y0]
    curves = [ellipse_curve(0.5 + 0.5 * x0, 0.5 + 0.5 * y0, 0.5 * a, 0.5 * b, np.deg2rad(ph))
              for (I, a, b, x0, y0, ph) in E]
    # parent = smallest ellipse that contains this one
    t = np.linspace(0, TWO_PI, 720, endpoint=False)
    area = [np.pi * 0.25 * e[1] * e[2] for e in E]
    parents = []
    for k, C in enumerate(curves):
        x, y = C.xy(t)
        cont = [m for m, D in enumerate(curves) if m != k and np.all(D.inside(x, y))]
        parents.append(min(cont, key=lambda m: area[m]) if cont else None)
    # region intensity = sum of intensities of all ellipses containing it
    tot = []
    for k in range(len(E)):
        s, p = E[k][0], parents[k]
        while p is not None:
            s += E[p][0]; p = parents[p]
        tot.append(s)
    one = Phase(np.array([1.0 + 0j]), np.zeros((1, 2), complex)) if coil is None else coil
    bg = scaled_phase(one, 0.0)
    phases = [scaled_phase(one, v) for v in tot]
    return MultiCurveImage(bg, curves, phases, parents)
