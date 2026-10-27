"""SAFI-2D: Fourier transform at arbitrary (u, v) of a piecewise-smooth image from pixel averages.

Pipeline (pixel = unit square in local scaled coordinates xi = (x - x_c)/h):
 1. mixed-pixel detection: 1-D ENO ring tests along rows and columns (+ step test), dilated by one;
 2. regions: connected components of the clean pixels;
 3. local phase fits: total-degree-P polynomial per pixel and region, least squares on the exact
    pixel-average basis over a (2R+1)^2 window of clean same-region pixels (own average enforced);
 4. edges: in every mixed pixel a line n.xi = d separating the two neighbouring phases, fitted to
    the 3x3 pixel averages (model averages computed exactly), then d corrected so that the pixel's
    own average is reproduced (2-D analogue of the subcell-resolution equation);
 5. evaluation: exact integrals of polynomial x exp over squares (separable) and over the two
    polygons of each mixed pixel (Green's theorem -> edge integrals -> Legendre/spherical Bessel).
"""
import numpy as np
from functools import lru_cache
from numpy.polynomial import legendre as npleg
from scipy.special import spherical_jn
from scipy.optimize import least_squares, brentq
from scipy import ndimage

TWO_PI = 2 * np.pi


def monos(P):
    return [(a, b) for a in range(P + 1) for b in range(P + 1 - a)]


# ------------------------------------------------------------------ 1-D helpers
def _eno_point_scores(v, M):
    n = v.size
    D = np.abs(np.diff(v, M + 1))
    sc = np.full(n, np.inf)
    for off in range(M + 2):
        j = np.arange(n); s = j - off
        ok = (s >= 0) & (s < D.size)
        sc[ok] = np.minimum(sc[ok], D[s[ok]])
    return sc


def _eno_pair_scores(v, M):
    n = v.size
    D = np.abs(np.diff(v, M + 1))
    sc = np.full(n - 1, np.inf)
    for off in range(M + 1):
        i = np.arange(n - 1); s = i - off
        ok = (s >= 0) & (s < D.size)
        sc[ok] = np.minimum(sc[ok], D[s[ok]])
    return sc


RING_MODE = 'two-stage'


def _ring_flags(sc, W, kappa, floor):
    if RING_MODE == 'max':
        return _ring_flags_max(sc, W, kappa, floor)
    return _ring_flags_2s(sc, W, kappa, floor)


def _ring_flags_max(sc, W, kappa, floor):
    n = sc.size
    out = np.zeros(n, bool)
    for i in range(n):
        if not np.isfinite(sc[i]) or sc[i] <= floor:
            continue
        ring = np.r_[max(0, i - 3 * W):max(0, i - W), min(n, i + W + 1):min(n, i + 3 * W + 1)]
        ring = ring[np.isfinite(sc[ring])]
        ref = np.max(sc[ring]) if ring.size else 0.0
        out[i] = sc[i] > kappa * ref
    return out


def _ring_flags_2s(sc, W, kappa, floor):
    """Two-stage ring test (as in 1-D): candidates by the ring median (robust to other edges inside
    the ring), confirmation by the ring maximum after removing all candidates (robust to zeros)."""
    n = sc.size
    fin = np.isfinite(sc)
    def ring(i):
        r = np.r_[max(0, i - 3 * W):max(0, i - W), min(n, i + W + 1):min(n, i + 3 * W + 1)]
        return r[fin[r]]
    cand = np.zeros(n, bool)
    for i in range(n):
        if not fin[i] or sc[i] <= floor:
            continue
        r = ring(i)
        cand[i] = r.size == 0 or sc[i] > kappa * np.median(sc[r])
    out = np.zeros(n, bool)
    for i in np.where(cand)[0]:
        r = ring(i); r = r[~cand[r]]
        out[i] = r.size == 0 or sc[i] > kappa * np.max(sc[r])
    return out


def detect_mixed(A, M=4, kappa=20.0, floor=1e-10, return_steps=False, two_stage=False):
    """1-D ENO tests along rows and columns.
    point anomaly (an edge crosses the pixel) -> mixed pixel;
    step between two pixels (edge on/near their common side) -> the pair is disconnected.
    Returns mixed (dilated by one) and, optionally, the step masks (sx: between (i,j),(i+1,j); sy: (i,j),(i,j+1))."""
    n = A.shape[0]
    W = M + 2
    fl = floor * np.max(np.abs(A))
    mixed = np.zeros_like(A, bool)
    sx = np.zeros((n - 1, n), bool); sy = np.zeros((n, n - 1), bool)
    for j in range(n):                                   # along x (axis 0) for each column j
        v = A[:, j]
        pt = _eno_point_scores(v, M)
        mixed[:, j] |= _ring_flags_max(pt, W, kappa, fl) | (_ring_flags_2s(pt, W, kappa, fl) if two_stage else False)
        ps = _eno_pair_scores(v, M)
        sx[:, j] |= _ring_flags_max(ps, W, kappa, fl) | _ring_flags_2s(ps, W, kappa, fl)
    for i in range(n):                                   # along y (axis 1)
        v = A[i, :]
        pt = _eno_point_scores(v, M)
        mixed[i, :] |= _ring_flags_max(pt, W, kappa, fl) | (_ring_flags_2s(pt, W, kappa, fl) if two_stage else False)
        ps = _eno_pair_scores(v, M)
        sy[i, :] |= _ring_flags_max(ps, W, kappa, fl) | _ring_flags_2s(ps, W, kappa, fl)
    mixed_d = ndimage.binary_dilation(mixed, structure=np.ones((3, 3), bool))
    if return_steps:
        return mixed_d, sx, sy
    return mixed_d


def edge_aware_labels(A, mixed, sx=None, sy=None):
    """Connected components of the clean pixels. Two clean 4-neighbours are connected unless the 1-D
    detector found a step between them (sx, sy) -- no global amplitude threshold, so smooth background
    gradients cannot bridge an edge. Components with < 6 pixels are returned to the mixed set."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n1, n2 = A.shape
    if sx is None:
        sx = np.zeros((n1 - 1, n2), bool); sy = np.zeros((n1, n2 - 1), bool)
    idx = np.arange(n1 * n2).reshape(n1, n2)
    clean = ~mixed
    okx = clean[:-1, :] & clean[1:, :] & ~sx
    oky = clean[:, :-1] & clean[:, 1:] & ~sy
    r = np.concatenate([idx[:-1, :][okx], idx[:, :-1][oky]])
    c = np.concatenate([idx[1:, :][okx], idx[:, 1:][oky]])
    G = coo_matrix((np.ones(r.size), (r, c)), shape=(n1 * n2, n1 * n2))
    _, comp = connected_components(G, directed=False)
    lab = comp.reshape(n1, n2) + 1
    lab[mixed] = 0
    vals, cnt = np.unique(lab[lab > 0], return_counts=True)
    for v_, c_ in zip(vals, cnt):
        if c_ < 6:
            lab[lab == v_] = 0
    return lab


# ------------------------------------------------------------------ polynomial fits
@lru_cache(maxsize=None)
def _avg_basis_c(di, dj, P):
    def m1(d, a):
        return ((d + 0.5) ** (a + 1) - (d - 0.5) ** (a + 1)) / (a + 1)
    out = np.array([m1(di, a) * m1(dj, b) for a, b in monos(P)])
    out.setflags(write=False)
    return out


def _avg_basis(di, dj, P):
    """Average over the unit pixel centred at (di, dj) of xi^a zeta^b, for all monomials
    (integer offsets; memoised, since the same few offsets recur for every pixel)."""
    return _avg_basis_c(int(di), int(dj), int(P))


class LocalFits:
    """Least-squares fits of pixel averages by polynomials of total degree <= P (in pixel units),
    using only clean pixels of one region in a window around (i, j).
    Own pixel in the region -> degree P with the own average enforced (interpolation).
    Otherwise (extrapolation into a mixed pixel) -> degree chosen automatically: among p = 0..P the one
    whose predicted average at (i, j) changes least from degree p-1 (stability-based order selection,
    the 2-D analogue of the 1-D 'reconstruction order matched to data accuracy')."""

    def __init__(self, A, labels, P=4, R=3, Rmax=7):
        self.A, self.lab, self.P, self.R, self.Rmax = A, labels, P, R, Rmax
        self.cache = {}
        self.deg = {}

    def _lsq(self, G, y, w, p):
        nb = len(monos(p))
        cols = [k for k, (a, b) in enumerate(monos(self.P)) if a + b <= p]
        c = np.zeros(len(monos(self.P)))
        c[cols] = np.linalg.lstsq(G[:, cols] * w[:, None], y * w, rcond=None)[0]
        return c

    def fit(self, i, j, region):
        key = (i, j, region)
        if key in self.cache:
            return self.cache[key]
        n = self.A.shape[0]
        nbP = len(monos(self.P))
        for R in range(self.R, self.Rmax + 1):
            ii, jj = np.meshgrid(np.arange(max(0, i - R), min(n, i + R + 1)),
                                 np.arange(max(0, j - R), min(n, j + R + 1)), indexing='ij')
            sel = self.lab[ii, jj] == region
            if sel.sum() >= 2 * nbP:
                break
        di, dj = ii[sel] - i, jj[sel] - j
        if di.size == 0:
            c = np.zeros(nbP); self.cache[key] = c; return c
        G = np.array([_avg_basis(a, b, self.P) for a, b in zip(di, dj)])
        y = self.A[ii[sel], jj[sel]]
        w = 1.0 / (1 + 0.15 * (di ** 2 + dj ** 2))
        own = (di == 0) & (dj == 0)
        g0 = _avg_basis(0, 0, self.P)
        # robust trimming: undetected partially-covered pixels carry an O(jump) misfit -> drop them
        p_tr = min(self.P, 2)
        keep = np.ones(y.size, bool)
        for _ in range(3):
            if keep.sum() < len(monos(p_tr)) + 3:
                break
            ct = self._lsq(G[keep], y[keep], w[keep], p_tr)
            r = np.abs(G @ ct - y)
            med = np.median(r[keep]) + 1e-14 * (np.abs(y).max() + 1)
            new = r <= 10 * med + 1e-12
            new[own] = True
            if (new == keep).all():
                break
            keep = new
        G, y, w, own, di, dj = G[keep], y[keep], w[keep], own[keep], di[keep], dj[keep]
        if own.any():
            w = w.copy(); w[own] = 1e3
            c = self._lsq(G, y, w, self.P); self.deg[key] = self.P
        else:
            pmax = self.P
            while pmax > 0 and di.size < 1.5 * len(monos(pmax)):
                pmax -= 1
            cs = [self._lsq(G, y, w, p) for p in range(pmax + 1)]
            preds = np.array([g0 @ c_ for c_ in cs])
            if pmax == 0:
                k = 0
            else:
                ch = np.abs(np.diff(preds))                  # change from p-1 to p, p = 1..pmax
                k = int(np.argmin(ch)) + 1
            c = cs[k]; self.deg[key] = k
        self.cache[key] = c
        return c


def peval(c, P, x, y):
    out = 0.0
    for k, (a, b) in enumerate(monos(P)):
        out = out + c[k] * x ** a * y ** b
    return out


# ------------------------------------------------------------------ geometry
_TG, _TW = npleg.leggauss(10)
_TG = 0.5 * (_TG + 1); _TW = 0.5 * _TW


def tri_quad(Pg, fun):
    tot = 0.0
    if Pg is None or len(Pg) < 3:
        return 0.0
    A0 = Pg[0]
    s, t = np.meshgrid(_TG, _TG, indexing='ij'); ws = np.outer(_TW, _TW)
    for k in range(1, len(Pg) - 1):
        B, C = Pg[k], Pg[k + 1]
        det = (B[0] - A0[0]) * (C[1] - A0[1]) - (B[1] - A0[1]) * (C[0] - A0[0])
        x = A0[0] + s * (B[0] - A0[0]) + s * t * (C[0] - B[0])
        y = A0[1] + s * (B[1] - A0[1]) + s * t * (C[1] - B[1])
        tot = tot + np.sum(ws * s * fun(x, y)) * det
    return tot


def clip_halfplane(Pg, nx, ny, d, keep_less=True):
    """Clip polygon by n.x <= d (keep_less) or n.x >= d."""
    sgn = 1.0 if keep_less else -1.0
    out = []
    m = len(Pg)
    for k in range(m):
        cur, prv = Pg[k], Pg[k - 1]
        fc = sgn * (nx * cur[0] + ny * cur[1] - d)
        fp = sgn * (nx * prv[0] + ny * prv[1] - d)
        if fc <= 0:
            if fp > 0:
                t = fp / (fp - fc); out.append(prv + t * (cur - prv))
            out.append(cur)
        elif fp <= 0:
            t = fp / (fp - fc); out.append(prv + t * (cur - prv))
    return np.array(out) if len(out) >= 3 else None



def _clip_poly_to_rect(Pg, x0, x1, y0, y1):
    """Sutherland-Hodgman clip of a simple polygon (any) by the rectangle; None if empty."""
    out = np.asarray(Pg, float)
    for nx, ny, d in ((-1, 0, -x0), (1, 0, x1), (0, -1, -y0), (0, 1, y1)):
        if out is None or len(out) < 3:
            return None
        out = clip_halfplane(out, nx, ny, d, True)
    if out is None or len(out) < 3:
        return None
    a = 0.5 * np.sum(out[:, 0] * np.roll(out[:, 1], -1) - np.roll(out[:, 0], -1) * out[:, 1])
    if abs(a) < 1e-30:
        return None
    return out if a > 0 else out[::-1]


def _dp_closed(Pg, tol):
    """Douglas-Peucker simplification of a closed polygon."""
    def dp(pts):
        if len(pts) < 3:
            return pts
        a, b = pts[0], pts[-1]
        ab = b - a; L = np.linalg.norm(ab) + 1e-300
        d = np.abs(ab[0] * (pts[:, 1] - a[1]) - ab[1] * (pts[:, 0] - a[0])) / L
        k = int(np.argmax(d))
        if d[k] > tol:
            return np.vstack([dp(pts[:k + 1])[:-1], dp(pts[k:])])
        return np.vstack([a, b])
    k = int(np.argmax(np.linalg.norm(Pg - Pg.mean(0), axis=1)))
    P2 = np.vstack([np.roll(Pg, -k, axis=0), Pg[k]])
    out = dp(P2)[:-1]
    return out if len(out) >= 3 else Pg


def _point_in_poly(Pg, p):
    x, y = p; inside = False
    m = len(Pg)
    for k in range(m):
        x1, y1 = Pg[k]; x2, y2 = Pg[(k + 1) % m]
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


_EG, _EW = npleg.leggauss(6)
_EG = 0.5 * (_EG + 1); _EW = 0.5 * _EW


_MONO_CACHE = {}


def poly_moments(Pg, P):
    """Exact int_Pg xi^a zeta^b dA for all monomials (a+b <= P): oint xi^{a+1} zeta^b/(a+1) d zeta."""
    if P not in _MONO_CACHE:
        mon = np.array(monos(P))
        _MONO_CACHE[P] = (mon[:, 0], mon[:, 1])
    A_, B_ = _MONO_CACHE[P]
    if Pg is None or len(Pg) < 3:
        return np.zeros(A_.size)
    D = np.empty_like(Pg); D[:-1] = Pg[1:] - Pg[:-1]; D[-1] = Pg[0] - Pg[-1]   # (E, 2)
    x = Pg[:, 0, None] + _EG[None, :] * D[:, 0, None]              # (E, q)
    y = Pg[:, 1, None] + _EG[None, :] * D[:, 1, None]
    xp = np.empty((P + 2,) + x.shape); xp[0] = 1.0
    yp = np.empty((P + 1,) + y.shape); yp[0] = 1.0
    for k in range(1, P + 2):
        xp[k] = xp[k - 1] * x
    for k in range(1, P + 1):
        yp[k] = yp[k - 1] * y
    T = xp[A_ + 1] * yp[B_]                                          # (nb, E, q)
    return ((T @ _EW) * D[None, :, 1]).sum(1) / (A_ + 1)


def square(di=0.0, dj=0.0):
    return np.array([[di - .5, dj - .5], [di + .5, dj - .5], [di + .5, dj + .5], [di - .5, dj + .5]])


# ------------------------------------------------------------------ oscillatory integrals
def _mono_leg(P):
    """Legendre coefficients of s^a on [-1,1], a = 0..P (rows)."""
    return [npleg.poly2leg(np.eye(P + 1)[a]) for a in range(P + 1)]


def mu1d(alpha, P):
    """mu_a(alpha) = int_{-1/2}^{1/2} xi^a exp(alpha xi) d xi, alpha = -2 pi i u h (imaginary). (n_u, P+1)."""
    om = 1j * np.asarray(alpha) / 2                      # exp(alpha xi) = exp(-i om s), xi = s/2
    om = np.real(om)
    L = _mono_leg(P)
    J = np.stack([2 * (-1j) ** m * spherical_jn(m, np.abs(om)) * (np.sign(om) ** m if m % 2 else 1.0)
                  for m in range(P + 1)], 1)
    out = np.zeros((om.size, P + 1), complex)
    for a in range(P + 1):
        la = L[a]
        out[:, a] = (0.5) ** (a + 1) * (J[:, :la.size] @ la)
    return out


def _line_poly_exp(r, c):
    """int_0^1 r(t) exp(c t) dt, r = monomial coefficients (low->high), c imaginary array."""
    deg = len(r) - 1
    s, _ = npleg.leggauss(deg + 1)
    vals = np.polynomial.polynomial.polyval((s + 1) / 2, r)
    lc = npleg.legfit(s, vals, deg)
    om = np.real(1j * np.asarray(c) / 2)
    J = np.stack([2 * (-1j) ** m * spherical_jn(m, np.abs(om)) * (np.sign(om) ** m if m % 2 else 1.0)
                  for m in range(deg + 1)], 1)
    return 0.5 * np.exp(np.asarray(c) / 2) * (J @ lc)


def poly_polygon_ft(Pg, coef, P, al, be):
    """int_Pg p(xi, zeta) exp(al xi + be zeta) d xi d zeta (local scaled coords), vectorised over (al, be).
    Green's theorem with repeated integration by parts; Gauss rule when |al|,|be| < 1."""
    al = np.asarray(al, complex); be = np.asarray(be, complex)
    out = np.zeros(al.size, complex)
    if Pg is None:
        return out
    mon = monos(P)
    C = np.zeros((P + 1, P + 1))
    for k, (a, b) in enumerate(mon):
        C[a, b] = coef[k]
    small = np.maximum(np.abs(al), np.abs(be)) < 1.0
    for k in np.where(small)[0]:
        out[k] = tri_quad(Pg, lambda x, y: np.polynomial.polynomial.polyval2d(x, y, C) * np.exp(al[k] * x + be[k] * y))
    big = ~small
    if not big.any():
        return out
    a_, b_ = al[big], be[big]
    usex = np.abs(a_) >= np.abs(b_)
    Pn = np.roll(Pg, -1, axis=0)
    res = np.zeros(a_.size, complex)
    for e in range(len(Pg)):
        p0, d = Pg[e], Pn[e] - Pg[e]
        c = a_ * d[0] + b_ * d[1]
        e0 = np.exp(a_ * p0[0] + b_ * p0[1])
        # Q_x = e * sum_k (-1)^k d_x^k p / a^{k+1};   oint Q_x d zeta
        # Q_y = e * sum_k (-1)^k d_y^k p / b^{k+1};  -oint Q_y d xi
        acc = np.zeros(a_.size, complex)
        Dk = C.copy()
        for kk in range(P + 1):
            # restrict Dk(x, y) to the edge: polynomial in t
            r = _restrict(Dk, p0, d)
            if np.any(r):
                I = _line_poly_exp(r, c)
                acc_x = ((-1) ** kk) * I / np.where(usex, a_, 1) ** (kk + 1) * d[1]
                acc = acc + np.where(usex, acc_x, 0)
            Dk = np.polynomial.polynomial.polyder(Dk, axis=0) if Dk.shape[0] > 1 else Dk * 0
        Dk = C.copy()
        for kk in range(P + 1):
            r = _restrict(Dk, p0, d)
            if np.any(r):
                I = _line_poly_exp(r, c)
                acc_y = -((-1) ** kk) * I / np.where(usex, 1, b_) ** (kk + 1) * d[0]
                acc = acc + np.where(usex, 0, acc_y)
            Dk = np.polynomial.polynomial.polyder(Dk, axis=1) if Dk.shape[1] > 1 else Dk * 0
        res = res + e0 * acc
    out[big] = res
    return out


def _restrict(C2, p0, d):
    """Coefficients in t of p(p0 + t d) for 2-D monomial coefficient matrix C2[a, b]."""
    from numpy.polynomial import polynomial as Pl
    px = np.array([p0[0], d[0]]); py = np.array([p0[1], d[1]])
    out = np.zeros(1)
    for a in range(C2.shape[0]):
        for b in range(C2.shape[1]):
            if C2[a, b] != 0:
                out = Pl.polyadd(out, C2[a, b] * Pl.polymul(Pl.polypow(px, a), Pl.polypow(py, b)))
    return out


# ------------------------------------------------------------------ SAFI-2D
class SAFI2D:
    def __init__(self, A, P=4, M_det=3, kappa=5.0, edges=True, known_polys=None, repair=True,
                 tol_ang=2e-3, tol_off=2e-3, two_stage=False, det_floor=1e-10):
        self.tol_ang, self.tol_off = tol_ang, tol_off
        self.A = np.asarray(A, float)
        n = self.A.shape[0]
        self.n, self.h, self.P = n, 1.0 / n, P
        if edges:
            self.mixed, sx, sy = detect_mixed(self.A, M=M_det, kappa=kappa, floor=det_floor, return_steps=True,
                                              two_stage=two_stage)
        else:
            self.mixed = np.zeros_like(self.A, bool); sx = sy = None
        # region labels: components of the clean set after two extra dilations (robust to missed
        # tiny-fraction pixels), then every clean pixel takes the label of the nearest component
        lab = edge_aware_labels(self.A, self.mixed, sx, sy)
        self.lab = lab                                 # 0 = mixed
        self.fits = LocalFits(self.A, lab, P=P)
        self.cuts = {}                                 # (i,j) -> (nx, ny, d, regA, regB)
        self.pure = {}                                 # (i,j) -> mass-corrected own coefficients
        self.known_polys = known_polys
        for i, j in zip(*np.where(self.mixed)):
            self._fit_pixel(i, j)
        # geometric closure: pixels crossed by a fitted line but not flagged are cut as well
        for _ in range(2):
            add = set()
            for (i, j), (nx, ny, d, rA, rB) in list(self.cuts.items()):
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        a, b = i + di, j + dj
                        if (di or dj) and 0 <= a < n and 0 <= b < n and not self.mixed[a, b]:
                            # line in neighbour's local coords: n.(xi + (di,dj)) = d
                            dd = d - (nx * di + ny * dj)
                            if abs(dd) < 0.5 * (abs(nx) + abs(ny)) - 1e-9 and (a, b) not in self.cuts:
                                add.add((a, b))
            if not add:
                break
            for a, b in add:
                self.mixed[a, b] = True
            for a, b in add:
                self._fit_pixel(a, b)
        self.corners = {}
        if edges and repair:
            self._repair()

    # -------------------------------------------------------------- per mixed pixel
    def _regions_near(self, i, j, R=3):
        n = self.n
        win = self.lab[max(0, i - R):i + R + 1, max(0, j - R):j + R + 1]
        vals, cnt = np.unique(win[win > 0], return_counts=True)
        order = np.argsort(-cnt)
        return [int(v) for v in vals[order]]

    def _model_avg(self, di, dj, nx, ny, d, cA, cB):
        sq = square(di, dj)
        PA = clip_halfplane(sq, nx, ny, d, True)
        P = self.P
        mA = poly_moments(PA, P)
        return mA @ cA + (_avg_basis(di, dj, P) - mA) @ cB         # moments of the rest of the unit square

    def _fit_pixel(self, i, j):
        """Choose the region pair and the line n.xi = d for mixed pixel (i, j).
        All pairs among the (up to 3) nearby regions are tried; a pair is admissible only if the
        conservation equation (own average) has a root; the admissible pair with the smallest 3x3
        misfit wins. If none is admissible the pixel is treated as pure (see self.pure)."""
        regs = self._regions_near(i, j)[:3]
        n, P = self.n, self.P
        target = self.A[i, j]
        if not regs:
            return
        if len(regs) == 1:
            self._set_pure(i, j, regs)
            return
        nbr = [(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1) if 0 <= i + di < n and 0 <= j + dj < n]
        best = None
        for a in range(len(regs)):
            for b in range(a + 1, len(regs)):
                rA, rB = regs[a], regs[b]
                cA, cB = self.fits.fit(i, j, rA), self.fits.fit(i, j, rB)
                gA0 = _avg_basis(0, 0, P) @ cA; gB0 = _avg_basis(0, 0, P) @ cB
                if abs(gA0 - gB0) < 1e-14:
                    continue
                phi = (target - gB0) / (gA0 - gB0)            # area fraction of A implied by conservation
                if not (0.02 < phi < 0.98):
                    continue                                  # (near-)pure for this pair or inconsistent
                sol = self._fit_line(i, j, nbr, cA, cB)
                if sol is None:
                    continue
                th, d, Jv = sol
                if best is None or Jv < best[0]:
                    best = (Jv, th, d, rA, rB)
        if best is None:
            self._set_pure(i, j, regs)
            return
        Jv, th, d, rA, rB = best
        nx, ny = np.cos(th), np.sin(th)
        if self.known_polys is not None:
            nx, ny, d = self._known_line(i, j, nx, ny, d)
        self.cuts[(i, j)] = (nx, ny, d, rA, rB)

    def _set_pure(self, i, j, regs):
        """Pure pixel: region whose fit reproduces the pixel average best; a constant shift then makes
        the pixel's own average exact (mass conservation)."""
        P = self.P
        errs = [abs(_avg_basis(0, 0, P) @ self.fits.fit(i, j, r) - self.A[i, j]) for r in regs]
        r = regs[int(np.argmin(errs))]
        c = self.fits.fit(i, j, r).copy()
        c[0] += self.A[i, j] - _avg_basis(0, 0, P) @ c
        self.pure[(i, j)] = c

    def _fit_line(self, i, j, nbr, cA, cB):
        P = self.P
        target = self.A[i, j]
        gA = np.array([_avg_basis(di, dj, P) @ cA for di, dj in nbr])
        gB = np.array([_avg_basis(di, dj, P) @ cB for di, dj in nbr])
        data = np.array([self.A[i + di, j + dj] for di, dj in nbr])
        den = np.where(np.abs(gA - gB) > 1e-14, gA - gB, 1e-14)
        phi = np.clip((data - gB) / den, 0, 1)
        gx = sum(p * di for p, (di, dj) in zip(phi, nbr)); gy = sum(p * dj for p, (di, dj) in zip(phi, nbr))
        th0 = np.arctan2(-gy, -gx) if (gx or gy) else 0.0     # normal points from A to B

        def d_of(th):
            nx, ny = np.cos(th), np.sin(th)
            r = 0.5 * (abs(nx) + abs(ny))
            g = lambda dd: self._model_avg(0, 0, nx, ny, dd, cA, cB) - target
            ga, gb = g(-r), g(r)
            if ga * gb < 0:
                return brentq(g, -r, r, xtol=1e-15)
            return None

        def J(th):
            d = d_of(th)
            if d is None:
                return np.inf
            nx, ny = np.cos(th), np.sin(th)
            res = [self._model_avg(di, dj, nx, ny, d, cA, cB) - self.A[i + di, j + dj]
                   for di, dj in nbr if (di, dj) != (0, 0)]
            return float(np.sum(np.square(res)))

        ths = th0 + np.linspace(-np.pi, np.pi, 36, endpoint=False)
        Js = np.array([J(t) for t in ths])
        if not np.isfinite(Js).any():
            return None
        k = int(np.argmin(Js))
        from scipy.optimize import minimize_scalar
        step = 2 * np.pi / 36
        sol = minimize_scalar(J, bounds=(ths[k] - step, ths[k] + step), method='bounded', options=dict(xatol=1e-10))
        th = sol.x if (np.isfinite(sol.fun) and sol.fun <= Js[k]) else ths[k]
        d = d_of(th)
        if d is None:
            th, d = ths[k], d_of(ths[k])
        return th, d, J(th)

    # -------------------------------------------------------------- straight-edge families and corners
    def _gline(self, i, j):
        """Global line of a cut pixel: unit normal n (A side: n.x <= D) and D."""
        nx, ny, d, rA, rB = self.cuts[(i, j)]
        h = self.h
        xc = np.array([(i + 0.5) * h, (j + 0.5) * h])
        nrm = np.array([nx, ny])
        return nrm, float(nrm @ xc + d * h), rA, rB

    def _same_line(self, L1, L2):
        n1, D1, a1, b1 = L1; n2, D2, a2, b2 = L2
        if {a1, b1} != {a2, b2}:
            return False
        if a1 != a2:                                    # opposite orientation convention
            n2, D2 = -n2, -D2
        ang = np.arccos(np.clip(n1 @ n2, -1, 1))
        return ang < self.tol_ang and abs(D1 - D2) < self.tol_off * self.h

    def _reliable(self):
        rel = {}
        keys = list(self.cuts)
        lines = {k: self._gline(*k) for k in keys}
        for (i, j) in keys:
            Lp = lines[(i, j)]
            # collinearity with cut pixels in a 5x5 window, judged at the pixel centre:
            cnt = 0
            for (a, b) in keys:
                if (a, b) == (i, j) or max(abs(a - i), abs(b - j)) > 2:
                    continue
                if self._same_line(Lp, lines[(a, b)]):
                    cnt += 1
            if cnt >= 2:
                rel[(i, j)] = Lp
        return rel

    def _families(self, rel, i, j, R=3):
        """Cluster reliable lines near (i, j) into collinear families; return fused lines."""
        cand = [(k, L) for k, L in rel.items() if max(abs(k[0] - i), abs(k[1] - j)) <= R and k != (i, j)]
        fams = []
        for k, L in cand:
            for F in fams:
                if self._same_line(F[0], L):
                    F[1].append(L); break
            else:
                fams.append([L, [L]])
        out = []
        for ref, mem in fams:
            n0, D0, a0, b0 = ref
            ns, Ds = [], []
            for n_, D_, a_, b_ in mem:
                if a_ != a0:
                    n_, D_ = -n_, -D_
                ns.append(n_); Ds.append(D_)
            nb = np.mean(ns, 0); nb /= np.linalg.norm(nb)
            Db = float(np.mean([D_ + (nb - n_) @ np.zeros(2) for n_, D_ in zip(ns, Ds)]))
            out.append((nb, Db, a0, b0, len(mem)))
        return out

    def _local_line(self, i, j, nrm, D):
        h = self.h
        xc = np.array([(i + 0.5) * h, (j + 0.5) * h])
        return nrm[0], nrm[1], (D - nrm @ xc) / h

    def _edge_families(self, tol_ang=0.05, tol_off=0.08):
        """Greedy clustering of cut pixels into straight-edge families (collinear + 8-connected)."""
        keys = sorted(self.cuts)
        lines = {k: self._gline(*k) for k in keys}
        unassigned = set(keys)
        fams = []
        def close(L1, L2):
            n1, D1, a1, b1 = L1; n2, D2, a2, b2 = L2
            if {a1, b1} != {a2, b2}:
                return False
            if a1 != a2:
                n2, D2 = -n2, -D2
            return np.arccos(np.clip(n1 @ n2, -1, 1)) < tol_ang and abs(D1 - D2) < tol_off * self.h
        while unassigned:
            seed = min(unassigned, key=lambda k: self._pix_resid.get(k, 0.0))
            fam = {seed}; unassigned.discard(seed)
            frontier = [seed]
            while frontier:
                p = frontier.pop()
                for q in list(unassigned):
                    if max(abs(q[0] - p[0]), abs(q[1] - p[1])) <= 1 and close(lines[seed], lines[q]):
                        fam.add(q); unassigned.discard(q); frontier.append(q)
            fams.append(sorted(fam))
        return fams, lines

    def _fit_family(self, members, L0):
        """Two-parameter (theta, D) least squares on the members' own averages (conservation residuals),
        with robust trimming of members contaminated by corners."""
        n0, D0, rA, rB = L0
        th0 = np.arctan2(n0[1], n0[0])
        h, P = self.h, self.P
        mem = list(members)
        coef = {q: (self.fits.fit(q[0], q[1], rA), self.fits.fit(q[0], q[1], rB)) for q in mem}
        def resid(p, use):
            th, D = p
            nrm = np.array([np.cos(th), np.sin(th)])
            out = []
            for q in use:
                xc = np.array([(q[0] + 0.5) * h, (q[1] + 0.5) * h])
                d = (D - nrm @ xc) / h
                cA, cB = coef[q]
                out.append(self._model_avg(0, 0, nrm[0], nrm[1], d, cA, cB) - self.A[q])
            return np.array(out)
        use = mem
        p = np.array([th0, D0])
        for _ in range(3):
            if len(use) < 3:
                break
            sol = least_squares(resid, p, args=(use,), x_scale=[1e-2, 1e-2 * h], xtol=1e-15, ftol=1e-15, gtol=1e-15)
            p = sol.x
            r = np.abs(resid(p, mem))
            med = np.median(r) + 1e-15
            use_new = [q for q, rr in zip(mem, r) if rr <= 5 * med]
            if len(use_new) == len(use):
                break
            use = use_new
        nrm = np.array([np.cos(p[0]), np.sin(p[0])])
        return nrm, float(p[1]), rA, rB, use

    def _repair(self):
        self._pix_resid = {}
        fams, lines = self._edge_families()
        fitted = []
        for fam in fams:
            if len(fam) < 3:
                continue
            nrm, D, rA, rB, use = self._fit_family(fam, lines[fam[0]])
            fitted.append(dict(n=nrm, D=D, rA=rA, rB=rB, mem=set(fam), m=len(use)))
        self.families = fitted
        if not fitted:
            return
        h = self.h
        # merge collinear families (an edge split into pieces)
        merged = True
        while merged:
            merged = False
            for a in range(len(fitted)):
                for b in range(a + 1, len(fitted)):
                    Fa, Fb = fitted[a], fitted[b]
                    if {Fa['rA'], Fa['rB']} != {Fb['rA'], Fb['rB']}:
                        continue
                    nb, Db = (Fb['n'], Fb['D']) if Fa['rA'] == Fb['rA'] else (-Fb['n'], -Fb['D'])
                    if np.arccos(np.clip(Fa['n'] @ nb, -1, 1)) < 0.01 and abs(Fa['D'] - Db) < 0.02 * h:
                        mem = Fa['mem'] | Fb['mem']
                        nrm, D, rA, rB, use = self._fit_family(sorted(mem), (Fa['n'], Fa['D'], Fa['rA'], Fa['rB']))
                        fitted[a] = dict(n=nrm, D=D, rA=rA, rB=rB, mem=mem, m=len(use))
                        fitted.pop(b); merged = True
                        break
                if merged:
                    break
        self.families = fitted
        # segment extents: member projections, then nearest vertex on each side
        for F in fitted:
            t = np.array([-F['n'][1], F['n'][0]])
            pts = np.array([[(a + 0.5) * h, (b + 0.5) * h] for a, b in F['mem']])
            proj = pts @ t
            F['t'] = t; F['pmin'], F['pmax'] = proj.min(), proj.max()
            F['lo'] = F['pmin'] - 0.75 * h; F['hi'] = F['pmax'] + 0.75 * h
            F['p0'] = F['n'] * F['D']
            F['vlo'] = F['vhi'] = None
        for a in range(len(fitted)):
            for b in range(a + 1, len(fitted)):
                Fa, Fb = fitted[a], fitted[b]
                if {Fa['rA'], Fa['rB']} != {Fb['rA'], Fb['rB']}:
                    continue
                M2 = np.array([Fa['n'], Fb['n']])
                if abs(np.linalg.det(M2)) < 0.05:
                    continue
                V = np.linalg.solve(M2, [Fa['D'], Fb['D']])
                ok = True
                for F in (Fa, Fb):
                    tv = V @ F['t']
                    gap = max(F['pmin'] - tv, tv - F['pmax'], 0.0)
                    if gap > 8 * h or F["pmin"] + 2 * h < tv < F["pmax"] - 2 * h:
                        ok = False
                if not ok:
                    continue
                for F in (Fa, Fb):
                    tv = V @ F['t']
                    if tv < 0.5 * (F['pmin'] + F['pmax']):
                        if F['vlo'] is None or tv > F['vlo']:
                            F['vlo'] = tv
                    else:
                        if F['vhi'] is None or tv < F['vhi']:
                            F['vhi'] = tv
        for F in fitted:
            if F['vlo'] is not None:
                F['lo'] = F['vlo']
            if F['vhi'] is not None:
                F['hi'] = F['vhi']
        cand = set(tuple(map(int, p)) for p in np.argwhere(self.mixed)) | set(self.cuts) | set(self.pure)
        for (i, j) in sorted(cand):
            sq = np.array([[i * h, j * h], [(i + 1) * h, j * h], [(i + 1) * h, (j + 1) * h], [i * h, (j + 1) * h]])
            crossing = []
            for F in fitted:
                if self._segment_hits(sq, F):
                    nx, ny, d = self._local_line(i, j, F['n'], F['D'])
                    crossing.append((nx, ny, d, F['rA'], F['rB'], F['m']))
            if not crossing:
                if (i, j) in self.cuts:                     # a cut not supported by any edge segment
                    nx, ny, d, rA, rB = self.cuts.pop((i, j))
                    self._set_pure(i, j, [rA, rB])
                continue
            crossing.sort(key=lambda c: -c[5])
            if len(crossing) == 1 or self._parallel(crossing[0], crossing[1]):
                nx, ny, d, rA, rB, _ = crossing[0]
                if abs(d) < 0.5 * (abs(nx) + abs(ny)):
                    self.cuts[(i, j)] = (nx, ny, d, rA, rB)
                    self.pure.pop((i, j), None); self.corners.pop((i, j), None)
            else:
                self._set_corner(i, j, crossing[0], crossing[1])

    @staticmethod
    def _segment_hits(sq, F):
        """Does the segment {p0 + s t : lo <= s <= hi} of family F meet the (closed) square sq?"""
        p0, t, n_ = F['p0'], F['t'], F['n']
        # the line must cross the square: corner signed distances change sign
        sd = sq @ n_ - F['D']
        if sd.max() <= 0 or sd.min() >= 0:
            return False
        # projection range of the square onto t must overlap [lo, hi]
        pr = sq @ t
        return pr.max() >= F['lo'] and pr.min() <= F['hi']

    def _parallel(self, c1, c2):
        return abs(c1[0] * c2[1] - c1[1] * c2[0]) < 0.05

    def _set_corner(self, i, j, l1, l2):
        """Inside region = A-side of both lines (convex corner) or of either (reflex corner):
        the variant reproducing the pixel average is kept. Region roles must agree."""
        nx1, ny1, d1, a1, b1, _ = l1
        nx2, ny2, d2, a2, b2, _ = l2
        if {a1, b1} != {a2, b2}:
            return
        if a2 != a1:                                    # align: A side = same region for both
            nx2, ny2, d2 = -nx2, -ny2, -d2
        rA, rB = a1, b1
        P = self.P
        cA, cB = self.fits.fit(i, j, rA), self.fits.fit(i, j, rB)
        sq = square()
        best = None
        for mode in ('and', 'or'):
            pa, pb = self._corner_polys(sq, (nx1, ny1, d1), (nx2, ny2, d2), mode)
            avg = sum(poly_moments(q, P) @ cA for q in pa) + sum(poly_moments(q, P) @ cB for q in pb)
            err = abs(avg - self.A[i, j])
            if best is None or err < best[0]:
                best = (err, mode)
        self.corners[(i, j)] = ((nx1, ny1, d1), (nx2, ny2, d2), best[1], rA, rB)
        self.cuts.pop((i, j), None); self.pure.pop((i, j), None)

    @staticmethod
    def _corner_polys(sq, L1, L2, mode):
        """Disjoint convex pieces of the pixel for region A ('and': H1 & H2, 'or': H1 | H2) and B."""
        H1 = clip_halfplane(sq, *L1, True)
        H1c = clip_halfplane(sq, *L1, False)
        if mode == 'and':
            pa = [clip_halfplane(H1, *L2, True)] if H1 is not None else []
            pb = [H1c] + ([clip_halfplane(H1, *L2, False)] if H1 is not None else [])
        else:
            pb = [clip_halfplane(H1c, *L2, False)] if H1c is not None else []
            pa = [H1] + ([clip_halfplane(H1c, *L2, True)] if H1c is not None else [])
        return [q for q in pa if q is not None], [q for q in pb if q is not None]

    def _corner_ft(self, i, j, al, be):
        L1, L2, mode, rA, rB = self.corners[(i, j)]
        cA, cB = self.fits.fit(i, j, rA), self.fits.fit(i, j, rB)
        pa, pb = self._corner_polys(square(), L1, L2, mode)
        out = 0
        for q in pa:
            out = out + poly_polygon_ft(q, cA, self.P, al, be)
        for q in pb:
            out = out + poly_polygon_ft(q, cB, self.P, al, be)
        return out

    # -------------------------------------------------------------- global conservative polygon fit
    def polygon_fit(self, max_insert=2, verbose=False):
        """Variational reconstruction: vertices of a star-shaped polygon separating region rin from rout,
        fitted by least squares on the conservation residuals of all pixels within one pixel of the
        boundary. Two initialisations are tried -- straight-edge families (+ their intersections) and the
        convex hull of the interior-phase pixels -- and the fit with the smaller residual on the union of
        both boundary bands is kept (guards against a bad local minimum)."""
        fams = getattr(self, 'families', [])
        h, n, P = self.h, self.n, self.P
        inits = []
        if len(fams) >= 3:
            pair = {fams[0]['rA'], fams[0]['rB']}
            ff = [F for F in fams if {F['rA'], F['rB']} == pair]
            if len(ff) >= 3:
                V = []
                for F in ff:
                    for tv in (F['lo'], F['hi']):
                        V.append(F['p0'] + tv * F['t'])
                inits.append((np.array(V), pair, max_insert))
        if not inits:                                      # fallback only (coarse grids / no families)
            Vh, pair_h = self._hull_init()
            if Vh is not None:
                inits.append((Vh, pair_h, max_insert))
        sols = []
        for V, pair, mi in inits:
            out = self._fit_from_init(V, pair, mi, verbose)
            if out is not None:
                sols.append(out)
        if not sols:
            return False
        if len(sols) > 1:
            union = sorted(set().union(*[set(o['band']) for o in sols]))
            scores = [np.sqrt(np.mean(o['resid'](o['V'].ravel(), union) ** 2)) for o in sols]
            best = sols[int(np.argmin(scores))]
            if verbose:
                print('  init scores', ['%.2e' % s_ for s_ in scores])
        else:
            best = sols[0]
        self.poly = dict(V=best['V'], rin=best['rin'], rout=best['rout'], band=best['band'], resid_max=best['rmax'])
        return True

    def _fit_from_init(self, V, pair, max_insert, verbose=False):
        h, n, P = self.h, self.n, self.P
        # merge close points
        keep = []
        for p in V:
            for k in keep:
                if np.linalg.norm(p - k[0]) < 2.5 * h:
                    k[1].append(p); break
            else:
                keep.append([p, [p]])
        V = np.array([np.mean(k[1], 0) for k in keep])
        ctr = V.mean(0)
        V = V[np.argsort(np.arctan2(V[:, 1] - ctr[1], V[:, 0] - ctr[0]))]
        # inside region: label of the clean pixel nearest to the centroid
        ci, cj = int(np.clip(ctr[0] / h, 0, n - 1)), int(np.clip(ctr[1] / h, 0, n - 1))
        lab = self.lab
        cand = np.argwhere(lab > 0)
        k = np.argmin(np.sum((cand - [ci, cj]) ** 2, 1))
        rin = int(lab[tuple(cand[k])])
        rout = [r for r in pair if r != rin]
        if len(rout) != 1:
            return None
        rout = rout[0]

        def band_pixels(Vx):
            pts = set()
            m = len(Vx)
            for a in range(m):
                p, q = Vx[a], Vx[(a + 1) % m]
                L = np.linalg.norm(q - p)
                for s_ in np.linspace(0, 1, max(2, int(4 * L / h) + 2)):
                    x = p + s_ * (q - p)
                    i, j = int(x[0] / h), int(x[1] / h)
                    for di in (-1, 0, 1):
                        for dj in (-1, 0, 1):
                            if 0 <= i + di < n and 0 <= j + dj < n:
                                pts.add((i + di, j + dj))
            return sorted(pts)

        band = band_pixels(V)
        coef = {}
        def cpair(i, j):
            if (i, j) not in coef:
                coef[(i, j)] = (self.fits.fit(i, j, rout), self.fits.fit(i, j, rin))
            return coef[(i, j)]

        def resid(x, pix):
            Vx = x.reshape(-1, 2)
            out = np.empty(len(pix))
            for k_, (i, j) in enumerate(pix):
                cO, cI = cpair(i, j)
                sq = np.array([[i * h, j * h], [(i + 1) * h, j * h], [(i + 1) * h, (j + 1) * h], [i * h, (j + 1) * h]])
                Q = _clip_poly_to_rect(Vx, i * h, (i + 1) * h, j * h, (j + 1) * h)
                avg = _avg_basis(0, 0, P) @ cO
                if Q is not None:
                    Ql = (Q - [(i + 0.5) * h, (j + 0.5) * h]) / h
                    avg = avg + poly_moments(Ql, P) @ (cI - cO)
                out[k_] = avg - self.A[i, j]
            return out

        x0 = V.ravel()
        sol = least_squares(resid, x0, args=(band,), x_scale=h, diff_step=1e-6, xtol=1e-14, ftol=1e-14, gtol=1e-14,
                            max_nfev=200 * x0.size)
        Vx = sol.x.reshape(-1, 2)
        r = np.abs(resid(sol.x, band)); rms = np.sqrt(np.mean(r ** 2))
        if verbose:
            print('  poly fit: m=%d  rms %.2e  max %.2e' % (len(Vx), rms, r.max()))
        for it in range(max_insert):
            # try a vertex on the edge nearest to the worst pixel; keep it only if rms drops by > 10 %
            i, j = band[int(np.argmax(r))]
            pc = np.array([(i + 0.5) * h, (j + 0.5) * h])
            m = len(Vx)
            dists = []
            for a in range(m):
                p, q = Vx[a], Vx[(a + 1) % m]
                t_ = np.clip((pc - p) @ (q - p) / ((q - p) @ (q - p)), 0, 1)
                dists.append(np.linalg.norm(p + t_ * (q - p) - pc))
            a = int(np.argmin(dists))
            p, q = Vx[a], Vx[(a + 1) % m]
            t_ = np.clip((pc - p) @ (q - p) / ((q - p) @ (q - p)), 0.1, 0.9)
            V2 = np.insert(Vx, a + 1, p + t_ * (q - p) + 0.5 * h * (pc - (p + t_ * (q - p))) / (dists[a] + 1e-300), axis=0)
            band2 = band_pixels(V2)
            sol2 = least_squares(resid, V2.ravel(), args=(band2,), x_scale=h, diff_step=1e-6, xtol=1e-14, ftol=1e-14,
                                 gtol=1e-14, max_nfev=200 * V2.size)
            r2 = np.abs(resid(sol2.x, band2)); rms2 = np.sqrt(np.mean(r2 ** 2))
            if verbose:
                print('  insert trial: m=%d  rms %.2e' % (len(V2), rms2))
            if rms2 > 0.9 * rms:
                break
            Vx, band, r, rms = sol2.x.reshape(-1, 2), band2, r2, rms2
        return dict(V=Vx, rin=rin, rout=rout, band=band_pixels(Vx), rmax=float(r.max()), resid=resid)

    def _own_coef(self, i, j):
        """Coefficients for a pixel outside the polygon band: own-region fit if clean, otherwise the
        region on the correct side of the fitted polygon, mass-corrected."""
        if self.lab[i, j] > 0:
            return self.fits.fit(i, j, self.lab[i, j])
        h = self.h
        inside = _point_in_poly(self.poly['V'], np.array([(i + 0.5) * h, (j + 0.5) * h]))
        c = self.fits.fit(i, j, self.poly['rin'] if inside else self.poly['rout']).copy()
        c[0] += self.A[i, j] - _avg_basis(0, 0, self.P) @ c
        return c

    def _hull_init(self):
        from scipy.spatial import ConvexHull
        h, n, P = self.h, self.n, self.P
        labs, cnt = np.unique(self.lab[self.lab > 0], return_counts=True)
        if labs.size < 2:
            return None, None
        top = labs[np.argsort(-cnt)[:2]]
        border = set(np.r_[self.lab[0, :], self.lab[-1, :], self.lab[:, 0], self.lab[:, -1]].tolist())
        inner = [r for r in top if r not in border]
        if len(inner) != 1:
            return None, None
        rin = int(inner[0]); rout = int([r for r in top if r != rin][0])
        pts = []
        for i in range(n):
            for j in range(n):
                if self.lab[i, j] == rin:
                    pts.append(((i + 0.5) * h, (j + 0.5) * h))
                elif self.lab[i, j] == 0:
                    gi = _avg_basis(0, 0, P) @ self.fits.fit(i, j, rin)
                    go = _avg_basis(0, 0, P) @ self.fits.fit(i, j, rout)
                    if abs(gi - go) > 1e-12 and (self.A[i, j] - go) / (gi - go) > 0.5:
                        pts.append(((i + 0.5) * h, (j + 0.5) * h))
        pts = np.array(pts)
        if len(pts) < 4:
            return None, None
        hull = pts[ConvexHull(pts).vertices]
        ctr = hull.mean(0)
        hull = ctr + (hull - ctr) * (1 + 0.5 * h / np.max(np.linalg.norm(hull - ctr, axis=1)))
        V = _dp_closed(hull, 0.75 * h)
        return V, {rin, rout}

    def _poly_eval(self, u, v, al, be, mx, my):
        """Contribution of the band pixels under the fitted polygon model."""
        h, P = self.h, self.P
        Vx, rin, rout = self.poly['V'], self.poly['rin'], self.poly['rout']
        mon = monos(P)
        Mu_all = np.stack([mx[:, a] * my[:, b] for a, b in mon], 1)
        F = np.zeros(u.size, complex)
        for (i, j) in self.poly['band']:
            cO, cI = self.fits.fit(i, j, rout), self.fits.fit(i, j, rin)
            ph = np.exp(-1j * TWO_PI * (u * (i + 0.5) * h + v * (j + 0.5) * h)) * h * h
            val = Mu_all @ cO
            Q = _clip_poly_to_rect(Vx, i * h, (i + 1) * h, j * h, (j + 1) * h)
            if Q is not None:
                Ql = (Q - [(i + 0.5) * h, (j + 0.5) * h]) / h
                val = val + poly_polygon_ft(Ql, cI - cO, P, al, be)
            F += ph * val
        return F

    def _known_line(self, i, j, nx, ny, d):
        """Oracle: nearest true polygon edge in local coords (for the 'known geometry' mode)."""
        h = self.h
        xc, yc = (i + 0.5) * h, (j + 0.5) * h
        best = None
        for Pg in self.known_polys:
            for k in range(len(Pg)):
                a, b = Pg[k], Pg[(k + 1) % len(Pg)]
                t = np.clip(np.dot([xc, yc] - a, b - a) / np.dot(b - a, b - a), 0, 1)
                dist = np.linalg.norm(a + t * (b - a) - [xc, yc])
                if best is None or dist < best[0]:
                    e = (b - a) / np.linalg.norm(b - a)
                    nrm = np.array([e[1], -e[0]])            # outward normal for CCW polygon
                    best = (dist, nrm, a)
        _, nrm, a = best
        dd = np.dot(nrm, (a - [xc, yc]) / h)
        # keep the fitted orientation's sense (which side is region A)
        if nrm[0] * nx + nrm[1] * ny < 0:
            nrm, dd = -nrm, -dd
        return nrm[0], nrm[1], dd

    # -------------------------------------------------------------- evaluation
    def __call__(self, u, v, chunk=256):
        u = np.atleast_1d(np.asarray(u, float)); v = np.atleast_1d(np.asarray(v, float))
        n, h, P = self.n, self.h, self.P
        al, be = -1j * TWO_PI * u * h, -1j * TWO_PI * v * h
        mx, my = mu1d(al, P), mu1d(be, P)                       # (n_u, P+1)
        mon = monos(P)
        if getattr(self, 'poly', None) is not None:
            band = set(self.poly['band'])
            rest = [(i, j) for i in range(n) for j in range(n) if (i, j) not in band]
            Cc = np.array([self._own_coef(i, j) for i, j in rest])
            rest = np.array(rest)
            xc = (rest[:, 0] + 0.5) * h; yc = (rest[:, 1] + 0.5) * h
            F = np.zeros(u.size, complex)
            for s in range(0, u.size, chunk):
                ph = np.exp(-1j * TWO_PI * (np.outer(u[s:s + chunk], xc) + np.outer(v[s:s + chunk], yc)))
                Mu = np.stack([mx[s:s + chunk, a] * my[s:s + chunk, b] for a, b in mon], 1)
                F[s:s + chunk] = ((ph @ Cc) * Mu).sum(1)
            F *= h * h
            return F + self._poly_eval(u, v, al, be, mx, my)
        # clean pixels: coefficients from own-region fits
        clean = np.argwhere(~self.mixed)
        Cc = np.array([self.fits.fit(i, j, self.lab[i, j]) for i, j in clean])   # (Np, nb)
        xc = (clean[:, 0] + 0.5) * h; yc = (clean[:, 1] + 0.5) * h
        F = np.zeros(u.size, complex)
        for s in range(0, u.size, chunk):
            ph = np.exp(-1j * TWO_PI * (np.outer(u[s:s + chunk], xc) + np.outer(v[s:s + chunk], yc)))   # (c, Np)
            S = ph @ Cc                                                            # (c, nb)
            Mu = np.stack([mx[s:s + chunk, a] * my[s:s + chunk, b] for a, b in mon], 1)
            F[s:s + chunk] = (S * Mu).sum(1)
        F *= h * h
        # mixed pixels
        for (i, j), (nx, ny, d, rA, rB) in self.cuts.items():
            cA, cB = self.fits.fit(i, j, rA), self.fits.fit(i, j, rB)
            sq = square()
            PA = clip_halfplane(sq, nx, ny, d, True); PB = clip_halfplane(sq, nx, ny, d, False)
            ph = np.exp(-1j * TWO_PI * (u * (i + 0.5) * h + v * (j + 0.5) * h)) * h * h
            F += ph * (poly_polygon_ft(PA, cA, P, al, be) + poly_polygon_ft(PB, cB, P, al, be))
        for (i, j) in self.corners:
            ph = np.exp(-1j * TWO_PI * (u * (i + 0.5) * h + v * (j + 0.5) * h)) * h * h
            F += ph * self._corner_ft(i, j, al, be)
        # mixed-flagged pixels found pure: mass-corrected fit of the best-matching region
        Mu_all = np.stack([mx[:, a] * my[:, b] for a, b in mon], 1)
        for (i, j), c in self.pure.items():
            if (i, j) in self.cuts or (i, j) in self.corners:
                continue
            ph = np.exp(-1j * TWO_PI * (u * (i + 0.5) * h + v * (j + 0.5) * h)) * h * h
            F += ph * (Mu_all @ c)
        return F


def pixel_box_ft(A, u, v):
    """Standard model: image = piecewise-constant pixels (exact FT of that model)."""
    n = A.shape[0]; h = 1.0 / n
    c = (np.arange(n) + 0.5) * h
    u = np.atleast_1d(u); v = np.atleast_1d(v)
    out = np.empty(u.size, complex)
    for s in range(0, u.size, 256):
        ex = np.exp(-1j * TWO_PI * np.outer(u[s:s + 256], c))
        ey = np.exp(-1j * TWO_PI * np.outer(v[s:s + 256], c))
        out[s:s + 256] = np.einsum('ki,ij,kj->k', ex, A, ey)
    return out * h * h * np.sinc(u * h) * np.sinc(v * h)
