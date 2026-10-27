"""SAFI-2D for images with several smooth closed interfaces (disjoint or nested).

Detection, region labels, local fits and per-pixel lines come from SAFI2D. Every region pair that
shares cut pixels is one interface; each interface is fitted as its own star-shaped Fourier curve
by the conservative fit of CurveSAFI2D (only that pair's band pixels enter its residual). The
transform is the sum of
  - pixels outside every band: exact transform of their own region polynomial;
  - band pixels of interface k: outer polynomial over the pixel plus (inner - outer) over the
    part of the pixel inside curve k (Green's theorem, as in CurveSAFI2D).
Assumes that the bands (one pixel on each side of each curve) do not overlap, i.e. interfaces are
at least about three pixels apart and do not cross.
"""
import numpy as np
from safi2d import SAFI2D, monos, _avg_basis, mu1d
from safi2d_curve import CurveSAFI2D
from curve2d import TWO_PI


class MultiCurveSAFI2D(SAFI2D):
    def __init__(self, A, P=4, two_stage=True, **kw):
        # several interfaces of very different contrast: the point test must not be masked by a
        # stronger edge inside the ring, hence the two-stage test as in one dimension
        kw.setdefault('repair', False)
        # scores below 1e-6 of the image range are treated as smooth: with exactly constant regions
        # (e.g. a zero background) relative tests would otherwise flag any smooth variation nearby
        kw.setdefault('det_floor', 1e-6)
        SAFI2D.__init__(self, A, P=P, two_stage=two_stage, **kw)

    def fit_all(self, min_pts=12, verbose=False, **kw):
        h = self.h
        count = {}
        for (i, j), (nx, ny, d, rA, rB) in self.cuts.items():
            nn = nx * nx + ny * ny
            if abs(d) / np.sqrt(nn) > 0.6:
                continue
            key = tuple(sorted((rA, rB)))
            count[key] = count.get(key, 0) + 1
        self.parts = []
        for pair, cnt in sorted(count.items(), key=lambda kv: -kv[1]):
            if cnt < min_pts:
                continue
            cs = CurveSAFI2D.__new__(CurveSAFI2D)
            cs.__dict__.update({k: v for k, v in self.__dict__.items() if k != 'parts'})
            ok = cs.curve_fit(pair=pair, **kw)
            if verbose:
                print('  pair', pair, 'pts', cnt, 'ok', ok, 'modes', (cs.coef.size - 1) // 2 if ok else None,
                      'rms %.2e' % cs.poly['rms'] if ok else '')
            if ok:
                self.parts.append(cs)
        # overlapping bands: the later (smaller) interface loses the shared pixels in the warning only
        seen = {}
        self.overlap = 0
        for k, cs in enumerate(self.parts):
            for p in cs._geom['band']:
                if p in seen:
                    self.overlap += 1
                seen[p] = k
        self.band_owner = seen
        # nesting: area of each curve for "innermost containing curve"
        t = np.linspace(0, TWO_PI, 1024, endpoint=False)
        self.areas = []
        for cs in self.parts:
            r = cs.curve.rho(t)
            self.areas.append(0.5 * np.mean(r ** 2) * TWO_PI)
        return len(self.parts) > 0

    def _region_at(self, i, j):
        """Region label of the clean side for a pixel outside all bands: inner region of the
        innermost curve containing its centre, otherwise the outer region of the outermost curve."""
        h = self.h
        x, y = (i + 0.5) * h, (j + 0.5) * h
        best, ba = None, np.inf
        for cs, a in zip(self.parts, self.areas):
            if cs.curve.inside(x, y) and a < ba:
                best, ba = cs, a
        if best is not None:
            return best.rin
        big = max(range(len(self.parts)), key=lambda k: self.areas[k])
        return self.parts[big].rout

    def _own(self, i, j):
        if self.lab[i, j] > 0:
            return self.fits.fit(i, j, self.lab[i, j])
        c = self.fits.fit(i, j, self._region_at(i, j)).copy()
        c[0] += self.A[i, j] - _avg_basis(0, 0, self.P) @ c
        return c

    def cartesian(self, N, sel=None):
        """F on the integer grid u, v = -N/2 .. N/2-1 (array [iu, iv]). Pixels outside all bands are
        summed by FFTs of their coefficient images (exact: e^{-2 pi i u x_i} is n-periodic in u up to
        the sign (-1)^m); band pixels by the curve transforms."""
        n, h, P = self.n, self.h, self.P
        mon = monos(P)
        k = np.arange(-N // 2, N // 2)
        C = np.zeros((len(mon), n, n), complex)
        for i in range(n):
            for j in range(n):
                if (i, j) not in self.band_owner:
                    C[:, i, j] = self._own(i, j)
        # D[a, ku, kv] = sum_ij C e^{-2 pi i (u x_i + v y_j)} for u, v integers, via FFT + sign tiling
        Fh = np.fft.fft2(C, axes=(1, 2))                       # sum C e^{-2 pi i (u i + v j)/n}
        ku, kv = np.mod(k, n), np.mod(k, n)
        su = np.exp(-1j * np.pi * k / n)                       # e^{-2 pi i u h/2}
        D = Fh[:, ku][:, :, kv] * su[None, :, None] * su[None, None, :]
        al, be = -1j * TWO_PI * k * h, -1j * TWO_PI * k * h
        mx, my = mu1d(al, P), mu1d(be, P)
        F = np.zeros((N, N), complex)
        for a_, (a, b) in enumerate(mon):
            F += D[a_] * np.outer(mx[:, a], my[:, b])
        F *= h * h
        U, V = np.meshgrid(k.astype(float), k.astype(float), indexing='ij')
        if sel is None:
            F += self.band_part(U.ravel(), V.ravel()).reshape(N, N)
            return F
        Fs = F.ravel()[sel] + self.band_part(U.ravel()[sel], V.ravel()[sel])
        return Fs

    def band_part(self, u, v):
        n, h, P = self.n, self.h, self.P
        al, be = -1j * TWO_PI * u * h, -1j * TWO_PI * v * h
        mx, my = mu1d(al, P), mu1d(be, P)
        F = np.zeros(u.size, complex)
        for k, cs in enumerate(self.parts):
            band = [p for p in cs._geom['band'] if self.band_owner[p] == k]
            keep = cs._geom['band']
            cs._geom = dict(cs._geom, band=band)
            cs.chunk_k = 512
            F += cs._poly_eval(u, v, al, be, mx, my)
            cs._geom = dict(cs._geom, band=keep)
        return F

    def __call__(self, u, v, chunk=256):
        u = np.atleast_1d(np.asarray(u, float)); v = np.atleast_1d(np.asarray(v, float))
        if not getattr(self, 'parts', None):
            return SAFI2D.__call__(self, u, v, chunk)
        n, h, P = self.n, self.h, self.P
        al, be = -1j * TWO_PI * u * h, -1j * TWO_PI * v * h
        mx, my = mu1d(al, P), mu1d(be, P)
        mon = monos(P)
        rest = [(i, j) for i in range(n) for j in range(n) if (i, j) not in self.band_owner]
        Cc = np.array([self._own(i, j) for i, j in rest])
        rest = np.array(rest)
        xc = (rest[:, 0] + 0.5) * h; yc = (rest[:, 1] + 0.5) * h
        F = np.zeros(u.size, complex)
        for s in range(0, u.size, chunk):
            ph = np.exp(-1j * TWO_PI * (np.outer(u[s:s + chunk], xc) + np.outer(v[s:s + chunk], yc)))
            Mu = np.stack([mx[s:s + chunk, a] * my[s:s + chunk, b] for a, b in mon], 1)
            F[s:s + chunk] = ((ph @ Cc) * Mu).sum(1)
        F *= h * h
        for k, cs in enumerate(self.parts):
            band = [p for p in cs._geom['band'] if self.band_owner[p] == k]
            keep = cs._geom['band']
            cs._geom = dict(cs._geom, band=band)
            F += cs._poly_eval(u, v, al, be, mx, my)
            cs._geom = dict(cs._geom, band=keep)
        return F


def _fast_nodes(S, N, Kfac=1.0):
    """Quadrature nodes of all band pixels for the Green's-theorem form of the region integrals.
    Returns local coordinates and weights per node, plus the pixel data they belong to."""
    from numpy.polynomial import legendre as npleg
    n, h = S.n, S.h
    Kmax = np.pi * N / n * np.sqrt(2) * Kfac
    X, Z, W, Q, I, Jp = [], [], [], [], [], []
    for k, cs in enumerate(S.parts):
        C = cs.curve
        for (i, j) in cs._geom['band']:
            if S.band_owner[(i, j)] != k:
                continue
            cO, cI = cs.fits.fit(i, j, cs.rout), cs.fits.fit(i, j, cs.rin)
            d = cI - cO
            xc, yc = (i + 0.5) * h, (j + 0.5) * h
            for (t0, t1) in cs._geom['pieces'].get((i, j), []):
                x0, y0 = C.xy(np.array([t0, t1]))
                Lloc = np.hypot(np.diff(x0), np.diff(y0))[0] / h + 1e-3
                g, w = npleg.leggauss(int(24 + np.ceil(Kmax * Lloc)))
                t = 0.5 * (t0 + t1) + 0.5 * (t1 - t0) * g
                x, y = C.xy(t)
                _, dy = C.dxy(t)
                X.append((x - xc) / h); Z.append((y - yc) / h); W.append(0.5 * (t1 - t0) * w * dy / h)
                I.append(np.full(t.size, i)); Jp.append(np.full(t.size, j)); Q.append(np.repeat(d[None, :], t.size, 0))
            for (ya, yb) in cs._geom['redge'].get((i, j), []):
                z0, z1 = (ya - yc) / h, (yb - yc) / h
                g, w = npleg.leggauss(int(12 + np.ceil(Kmax)))
                z = 0.5 * (z0 + z1) + 0.5 * (z1 - z0) * g
                X.append(np.full(z.size, 0.5)); Z.append(z); W.append(0.5 * (z1 - z0) * w)
                I.append(np.full(z.size, i)); Jp.append(np.full(z.size, j)); Q.append(np.repeat(d[None, :], z.size, 0))
    return (np.concatenate(X), np.concatenate(Z), np.concatenate(W), np.concatenate(I), np.concatenate(Jp),
            np.concatenate(Q, 0))


def fast_cartesian(S, N, msp=10):
    """F on the integer grid u, v = -N/2..N/2-1 with all band terms evaluated by 2-D type-1 NUFFTs.
    |alpha| >= 1 (alpha = -2 pi i u h): closed-form inner integrals, one NUFFT per power alpha^-p;
    |alpha| < 1: Gauss rule for the inner integral, one NUFFT over the product nodes."""
    from math import factorial
    from numpy.polynomial import legendre as npleg
    from nufft import nufft1_2d
    n, h, P = S.n, S.h, S.P
    mon = monos(P)
    k = np.arange(-N // 2, N // 2)
    # pixels outside all bands, and the outer polynomial of band pixels, by FFT
    Cimg = np.zeros((len(mon), n, n), complex)
    for i in range(n):
        for j in range(n):
            if (i, j) in S.band_owner:
                cs = S.parts[S.band_owner[(i, j)]]
                Cimg[:, i, j] = cs.fits.fit(i, j, cs.rout)
            else:
                Cimg[:, i, j] = S._own(i, j)
    Fh = np.fft.fft2(Cimg, axes=(1, 2))
    km = np.mod(k, n)
    sh = np.exp(-1j * np.pi * k / n)
    D = Fh[:, km][:, :, km] * sh[None, :, None] * sh[None, None, :]
    al = -1j * TWO_PI * k * h
    mx = mu1d(al, P)
    F = np.zeros((N, N), complex)
    for a_, (a, b) in enumerate(mon):
        F += D[a_] * np.outer(mx[:, a], mx[:, b])
    F *= h * h
    # band nodes
    xi, ze, W, I, Jp, d = _fast_nodes(S, N)
    xq, yq = (I + 0.5) * h, (Jp + 0.5) * h
    zb = np.stack([ze ** b for b in range(P + 1)], 1)                   # (nodes, P+1)
    # |alpha| >= 1 family
    Fx = np.zeros((N, N), complex)
    for p in range(1, P + 2):
        A1 = np.zeros(xi.size, complex); A0 = np.zeros(xi.size, complex)
        for a_, (a, b) in enumerate(mon):
            if a < p - 1:
                continue
            cf = (-1) ** (p - 1) * factorial(a) / factorial(a - p + 1)
            A1 += d[:, a_] * zb[:, b] * cf * xi ** (a - p + 1)
            A0 += d[:, a_] * zb[:, b] * cf * (-0.5) ** (a - p + 1)
        px = np.r_[xq + h * xi, xq - 0.5 * h]
        py = np.r_[yq + h * ze, yq + h * ze]
        cc = h * h * np.r_[W * A1, -W * A0]
        G = nufft1_2d(px, py, cc, N, msp=msp)
        with np.errstate(divide='ignore', invalid='ignore'):
            Fx += G * (al ** (-p))[:, None]
    # |alpha| < 1 family: Gauss rule in the inner variable
    gl, wl = npleg.leggauss(12)
    s = -0.5 + (xi[:, None] + 0.5) * (gl[None, :] + 1) / 2                # (nodes, 12)
    ws = W[:, None] * (xi[:, None] + 0.5) / 2 * wl[None, :]
    poly = np.zeros(s.shape, complex)
    for a_, (a, b) in enumerate(mon):
        poly += d[:, a_, None] * s ** a * zb[:, b, None]
    px = (xq[:, None] + h * s).ravel(); py = np.repeat(yq + h * ze, 12)
    Fs = nufft1_2d(px, py, (h * h * ws * poly).ravel(), N, msp=msp)
    small = (np.abs(al) < 1.0)[:, None]
    F += np.where(small, Fs, Fx)
    return F
