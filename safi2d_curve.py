"""SAFI-2D for a smooth curved interface: conservative fit of a star-shaped Fourier curve.

Uses SAFI2D (detection, region labels, local phase fits, per-pixel lines) and replaces the polygon by
  r(t) = c + rho(t) (cos t, sin t),  rho(t) = a0 + sum_{j<=N} (a_j cos jt + b_j sin jt).
The coefficients minimise the conservation residuals of all pixels within one pixel of the curve,
  r_q = [ <p_out,q> + int_{q n Omega} (p_in,q - p_out,q) ] - abar_q   (local pixel coordinates),
by Gauss-Newton/Levenberg-Marquardt with the exact shape derivative
  d/d theta int_{q n Omega} p dA = int_{curve in q} p(r(t)) rho(t) d rho/d theta (t) dt,
and the number of modes N grows while the residual drops by more than 10 %.
Transform of a band pixel: exact Green's theorem on q n Omega with oscillatory Gauss rules on the
curve pieces and closed forms on the pixel edges.
"""
import numpy as np
from numpy.polynomial import legendre as npleg
from safi2d import SAFI2D, monos, _avg_basis, mu1d
from curve2d import StarCurve, fourier_rho, TWO_PI

_G40, _W40 = npleg.leggauss(40)
_G12, _W12 = npleg.leggauss(12)
_FACT = [1, 1, 2, 6, 24, 120, 720, 5040]
_GL = {}


def _gauss(N):
    N = int(32 * np.ceil(N / 32))
    if N not in _GL:
        _GL[N] = npleg.leggauss(N)
    return _GL[N]


def _prim_exp(al, s, a):
    """int s^a exp(al s) ds antiderivative at s (|al| >= 1): exp(al s) sum_j (-1)^j a!/(a-j)! s^(a-j)/al^(j+1)."""
    out = 0
    for j in range(a + 1):
        out = out + ((-1) ** j) * (_FACT[a] / _FACT[a - j]) * s ** (a - j) / al ** (j + 1)
    return np.exp(al * s) * out


def _int_pow_exp(al, lo, hi, a):
    """int_lo^hi s^a exp(al s) ds for arrays al (k,) and lo, hi broadcastable to (k, m)."""
    al = np.asarray(al, complex)
    big = np.abs(al) >= 1.0
    res = np.zeros(np.broadcast(al[:, None], lo, hi).shape, complex)
    if big.any():
        A = al[big][:, None]
        res[big] = _prim_exp(A, np.broadcast_to(hi, res.shape)[big], a) - _prim_exp(A, np.broadcast_to(lo, res.shape)[big], a)
    if (~big).any():
        A = al[~big][:, None, None]
        L = np.broadcast_to(lo, res.shape)[~big][..., None]
        H = np.broadcast_to(hi, res.shape)[~big][..., None]
        s = 0.5 * (L + H) + 0.5 * (H - L) * _G12
        res[~big] = 0.5 * (H[..., 0] - L[..., 0]) * (_W12 * s ** a * np.exp(A * s)).sum(-1)
    return res


class CurveSAFI2D(SAFI2D):
    def curve_fit(self, modes=(4, 6, 8, 12, 16, 24, 32), verbose=False, maxit=40, pair=None):
        """Fit one closed curve. pair = (region A, region B) selects the interface; by default the
        pair with the most cut pixels. Only cut points of that pair are used."""
        h, n, P = self.h, self.n, self.P
        pts, pairs = [], {}
        for (i, j), (nx, ny, d, rA, rB) in self.cuts.items():
            nn = nx * nx + ny * ny
            if abs(d) / np.sqrt(nn) > 0.6:
                continue
            key = tuple(sorted((rA, rB)))
            pairs[key] = pairs.get(key, 0) + 1
            if pair is None or key == tuple(sorted(pair)):
                pts.append(((i + 0.5) * h + h * d * nx / nn, (j + 0.5) * h + h * d * ny / nn, key))
        if not pairs:
            return False
        if pair is None:
            pair = max(pairs, key=pairs.get)
        pair = tuple(sorted(pair))
        pts = np.array([p[:2] for p in pts if p[2] == pair])
        if len(pts) < 8:
            return False
        c0 = pts.mean(0)
        ci, cj = int(np.clip(c0[0] / h, 0, n - 1)), int(np.clip(c0[1] / h, 0, n - 1))
        cand = np.argwhere(np.isin(self.lab, pair))
        k = np.argmin(np.sum((cand - [ci, cj]) ** 2, 1))
        rin = int(self.lab[tuple(cand[k])])
        rout = pair[0] if pair[1] == rin else pair[1]
        self.rin, self.rout, self.c0 = rin, rout, c0
        # initial Fourier fit of the per-pixel line points (robust)
        th = np.mod(np.arctan2(pts[:, 1] - c0[1], pts[:, 0] - c0[0]), TWO_PI)
        rr = np.hypot(pts[:, 0] - c0[0], pts[:, 1] - c0[1])
        N0 = modes[0]
        B = fourier_rho(np.zeros(2 * N0 + 1))[3](th)
        keep = np.ones(th.size, bool)
        for _ in range(4):
            coef = np.linalg.lstsq(B[keep], rr[keep], rcond=None)[0]
            res = np.abs(B @ coef - rr)
            keep = res <= 4 * np.median(res[keep]) + 0.1 * h
        self._cp = {}
        best = None
        for N in modes:
            if best is not None:
                coef = np.r_[best['coef'], np.zeros(2 * (N - (best['coef'].size - 1) // 2))]
            coef, rms, band = self._gn(coef, maxit, verbose)
            if verbose:
                print('  modes %d: rms %.3e  band %d' % (N, rms, len(band)))
            if best is not None and rms > 0.9 * best['rms']:
                break
            best = dict(coef=coef, rms=rms, band=band)
        self.coef = best['coef']
        self.curve = self._make_curve(self.coef)
        self._geom = self._geometry(self.curve)
        self.poly = dict(band=self._geom['band'], rin=rin, rout=rout, curve=self.curve, rms=best['rms'])
        return True

    # ------------------------------------------------------------ geometry for one parameter vector
    def _make_curve(self, coef):
        rho, drho, d2rho, _ = fourier_rho(coef)
        return StarCurve(self.c0, rho, drho, d2rho)

    def _geometry(self, C):
        h, n = self.h, self.n
        cr = C.crossings(h)
        pcs = {}
        for (i, j, t0, t1) in C.pieces(h, cr):
            if 0 <= i < n and 0 <= j < n:
                pcs.setdefault((i, j), []).append((t0, t1))
        redge = C.right_edges(h, n, cr)
        band = set()
        for (i, j) in pcs:
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if 0 <= i + di < n and 0 <= j + dj < n:
                        band.add((i + di, j + dj))
        return dict(pieces=pcs, redge=redge, band=sorted(band))

    def _cpair(self, i, j):
        if (i, j) not in self._cp:
            self._cp[(i, j)] = (self.fits.fit(i, j, self.rout), self.fits.fit(i, j, self.rin))
        return self._cp[(i, j)]

    def _moments(self, C, geo, i, j, deriv_basis=None):
        """Local moments m_ab = int_{q n Omega} xi^a zeta^b (q in local coords), and, if deriv_basis is
        given, the matrix D[k, ab] = d m_ab / d coef_k (shape derivative)."""
        h, P = self.h, self.P
        mon = monos(P)
        xc, yc = (i + 0.5) * h, (j + 0.5) * h
        m = np.zeros(len(mon))
        D = None if deriv_basis is None else np.zeros((deriv_basis(np.zeros(1)).shape[-1], len(mon)))
        for (t0, t1) in geo['pieces'].get((i, j), []):
            t = 0.5 * (t0 + t1) + 0.5 * (t1 - t0) * _G40
            w = 0.5 * (t1 - t0) * _W40
            x, y = C.xy(t)
            _, dy = C.dxy(t)
            xi, ze, dze = (x - xc) / h, (y - yc) / h, dy / h
            for k, (a, b) in enumerate(mon):
                Q = (xi ** (a + 1) - (-0.5) ** (a + 1)) / (a + 1) * ze ** b
                m[k] += np.sum(w * Q * dze)
            if D is not None:
                rho = C.rho(t)
                Bt = deriv_basis(t)                                     # (q, ncoef)
                pv = np.stack([xi ** a * ze ** b for a, b in mon], 1)    # (q, nb)
                D += (Bt * (w * rho)[:, None]).T @ pv / h ** 2
        for (y0, y1) in geo['redge'].get((i, j), []):
            z0, z1 = (y0 - yc) / h, (y1 - yc) / h
            for k, (a, b) in enumerate(mon):
                m[k] += ((0.5) ** (a + 1) - (-0.5) ** (a + 1)) / (a + 1) * (z1 ** (b + 1) - z0 ** (b + 1)) / (b + 1)
        return m, D

    def _resid(self, coef, jac=False):
        C = self._make_curve(coef)
        geo = self._geometry(C)
        P = self.P
        g0 = _avg_basis(0, 0, P)
        basis = fourier_rho(coef)[3] if jac else None
        rows, J = [], []
        for (i, j) in geo['band']:
            cO, cI = self._cpair(i, j)
            m, D = self._moments(C, geo, i, j, basis)
            rows.append(g0 @ cO + m @ (cI - cO) - self.A[i, j])
            if jac:
                J.append(D @ (cI - cO))
        return np.array(rows), (np.array(J) if jac else None), geo

    def _gn(self, coef, maxit, verbose):
        lam = 1e-3
        r, J, geo = self._resid(coef, True)
        rms = np.sqrt(np.mean(r ** 2))
        for it in range(maxit):
            A_ = J.T @ J
            g = J.T @ r
            step = np.linalg.solve(A_ + lam * np.diag(np.diag(A_) + 1e-30), -g)
            r2, J2, geo2 = self._resid(coef + step, True)
            rms2 = np.sqrt(np.mean(r2 ** 2))
            if rms2 < rms:
                coef, r, J, geo, lam = coef + step, r2, J2, geo2, max(lam / 5, 1e-9)
                done = rms - rms2 < 1e-6 * rms or np.max(np.abs(step)) < 1e-14
                rms = rms2
                if done:
                    break
            else:
                lam *= 10
                if lam > 1e8:
                    break
        return coef, rms, geo['band']

    # ------------------------------------------------------------ evaluation
    def _own_coef(self, i, j):
        if self.lab[i, j] > 0:
            return self.fits.fit(i, j, self.lab[i, j])
        h = self.h
        inside = bool(self.curve.inside((i + 0.5) * h, (j + 0.5) * h))
        c = self.fits.fit(i, j, self.rin if inside else self.rout).copy()
        c[0] += self.A[i, j] - _avg_basis(0, 0, self.P) @ c
        return c

    @staticmethod
    def _I_all(al, s, P):
        """I_a(al, s) = int_{-1/2}^{s} t^a exp(al t) dt for a = 0..P; al (k,), s (q,) -> list of (k, q)."""
        al = np.asarray(al, complex)
        big = np.abs(al) >= 1.0
        out = [np.zeros((al.size, s.size), complex) for _ in range(P + 1)]
        if big.any():
            A = al[big][:, None]
            E1, E0 = np.exp(A * s[None, :]), np.exp(-0.5 * A)
            S1 = np.zeros((A.shape[0], s.size), complex); S0 = np.zeros((A.shape[0], 1), complex)
            for a_ in range(P + 1):
                # S_a(x) = x^a / al - (a / al) S_{a-1}(x)  (antiderivative factor of x^a e^{al x})
                S1 = s[None, :] ** a_ / A - (a_ / A) * S1
                S0 = (-0.5) ** a_ / A - (a_ / A) * S0
                out[a_][big] = E1 * S1 - E0 * S0
        if (~big).any():
            A = al[~big][:, None, None]
            L, H = -0.5, s[None, :, None]
            x = 0.5 * (L + H) + 0.5 * (H - L) * _G12                     # (1, q, 12)
            ex = np.exp(A * x)                                          # (ks, q, 12)
            for a_ in range(P + 1):
                out[a_][~big] = 0.5 * (s[None, :] + 0.5) * (_W12 * x ** a_ * ex).sum(-1)
        return out

    def _region_ft(self, i, j, al, be):
        """G_ab(al, be) = int_{q n Omega} xi^a zeta^b exp(al xi + be zeta), all monomials. (k, nb).
        al, be: one chunk of frequencies of similar magnitude (sets the Gauss order on the curve)."""
        h, P, C, geo = self.h, self.P, self.curve, self._geom
        mon = monos(P)
        xc, yc = (i + 0.5) * h, (j + 0.5) * h
        K = float(np.max(np.hypot(np.abs(al), np.abs(be))))
        out = np.zeros((al.size, len(mon)), complex)
        for (t0, t1) in geo['pieces'].get((i, j), []):
            x0, y0 = C.xy(np.array([t0, t1]))
            Lloc = np.hypot(np.diff(x0), np.diff(y0))[0] / h + 1e-3
            Nt = int(24 + np.ceil(1.0 * K * Lloc))
            g, w = _gauss(Nt)
            t = 0.5 * (t0 + t1) + 0.5 * (t1 - t0) * g
            w = 0.5 * (t1 - t0) * w
            x, y = C.xy(t)
            _, dy = C.dxy(t)
            xi, ze, dze = (x - xc) / h, (y - yc) / h, dy / h
            ez = np.exp(np.outer(be, ze)) * (w * dze)[None, :]             # (k, q)
            Ia = self._I_all(al, xi, P)
            zb = [ze ** b for b in range(P + 1)]
            for k_, (a, b) in enumerate(mon):
                out[:, k_] += (Ia[a] * ez) @ zb[b]
        for (ya, yb) in geo['redge'].get((i, j), []):
            z0, z1 = (ya - yc) / h, (yb - yc) / h
            Ix = [_int_pow_exp(al, -0.5, np.array([[0.5]]), a)[:, 0] for a in range(P + 1)]
            Iz = [_int_pow_exp(be, np.array([[z0]]), np.array([[z1]]), b)[:, 0] for b in range(P + 1)]
            for k_, (a, b) in enumerate(mon):
                out[:, k_] += Ix[a] * Iz[b]
        return out

    def _poly_eval(self, u, v, al, be, mx, my):
        h, P = self.h, self.P
        mon = monos(P)
        Mu_all = np.stack([mx[:, a] * my[:, b] for a, b in mon], 1)
        F = np.zeros(u.size, complex)
        order = np.argsort(np.hypot(np.abs(al), np.abs(be)))
        ck = getattr(self, 'chunk_k', 16)
        chunks = [order[s_:s_ + ck] for s_ in range(0, order.size, ck)]
        for (i, j) in self._geom['band']:
            cO, cI = self.fits.fit(i, j, self.rout), self.fits.fit(i, j, self.rin)
            ph = np.exp(-1j * TWO_PI * (u * (i + 0.5) * h + v * (j + 0.5) * h)) * h * h
            val = Mu_all @ cO
            if (i, j) in self._geom['pieces'] or (i, j) in self._geom['redge']:
                d = cI - cO
                for ch in chunks:
                    val[ch] = val[ch] + self._region_ft(i, j, al[ch], be[ch]) @ d
            elif self.curve.inside((i + 0.5) * h, (j + 0.5) * h):
                val = val + Mu_all @ (cI - cO)
            F += ph * val
        return F
