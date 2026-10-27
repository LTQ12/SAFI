"""SAFI cell-average mode: data abar_j = (1/h) int_{x_j}^{x_{j+1}} f.

Reconstruction through the primitive V(x_j) = h sum_{i<j} abar_i (continuous even at jumps):
on each cell, P = d/dx of the degree-(M+1) interpolant of V on a stencil inside one smooth
piece.  A jump cell is split at c solving Harten's subcell-resolution equation
    int_{x_j}^{c} P_L + int_{c}^{x_{j+1}} P_R = h abar_j ,
which locates c to O(h^{M+2}/|J|) without any extra data.
"""
import numpy as np
from numpy.polynomial import legendre as npleg
from numpy.polynomial import polynomial as P
from scipy.optimize import brentq
from safi import leg_ft_cells, Break, TWO_PI, local_ratio_flags, eno_scores, DETECT_LOG


def _prim_poly(X, V, idx):
    """Monomial coefficients (in s = x - X[idx[0]]) of the interpolant of V on nodes idx."""
    s = X[idx] - X[idx[0]]
    c = np.polyfit(s, V[idx], len(idx) - 1)[::-1]
    return X[idx[0]], c


def _deriv_eval(x0, c, x):
    return P.polyval(np.asarray(x) - x0, P.polyder(c))


def _deriv_int(x0, c, a, b):
    return P.polyval(b - x0, c) - P.polyval(a - x0, c)


class SAFICell:
    def __init__(self, xe, abar, M=6, kappa=30.0, floor=1e-9, breaks_known=None, zones=False, zmax=4,
                 auto_jump_degree=False):
        """xe: n+1 cell edges (uniform), abar: n cell averages.
        zones=True: a detected jump may occupy several contaminated cells (e.g. a shock smeared by
        a finite-volume scheme); the zone [j0, j1] grows while neighbouring cells are inconsistent
        with the smooth one-sided reconstruction, and c solves the zone-wide mass balance."""
        self.xe, self.ab, self.M = np.asarray(xe, float), np.asarray(abar, float), M
        self.h = self.xe[1] - self.xe[0]
        n = self.ab.size
        self.V = np.concatenate([[0.0], np.cumsum(self.ab) * self.h])
        if breaks_known is None:
            self.breaks = []
            if zones:
                for j0, j1 in self._detect_zones(kappa, floor, zmax):
                    if auto_jump_degree:
                        # extrapolation degree matched to data accuracy: minimise the location
                        # uncertainty estimate over 1..M (noisy/low-order data -> low degree)
                        cands = []
                        for Mj in range(1, M + 1):
                            self.M = Mj
                            try:
                                cands.append(self._subcell(j0, j1))
                            except Exception:
                                pass
                        self.M = M
                        self.breaks.append(min(cands, key=lambda b: b.delta))
                    else:
                        self.breaks.append(self._subcell(j0, j1))
            else:
                for j in self._detect(kappa, floor):
                    self.breaks.append(self._subcell(j, j))
        else:
            self.breaks = []
            for c in breaks_known:
                j = int(np.searchsorted(self.xe, c) - 1)
                self.breaks.append(Break(c, c, 0.0))
                self.breaks[-1].j = j; self.breaks[-1].j1 = j
        self._build()

    # --- stencil helpers on the primitive (nodes = cell edges)
    def _left_nodes(self, j, k):      # primitive nodes using only cells < j  (edges <= x_j)
        return np.arange(max(0, j - k), j + 1)

    def _right_nodes(self, j, k):     # only cells > j  (edges >= x_{j+1})
        n = self.ab.size
        return np.arange(j + 1, min(n, j + 1 + k) + 1)

    def _detect(self, kappa, floor):
        """ENO scores on the averages: a jump makes the single value abar_j anomalous, so the
        score of cell j is min |Delta^{q} abar| over the (q+1)-windows that contain j. Orders
        q = M+1 and q = M are tested and united: each order has isolated subcell positions at
        which its score vanishes (k/(M+1) and k/M for a jump), and the two sets are disjoint."""
        n, M = self.ab.size, self.M
        fl_abs = floor * np.max(np.abs(self.ab))
        ab = self.ab

        def scores(q):
            D = np.abs(np.diff(ab, q))
            sq = np.full(n, np.inf)
            for off in range(q + 1):
                j = np.arange(n); s = j - off
                ok = (s >= 0) & (s < D.size)
                sq[ok] = np.minimum(sq[ok], D[s[ok]])
            return sq
        sc = scores(M + 1)

        def flank_min(j, q):
            w = [abs(np.diff(ab[j - 1 - q:j], q)[0])] if j - 1 - q >= 0 else []
            w += [abs(np.diff(ab[j + 1:j + 2 + q], q)[0])] if j + 1 + q <= n - 1 else []
            return min(w) if w else None
        # a flag must stand out against at least one clean flanking window (see safi.detect_jumps)
        flag = set(int(j) for j in local_ratio_flags(sc, M, kappa, fl_abs)
                   if flank_min(int(j), M + 1) is None or sc[j] > kappa * flank_min(int(j), M + 1))
        # step between two averages (jump close to a cell edge): pair scores
        sp = eno_scores(ab, M)
        for i in local_ratio_flags(sp, M, kappa, fl_abs):
            if i in flag or i + 1 in flag:
                continue
            # pick the neighbouring cell in which the subcell equation has a root
            flag.add(int(i) if self._has_root(int(i)) else int(i) + 1)
        # order M covers the zeros of the order-(M+1) kernel; confirmed against the clean
        # windows of M+1 averages that flank cell j (see detect_jumps in safi.py)
        s0 = scores(M)
        for j in local_ratio_flags(s0, M, kappa, fl_abs):
            j = int(j)
            if any(abs(j - k) <= 1 for k in flag) or j < M + 1 or j + 1 + M > n - 1:
                continue
            flank = max(abs(np.diff(ab[j - 1 - M:j], M)[0]), abs(np.diff(ab[j + 1:j + 2 + M], M)[0]))
            if s0[j] > kappa * flank:
                flag.add(j); DETECT_LOG.append(('cell', j))
        self.scores = sc
        # resolvability: a break needs M+1 cells on each side (as for point data)
        flag = {j for j in flag if M + 1 <= j <= n - 2 - M}
        return sorted(flag)

    def _detect_zones(self, kappa, floor, zmax):
        """Smeared discontinuities (finite-volume data): ring test. Cell i is contaminated if its
        ENO score exceeds kappa x the largest score in the ring W < |k - i| <= 3W (W = M+2),
        i.e. the background a few cells away. Contiguous contaminated cells form one zone,
        widened by `_zone` and capped at the strongest cell +- zmax."""
        n, M = self.ab.size, self.M
        D = np.abs(np.diff(self.ab, M + 1))
        sc = np.full(n, np.inf)
        for off in range(M + 2):
            j = np.arange(n); s = j - off
            ok = (s >= 0) & (s < D.size)
            sc[ok] = np.minimum(sc[ok], D[s[ok]])
        self.scores = sc
        W = M + 2
        fl_abs = floor * np.max(np.abs(self.ab))
        bad = np.zeros(n, bool)
        for i in range(M + 1, n - M - 1):
            if sc[i] <= fl_abs:
                continue
            ring = np.r_[max(0, i - 3 * W):max(0, i - W), min(n, i + W + 1):min(n, i + 3 * W + 1)]
            ref = np.max(sc[ring]) if ring.size else 0.0
            bad[i] = sc[i] > kappa * ref
        # jump on a cell edge: no single contaminated value, but a step between two values
        sp = eno_scores(self.ab, M)                       # pair (i, i+1) scores
        for i in range(M + 1, n - M - 2):
            if sp[i] <= fl_abs:
                continue
            ring = np.r_[max(0, i - 3 * W):max(0, i - W), min(n - 1, i + W + 1):min(n - 1, i + 3 * W + 1)]
            ref = np.max(sp[ring]) if ring.size else 0.0
            if sp[i] > kappa * ref:
                bad[i] = bad[i + 1] = True
        zones, i = [], 0
        while i < n:
            if bad[i]:
                j = i
                while j + 1 < n and bad[j + 1]:
                    j += 1
                pk = i + int(np.argmax(sc[i:j + 1]))
                z0, z1 = self._zone(pk, zmax)
                zones.append((min(z0, max(i, pk - zmax)), max(z1, min(j, pk + zmax))))
                i = j + 1
            else:
                i += 1
        return zones

    def _has_root(self, j):
        n, M = self.ab.size, self.M
        if j - (M + 1) < 0 or j + 1 + (M + 1) > n:
            return False
        xL0, cL = _prim_poly(self.xe, self.V, self._left_nodes(j, M + 1))
        xR0, cR = _prim_poly(self.xe, self.V, self._right_nodes(j, M + 1))
        a, b = self.xe[j], self.xe[j + 1]
        tot = self.h * self.ab[j]
        G = lambda c: _deriv_int(xL0, cL, a, c) + _deriv_int(xR0, cR, c, b) - tot
        return G(a) * G(b) <= 0

    def _misfit_left(self, c):
        """|average of cell c predicted from cells < c| - actual| (smooth-left consistency)."""
        M = self.M
        if c - (M + 1) < 0:
            return np.inf
        x0, cc = _prim_poly(self.xe, self.V, self._left_nodes(c, M + 1))
        return abs(_deriv_int(x0, cc, self.xe[c], self.xe[c + 1]) / self.h - self.ab[c])

    def _misfit_right(self, c):
        M, n = self.M, self.ab.size
        if c + 1 + (M + 1) > n:
            return np.inf
        x0, cc = _prim_poly(self.xe, self.V, self._right_nodes(c, M + 1))
        return abs(_deriv_int(x0, cc, self.xe[c], self.xe[c + 1]) / self.h - self.ab[c])

    def _zone(self, j, zmax):
        """Grow [j0, j1] while the neighbouring cell is inconsistent with its own side."""
        lev_l = max([self._misfit_left(c) for c in range(j - zmax - 6, j - zmax - 1) if c >= 0] + [0.0])
        lev_r = max([self._misfit_right(c) for c in range(j + zmax + 2, j + zmax + 7) if c < self.ab.size] + [0.0])
        tl = 30 * lev_l + 1e-12 * np.max(np.abs(self.ab))
        tr = 30 * lev_r + 1e-12 * np.max(np.abs(self.ab))
        j0, j1 = j, j
        while j0 - j > -zmax and j0 - 1 >= 0 and self._misfit_left(j0 - 1) > tl:
            j0 -= 1
        while j1 - j < zmax and j1 + 1 < self.ab.size and self._misfit_right(j1 + 1) > tr:
            j1 += 1
        return j0, j1

    def _subcell(self, j, j1=None, moved=False):
        M = self.M
        j1 = j if j1 is None else j1
        xL0, cL = _prim_poly(self.xe, self.V, self._left_nodes(j, M + 1))
        xR0, cR = _prim_poly(self.xe, self.V, self._right_nodes(j1, M + 1))
        a, b = self.xe[j], self.xe[j1 + 1]
        # kink test: the reconstructed one-sided values intersect inside the cell (f continuous)
        D = lambda c: _deriv_eval(xR0, cR, c) - _deriv_eval(xL0, cL, c)
        g8 = (b - a) / 8                          # kinks at a cell edge: root may lie just outside
        tt = np.linspace(a - g8, b + g8, 81); dd = D(tt)
        kk = np.where(np.sign(dd[:-1]) * np.sign(dd[1:]) <= 0)[0]
        if kk.size and j1 == j:
            ck = brentq(D, tt[kk[0]], tt[kk[0] + 1], xtol=1e-16, rtol=1e-15) if D(tt[kk[0]]) != 0 else tt[kk[0]]
            if not moved and not a <= ck <= b:
                # the intersection lies in a neighbouring cell: localise from that cell's stencils
                j2 = j - 1 if ck < a else j + 1
                if M + 1 <= j2 <= self.ab.size - 2 - (M + 1):
                    br2 = self._subcell(j2, j2, moved=True)
                    if br2.kind == 'kink' and self.xe[j2] <= br2.cstar <= self.xe[j2 + 1]:
                        return br2
            ck = min(max(ck, a), b)
            xL1, cL1 = _prim_poly(self.xe, self.V, self._left_nodes(j, M))
            xR1, cR1 = _prim_poly(self.xe, self.V, self._right_nodes(j, M))
            D1 = lambda c: _deriv_eval(xR1, cR1, c) - _deriv_eval(xL1, cL1, c)
            try:
                c1 = brentq(D1, a - g8, b + g8, xtol=1e-16, rtol=1e-15)
                dl = 2 * abs(c1 - ck)
            except ValueError:
                dl = 0.5 * self.h
            dl = min(dl + 8 * np.finfo(float).eps * max(1.0, abs(ck)), 0.5 * self.h)
            br = Break(ck - dl, ck + dl, 0.0, kind='kink')
            br.j, br.j1, br.cstar, br.polys = j, j, ck, (xL0, cL, xR0, cR)
            return br
        tot = self.h * self.ab[j:j1 + 1].sum()
        G = lambda c: _deriv_int(xL0, cL, a, c) + _deriv_int(xR0, cR, c, b) - tot
        Ga, Gb = G(a), G(b)
        sc_ = 1e-9 * self.h * (np.max(np.abs(self.ab)) + 1e-300)
        if Ga * Gb < 0:
            c = brentq(G, a, b, xtol=1e-15 * max(1, abs(b)), rtol=1e-15)
        else:   # jump on (or within the extrapolation error of) a cell edge: nearest endpoint
            c = a if abs(Ga) <= abs(Gb) else b
        J = _deriv_eval(xR0, cR, c) - _deriv_eval(xL0, cL, c)
        # location uncertainty: perturb by degree-M one-sided fits (error proxy) -> delta
        xL1, cL1 = _prim_poly(self.xe, self.V, self._left_nodes(j, M))
        xR1, cR1 = _prim_poly(self.xe, self.V, self._right_nodes(j1, M))
        dG = abs(_deriv_int(xL0, cL, a, c) - _deriv_int(xL1, cL1, a, c)) + \
             abs(_deriv_int(xR0, cR, c, b) - _deriv_int(xR1, cR1, c, b))
        dG += abs(G(c))                           # nonzero only when the root lies outside [a, b]
        delta = min(2 * dG / max(abs(J), 1e-300) + 8 * np.finfo(float).eps * max(1.0, abs(c)), 0.5 * (b - a))
        br = Break(c - delta, c + delta, complex(J))
        br.j, br.j1, br.cstar = j, j1, c
        br.polys = (xL0, cL, xR0, cR)
        return br

    def _build(self):
        n, M = self.ab.size, self.M
        zones = sorted(((b.j, getattr(b, 'j1', b.j), b) for b in self.breaks), key=lambda z: z[0])
        A, B, C = [], [], []
        t, _ = npleg.leggauss(M + 1)
        bounds = [(-1, -1)] + [(z[0], z[1]) for z in zones] + [(n, n)]
        for p in range(len(bounds) - 1):
            j0, j1 = bounds[p][1] + 1, bounds[p + 1][0] - 1        # smooth cells j0..j1
            if j1 < j0:
                continue
            for j in range(j0, j1 + 1):
                k = min(M + 1, j1 + 1 - j0)
                s = j - k // 2
                s = max(j0, min(s, j1 + 1 - k))
                x0, c = _prim_poly(self.xe, self.V, np.arange(s, s + k + 1))
                l, r = self.xe[j], self.xe[j + 1]
                xx = 0.5 * (l + r) + 0.5 * (r - l) * t
                A.append(l); B.append(r); C.append(npleg.legfit(t, _deriv_eval(x0, c, xx), M))
        for j, j1, b in zones:
            if not hasattr(b, 'polys'):           # known-break mode: build side polys now
                xL0, cL = _prim_poly(self.xe, self.V, self._left_nodes(j, M + 1))
                xR0, cR = _prim_poly(self.xe, self.V, self._right_nodes(j1, M + 1))
                b.polys, b.cstar = (xL0, cL, xR0, cR), b.lo
                b.J = complex(_deriv_eval(xR0, cR, b.lo) - _deriv_eval(xL0, cL, b.lo))
            xL0, cL, xR0, cR = b.polys
            c = b.cstar
            for (l, r, x0, cc) in ((self.xe[j], c, xL0, cL), (c, self.xe[j1 + 1], xR0, cR)):
                if r - l <= 0:
                    continue
                xx = 0.5 * (l + r) + 0.5 * (r - l) * t
                A.append(l); B.append(r); C.append(npleg.legfit(t, _deriv_eval(x0, cc, xx), M))
        o = np.argsort(A)
        self.A, self.B, self.C = np.array(A)[o], np.array(B)[o], np.array(C)[o]

    def __call__(self, u):
        return leg_ft_cells(self.A, self.B, self.C, u)
