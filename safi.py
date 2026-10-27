"""SAFI: Singularity-Aware Fourier Integration.

Evaluate F(u) = int_{p0}^{p1} f(x) exp(-2 pi i u x) dx at arbitrary frequencies u
for a compactly supported, piecewise-smooth f, from one of several data modes:

  'point'    : fixed uniform point samples f(x_j)
  'adaptive' : point samples + a callable f that may be queried (bisection)
  'known'    : point samples + exact interior break locations
  (cell-average mode lives in safi_cell.py, second round)

Engine: piecewise Legendre interpolant aligned with (estimated) breaks; its Fourier
integral is exact for every u through  int_{-1}^{1} P_m(t) e^{-i w t} dt = 2 (-i)^m j_m(w).
Every output frequency comes with an a-posteriori error estimate e(u).
"""
import numpy as np
from scipy.special import spherical_jn
from scipy.interpolate import BarycentricInterpolator
from numpy.polynomial import legendre as npleg

TWO_PI = 2.0 * np.pi


# ----------------------------------------------------------------------------- engine
def leg_ft_cells(a, b, coef, u, chunk=512):
    """Exact Fourier integral of a piecewise polynomial.

    a, b : (K,) cell endpoints; coef : (K, M+1) Legendre coefficients on t in [-1, 1].
    Returns F(u) = sum_k int_{a_k}^{b_k} p_k(x) exp(-2 pi i u x) dx.
    Dense O(n_u K M); the inner sum over k is a type-3 exponential sum per mode m.
    """
    u = np.atleast_1d(np.asarray(u, float))
    a = np.asarray(a, float); b = np.asarray(b, float)
    hc, ha = 0.5 * (a + b), 0.5 * (b - a)
    M = coef.shape[1] - 1
    m = np.arange(M + 1)
    phase_m = 2.0 * (-1j) ** m
    out = np.empty(u.shape, complex)
    for s in range(0, u.size, chunk):
        uu = u[s:s + chunk]
        w = TWO_PI * uu[:, None] * ha[None, :]                 # (U,K)
        aw = np.abs(w)
        sg = np.sign(w)
        acc = np.zeros(w.shape, complex)
        for mm in m:
            jm = spherical_jn(mm, aw) * (sg ** mm if mm % 2 else 1.0)
            acc += (coef[:, mm] * phase_m[mm])[None, :] * jm
        out[s:s + chunk] = (acc * ha[None, :] * np.exp(-1j * TWO_PI * uu[:, None] * hc[None, :])).sum(1)
    return out


def _cell_coef(poly, l, r, M):
    """Legendre coefficients (degree M) of a polynomial callable on [l, r]."""
    t, _ = npleg.leggauss(M + 1)
    x = 0.5 * (l + r) + 0.5 * (r - l) * t
    return npleg.legfit(t, poly(x), M)


def _stencil(N, i, M):
    """Start index of an (M+1)-point consecutive stencil in 0..N-1 containing i and i+1."""
    k = min(M, N - 1)
    s = i - (k - 1) // 2
    return max(0, min(s, N - 1 - k)), k


_MAPCACHE = {}


def _lagrange_matrix(nodes, z):
    """L[g, m] = ell_m(z_g) for the Lagrange basis on `nodes`."""
    L = np.ones((z.size, nodes.size))
    for m in range(nodes.size):
        for q in range(nodes.size):
            if q != m:
                L[:, m] *= (z - nodes[q]) / (nodes[m] - nodes[q])
    return L


def _cell_map(M, k, j):
    """Matrix taking the k+1 stencil values (stencil = cell nodes -j..k-j, unit spacing) to the
    degree-M Legendre coefficients of their interpolant on the unit cell [0, 1]."""
    key = (M, k, j)
    if key not in _MAPCACHE:
        t, _ = npleg.leggauss(M + 1)
        Pinv = np.linalg.inv(npleg.legvander(t, M))
        L = _lagrange_matrix(np.arange(k + 1, dtype=float) - j, 0.5 * (1 + t))
        _MAPCACHE[key] = Pinv @ L
    return _MAPCACHE[key]


def piece_cells(xs, ys, b0, b1, M):
    """Cells and Legendre coefficients of the local degree-M interpolant on one smooth piece.

    xs, ys : samples lying in [b0, b1] (sorted, uniform spacing). b0/b1 may be off-grid
    (estimated breaks): the partial end cells use the nearest one-sided stencil (extrapolation).
    The interpolant is continuous at every sample node (each cell stencil contains both
    cell endpoints that are samples). Regular cells are vectorised by stencil offset.
    """
    xs = np.asarray(xs, float); ys = np.asarray(ys)
    N = xs.size
    A, B, C = [], [], []
    def slow(l, r, i):
        s, k = _stencil(N, i, M)
        if k == 0:
            poly = lambda x, v=ys[0]: np.full_like(x, v, dtype=complex if np.iscomplexobj(ys) else float)
        else:
            poly = BarycentricInterpolator(xs[s:s + k + 1], ys[s:s + k + 1])
        A.append(np.array([l])); B.append(np.array([r])); C.append(_cell_coef(poly, l, r, M)[None, :])
    if b0 < xs[0] - 1e-15:
        slow(b0, xs[0], 0)
    if N >= 2:
        i = np.arange(N - 1)
        k = min(M, N - 1)
        s = np.clip(i - (k - 1) // 2, 0, N - 1 - k)
        j = i - s
        for jj in np.unique(j):
            sel = i[j == jj]
            W = _cell_map(M, k, int(jj))                     # (M+1, k+1)
            idx = (sel - jj)[:, None] + np.arange(k + 1)[None, :]
            A.append(xs[sel]); B.append(xs[sel + 1]); C.append(ys[idx] @ W.T)
    if b1 > xs[-1] + 1e-15:
        slow(xs[-1], b1, N - 2 if N >= 2 else 0)
    A = np.concatenate(A); B = np.concatenate(B); C = np.vstack(C)
    o = np.argsort(A, kind='stable')
    return A[o], B[o], C[o]


# ----------------------------------------------------------------------------- detection
def _extrap(xs, ys, idx, xq):
    return BarycentricInterpolator(xs[idx], ys[idx])(xq)


def eno_scores(v, M):
    """ENO smoothness score per cell (v_i, v_{i+1}) of a uniform sequence v:
    min over all (M+2)-point windows containing both i and i+1 of |Delta^{M+1} v|.
    Every window around a jump cell straddles the jump (score ~ |J|); any other cell whose
    smooth piece holds >= M+2 values has a clean window (score ~ h^{M+1}|f^{(M+1)}|)."""
    n = v.size
    D = np.abs(np.diff(v, M + 1))                 # window s covers s..s+M+1
    sc = np.full(n - 1, np.inf)
    for off in range(M + 1):                     # s = i - off
        i = np.arange(n - 1)
        s = i - off
        ok = (s >= 0) & (s < D.size)
        sc[ok] = np.minimum(sc[ok], D[s[ok]])
    return sc


def jump_scores(x, y, M):
    return eno_scores(np.asarray(y), M)


def local_ratio_flags(sc, M, kappa, floor_abs):
    """Two-stage local test on ENO scores.
    Stage 1 (candidates): sc_i > kappa * median of the neighbourhood (robust to other jumps).
    Stage 2 (confirm):    sc_i > kappa * max of the neighbourhood with all candidates removed
                          (robust to zeros of f^{(M+1)}, which make the median too small).
    Neighbourhood: cells i-W..i+W, W = M+2."""
    n = sc.size
    W = M + 2
    def nbr(i, excl):
        idx = np.r_[max(0, i - W):i, i + 1:min(n, i + 1 + W)]
        idx = idx[~excl[idx]]
        v = sc[idx]
        return v[np.isfinite(v)]
    # stage 1, vectorised: median of the finite neighbours (self excluded, windows truncated)
    from numpy.lib.stride_tricks import sliding_window_view as _sw
    scf = np.where(np.isfinite(sc), sc, np.nan).astype(float)
    P = np.r_[np.full(W, np.nan), scf, np.full(W, np.nan)]
    win = _sw(P, 2 * W + 1).copy()
    win[:, W] = np.nan
    good = np.isfinite(scf) & (np.nan_to_num(scf, nan=-np.inf) > floor_abs)
    cand = np.zeros(n, bool)
    idx = np.where(good)[0]
    if idx.size:
        with np.errstate(all='ignore'):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                med = np.nanmedian(win[idx], axis=1)
        cand[idx] = np.isnan(med) | (scf[idx] > kappa * med)
    fl = []
    for i in np.where(cand)[0]:
        v = nbr(i, cand)
        if v.size == 0 or sc[i] > kappa * np.max(v):
            fl.append(i)
    return np.array(fl, int)


DETECT_LOG = []                                   # order-M-only detections (diagnostics)


def local_ratio(sc, i, M):
    """Score of cell i relative to the median of its neighbourhood (ranking key)."""
    W = M + 2
    v = sc[np.r_[max(0, i - W):i, i + 1:min(sc.size, i + 1 + W)]]
    v = v[np.isfinite(v)]
    return sc[i] / max(np.median(v) if v.size else 0.0, 1e-300)


def detect_jumps(x, y, M, kappa=30.0, floor=1e-9):
    """Return list of cell indices i whose gap (x_i, x_{i+1}) contains a jump or a kink.

    Two difference orders are tested and their flags united. The order-(M+1) score of a kink
    vanishes when the kink sits at a fraction k/M of a gap (k = 1..M-1): an odd derivative of the
    B-spline kernel of Delta^{M+1} has a zero there. The order-M score vanishes at k/(M-1) instead,
    and the two sets are disjoint because gcd(M, M-1) = 1 (Lemma in Appendix A)."""
    scale = max(np.max(np.abs(y)), 1e-300)
    y = np.asarray(y)

    def flank(i, q):
        """Largest |Delta^q y| over the clean windows of q+1 samples that flank gap i
        (x_{i-q..i} and x_{i+1..i+1+q}); None if neither fits in the data."""
        w = [abs(np.diff(y[i - q:i + 1], q)[0])] if i - q >= 0 else []
        w += [abs(np.diff(y[i + 1:i + 2 + q], q)[0])] if i + 1 + q <= y.size - 1 else []
        return max(w) if w else None

    sc = jump_scores(x, y, M)                     # order M+1
    # resolvability: a break needs >= M+1 samples on each side (else its piece is unresolvable)
    # (NaN: these cells are neither flagged nor used as background for their neighbours)
    sc[:M] = np.nan; sc[max(0, sc.size - M):] = np.nan
    flag = [int(i) for i in local_ratio_flags(sc, M, kappa, floor * scale)]
    # a flag must stand out against at least one of its flanking windows (a break next to
    # another one has a clean window on its other side); flags that do not are artefacts of
    # score plateaus, e.g. around a kink at a zero of the order-(M+1) kernel
    def flank_min(i, q):
        w = [abs(np.diff(y[i - q:i + 1], q)[0])] if i - q >= 0 else []
        w += [abs(np.diff(y[i + 1:i + 2 + q], q)[0])] if i + 1 + q <= y.size - 1 else []
        return min(w) if w else None
    flag = [i for i in flag if flank_min(i, M + 1) is None or sc[i] > kappa * flank_min(i, M + 1)]
    # ranking: flags confirmed against both flanking windows first, then order-(M+1) flags
    # (the original rule), then the size of the score
    key = {}
    for i in flag:
        fw = flank(i, M + 1)
        key[i] = (fw is None or sc[i] > kappa * fw, 1, sc[i])
    # order M covers the zeros of the order-(M+1) kernel. Its min-scores have plateaus (one
    # window with a zero of f^{(M)} serves many cells), so an order-M flag is kept only if it is
    # confirmed against the two clean windows that flank the gap.
    s0 = jump_scores(x, y, M - 1)
    s0[:M] = np.nan; s0[max(0, s0.size - M):] = np.nan
    for i in local_ratio_flags(s0, M, kappa, floor * scale):
        i = int(i)
        if any(abs(i - j) <= 1 for j in flag) or i < M or i + 1 + M > y.size - 1:
            continue
        fw = flank(i, M)
        if s0[i] > kappa * fw:
            flag.append(i); key[i] = (True, 0, s0[i] / max(fw, 1e-300))
            DETECT_LOG.append(('point', i))
    # kinks at (or extremely close to) a node x_k: every gap has a clean window (one ending or
    # starting at x_k), so gap scores miss them. Node score: min over windows with x_k strictly
    # inside; a kink there gives |Delta^q| = |J1| h binom(q-2, j-1) != 0. (Vectorised.)
    from numpy.lib.stride_tricks import sliding_window_view as _sw
    near = np.zeros(y.size + 2, bool)
    for j in flag:
        near[max(0, j - 1):j + 3] = True              # nodes j-1 .. j+2 border flagged gaps
    for q in (M + 1, M):
        if q < 2:
            continue
        D = np.abs(np.diff(y, q))                    # window s covers s..s+q
        k = np.arange(M + 1, y.size - 1 - M)
        k = k[(k - q >= 0) & (k + q <= y.size - 1)]
        if k.size == 0:
            continue
        sk = _sw(D, q - 1).min(axis=1)[k - q + 1]     # min over windows s = k-q+1 .. k-1
        fw = np.maximum(D[k - q], D[k])               # windows x_{k-q..k} and x_{k..k+q}
        W = M + 2                                     # background: W windows on each side
        Dp = np.r_[np.zeros(W), D, np.zeros(W)]
        mx = _sw(Dp, W).max(axis=1)                   # mx[t] = max Dp[t:t+W] = max D[t-W:t]
        refl = mx[k - q]                              # max D[k-q-W : k-q]
        refr = mx[k + 1 + W]                          # max D[k+1 : k+1+W]
        ok = (sk > floor * scale) & (sk > kappa * fw) & (sk > kappa * np.maximum(refl, refr)) & ~near[k]
        for kk, v, w in zip(k[ok], sk[ok], fw[ok]):
            kk = int(kk)
            if near[kk]:
                continue
            flag.append(kk); key[kk] = (True, 0, v / max(w, 1e-300))
            near[max(0, kk - 1):kk + 3] = True
            DETECT_LOG.append(('node', kk))
    flag = sorted(flag)
    rank = lambda j: key[j]
    # keep the strongest cell in each run of consecutive flags
    out, run = [], []
    for i in flag:
        if run and i != run[-1] + 1:
            out.append(max(run, key=rank)); run = []
        run.append(i)
    if run:
        out.append(max(run, key=rank))
    # unresolvable clusters: breaks closer than M+2 cells -> keep the strongest; the missing one
    # shows up as a large smooth-error indicator and is resolved by (local) refinement
    keep = []
    for i in sorted(out, key=rank, reverse=True):
        if all(abs(i - k) >= M + 2 for k in keep):
            keep.append(i)
    return sorted(keep), sc


# ----------------------------------------------------------------------------- SAFI
class Break:
    """A located interior break: gap [lo, hi] known to contain it, estimate c = midpoint.
    kind = 'jump' (f discontinuous) or 'kink' (f continuous, f' discontinuous; J1 = slope jump)."""
    def __init__(self, lo, hi, J, eta=0.0, nq=0, i=None, kind='jump', J1=0.0):
        self.lo, self.hi, self.J, self.eta, self.nq, self.i = lo, hi, J, eta, nq, i
        self.kind, self.J1 = kind, J1
        self.ambiguous, self.J_amb, self.d_res = False, 0.0, 0.0
        self.cluster, self.cluster_amp = False, 0.0

    @property
    def c(self):
        return 0.5 * (self.lo + self.hi)

    @property
    def delta(self):
        return 0.5 * (self.hi - self.lo)


def _side_polys(x, y, i, M):
    sL = np.arange(max(0, i - M), i + 1)
    sR = np.arange(i + 1, min(x.size, i + 2 + M))
    return (BarycentricInterpolator(x[sL], y[sL]), BarycentricInterpolator(x[sR], y[sR]))


def _kink_root(PL, PR, lo, hi, ngrid=64):
    """Root of PL - PR in [lo, hi] (continuity => kink location), or None."""
    from scipy.optimize import brentq
    t = np.linspace(lo, hi, ngrid + 1)
    d = PL(t) - PR(t)
    if np.iscomplexobj(d):
        # complex data: use the component with the larger variation
        d = d.real if np.ptp(d.real) >= np.ptp(d.imag) else d.imag
        g = (lambda z: (PL(z) - PR(z)).real) if np.ptp((PL(t) - PR(t)).real) >= np.ptp((PL(t) - PR(t)).imag) \
            else (lambda z: (PL(z) - PR(z)).imag)
    else:
        g = lambda z: PL(z) - PR(z)
    sgn = np.sign(d)
    k = np.where(sgn[:-1] * sgn[1:] <= 0)[0]
    if k.size == 0:
        return None
    a, b = t[k[0]], t[k[0] + 1]
    ga, gb = g(a), g(b)
    if ga == 0 or gb == 0 or np.sign(ga) == np.sign(gb):   # grid sign change at a zero sample
        return a if abs(ga) <= abs(gb) else b
    return brentq(g, a, b, xtol=1e-16, rtol=4 * np.finfo(float).eps)


def _kink_break(x, y, i, M, c):
    """Kink at c: continuity holds; slope jump J1; uncertainty from degree variation."""
    PL, PR = _side_polys(x, y, i, M)
    PLm, PRm = _side_polys(x, y, i, max(M - 1, 1))
    g = (x[i + 1] - x[i]) / 8
    cm = _kink_root(PLm, PRm, x[i] - g, x[i + 1] + g)
    d = 2 * abs(cm - c) if cm is not None else 0.5 * (x[i + 1] - x[i])
    d = min(d + 8 * np.finfo(float).eps * max(1.0, abs(c)), 0.5 * (x[i + 1] - x[i]))
    J1 = complex(PR.derivative(c) - PL.derivative(c))
    lo, hi = max(c - d, x[i]), min(c + d, x[i + 1])
    b = Break(lo, hi, 0.0, 0.0, 0, i, kind='kink', J1=J1)
    return b


def _kink_verified(x, y, i, M, c, f):
    """Two queries at c -/+ d: a kink has f = PL left of c and f = PR right of c.
    A jump at c' with |c'-c| > d would show a misfit ~|J1| d at one of them, so d is chosen as
    d = 4 tol/|J1| with tol ~ the one-sided extrapolation error at c. Returns (ok, d); a jump
    with |c'-c| <= d (height <= |J1| d ~ 4 tol) is indistinguishable and is accounted for in the
    error estimate by the residual term |J1| d S(u, d)."""
    PL, PR = _side_polys(x, y, i, M)
    PLm, PRm = _side_polys(x, y, i, max(M - 1, 1))
    h = x[i + 1] - x[i]
    J1 = abs(PR.derivative(c) - PL.derivative(c))
    eta = abs(PL(c) - PLm(c)) + abs(PR(c) - PRm(c)) + 1e-13 * (abs(PL(c)) + 1)
    d = min(max(4 * eta / max(J1, 1e-300), 64 * np.finfo(float).eps * max(1.0, abs(c))), h / 4)
    a, b = max(c - d, x[i]), min(c + d, x[i + 1])
    va, vb = f(np.array([a, b]))
    ok = abs(va - PL(a)) <= 0.5 * J1 * d + eta and abs(vb - PR(b)) <= 0.5 * J1 * d + eta
    return bool(ok), d


def _node_kink_evidence(y, i, M, kappa=30.0):
    """True if the flag of gap i comes from a kink at (or extremely close to) one of its nodes:
    the gap score is ordinary while a node score (windows with the node strictly inside) is
    kappa times larger. A jump inside the gap has a large gap score instead."""
    q = M + 1
    D = np.abs(np.diff(y, q))
    gap = min([D[s] for s in range(i - q + 1, i + 1) if 0 <= s < D.size] or [np.inf])
    node = 0.0
    for k in (i, i + 1):
        v = [D[s] for s in range(k - q + 1, k) if 0 <= s < D.size]
        if len(v) == q - 1:
            node = max(node, min(v))
    return node > kappa * gap


def locate(x, y, cells, M, f=None, q=0):
    """Turn detected cells into Break objects. With callable f, bisect q times per jump.
    A cell whose one-sided extrapolants intersect inside the gap is a kink: its location is the
    intersection (O(h^{M+1}/|J1|), no queries needed)."""
    brks = []
    for i in cells:
        PL, PR = _side_polys(x, y, i, M)
        lo, hi = x[i], x[i + 1]
        ck = _kink_root(PL, PR, lo, hi)
        if ck is None and _node_kink_evidence(y, i, M):
            # kink at (or within the extrapolation error of) a node: the intersection may lie
            # just outside the gap; search a slightly wider interval and clamp to the gap
            ck = _kink_root(PL, PR, lo - (hi - lo) / 8, hi + (hi - lo) / 8)
            if ck is not None:
                ck = min(max(ck, lo), hi)
        if ck is not None:
            kb = _kink_break(x, y, i, M, ck)
            # a small jump |J| < |J1| h is consistent with the same samples (ambiguity)
            tt = np.linspace(lo, hi, 33)
            kb.J_amb = float(np.max(np.abs(PL(tt) - PR(tt))))
            if f is not None and q > 0:
                ok, dv = _kink_verified(x, y, i, M, ck, f)
                if ok:
                    kb.nq, kb.d_res = 2, dv
                    brks.append(kb)
                    continue
                # not a kink: fall through to the jump branch (bisection)
            else:
                kb.ambiguous = True
                brks.append(kb)
                continue
        b = Break(lo, hi, 0.0, 0.0, 0, i)
        bisect_break(x, y, b, M, f, q if f is not None else 0)
        brks.append(b)
    return brks


def bisect_break(x, y, b, M, f, steps, tol_abs=0.0):
    """Refine an existing Break by `steps` more bisections (queries f).

    Consistency check: every query must agree with the left OR the right extrapolant. A value
    that matches neither lies between two singularities inside the gap (a cluster); bisection
    stops, the break is flagged `cluster` (to be split by local refinement) and its error term
    uses the observed middle-level amplitude. An ambiguous kink is first verified with two queries."""
    if getattr(b, 'cluster', False):
        return b
    if b.kind == 'kink':
        if not b.ambiguous:
            return b
        ok, dv = _kink_verified(x, y, b.i, M, b.c, f)
        if ok:
            b.ambiguous = False; b.nq += 2; b.d_res = dv
            return b
        b.kind, b.ambiguous, b.nq = 'jump', False, b.nq + 2
        b.lo, b.hi = x[b.i], x[b.i + 1]
    PL, PR = _side_polys(x, y, b.i, M)
    PLm, PRm = _side_polys(x, y, b.i, max(M - 1, 0))
    if steps > 0:
        # resolvability: if a one-sided stencil straddles another singularity, its extrapolant
        # and that of one degree less disagree by an amount comparable to the jump, and the
        # side test below has no meaning. Treat the gap as a cluster (local refinement).
        mid = 0.5 * (x[b.i] + x[b.i + 1])
        e_mid = abs(PL(mid) - PLm(mid)) + abs(PR(mid) - PRm(mid))
        if e_mid > 0.1 * abs(PL(mid) - PR(mid)):
            b.cluster, b.cluster_amp = True, abs(PL(mid) - PR(mid)) + e_mid
            return b
    for _ in range(steps):
        mpt = 0.5 * (b.lo + b.hi)
        v = f(np.array([mpt]))[0]
        dL, dR = abs(v - PL(mpt)), abs(v - PR(mpt))
        tol = 10 * (abs(PL(mpt) - PLm(mpt)) + abs(PR(mpt) - PRm(mpt))) + 1e-12 * (abs(v) + 1) + tol_abs
        tau = tol + 0.02 * abs(PL(mpt) - PR(mpt))
        b.nq += 1
        if dL > tau and dR > tau:
            b.cluster, b.cluster_amp = True, max(dL, dR)
            break
        if dL <= dR:
            b.lo = mpt
        else:
            b.hi = mpt
    if steps > 0 and not b.cluster:
        # end-point check: if bisection never moved an end of the gap, the node at that end
        # must belong to its own side. A node that matches the extrapolant of the OTHER side
        # means the singularity lies outside the flagged gap (wrong cell, e.g. inside a
        # cluster): treat it as a cluster so that the driver refines locally.
        for xe, ye, P, Q in ((x[b.i], y[b.i], PR, PL), (x[b.i + 1], y[b.i + 1], PL, PR)):
            if b.lo == xe or b.hi == xe:
                tol = 10 * (abs(PL(xe) - PLm(xe)) + abs(PR(xe) - PRm(xe))) + 1e-12 * (abs(ye) + 1) + tol_abs
                if abs(ye - P(xe)) <= tol + 0.02 * abs(PL(xe) - PR(xe)) and abs(PL(xe) - PR(xe)) > 2 * tol:
                    b.cluster, b.cluster_amp = True, abs(PL(xe) - PR(xe))
    b.J = complex(PR(b.c) - PL(b.c))
    b.eta = abs(PL(b.c) - PLm(b.c)) + abs(PR(b.c) - PRm(b.c))
    return b


class SAFI:
    """Fit once from data, then evaluate F(u) and its error estimate on any frequency set."""

    def __init__(self, x, y, M=6, mode='point', f=None, q=0, breaks=None, kappa=30.0, sigma=0.0,
                 continuous=False):
        x = np.asarray(x, float); y = np.asarray(y)
        self.continuous = continuous
        self.x, self.y, self.M, self.mode, self.sigma = x, y, M, mode, float(sigma)
        self.p0, self.p1 = x[0], x[-1]
        if mode == 'known':
            self.breaks = []
            for c in sorted(breaks or []):
                i = int(np.searchsorted(x, c) - 1)
                PL, PR = _side_polys(x, y, i, M)
                self.breaks.append(Break(c, c, complex(PR(c) - PL(c))))
        elif mode == 'none':
            self.breaks = []
        else:
            # noise floor for (M+1)-th differences: 6 sigma sqrt(binom(2M+2, M+1))
            from math import comb
            fl = max(1e-9, 6 * self.sigma * np.sqrt(comb(2 * M + 2, M + 1)) / max(np.max(np.abs(y)), 1e-300))
            cells, _ = detect_jumps(x, y, M, kappa=kappa, floor=fl)
            self.breaks = locate(x, y, cells, M, f=f if mode == 'adaptive' else None, q=q)
        self.nquery = sum(b.nq for b in self.breaks)
        self._build()

    def _pieces(self, M):
        cuts = [self.p0] + [b.c for b in self.breaks] + [self.p1]
        A, B, C = [], [], []
        nb = len(self.breaks)
        for k in range(len(cuts) - 1):
            b0, b1 = cuts[k], cuts[k + 1]
            sel = np.ones(self.x.size, bool)
            if k > 0:                      # right of the previous gap
                sel &= self.x >= self.breaks[k - 1].hi - 1e-15
            if k < nb:                     # left of the next gap
                sel &= self.x <= self.breaks[k].lo + 1e-15
            a, b, c = piece_cells(self.x[sel], self.y[sel], b0, b1, M)
            A.append(a); B.append(b); C.append(c)
        return np.concatenate(A), np.concatenate(B), np.vstack(C)

    def _build(self):
        self.A, self.B, self.C = self._pieces(self.M)
        # error-estimator proxy: degree M+1 interpolant on the same cells
        A2, B2, C2 = self._pieces(self.M + 1)
        self.D = (A2, B2, C2)

    def __call__(self, u):
        return leg_ft_cells(self.A, self.B, self.C, u)

    def eval_fast(self, u, msp=12):
        """Same F(u) via M+1 type-2 NUFFTs on the regular cells (uniform centres, width h)
        + dense sums on the few partial cells next to breaks. Cost O((M+1)(n log n + n_u))."""
        from nufft import nufft2
        u = np.atleast_1d(np.asarray(u, float))
        h = self.x[1] - self.x[0]
        n = self.x.size - 1
        k = np.rint((self.A - self.x[0]) / h).astype(int)
        reg = (np.abs(self.B - self.A - h) < 1e-12 * h) & (np.abs(self.A - (self.x[0] + k * h)) < 1e-12 * h)
        out = leg_ft_cells(self.A[~reg], self.B[~reg], self.C[~reg], u) if (~reg).any() else np.zeros(u.size, complex)
        Cf = np.zeros((n, self.M + 1), complex)
        Cf[k[reg]] = self.C[reg]
        th = TWO_PI * u * h
        w = np.pi * u * h
        aw, sg = np.abs(w), np.sign(w)
        acc = np.zeros(u.size, complex)
        for m in range(self.M + 1):
            jm = spherical_jn(m, aw) * (sg ** m if m % 2 else 1.0)
            acc += 2 * (-1j) ** m * jm * nufft2(Cf[:, m], th, msp=msp)
        return out + acc * (h / 2) * np.exp(-1j * TWO_PI * u * (self.x[0] + h / 2))

    # ------------------------------------------------------------- a-posteriori estimate
    def noise_term(self, u, nprobe=8, seed=0):
        """3 x RMS of F[p(noise)] over random probes: the map y -> F[p](u) is linear."""
        rg = np.random.default_rng(seed)
        acc = np.zeros(np.size(u))
        for _ in range(nprobe):
            z = self.sigma * rg.normal(size=self.y.size)
            A, B, C = self._pieces_from(z, self.M)
            acc += np.abs(leg_ft_cells(A, B, C, u)) ** 2
        return 3 * np.sqrt(acc / nprobe)

    def _pieces_from(self, yy, M):
        y0 = self.y
        self.y = yy
        try:
            return self._pieces(M)
        finally:
            self.y = y0

    def refine(self, k, f, steps):
        bisect_break(self.x, self.y, self.breaks[k], self.M, f, steps)
        self.nquery = sum(b.nq for b in self.breaks)
        self._build()

    def cell_indicators(self):
        """Per-cell L1 norms of d = p_{M+1} - p_M and of d' (cells as in self.A, self.B)."""
        A2, B2, C2 = self.D
        Cd = C2.copy(); Cd[:, :self.M + 1] -= self.C
        t, w = npleg.leggauss(self.M + 4)
        ha = 0.5 * (B2 - A2)
        V = npleg.legvander(t, self.M + 1)
        dV = np.stack([npleg.legval(t, npleg.legder(np.eye(self.M + 2)[j])) for j in range(self.M + 2)], 1)
        L1k = (w[:, None] * np.abs(V @ Cd.T)).sum(0) * ha
        L1dk = (w[:, None] * np.abs(dV @ Cd.T)).sum(0)
        return A2, B2, L1k, L1dk

    def error_estimate(self, u, parts=False):
        """e(u) = e_smooth(u) + sum_breaks e_loc(u).

        e_smooth: |F[p_{M+1}] - F[p_M]| is replaced by the u-explicit majorant
                  min( sum_k int|d_k|, sum_k int|d_k'| / (2 pi |u|) ), d = p_{M+1}-p_M,
                  valid because d is continuous at every sample node.
        e_loc   : (|J| + eta) * S(u, delta),  S = sin(pi|u|delta)/(pi|u|) for |u|delta<=1/2,
                  1/(pi|u|) beyond  (worst case over the unresolved gap).
        """
        u = np.atleast_1d(np.asarray(u, float))
        A2, B2, C2 = self.D
        Cd = C2.copy(); Cd[:, :self.M + 1] -= self.C
        t, w = npleg.leggauss(self.M + 4)
        ha = 0.5 * (B2 - A2)
        V = npleg.legvander(t, self.M + 1)                        # (q, M+2)
        dV = np.stack([npleg.legval(t, npleg.legder(np.eye(self.M + 2)[j])) for j in range(self.M + 2)], 1)
        d = np.abs(V @ Cd.T)                                       # (q, K)
        dd = np.abs(dV @ Cd.T) / ha[None, :]
        L1 = float((w[:, None] * d * ha[None, :]).sum())
        L1d = float((w[:, None] * dd * ha[None, :]).sum())
        au = np.maximum(np.abs(u), 1e-300)
        es = np.minimum(L1, L1d / (TWO_PI * au))
        # floating-point floor of the evaluation (u-independent)
        es = es + np.finfo(float).eps * np.sqrt(self.C.shape[0]) * float((np.abs(self.C).sum(1) * 2 * 0.5 * (self.B - self.A)).sum())
        if self.sigma > 0:
            es = es + self.noise_term(u)
        eloc = np.zeros((len(self.breaks), u.size))
        for k, b in enumerate(self.breaks):
            if b.delta == 0:
                continue
            if b.kind == 'kink':
                # |int_c^{c^} J1 (x-c) e^{-2 pi i u x}| <= |J1| min(delta^2/2, delta/(pi|u|))  (x2: interval of c)
                eloc[k] = abs(b.J1) * np.minimum(2 * b.delta ** 2, 2 * b.delta / (np.pi * au))
                if b.d_res > 0 and not self.continuous:      # residual after verification
                    xa = np.pi * au * b.d_res
                    eloc[k] += abs(b.J1) * b.d_res * np.where(xa <= np.pi / 2, np.sin(xa) / (np.pi * au), 1.0 / (np.pi * au))
                if b.ambiguous and not self.continuous:
                    hg = 0.5 * (self.x[b.i + 1] - self.x[b.i])
                    xa = np.pi * au * hg
                    eloc[k] += b.J_amb * np.where(xa <= np.pi / 2, np.sin(xa) / (np.pi * au), 1.0 / (np.pi * au))
                continue
            x_ = np.pi * au * b.delta
            S = np.where(x_ <= np.pi / 2, np.sin(x_) / (np.pi * au), 1.0 / (np.pi * au))
            eloc[k] = (abs(b.J) + b.eta + (b.cluster_amp if b.cluster else 0.0)) * S
        if parts:
            return es, eloc
        return es + eloc.sum(0)


# ----------------------------------------------------------------------------- adaptive driver
class Counted:
    """Memoised callable that counts distinct evaluation points."""
    def __init__(self, f):
        self.f, self.cache = f, {}

    def __call__(self, x):
        x = np.atleast_1d(np.asarray(x, float))
        new = [v for v in x.tolist() if v not in self.cache]
        if new:
            vals = self.f(np.array(new))
            self.cache.update(zip(new, vals.tolist()))
        return np.array([self.cache[v] for v in x.tolist()])

    @property
    def n(self):
        return len(self.cache)


def _band_ratio(e, F, u):
    lab = np.floor(np.log2(np.maximum(np.abs(u), 1e-300))).astype(int)
    r = []
    for k in np.unique(lab):
        s = lab == k
        r.append(np.linalg.norm(e[s]) / max(np.linalg.norm(F[s]), 1e-300))
    return np.array(r), np.unique(lab)


def safi_auto(f, p0, p1, u, eps, M=6, n0=64, nmax=2 ** 15, maxit=60, log=None):
    """Certified-adaptive SAFI: returns (model, F_hat, e_hat, stats).

    Stops when the a-posteriori estimate gives band relative error <= eps on every dyadic
    band of the target set u. Each iteration fixes the dominant error source predictively:
      location term of break k  ~ delta_k     -> bisect ceil(log2(ratio/(eps/2))) times
      smooth term               ~ h^{M+1}     -> grow n by (ratio/(eps/2))^{1/(M+1)}, power of 2
    """
    fc = Counted(f)
    n = n0
    u = np.asarray(u, float)
    hist = []
    S = None
    prev_q = {}
    for it in range(maxit):
        if S is None or S.x.size != n + 1:
            x = np.linspace(p0, p1, n + 1)
            S = SAFI(x, fc(x), M, mode='point')
            for k, b in enumerate(S.breaks):          # re-apply previous depth (cached points)
                q0 = max(0, prev_q.get(round(b.c, 6) // 1e-3, 0) - 1)
                if q0:
                    bisect_break(S.x, S.y, b, M, fc, q0)
            S._build()
        Fh = S(u)
        es, el = S.error_estimate(u, parts=True)
        tot = es + el.sum(0)
        rt, _ = _band_ratio(tot, Fh, u)
        hist.append(dict(it=it, n=n, nf=fc.n, ratio=float(rt.max()), nb=len(S.breaks)))
        if log:
            log(hist[-1])
        if rt.max() <= eps:
            break
        rs = _band_ratio(es, Fh, u)[0].max()
        rl = [_band_ratio(el[k], Fh, u)[0].max() for k in range(len(S.breaks))]
        stalled = len(hist) >= 2 and hist[-2]['n'] == n and rt.max() > 0.9 * hist[-2]['ratio']
        if rl and max(rl) >= rs and not stalled:
            k = int(np.argmax(rl))
            steps = int(np.ceil(np.log2(max(rl[k] / (eps / 2), 2.0))))
            S.refine(k, fc, steps)
            prev_q = {round(b.c, 6) // 1e-3: b.nq for b in S.breaks}
        else:
            if n >= nmax:
                break
            g = (rs / (eps / 2)) ** (1.0 / (M + 1))
            n = int(min(nmax, n * 2 ** min(2, max(1, int(np.ceil(np.log2(g)))))))
            prev_q = {round(b.c, 6) // 1e-3: b.nq for b in S.breaks}
    Fh = S(u)
    eh = S.error_estimate(u)
    cert = bool(_band_ratio(eh, Fh, u)[0].max() <= eps)
    return S, Fh, eh, dict(nf=fc.n, n=n, it=len(hist), hist=hist, nq=S.nquery, nb=len(S.breaks),
                           certified=cert)


# ----------------------------------------------------------------------------- baselines
def riemann_sum(x, y, u, trap=True, block=2 ** 22):
    """Frozen-node exponential sum on the samples (= type-3 NUFFT target / sampling sum).
    Dense, blocked over (u, x) so memory stays ~ block complex numbers."""
    h = x[1] - x[0]
    wts = np.full(x.size, h)
    if trap:
        wts[0] = wts[-1] = h / 2
    wy = wts * y
    u = np.atleast_1d(u)
    out = np.zeros(u.size, complex)
    bx = min(x.size, block)
    bu = max(1, block // bx)
    for s in range(0, u.size, bu):
        for t in range(0, x.size, bx):
            out[s:s + bu] += np.exp(-1j * TWO_PI * np.outer(u[s:s + bu], x[t:t + bx])) @ wy[t:t + bx]
    return out
