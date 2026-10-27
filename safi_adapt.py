"""Locally refined SAFI: piecewise-uniform sampling (segments), h-adaptive near singular clusters.

The domain is a chain of segments [a_s, b_s], each with its own uniform grid; F = sum_s F_s and
e = sum_s e_s.  Segment ends lie in smooth regions, where one-sided stencils are accurate, so
no coupling is needed.  Refinement uses Doerfler marking on per-cell smooth-error indicators:
the marked cells (plus a buffer of M+3 cells) become a new segment with half the spacing.
Jump locations found by bisection are remembered and re-used on refined grids.
"""
import numpy as np
from safi import SAFI, Counted, _band_ratio, bisect_break


class Composite:
    def __init__(self, segs, fc, M, memory):
        self.segs, self.M = segs, M
        self.parts = []
        for a, b, n in segs:
            x = np.linspace(a, b, n + 1)
            S = SAFI(x, fc(x), M, mode='point')
            for br in S.breaks:                       # re-use earlier bisection results
                for (lo, hi, kind, amb, dres, nq) in memory:
                    if br.lo - 1e-15 <= lo and hi <= br.hi + 1e-15 and (hi - lo) < (br.hi - br.lo):
                        br.lo, br.hi, br.kind, br.ambiguous, br.d_res = lo, hi, kind, amb, dres
                        bisect_break(S.x, S.y, br, M, fc, 0)
                        break
            S._build()
            self.parts.append(S)

    def __call__(self, u):
        return sum(S(u) for S in self.parts)

    def error_parts(self, u):
        es = 0
        el = []
        for s, S in enumerate(self.parts):
            e1, e2 = S.error_estimate(u, parts=True)
            es = es + e1
            for k in range(e2.shape[0]):
                el.append(((s, k), e2[k]))
        return es, el

    @property
    def breaks(self):
        return [b for S in self.parts for b in S.breaks]

    @property
    def nquery(self):
        return sum(S.nquery for S in self.parts)


def _refine_segments(comp, M, frac=0.6, buf=None, force=None):
    """Doerfler marking on w_k = L1_k/sum L1 + L1d_k/sum L1d over all cells of all segments.
    force: list of (segment, x) positions whose cells must be refined (clusters); if given,
    only those are marked."""
    buf = M + 3 if buf is None else buf
    info = []
    for s, S in enumerate(comp.parts):
        A, B, L1, L1d = S.cell_indicators()
        info.append((s, A, L1, L1d))
    T1 = sum(i[2].sum() for i in info); T2 = sum(i[3].sum() for i in info)
    allw = np.concatenate([i[2] / T1 + i[3] / T2 for i in info])
    order = np.argsort(allw)[::-1]
    cum = np.cumsum(allw[order]) / allw.sum()
    nmark = int(np.searchsorted(cum, frac) + 1)
    thr = allw[order[nmark - 1]]
    new = []
    for (s, A, L1, L1d), (a, b, n) in zip(info, comp.segs):
        h = (b - a) / n
        w = L1 / T1 + L1d / T2
        if force is not None:
            xs = np.array([xx for ss, xx in force if ss == s])
            cells = np.unique(np.clip(np.floor((xs - a) / h).astype(int), 0, n - 1)) if xs.size else np.array([], int)
        else:
            cells = np.unique(np.clip(np.floor((A[w >= thr] - a) / h + 1e-9).astype(int), 0, n - 1))
        if cells.size == 0:
            new.append((a, b, n)); continue
        if cells.size > 0.5 * n:
            new.append((a, b, 2 * n)); continue
        runs, cur = [], [max(0, cells[0] - buf), min(n, cells[0] + 1 + buf)]
        for c in cells[1:]:
            lo, hi = max(0, c - buf), min(n, c + 1 + buf)
            if lo <= cur[1]:
                cur[1] = max(cur[1], hi)
            else:
                runs.append(cur); cur = [lo, hi]
        runs.append(cur)
        pos = 0
        minc = 2 * (M + 2)
        for lo, hi in runs:
            lo = max(lo, pos)
            if hi <= pos:
                continue
            if lo - pos < minc:            # too small a remainder: absorb into the patch
                lo = pos
            if n - hi < minc:
                hi = n
            if lo > pos:
                new.append((a + pos * h, a + lo * h, lo - pos))
            new.append((a + lo * h, a + hi * h, 2 * (hi - lo)))
            pos = hi
        if pos < n:
            new.append((a + pos * h, b, n - pos))
    return new


def safi_auto_local(f, p0, p1, u, eps, M=6, n0=64, maxit=120, hmin=1e-9, s_min=0.0, log=None):
    """Certified-adaptive SAFI with local refinement.

    Certification is relative to the class with singularity separation >= s_min: every jump is
    bisected (with consistency checks) until its gap is <= s_min, so any cluster wider than s_min
    is detected and split by local refinement."""
    fc = Counted(f)
    u = np.asarray(u, float)
    segs = [(p0, p1, n0)]
    memory = []
    hist = []
    comp = Composite(segs, fc, M, memory)

    def remember():
        for S in comp.parts:
            for b in S.breaks:
                if not b.cluster:
                    memory.append((b.lo, b.hi, b.kind, b.ambiguous, b.d_res, b.nq))

    for it in range(maxit):
        Fh = comp(u)
        es, el = comp.error_parts(u)
        tot = es + sum(e for _, e in el) if el else es
        rt = _band_ratio(tot, Fh, u)[0].max()
        hist.append(dict(it=it, nseg=len(comp.segs), nf=fc.n, ratio=float(rt),
                         hmin=min((b - a) / n for a, b, n in comp.segs)))
        if log:
            log(hist[-1])
        clusters = [(s, b) for s, S in enumerate(comp.parts) for b in S.breaks if b.cluster]
        if clusters:                                            # split clusters by local refinement
            if min((b - a) / n for a, b, n in comp.segs) < hmin:
                break
            remember()
            segs = _refine_segments(comp, M, force=[(s, b.c) for s, b in clusters])
            comp = Composite(segs, fc, M, memory)
            hist[-1]['act'] = 'split-cluster'
            continue
        if rt <= eps:
            # class guarantee: every jump gap must be <= s_min before certifying
            wide = [(S, k, b) for S in comp.parts for k, b in enumerate(S.breaks)
                    if b.kind == 'jump' and (b.hi - b.lo) > s_min > 0]
            if not wide:
                break
            for S, k, b in wide:
                S.refine(k, fc, int(np.ceil(np.log2((b.hi - b.lo) / s_min))))
            hist[-1]['act'] = 's_min'
            continue
        rs = _band_ratio(es, Fh, u)[0].max()
        rl = [_band_ratio(e, Fh, u)[0].max() for _, e in el]
        last_bisect_stalled = len(hist) >= 2 and rt > 0.9 * hist[-2]['ratio'] and hist[-2].get('act') == 'bisect'
        if rl and max(rl) >= rs and not last_bisect_stalled:
            j = int(np.argmax(rl)); (s, k) = el[j][0]
            S = comp.parts[s]; b = S.breaks[k]
            steps = int(np.ceil(np.log2(max(rl[j] / (eps / 2), 2.0))))
            S.refine(k, fc, steps)
            if not b.cluster:
                memory.append((b.lo, b.hi, b.kind, b.ambiguous, b.d_res, b.nq))
            hist[-1]['act'] = 'bisect'
        else:
            if min((b - a) / n for a, b, n in comp.segs) < hmin:
                break
            remember()
            segs = _refine_segments(comp, M)
            comp = Composite(segs, fc, M, memory)
            hist[-1]['act'] = 'refine'
    Fh = comp(u)
    es, el = comp.error_parts(u)
    eh = es + sum(e for _, e in el) if el else es
    cert = bool(_band_ratio(eh, Fh, u)[0].max() <= eps) and not any(b.cluster for b in comp.breaks)
    return comp, Fh, eh, dict(nf=fc.n, it=len(hist), hist=hist, nseg=len(comp.segs), nb=len(comp.breaks),
                              certified=cert, segs=comp.segs)
