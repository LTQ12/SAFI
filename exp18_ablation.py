"""Exp 18: which part of the method does what (ablation on the phantoms of Exp 16).

Variants, all from the same pixel averages:
  full      conservative curve fit, phase polynomials of degree P = 4, exact transform (the method);
  lowdeg    the same with P = 1 (low-order extrapolants: the conservation equation alone);
  noncons   P = 4 phases and exact transform, but each interface is a 16-mode Fourier curve fitted by
            least squares to the per-pixel line points (a piecewise linear, cell-wise conservative
            VOF/PLIC reconstruction smoothed into a curve), without the global conservative fit;
  raster4   the full model sampled at the centres of a 4n x 4n grid and transformed by the DFT
            (the recovered model, but evaluated on a grid, so sampling folds it again).
Relative k-space error per ring, 1024 random grid points per ring, N = n and N = 4n.
"""
import sys, json, time
sys.path.insert(0, '.')
import numpy as np
from bench2d_multi import MultiCurveImage
from safi2d_multi import MultiCurveSAFI2D, fast_cartesian
from safi2d import peval
from exp16_mri import sl_phantom, raster_dft

S_UP = 4


def model_points(S, s):
    """Point values of the fitted model at the centres of an (s n) x (s n) grid."""
    n, h, P = S.n, S.h, S.P
    off = (np.arange(s) + 0.5) / s - 0.5                    # local coordinates in pixel units
    XI, ZE = np.meshgrid(off, off, indexing='ij')
    out = np.zeros((s * n, s * n))
    for i in range(n):
        for j in range(n):
            xs, ys = (i + 0.5 + XI) * h, (j + 0.5 + ZE) * h
            if (i, j) in S.band_owner:
                cs = S.parts[S.band_owner[(i, j)]]
                cin, cout = cs.fits.fit(i, j, cs.rin), cs.fits.fit(i, j, cs.rout)
                ins = cs.curve.inside(xs.ravel(), ys.ravel()).reshape(xs.shape)
                v = np.where(ins, peval(cin, P, XI, ZE), peval(cout, P, XI, ZE))
            else:
                v = peval(S._own(i, j), P, XI, ZE)
            out[i * s:(i + 1) * s, j * s:(j + 1) * s] = np.real(v)
    return out


def run(img, n, tag):
    A = img.pixel_averages(n)
    res = dict(tag=tag, n=n, acc={}, fit_s={})
    models = {}
    for name, P, kw in (('full', 4, {}), ('lowdeg', 1, {}), ('noncons', 4, dict(modes=(16,), maxit=0))):
        t = time.time()
        S = MultiCurveSAFI2D(A, P=P)
        ok = S.fit_all(**kw)
        res['fit_s'][name] = time.time() - t
        models[name] = S if ok else None
        print(tag, name, 'fitted', len(S.parts) if ok else 0, '%.0fs' % res['fit_s'][name], flush=True)
    for N in (n, 4 * n):
        k = np.arange(-N // 2, N // 2)
        U, V = np.meshgrid(k.astype(float), k.astype(float), indexing='ij')
        R = np.hypot(U, V).ravel()
        edges = [e for e in [0, n / 16, n / 8, n / 4, n / 2, n, 2 * n, N / 2 * 1.4143] if e <= N / 2 * 1.4143]
        edges = sorted(set(edges))
        rr = np.random.default_rng(N)
        sel = np.concatenate([rr.choice(np.where((R >= lo) & (R < hi))[0], min(1024, int(((R >= lo) & (R < hi)).sum())), replace=False)
                              for lo, hi in zip(edges[:-1], edges[1:])])
        F = img.ft(U.ravel()[sel], V.ravel()[sel])
        G = {m: fast_cartesian(S, N).ravel()[sel] for m, S in models.items() if S is not None}
        B = model_points(models['full'], S_UP)
        kk = np.arange(-N // 2, N // 2)
        G['raster4'] = raster_dft(B, kk).ravel()[sel]
        rg = [(R[sel] >= lo) & (R[sel] < hi) for lo, hi in zip(edges[:-1], edges[1:])]
        res['acc'][N] = dict(edges=edges, rings={m: [float(np.linalg.norm((F - g)[r]) / np.linalg.norm(F[r])) for r in rg]
                                                  for m, g in G.items()})
        print(tag, N, {m: ' '.join('%.1e' % x for x in v) for m, v in res['acc'][N]['rings'].items()}, flush=True)
    return res


if __name__ == '__main__':
    out = {}
    out['SL_n256'] = run(sl_phantom(), 256, 'SL')
    json.dump(out, open('results/exp18_ablation.json', 'w'), indent=1)
    for seed in range(1, 6):
        img = MultiCurveImage.random(np.random.default_rng(100 + seed), ninc=3, nested=1)
        out[f'RND{seed}_n128'] = run(img, 128, f'RND{seed}')
        json.dump(out, open('results/exp18_ablation.json', 'w'), indent=1)
