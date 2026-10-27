"""Exp 16: k-space of MRI phantoms given as partial-volume pixel images (the inverse-crime setting).

Phantoms (analytic, so the exact k-space is known):
  SL  : Shepp-Logan structure (Toft's intensities) with ellipses moved where the original ones
        cross or touch (crossing interfaces create corners, outside the smooth-curve model);
  RND : random images with three smooth star-shaped inclusions, one of them containing another;
  both multiplied by a smooth coil sensitivity.
Data: exact pixel averages (partial-volume / fuzzy model) at n x n.
Task 1, k-space accuracy on the Cartesian grid u, v = -N/2..N/2-1 for N = n (same matrix) and
  N = 2n, 4n (acquisition finer than the phantom), relative error per radial ring.
Methods: raster DFT of the pixel image (standard), DFT corrected by the pixel transfer function,
  bilinear upsampling to N then DFT (standard for N > n), exact transform of the box model,
  edge-unaware smooth reconstruction, SAFI-2D (multi-interface conservative fit).
Task 2, inverse crime: zero-filled inverse DFT of N = n data. Raster data return the pixel image
  itself (no truncation artefact); exact data show the true Gibbs ringing.
"""
import sys, json, time
sys.path.insert(0, '.')
import numpy as np
from bench2d import Phase
from bench2d_multi import MultiCurveImage, shepp_logan
from safi2d_multi import MultiCurveSAFI2D, fast_cartesian
import safi2d as s2

TWO_PI = 2 * np.pi

SL_SEP = dict(move={4: (0, .5), 5: (0, .15), 6: (0.04, -.4), 7: (-.1, -.62), 8: (.1, -.62)}, drop=(9,))


def coil(rng=None):
    """Smooth real coil sensitivity: 1 + low-order spatial variation (a few cycles at most)."""
    rng = np.random.default_rng(0) if rng is None else rng
    a = [1.0 + 0j] + list(0.25 * (rng.normal(size=3) + 1j * rng.normal(size=3)))
    b = [[0, 0]] + [list(rng.uniform(-1.0, 1.0, 2) + 1j * TWO_PI * rng.uniform(-0.8, 0.8, 2)) for _ in range(3)]
    return Phase(np.array(a), np.array(b, complex))


def sl_phantom():
    from bench2d_multi import SHEPP_LOGAN
    import bench2d_multi as bm
    E = [list(e) for e in SHEPP_LOGAN]
    E[1][1:3] = [.62, .83]                      # thicker skull (the original is 1.8 pixels wide at n = 128)
    E[4][1:3] = [.21, .21]                      # ellipse 5 made circular and moved up (crossed 4 and 6)
    E[7][1:3] = [.07, .04]; E[8][1:3] = [.05, .05]
    for k, (x0, y0) in SL_SEP['move'].items():
        E[k][3:5] = [x0, y0]
    E = [e for k, e in enumerate(E) if k not in SL_SEP['drop']]
    saved = bm.SHEPP_LOGAN
    bm.SHEPP_LOGAN = [tuple(e) for e in E]
    try:
        img = shepp_logan(coil=coil())
    finally:
        bm.SHEPP_LOGAN = saved
    return img


def raster_dft(A, k):
    """h^2 sum A_ij e^{-2 pi i (u x_i + v y_j)} on the integer grid k x k (pixel centres)."""
    n = A.shape[0]
    Fh = np.fft.fft2(A)
    km = np.mod(k, n)
    s = np.exp(-1j * np.pi * k / n)
    return Fh[km][:, km] * np.outer(s, s) / n ** 2


def upsample_dft(A, N, k):
    """Bilinear upsampling of the pixel image to N x N, then raster DFT (common way to simulate a
    finer acquisition from a coarse phantom)."""
    from scipy.ndimage import zoom
    B = zoom(A, N / A.shape[0], order=1, mode='nearest', grid_mode=True)
    return raster_dft(B, k)


def rings(N, n, edges):
    k = np.arange(-N // 2, N // 2)
    R = np.hypot(*np.meshgrid(k, k, indexing='ij'))
    return [(R >= lo) & (R < hi) for lo, hi in zip(edges[:-1], edges[1:])]


def rel(F, G, m):
    return float(np.linalg.norm((F - G)[m]) / np.linalg.norm(F[m]))


def run(img, n, Ns, tag):
    A = img.pixel_averages(n)
    t0 = time.time()
    S = MultiCurveSAFI2D(A, P=4)
    ok = S.fit_all()
    fit_s = time.time() - t0
    out = dict(tag=tag, n=n, curves=len(img.curves), fitted=len(S.parts) if ok else 0, overlap=S.overlap if ok else None,
               fit_s=fit_s, acc={})
    for N in Ns:
        k = np.arange(-N // 2, N // 2)
        U, V = np.meshgrid(k.astype(float), k.astype(float), indexing='ij')
        R = np.hypot(U, V).ravel()
        edges = [0, n / 16, n / 8, n / 4, n / 2, n, 2 * n, N / 2 * 1.4143]
        edges = sorted(set([e_ for e_ in edges if e_ <= N / 2 * 1.4143]))
        if N == n:
            sel = np.arange(N * N)
        else:                      # finer acquisition: 1024 random grid points per ring
            rr = np.random.default_rng(N)
            sel = np.concatenate([rr.choice(np.where((R >= lo) & (R < hi))[0], min(1024, int(((R >= lo) & (R < hi)).sum())), replace=False)
                                  for lo, hi in zip(edges[:-1], edges[1:])])
        t1 = time.time()
        F = img.ft(U.ravel()[sel], V.ravel()[sel])
        t2 = time.time()
        G = fast_cartesian(S, N).ravel()[sel]          # whole N x N grid in seconds (NUFFT)
        t3 = time.time()
        sx = np.sinc(k / n)
        D = raster_dft(A, k).ravel()[sel]
        Sx = np.outer(sx, sx).ravel()[sel]
        meth = dict(raster=D, box=D * Sx, safi=G)
        if N <= n:
            meth['raster_deconv'] = D / Sx
        else:
            meth['upsampled'] = upsample_dft(A, N, k).ravel()[sel]
        if N == n:
            meth['smooth'] = s2.SAFI2D(A, P=4, edges=False)(U.ravel(), V.ravel())
        rg = [(R[sel] >= lo) & (R[sel] < hi) for lo, hi in zip(edges[:-1], edges[1:])]
        allm = np.ones(sel.size, bool)
        out['acc'][N] = dict(edges=edges, total={m: rel(F, G_, allm) for m, G_ in meth.items()},
                             rings={m: [rel(F, G_, r) if r.any() else None for r in rg] for m, G_ in meth.items()},
                             time_truth_s=t2 - t1, time_safi_s=t3 - t2, npts=int(sel.size))
        print(tag, n, N, {m: '%.1e' % v for m, v in out['acc'][N]['total'].items()}, 'safi %.0fs truth %.0fs' % (t3 - t2, t2 - t1), flush=True)
        F = F.reshape(N, N) if N == n else F
        meth = {m: (G_.reshape(N, N) if N == n else G_) for m, G_ in meth.items()}
        if N == n:
            # inverse crime: zero-filled reconstruction on the n x n grid at pixel centres
            x = (np.arange(n) + 0.5) / n
            E = np.exp(1j * TWO_PI * np.outer(x, k))           # f(x_i) ~ sum_k F(k) e^{2 pi i k x_i}

            def recon(Fk):
                return np.real(E @ Fk @ E.T)
            Rt = recon(F)
            out['crime'] = {m: dict(vs_truth_recon=float(np.linalg.norm(recon(G_) - Rt) / np.linalg.norm(Rt)),
                                     vs_pixels=float(np.linalg.norm(recon(G_) - A) / np.linalg.norm(A)))
                            for m, G_ in meth.items()}
            out['crime']['exact'] = dict(vs_truth_recon=0.0, vs_pixels=float(np.linalg.norm(Rt - A) / np.linalg.norm(A)))
            print(tag, 'crime', {m: '%.1e/%.1e' % (v['vs_truth_recon'], v['vs_pixels']) for m, v in out['crime'].items()}, flush=True)
            np.savez_compressed(f'results/exp16_{tag}_n{n}_recon.npz', A=A, Rt=Rt, Rr=recon(meth['raster']), Rs=recon(meth['safi']))
    return out


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'sl'
    import os
    path = 'results/exp16_mri.json'
    res = json.load(open(path)) if os.path.exists(path) else {}
    if which == 'sl':
        img = sl_phantom()
        for n, Ns in ((256, (256, 512, 1024)),):
            res[f'SL_n{n}'] = run(img, n, Ns, 'SL')
            json.dump(res, open(path, 'w'), indent=1)
    else:
        for seed in range(1, 6):
            img = MultiCurveImage.random(np.random.default_rng(100 + seed), ninc=3, nested=1)
            res[f'RND{seed}_n128'] = run(img, 128, (128, 512), f'RND{seed}')
            json.dump(res, open(path, 'w'), indent=1)
