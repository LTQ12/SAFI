"""Exp 13: pixel images with a smooth curved inclusion.

Images: oscillatory background + star-shaped inclusion rho(t) = R0 exp(sum_{j=2..4} eps_j cos(j t + phi_j))
(analytic boundary with infinitely many Fourier modes, curvature >= 0.5) carrying another phase.
Data: exact pixel averages, n in {64, 128, 256}; 5 images (3 for n = 256).
Frequencies: radial dyadic bands up to 128 n, 16 random directions per band.
Methods: pixel box; smooth fits without edges; SAFI-2D with a polygon (Exp 11 method); SAFI-2D with a
conservative Fourier curve (safi2d_curve.py); the same with the true curve (geometry oracle).
"""
import sys, json, time
sys.path.insert(0, '.')
import numpy as np
from bench2d_curve import CurveImage
from curve2d import TWO_PI
import safi2d as s2
from safi2d_curve import CurveSAFI2D

NS = [64, 128, 256]
RUN = [int(a) for a in sys.argv[1:]] or NS          # resolutions to (re)compute; others are loaded
POLY_NS = (64,)                                     # polygon baseline only at n = 64 (minutes per image)
SEEDS = [1, 2, 3, 4, 5]
PER = 16
import os
res = {int(k): v for k, v in json.load(open('results/exp13_curved.json')).items()} if os.path.exists('results/exp13_curved.json') else {}
t00 = time.time()
for n in RUN:
    edges = 2.0 ** np.arange(0, int(np.ceil(np.log2(128 * n))) + 1)
    rng = np.random.default_rng(1000 + n)
    R = np.concatenate([rng.uniform(edges[k], edges[k + 1], PER) for k in range(edges.size - 1)])
    lab = np.repeat(np.arange(edges.size - 1), PER)
    th = rng.uniform(0, TWO_PI, R.size)
    u, v = R * np.cos(th), R * np.sin(th)
    rows = []
    for seed in (SEEDS if n <= 128 else SEEDS[:3]):
        img = CurveImage.random(np.random.default_rng(seed))
        A = img.pixel_averages(n)
        F = img.ft(u, v)
        brel = lambda G: [float(np.linalg.norm((F - G)[lab == k]) / np.linalg.norm(F[lab == k])) for k in range(edges.size - 1)]
        rec = dict(seed=seed)
        rec['box'] = brel(s2.pixel_box_ft(A, u, v))
        rec['smooth'] = brel(s2.SAFI2D(A, P=4, edges=False)(u, v))
        tb = time.time()
        if n in POLY_NS:
          try:
            Sp = s2.SAFI2D(A, P=4)
            okp = Sp.polygon_fit()
            rec['polygon'] = brel(Sp(u, v)); rec['polygon_ok'] = bool(okp)
            rec['polygon_m'] = int(len(Sp.poly['V'])) if okp else None
          except Exception as e:                              # report, do not hide
            rec['polygon_error'] = repr(e)
        rec['polygon_s'] = time.time() - tb
        tb = time.time()
        S = CurveSAFI2D(A, P=4, repair=False)
        ok = S.curve_fit()
        rec['build_s'] = time.time() - tb
        rec['curve_ok'] = bool(ok)
        if ok:
            tt = np.linspace(0, TWO_PI, 4000, endpoint=False)
            x, y = img.curve.xy(tt)
            phi = np.mod(np.arctan2(y - S.c0[1], x - S.c0[0]), TWO_PI)
            err = np.abs(np.hypot(x - S.c0[0], y - S.c0[1]) - S.curve.rho(phi))
            rec['bnd_err_max_px'] = float(err.max() * n); rec['bnd_err_med_px'] = float(np.median(err) * n)
            rec['modes'] = int((S.coef.size - 1) // 2)
            rec['curve'] = brel(S(u, v))
            S.curve = img.curve; S._geom = S._geometry(img.curve); S.poly['band'] = S._geom['band']
            rec['known'] = brel(S(u, v))
        rows.append(rec)
        print(n, seed, 'curve ok', ok, 'modes', rec.get('modes'), 'bnd max %.1e med %.1e px' % (rec.get('bnd_err_max_px', np.nan), rec.get('bnd_err_med_px', np.nan)),
              'build %.0fs' % rec['build_s'],
              '| curve [0,mid,last] %.1e %.1e %.1e' % (rec['curve'][0], rec['curve'][len(edges) // 2], rec['curve'][-1]) if ok else '',
              '| polygon last %.1e' % rec['polygon'][-1] if 'polygon' in rec else rec.get('polygon_error', ''),
              '| box last %.1e' % rec['box'][-1], '%.0fs' % (time.time() - t00), flush=True)
    res[n] = dict(edges=edges.tolist(), rows=rows)
    json.dump({str(k): v for k, v in sorted(res.items())}, open('results/exp13_curved.json', 'w'), indent=1)

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
sty = {'box': ('#888', 'pixel box (standard)'), 'smooth': ('#c77', 'smooth fits, edge-unaware'),
       'polygon': ('#d62', 'SAFI-2D, polygon'), 'curve': ('#126', 'SAFI-2D, curve (edges from data)'),
       'known': ('#2a5', 'SAFI-2D, true curve')}
nn = 128
E = np.array(res[nn]['edges']); uc = np.sqrt(E[:-1] * E[1:])
for k, (c, l) in sty.items():
    vals = [r[k] for r in res[nn]['rows'] if k in r]
    if vals:
        ax[0].loglog(uc, np.median(vals, 0), '-o', ms=3, color=c, label=l)
ax[0].axvline(nn / 2, color='k', ls=':', lw=.7)
ax[0].set(xlabel='|k|', ylabel='band relative error (median)', title=f'curved inclusion, n = {nn}'); ax[0].legend(fontsize=7)
NS = sorted(res)
be = [np.median([r['bnd_err_max_px'] for r in res[n_]['rows'] if r.get('curve_ok')]) for n_ in NS]
ax[1].loglog(NS, be, '-o', color='#126', label='max boundary error (median over images)')
ax[1].set_xscale('log', base=2)
ax[1].set(xlabel='n', ylabel='pixels', title='boundary from pixel averages'); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig('figs/exp13_curved.png', dpi=140)
