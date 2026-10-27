"""Exp 11: 2-D pixel images with a polygonal interface (pixel averages = 2-D cell data).

Images: smooth background + a random convex hexagon with its own smooth phase (jump 1.5-3).
Data: exact pixel averages, n x n, n in {32, 64, 128, 256}; 6 images.
Frequencies: radial dyadic bands |k| in [2^k, 2^{k+1}], up to 128 n, 32 random directions per band.
Methods: pixel-box (standard: piecewise-constant pixels), smooth-unaware (local polynomial fits, no
edges), SAFI-2D (edges from the data: detection -> straight-edge families -> conservative polygon fit),
SAFI-2D with the true polygon (geometry oracle; phases still fitted).
"""
import sys, json, time, copy
sys.path.insert(0, '.')
import numpy as np
from bench2d import Phase, PolyImage
import safi2d as s2

TWO_PI = 2 * np.pi
NS = [32, 64, 128, 256]
SEEDS = [3, 4, 5, 6, 7, 8]


def make_img(seed, osc=0.7):
    rng = np.random.default_rng(seed)
    bg = Phase.random(rng, osc=osc, decay=1.0)
    img = PolyImage.random(rng, npoly=1)
    return PolyImage(bg, img.polys, [Phase.random(np.random.default_rng(seed + 100), osc=osc, decay=1.0, offset=1.5)])


def band_px(V, h, n):
    pts = set()
    m = len(V)
    for a in range(m):
        p, q = V[a], V[(a + 1) % m]
        L = np.linalg.norm(q - p)
        for s_ in np.linspace(0, 1, max(2, int(4 * L / h) + 2)):
            x = p + s_ * (q - p); i, j = int(x[0] / h), int(x[1] / h)
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if 0 <= i + di < n and 0 <= j + dj < n:
                        pts.add((i + di, j + dj))
    return sorted(pts)


res = {}
t0 = time.time()
for n in NS:
    h = 1.0 / n
    kmax = int(np.ceil(np.log2(128 * n)))
    edges = 2.0 ** np.arange(0, kmax + 1)
    rng = np.random.default_rng(n)
    R = np.concatenate([rng.uniform(edges[k], edges[k + 1], 32) for k in range(edges.size - 1)])
    lab = np.repeat(np.arange(edges.size - 1), 32)
    th = rng.uniform(0, TWO_PI, R.size)
    u, v = R * np.cos(th), R * np.sin(th)
    rows = []
    for seed in (SEEDS if n <= 128 else SEEDS[:3]):
        img = make_img(seed)
        A = img.pixel_averages(n)
        F = img.ft(u, v)
        brel = lambda G: [float(np.linalg.norm((F - G)[lab == k]) / np.linalg.norm(F[lab == k])) for k in range(edges.size - 1)]
        rec = dict(seed=seed)
        rec['box'] = brel(s2.pixel_box_ft(A, u, v))
        rec['smooth'] = brel(s2.SAFI2D(A, P=4, edges=False)(u, v))
        tb = time.time()
        S = s2.SAFI2D(A, P=4)
        ok = S.polygon_fit()
        rec['build_s'] = time.time() - tb
        rec['poly_ok'] = bool(ok)
        if ok:
            Vt, Vf = img.polys[0], S.poly['V']
            rec['vertex_err_px'] = float(max(min(np.linalg.norm(Vf - p, axis=1)) for p in Vt) * n)
            rec['m'] = int(len(Vf))
            rec['safi'] = brel(S(u, v))
            K = copy.copy(S); K.poly = dict(S.poly)
            Vt_ccw = Vt if 0.5 * np.sum(Vt[:, 0] * np.roll(Vt[:, 1], -1) - np.roll(Vt[:, 0], -1) * Vt[:, 1]) > 0 else Vt[::-1]
            K.poly['V'] = Vt_ccw
            K.poly['band'] = sorted(set(band_px(Vt_ccw, h, n)) | set(S.poly['band']))
            rec['known'] = brel(K(u, v))
        else:
            rec['safi'] = brel(S(u, v))
        rows.append(rec)
        print(n, seed, 'ok', ok, 'vert err %.1e px' % rec.get('vertex_err_px', np.nan), 'build %.0fs' % rec['build_s'],
              'safi bands[0,mid,last] %.1e %.1e %.1e' % (rec['safi'][0], rec['safi'][len(rec['safi']) // 2], rec['safi'][-1]),
              'box last %.1e' % rec['box'][-1], f'{time.time() - t0:.0f}s', flush=True)
    res[n] = dict(edges=edges.tolist(), rows=rows)
    json.dump(res, open('results/exp11_2d.json', 'w'), indent=1)

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
sty = {'box': ('#888', 'pixel box (standard)'), 'smooth': ('#c77', 'smooth fits, edge-unaware'),
       'safi': ('#126', 'SAFI-2D (edges from data)'), 'known': ('#2a5', 'SAFI-2D, true polygon')}
nn = 128
E = np.array(res[nn]['edges']); uc = np.sqrt(E[:-1] * E[1:])
for k, (c, l) in sty.items():
    vals = [r[k] for r in res[nn]['rows'] if k in r]
    if vals:
        ax[0].loglog(uc, np.median(vals, 0), '-o', ms=3, color=c, label=l)
ax[0].axvline(nn / 2, color='k', ls=':', lw=.7)
ax[0].set(xlabel='|k|', ylabel='band relative error (median of 6)', title=f'n = {nn} pixels per side'); ax[0].legend(fontsize=7)
ve = [np.median([r['vertex_err_px'] for r in res[n_]['rows'] if r.get('poly_ok')]) if any(r.get('poly_ok') for r in res[n_]['rows']) else np.nan for n_ in NS]
ax[1].semilogy(NS, ve, '-o', color='#126')
ax[1].set_xscale('log', base=2)
ax[1].set(xlabel='n', ylabel='median vertex error (pixels)', title='polygon vertices from pixel averages')
for k, (c, l) in sty.items():
    hz = []
    for n_ in NS:
        E_ = np.array(res[n_]['edges'])
        vals = [r[k] for r in res[n_]['rows'] if k in r]
        if not vals:
            hz.append(np.nan); continue
        med = np.median(vals, 0); bad = np.where(med > 1e-3)[0]
        hz.append(E_[bad[0]] if bad.size else E_[-1] * 2)
    ax[2].loglog(NS, hz, '-o', color=c, label=l)
ax[2].loglog(NS, np.array(NS) / 2, 'k:', lw=1, label='Nyquist n/2')
ax[2].set(xlabel='n', ylabel='horizon |k|*(eps=1e-3)', title='2-D horizon (capped at 256 n)'); ax[2].legend(fontsize=7)
plt.tight_layout(); plt.savefig('figs/exp11_2d.png', dpi=140)
