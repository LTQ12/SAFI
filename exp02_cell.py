"""Exp 2: cell-average data. Subcell location error vs h and the horizon it buys.

Same functions/frequencies as Exp 1. Compared: SAFI-cell (subcell resolution, no extra data),
SAFI-cell with the true break (oracle), break-unaware finite-volume reconstruction,
midpoint sum of the averages (the frozen sum such data usually feed), SAFI-point on point
samples at the same n (from Exp 1 protocol).
"""
import sys, json, time
sys.path.insert(0, '.')
import numpy as np
from bench import PiecewiseExp, band_grid, band_rel, horizon
from safi import SAFI
from safi_cell import SAFICell

M = 6
NS = [64, 128, 256, 512, 1024, 2048]
NF = 12
u, lab, edges = band_grid(2.0 ** 24, per_band=40, rng=np.random.default_rng(7))
rng = np.random.default_rng(2026)
funcs = [PiecewiseExp.random(rng, rng.uniform(0.2, 0.8)) for _ in range(NF)]
methods = ['cell_mid_sum', 'cell_unaware', 'cell_subcell', 'cell_known', 'point']
rel = {m: np.zeros((len(NS), NF, edges.size - 1)) for m in methods}
locerr = np.zeros((len(NS), NF)); locdel = np.zeros((len(NS), NF)); fails = []
t0 = time.time()
for a, n in enumerate(NS):
    xe = np.linspace(0, 1, n + 1); h = 1 / n
    for b, f in enumerate(funcs):
        F = f.ft(u); ab = f.cell_avg(xe); c = f.breaks[1]
        xm = 0.5 * (xe[:-1] + xe[1:])
        G = {'cell_mid_sum': (np.exp(-2j * np.pi * np.outer(u, xm)) @ ab) * h}
        S = SAFICell(xe, ab, M)
        if len(S.breaks) != 1:
            fails.append((n, b, len(S.breaks)))
        else:
            locerr[a, b] = abs(S.breaks[0].cstar - c); locdel[a, b] = S.breaks[0].delta
        G['cell_subcell'] = S(u)
        G['cell_known'] = SAFICell(xe, ab, M, breaks_known=[c])(u)
        Su = SAFICell.__new__(SAFICell)                    # unaware: no breaks at all
        Su.xe, Su.ab, Su.M, Su.h = xe, ab, M, h
        Su.V = np.concatenate([[0.0], np.cumsum(ab) * h]); Su.breaks = []; Su._build()
        G['cell_unaware'] = Su(u)
        x = np.linspace(0, 1, n + 1)
        G['point'] = SAFI(x, f(x), M, 'point')(u)
        for k in methods:
            rel[k][a, b] = band_rel(F, G[k], lab)
    print(n, f'{time.time() - t0:.0f}s', flush=True)

hz = {f'{e:g}': {k: [float(np.median([horizon(rel[k][a, b], edges, e) for b in range(NF)])) for a in range(len(NS))]
                 for k in methods} for e in [1e-2, 1e-4]}
ok = locerr > 0
slope = np.polyfit(np.log(1 / np.array(NS))[:4], np.log(np.median(np.where(ok, locerr, np.nan), 1)[:4]), 1)[0]
out = dict(M=M, NS=NS, edges=edges.tolist(), fails=fails, horizon_median=hz,
           loc_err_median=np.median(locerr, 1).tolist(), loc_delta_median=np.median(locdel, 1).tolist(),
           loc_err_slope_first4=float(slope),
           delta_covers_true_frac=float(np.mean(locdel[ok] >= locerr[ok])),
           rel_median={k: np.median(v, 1).tolist() for k, v in rel.items()})
json.dump(out, open('results/exp02_cell.json', 'w'), indent=1)
print(json.dumps({k: out[k] for k in ['fails', 'horizon_median', 'loc_err_median', 'loc_err_slope_first4', 'delta_covers_true_frac']}, indent=0))

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
uc = np.sqrt(edges[:-1] * edges[1:])
sty = {'cell_mid_sum': ('#888', 'midpoint sum of averages'), 'cell_unaware': ('#c77', 'break-unaware FV recon.'),
       'cell_subcell': ('#126', 'SAFI cell (subcell)'), 'cell_known': ('#2a5', 'SAFI cell, known break'),
       'point': ('#d62', 'SAFI point samples (same n)')}
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
ia = NS.index(256)
for k in methods:
    ax[0].loglog(uc, np.median(rel[k][ia], 0), '-o', ms=3, color=sty[k][0], label=sty[k][1])
ax[0].axvline(128, color='k', lw=.6, ls=':')
ax[0].set(xlabel='frequency u', ylabel='band relative L2 error', title='n = 256 cells, M = 6', ylim=(1e-15, 1e3))
ax[0].legend(fontsize=7)
hh = 1 / np.array(NS)
ax[1].loglog(hh, np.median(locerr, 1), '-o', color='#126', label='|c_hat - c| (median)')
ax[1].loglog(hh, np.median(locdel, 1), '--', color='#126', label='estimated delta')
ax[1].loglog(hh, hh ** 8 * np.median(locerr, 1)[0] / hh[0] ** 8, 'k:', label='O(h^{M+2})')
ax[1].set(xlabel='h', ylabel='break location error', title='subcell localisation')
ax[1].legend(fontsize=7)
for k in methods:
    ax[2].loglog(NS, np.minimum(hz['0.0001'][k], 2 ** 25), '-o', color=sty[k][0], label=sty[k][1])
ax[2].loglog(NS, np.array(NS) / 2, 'k:', lw=1, label='Nyquist n/2')
ax[2].set(xlabel='n', ylabel='horizon u*(eps=1e-4)', title='horizon (capped at 2^25)')
ax[2].legend(fontsize=7)
plt.tight_layout(); plt.savefig('figs/exp02_cell.png', dpi=140)
