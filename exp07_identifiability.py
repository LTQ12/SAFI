"""Exp 7: identifiability of a singularity = (singularity order r) x (data integration order k).

r = 0: jump in f;  r = 1: kink (f continuous, f' jumps).
k = 0: point samples;  k = 1: cell averages.
Prediction: the location is identifiable without extra data iff r + k >= 1:
  (r,k)=(0,0): delta = h/2 (no better possible)      (1,0): intersection, delta ~ h^{M+1}/|J1|
  (0,1): subcell, delta ~ h^{M+2}/|J0|                 (1,1): intersection, delta ~ h^{M+1}/|J1|
"""
import sys, json
sys.path.insert(0, '.')
import numpy as np
from bench import PiecewiseExp, band_grid, band_rel, horizon
from safi import SAFI
from safi_cell import SAFICell

M = 6
NS = [64, 128, 256, 512, 1024]
NF = 16
TWO_PI = 2 * np.pi
u, lab, edges = band_grid(2.0 ** 22, per_band=24, rng=np.random.default_rng(4))


def random_fn(rng, kink):
    c = rng.uniform(0.25, 0.75)
    f = PiecewiseExp.random(rng, c)
    if kink:  # add a constant on the right piece so that f is continuous at c
        J = f.jumps()[0]
        f.alphas[1] = np.r_[f.alphas[1], -J]
        f.betas[1] = np.r_[f.betas[1], 0.0]
    return f, c


res = {}
for r in [0, 1]:
    rng = np.random.default_rng(700 + r)
    funcs = [random_fn(rng, r == 1) for _ in range(NF)]
    for k in [0, 1]:
        loc = np.full((len(NS), NF), np.nan); rel = np.full((len(NS), NF, edges.size - 1), np.nan)
        kinds = []
        for a, n in enumerate(NS):
            for b, (f, c) in enumerate(funcs):
                if k == 0:
                    x = np.linspace(0, 1, n + 1); S = SAFI(x, f(x), M, 'point')
                    est = [bb.c for bb in S.breaks]
                else:
                    xe = np.linspace(0, 1, n + 1); S = SAFICell(xe, f.cell_avg(xe), M)
                    est = [bb.cstar for bb in S.breaks]
                kinds += [getattr(bb, 'kind', 'jump') for bb in S.breaks]
                if len(est) == 1:
                    loc[a, b] = abs(est[0] - c)
                    rel[a, b] = band_rel(f.ft(u), S(u), lab)
        med = np.nanmedian(loc, 1)
        ok = med > 1e-14
        slope = float(np.polyfit(np.log(1 / np.array(NS))[ok], np.log(med[ok]), 1)[0]) if ok.sum() >= 2 else np.nan
        hz = [float(np.median([horizon(rel[a, b], edges, 1e-6) for b in range(NF)])) for a in range(len(NS))]
        res[f'r{r}_k{k}'] = dict(loc_median=med.tolist(), slope=slope, horizon_1e6=hz,
                                 detected_frac=float(np.mean(np.isfinite(loc))),
                                 kind_counts={kk: kinds.count(kk) for kk in set(kinds)},
                                 rel_median_n256=np.nanmedian(rel[NS.index(256)], 0).tolist())
        print(f'r={r} k={k}', {kk: v for kk, v in res[f'r{r}_k{k}'].items() if kk != 'rel_median_n256'}, flush=True)
json.dump(dict(M=M, NS=NS, edges=edges.tolist(), res=res), open('results/exp07_identifiability.json', 'w'), indent=1)

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
hh = 1 / np.array(NS)
sty = {'r0_k0': ('#d62', 'jump, point samples'), 'r1_k0': ('#37b', 'kink, point samples'),
       'r0_k1': ('#126', 'jump, cell averages'), 'r1_k1': ('#2a5', 'kink, cell averages')}
for key, (c, l) in sty.items():
    ax[0].loglog(hh, res[key]['loc_median'], '-o', color=c, label=f"{l} (slope {res[key]['slope']:.1f})")
ax[0].loglog(hh, hh / 4, 'k:', lw=1, label='h/4 (gap midpoint)')
ax[0].set(xlabel='h', ylabel='median |c_hat - c|', title='singularity location from fixed data', ylim=(1e-17, 1e-1))
ax[0].legend(fontsize=7)
uc = np.sqrt(edges[:-1] * edges[1:])
for key, (c, l) in sty.items():
    ax[1].loglog(uc, res[key]['rel_median_n256'], '-o', ms=3, color=c, label=l)
ax[1].axvline(128, color='k', ls=':', lw=.7)
ax[1].set(xlabel='u', ylabel='band relative error', title='n = 256, M = 6'); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig('figs/exp07_identifiability.png', dpi=140)
