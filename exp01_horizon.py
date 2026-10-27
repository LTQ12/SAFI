"""Exp 1: frequency horizon of each data mode (one interior jump, unknown location).

Methods: frozen trapezoid sum (NUFFT / sampling-sum object), break-unaware hp interpolant
(original A_smp-type), SAFI-point, SAFI-adaptive (q bisections), SAFI-known (oracle breaks).
Reference: minimax lower bound for point data, |J| min(sin(pi u h)/(2 pi u), 1/(2 pi u)).
"""
import sys, json, time
sys.path.insert(0, '.')
import numpy as np
from bench import PiecewiseExp, band_grid, band_rel, horizon
from safi import SAFI, riemann_sum

M = 6
NS = [64, 128, 256, 512, 1024, 2048]
NF = 12
QS = [10, 20, 30]
EPS = [1e-2, 1e-4]
u, lab, edges = band_grid(2.0 ** 20, per_band=40, rng=np.random.default_rng(7))
rng = np.random.default_rng(2026)
funcs = [PiecewiseExp.random(rng, rng.uniform(0.2, 0.8)) for _ in range(NF)]

methods = ['trap', 'unaware', 'point'] + [f'adapt_q{q}' for q in QS] + ['known']
rel = {m: np.zeros((len(NS), NF, edges.size - 1)) for m in methods}
lb = np.zeros((len(NS), NF, edges.size - 1))
est_ratio = {m: [] for m in ['point', 'adapt_q20', 'known']}
t0 = time.time()
detfail = []
for a, n in enumerate(NS):
    x = np.linspace(0, 1, n + 1); h = 1.0 / n
    for b, f in enumerate(funcs):
        y = f(x); F = f.ft(u); J = abs(f.jumps()[0])
        res = {'trap': riemann_sum(x, y, u)}
        res['unaware'] = SAFI(x, y, M, mode='none')(u)
        objs = {'point': SAFI(x, y, M, mode='point'), 'known': SAFI(x, y, M, mode='known', breaks=[f.breaks[1]])}
        for q in QS:
            objs[f'adapt_q{q}'] = SAFI(x, y, M, mode='adaptive', f=f, q=q)
        for k, o in objs.items():
            if k != 'known' and (len(o.breaks) != 1 or not (o.breaks[0].lo <= f.breaks[1] <= o.breaks[0].hi)):
                detfail.append((k, n, b, len(o.breaks)))
            res[k] = o(u)
            if k in est_ratio:
                e = o.error_estimate(u)
                est_ratio[k].append(np.median(e / np.abs(F - res[k])))
                est_ratio[k].append(np.min(e / np.abs(F - res[k])))
        for k in methods:
            rel[k][a, b] = band_rel(F, res[k], lab)
        LB = J * np.where(u <= 1 / (2 * h), np.sin(np.pi * u * h) / (2 * np.pi * u), 1 / (2 * np.pi * u))
        lb[a, b] = np.array([np.linalg.norm(LB[lab == k]) / np.linalg.norm(F[lab == k]) for k in np.unique(lab)])
    print(n, f'{time.time() - t0:.1f}s', flush=True)

out = {'M': M, 'NS': NS, 'NF': NF, 'edges': edges.tolist(), 'methods': methods,
       'rel_median': {k: np.median(v, 1).tolist() for k, v in rel.items()},
       'rel_max': {k: np.max(v, 1).tolist() for k, v in rel.items()},
       'lb_median': np.median(lb, 1).tolist(),
       'horizon_median': {f'{e:g}': {k: [float(np.median([horizon(rel[k][a, b], edges, e) for b in range(NF)]))
                                        for a in range(len(NS))] for k in methods} for e in EPS},
       'est_ratio_median_of_medians': {k: float(np.median(v[0::2])) for k, v in est_ratio.items()},
       'detect_failures': detfail,
       'est_ratio_min': {k: float(np.min(v[1::2])) for k, v in est_ratio.items()}}
json.dump(out, open('results/exp01_horizon.json', 'w'), indent=1)

# ---- figure
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
uc = np.sqrt(edges[:-1] * edges[1:])
sty = {'trap': ('#888', 'trapezoid sum (NUFFT obj.)'), 'unaware': ('#c77', 'break-unaware hp'),
       'point': ('#d62', 'SAFI point'), 'adapt_q10': ('#7ad', 'SAFI adaptive q=10'),
       'adapt_q20': ('#37b', 'SAFI adaptive q=20'), 'adapt_q30': ('#126', 'SAFI adaptive q=30'),
       'known': ('#2a5', 'SAFI known breaks')}
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
ia = NS.index(128)
for k in methods:
    ax[0].loglog(uc, np.median(rel[k][ia], 0), '-o', ms=3, color=sty[k][0], label=sty[k][1])
ax[0].loglog(uc, np.median(lb[ia], 0), 'k--', lw=1, label='minimax lower bound (point data)')
ax[0].axvline(128 / 2, color='k', lw=.6, ls=':')
ax[0].set(xlabel='frequency u (band centre)', ylabel='band relative L2 error', title='n = 128, M = 6', ylim=(1e-12, 1e4))
ax[0].legend(fontsize=7)
for k in methods:
    hz = out['horizon_median']['0.01'][k]
    ax[1].loglog(NS, np.minimum(hz, 2 ** 21), '-o', color=sty[k][0], label=sty[k][1])
ax[1].loglog(NS, np.array(NS) / 2, 'k:', lw=1, label='Nyquist n/2')
ax[1].set(xlabel='n samples', ylabel='horizon u*(eps=1e-2)', title='frequency horizon (capped at 2^21)')
for k in ['unaware', 'point', 'adapt_q20', 'known']:
    ax[2].loglog(1 / np.array(NS), np.median(rel[k][:, :, 0], 1), '-o', color=sty[k][0], label=sty[k][1])
hh = 1 / np.array(NS)
ax[2].loglog(hh, hh * 2, 'k:', lw=1, label='O(h)')
ax[2].loglog(hh, 2e2 * hh ** 7, 'k--', lw=1, label='O(h^7)')
ax[2].set(xlabel='h', ylabel='rel. error, band [1,2]', title='low-frequency order')
ax[2].legend(fontsize=7)
plt.tight_layout(); plt.savefig('figs/exp01_horizon.png', dpi=140)
print(json.dumps(out['horizon_median'], indent=0))
print('detect failures', detfail)
print('est ratios', out['est_ratio_median_of_medians'], out['est_ratio_min'])
