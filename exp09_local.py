"""Exp 9: clustered singularities. Global vs local refinement, separation sweep.

Two jumps at distance s in {0.1, ..., 1e-4}; target band [1, 2^14], eps = 1e-8.
Prediction: global uniform refinement needs n ~ (M+2)/s (resolvability) -> evaluations ~ 1/s;
local refinement needs ~ log2(1/s) refinement levels of O(M) cells each -> ~ log(1/s).
Also: application A (switched transient) with the local driver.
"""
import sys, json, time
sys.path.insert(0, '.')
import numpy as np
from bench import PiecewiseExp, band_grid, band_rel
from safi import safi_auto
from safi_adapt import safi_auto_local

u, lab, edges = band_grid(2.0 ** 14, per_band=16, rng=np.random.default_rng(9))
SEP = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5]
out = {'sweep': []}
for s in SEP:
    rows = []
    for seed in range(5):
        rng = np.random.default_rng(900 + seed)
        c0 = rng.uniform(0.3, 0.6)
        f = PiecewiseExp.random(rng, [c0, c0 + s])
        F = f.ft(u)
        t = time.time(); C, Fh, eh, st = safi_auto_local(f, 0, 1, u, 1e-8, s_min=1e-6); tl = time.time() - t
        r = dict(local_evals=st['nf'], local_cert=st['certified'], local_true=float(band_rel(F, Fh, lab).max()),
                 local_segs=st['nseg'], local_s=tl)
        if s >= 1e-3:
            t = time.time(); S, Gh, _, st0 = safi_auto(f, 0, 1, u, 1e-8, nmax=2 ** 17); tg = time.time() - t
            r.update(global_evals=st0['nf'], global_cert=st0['certified'], global_true=float(band_rel(F, Gh, lab).max()), global_s=tg)
        rows.append(r)
    agg = {k: float(np.median([r[k] for r in rows])) for k in rows[0] if isinstance(rows[0][k], (int, float)) and not isinstance(rows[0][k], bool)}
    agg.update(sep=s, local_cert_frac=float(np.mean([r['local_cert'] for r in rows])),
               local_true_max=float(max(r['local_true'] for r in rows)))
    if 'global_cert' in rows[0]:
        agg.update(global_cert_frac=float(np.mean([r['global_cert'] for r in rows])),
                   global_true_max=float(max(r['global_true'] for r in rows)))
    out['sweep'].append(agg)
    print(agg, flush=True)

# application A with the local driver
TWO_PI = 2 * np.pi
rng = np.random.default_rng(8)
ts = np.sort(rng.uniform(0.1, 0.9, 3)); br = np.r_[0, ts, 1]
A, B = [], []
for k in range(4):
    V = [1.0, -0.6, 0.8, -0.3][k]; tau, a, w0 = 0.02, 8.0, TWO_PI * 37.0
    A.append([V * np.exp(br[k] / tau), 0.4 * V * np.exp((a - 1j * w0) * br[k]), 0.15]); B.append([-1 / tau, -a + 1j * w0, 0.0])
fA = PiecewiseExp(br, A, B)
uA = np.logspace(0, 6, 241); FA = fA.ft(uA)
dec = np.floor(np.log10(uA)).clip(max=5)
C, Fh, eh, st = safi_auto_local(fA, 0, 1, uA, 1e-6, M=8, n0=128, s_min=1e-6)
out['appA_local'] = dict(evals=st['nf'], certified=st['certified'], segs=st['nseg'],
                         true=float(max(np.linalg.norm((FA - Fh)[dec == d]) / np.linalg.norm(FA[dec == d]) for d in np.unique(dec))))
print('appA', out['appA_local'], flush=True)
json.dump(out, open('results/exp09_local.json', 'w'), indent=1)

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(5.5, 4.2))
sw = out['sweep']
ax.loglog([r['sep'] for r in sw], [r['local_evals'] for r in sw], '-o', color='#126', label='SAFI local refinement')
g = [r for r in sw if 'global_evals' in r]
ax.loglog([r['sep'] for r in g], [r['global_evals'] for r in g], '-o', color='#d62', label='SAFI global refinement')
ss = np.array(SEP)
ax.loglog(ss, 10 / ss, 'k:', lw=1, label='~1/s')
ax.set(xlabel='separation of the two jumps s', ylabel='function evaluations (median)', title='eps = 1e-8, band [1, 2^14]')
ax.legend(fontsize=8); plt.tight_layout(); plt.savefig('figs/exp09_local.png', dpi=140)
