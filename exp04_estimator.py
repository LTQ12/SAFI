"""Exp 4: reliability/efficiency of the a-posteriori estimate e(u), many random functions.

Effectivity index theta(u) = e(u)/|F(u) - F_hat(u)| per frequency. Reliable <=> theta >= 1.
We report, per mode, the fraction of frequencies with theta >= 1 and band-level effectivity
(||e||_band / ||err||_band) quantiles. 1-3 interior jumps, n in {128, 512}, M in {4, 6, 8}.
"""
import sys, json
sys.path.insert(0, '.')
import numpy as np
from bench import PiecewiseExp, band_grid
from safi import SAFI

rng = np.random.default_rng(404)
u, lab, edges = band_grid(2.0 ** 18, per_band=24, rng=np.random.default_rng(3))
out = {}
for M in [4, 6, 8]:
    for n in [128, 512]:
        for mode in ['point', 'adaptive', 'known']:
            th_pt, th_band, miss = [], [], 0
            for trial in range(30):
                nj = rng.integers(1, 4)
                while True:
                    c = np.sort(rng.uniform(0.1, 0.9, nj))
                    if nj == 1 or np.min(np.diff(c)) > 0.08:
                        break
                f = PiecewiseExp.random(rng, c)
                x = np.linspace(0, 1, n + 1)
                kw = dict(f=f, q=int(rng.integers(5, 40))) if mode == 'adaptive' else (dict(breaks=list(c)) if mode == 'known' else {})
                S = SAFI(x, f(x), M, mode, **kw)
                if mode != 'known' and len(S.breaks) != nj:
                    miss += 1
                    continue
                err = np.abs(f.ft(u) - S(u)); e = S.error_estimate(u)
                th_pt.append(e / np.maximum(err, 1e-300))
                th_band += [np.linalg.norm(e[lab == k]) / np.linalg.norm(err[lab == k]) for k in np.unique(lab)]
            th_pt = np.concatenate(th_pt); th_band = np.array(th_band)
            out[f'M{M}_n{n}_{mode}'] = dict(frac_pointwise_reliable=float(np.mean(th_pt >= 1)),
                                            band_eff_q=[float(v) for v in np.quantile(th_band, [0, .05, .5, .95, 1])],
                                            detect_miss=miss)
            print(M, n, mode, out[f'M{M}_n{n}_{mode}'], flush=True)
json.dump(out, open('results/exp04_estimator.json', 'w'), indent=1)
