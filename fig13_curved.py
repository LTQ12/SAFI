"""Figure for Exp 13 (curved inclusions) from results/exp13_curved.json."""
import json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = {int(k): v for k, v in json.load(open('results/exp13_curved.json')).items()}
sty = {'box': ('#888', 'pixel box (standard)'), 'smooth': ('#c77', 'smooth fits, edge-unaware'),
       'polygon': ('#d62', 'SAFI-2D, polygon model'), 'curve': ('#126', 'SAFI-2D, curve from data'),
       'known': ('#2a5', 'SAFI-2D, true curve')}
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for a, nn in zip(ax[:2], (64, 128)):
    E = np.array(R[nn]['edges']); uc = np.sqrt(E[:-1] * E[1:])
    for k, (c, l) in sty.items():
        vals = [r[k] for r in R[nn]['rows'] if k in r]
        if vals:
            a.loglog(uc, np.median(vals, 0), '-o', ms=3, color=c, label=l)
    a.axvline(nn / 2, color='k', ls=':', lw=.7)
    a.set(xlabel='|k|', ylabel='band relative error (median)', title=f'curved inclusion, n = {nn}', ylim=(1e-9, 10))
    a.legend(fontsize=7, loc='lower right')
NS = sorted(R)
mx = [np.median([r['bnd_err_max_px'] / n for r in R[n]['rows'] if r['seed'] in (1, 2, 3)]) for n in NS]
md = [np.median([r['bnd_err_med_px'] / n for r in R[n]['rows'] if r['seed'] in (1, 2, 3)]) for n in NS]
ax[2].loglog(NS, mx, '-o', color='#126', label='max over the boundary')
ax[2].loglog(NS, md, '-s', color='#6a8fd0', label='median over the boundary')
ref = mx[0] * (np.array(NS) / NS[0]) ** -6.0
ax[2].loglog(NS, ref, 'k:', lw=1, label='h^6')
ax[2].set_xscale('log', base=2)
ax[2].set(xlabel='n', ylabel='radial boundary error (length)', title='boundary from pixel averages (3 images)')
ax[2].legend(fontsize=7)
plt.tight_layout(); plt.savefig('figs/exp13_curved.png', dpi=140)
