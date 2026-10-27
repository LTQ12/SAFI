"""Figure: MRI phantom k-space from partial-volume pixels (Exp 16)."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

r = json.load(open('../results/exp16_mri.json'))['SL_n256']
d = np.load('../results/exp16_SL_n256_recon.npz')
A, Rt, Rr, Rs = d['A'], d['Rt'], d['Rr'], d['Rs']
plt.rcParams.update({'font.size': 8})
fig, ax = plt.subplots(1, 4, figsize=(13, 3.2), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.35]))
ext = (0, 1, 0, 1)
ax[0].imshow(A.T, origin='lower', cmap='gray', extent=ext)
ax[0].set_title('(a) partial-volume phantom, $256^2$')
v = np.abs(Rt - A).max() * 0.5
im = ax[1].imshow((Rt - A).T, origin='lower', cmap='RdBu_r', vmin=-v, vmax=v, extent=ext)
ax[1].set_title('(b) recon. of exact data $-$ pixels')
plt.colorbar(im, ax=ax[1], fraction=0.046, pad=0.03)
E = np.abs(Rs - Rt) + 1e-12
im = ax[2].imshow(E.T, origin='lower', cmap='magma', norm=LogNorm(vmin=1e-9, vmax=max(E.max(), 1e-8)), extent=ext)
ax[2].set_title('(c) recon. of our data $-$ recon. of exact')
plt.colorbar(im, ax=ax[2], fraction=0.046, pad=0.03)
for a in ax[:3]:
    a.set_xticks([]); a.set_yticks([])

n = 256
sty = [('raster', '#d62', 'o', 'DFT of pixels (periodic beyond $n/2$)'), ('box', '#a85', 's', 'exact transform of pixel boxes'),
       ('upsampled', '#e9a', 'v', 'bilinear upsampling, then DFT'), ('raster_deconv', '#777', 'x', 'DFT / pixel transfer function'),
       ('smooth', '#5a5', '^', 'degree 4, edges ignored'), ('safi', '#126', 'D', 'conservative fit + exact transform')]
for key, col, mk, lab in sty:
    for N in ('1024', '256'):
        acc = r['acc'][N]
        if key in acc['rings']:
            e = np.array(acc['edges'])
            mid = np.sqrt(np.maximum(e[:-1], 4) * e[1:])
            vals = np.array([np.nan if x is None else x for x in acc['rings'][key]], float)
            ax[3].loglog(mid / (n / 2), vals, '-' + mk, color=col, ms=3.5, lw=1, label=lab)
            break
ax[3].axvline(1, color='k', ls=':', lw=0.8)
ax[3].set(xlabel='$|\\mathbf{k}|$ / pixel Nyquist frequency', ylabel='relative error per ring', title='(d) k-space error, $N$ up to $4n$')
ax[3].set_ylim(1e-9, 1e7)
ax[3].legend(fontsize=5.6, loc='upper left', frameon=False)
plt.tight_layout()
plt.savefig('figs/mri.pdf')
plt.savefig('figs/mri.png', dpi=160)
print('Rt-A max', np.abs(Rt - A).max(), 'Rs-Rt max', np.abs(Rs - Rt).max(), 'Rr-A', np.abs(Rr - A).max())
