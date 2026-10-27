"""Figure 1: the failure mechanism (folded singular tail) and what removes it."""
import sys, json
sys.path.insert(0, '..')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import exp15_mechanism as e
from bench import PiecewiseExp

r = json.load(open('../results/exp15_mechanism.json'))
TWO_PI = 2 * np.pi
plt.rcParams.update({'font.size': 8})
fig, ax = plt.subplots(1, 4, figsize=(13, 3.1))

# (a) what the data contain: average and point value of a unit step at x_i + theta h
th = np.linspace(0, 1, 201)
ax[0].plot(th, 1 - th, color='#126', lw=1.6, label='cell average $\\bar a_i$')
ax[0].plot(th, np.zeros_like(th), color='#d62', lw=1.6, label='point value $f(x_i)$')
ax[0].plot(th, np.ones_like(th), color='#d62', lw=1.6, ls='--', label='point value $f(x_{i+1})$')
ax[0].set(xlabel='sub-cell position $\\theta$ of the jump', ylabel='datum (unit jump)',
          title='(a) what the data record', ylim=(-0.1, 1.15))
ax[0].legend(fontsize=6.5, loc='center right')

# (b) the DFT error is the folded singular tail
n, h = 256, 1 / 256
rng = np.random.default_rng(1500)
c = rng.uniform(0.3, 0.7)
f = e.taper(PiecewiseExp.random(rng, [c]))
u = np.linspace(2, 0.49 * n, 400)
a = f.cell_avg(np.linspace(0, 1, n + 1))
err = e.dft_avg(a, u, h) / e.S(u, h) - f.ft(u)
m = np.r_[np.arange(-4000, 0), np.arange(1, 4001)]
xi = u[:, None] + m[None, :] * n
J, J1 = f.jumps()[0], e.dslope(f, c)
E = np.exp(-1j * TWO_PI * xi * c)
T2 = np.sum((-1.0) ** m * (J * E / (1j * TWO_PI * xi) + J1 * E / (1j * TWO_PI * xi) ** 2) * e.S(xi, h), 1) / e.S(u, h)
ax[1].semilogy(u / n, np.abs(f.ft(u)), color='#999', lw=0.8, label='$|F(u)|$')
ax[1].semilogy(u / n, np.abs(err), color='#d62', lw=1.0, label='error of the DFT of the averages')
ax[1].semilogy(u / n, np.abs(T2), 'k:', lw=1.0, label='folded tail of the jump ($c$, $J$, $J_1$ only)')
ax[1].set(xlabel='$u/n$', title='(b) the error is the folded singular tail')
ax[1].legend(fontsize=6.5, loc='lower left')

# (c), (d) error at u = n/4 versus resolution, 1-D and 2-D
sty = [('dft', '#d62', 'o', 'DFT of averages'), ('dft_hann', '#e9a', 'v', 'apodised DFT'),
       ('smooth', '#a85', 's', 'high degree, edges ignored'), ('shift_h10', '#777', 'x', 'exact geometry shifted by $h/10$'),
       ('safi', '#126', 'D', 'conservative fit + exact transform')]
NS = np.array(r['1d']['NS'])
q = r['1d']['RATIOS'].index(1 / 4)
for key, col, mk, lab in sty:
    ax[2].loglog(NS, np.array(r['1d']['scaling']['jump'][key])[:, q], '-' + mk, color=col, ms=3.5, lw=1, label=lab)
ax[2].set(xlabel='cells $n$', ylabel='relative error at $u\\approx n/4$', title='(c) one dimension, one jump')
ax[2].set_xscale('log', base=2)
ns2 = sorted(int(k) for k in r['2d'])
for key, col, mk, lab in sty:
    vals = [np.median([row[key][3] for row in r['2d'][str(k)] if key in row]) for k in ns2]
    ax[3].loglog(ns2, vals, '-' + mk, color=col, ms=3.5, lw=1, label=lab)
ax[3].set(xlabel='pixels per side $n$', title='(d) two dimensions, curved inclusion')
ax[3].set_xscale('log', base=2)
ax[3].legend(fontsize=6.2, loc='lower left')
plt.tight_layout()
plt.savefig('figs/mechanism2.pdf')
plt.savefig('figs/mechanism2.png', dpi=160)
