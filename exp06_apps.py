"""Exp 6: three applications, one per data mode.

A (adaptive)  switched RLC transient from a black-box simulator, switch times unknown;
              spectrum on a log grid 1 .. 1e6 (cycles per record), target eps = 1e-6.
B (cell)      image row recorded by integrating pixels (n = 256), reflectance edges at unknown
              sub-pixel positions under smooth illumination; spectrum up to 100x pixel Nyquist.
C (known)     1-D aperture: Hann taper x designed binary/quaternary phase mask (step positions
              known by design), field measured at n = 256 points; far field on radiation circles
              u = sin(theta)/lambda for three wavelengths, one fit for all.
"""
import sys, json, time
sys.path.insert(0, '.')
import numpy as np
from scipy.integrate import quad
from bench import PiecewiseExp
from safi import SAFI, safi_auto, riemann_sum
from safi_cell import SAFICell

TWO_PI = 2 * np.pi
rel = lambda F, G: float(np.linalg.norm(F - G) / np.linalg.norm(F))
out = {}

# ------------------------------------------------------------------ A: switched transient
rng = np.random.default_rng(8)
ts = np.sort(rng.uniform(0.1, 0.9, 3))
br = np.r_[0, ts, 1]
A, B = [], []
for k in range(4):
    V = [1.0, -0.6, 0.8, -0.3][k]
    # RC decay + damped ringing (RLC) after each switching event, written as exponentials in t
    tau, a, w0 = 0.02, 8.0, TWO_PI * 37.0
    A.append([V * np.exp(br[k] / tau), 0.4 * V * np.exp((a - 1j * w0) * br[k]), 0.15])
    B.append([-1 / tau, -a + 1j * w0, 0.0])
fA = PiecewiseExp(br, A, B)
uA = np.logspace(0, 6, 241)
FA = fA.ft(uA)
t0 = time.time()
S, Fh, eh, st = safi_auto(fA, 0, 1, uA, 1e-6, M=8, n0=128)
tA = time.time() - t0
# log-grid relative error: normalise per decade
dec = np.floor(np.log10(uA)).clip(max=5)
rdec = lambda G: max(np.linalg.norm((FA - G)[dec == d]) / np.linalg.norm(FA[dec == d]) for d in np.unique(dec))
xs = lambda n: np.linspace(0, 1, n + 1)
budget = st['nf']
resA = dict(switch_times=ts.tolist(), safi_evals=budget, safi_true=rdec(Fh), safi_est=float(max(
    np.linalg.norm(eh[dec == d]) / np.linalg.norm(Fh[dec == d]) for d in np.unique(dec))), safi_time_s=tA,
    breaks_found=[b.c for b in S.breaks])
nb = 2 ** int(np.ceil(np.log2(budget)))
resA['trap_same_budget'] = rdec(riemann_sum(xs(nb), fA(xs(nb)), uA))
resA['trap_2^20'] = rdec(riemann_sum(xs(2 ** 20), fA(xs(2 ** 20)), uA))
resA['hp_unaware_2^14'] = rdec(SAFI(xs(2 ** 14), fA(xs(2 ** 14)), 8, 'none')(uA))
nev = 0; G = np.empty_like(FA); sub = np.arange(0, uA.size, 10)
for i in sub:
    w = TWO_PI * uA[i]; g = lambda t: fA(np.array([t]))[0]
    rc = quad(g, 0, 1, weight='cos', wvar=w, epsabs=5e-7 * abs(FA[i]), epsrel=0, limit=5000, full_output=1)
    rs = quad(g, 0, 1, weight='sin', wvar=w, epsabs=5e-7 * abs(FA[i]), epsrel=0, limit=5000, full_output=1)
    nev += rc[2]['neval'] + rs[2]['neval']; G[i] = rc[0] - 1j * rs[0]
resA['quadpack_evals_per_freq'] = nev / sub.size
resA['quadpack_evals_all_241'] = nev / sub.size * uA.size
resA['quadpack_worst_rel'] = float(np.max(np.abs(FA[sub] - G[sub]) / np.abs(FA[sub])))
out['A_switched_transient'] = resA
print('A', resA, flush=True)

# ------------------------------------------------------------------ B: pixel row
rng = np.random.default_rng(21)
edges = np.sort(rng.uniform(0.12, 0.88, 3))
br = np.r_[0, edges, 1]
refl = [0.2, 0.9, 0.35, 0.7]
A, B = [], []
for k in range(4):
    # illumination 1 + 0.3 cos(2 pi 1.7 x) + 0.2 exp(-x), times reflectance
    A.append([refl[k], 0.3 * refl[k], 0.2 * refl[k]]); B.append([0.0, 1j * TWO_PI * 1.7, -1.0])
fB = PiecewiseExp(br, A, B)
n = 256; xe = np.linspace(0, 1, n + 1); ab = fB.cell_avg(xe)
uB = np.r_[np.linspace(1, 128, 128), np.logspace(np.log10(128), np.log10(12800), 200)]
FB = fB.ft(uB)
Sc = SAFICell(xe, ab, M=6)
xm = 0.5 * (xe[:-1] + xe[1:])
pix_fft = (np.exp(-1j * TWO_PI * np.outer(uB, xm)) @ ab) / n
Su = SAFICell.__new__(SAFICell); Su.xe, Su.ab, Su.M, Su.h = xe, ab, 6, 1 / n
Su.V = np.concatenate([[0.0], np.cumsum(ab) / n]); Su.breaks = []; Su._build()
GB = Sc(uB)
seg = {'below Nyquist (u<=128)': uB <= 128, '1-10x Nyquist': (uB > 128) & (uB <= 1280), '10-100x Nyquist': uB > 1280}
resB = dict(edges=edges.tolist(), edges_found=[b.cstar for b in Sc.breaks],
            loc_err=[float(abs(b.cstar - e)) for b, e in zip(Sc.breaks, edges)])
for name, s in seg.items():
    resB[name] = dict(pixel_sum=rel(FB[s], pix_fft[s]), fv_unaware=rel(FB[s], Su(uB)[s]), safi_cell=rel(FB[s], GB[s]))
out['B_pixel_row'] = resB
print('B', resB, flush=True)

# ------------------------------------------------------------------ C: phase-mask aperture
steps = np.array([-0.5, -0.27, -0.05, 0.18, 0.31, 0.5])
phase = np.exp(1j * np.pi / 2 * np.array([0, 1, 3, 2, 1]))
A, B = [], []
for k in range(5):
    A.append(phase[k] * np.array([0.5, 0.25, 0.25])); B.append([0.0, 1j * TWO_PI, -1j * TWO_PI])
fC = PiecewiseExp(steps, A, B, real=False)
n = 256; x = np.linspace(-0.5, 0.5, n + 1); y = fC(x)
Sk = SAFI(x, y, 6, 'known', breaks=list(steps[1:-1]))
Sp = SAFI(x, y, 6, 'point')
Sn = SAFI(x, y, 6, 'none')
th = np.linspace(-np.pi / 2, np.pi / 2, 721)
resC = {}
for inv_lam in [32, 256, 2048]:
    uC = inv_lam * np.sin(th)
    FC = fC.ft(uC)
    resC[f'aperture/lambda={inv_lam}'] = dict(trap=rel(FC, riemann_sum(x, y, uC)), hp_unaware=rel(FC, Sn(uC)),
                                              safi_point=rel(FC, Sp(uC)), safi_known=rel(FC, Sk(uC)))
out['C_phase_aperture'] = resC
print('C', resC, flush=True)
json.dump(out, open('results/exp06_apps.json', 'w'), indent=1, default=float)

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
ax[0].loglog(uA, np.abs(FA), 'k', lw=1, label='|F| exact')
ax[0].loglog(uA, np.abs(FA - Fh), color='#126', label=f'SAFI-auto err ({budget} evals)')
ax[0].loglog(uA, eh, '--', color='#126', lw=.8, label='SAFI estimate')
ax[0].loglog(uA, np.abs(FA - riemann_sum(xs(nb), fA(xs(nb)), uA)), color='#888', label=f'trapezoid n={nb}')
ax[0].set(xlabel='frequency (cycles/record)', title='A: switched transient, unknown switch times'); ax[0].legend(fontsize=7)
ax[1].loglog(uB, np.abs(FB), 'k', lw=1, label='|F| exact')
ax[1].loglog(uB, np.abs(FB - pix_fft), color='#888', label='pixel sum')
ax[1].loglog(uB, np.abs(FB - Su(uB)), color='#c77', label='FV, edge-unaware')
ax[1].loglog(uB, np.abs(FB - GB), color='#126', label='SAFI cell')
ax[1].axvline(128, color='k', ls=':', lw=.7)
ax[1].set(xlabel='frequency (cycles/row)', title='B: 256 integrating pixels'); ax[1].legend(fontsize=7)
uC = 2048 * np.sin(th); FC = fC.ft(uC)
ax[2].semilogy(np.degrees(th), np.abs(FC), 'k', lw=1, label='|E| exact')
ax[2].semilogy(np.degrees(th), np.abs(FC - riemann_sum(x, y, uC)), color='#888', lw=.8, label='trapezoid / NUFFT')
ax[2].semilogy(np.degrees(th), np.abs(FC - Sn(uC)), color='#c77', lw=.8, label='hp, steps ignored')
ax[2].semilogy(np.degrees(th), np.abs(FC - Sk(uC)), color='#2a5', lw=.8, label='SAFI known steps')
ax[2].set(xlabel='angle (deg)', title='C: phase-mask aperture, 1/lambda = 2048'); ax[2].legend(fontsize=7)
plt.tight_layout(); plt.savefig('figs/exp06_apps.png', dpi=140)
