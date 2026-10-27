"""Exp 10: spectrum of a shock solution beyond the grid Nyquist, from finite-volume data.

Inviscid Burgers, u0 = 0.3 + sin(2 pi x), periodic, T = 0.4 (shock at x = 0.62), WENO5-JS + SSP-RK3.
Data = the scheme's cell averages (shock smeared over a few cells).
Compared:  FFT of the averages with sinc deconvolution (standard; k <= N/2 only),
           SAFI-cell single-cell subcell, SAFI-cell zone mass balance (zones=True),
           and SAFI-cell on the exact cell averages (what perfect data would give).
Reference: exact entropy-solution Fourier coefficients via the Lagrangian integral.
"""
import sys, json, time
sys.path.insert(0, '.')
import numpy as np
from burgers import solve_fv, exact_coeffs, shock_position, exact_cell_averages
from safi_cell import SAFICell

T = 0.4
TWO_PI = 2 * np.pi
out = {}


ks_all = np.unique(np.round(np.logspace(0, np.log10(60000), 160)).astype(int))


def midpoint_crossing(xe, ab, cs, N):
    j = int(cs * N); seg = ab[j - 6:j + 7]; mid = 0.5 * (seg[:3].mean() + seg[-3:].mean())
    xm = 0.5 * (xe[:-1] + xe[1:])[j - 6:j + 7]
    k = np.where(np.sign(seg[:-1] - mid) != np.sign(seg[1:] - mid))[0][0]
    return xm[k] + (mid - seg[k]) * (xm[k + 1] - xm[k]) / (seg[k + 1] - seg[k])


for m in [0.3, 0.0]:
    Fex = exact_coeffs(ks_all, T, m=m)
    cs = shock_position(T, m)
    for N in [128, 256, 512, 1024, 2048]:
        xe, ab = solve_fv(N, T, m=m)
        kk = ks_all.astype(float)
        fft = np.fft.fft(ab) / N
        kin = ks_all[ks_all <= N // 2]
        std = fft[kin] * np.exp(-1j * np.pi * kin / N) / np.sinc(kin / N)
        Sz = SAFICell(xe, ab, M=6, zones=True, auto_jump_degree=True)
        S6 = SAFICell(xe, ab, M=6, zones=True)
        xe2, abx = exact_cell_averages(N, T, m=m)
        Se = SAFICell(xe2, abx, M=6, zones=True)
        Gz, G6, Ge = Sz(kk), S6(kk), Se(kk)
        bands = {'k<=N/2': ks_all <= N // 2, 'N/2<k<=10N': (ks_all > N // 2) & (ks_all <= 10 * N), 'k>10N': ks_all > 10 * N}
        rel = lambda G, s: float(np.linalg.norm(Fex[s] - G[s]) / np.linalg.norm(Fex[s])) if s.any() else None
        amp = lambda G, s: float(np.linalg.norm(np.abs(Fex[s]) - np.abs(G[s])) / np.linalg.norm(Fex[s])) if s.any() else None
        res = {'bands': {}}
        for name, s in bands.items():
            r = dict(safi_auto=rel(Gz, s), safi_auto_amp=amp(Gz, s), safi_M6=rel(G6, s),
                     safi_exact_data=rel(Ge, s))
            if name == 'k<=N/2':
                r['std_fft'] = float(np.linalg.norm(Fex[s] - std) / np.linalg.norm(Fex[s]))
                r['std_fft_amp'] = float(np.linalg.norm(np.abs(Fex[s]) - np.abs(std)) / np.linalg.norm(Fex[s]))
            res['bands'][name] = r
        res['shock_err'] = dict(safi_auto=float(abs(Sz.breaks[0].cstar - cs)) if Sz.breaks else None,
                                safi_auto_delta=float(Sz.breaks[0].delta) if Sz.breaks else None,
                                safi_M6=float(abs(S6.breaks[0].cstar - cs)) if S6.breaks else None,
                                midpoint=float(abs(midpoint_crossing(xe, ab, cs, N) - cs)),
                                exact_data=float(abs(Se.breaks[0].cstar - cs)) if Se.breaks else None)
        sel = (ks_all > N) & (ks_all < 50 * N)
        res['tail_slope'] = dict(safi_auto=float(np.polyfit(np.log(kk[sel]), np.log(np.abs(Gz[sel]) ** 2), 1)[0]),
                                 exact=float(np.polyfit(np.log(kk[sel]), np.log(np.abs(Fex[sel]) ** 2), 1)[0]))
        out[f'm{m}_N{N}'] = res
        if N == 512:
            out[f'curves_m{m}'] = dict(k=ks_all.tolist(), exact=np.abs(Fex).tolist(), safi=np.abs(Gz).tolist(),
                                       safi_err=np.abs(Fex - Gz).tolist(), std_k=kin.tolist(), std=np.abs(std).tolist())
        print(f'm={m} N={N}', json.dumps(res), flush=True)

json.dump(out, open('results/exp10_shock.json', 'w'), indent=1, default=str)

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
c = out['curves_m0.3']; k = np.array(c['k'], float)
ax[0].loglog(k, np.array(c['exact']) ** 2, 'k', lw=1.2, label='exact entropy solution')
ax[0].loglog(c['std_k'], np.array(c['std']) ** 2, 'o', ms=2, color='#888', label='FFT of FV averages (k <= N/2)')
ax[0].loglog(k, np.array(c['safi']) ** 2, '--', color='#126', label='SAFI-cell (zone, auto degree)')
ax[0].loglog(k, 0.03 * k ** -2.0, 'k:', lw=1, label='k^-2')
ax[0].axvline(256, color='k', ls=':', lw=.7)
ax[0].set(xlabel='k', ylabel='|u_k|^2', title='moving shock, WENO5, N = 512'); ax[0].legend(fontsize=7)
for mm, col in [(0.3, '#d62'), (0.0, '#126')]:
    c = out[f'curves_m{mm}']
    ax[1].loglog(c['k'], np.array(c['safi_err']) / np.array(c['exact']), color=col, lw=.8, label=f'm={mm}')
ax[1].axvline(256, color='k', ls=':', lw=.7)
ax[1].set(xlabel='k', ylabel='|u_k - u_k^exact| / |u_k^exact|', title='SAFI-cell coefficient error, N=512'); ax[1].legend(fontsize=7)
NS = [128, 256, 512, 1024, 2048]
for key, col, lab in [('safi_auto', '#126', 'SAFI zone, auto degree'), ('safi_M6', '#7ad', 'SAFI zone, M=6'),
                      ('midpoint', '#888', 'midpoint crossing')]:
    ax[2].loglog(NS, [out[f'm0.3_N{N}']['shock_err'][key] for N in NS], '-o', color=col, label=lab)
ax[2].loglog(NS, 2.0 * np.array(NS, float) ** -2, 'k:', lw=1, label='N^-2')
ax[2].set(xlabel='N', ylabel='|shock position error|', title='moving shock location from WENO data'); ax[2].legend(fontsize=7)
plt.tight_layout(); plt.savefig('figs/exp10_shock.png', dpi=140)
