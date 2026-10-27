"""Exp 17: finite-rate-of-innovation (FRI) localisation versus SAFI from cell averages.

FRI recovers the jump positions of a piecewise-constant signal exactly from suitable samples.
Two FRI estimators are applied to box (cell-average) data:
  spectral FRI : annihilating filter (Prony / total least squares) on the low-order Fourier
                 coefficients of the derivative, y_k = 2 pi i k F_hat(k), F_hat from the DFT of the
                 averages corrected by the box transfer function, |k| <= K;
  local FRI    : the Strang-Fix moment formula for one step per support with the box kernel,
                 c = x_{j+1} - h (a_j - a_{j-1}) / (a_{j+1} - a_{j-1}); for box data this is the
                 conservation equation with constant extrapolants (SAFI-cell with M = 0).
SAFI-cell uses extrapolants of degree M = 6 in the same conservation equation.
Signals: piecewise constant (the FRI model) and piecewise exponential-trigonometric (smooth,
non-polynomial pieces), 1-3 jumps, n = 64..1024, noise sigma in {0, 1e-6, 1e-3}.
Metric: median position error; the induced 1e-6 horizon is about 1e-6 / (2 pi delta).
"""
import sys, json
sys.path.insert(0, '.')
import numpy as np
from math import comb
from bench import PiecewiseExp
from safi_cell import SAFICell

TWO_PI = 2 * np.pi
NS = [64, 128, 256, 512, 1024]
SIG = [0.0, 1e-6, 1e-3]
NF = 20


def pw_const(rng, cs):
    br = np.r_[0, cs, 1]
    A = [np.array([rng.uniform(-2, 2) + 0j]) for _ in range(br.size - 1)]
    B = [np.array([0j]) for _ in range(br.size - 1)]
    return PiecewiseExp(br, A, B)


def fri_spectral(a, n, L, K):
    """Positions of L jumps from y_k = 2 pi i k F(k), k = -K..K (F from deconvolved DFT)."""
    h = 1.0 / n
    k = np.arange(-K, K + 1)
    xc = (np.arange(n) + 0.5) * h
    F = h * np.exp(-1j * TWO_PI * np.outer(k, xc)) @ a / np.sinc(k / n)
    # derivative of f on the periodic interval: Diracs at the jumps plus one at 0 (f(0) - f(1))
    y = 1j * TWO_PI * k * F
    Lp = L + 1
    # annihilating filter of length Lp+1 by total least squares on the Toeplitz system
    T = np.array([y[i:i + Lp + 1][::-1] for i in range(y.size - Lp)])
    _, _, Vh = np.linalg.svd(T)
    hcoef = np.conj(Vh[-1])
    z = np.roots(hcoef)
    t = np.mod(-np.angle(z) / TWO_PI, 1.0)
    t = np.sort(t)
    # drop the boundary Dirac (closest to 0 or 1)
    d0 = np.minimum(t, 1 - t)
    t = np.delete(t, np.argmin(d0))
    return np.sort(t)


def fri_local(a, xe, cells):
    h = xe[1] - xe[0]
    out = []
    for j in cells:
        if 1 <= j < a.size - 1 and a[j + 1] != a[j - 1]:
            th = (a[j] - a[j - 1]) / (a[j + 1] - a[j - 1])
            out.append(xe[j + 1] - h * np.clip(th, 0, 1))
        else:
            out.append(0.5 * (xe[j] + xe[j + 1]))
    return np.array(out)


def match(est, cs):
    est = np.asarray(est)
    if est.size != len(cs):
        return np.nan
    return float(np.max(np.abs(np.sort(est) - np.sort(cs))))


if __name__ == '__main__':
    res = {}
    for kind in ('const', 'smooth'):
        for sig in SIG:
            E = {m: np.full((len(NS), NF), np.nan) for m in ('fri_spectral', 'fri_local', 'safi', 'safi_m2')}
            for a_, n in enumerate(NS):
                xe = np.linspace(0, 1, n + 1)
                for b in range(NF):
                    rng = np.random.default_rng(1700 + 37 * b)
                    L = int(rng.integers(1, 4))
                    while True:
                        cs = np.sort(rng.uniform(0.15, 0.85, L))
                        if L == 1 or np.min(np.diff(cs)) > 0.1:
                            break
                    f = pw_const(rng, cs) if kind == 'const' else PiecewiseExp.random(rng, cs)
                    a = f.cell_avg(xe) + sig * np.random.default_rng(b).normal(size=n)
                    try:
                        E['fri_spectral'][a_, b] = match(fri_spectral(a, n, L, min(n // 4, 64)), cs)
                    except Exception:
                        pass
                    # noise floor for differences of order M+1 (as in the point mode): 6 sigma binom(2M+2, M+1)^(1/2)
                    fl = max(1e-9, 6 * sig * np.sqrt(comb(14, 7)) / np.max(np.abs(a)))
                    S = SAFICell(xe, a, M=6, floor=fl)
                    cells = sorted(br.j for br in S.breaks)
                    E['fri_local'][a_, b] = match(fri_local(a, xe, cells), cs)
                    E['safi'][a_, b] = match([br.cstar for br in S.breaks], cs)
                    # lower degree for noisy data: differences of order 3 amplify noise 4.5x, not 59x
                    fl2 = max(1e-9, 6 * sig * np.sqrt(comb(6, 3)) / np.max(np.abs(a)))
                    S2 = SAFICell(xe, a, M=2, floor=fl2)
                    E['safi_m2'][a_, b] = match([br.cstar for br in S2.breaks], cs)
                    if np.isnan(E['fri_local'][a_, b]) and len(S2.breaks) == len(cs):
                        E['fri_local'][a_, b] = match(fri_local(a, xe, sorted(br.j for br in S2.breaks)), cs)
                print(kind, sig, n, {m: '%.1e' % np.nanmedian(v[a_]) for m, v in E.items()},
                      'found', {m: int(np.isfinite(v[a_]).sum()) for m, v in E.items()}, flush=True)
            res[f'{kind}_{sig:g}'] = {m: dict(median=np.nanmedian(v, 1).tolist(), found=np.isfinite(v).sum(1).tolist())
                                      for m, v in E.items()}
    json.dump(dict(NS=NS, SIG=SIG, NF=NF, res=res), open('results/exp17_fri.json', 'w'), indent=1)
