"""Exp 15: why pixel/cell data give wrong spectra, and why the error can be removed.

Part 1 (1D, exact).  For cell averages a_j of f, the standard estimate (DFT of the averages,
corrected by the box transfer function S) satisfies, by Poisson summation,
    F_dft(u) = F(u) + T(u),   T(u) = sum_{m != 0} (-1)^m F(u + m n) S(u + m n) / S(u),
i.e. its error is the high-frequency tail of F folded into the band.  For piecewise smooth f
the tail is dominated by the singularities: F(xi) ~ sum_l J_l exp(-2 pi i xi c_l)/(2 pi i xi).
We check that T computed from (c_l, J_l) alone reproduces the actual error, and that the
dependence on c_l enters only through exp(-2 pi i m c_l / h): the sub-cell position.
Then: relative error at fixed u/n versus n for the DFT, an apodised DFT, the exact transform of
the box model, a high-order smooth reconstruction, SAFI-cell, SAFI with the true position
(oracle) and with the position shifted by h/10 and h/100; jump vs no jump; averages vs points.

Part 2 (2D).  The same comparison for pixel images with a curved inclusion (bench2d_curve).
"""
import sys, json, time
sys.path.insert(0, '.')
import numpy as np
from bench import PiecewiseExp
from safi_cell import SAFICell
from safi import SAFI

TWO_PI = 2 * np.pi
rng0 = np.random.default_rng(15)
RATIOS = [1 / 32, 1 / 16, 1 / 8, 1 / 4, 0.45]          # u / n below the Nyquist frequency n/2
BEYOND = [2.0, 8.0, 32.0]                              # u / n above it (continuous models only)
NS = [32, 64, 128, 256, 512, 1024]
NF = 16
PER = 12


def taper(f):
    """f * sin^4(pi x): kills the end-point singularities, keeps interior ones."""
    ck = {0: 3 / 8, 1: -1 / 4, -1: -1 / 4, 2: 1 / 16, -2: 1 / 16}
    A, B = [], []
    for al, be in zip(f.alphas, f.betas):
        A.append(np.concatenate([al * c for c in ck.values()]))
        B.append(np.concatenate([be + 1j * TWO_PI * k for k in ck.keys()]))
    return PiecewiseExp(f.breaks, A, B)


def dslope(f, c):
    """f'(c+) - f'(c-) for a PiecewiseExp with one interior break at c (real part convention)."""
    d = lambda j: np.real(np.sum(f.alphas[j] * f.betas[j] * np.exp(f.betas[j] * c)))
    return d(1) - d(0)


def S(xi, h):
    return np.sinc(xi * h)                              # sin(pi xi h)/(pi xi h)


def dft_avg(a, u, h):
    xc = (np.arange(a.size) + 0.5) * h
    return h * np.exp(-1j * TWO_PI * np.outer(u, xc)) @ a


def band(F, G):
    return float(np.linalg.norm(F - G) / np.linalg.norm(F))


def part1():
    out = {'folding': [], 'scaling': {}}
    funcs = []
    for k in range(NF):
        rng = np.random.default_rng(1500 + k)
        c = rng.uniform(0.3, 0.7)
        fj = taper(PiecewiseExp.random(rng, [c]))
        fs = taper(PiecewiseExp.random(np.random.default_rng(2500 + k), []))
        funcs.append((fj, fs, c))
    # ---- folding identity and its singular part
    for n in (64, 256):
        h = 1.0 / n
        for fj, fs, c in funcs[:6]:
            u = np.linspace(0.02, 0.45, 40) * n
            a = fj.cell_avg(np.linspace(0, 1, n + 1))
            err = dft_avg(a, u, h) / S(u, h) - fj.ft(u)
            m = np.r_[np.arange(-4000, 0), np.arange(1, 4001)]
            xi = u[:, None] + m[None, :] * n
            T_exact = np.sum((-1.0) ** m * fj.ft(xi.ravel()).reshape(xi.shape) * S(xi, h), 1) / S(u, h)
            J = fj.jumps()[0]
            J1 = dslope(fj, c)
            Fs = J * np.exp(-1j * TWO_PI * xi * c) / (1j * TWO_PI * xi)
            Fs2 = Fs + J1 * np.exp(-1j * TWO_PI * xi * c) / (1j * TWO_PI * xi) ** 2
            T_sing2 = np.sum((-1.0) ** m * Fs2 * S(xi, h), 1) / S(u, h)
            T_sing = np.sum((-1.0) ** m * Fs * S(xi, h), 1) / S(u, h)
            out['folding'].append(dict(n=n, c=c, theta=float(c * n % 1),
                                       err_vs_Texact=float(np.max(np.abs(err - T_exact)) / np.max(np.abs(err))),
                                       err_vs_Tsing=float(np.max(np.abs(err - T_sing)) / np.max(np.abs(err))),
                                       err_vs_Tsing2=float(np.max(np.abs(err - T_sing2)) / np.max(np.abs(err))),
                                       rel_err_dft=float(np.linalg.norm(err) / np.linalg.norm(fj.ft(u)))))
    # ---- scaling of the error at fixed u/n
    methods = ['dft', 'dft_hann', 'box', 'smooth', 'points_best', 'safi', 'oracle', 'shift_h10', 'shift_h100']
    for case in ('jump', 'smooth'):
        R = {mth: np.full((len(NS), NF, len(RATIOS) + len(BEYOND)), np.nan) for mth in methods}
        for a_, n in enumerate(NS):
            h = 1.0 / n
            xe = np.linspace(0, 1, n + 1)
            for b_, (fj, fs, c) in enumerate(funcs):
                f = fj if case == 'jump' else fs
                a = f.cell_avg(xe)
                rr = np.random.default_rng(7 * n + b_)
                for q, r in enumerate(RATIOS + BEYOND):
                    u = rr.uniform(r, 1.25 * r if r < 0.4 else 0.49, PER) * n if r < 1 else rr.uniform(r, 1.25 * r, PER) * n
                    F = f.ft(u)
                    if r < 1:
                        D = dft_avg(a, u, h)
                        R['dft'][a_, b_, q] = band(F, D / S(u, h))
                        R['dft_hann'][a_, b_, q] = band(F, D / S(u, h) * np.cos(np.pi * u / n) ** 2)
                    R['box'][a_, b_, q] = band(F, dft_avg(a, u, h))
                    try:
                        R['smooth'][a_, b_, q] = band(F, SAFICell(xe, a, M=6, breaks_known=[])(u))
                    except Exception:
                        pass
                    R['safi'][a_, b_, q] = band(F, SAFICell(xe, a, M=6)(u))
                    if case == 'jump':
                        R['oracle'][a_, b_, q] = band(F, SAFICell(xe, a, M=6, breaks_known=[c])(u))
                        R['shift_h10'][a_, b_, q] = band(F, SAFICell(xe, a, M=6, breaks_known=[c + h / 10])(u))
                        R['shift_h100'][a_, b_, q] = band(F, SAFICell(xe, a, M=6, breaks_known=[c + h / 100])(u))
                        # point values at the same nodes: no method can beat the break at the cell midpoint
                        y = f(xe)
                        R['points_best'][a_, b_, q] = band(F, SAFI(xe, y, 6, 'point')(u))
            print(case, n, {mth: '%.1e' % np.nanmedian(R[mth][a_, :, 3]) for mth in methods}, flush=True)
        out['scaling'][case] = {mth: np.nanmedian(R[mth], 1).tolist() for mth in methods}
    out.update(NS=NS, RATIOS=RATIOS, BEYOND=BEYOND)
    return out


def part2(ns=(32, 64, 128, 256), seeds=(1, 2, 3)):
    from bench2d_curve import CurveImage
    import safi2d as s2
    from safi2d_curve import CurveSAFI2D
    from curve2d import StarCurve
    res = {}
    for n in ns:
        rows = []
        rng = np.random.default_rng(3000 + n)
        r = np.concatenate([rng.uniform(rt, 1.25 * rt if rt < 0.4 else 0.49, 16) for rt in RATIOS]) * n
        lab = np.repeat(np.arange(len(RATIOS)), 16)
        th = rng.uniform(0, TWO_PI, r.size)
        u, v = r * np.cos(th), r * np.sin(th)
        h = 1.0 / n
        for seed in seeds:
            img = CurveImage.random(np.random.default_rng(seed))
            A = img.pixel_averages(n)
            F = img.ft(u, v)
            br = lambda G: [band(F[lab == k], G[lab == k]) for k in range(len(RATIOS))]
            xc = (np.arange(n) + 0.5) * h
            D = np.array([h * h * np.exp(-1j * TWO_PI * u[k] * xc) @ A @ np.exp(-1j * TWO_PI * v[k] * xc) for k in range(u.size)])
            Sx = np.sinc(u * h) * np.sinc(v * h)
            rec = dict(seed=seed, dft=br(D / Sx), dft_hann=br(D / Sx * np.cos(np.pi * u / n) ** 2 * np.cos(np.pi * v / n) ** 2),
                       box=br(D), smooth=br(s2.SAFI2D(A, P=4, edges=False)(u, v)))
            t0 = time.time()
            Sc = CurveSAFI2D(A, P=4, repair=False)
            ok = Sc.curve_fit()
            rec['fit_s'] = time.time() - t0
            if ok:
                rec['safi'] = br(Sc(u, v))
                C = img.curve
                for name, d in (('oracle', 0.0), ('shift_h10', h / 10), ('shift_h100', h / 100)):
                    Cd = StarCurve(C.c if hasattr(C, 'c') else C.center, (lambda t, d=d: C.rho(t) + d), C.drho, getattr(C, 'd2rho', None))
                    Sc.curve = Cd; Sc._geom = Sc._geometry(Cd); Sc.poly['band'] = Sc._geom['band']
                    rec[name] = br(Sc(u, v))
            rows.append(rec)
            print('2D', n, seed, {k: '%.1e' % v_[3] for k, v_ in rec.items() if isinstance(v_, list)}, flush=True)
        res[n] = rows
    return res


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    import os
    path = 'results/exp15_mechanism.json'
    out = json.load(open(path)) if os.path.exists(path) else {}
    if which in ('1d', 'all'):
        out['1d'] = part1()
        json.dump(out, open(path, 'w'), indent=1)
    if which in ('2d', 'all'):
        out['2d'] = {str(k): v for k, v in part2().items()}
        json.dump(out, open(path, 'w'), indent=1)
