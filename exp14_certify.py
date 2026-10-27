"""Exp 14: certification without a separation bound (Theorem verify, both directions).

Class P0^B: s = 0, |f| <= B, jumps in [J0, J1], |f^{(M+1)}| <= K.
Upper bound: degree-M interpolation on a uniform grid with fixed, centred stencils is sound on P0^B
with the a priori bound
    e_u <= K e_M g^{M+1} + 2M L (1 + Lam_M) B g           (all u; <= 2M cells per breakpoint),
so it certifies sup_{|u|<=U} e_hat_u <= eps once N ~ L B U/(eps J0).
Lower bound: +/- pulse trains of height J1 in the m largest unsampled gaps give two functions with
identical data whose transforms differ by >= (8/pi) m J1 g/(...); the measured difference is
compared with the bound of the theorem.
"""
import sys, json
sys.path.insert(0, '.')
import numpy as np
from math import factorial
from bench import PiecewiseExp
from safi import leg_ft_cells

TWO_PI = 2 * np.pi
M = 6


def stencil(j, n, M):
    s = min(max(j - M // 2, 0), n - M)            # nodes s..s+M contain x_j, x_{j+1}
    return np.arange(s, s + M + 1)


def constants(M, n=64, ngrid=400):
    """Lam_M = max cell Lebesgue constant, e_M = max |omega| / ((M+1)! g^{M+1}) over all cells."""
    lam = em = 0.0
    for j in range(n):
        S = stencil(j, n, M).astype(float)
        t = np.linspace(j, j + 1, ngrid)
        L = np.ones((S.size, t.size))
        for a in range(S.size):
            for b in range(S.size):
                if a != b:
                    L[a] *= (t - S[b]) / (S[a] - S[b])
        lam = max(lam, np.abs(L).sum(0).max())
        em = max(em, np.abs(np.prod([t - z for z in S], 0)).max() / factorial(M + 1))
    return lam, em


def interp_ft(y, n, M, u):
    """Exact FT of the piecewise interpolant (degree M on each cell) at frequencies u."""
    g = 1.0 / n
    x = np.arange(n + 1) * g
    F = np.zeros(u.size, complex)
    # Legendre coefficients of each cell polynomial from M+1 Gauss values
    tq, wq = np.polynomial.legendre.leggauss(M + 1)
    C = np.zeros((n, M + 1), complex)
    for j in range(n):
        S = stencil(j, n, M)
        xs = x[j] + (tq + 1) / 2 * g
        V = np.polynomial.polynomial.polyvander((xs - x[j]) / g, M) @ np.linalg.solve(
            np.polynomial.polynomial.polyvander((x[S] - x[j]) / g, M), y[S])
        P = np.polynomial.legendre.legvander(tq, M)
        C[j] = (P.T @ (wq * V)) * (2 * np.arange(M + 1) + 1) / 2
    return leg_ft_cells(x[:-1], x[1:], C, u)


if __name__ == '__main__':
    lam, em = constants(M)
    print('M', M, 'Lam_M %.3f e_M %.3e' % (lam, em))
    rng = np.random.default_rng(14)
    U, J0, J1, K = 2.0 ** 10, 1.0, 1.0, None
    rows = []
    for L in (2, 4, 8):
        for n in (256, 1024, 4096):
            g = 1.0 / n
            u = np.r_[np.linspace(1, U, 400)]
            for trial in range(8):
                base = PiecewiseExp.random(rng, [], nterm=3, osc=2.0, decay=1.0)
                xx = np.linspace(0, 1, 20001)
                B0 = np.max(np.abs(base(xx)))
                B = B0 + J1
                # K of the base: max |f^{(M+1)}| from the exponential terms
                Kf = float(np.max(np.abs(sum(a * b ** (M + 1) * np.exp(b * xx) for a, b in zip(base.alphas[0], base.betas[0])))))
                # hidden pulses in m = L/2 random cells, random widths and signs; data unchanged
                m = L // 2
                cells = rng.choice(np.arange(n), m, replace=False)
                pulses = []
                for j in cells:
                    w = g * rng.uniform(0.05, 0.95)
                    a = j * g + rng.uniform(0, g - w)
                    pulses.append((a, w, rng.choice([-1, 1]) * J1))
                x = np.arange(n + 1) * g
                y = base(x)                         # pulses lie strictly between nodes
                Fh = interp_ft(y, n, M, u)
                Ftrue = base.ft(u) + sum(s * (np.exp(-TWO_PI * 1j * u * a) - np.exp(-TWO_PI * 1j * u * (a + w))) / (TWO_PI * 1j * u)
                                         for a, w, s in pulses)
                err = np.max(np.abs(Ftrue - Fh))
                bound = Kf * em * g ** (M + 1) + 2 * M * L * (1 + lam) * B * g
                rows.append(dict(L=L, n=n, err=float(err), bound=float(bound), ratio=float(err / bound)))
            r = [z for z in rows if z['L'] == L and z['n'] == n]
            print(f'L={L} n={n}: max err {max(z["err"] for z in r):.2e}  bound {np.median([z["bound"] for z in r]):.2e}  max err/bound {max(z["ratio"] for z in r):.3f}', flush=True)
    # lower bound: two functions with identical data; transform difference at u = U vs theorem
    low = []
    for n in (2048, 4096, 8192):                  # g <= 1/(2U): the regime of the theorem
        g = 1.0 / n
        for m in (1, 2, 4):
            # pulses fill the m chosen gaps (width g minus a margin), signs aligned at u = U
            js = rng.choice(np.arange(n), m, replace=False)
            w = g * (1 - 1e-9)
            z = np.array([J1 * (np.exp(-TWO_PI * 1j * U * j * g) - np.exp(-TWO_PI * 1j * U * (j * g + w))) / (TWO_PI * 1j * U) for j in js])
            best = max(abs(np.sum(np.sign(np.real(z * np.exp(-1j * th))) * z)) for th in np.linspace(0, np.pi, 721))
            # two claimed functions differ by 2*best; theorem: best >= (4/pi^2) J1 m g
            low.append(dict(n=n, m=m, diff=float(2 * best), theorem=float(2 * 4 / np.pi ** 2 * J1 * m * g)))
            print(f'n={n} m={m}: |F+ - F-|(U) = {2 * best:.3e} >= theorem {2 * 4 / np.pi ** 2 * J1 * m * g:.3e}')
    json.dump(dict(M=M, Lam=lam, eM=em, upper=rows, lower=low), open('results/exp14_certify.json', 'w'), indent=1)
