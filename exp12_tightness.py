"""Exp 12: how close is SAFI's localisation to the information limit?

For the two localisable cells of the identifiability table that SAFI uses most,
  (r,k) = (1,0): kink from point samples      -> extrapolation of degree M from M+1 samples
  (r,k) = (0,1): jump from cell averages      -> primitive of degree M+1 from M+2 nodes,
we build, for several positions theta of the singularity inside its cell,

  (a) a FOOLING PAIR f_+, f_- in the class (|f^{(M+1)}| <= K on every piece) whose data are
      identical but whose singularities are 2 rho_lo apart, from the one-sided optimal-recovery
      extremals of optrec.py.  Every method errs by >= rho_lo on one of them (information radius).
  (b) a WORST CASE FOR SAFI: the polynomial perturbation K*omega/m! that vanishes at SAFI's stencil
      nodes; SAFI's location error on it equals rho_up, the constant in our upper bound.

rho_lo = (E_m(theta) + E_m(1-theta)) K h^m / |J|,  rho_up = (e_m(theta) + e_m(1-theta)) K h^m / |J|
with m = M+1 (kink, points) or m = M+2 (jump, cells).
"""
import sys, json
sys.path.insert(0, '.')
import numpy as np
from math import factorial
from scipy.optimize import brentq
from optrec import Extremal, e_poly
from safi import SAFI
from safi_cell import SAFICell

M = 6
K = 1.0e5
THETAS = [0.2, 1 / 3, 3 / 7, 0.5, 4 / 7, 2 / 3, 0.8]   # includes the zeros k/M (kinks) and k/(M+1) (cell jumps) of the order-(M+1) ENO kernel
res = {}


def omega(x, nodes):
    return np.prod([x - z for z in nodes], axis=0)


# ------------------------------------------------------------------ (1,0): kink, point samples
m = M + 1
N = 2 * m + 1                      # samples on each side of the singular cell
n = 2 * N - 1                      # cells: left nodes x_0..x_{N-1}, right nodes x_N..x_{2N-1}
h = 1.0 / n
x = np.linspace(0, 1, n + 1)
i = N - 1
J1 = 1.0
fL = lambda z: 0.4 + 0.3 * z
rows = []
for th in THETAS:
    c = x[i] + th * h
    fR = lambda z, c=c: fL(z) + J1 * (z - c)
    gL, gR = Extremal(m, th, N=N), Extremal(m, 1 - th, N=N)
    sL = lambda z, s: -s * K * h ** m * gL((z - x[i]) / h)
    sR = lambda z, s: s * K * h ** m * gR((x[i + 1] - z) / h)
    cs, ys = {}, {}
    for s in (+1, -1):
        cs[s] = brentq(lambda z: (fR(z) + sR(z, s)[0]) - (fL(z) + sL(z, s)[0]), x[i], x[i + 1], xtol=1e-16, rtol=1e-15)
        ys[s] = np.r_[fL(x[:i + 1]) + sL(x[:i + 1], s), fR(x[i + 1:]) + sR(x[i + 1:], s)]
    data_diff = float(np.max(np.abs(ys[1] - ys[-1])))
    sep_half = abs(cs[1] - cs[-1]) / 2
    rho_lo = (gL.value + gR.value) * K * h ** m / J1
    # SAFI on the fooling pair (identical data -> identical estimate)
    S = SAFI(x, ys[1], M, mode='point')
    chat = S.breaks[0].c
    err_pair = max(abs(chat - cs[1]), abs(chat - cs[-1]))
    # worst case for SAFI: perturbations vanishing at SAFI's stencils, opposite signs on the two sides
    stL, stR = x[i - M:i + 1], x[i + 1:i + 2 + M]
    pL = lambda z: -K * omega(z, stL) / factorial(m)
    pR = lambda z: K * omega(z, stR) / factorial(m) * (-1) ** m
    cw = brentq(lambda z: (fR(z) + pR(z)) - (fL(z) + pL(z)), x[i], x[i + 1], xtol=1e-16, rtol=1e-15)
    yw = np.r_[fL(x[:i + 1]) + pL(x[:i + 1]), fR(x[i + 1:]) + pR(x[i + 1:])]
    Sw = SAFI(x, yw, M, mode='point')
    err_w = abs(Sw.breaks[0].c - cw)
    rho_up = (e_poly(m, th) + e_poly(m, 1 - th)) * K * h ** m / J1
    rows.append(dict(theta=th, data_diff=data_diff, sep_half=sep_half, rho_lo=rho_lo, safi_err_pair=err_pair,
                     safi_err_worst=err_w, rho_up=rho_up, kind=[S.breaks[0].kind, Sw.breaks[0].kind]))
    print('kink/points th=%.2f  data diff %.1e  half-sep %.4e rho_lo %.4e | SAFI worst %.4e rho_up %.4e  ratio %.3f'
          % (th, data_diff, sep_half, rho_lo, err_w, rho_up, err_w / sep_half), rows[-1]['kind'], flush=True)
res['kink_points'] = dict(M=M, m=m, n=n, K=K, J1=J1, rows=rows)

# ------------------------------------------------------------------ (0,1): jump, cell averages
m = M + 2
N = 2 * m + 1                      # primitive nodes on each side
n = 2 * N - 1
h = 1.0 / n
xe = np.linspace(0, 1, n + 1)
i = N - 1                          # jump in cell [x_i, x_{i+1}]
J = 1.0
FL = lambda z: 0.4 * z + 0.15 * z ** 2         # primitive of f_L = 0.4 + 0.3 z
rows = []
for th in THETAS:
    c = xe[i] + th * h
    FR = lambda z, c=c: FL(z) + J * (z - c)    # primitive of f_R = f_L + J (continuous at c)
    gL, gR = Extremal(m, th, N=N), Extremal(m, 1 - th, N=N)
    SL = lambda z, s: -s * K * h ** m * gL((z - xe[i]) / h)
    SR = lambda z, s: s * K * h ** m * gR((xe[i + 1] - z) / h)
    cs, ab = {}, {}
    for s in (+1, -1):
        # mass balance in the singular cell: (S_L - S_R)(c') = J (c' - c)
        cs[s] = brentq(lambda z: (SL(z, s)[0] - SR(z, s)[0]) - J * (z - c), xe[i], xe[i + 1], xtol=1e-16, rtol=1e-15)
        cp = cs[s]
        # primitive of f_s, continuous: left part FL+SL up to c', right part FR+SR shifted to match at c'
        V = lambda z, s=s, cp=cp: np.where(z <= cp, FL(z) + SL(z, s), FR(z) + SR(z, s) - (FR(cp) + SR(cp, s)[0]) + (FL(cp) + SL(cp, s)[0]))
        Vn = V(xe)
        ab[s] = np.diff(Vn) / h
    data_diff = float(np.max(np.abs(ab[1] - ab[-1])))
    sep_half = abs(cs[1] - cs[-1]) / 2
    rho_lo = (gL.value + gR.value) * K * h ** m / J
    C = SAFICell(xe, ab[1], M=M)
    chat = C.breaks[0].c
    err_pair = max(abs(chat - cs[1]), abs(chat - cs[-1]))
    # worst case for SAFI-cell: primitive perturbations vanishing at its stencils (M+2 nodes each side)
    stL, stR = xe[i - M - 1:i + 1], xe[i + 1:i + 3 + M]
    PL = lambda z: -K * omega(z, stL) / factorial(m)
    PR = lambda z: K * omega(z, stR) / factorial(m) * (-1) ** m
    cw = brentq(lambda z: (PL(z) - PR(z)) - J * (z - c), xe[i], xe[i + 1], xtol=1e-16, rtol=1e-15)
    Vw = np.where(xe <= cw, FL(xe) + PL(xe), FR(xe) + PR(xe) - (FR(cw) + PR(cw)) + (FL(cw) + PL(cw)))
    Cw = SAFICell(xe, np.diff(Vw) / h, M=M)
    err_w = abs(Cw.breaks[0].c - cw)
    rho_up = (e_poly(m, th) + e_poly(m, 1 - th)) * K * h ** m / J
    rows.append(dict(theta=th, data_diff=data_diff, sep_half=sep_half, rho_lo=rho_lo, safi_err_pair=err_pair,
                     safi_err_worst=err_w, rho_up=rho_up))
    print('jump/cells  th=%.2f  data diff %.1e  half-sep %.4e rho_lo %.4e | SAFI worst %.4e rho_up %.4e  ratio %.3f'
          % (th, data_diff, sep_half, rho_lo, err_w, rho_up, err_w / sep_half), flush=True)
res['jump_cells'] = dict(M=M, m=m, n=n, K=K, J=J, rows=rows)

# ------------------------------------------------------------------ constants across degrees
tab = []
for mm in (5, 6, 7, 8):
    for th in (0.25, 0.5, 0.75):
        g = Extremal(mm, th, N=2 * mm + 1)
        tab.append(dict(m=mm, theta=th, E=g.value, e=e_poly(mm, th), ratio=e_poly(mm, th) / g.value))
res['constants'] = tab
json.dump(res, open('results/exp12_tightness.json', 'w'), indent=1)
for t in tab:
    print('m=%d theta=%.2f E=%.4e e=%.4e ratio=%.3f' % (t['m'], t['theta'], t['E'], t['e'], t['ratio']))
