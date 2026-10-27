"""Minimal type-2 NUFFT (Greengard-Lee Gaussian gridding), numpy only.

nufft2(c, theta): s_j = sum_{k=0}^{n-1} c_k exp(-i k theta_j),  theta_j arbitrary real.
Derivation: with the periodic heat kernel g(t) = sum_p exp(-(t-2 pi p)^2/(4 tau)), whose Fourier
coefficients are sqrt(tau/pi) exp(-k^2 tau), and a_k = c_k exp(tau k^2) (modes centred),
  s(theta) = sqrt(pi/tau) (1/2pi) int F(t) g(theta - t) dt,   F(t) = sum_k a_k e^{-ikt},
discretised on an R-times oversampled grid (FFT) and a 2*msp-point Gaussian stencil.
Oversampling R = 2; msp = 12 gives ~1e-12 relative accuracy. Cost O(n log n + n_u msp).
"""
import numpy as np


def nufft2(c, theta, msp=12, R=2):
    c = np.asarray(c, complex)
    theta = np.asarray(theta, float)
    n = c.size
    k0 = n // 2
    kk = np.arange(n) - k0                                  # centred modes
    Mr = max(int(R * n), 2 * msp + 2)
    Mr += Mr % 2
    tau = np.pi * msp / (n * n * R * (R - 0.5)) if n > 1 else 1.0 / Mr
    arr = np.zeros(Mr, complex)
    arr[kk % Mr] = c * np.exp(tau * kk.astype(float) ** 2)
    Fl = np.fft.fft(arr)                                    # F(t_l) = sum_k a_k e^{-i k t_l}
    h = 2 * np.pi / Mr
    th = np.mod(theta, 2 * np.pi)
    m0 = np.floor(th / h).astype(int)
    offs = np.arange(-msp + 1, msp + 1)
    gl = m0[:, None] + offs[None, :]
    d = th[:, None] - gl * h
    w = np.exp(-d * d / (4 * tau))
    s = (Fl[gl % Mr] * w).sum(1) * np.sqrt(np.pi / tau) / Mr
    return s * np.exp(-1j * k0 * theta)


def nufft1_2d(x, y, c, N, msp=8, R=2, chunk=20000):
    """Type-1 2-D NUFFT: F[k1, k2] = sum_j c_j exp(-2 pi i (k1 x_j + k2 y_j)), k = -N/2..N/2-1.
    Gaussian gridding (Greengard-Lee) on an R-times oversampled grid with a (2 msp)^2 stencil;
    msp = 8 gives about 1e-9 relative accuracy, msp = 12 about 1e-12."""
    x = np.asarray(x, float).ravel(); y = np.asarray(y, float).ravel(); c = np.asarray(c, complex).ravel()
    Mr = max(int(R * N), 2 * msp + 2)
    Mr += Mr % 2
    tau = np.pi * msp / (N * N * R * (R - 0.5))
    hg = 2 * np.pi / Mr
    offs = np.arange(-msp + 1, msp + 1)
    gre = np.zeros(Mr * Mr); gim = np.zeros(Mr * Mr)
    for s in range(0, x.size, chunk):
        tx = np.mod(TWO_PI_ * x[s:s + chunk], TWO_PI_); ty = np.mod(TWO_PI_ * y[s:s + chunk], TWO_PI_)
        mx = np.floor(tx / hg).astype(int); my = np.floor(ty / hg).astype(int)
        gx = mx[:, None] + offs[None, :]; gy = my[:, None] + offs[None, :]
        wx = np.exp(-(tx[:, None] - gx * hg) ** 2 / (4 * tau))
        wy = np.exp(-(ty[:, None] - gy * hg) ** 2 / (4 * tau))
        idx = ((gx % Mr)[:, :, None] * Mr + (gy % Mr)[:, None, :]).ravel()
        w = (wx[:, :, None] * wy[:, None, :] * c[s:s + chunk, None, None]).ravel()
        gre += np.bincount(idx, weights=w.real, minlength=Mr * Mr)
        gim += np.bincount(idx, weights=w.imag, minlength=Mr * Mr)
    G = np.fft.fft2((gre + 1j * gim).reshape(Mr, Mr))
    k = np.arange(-N // 2, N // 2)
    Fk = G[np.ix_(k % Mr, k % Mr)]
    dec = np.exp(tau * k.astype(float) ** 2)
    return Fk * (np.pi / tau) / Mr ** 2 * dec[:, None] * dec[None, :]


TWO_PI_ = 2 * np.pi
