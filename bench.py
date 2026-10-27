"""Test functions with exact Fourier integrals at every frequency, and error metrics."""
import numpy as np

TWO_PI = 2.0 * np.pi


class PiecewiseExp:
    """f(x) = Re sum_r alpha_{k,r} exp(beta_{k,r} x) on piece k = [b_k, b_{k+1}); 0 outside.

    Real-valued, smooth (entire) on each piece, jumps at the interior breaks and at the
    support endpoints. F(u) is exact in closed form for every u.
    """

    def __init__(self, breaks, alphas, betas, real=True):
        self.breaks = np.asarray(breaks, float)
        self.alphas = [np.asarray(a, complex) for a in alphas]
        self.betas = [np.asarray(b, complex) for b in betas]
        self.real = real

    @classmethod
    def random(cls, rng, interior, p0=0.0, p1=1.0, nterm=3, osc=3.0, decay=2.0):
        br = np.concatenate([[p0], np.sort(np.atleast_1d(interior)), [p1]])
        A, Bt = [], []
        for _ in range(br.size - 1):
            A.append(rng.normal(size=nterm) + 1j * rng.normal(size=nterm))
            Bt.append(rng.uniform(-decay, decay, nterm) + 1j * TWO_PI * rng.uniform(-osc, osc, nterm))
        return cls(br, A, Bt)

    def __call__(self, x):
        x = np.atleast_1d(np.asarray(x, float))
        out = np.zeros(x.shape, float if self.real else complex)
        k = np.clip(np.searchsorted(self.breaks, x, side='right') - 1, 0, self.breaks.size - 2)
        inside = (x >= self.breaks[0]) & (x <= self.breaks[-1])
        for j in range(self.breaks.size - 1):
            s = inside & (k == j)
            if s.any():
                v = np.exp(np.outer(x[s], self.betas[j])) @ self.alphas[j]
                out[s] = np.real(v) if self.real else v
        return out

    def ft(self, u):
        u = np.atleast_1d(np.asarray(u, float))
        F = np.zeros(u.shape, complex)
        for j in range(self.breaks.size - 1):
            a, b = self.breaks[j], self.breaks[j + 1]
            for al, be in zip(self.alphas[j], self.betas[j]):
                # Re(al e^{be x}) = (al e^{be x} + conj(al) e^{conj(be) x}) / 2
                terms = ((al / 2, be), (np.conj(al) / 2, np.conj(be))) if self.real else ((al, be),)
                for c, g in terms:
                    z = g - 1j * TWO_PI * u
                    small = np.abs(z * (b - a)) < 1e-8
                    val = np.where(small, np.exp(z * a) * (b - a),
                                   (np.exp(z * b) - np.exp(z * a)) / np.where(small, 1, z))
                    F += c * val
        return F

    def integral(self, a, b):
        """Exact int_a^b f (vectorised over arrays a <= b)."""
        a = np.atleast_1d(np.asarray(a, float)); b = np.atleast_1d(np.asarray(b, float))
        out = np.zeros(a.shape, float if self.real else complex)
        for j in range(self.breaks.size - 1):
            lo = np.clip(a, self.breaks[j], self.breaks[j + 1])
            hi = np.clip(b, self.breaks[j], self.breaks[j + 1])
            for al, be in zip(self.alphas[j], self.betas[j]):
                if abs(be) < 1e-14:
                    v = al * (hi - lo)
                else:
                    v = al * (np.exp(be * hi) - np.exp(be * lo)) / be
                out = out + (np.real(v) if self.real else v)
        return out

    def cell_avg(self, xe):
        xe = np.asarray(xe, float)
        return self.integral(xe[:-1], xe[1:]) / np.diff(xe)

    def jumps(self):
        """Jump sizes f(c+) - f(c-) at interior breaks."""
        out = []
        for j in range(1, self.breaks.size - 1):
            c = self.breaks[j]
            fl = np.exp(c * self.betas[j - 1]) @ self.alphas[j - 1]
            fr = np.exp(c * self.betas[j]) @ self.alphas[j]
            out.append(np.real(fr - fl) if self.real else fr - fl)
        return np.array(out)


def band_grid(umax, umin=1.0, per_band=48, rng=None):
    """Dyadic bands [2^k, 2^{k+1}] with per_band random (irrational-offset) frequencies each."""
    rng = rng or np.random.default_rng(0)
    edges = 2.0 ** np.arange(np.floor(np.log2(umin)), np.ceil(np.log2(umax)) + 1)
    u, lab = [], []
    for k in range(edges.size - 1):
        uu = rng.uniform(edges[k], edges[k + 1], per_band)
        u.append(uu); lab.append(np.full(per_band, k))
    return np.concatenate(u), np.concatenate(lab), edges


def band_rel(F, G, lab):
    """Band-wise relative L2 error ||F-G||/||F|| per dyadic band."""
    ks = np.unique(lab)
    return np.array([np.linalg.norm(F[lab == k] - G[lab == k]) / np.linalg.norm(F[lab == k]) for k in ks])


def horizon(rel, edges, eps):
    """Upper edge of the last band before the first band whose relative error exceeds eps."""
    bad = np.where(rel > eps)[0]
    if bad.size == 0:
        return np.inf
    k = bad[0]
    return edges[k] if k > 0 else edges[0]
