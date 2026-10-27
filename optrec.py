"""One-sided optimal recovery constants and extremal functions (localisation lower bounds).

E_m(theta) = sup { g(theta) : g(-j) = 0 for j = 0..N-1,  |g^{(m)}| <= 1 }      (unit spacing)

is computed as a linear programme: g^{(m)} = v, piecewise constant on a fine grid with |v| <= 1,
g = polynomial of degree m-1 + m-fold integral of v.  The value is the largest error any method can
be forced to make when extrapolating a C^m function a fraction theta of a cell beyond its last
sample; the extremal g vanishes at every sample.  Compare with the error constant of one-sided
polynomial extrapolation from the m nearest samples, e_m(theta) = prod_{j<m} (theta + j) / m!.
"""
import numpy as np
from math import factorial
from scipy.optimize import linprog


class Extremal:
    """The LP extremal g on [a, b] (unit spacing, samples at 0, -1, ..., -(N-1))."""

    def __init__(self, m, theta, N=None, per=32, b=1.0):
        N = N or 3 * m + 8
        self.m, self.theta = m, theta
        self.a = a = -float(N)
        self.L = b - a
        self.t = t = np.linspace(a, b, int((b - a) * per) + 1)
        nodes = -np.arange(N, dtype=float)
        Aeq = np.array([self._row(x) for x in nodes])
        r = linprog(-self._row(theta), A_eq=Aeq, b_eq=np.zeros(N),
                    bounds=[(None, None)] * m + [(-1, 1)] * (t.size - 1), method='highs')
        assert r.status == 0, r.message
        self.coef, self.v = r.x[:m], r.x[m:]
        # remove the LP's residuals at the samples with a cardinal-sine correction, then rescale
        # so that |g^{(m)}| <= 1 still holds (the correction's m-th derivative is <= sum|r| pi^m)
        self.nodes = nodes
        self.res = np.array([self._raw(x) for x in nodes])
        self.scale = 1.0 / (1.0 + np.sum(np.abs(self.res)) * np.pi ** m)
        self.value = float(self(theta)[0])

    def _row(self, x, d=0):
        m, a, t = self.m, self.a, self.t
        L = self.L
        poly = [0.0 if k < d else factorial(k) / factorial(k - d) * ((x - a) / L) ** (k - d) / L ** d for k in range(m)]
        lo, hi = np.minimum(t[:-1], x), np.minimum(t[1:], x)
        w = ((x - lo) ** (m - d) - (x - hi) ** (m - d)) / factorial(m - d)
        return np.r_[poly, w]

    def _raw(self, x, d=0):
        return float(self._row(x, d) @ np.r_[self.coef, self.v])

    def __call__(self, x, d=0):
        x = np.atleast_1d(np.asarray(x, float))
        out = np.array([self._raw(xx, d) for xx in x])
        for xj, rj in zip(self.nodes, self.res):
            z = x - xj
            if d == 0:
                s = np.sinc(z)
            else:
                with np.errstate(invalid='ignore', divide='ignore'):
                    s = np.where(np.abs(z) < 1e-8, 0.0,
                                 (np.cos(np.pi * z) * np.pi * z - np.sin(np.pi * z)) / (np.pi * z ** 2))
            out -= rj * s
        return self.scale * out


def E(m, theta, **kw):
    return Extremal(m, theta, **kw).value


def e_poly(m, theta):
    return float(np.prod([theta + j for j in range(m)]) / factorial(m))


if __name__ == '__main__':
    for m in (7, 8):
        for th in (0.25, 0.5, 0.75, 1.0):
            g = Extremal(m, th)
            print(f'm={m} theta={th}: E={g.value:.4e}  e_poly={e_poly(m, th):.4e}  ratio={e_poly(m, th) / g.value:.3f}'
                  f'  LP residual {np.max(np.abs(g.res)):.1e} -> {np.max(np.abs(g(g.nodes))):.1e}')
