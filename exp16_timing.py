"""Exp 16 timing with the final code (memoised pixel bases, halved polygon moments): fit and
whole-grid evaluation for the Shepp-Logan variant at n = 256; spot check against the exact transform."""
import sys, json, time
sys.path.insert(0, '.')
import numpy as np
from exp16_mri import sl_phantom
from safi2d_multi import MultiCurveSAFI2D, fast_cartesian

img = sl_phantom()
n = 256
A = img.pixel_averages(n)
t0 = time.time(); S = MultiCurveSAFI2D(A, P=4); t1 = time.time(); S.fit_all(); t2 = time.time()
out = dict(n=n, detect_init_s=t1 - t0, fit_s=t2 - t1, eval_s={}, check={})
for N in (256, 512, 1024):
    t = time.time(); G = fast_cartesian(S, N); out['eval_s'][N] = time.time() - t
    k = np.arange(-N // 2, N // 2)
    rr = np.random.default_rng(7)
    iu, iv = rr.integers(0, N, 200), rr.integers(0, N, 200)
    F = img.ft(k[iu].astype(float), k[iv].astype(float))
    out['check'][N] = float(np.linalg.norm(G[iu, iv] - F) / np.linalg.norm(F))
    print(N, 'eval %.1fs' % out['eval_s'][N], 'spot error %.1e' % out['check'][N], flush=True)
print('init %.1fs fit %.1fs' % (out['detect_init_s'], out['fit_s']))
json.dump(out, open('results/exp16_timing.json', 'w'), indent=1)
