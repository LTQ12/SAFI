# SAFI: Fourier transforms of piecewise smooth objects from samples, cell averages and pixel averages

Code for the paper *Inverting the folded singular spectrum: Fourier transforms of piecewise smooth objects from cell and pixel averages* (T. Liu, L. Liu, C. Zhu).

The discrete Fourier transform of pixel or cell averages of an object with sharp interfaces has an error that does not decrease with resolution: sampling folds the spectral tail of the interfaces back into the band. SAFI locates the interfaces from the data by conservation (the model reproduces every cut average exactly), extrapolates the smooth parts with polynomials of high degree, and evaluates the transform of the resulting piecewise polynomial model exactly at every frequency.

## Requirements

Python 3.10 or later with NumPy, SciPy and Matplotlib (`pip install -r requirements.txt`). Everything runs on one CPU core; the nonuniform FFT is a pure NumPy implementation (`nufft.py`).

## Library

| File | Contents |
|---|---|
| `safi.py` | one dimension, point samples and queries: detection, localisation, exact transform, a posteriori estimate, adaptive driver `safi_auto` |
| `safi_cell.py` | one dimension, cell averages (subcell resolution with extrapolants of degree M) |
| `safi_adapt.py` | local refinement and consistency-checked bisection for clustered singularities |
| `safi2d.py` | two dimensions, pixel averages: detection, local polynomial fits, polygonal interfaces |
| `safi2d_curve.py` | conservative fit of one smooth star-shaped interface (Fourier curve) |
| `safi2d_multi.py` | several nested or disjoint interfaces; `fast_cartesian` evaluates k-space on an N x N grid by NUFFTs |
| `nufft.py` | type-1 two-dimensional NUFFT (Gaussian gridding) |
| `bench.py`, `bench2d*.py`, `curve2d.py`, `burgers.py` | test functions, images and phantoms with exact transforms; WENO5 Burgers solver |
| `optrec.py` | information-theoretic localisation bounds (fooling pairs) |

Minimal use:

```python
import numpy as np
from safi_cell import SAFICell
from bench import PiecewiseExp

f = PiecewiseExp.random(np.random.default_rng(0), [0.37, 0.71])
xe = np.linspace(0, 1, 257)
S = SAFICell(xe, f.cell_avg(xe), M=6)
u = np.array([10.0, 1e3, 1e5])
print(np.abs(S(u) - f.ft(u)) / np.abs(f.ft(u)))
```

## Experiments

Each script writes `results/<name>.json` (and usually `figs/<name>.png`) and is run from the repository root, for example `python exp16_mri.py sl`.

| Script | Experiment |
|---|---|
| `exp15_mechanism.py` | the folded singular tail in one and two dimensions; methods that do not remove it |
| `exp16_mri.py`, `exp16_timing.py` | Shepp–Logan variant and random phantoms with coil sensitivity: k-space accuracy and the inverse crime |
| `exp17_fri.py` | comparison with finite-rate-of-innovation localisation |
| `exp18_ablation.py` | ablation: low-degree extrapolants, interfaces without the global conservative fit, model evaluated on a grid |
| `exp01_horizon.py`, `exp02_cell.py`, `exp07_identifiability.py`, `exp12_tightness.py` | frequency horizons for point and cell data, identifiability, localisation bounds |
| `exp03_complexity.py`, `exp09_local.py`, `exp14_certify.py` | sample complexity over a band, clustered singularities, cost of certification |
| `exp04_estimator.py`, `exp05_noise.py`, `exp08_timing.py` | a posteriori estimate, noise, timing |
| `exp06_apps.py`, `exp10_shock.py` | applications; spectra of Burgers shocks from finite-volume data |
| `exp11_2d.py`, `exp13_curved.py`, `fig13_curved.py` | pixel images with polygonal and curved inclusions |
| `paper_figures/` | Figures 1 and 12 of the paper (run after `exp15` and `exp16`, from inside the folder) |
