# Changelog

## 0.2.0

Verified against official BBOB, and restructured to match how COCO defines a
problem. **This release changes both the API and the numbers**: several
functions were wrong, so results from 0.1.0 are not comparable.

### Fixed — alignment with official BBOB

Every function and noise model is now checked numerically against the official
2009 implementation on identical instances (`tests/test_alignment.py`), which
found four real defects:

- **`bueche_rastrigin`** skewed the wrong half of the coordinates. The paper's
  1-based "i odd" is 0-based *even*, as both official implementations have it.
- **`step_ellipsoidal`** carried an extra factor of 100 on the ellipsoid sum,
  so values ran ~98x high and the plateau branch almost never selected.
- **`gallagher_101_me` / `gallagher_21_hi`** derived their peak layout from
  `fold_in(key, q[0, 0])`, which int-casts the float; every rotated instance
  shared one frozen layout. Layouts now come from the instance's own key.
- **`katsuura`** truncated its inner sum at 30 terms instead of 32.

And three in the noise models, against the noisy-functions paper:

- **Cauchy noise** divided by `|Uniform(0, 1)|` rather than `|N(0, 1)|`, giving
  ~30% heavier tails and the wrong central law.
- **Uniform noise** used an epsilon of 1e-8 where the paper specifies 1e-99,
  distorting the ratio by up to 2x in the 1e-8 target-precision band, and
  dropped the paper's `(0.49 + 1/D)` dimension term from alpha.

### Changed — one task is one function at one dimension

- `BBOB("sphere", num_dims=10)` replaces a list or dict of functions plus a
  `min_num_dims`/`max_num_dims` pair. `bbobax.suite()` builds the standard 24
  as separate tasks; loop over it to cover the suite.
- `QDBBOB(fitness_fn, descriptor_fn, ...)` takes one descriptor function.
- `lax.switch` is gone. Under `vmap` with a varying function it had to evaluate
  every branch for every solution.
- The masking layer is gone with the dynamic dimension. A sampled dimension
  range never resized the search space -- solutions stayed `max_num_dims` long
  and the extra coordinates were inert -- so it was never COCO's D-dimensional
  problem. `num_dims < 2` now raises.
- `params.x_opt` is always the function's true argmin: the per-function
  conventions (sign vectors, scalings, sign-forcing) are applied at sampling
  time, as COCO stores the post-convention optimum.
- Rotations moved from `BBOBState` to `BBOBParams` -- they are instance data,
  drawn once and never mutated -- so `init(params)` takes no key and cannot
  silently redraw the instance. `BBOBState` carries the evaluation counter.
- `BBOBParams` is now `(x_opt, f_opt, r, q, key, noise_params)`. `fn_id`,
  `num_dims` and `descriptor_id` are gone.
- **`sample_rotation` defaults to `True`.** BBOB always rotates; with identity
  rotations every rotated variant collapses onto its base function.
- **The default `BBOB()` is the plain noiseless suite.** The old default
  enabled all five noise models with stabilization, so ~80% of instances were
  noisy and even the noiseless path was biased by +1.01e-8.
- `num_dims` sampling had an exclusive upper bound, so the top of the range was
  never drawn.
- `Gallagher` is ~2.4x faster: its diagonal conditioning is carried as a vector
  and the quadratic form is evaluated as one matmul over all peaks.
- Registries are `SCREAMING_CASE`, as module constants should be: `bbob_fns` ->
  `BBOB_FNS`, plus `X_OPT_CONVENTIONS` and `NOISE_MODELS`. The noise functions
  lose their redundant `_noise` suffix, so a registry key matches its function
  name exactly (`"gaussian"` -> `gaussian`).
- `NoiseParams` moved to `types.py` with every other dataclass, which removes a
  circular import and the duplicate `IntScalar` alias it forced.
- Tooling: `ty` replaces `mypy`; both it and ruff target 3.11, the oldest
  supported version, while the dev environment runs 3.14. CI now runs ruff,
  ty, the tests on 3.11-3.14, and executes every notebook.

## 0.1.0

Initial release.
