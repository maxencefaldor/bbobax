# Changelog

## 0.3.0

Each of the 24 functions is now a class satisfying one protocol. **This release
changes the API**; the numbers are unchanged except where noted, and the
alignment suite still passes at 1e-9 relative against official BBOB.

### Changed — a problem is an object

- `BBOB("sphere", num_dims=10)` becomes `Sphere(num_dims=10)`. Every function
  is a `BBOBProblem` subclass supplying `_value`; `BBOB_FNS` becomes
  `BBOB_PROBLEMS`, mapping the same names to classes. `suite()` is unchanged.
- **Instance generation is per function.** A function whose definition
  constrains its optimum overrides `_sample_x_opt` and draws what it actually
  needs, rather than a uniform draw being reshaped afterwards.
- **`QDBBOB` becomes `QDProblem`, composed rather than inherited**: it pairs
  any problem with any `Descriptor`, so a descriptor is no longer bound to a
  function. `get_random_projection_descriptor()` becomes the `RandomProjection`
  descriptor. `QDBBOBParams` becomes `QDParams(problem, descriptor)`.
- **`BBOBState` is gone, and `evaluate` no longer takes or returns it**:
  `evaluate(key, x, params) -> BBOBEval`. All 24 functions are memoryless and
  all 24 ignored it. `init()` is gone with it.
- **`DIMENSIONS`** records COCO's own dimension set, `(2, 3, 5, 10, 20, 40)`.
- Modules follow: `bbob`/`fitness_fns`/`descriptor_fns` become
  `problem`/`functions`/`qd`.

### Fixed — noise stabilization was optional, and off

In official BBOB the `+1.01 * 1e-8` offset and below-tolerance passthrough are
*inside* `fGauss`, `fUniform` and `fCauchy`, applied unconditionally. bbobax had
them as `use_stabilization`, a flag on the noise container defaulting to
**False**, so every noisy model was missing them unless you knew to ask. They
are now part of the three official models and cannot be switched off, while
`Noiseless` (as in official) and `Additive` (a bbobax extension) have none.

`tests/test_noise.py` now checks each model *by formula* against the official
one transcribed into numpy, on the draws bbobax actually makes — the same
approach `tests/test_alignment.py` takes for the functions.

### Fixed — the noise epsilons were zero in float32

The uniform and Cauchy models divide by a quantity that can be zero, and the
paper guards each with a literal (`1e-99`, `1e-199`). Both round to exactly
`0.0` in float32 — JAX's default — so neither guarded anything unless
`jax_enable_x64` happened to be on. The guard is now the smallest positive
normal of the working dtype, which holds in either precision and is
indistinguishable from the paper's constant in float64.

### Changed — a noise model is an object too

- The `NoiseModel(noise_model_names=..., noise_ranges=...,
  use_stabilization=...)` **container is gone**, replaced by one model per
  class: `Noiseless`, `Gaussian`, `Uniform`, `Cauchy`, `Additive`. A problem
  holds one: `Sphere(noise_model=Gaussian())`.
  **Note the name is reused, not kept**: `NoiseModel` in 0.2.0 was that
  container; in 0.3.0 it is the protocol each individual model satisfies. Code
  that constructed `NoiseModel(...)` must construct a model instead.
- **The `lax.switch` over noise models is gone.** Under `vmap` with a varying
  `noise_id` it evaluated every model for every solution — the same cost the
  function redesign removed. `NoiseParams` no longer carries all five models'
  settings at once, either; each model draws only what it uses.
- Severity ranges move onto the model that uses them
  (`Gaussian(beta_range=...)`), replacing the `DEFAULT_RANGES` dictionary.
- `Mixture(Gaussian(), Uniform(), Cauchy())` restores drawing a noise *family*
  per instance, for meta-learning batches that need instances to disagree about
  which noise they carry. It is the only thing here that dispatches, and says
  so: under `vmap` it evaluates every model it holds, for every solution.

### Added — coverage at the dimensions the suite actually defines

No test reached above D = 10, so half of `DIMENSIONS` was never exercised —
and the dimension is not incidental: it enters `katsuura` as `10 / D**1.2`,
`lunacek` as `sqrt(D + 20) - 4.1`, and `weierstrass` and `schwefel` divide by
it. Every function is now swept across all six standard dimensions for
finiteness (inside the box, outside it, at the corners, around and at the
optimum) and for `f(x_opt) == 0`. The same sweep runs in a fresh interpreter at
float32, JAX's default, which the float64 suite otherwise never covers. Every
noise model is checked for finiteness too, including at `f = 0` exactly — the
divide-by-zero case the epsilon above exists for.

### Changed — one word for a noise model

The library used `Noise`, `noise`, `noise_model` and `noise_params` for four
views of the same thing. There is now one word. `NoiseModel` is the protocol
(the paper's term, and parallel to `BBOBProblem`), `NOISE_MODELS` the registry
(parallel to `BBOB_PROBLEMS`), `noise_model` both the problem's attribute and
the field of `BBOBParams` holding that model's parameters.

That last pairing is the rule the whole package now follows: **a field of a
params container is named for the component whose parameters it holds**, so
`problem.noise_model` and `params.noise_model` line up exactly as
`QDProblem.descriptor` lines up with `QDParams.descriptor`, and `evaluate`
reads `self.X.apply(..., params.X)` on both sides.

`TARGET_VALUE` becomes `TARGET_PRECISION`, which is what the paper calls it.

### Added — the paper's two severities are reachable

Sampling severity continuously across the moderate-to-severe span is bbobax's
deviation, and a useful one — difficulty then varies per instance for free,
which is what meta-learning wants. But it means nothing here is comparable to a
published f101-f130 result. `Gaussian.moderate()`, `Gaussian.severe()` and the
same on `Uniform` and `Cauchy` pin the paper's exact points, so the deviation
is a default rather than a constraint.

### Fixed — the notebooks' stored outputs could go stale

`scripts/run_notebooks.py` executed a copy and never wrote back, so a notebook
whose code was updated kept the outputs of an older run — and the docs render
*stored* outputs, so `00_getting_started` was showing readers a
`noise_params=NoiseParams(noise_id=...)` repr for an API that no longer exists.
The script takes `--save` now, and the outputs are regenerated.

### Changed — every module owns its own types

`types.py` is gone. `BBOBParams`/`BBOBEval` live with `BBOBProblem`,
`QDParams`/`QDEval` with `QDProblem`, and each noise model's parameters with
that model — the rule the noise and descriptor redesigns established, now
applied to the whole package. Public imports (`from bbobax import BBOBParams`)
are unchanged.

`_lambda_alpha_vector` becomes `lambda_alpha`, public alongside the three other
transforms it sits with (`transform_osz`, `transform_asy`, `f_pen`) and named
for the paper's symbol like they are.

### Changed — numbers

- `linear_slope`, `schwefel` and `lunacek` draw the sign of each optimum
  coordinate directly instead of taking the sign of a uniform draw. The
  distribution of instances is identical; the specific instance drawn from a
  given key is not.
- Any use of the noise models now includes the stabilization they were missing.
  The noiseless suite — the default, and what `qd` uses — is unaffected.

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
- `params.x_opt` is always the function's true argmin. Six of the 24 constrain
  where their optimum may sit -- `linear_slope` is linear, so its minimum is
  always a corner of the box; `schwefel` and `lunacek` are built around fixed
  constants; `rosenbrock` and `gallagher_21_hi` are scaled; `bueche_rastrigin`
  forces its skewed coordinates non-negative. `BBOB_FNS` now maps a name to a
  `BBOBFunction` bundling the function with that constraint, so per-function
  knowledge lives in one entry and the default ("anywhere in the box") is
  written out rather than implied by absence from a second table.
- The PRNG key comes first everywhere -- in every signature, as `jax.random`
  has it, and now in `BBOBParams` too, since everything else is drawn from it.
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
