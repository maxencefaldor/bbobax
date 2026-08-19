# Changelog

## 0.2.0

Verified against official BBOB, and rebuilt around how COCO defines a problem.
**This release changes both the API and the numbers**: several functions were
wrong, so no result from 0.1.0 is comparable.

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
  distorting the ratio by up to 2x in the 1e-8 target-precision band.
- **Uniform noise** dropped the paper's `(0.49 + 1/D)` dimension term from
  alpha, so its severity did not vary with the dimension.

### Fixed — noise stabilization was optional, and off

In official BBOB the `+1.01 * 1e-8` offset and below-tolerance passthrough are
*inside* `fGauss`, `fUniform` and `fCauchy`, applied unconditionally. bbobax
had them as `use_stabilization`, a flag on the noise container defaulting to
**False**, so every noisy model was missing them unless you knew to ask. They
are now part of the three official models and cannot be switched off, while
`Noiseless` (as in official) and `Additive` (a bbobax extension) have none.

`tests/test_noise.py` checks each model *by formula* against the official one
transcribed into numpy, on the draws bbobax actually makes — the same approach
`tests/test_alignment.py` takes for the functions.

### Fixed — the noise epsilons were zero in float32

The uniform and Cauchy models divide by a quantity that can be zero, and the
paper guards each with a literal (`1e-99`, `1e-199`). Both round to exactly
`0.0` in float32 — JAX's default — so neither guarded anything unless
`jax_enable_x64` happened to be on. The guard is now the smallest positive
normal of the working dtype, which holds in either precision and is
indistinguishable from the paper's constant in float64.

### Fixed — the dimension was never really the problem's

The masking layer is gone along with the sampled dimension range. A sampled
range never resized the search space: solutions stayed `max_num_dims` long and
the extra coordinates were inert, so it was never COCO's D-dimensional problem.
A problem now fixes one dimension, as COCO enumerates them, and `num_dims < 2`
raises. `bbobax.DIMENSIONS` records COCO's own set, `(2, 3, 5, 10, 20, 40)`.

### Fixed — the notebooks' stored outputs could go stale

`scripts/run_notebooks.py` executed a copy and never wrote back, and the docs
render *stored* outputs — so a notebook whose code was updated could show a
reader an API that no longer exists. The script takes `--save` now, and pins a
portable kernelspec so a contributor's local environment name cannot ship.

### Changed — a problem is an object

- `BBOB(fn_id=..., min_num_dims=..., max_num_dims=...)` becomes one class per
  function: `Sphere(num_dims=10)`. `BBOB_PROBLEMS` maps each name to its class,
  and `suite()` builds the standard 24.
- **Instance generation is per function.** A function whose definition
  constrains its optimum overrides `_sample_x_opt` and draws what it actually
  needs. `params.x_opt` is always the true argmin: `linear_slope` is linear so
  its minimum is a corner of the box, `schwefel` and `lunacek` are built around
  fixed constants, `rosenbrock` and `gallagher_21_hi` are scaled, and
  `bueche_rastrigin` forces its skewed coordinates non-negative.
- **Quality-Diversity is composed, not inherited.** `QDProblem(problem,
  descriptor)` pairs any of the 24 with any `Descriptor`, so a descriptor is no
  longer bound to a function. `RandomProjection` replaces
  `get_random_projection_descriptor()`.
- **`BBOBState` is gone**, and `evaluate` no longer takes or returns it:
  `evaluate(key, x, params) -> BBOBEval`. All 24 functions are memoryless and
  all 24 ignored it. `init()` is gone with it.
- **`lax.switch` is gone.** Under `vmap` with a varying function it had to
  evaluate every branch for every solution.
- The PRNG key comes first everywhere — in every signature, as `jax.random`
  has it, and in `BBOBParams`, since everything else is drawn from it.
- Rotations are instance parameters rather than state: drawn once by `sample`
  and never mutated.

### Changed — a noise model is an object too

- The `NoiseModel(noise_model_names=..., noise_ranges=...,
  use_stabilization=...)` **container is gone**, replaced by one model per
  class: `Noiseless`, `Gaussian`, `Uniform`, `Cauchy`, `Additive`. A problem
  holds one: `Sphere(noise_model=Gaussian())`.
  **Note the name is reused, not kept**: `NoiseModel` was that container; it is
  now the protocol each individual model satisfies.
- **The `lax.switch` over noise models is gone.** Under `vmap` with a varying
  `noise_id` it evaluated every model for every solution. `NoiseParams` no
  longer carries all five models' settings at once, either; each model draws
  only what it uses.
- Severity ranges move onto the model that uses them
  (`Gaussian(beta_range=...)`), replacing the `DEFAULT_RANGES` dictionary.
- `Mixture(Gaussian(), Uniform(), Cauchy())` restores drawing a noise *family*
  per instance, for meta-learning batches that need instances to disagree about
  which noise they carry. It is the only thing here that dispatches, and says
  so: under `vmap` it evaluates every model it holds, for every solution.

### Changed — one word for a noise model

`NoiseModel` is the protocol (the paper's term, parallel to `BBOBProblem`),
`NOISE_MODELS` the registry (parallel to `BBOB_PROBLEMS`), and `noise_model`
names both the problem's attribute and the `BBOBParams` field holding that
model's settings.

That pairing is the rule the package follows throughout: **a field of a params
container is named for the component whose parameters it holds**, so
`problem.noise_model` lines up with `params.noise_model` exactly as
`QDProblem.descriptor` lines up with `QDParams.descriptor`.

`TARGET_VALUE` becomes `TARGET_PRECISION`, which is what the paper calls it.

### Changed — every module owns its own types

`types.py` is gone. `BBOBParams`/`BBOBEval` live with `BBOBProblem`,
`QDParams`/`QDEval` with `QDProblem`, and each noise model's parameters with
that model. Public imports (`from bbobax import BBOBParams`) are unchanged.
Modules follow the same rule: `bbob`/`fitness_fns`/`descriptor_fns` become
`problem`/`functions`/`qd`.

Registries are `SCREAMING_CASE`, and the noise functions lose their redundant
`_noise` suffix so a registry key matches its class name.
`_lambda_alpha_vector` becomes `lambda_alpha`, public alongside the three other
transforms it sits with and named for the paper's symbol as they are.

### Changed — defaults

- **`sample_rotation` defaults to `True`.** BBOB always rotates; with identity
  rotations every rotated variant collapses onto its base function.
- **The default problem is the plain noiseless suite.** The old default enabled
  all five noise models with stabilization, so ~80% of instances were noisy and
  even the noiseless path was biased by +1.01e-8.

### Added — the noisy suite, f101-f130

`noisy_suite()` builds the thirty official noisy problems. It is not "the 24
plus noise", and the two things that make it so were read out of the vendored
reference rather than assumed:

- **Boundary handling is uniform.** Every noisy problem uses
  `defaultboundaryhandling(x, 100)`, replacing whatever factor its noiseless
  counterpart used -- including the several that have no penalty at all.
- **Two of the bases are reparameterized.** f116-f118 use an ellipsoid of
  conditioning 1e4 where f10 has 1e6, and f125-f127 scale Griewank-Rosenbrock
  by 1 where f19 uses 10. (`GriewankRosenbrock.facftrue` is now a class
  attribute, since the reference names it.)

Each problem is pinned to one of the paper's two severities, so a number here
is comparable to a published f1xx number.

### Removed — the `Additive` noise model

It applied noise at an *absolute* scale (`f + std * N(0, 1)`) to functions
whose values span eight orders of magnitude, so `std = 0.1` was 1% of
`katsuura`'s whole range and 1e-9 of `bent_cigar`'s -- the parameter meant
something different on every one of the 24, which is not a benchmark axis. It
was also unstabilized, putting a hard floor under BBOB's 1e-8 target. All three
official models are relative to the value instead; `Gaussian` with a small beta
is the scale-aware, stabilized version of what it was reaching for.

### Added — Many-Affine BBOB

`ManyAffine` combines all 24 under a sparse weight vector in log space, following
IOHexperimenter's `ManyAffine`. It turns the suite from 24 atoms into a
continuous space of problems, which is what meta-learning wants from it.

It costs what evaluating 24 functions costs and no more — measured at 0.92x the
time of the 24 run separately at D=10 and 0.78x at D=40, because XLA fuses them
and hoists the shared instance derivation.

### Added — the paper's two severities are reachable

Sampling severity continuously across the moderate-to-severe span is bbobax's
deviation, and a useful one — difficulty then varies per instance for free. But
it means nothing here is comparable to a published f101-f130 result.
`Gaussian.moderate()`, `Gaussian.severe()` and the same on `Uniform` and
`Cauchy` pin the paper's exact points.

### Added — coverage at the dimensions the suite actually defines

No test reached above D = 10, so half of `DIMENSIONS` was never exercised —
and the dimension is not incidental: it enters `katsuura` as `10 / D**1.2`,
`lunacek` as `sqrt(D + 20) - 4.1`, and `weierstrass` and `schwefel` divide by
it. Every function is now swept across all six standard dimensions for
finiteness and for `f(x_opt) == 0`. The same sweep runs in a fresh interpreter
at float32, JAX's default, which the float64 suite otherwise never covers.
Every noise model is swept too, including at `f = 0` exactly.

### Performance

`Gallagher` is ~2.4x faster: its diagonal conditioning is carried as a vector
and the quadratic form is evaluated as one matmul over all peaks.

### Tooling

`ty` replaces `mypy`; both it and ruff target 3.11, the oldest supported
version, while the dev environment runs 3.14. CI runs ruff, ty, the tests on
3.11-3.14, and executes every notebook.

## 0.1.0

Initial release.
