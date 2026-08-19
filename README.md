# BBOBax

<div align="center">
  <p>Accelerated Black-Box Optimization Benchmark in JAX.</p>
  <a href="https://pypi.python.org/pypi/bbobax"><img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/bbobax.svg?style=flat"></img></a>
  <a href="https://pypi.python.org/pypi/bbobax"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/bbobax.svg?style=flat"></img></a>
  <a href="https://github.com/google/jax"><img alt="JAX" src="https://img.shields.io/badge/JAX-Accelerated-9cf"></img></a>
</div>
</br>

A high-performance reimplementation of the [COCO](https://coco-platform.org/) (COmparing Continuous Optimizers) test suite in JAX. BBOBax allows for massive parallelization of function evaluations on hardware accelerators (GPUs/TPUs), enabling efficient benchmarking of black-box optimization algorithms.

## Features

*   **JAX-based**: Fully differentiable (where applicable) and JIT-compilable.
*   **Hardware Acceleration**: Run benchmarks on GPUs and TPUs for massive speedups.
*   **Standard BBOB**: Includes standard single-objective BBOB functions (noiseless).
*   **Noise Support**: Configurable noise models (Gaussian, Uniform, Cauchy, etc.) for robust optimization benchmarking.
*   **Quality-Diversity (QD)**: Any of the 24 functions composes with any descriptor.
*   **Flexible API**: Easy integration with existing JAX-based evolutionary computation libraries (e.g., EvoJAX, evosax).

## Fidelity to official BBOB

Every function and noise model is verified **numerically** against the official
2009 BBOB implementation (`bbobbenchmarks.py`, vendored as test ground truth):
identical instances are injected and outputs must agree to ≤1e-9 relative in
float64 — the suite in `tests/test_alignment.py` re-proves this on every run.
Where the paper (Hansen, Finck, Ros & Auger, INRIA RR-6829/RR-6869) and the
official code disagree, the code wins, with a comment at the site.

Guarantees you can rely on:

*   `params.x_opt` is always the function's **true argmin**, so
    `f(params.x_opt) = f_opt` for all 24. Six functions constrain where their
    optimum may sit — a linear function is minimized on a corner, not inside
    the box — and each declares that constraint itself by overriding
    `_place_x_opt`, applied when the instance is drawn, exactly as COCO stores
    the post-constraint optimum.
*   Noise applies to the raw function value only; the boundary penalty and
    `f_opt` are added outside it, as the paper prescribes.
*   The default problem is the plain **noiseless** suite with rotations on —
    exactly COCO's noiseless BBOB. Noise is opt-in via `noise_config`.

Deliberate, documented deviations from COCO (design choices, not accidents):

*   **Instance distribution**: x_opt is drawn continuously in `x_opt_range`
    (COCO uses a 0.0008 grid) and `f_opt` defaults to 0 (COCO draws a 2-decimal
    Cauchy clipped to ±1000); there is no seed table, so official numbered
    instances are not reproducible — instances are sampled, not enumerated.
*   **f9/f19**: the optimum is placeable via `x_opt` (official derives it from
    the rotation, always near the origin). Every bbobax instance is an exact
    translation of an official landscape — a strict superset of the official
    family, verified bit-for-bit at the derived point.
*   **Rotations** are Haar on SO(n) (COCO: O(n)); orientation is the only
    difference, and no benchmark property distinguishes the cosets.
*   **Nothing about a problem is sampled except the instance.** A problem is
    one function at one dimension, as COCO enumerates them; `sample` draws an
    instance of it. To cover many functions or dimensions, hold many problems
    and loop — `bbobax.suite()` builds the standard 24 and `bbobax.DIMENSIONS`
    is COCO's own dimension set. Under `jit` that loop unrolls, so every
    problem keeps its own compiled code and none pays for dispatch.
*   **There is no evaluation state.** All 24 functions are memoryless: the
    value at `x` does not depend on when `x` was asked. A dynamic benchmark
    would be a different contract, not a parameter these 24 carry and ignore.
*   The **`additive`** noise model is a bbobax extension with no COCO
    counterpart.

## Installation

We recommend using [uv](https://github.com/astral-sh/uv) for a fast and reliable installation, but standard `pip` is also supported.

### Using uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/maxencefaldor/bbobax.git
cd bbobax

# Create a virtual environment
uv venv
source .venv/bin/activate

# Install dependencies and the package in editable mode
uv pip install -e .
```

### Using pip

```bash
pip install bbobax
# Or install from source
pip install -e .
```

**Note on JAX**: You may need to install the specific version of JAX compatible with your CUDA/cuDNN version. Please refer to the [JAX installation guide](https://github.com/google/jax#installation) for details.

## Usage

### Basic BBOB Example

Each of the 24 functions is a class. Instantiating one fixes the function and
the dimension; `sample` draws an instance of it.

```python
import jax
from bbobax import Sphere

# One function at one dimension -- that is the whole problem
problem = Sphere(num_dims=10)

key_sample, key_x, key_eval = jax.random.split(jax.random.key(0), 3)

# Draw an instance: its optimum, rotations, and noise settings
params = problem.sample(key_sample)

# Sample a random solution in the search space
x = problem.sample_x(key_x)

# Evaluate it
evaluation = problem.evaluate(key_eval, x, params)

print(f"Function: {problem.name}")
print(f"Dimensions: {problem.num_dims}")
print(f"Fitness: {evaluation.fitness}")
```

### Quality-Diversity (QD) Example

BBOBax also supports Quality-Diversity, where solutions are scored on both
fitness and a behaviour descriptor. A descriptor is orthogonal to a function —
any of the 24 pairs with any descriptor — so a QD problem is **composed**:

```python
import jax
from bbobax import QDProblem, RandomProjection, Rastrigin

problem = QDProblem(
    Rastrigin(num_dims=10),
    RandomProjection(descriptor_size=2),
)

key_sample, key_x, key_eval = jax.random.split(jax.random.key(42), 3)

params = problem.sample(key_sample)
x = problem.sample_x(key_x)
evaluation = problem.evaluate(key_eval, x, params)

print(f"Fitness: {evaluation.fitness}")
print(f"Descriptor: {evaluation.descriptor}")
```

### Covering the suite

A problem is one function, so covering many means holding many problems. The
loop unrolls under `jit`, which is both faithful to COCO's structure and faster
than dispatching inside one problem.

```python
import jax
import bbobax

problems = bbobax.suite(num_dims=10)  # the 24 standard functions

key = jax.random.key(0)
for name, problem in problems.items():
    key, key_sample, key_x, key_eval = jax.random.split(key, 4)
    params = problem.sample(key_sample)
    x = problem.sample_x(key_x)
    evaluation = problem.evaluate(key_eval, x, params)
    print(f"{name:>28}: {evaluation.fitness:.4g}")
```

### Meta-learning across functions and dimensions

Array shapes are static in JAX, so a batch cannot mix dimensions. The dimension
is therefore a Python loop variable and the batch axis is instances *within* a
dimension — which covers every dimension on every meta-step instead of sampling
one, and needs only the six compilations `bbobax.DIMENSIONS` names.

```python
import jax
import bbobax

for num_dims in bbobax.DIMENSIONS:  # (2, 3, 5, 10, 20, 40), COCO's own set
    for name, problem in bbobax.suite(num_dims=num_dims).items():
        keys = jax.random.split(jax.random.key(0), 32)
        params = jax.vmap(problem.sample)(keys)  # 32 instances, batched
        ...
```

This requires the meta-learned parameters to be dimension-independent, which is
a property of the algorithm rather than of the benchmark.

## Documentation

Full documentation is available at [https://maxencefaldor.github.io/bbobax/](https://maxencefaldor.github.io/bbobax/).

To build the documentation locally:

```bash
# Install documentation dependencies
uv pip install -e ".[docs]"

# Serve the documentation
mkdocs serve
```

## Citation

If you use BBOBax in your research, please cite it using the following metadata:

```yaml
title: "BBOBax"
abstract: "BBOBax: Accelerated Black-Box Optimization Benchmark in JAX."
authors:
  - family-names: "Faldor"
    given-names: "Maxence"
    orcid: "https://orcid.org/0000-0003-4743-9494"
repository-code: "https://github.com/maxencefaldor/bbobax"
type: software
```

## References

This library is built upon the standard BBOB function definitions. Please verify the details in the provided documentation:

*   **Noiseless Functions**: Hansen, N., Finck, S., Ros, R., & Auger, A. (2009). *Real-parameter black-box optimization benchmarking 2009: Noiseless functions definitions*. [PDF](https://github.com/maxencefaldor/bbobax/raw/main/docs/assets/bbob-noiseless.pdf)
*   **Noisy Functions**: Hansen, N., Finck, S., Ros, R., & Auger, A. (2009). *Real-parameter black-box optimization benchmarking 2009: Noisy functions definitions*. [PDF](https://github.com/maxencefaldor/bbobax/raw/main/docs/assets/bbob-noisy.pdf)
