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
*   **Quality-Diversity (QD)**: Support for Quality-Diversity benchmarks with customizable descriptor functions.
*   **Flexible API**: Easy integration with existing JAX-based evolutionary computation libraries (e.g., EvoJAX, evosax).

## Fidelity to official BBOB

Every function and noise model is verified **numerically** against the official
2009 BBOB implementation (`bbobbenchmarks.py`, vendored as test ground truth):
identical instances are injected and outputs must agree to ≤1e-9 relative in
float64 — the suite in `tests/test_alignment.py` re-proves this on every run.
Where the paper (Hansen, Finck, Ros & Auger, INRIA RR-6829/RR-6869) and the
official code disagree, the code wins, with a comment at the site.

Guarantees you can rely on:

*   `params.x_opt` is always the function's **true argmin** — the per-function
    x_opt conventions of the official suite (sign vectors, scalings,
    sign-forcing) are applied at sampling time, exactly as COCO stores the
    post-convention optimum. `f(params.x_opt) = f_opt` for all 24 functions.
*   Noise applies to the raw function value only; the boundary penalty and
    `f_opt` are added outside it, as the paper prescribes.
*   The default `BBOB()` is the plain **noiseless** suite with rotations on —
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
*   **A sampled dimension range embeds rather than resizes.** `num_dims=10`
    gives a genuine 10-D problem, identical to COCO's. `num_dims=(2, 10)`
    samples the dimension per instance, but solutions are always 10 long and
    the coordinates beyond an instance's own dimension are inert — they move
    neither fitness nor descriptor. That is a D-dimensional problem *embedded
    in a 10-dimensional search space*: an optimizer still adapts over all ten
    coordinates and has to discover which do nothing. It is the right shape
    for meta-learning across dimensions, and the wrong one for comparing
    against published BBOB results — fix the dimension for that.
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

Here is a minimal example of how to initialize a BBOB task, sample a problem instance, and evaluate a solution.

```python
import jax
from bbobax import BBOB

# Initialize BBOB task with default functions
bbob = BBOB.create_default(num_dims=(2, 10))

# Sample a task instance (function ID, dimensions, optimal values, etc.)
key = jax.random.key(0)
key_task, key_init, key_eval, key_x = jax.random.split(key, 4)
task_params = bbob.sample(key_task)

# Initialize the evaluation state (rotations live in the params)
state = bbob.init(task_params)

# Sample a random solution in the search space
x = bbob.sample_x(key_x)

# Evaluate the solution
# Returns updated state and evaluation metrics (fitness)
state, eval_metrics = bbob.evaluate(key_eval, x, state, task_params)

print(f"Function ID: {task_params.fn_id}")
print(f"Dimensions: {task_params.num_dims}")
print(f"Fitness: {eval_metrics.fitness}")
```

### Quality-Diversity (QD) Example

BBOBax also supports Quality-Diversity optimization tasks where solutions are evaluated based on both fitness and a behavior descriptor.

```python
import jax
from bbobax import QDBBOB, bbob_fns, get_random_projection_descriptor

# Define descriptor functions (e.g., random projection)
descriptor_fns = [get_random_projection_descriptor()]

# Initialize QD-BBOB task
qd_bbob = QDBBOB(
    descriptor_fns=descriptor_fns,
    fitness_fns=bbob_fns,
    descriptor_size=2,
    num_dims=10
)

# Sample task
key = jax.random.key(42)
key_task, key_init, key_eval, key_x = jax.random.split(key, 4)
task_params = qd_bbob.sample(key_task)

# Initialize state
state = qd_bbob.init(task_params)

# Sample solution
x = qd_bbob.sample_x(key_x)

# Evaluate
state, eval_metrics = qd_bbob.evaluate(key_eval, x, state, task_params)

print(f"Fitness: {eval_metrics.fitness}")
print(f"Descriptor: {eval_metrics.descriptor}")
```

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
