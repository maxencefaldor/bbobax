<!-- rumdl-disable MD033 MD041 -->
<div align="center">
  <h1>BBOBax</h1>
  <p>Black-Box Optimization Benchmarking in JAX.</p>
  <a href="https://pypi.python.org/pypi/bbobax"><img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/bbobax.svg?style=flat"></img></a>
  <a href="https://pypi.python.org/pypi/bbobax"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/bbobax.svg?style=flat"></img></a>
  <a href="https://github.com/maxencefaldor/bbobax/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/pypi/l/bbobax.svg?style=flat"></img></a>
  <a href="https://github.com/google/jax"><img alt="JAX" src="https://img.shields.io/badge/JAX-Accelerated-9cf"></img></a>
</div>
<br>
<!-- rumdl-enable MD033 MD041 -->

BBOBax is a JAX implementation of [COCO](https://coco-platform.org/)'s BBOB benchmark: the 24 standard noiseless functions, the noisy suite f101–f130, and Many-Affine BBOB.
Every function is jit-compilable and vmap-friendly, so populations, instances, and whole experiments run as single compiled programs on CPU, GPU, or TPU — and every value is verified numerically against the official implementation.

## Why BBOBax?

- **Faithful** 🎯 — verified against the official 2009 BBOB implementation on identical instances, to ≤ 1e-9 relative in float64, re-proven on every test run.
  Where the papers and the official code disagree, the code wins, with a comment at the site.
- **Fast** ⚡ — one `vmap` evaluates a population, another batches instances, and `jit` compiles the whole optimization loop.
  Nothing is stateful, so nothing blocks vectorization.
- **Simple** 🌱 — one contract: a problem is one function at one dimension, `sample(key)` draws an instance, `evaluate(key, x, params)` scores a solution.
  A suite is a plain `dict`.
- **Composable** 🧩 — noise models are held by problems, any function pairs with any descriptor for Quality-Diversity, and Many-Affine blends all 24 into a continuous space of problems.

## Suites

| Suite | Problems | Reference |
| --- | --- | --- |
| `bbob_suite()` | f1–f24, the standard noiseless functions | [Hansen et al., 2009](https://github.com/maxencefaldor/bbobax/raw/main/docs/assets/bbob-noiseless.pdf) |
| `bbob_noisy_suite()` | f101–f130, eight functions under the three official noise models | [Hansen et al., 2009](https://github.com/maxencefaldor/bbobax/raw/main/docs/assets/bbob-noisy.pdf) |
| `ManyAffine` | a continuous space of problems blended from the 24 | [Vermetten et al., 2023](https://arxiv.org/abs/2312.11083) |

A suite is a `dict[str, BBOBProblem]` — composing your own is a dict comprehension.

## Getting Started

```python
import jax

from bbobax import Rastrigin

problem = Rastrigin(num_dims=10)  # one function at one dimension

key_instance, key_x, key_eval = jax.random.split(jax.random.key(0), 3)
params = problem.sample(key_instance)  # draw an instance: optimum, rotations, noise

# Evaluate a whole population in one compiled call
population = jax.vmap(problem.sample_x)(jax.random.split(key_x, 1024))
keys = jax.random.split(key_eval, 1024)
fitness = jax.vmap(problem.evaluate, in_axes=(0, 0, None))(
    keys, population, params
).fitness  # (1024,)
```

Quality-Diversity is composition — any function pairs with any of six descriptor families (`RandomProjection`, `IrregularProjection`, `QuantizedProjection`, `FourierProjection`, `SubsetProjection`, `AlignedProjection`), each isolating one phenomenon of real behavior maps.
Descriptor space is `[-1, 1]^k` by construction, so `problem.descriptor_range` is exact ground truth — whether your algorithm is told is your experiment's choice:

```python
from bbobax import QDProblem, RandomProjection, Rastrigin

problem = QDProblem(Rastrigin(num_dims=10), RandomProjection(descriptor_size=2))
evaluation = problem.evaluate(key, x, params)  # .fitness and .descriptor
```

For `Sphere` under a linear descriptor, the best fitness achievable at any descriptor value has a closed form — `sphere_descriptor_optimum` — giving QD archives an exact per-cell reference no other benchmark provides.

To cover many functions or dimensions, hold many problems and loop — the loop unrolls under `jit`, so every problem keeps its own compiled code and none pays for dispatch:

```python
import bbobax

for num_dims in bbobax.DIMENSIONS:  # (2, 3, 5, 10, 20, 40), COCO's own set
    for problem in bbobax.bbob_suite(num_dims=num_dims).values():
        params = problem.sample(key)
        ...
```

## Tutorials

| Notebook | Colab |
| --- | --- |
| Getting Started — the contract on one function | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxencefaldor/bbobax/blob/main/notebooks/00_getting_started.ipynb) |
| The Suites — bbob, bbob-noisy, and Many-Affine | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxencefaldor/bbobax/blob/main/notebooks/01_suites.ipynb) |
| Black-Box Optimization — one problem, one evolution strategy | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxencefaldor/bbobax/blob/main/notebooks/02_bbo.ipynb) |
| Benchmarking — six algorithms across the suite, the COCO way | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxencefaldor/bbobax/blob/main/notebooks/03_benchmarking.ipynb) |
| Meta-Black-Box Optimization — meta-learning a component over the suite | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxencefaldor/bbobax/blob/main/notebooks/04_meta_bbo.ipynb) |
| Quality-Diversity — five algorithms on one skeleton | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxencefaldor/bbobax/blob/main/notebooks/05_qd.ipynb) |

## Fidelity

The official 2009 implementation (`bbobbenchmarks.py`) is vendored as test ground truth, and `tests/test_alignment.py` re-proves the numerical agreement on every run.
Guarantees you can rely on:

- `params.x_opt` is always the function's true argmin: `f(x_opt) = f_opt` for all 24.
  The six functions that constrain where their optimum may sit draw it accordingly.
- Noise applies to the raw function value only; the boundary penalty and `f_opt` are added outside it, and the three official noise models stabilize themselves — below the 1e-8 target precision the undisturbed value comes back, so noise can never hide a solved problem.
- The bbob-noisy suite is not "the 24 plus noise": boundary handling is replaced with a uniform factor of 100, each problem is pinned to one of the paper's two severities, and two of the bases are reparameterized — exactly as in the reference.

Deliberate, documented deviations: instances are *sampled* from a key rather than enumerated from COCO's seed table, `f_opt` defaults to 0, and rotations are Haar on SO(n).
Each deviation is documented where it lives, with the official behavior quoted.

## Precision

BBOBax follows JAX's float32 default, which is right for comparing algorithms and for meta-learning.
Measuring proximity to the optimum at BBOB's 1e-8 target precision is a float64 question — float32 carries about 7 decimal digits, so on a landscape of order 1 the target sits below what the format can represent:

```python
import jax

jax.config.update("jax_enable_x64", True)  # before any array is created
```

The test suite runs in float64; a separate sweep proves all 24 functions stay finite in float32 across all six standard dimensions.

## Installation

BBOBax requires Python 3.11 or later, and a working [JAX](https://docs.jax.dev/en/latest/installation.html) installation.

```bash
pip install bbobax
```

or with [uv](https://github.com/astral-sh/uv):

```bash
uv add bbobax
```

## Citation

If you use BBOBax in your research, please cite:

```bibtex
@software{faldor2025bbobax,
  author = {Faldor, Maxence},
  title = {{BBOBax}: Black-Box Optimization Benchmarking in {JAX}},
  url = {https://github.com/maxencefaldor/bbobax},
  year = {2025},
}
```

## References

- Hansen, N., Finck, S., Ros, R., & Auger, A. (2009).
  *Real-parameter black-box optimization benchmarking 2009: Noiseless functions definitions.*
  [PDF](https://github.com/maxencefaldor/bbobax/raw/main/docs/assets/bbob-noiseless.pdf)
- Hansen, N., Finck, S., Ros, R., & Auger, A. (2009).
  *Real-parameter black-box optimization benchmarking 2009: Noisy functions definitions.*
  [PDF](https://github.com/maxencefaldor/bbobax/raw/main/docs/assets/bbob-noisy.pdf)
- Vermetten, D., Ye, F., Bäck, T., & Doerr, C. (2023).
  *MA-BBOB: A problem generator for black-box optimization using affine combinations and shifts.*
  [arXiv](https://arxiv.org/abs/2312.11083)

## Contributing

Contributions are welcome.
Fidelity is the contract: any new function, model, or suite comes with a numerical alignment test against its official reference.
