"""Black-box Optimization Benchmarking Types."""

from typing import Any, TypeAlias

import jax
from flax.struct import dataclass

# A scalar that is a Python value when built by hand and a traced array when it
# comes out of `sample`; both are valid everywhere it appears.
IntScalar: TypeAlias = int | jax.Array


@dataclass
class NoiseParams:
    """The noise model and its settings, drawn per instance."""

    noise_id: jax.Array
    gaussian_beta: jax.Array
    uniform_alpha: jax.Array
    uniform_beta: jax.Array
    cauchy_alpha: jax.Array
    cauchy_p: jax.Array
    additive_std: jax.Array


@dataclass
class BBOBParams:
    """One sampled instance of a problem.

    Everything that varies between instances lives here, drawn once by
    `BBOBProblem.sample` and never mutated. The function and the dimension are
    the *problem's*, not the instance's -- COCO enumerates those and draws only
    the instance, and so does bbobax.

    - `key` is the instance's own PRNG key, for instance structure that is
      cheaper to derive than to store -- the Gallagher functions draw their
      peak layouts from it.
    - `x_opt` is the function's true argmin. The constraint each definition
      places on its optimum is applied at sampling time, mirroring how COCO
      stores the post-constraint optimum.
    - `r` and `q` are the instance's rotation matrices, the R and Q of the
      function definitions.
    """

    key: jax.Array
    x_opt: jax.Array
    f_opt: jax.Array
    r: jax.Array
    q: jax.Array
    noise_params: NoiseParams


@dataclass
class BBOBEval:
    """What evaluating a solution yields."""

    fitness: jax.Array


@dataclass
class QDParams:
    """One sampled instance of a Quality-Diversity problem.

    Composed, not inherited: a QD problem is a problem paired with a
    descriptor, and each half draws its own instance data. `descriptor_params`
    is whatever that descriptor's `sample` returns -- a projection matrix for
    `RandomProjection`, something else for a descriptor that needs more.
    """

    problem_params: BBOBParams
    descriptor_params: Any


@dataclass
class QDEval:
    """What evaluating a solution on a Quality-Diversity problem yields."""

    fitness: jax.Array
    descriptor: jax.Array
