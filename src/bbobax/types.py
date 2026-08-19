"""Black-box Optimization Benchmarking Types."""

from collections.abc import Callable
from typing import TypeAlias

import jax
from flax.struct import dataclass

# A scalar that is a Python value when built by hand and a traced array when it
# comes out of `sample`; both are valid everywhere it appears.
IntScalar: TypeAlias = int | jax.Array

# A fitness function returns (value, boundary penalty); `BBOB.evaluate` adds
# f_opt and applies noise to the value alone. A descriptor function returns the
# descriptor vector.
FitnessFn: TypeAlias = Callable[
    [jax.Array, "BBOBState", "BBOBParams"], tuple[jax.Array, jax.Array]
]
DescriptorFn: TypeAlias = Callable[[jax.Array, "BBOBState", "QDBBOBParams"], jax.Array]


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
    """BBOB task parameters.

    Everything that defines the instance lives here, drawn once by
    `BBOB.sample` and never mutated:

    The function and the dimension are the task's, not the instance's -- COCO
    enumerates those and draws only the instance, and so does bbobax.

    - `x_opt` is the function's true argmin. The per-function conventions
      (sign vectors, scalings, sign-forcing) are applied at sampling time,
      mirroring how COCO stores the post-convention optimum.
    - `r` and `q` are the instance's rotation matrices, the R and Q of the
      function definitions.
    - `key` is the instance's own PRNG key, for instance structure that is
      cheaper to derive than to store -- the Gallagher functions draw their
      peak layouts from it.
    """

    x_opt: jax.Array
    f_opt: jax.Array
    r: jax.Array
    q: jax.Array
    key: jax.Array
    noise_params: NoiseParams


@dataclass
class QDBBOBParams(BBOBParams):
    """QD-BBOB task parameters."""

    descriptor_params: jax.Array


@dataclass
class BBOBState:
    """BBOB task state: what changes as an instance is evaluated.

    Only the evaluation counter. The rotation matrices used to live here, but
    they are drawn once per instance and never mutated, which makes them
    parameters -- see `BBOBParams`.
    """

    counter: IntScalar = 0


@dataclass
class BBOBEval:
    """BBOB evaluation results."""

    fitness: jax.Array


@dataclass
class QDBBOBEval(BBOBEval):
    """QD-BBOB evaluation results."""

    descriptor: jax.Array
