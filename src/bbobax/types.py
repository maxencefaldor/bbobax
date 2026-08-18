"""Black-box Optimization Benchmarking Types."""

import jax
from flax.struct import dataclass

from .noise import NoiseParams


@dataclass
class BBOBParams:
    """BBOB task parameters.

    ``x_opt`` is the function's true argmin: per-function conventions (sign
    vectors, scalings, sign-forcing) are applied at sampling time, mirroring
    how COCO stores the post-convention optimum. ``key`` is the instance's own
    PRNG key, fixed at sampling, for instance structure beyond x_opt -- the
    Gallagher functions draw their peak layouts from it.
    """

    fn_id: jax.Array
    num_dims: jax.Array
    x_opt: jax.Array
    f_opt: jax.Array
    key: jax.Array
    noise_params: NoiseParams


@dataclass
class QDBBOBParams(BBOBParams):
    """QD-BBOB task parameters."""

    descriptor_params: jax.Array
    descriptor_id: jax.Array


@dataclass
class BBOBState:
    """BBOB task state."""

    r: jax.Array
    q: jax.Array
    counter: int = 0


@dataclass
class BBOBEval:
    """BBOB evaluation results."""

    fitness: jax.Array


@dataclass
class QDBBOBEval(BBOBEval):
    """QD-BBOB evaluation results."""

    descriptor: jax.Array
