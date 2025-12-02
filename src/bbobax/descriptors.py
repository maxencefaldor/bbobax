"""Black-box Optimization Benchmarking Descriptors Definitions."""

from collections.abc import Callable
from typing import Any

import jax


def get_random_projection_descriptor() -> Callable[[jax.Array, Any, Any], jax.Array]:
    """Return a random projection descriptor function.

    Returns:
        A descriptor function that takes (x, state, params) and returns the
        descriptor.

    """

    def descriptor_fn(x: jax.Array, state: Any, params: Any) -> jax.Array:
        # x shape: (batch_size, max_num_dims)
        # params.descriptor_params shape: (descriptor_size, max_num_dims)
        # Output shape: (batch_size, descriptor_size)
        return x @ params.descriptor_params.T

    return descriptor_fn
