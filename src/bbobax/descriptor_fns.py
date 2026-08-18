"""Black-box Optimization Benchmarking Descriptors Definitions.

A descriptor function has the same shape of contract as a fitness function: it
scores **one** solution, and callers ``vmap`` it to cover a batch. The
Quality-Diversity extension is bbobax's own -- COCO has no descriptor notion.
"""

import jax

from .types import BBOBState, DescriptorFn, QDBBOBParams


def get_random_projection_descriptor() -> DescriptorFn:
    """Return a random-projection descriptor function.

    The projection matrix is instance data, drawn once per instance by
    ``QDBBOB.sample`` and carried in ``params.descriptor_params``: entries are
    ``N(0, 1) / sqrt(descriptor_size)``, so the projection preserves expected
    squared norms, with the columns beyond ``num_dims`` zeroed to match the
    fitness side's masking.

    Returns:
        A descriptor function mapping one solution of shape
        ``(max_num_dims,)`` to a descriptor of shape ``(descriptor_size,)``.

    """

    def descriptor_fn(
        x: jax.Array, state: BBOBState, params: QDBBOBParams
    ) -> jax.Array:
        # x: (max_num_dims,); descriptor_params: (descriptor_size, max_num_dims)
        # -> (descriptor_size,)
        return params.descriptor_params @ x

    return descriptor_fn
