"""Tests for BBOBax descriptor functions."""

import jax
import jax.numpy as jnp

from bbobax.descriptor_fns import get_random_projection_descriptor
from bbobax.types import BBOBState


def test_random_projection_descriptor_shape(mock_qdbbob_params):
    """Test shape of random projection descriptor."""
    max_dims = 10
    num_dims = 5
    descriptor_size = 2

    # Initialize descriptor function
    descriptor_fn_builder = get_random_projection_descriptor()

    # Create inputs
    x = jnp.ones(max_dims)
    state = BBOBState(r=jnp.eye(max_dims), q=jnp.eye(max_dims))
    params = mock_qdbbob_params(num_dims, max_dims, descriptor_size)

    # Evaluate
    descriptor = descriptor_fn_builder(x, state, params)

    assert descriptor.shape == (descriptor_size,)


def test_random_projection_descriptor_jit_vmap(mock_qdbbob_params):
    """Test JAX transformations on descriptor function."""
    max_dims = 10
    num_dims = 5
    descriptor_size = 3
    batch_size = 8

    descriptor_fn = get_random_projection_descriptor()

    state = BBOBState(r=jnp.eye(max_dims), q=jnp.eye(max_dims))
    params = mock_qdbbob_params(num_dims, max_dims, descriptor_size)

    # Test JIT
    jitted_fn = jax.jit(descriptor_fn)
    x = jnp.ones(max_dims)
    desc = jitted_fn(x, state, params)
    assert desc.shape == (descriptor_size,)

    # Test VMAP
    vmapped_fn = jax.vmap(descriptor_fn, in_axes=(0, None, None))
    x_batch = jnp.ones((batch_size, max_dims))
    desc_batch = vmapped_fn(x_batch, state, params)

    assert desc_batch.shape == (batch_size, descriptor_size)
