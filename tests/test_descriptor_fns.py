"""Tests for BBOBax descriptor functions."""

import jax
import jax.numpy as jnp
import numpy as np

from bbobax.bbob import QDBBOB
from bbobax.descriptor_fns import get_random_projection_descriptor
from bbobax.types import BBOBState


def test_random_projection_descriptor_shape(mock_qdbbob_params):
    """Test shape of random projection descriptor."""
    num_dims = 10
    descriptor_size = 2

    descriptor_fn = get_random_projection_descriptor()

    x = jnp.ones(num_dims)
    state = BBOBState()
    params = mock_qdbbob_params(num_dims, descriptor_size)

    descriptor = descriptor_fn(x, state, params)

    assert descriptor.shape == (descriptor_size,)
    # It is exactly the projection of x, no masking anywhere.
    np.testing.assert_allclose(
        np.asarray(descriptor), np.asarray(params.descriptor_params) @ np.ones(num_dims)
    )


def test_random_projection_descriptor_jit_vmap(mock_qdbbob_params):
    """Test JAX transformations on descriptor function."""
    num_dims = 10
    descriptor_size = 3
    batch_size = 8

    descriptor_fn = get_random_projection_descriptor()

    state = BBOBState()
    params = mock_qdbbob_params(num_dims, descriptor_size)

    # Test JIT
    jitted_fn = jax.jit(descriptor_fn)
    x = jnp.ones(num_dims)
    desc = jitted_fn(x, state, params)
    assert desc.shape == (descriptor_size,)

    # Test VMAP
    vmapped_fn = jax.vmap(descriptor_fn, in_axes=(0, None, None))
    x_batch = jnp.ones((batch_size, num_dims))
    desc_batch = vmapped_fn(x_batch, state, params)

    assert desc_batch.shape == (batch_size, descriptor_size)


def test_random_projection_descriptor_on_a_real_task():
    """The projection is instance data of the task's own dimension."""
    num_dims = 6
    descriptor_size = 2

    task = QDBBOB(
        fitness_fn="rastrigin",
        descriptor_fn=get_random_projection_descriptor(),
        descriptor_size=descriptor_size,
        num_dims=num_dims,
    )
    params = task.sample(jax.random.key(0))
    state = task.init(params)

    assert params.descriptor_params.shape == (descriptor_size, num_dims)

    x = task.sample_x(jax.random.key(1))
    _, result = task.evaluate(jax.random.key(2), x, state, params)
    np.testing.assert_allclose(
        np.asarray(result.descriptor),
        np.asarray(params.descriptor_params) @ np.asarray(x),
        rtol=1e-12,
    )
