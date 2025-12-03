"""Tests for BBOBax task management."""

import jax

from bbobax.bbob import BBOB, QDBBOB, BBOBParams, BBOBState, QDBBOBParams
from bbobax.descriptor_fns import get_random_projection_descriptor
from bbobax.fitness_fns import bbob_fns


def test_bbob_initialization():
    """Test BBOB class initialization."""
    # Default
    task = BBOB.create_default()
    assert len(task.fitness_fns) == len(bbob_fns)
    assert task.min_num_dims == 2
    assert task.max_num_dims == 10

    # Custom
    task_custom = BBOB(fitness_fns=[bbob_fns["sphere"]], min_num_dims=5, max_num_dims=5)
    assert len(task_custom.fitness_fns) == 1
    assert task_custom.min_num_dims == 5


def test_bbob_workflow():
    """Test complete BBOB workflow: sample -> init -> evaluate."""
    task = BBOB.create_default(max_num_dims=5)
    key = jax.random.key(0)

    # Sample task parameters
    key_sample, key_init, key_eval = jax.random.split(key, 3)
    params = task.sample(key_sample)

    assert isinstance(params, BBOBParams)
    assert params.num_dims >= task.min_num_dims
    assert params.num_dims <= task.max_num_dims

    # Initialize state
    state = task.init(key_init, params)
    assert isinstance(state, BBOBState)
    assert state.r.shape == (task.max_num_dims, task.max_num_dims)

    # Sample solution
    x = task.sample_x(key_eval)
    assert x.shape == (task.max_num_dims,)

    # Evaluate
    new_state, result = task.evaluate(key_eval, x, state, params)

    assert new_state.counter == state.counter + 1
    assert result.fitness.shape == ()


def test_qdbbob_workflow():
    """Test complete QD-BBOB workflow."""
    descriptor_size = 2
    descriptor_fn = get_random_projection_descriptor()

    task = QDBBOB(
        descriptor_fns=[descriptor_fn],
        fitness_fns=[bbob_fns["sphere"]],
        descriptor_size=descriptor_size,
        max_num_dims=5,
    )

    key = jax.random.key(0)

    # Sample
    params = task.sample(key)
    assert isinstance(params, QDBBOBParams)
    assert params.descriptor_params.shape == (descriptor_size, task.max_num_dims)

    # Init
    state = task.init(key, params)

    # Evaluate
    x = task.sample_x(key)
    new_state, result = task.evaluate(key, x, state, params)

    assert result.fitness.shape == ()
    assert result.descriptor.shape == (descriptor_size,)


def test_task_jit_vmap():
    """Test JAX transformations on task evaluation."""
    task = BBOB.create_default(max_num_dims=5)
    key = jax.random.key(0)
    params = task.sample(key)
    state = task.init(key, params)

    # JIT evaluate
    # Note: we usually JIT the method bound to the instance or a wrapper
    # Here we verify the method is JIT-able

    @jax.jit
    def step(k, x, s, p):
        return task.evaluate(k, x, s, p)

    x = task.sample_x(key)
    step(key, x, state, params)

    # VMAP evaluate (batch of solutions)
    batch_size = 10

    def batch_step(k, x, s, p):
        # We vmap over keys and x, keep state and params fixed
        return task.evaluate(k, x, s, p)

    batch_step_vmapped = jax.vmap(batch_step, in_axes=(0, 0, None, None))

    keys = jax.random.split(key, batch_size)
    xs = jax.random.uniform(key, (batch_size, task.max_num_dims))

    new_state_batch, results_batch = batch_step_vmapped(keys, xs, state, params)

    assert results_batch.fitness.shape == (batch_size,)
