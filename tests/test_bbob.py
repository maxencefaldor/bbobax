"""Tests for BBOBax task management."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bbobax.bbob import BBOB, QDBBOB, BBOBParams, BBOBState, QDBBOBParams, suite
from bbobax.descriptor_fns import get_random_projection_descriptor
from bbobax.fitness_fns import bbob_fns
from bbobax.noise import noiseless_noise


def test_bbob_initialization():
    """One task = one function at one fixed dimension."""
    # A name selects the function; the default dimension is BBOB's usual 10.
    task = BBOB("sphere")
    assert task.name == "sphere"
    assert task.fitness_fn is bbob_fns["sphere"]
    assert task.num_dims == 10

    # num_dims is a single int, fixed for the task.
    task_custom = BBOB("rastrigin", num_dims=5)
    assert task_custom.num_dims == 5
    assert task_custom.fitness_fn is bbob_fns["rastrigin"]

    # A bare callable is accepted, and gets the raw x_opt draw.
    def my_fn(x, state, params):
        return jnp.sum(jnp.square(x - params.x_opt)), jnp.array(0.0)

    task_callable = BBOB(my_fn, num_dims=4)
    assert task_callable.name == "my_fn"
    assert task_callable.fitness_fn is my_fn


def test_bbob_initialization_rejects_bad_arguments():
    """Unknown function -> KeyError; num_dims below 2 -> ValueError."""
    with pytest.raises(KeyError):
        BBOB("not_a_bbob_function")

    with pytest.raises(ValueError, match="num_dims"):
        BBOB("sphere", num_dims=1)

    # 2 is the smallest dimension BBOB is defined for, and is accepted.
    assert BBOB("sphere", num_dims=2).num_dims == 2


def test_bbob_workflow():
    """Test complete BBOB workflow: sample -> init -> evaluate."""
    num_dims = 5
    task = BBOB("sphere", num_dims=num_dims)
    key = jax.random.key(0)

    key_sample, key_eval = jax.random.split(key)

    # Sample an instance: every array is exactly num_dims long.
    params = task.sample(key_sample)
    assert isinstance(params, BBOBParams)
    assert params.x_opt.shape == (num_dims,)
    assert params.f_opt.shape == ()
    # Rotations are instance parameters, not state: drawn once by sample().
    assert params.r.shape == (num_dims, num_dims)
    assert params.q.shape == (num_dims, num_dims)

    # Initialize state
    state = task.init(params)
    assert isinstance(state, BBOBState)
    assert int(state.counter) == 0

    # Sample solution
    x = task.sample_x(key_eval)
    assert x.shape == (num_dims,)

    # Evaluate
    new_state, result = task.evaluate(key_eval, x, state, params)

    assert new_state.counter == state.counter + 1
    assert result.fitness.shape == ()


def test_qdbbob_workflow():
    """Test complete QD-BBOB workflow."""
    num_dims = 5
    descriptor_size = 2

    task = QDBBOB(
        fitness_fn="sphere",
        descriptor_fn=get_random_projection_descriptor(),
        descriptor_size=descriptor_size,
        num_dims=num_dims,
    )

    key = jax.random.key(0)

    # Sample
    params = task.sample(key)
    assert isinstance(params, QDBBOBParams)
    assert params.descriptor_params.shape == (descriptor_size, num_dims)

    # Init
    state = task.init(params)

    # Evaluate
    x = task.sample_x(key)
    new_state, result = task.evaluate(key, x, state, params)

    assert new_state.counter == state.counter + 1
    assert result.fitness.shape == ()
    assert result.descriptor.shape == (descriptor_size,)


def test_task_jit_vmap():
    """Test JAX transformations on task evaluation."""
    num_dims = 5
    task = BBOB("rastrigin", num_dims=num_dims)
    key = jax.random.key(0)
    params = task.sample(key)
    state = task.init(params)

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
    xs = jax.random.uniform(key, (batch_size, num_dims))

    new_state_batch, results_batch = batch_step_vmapped(keys, xs, state, params)

    assert results_batch.fitness.shape == (batch_size,)


def test_suite_builds_the_standard_24():
    """suite() is the 24 standard functions as 24 separate tasks."""
    tasks = suite(num_dims=6)

    assert list(tasks) == list(bbob_fns)
    assert len(tasks) == 24

    # 24 distinct task objects, each naming and holding its own function.
    assert len({id(task) for task in tasks.values()}) == 24
    for name, task in tasks.items():
        assert task.name == name
        assert task.num_dims == 6
        # The point of the refactor: the task holds the function directly, so
        # evaluation calls it -- no lax.switch over 24 branches, and nothing
        # pays for the 23 branches it does not want.
        assert task.fitness_fn is bbob_fns[name]

    # A subset is selectable, and kwargs reach every task.
    subset = suite(["sphere", "discus"], num_dims=3, clip_x=True)
    assert list(subset) == ["sphere", "discus"]
    assert all(task.num_dims == 3 and task.clip_x for task in subset.values())


def test_suite_evaluation_has_no_dispatch():
    """Each suite task evaluates through its own function, at its own optimum."""
    tasks = suite(num_dims=4)
    key = jax.random.key(11)

    for name, task in tasks.items():
        params = task.sample(key)
        state = task.init(params)
        _, result = task.evaluate(key, params.x_opt, state, params)
        # Noiseless default and f_opt pinned to 0: the optimum evaluates to 0.
        assert float(result.fitness) == pytest.approx(0.0, abs=1e-9), name


@pytest.mark.parametrize("num_dims", [2, 5, 10])
def test_generate_random_rotation(num_dims):
    """The static rotation generator returns a matrix in SO(n)."""
    rotation = np.asarray(BBOB.generate_random_rotation(jax.random.key(3), num_dims))

    assert rotation.shape == (num_dims, num_dims)
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(num_dims), atol=1e-10)
    np.testing.assert_allclose(np.linalg.det(rotation), 1.0, atol=1e-10)

    # Deterministic in the key, and a different key gives a different rotation.
    again = np.asarray(BBOB.generate_random_rotation(jax.random.key(3), num_dims))
    np.testing.assert_array_equal(rotation, again)
    other = np.asarray(BBOB.generate_random_rotation(jax.random.key(4), num_dims))
    assert not np.allclose(rotation, other)


def test_sample_rotation_off_gives_identity():
    """With sample_rotation off, R = Q = I -- deliberately, not by default."""
    task = BBOB("ellipsoidal_rotated", num_dims=5, sample_rotation=False)
    params = task.sample(jax.random.key(0))

    np.testing.assert_array_equal(np.asarray(params.r), np.eye(5))
    np.testing.assert_array_equal(np.asarray(params.q), np.eye(5))


def test_default_noise_is_noiseless():
    """The default task is COCO's noiseless suite: evaluation is deterministic."""
    task = BBOB("sphere", num_dims=5)

    # Only the noiseless model is in the pool, and stabilization is off.
    assert task.noise_model.noise_models == [noiseless_noise]
    assert task.noise_model.use_stabilization is False

    params = task.sample(jax.random.key(0))
    state = task.init(params)
    x = task.sample_x(jax.random.key(1))

    # Unbiased: the same solution evaluates identically under any eval key, and
    # equals the raw function value plus penalty plus f_opt.
    values = [
        float(task.evaluate(jax.random.key(k), x, state, params)[1].fitness)
        for k in range(8)
    ]
    assert len(set(values)) == 1

    val, pen = task.fitness_fn(x, state, params)
    assert values[0] == pytest.approx(float(val) + float(pen) + float(params.f_opt))
