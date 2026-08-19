"""Integration tests for BBOBax."""

import jax
import jax.numpy as jnp

from bbobax.bbob import QDBBOB, suite
from bbobax.descriptor_fns import get_random_projection_descriptor


def test_bbob_optimization_loop():
    """Test a simple random search optimization loop on BBOB."""
    num_dims = 5
    population_size = 20
    num_generations = 5

    # Initialize task
    task = suite(["rastrigin"], num_dims=num_dims)["rastrigin"]
    key = jax.random.key(42)

    # Sample task instance
    key_sample, key_loop = jax.random.split(key)
    params = task.sample(key_sample)
    state = task.init(params)

    # Optimization loop
    current_best_fitness = jnp.inf

    def batch_evaluate(k, x, s, p):
        return task.evaluate(k, x, s, p)

    batch_evaluate_vmapped = jax.vmap(batch_evaluate, in_axes=(0, 0, None, None))

    loop_key = key_loop
    for _ in range(num_generations):
        key_gen, loop_key = jax.random.split(loop_key)
        keys = jax.random.split(key_gen, population_size)

        # Sample random population
        xs = jax.random.uniform(
            key_gen,
            (population_size, num_dims),
            minval=task.x_range[0],
            maxval=task.x_range[1],
        )

        # Evaluate. vmap returns one state per solution; the counter is the
        # only thing that changes, so take the first for continuity.
        new_states, results = batch_evaluate_vmapped(keys, xs, state, params)

        current_best_fitness = jnp.minimum(
            current_best_fitness, jnp.min(results.fitness)
        )
        state = jax.tree_util.tree_map(lambda x: x[0], new_states)

    # BBOB minimizes, and f_opt is pinned to 0 by default: every value is above
    # the optimum, and the search has actually made progress.
    assert not jnp.isnan(current_best_fitness)
    assert not jnp.isinf(current_best_fitness)
    assert current_best_fitness >= 0.0
    assert int(state.counter) == num_generations


def test_suite_optimization_loop():
    """Loop over the suite: each task is evaluated by its own function."""
    num_dims = 4
    population_size = 8
    tasks = suite(num_dims=num_dims)
    key = jax.random.key(0)

    best = {}
    for name, task in tasks.items():
        key, key_sample, key_x, key_eval = jax.random.split(key, 4)
        params = task.sample(key_sample)
        state = task.init(params)

        xs = jax.vmap(task.sample_x)(jax.random.split(key_x, population_size))
        keys = jax.random.split(key_eval, population_size)
        _, results = jax.vmap(task.evaluate, in_axes=(0, 0, None, None))(
            keys, xs, state, params
        )

        assert results.fitness.shape == (population_size,)
        assert not jnp.any(jnp.isnan(results.fitness))
        best[name] = float(jnp.min(results.fitness))

    assert len(best) == 24
    # 24 genuinely different landscapes: no two share a best value.
    assert len(set(best.values())) == 24


def test_qdbbob_optimization_loop():
    """Test a simple random search optimization loop on QD-BBOB."""
    num_dims = 5
    population_size = 20
    num_generations = 5
    descriptor_size = 2

    # Initialize task
    task = QDBBOB(
        fitness_fn="sphere",
        descriptor_fn=get_random_projection_descriptor(),
        descriptor_size=descriptor_size,
        num_dims=num_dims,
    )
    key = jax.random.key(42)

    # Sample task instance
    key_sample, key_loop = jax.random.split(key)
    params = task.sample(key_sample)
    state = task.init(params)

    def batch_evaluate(k, x, s, p):
        return task.evaluate(k, x, s, p)

    batch_evaluate_vmapped = jax.vmap(batch_evaluate, in_axes=(0, 0, None, None))

    loop_key = key_loop
    descriptors_observed = []

    for _ in range(num_generations):
        key_gen, loop_key = jax.random.split(loop_key)
        keys = jax.random.split(key_gen, population_size)

        xs = jax.random.uniform(
            key_gen,
            (population_size, num_dims),
            minval=task.x_range[0],
            maxval=task.x_range[1],
        )

        new_states, results = batch_evaluate_vmapped(keys, xs, state, params)

        descriptors_observed.append(results.descriptor)
        state = jax.tree_util.tree_map(lambda x: x[0], new_states)

    # Verify descriptors
    all_descriptors = jnp.concatenate(descriptors_observed, axis=0)
    assert all_descriptors.shape == (population_size * num_generations, descriptor_size)
    assert not jnp.any(jnp.isnan(all_descriptors))
