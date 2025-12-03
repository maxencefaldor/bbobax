"""Integration tests for BBOBax."""

import jax
import jax.numpy as jnp

from bbobax.bbob import BBOB, QDBBOB
from bbobax.descriptor_fns import get_random_projection_descriptor


def test_bbob_optimization_loop():
    """Test a simple random search optimization loop on BBOB."""
    max_dims = 5
    population_size = 20
    num_generations = 5

    # Initialize task
    task = BBOB.create_default(max_num_dims=max_dims)
    key = jax.random.key(42)

    # Sample task instance
    key_sample, key_loop = jax.random.split(key)
    params = task.sample(key_sample)
    state = task.init(key_sample, params)

    # Optimization loop
    current_best_fitness = -jnp.inf

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
            (population_size, max_dims),
            minval=task.x_range[0],
            maxval=task.x_range[1],
        )

        # Evaluate
        # Note: state update in batch is tricky because we get N new states.
        # BBOB state only contains counters and rotation matrices (constant).
        # The counter increment is the only change.
        # In a real setting, we might just take one state or not update state in vmap
        # if read-only.
        # Here we just take the first state from the batch for next iteration,
        # accepting that counters might diverge if we were tracking exact evals.
        new_states, results = batch_evaluate_vmapped(keys, xs, state, params)

        # Update best fitness
        gen_best = jnp.max(results.fitness)
        current_best_fitness = jnp.maximum(current_best_fitness, gen_best)

        # Update state (just picking one for continuity)
        # In reality, BBOB state is mostly static or per-eval-independent
        # (except counter for logging/termination)
        state = jax.tree_util.tree_map(lambda x: x[0], new_states)

    # Check that we got a valid fitness
    assert not jnp.isnan(current_best_fitness)
    assert not jnp.isinf(current_best_fitness)
    assert current_best_fitness > -1e9  # Reasonable lower bound


def test_qdbbob_optimization_loop():
    """Test a simple random search optimization loop on QD-BBOB."""
    max_dims = 5
    population_size = 20
    num_generations = 5
    descriptor_size = 2

    # Initialize task
    descriptor_fn = get_random_projection_descriptor()
    task = QDBBOB.create_default(
        descriptor_fns=[descriptor_fn],
        descriptor_size=descriptor_size,
        max_num_dims=max_dims,
    )
    key = jax.random.key(42)

    # Sample task instance
    key_sample, key_loop = jax.random.split(key)
    params = task.sample(key_sample)
    state = task.init(key_sample, params)

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
            (population_size, max_dims),
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
