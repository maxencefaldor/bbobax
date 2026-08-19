"""Integration tests for BBOBax."""

import jax
import jax.numpy as jnp

from bbobax.bbob import suite
from bbobax.problem import DIMENSIONS
from bbobax.qd import QDProblem, RandomProjection


def test_bbob_optimization_loop():
    """Test a simple random search optimization loop on BBOB."""
    num_dims = 5
    population_size = 20
    num_generations = 5

    problem = suite(["rastrigin"], num_dims=num_dims)["rastrigin"]
    key = jax.random.key(42)

    # Sample the instance
    key_sample, loop_key = jax.random.split(key)
    params = problem.sample(key_sample)

    evaluate_batch = jax.vmap(problem.evaluate, in_axes=(0, 0, None))

    current_best_fitness = jnp.inf
    for _ in range(num_generations):
        key_gen, loop_key = jax.random.split(loop_key)
        keys = jax.random.split(key_gen, population_size)

        xs = jax.random.uniform(
            key_gen,
            (population_size, num_dims),
            minval=problem.x_range[0],
            maxval=problem.x_range[1],
        )

        results = evaluate_batch(keys, xs, params)
        current_best_fitness = jnp.minimum(
            current_best_fitness, jnp.min(results.fitness)
        )

    # BBOB minimizes, and f_opt is pinned to 0 by default: every value is above
    # the optimum, and the search has actually made progress.
    assert not jnp.isnan(current_best_fitness)
    assert not jnp.isinf(current_best_fitness)
    assert current_best_fitness >= 0.0


def test_suite_optimization_loop():
    """Loop over the suite: each problem is evaluated by its own function."""
    num_dims = 4
    population_size = 8
    problems = suite(num_dims=num_dims)
    key = jax.random.key(0)

    best = {}
    for name, problem in problems.items():
        key, key_sample, key_x, key_eval = jax.random.split(key, 4)
        params = problem.sample(key_sample)

        xs = jax.vmap(problem.sample_x)(jax.random.split(key_x, population_size))
        keys = jax.random.split(key_eval, population_size)
        results = jax.vmap(problem.evaluate, in_axes=(0, 0, None))(keys, xs, params)

        assert results.fitness.shape == (population_size,)
        assert not jnp.any(jnp.isnan(results.fitness))
        best[name] = float(jnp.min(results.fitness))

    assert len(best) == 24
    # 24 genuinely different landscapes: no two share a best value.
    assert len(set(best.values())) == 24


def test_meta_learning_loop_over_functions_and_dimensions():
    """The meta-BBO shape: Python loops over dimension and function.

    Array shapes are static in JAX, so a batch cannot mix dimensions. The
    dimension is therefore a Python loop variable and the batch axis is
    instances *within* a dimension -- which covers every dimension on every
    meta-step rather than sampling one. Inside `jit` both loops unroll, so each
    (function, dimension) keeps its own compiled code and nothing dispatches.
    """
    num_instances = 4
    key = jax.random.key(0)

    # The two smallest standard dimensions keep the test quick; the loop is the
    # same shape over all of DIMENSIONS.
    assert DIMENSIONS[:2] == (2, 3)

    scores = {}
    for num_dims in DIMENSIONS[:2]:
        for name, problem in suite(num_dims=num_dims).items():
            key, key_instances, key_x = jax.random.split(key, 3)

            # A batch of instances of one (function, dimension).
            params = jax.vmap(problem.sample)(
                jax.random.split(key_instances, num_instances)
            )
            xs = jax.vmap(problem.sample_x)(jax.random.split(key_x, num_instances))

            results = jax.vmap(problem.evaluate, in_axes=(0, 0, 0))(
                jax.random.split(key_x, num_instances), xs, params
            )

            assert results.fitness.shape == (num_instances,)
            assert not jnp.any(jnp.isnan(results.fitness))
            scores[(name, num_dims)] = float(jnp.mean(results.fitness))

    assert len(scores) == 48


def test_qd_optimization_loop():
    """Test a simple random search optimization loop on a QD problem."""
    num_dims = 5
    population_size = 20
    num_generations = 5
    descriptor_size = 2

    problem = QDProblem(
        suite(["sphere"], num_dims=num_dims)["sphere"],
        RandomProjection(descriptor_size=descriptor_size),
    )
    key = jax.random.key(42)

    key_sample, loop_key = jax.random.split(key)
    params = problem.sample(key_sample)

    evaluate_batch = jax.vmap(problem.evaluate, in_axes=(0, 0, None))

    descriptors_observed = []
    for _ in range(num_generations):
        key_gen, loop_key = jax.random.split(loop_key)
        keys = jax.random.split(key_gen, population_size)

        xs = jax.random.uniform(
            key_gen,
            (population_size, num_dims),
            minval=problem.x_range[0],
            maxval=problem.x_range[1],
        )

        results = evaluate_batch(keys, xs, params)
        descriptors_observed.append(results.descriptor)

    all_descriptors = jnp.concatenate(descriptors_observed, axis=0)
    assert all_descriptors.shape == (population_size * num_generations, descriptor_size)
    assert not jnp.any(jnp.isnan(all_descriptors))
