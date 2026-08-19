"""Tests for the Quality-Diversity extension."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bbobax.functions import BBOB_PROBLEMS, Rastrigin, Sphere
from bbobax.qd import Descriptor, QDProblem, RandomProjection
from bbobax.types import QDEval, QDParams


def test_random_projection_is_a_descriptor():
    """RandomProjection satisfies the Descriptor protocol structurally."""
    assert isinstance(RandomProjection(descriptor_size=2), Descriptor)


@pytest.mark.parametrize("descriptor_size", [1, 2, 5])
def test_random_projection_sample_and_evaluate(descriptor_size):
    """The projection is drawn per instance and applied to one solution."""
    num_dims = 10
    descriptor = RandomProjection(descriptor_size=descriptor_size)

    projection = descriptor.sample(jax.random.key(0), num_dims)
    assert projection.shape == (descriptor_size, num_dims)

    x = jnp.ones(num_dims)
    result = descriptor.evaluate(x, projection)
    assert result.shape == (descriptor_size,)
    np.testing.assert_allclose(
        np.asarray(result), np.asarray(projection) @ np.ones(num_dims), rtol=1e-12
    )


def test_random_projection_jit_vmap():
    """Test JAX transformations on the descriptor."""
    num_dims, descriptor_size, batch_size = 10, 3, 8
    descriptor = RandomProjection(descriptor_size=descriptor_size)
    projection = descriptor.sample(jax.random.key(0), num_dims)

    result = jax.jit(descriptor.evaluate)(jnp.ones(num_dims), projection)
    assert result.shape == (descriptor_size,)

    batch = jax.vmap(descriptor.evaluate, in_axes=(0, None))(
        jnp.ones((batch_size, num_dims)), projection
    )
    assert batch.shape == (batch_size, descriptor_size)


def test_qd_problem_workflow():
    """The QD contract: sample -> evaluate, fitness and descriptor together."""
    num_dims, descriptor_size = 5, 2
    problem = QDProblem(
        Sphere(num_dims=num_dims), RandomProjection(descriptor_size=descriptor_size)
    )
    key = jax.random.key(0)

    params = problem.sample(key)
    assert isinstance(params, QDParams)
    assert params.descriptor_params.shape == (descriptor_size, num_dims)
    # Composed, not inherited: the problem's own instance is nested whole.
    assert params.problem_params.x_opt.shape == (num_dims,)

    x = problem.sample_x(key)
    evaluation = problem.evaluate(key, x, params)

    assert isinstance(evaluation, QDEval)
    assert evaluation.fitness.shape == ()
    assert evaluation.descriptor.shape == (descriptor_size,)


def test_qd_problem_delegates_what_it_wraps():
    """A QD problem answers for the function it wraps."""
    problem = QDProblem(Rastrigin(num_dims=7), RandomProjection(descriptor_size=3))

    assert problem.name == "rastrigin"
    assert problem.num_dims == 7
    assert problem.x_range == (-5.0, 5.0)
    assert problem.descriptor_size == 3


def test_qd_fitness_is_the_underlying_fitness():
    """Pairing a descriptor onto a function does not change the function."""
    num_dims = 6
    base = Rastrigin(num_dims=num_dims)
    problem = QDProblem(base, RandomProjection(descriptor_size=2))

    key = jax.random.key(0)
    params = problem.sample(key)
    x = problem.sample_x(jax.random.key(1))

    qd_evaluation = problem.evaluate(jax.random.key(2), x, params)
    base_evaluation = base.evaluate(jax.random.key(2), x, params.problem_params)

    assert float(qd_evaluation.fitness) == float(base_evaluation.fitness)
    # And the descriptor is exactly the projection of x.
    np.testing.assert_allclose(
        np.asarray(qd_evaluation.descriptor),
        np.asarray(params.descriptor_params) @ np.asarray(x),
        rtol=1e-12,
    )


def test_any_function_pairs_with_the_descriptor():
    """Composition is why this is not 24 subclasses: all 24 pair the same way."""
    num_dims, descriptor_size = 4, 2
    descriptor = RandomProjection(descriptor_size=descriptor_size)
    key = jax.random.key(3)

    for name, problem_class in BBOB_PROBLEMS.items():
        problem = QDProblem(problem_class(num_dims=num_dims), descriptor)
        params = problem.sample(key)
        evaluation = problem.evaluate(key, problem.sample_x(key), params)

        assert evaluation.descriptor.shape == (descriptor_size,), name
        assert not jnp.isnan(evaluation.fitness), name


def test_qd_problem_jit_vmap():
    """Test JAX transformations on QD evaluation."""
    num_dims, descriptor_size, batch_size = 5, 2, 8
    problem = QDProblem(
        Sphere(num_dims=num_dims), RandomProjection(descriptor_size=descriptor_size)
    )
    key = jax.random.key(0)
    params = problem.sample(key)

    keys = jax.random.split(key, batch_size)
    xs = jax.vmap(problem.sample_x)(keys)
    results = jax.jit(jax.vmap(problem.evaluate, in_axes=(0, 0, None)))(
        keys, xs, params
    )

    assert results.fitness.shape == (batch_size,)
    assert results.descriptor.shape == (batch_size, descriptor_size)
