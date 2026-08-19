"""Tests for the BBOB problem contract."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bbobax.bbob import BBOB_PROBLEMS, Rastrigin, Sphere, bbob_suite
from bbobax.noise import Noiseless
from bbobax.problem import BBOBEval, BBOBParams, BBOBProblem


def test_problem_initialization():
    """One problem = one function at one fixed dimension."""
    # The class selects the function; the default dimension is BBOB's usual 10.
    problem = Sphere()
    assert problem.name == "sphere"
    assert problem.num_dims == 10

    # num_dims is a single int, fixed for the problem.
    assert Rastrigin(num_dims=5).num_dims == 5


def test_base_class_supplies_no_function():
    """BBOBProblem is the contract, not a problem: it defines no `_value`."""
    problem = BBOBProblem(num_dims=5)
    params = problem.sample(jax.random.key(0))

    with pytest.raises(NotImplementedError, match="BBOBProblem"):
        problem.evaluate(jax.random.key(1), problem.sample_x(jax.random.key(2)), params)


def test_problem_initialization_rejects_bad_dimension():
    """num_dims below 2 -> ValueError; 2 is the smallest BBOB is defined for."""
    with pytest.raises(ValueError, match="num_dims"):
        Sphere(num_dims=1)

    assert Sphere(num_dims=2).num_dims == 2


def test_problem_workflow():
    """The whole contract: sample -> evaluate. There is no state."""
    num_dims = 5
    problem = Sphere(num_dims=num_dims)
    key_sample, key_eval = jax.random.split(jax.random.key(0))

    # Sample an instance: every array is exactly num_dims long.
    params = problem.sample(key_sample)
    assert isinstance(params, BBOBParams)
    assert params.x_opt.shape == (num_dims,)
    assert params.f_opt.shape == ()
    # Rotations are instance parameters: drawn once by sample().
    assert params.r.shape == (num_dims, num_dims)
    assert params.q.shape == (num_dims, num_dims)

    # Sample solution
    x = problem.sample_x(key_eval)
    assert x.shape == (num_dims,)

    # Evaluate: one call, one result, nothing to thread through.
    evaluation = problem.evaluate(key_eval, x, params)
    assert isinstance(evaluation, BBOBEval)
    assert evaluation.fitness.shape == ()


def test_evaluation_is_memoryless():
    """The value at x does not depend on how many times x has been asked.

    This is why there is no evaluation state: repeating a call returns the
    identical value, for every one of the 24.
    """
    for name in BBOB_PROBLEMS:
        problem = BBOB_PROBLEMS[name](num_dims=4)
        params = problem.sample(jax.random.key(0))
        x = problem.sample_x(jax.random.key(1))

        values = [
            float(problem.evaluate(jax.random.key(2), x, params).fitness)
            for _ in range(3)
        ]
        assert len(set(values)) == 1, name


def test_problem_jit_vmap():
    """Test JAX transformations on evaluation."""
    num_dims = 5
    problem = Rastrigin(num_dims=num_dims)
    key = jax.random.key(0)
    params = problem.sample(key)

    x = problem.sample_x(key)
    jax.jit(problem.evaluate)(key, x, params)

    # VMAP evaluate (batch of solutions), keeping params fixed.
    batch_size = 10
    keys = jax.random.split(key, batch_size)
    xs = jax.random.uniform(key, (batch_size, num_dims))

    results = jax.vmap(problem.evaluate, in_axes=(0, 0, None))(keys, xs, params)

    assert results.fitness.shape == (batch_size,)


def test_suite_builds_the_standard_24():
    """bbob_suite() is the 24 standard functions as 24 separate problems."""
    problems = bbob_suite(num_dims=6)

    assert list(problems) == list(BBOB_PROBLEMS)
    assert len(problems) == 24

    # 24 distinct objects, each of its own class.
    assert len({id(problem) for problem in problems.values()}) == 24
    for name, problem in problems.items():
        assert problem.name == name
        assert problem.num_dims == 6
        # The point of the design: the problem *is* the function, so evaluation
        # calls it directly -- no lax.switch over 24 branches, and nothing pays
        # for the 23 branches it does not want.
        assert type(problem) is BBOB_PROBLEMS[name]

    # A subset is selectable, and kwargs reach every problem.
    subset = bbob_suite(["sphere", "discus"], num_dims=3, clip_x=True)
    assert list(subset) == ["sphere", "discus"]
    assert all(p.num_dims == 3 and p.clip_x for p in subset.values())


def test_suite_rejects_unknown_names():
    """An unknown function is named in the error, with the available list."""
    with pytest.raises(KeyError, match="not_a_bbob_function"):
        bbob_suite(["sphere", "not_a_bbob_function"])


def test_suite_evaluation_has_no_dispatch():
    """Each suite problem evaluates through its own function, at its own optimum."""
    problems = bbob_suite(num_dims=4)
    key = jax.random.key(11)

    for name, problem in problems.items():
        params = problem.sample(key)
        evaluation = problem.evaluate(key, params.x_opt, params)
        # Noiseless default and f_opt pinned to 0: the optimum evaluates to 0.
        assert float(evaluation.fitness) == pytest.approx(0.0, abs=1e-9), name


@pytest.mark.parametrize("num_dims", [2, 5, 10])
def test_generate_random_rotation(num_dims):
    """The static rotation generator returns a matrix in SO(n)."""
    rotation = np.asarray(
        BBOBProblem.generate_random_rotation(jax.random.key(3), num_dims)
    )

    assert rotation.shape == (num_dims, num_dims)
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(num_dims), atol=1e-10)
    np.testing.assert_allclose(np.linalg.det(rotation), 1.0, atol=1e-10)

    # Deterministic in the key, and a different key gives a different rotation.
    again = np.asarray(
        BBOBProblem.generate_random_rotation(jax.random.key(3), num_dims)
    )
    np.testing.assert_array_equal(rotation, again)
    other = np.asarray(
        BBOBProblem.generate_random_rotation(jax.random.key(4), num_dims)
    )
    assert not np.allclose(rotation, other)


def test_sample_rotation_off_gives_identity():
    """With sample_rotation off, R = Q = I -- deliberately, not by default."""
    problem = BBOB_PROBLEMS["ellipsoidal_rotated"](num_dims=5, sample_rotation=False)
    params = problem.sample(jax.random.key(0))

    np.testing.assert_array_equal(np.asarray(params.r), np.eye(5))
    np.testing.assert_array_equal(np.asarray(params.q), np.eye(5))


def test_default_noise_is_noiseless():
    """The default problem is COCO's noiseless suite: evaluation is deterministic."""
    problem = Sphere(num_dims=5)

    # The default model is Noiseless, held directly -- no pool, no switch.
    assert isinstance(problem.noise_model, Noiseless)

    params = problem.sample(jax.random.key(0))
    x = problem.sample_x(jax.random.key(1))

    # Unbiased: the same solution evaluates identically under any eval key, and
    # equals the raw function value plus penalty plus f_opt.
    values = [
        float(problem.evaluate(jax.random.key(k), x, params).fitness) for k in range(8)
    ]
    assert len(set(values)) == 1

    val, pen = problem._value(x, params)
    assert values[0] == pytest.approx(float(val) + float(pen) + float(params.f_opt))


def test_clip_x_clips():
    """clip_x pulls solutions back into the box before the function sees them."""
    problem = Sphere(num_dims=3, clip_x=True)
    params = problem.sample(jax.random.key(0))

    outside = jnp.array([9.0, -9.0, 0.0])
    clipped = jnp.clip(outside, -5.0, 5.0)

    assert float(problem.evaluate(jax.random.key(1), outside, params).fitness) == (
        pytest.approx(
            float(problem.evaluate(jax.random.key(1), clipped, params).fitness)
        )
    )
