"""Tests for the 24 standard BBOB functions."""

import jax
import jax.numpy as jnp
import pytest

from bbobax.functions import (
    BBOB_PROBLEMS,
    DIMENSIONS,
    SchaffersF7,
    SchaffersF7IllConditioned,
    _lambda_alpha_vector,
    f_pen,
    transform_asy,
    transform_osz,
)
from bbobax.problem import BBOBProblem


def test_lambda_alpha_vector():
    """The conditioning vector is 10**(0.5 i/(D-1)) over exactly D coordinates."""
    num_dims, alpha = 5, 10.0

    res = _lambda_alpha_vector(alpha, num_dims)
    assert res.shape == (num_dims,)

    expected = alpha ** (0.5 * jnp.arange(num_dims) / (num_dims - 1))
    assert jnp.allclose(res, expected)

    # The ramp runs from 1 to sqrt(alpha), the BBOB conditioning span.
    assert jnp.isclose(res[0], 1.0)
    assert jnp.isclose(res[-1], jnp.sqrt(alpha))


def test_transform_osz():
    """Test oscillation transformation."""
    x = jnp.array([0.0, 1.0, -1.0, 2.0])
    res = transform_osz(x)
    assert res.shape == x.shape
    # 0 maps to 0
    assert res[0] == 0.0
    # Sign should be preserved
    assert jnp.all(jnp.sign(res) == jnp.sign(x))


def test_transform_asy():
    """Test asymmetry transformation."""
    x = jnp.array([1.0, 2.0, 0.5])
    beta = 0.2
    res = transform_asy(x, beta)
    assert res.shape == x.shape
    # Check positive values are transformed
    assert not jnp.allclose(res, x)

    # The implementation uses jnp.where(x > 0, ..., x), so non-positive
    # coordinates pass through unchanged.
    x_neg = jnp.array([-1.0, -2.0, 0.0])
    res_neg = transform_asy(x_neg, beta)
    assert jnp.allclose(res_neg, x_neg)


def test_f_pen():
    """Test boundary penalty."""
    # Within bounds [-5, 5] -> penalty 0
    x_in = jnp.array([4.0, -4.0, 0.0])
    assert f_pen(x_in) == 0.0

    # Out of bounds -> the squared excess, on every coordinate
    x_out = jnp.array([6.0, 0.0, -8.0])
    assert f_pen(x_out) == pytest.approx(1.0 + 9.0)


def test_registry_is_the_standard_24():
    """BBOB_PROBLEMS holds the 24, keyed by each class's own name."""
    assert len(BBOB_PROBLEMS) == 24

    for name, problem_class in BBOB_PROBLEMS.items():
        assert issubclass(problem_class, BBOBProblem)
        assert problem_class.name == name

    # f1 and f24 pin the canonical order the dict preserves.
    assert list(BBOB_PROBLEMS)[0] == "sphere"
    assert list(BBOB_PROBLEMS)[-1] == "lunacek"


def test_dimensions_are_cocos():
    """DIMENSIONS is COCO's own bbob suite dimension set."""
    assert DIMENSIONS == (2, 3, 5, 10, 20, 40)


@pytest.mark.parametrize("name", BBOB_PROBLEMS.keys())
def test_value_shapes_and_optimum(name, mock_params):
    """Each function returns scalars and bottoms out at x_opt.

    The x_opt constraints are applied by ``sample``, so a real instance is the
    honest way to build params; the identity-rotation mock is kept for the
    shape checks.
    """
    num_dims = 5
    problem = BBOB_PROBLEMS[name](num_dims=num_dims)

    # Shapes, on hand-built params with x_opt = 0 and r = q = I.
    val, pen = problem._value(jnp.zeros(num_dims), mock_params(num_dims))
    assert val.shape == ()
    assert pen.shape == ()

    # Optimum, on a real instance: f(x_opt) == 0 before f_opt is added.
    params = problem.sample(jax.random.key(0))
    val, pen = problem._value(params.x_opt, params)
    assert float(val) + float(pen) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("name", BBOB_PROBLEMS.keys())
def test_value_jit_vmap(name, mock_params):
    """Test JAX transformations on the raw function."""
    num_dims = 5
    batch_size = 10
    problem = BBOB_PROBLEMS[name](num_dims=num_dims)
    params = mock_params(num_dims)

    # JIT test
    val, pen = jax.jit(problem._value)(jnp.ones(num_dims), params)
    assert val.shape == ()

    # VMAP test
    vmapped = jax.vmap(problem._value, in_axes=(0, None))
    val_batch, pen_batch = vmapped(jnp.ones((batch_size, num_dims)), params)

    assert val_batch.shape == (batch_size,)
    assert pen_batch.shape == (batch_size,)


@pytest.mark.parametrize("num_dims", [2, 5, 10])
def test_problem_dimension_is_exact(num_dims):
    """A problem at num_dims=D works in exactly D dimensions -- no padding.

    Masking is gone: solutions and every instance array are (D,), and every one
    of the 24 functions still bottoms out at 0 on its true optimum, with
    rotations on. (Replaces the old masking test, which asserted that
    coordinates beyond num_dims were ignored.)
    """
    key = jax.random.key(7)

    for name, problem_class in BBOB_PROBLEMS.items():
        problem = problem_class(num_dims=num_dims)
        assert problem.sample_rotation is True

        params = problem.sample(key)

        assert problem.sample_x(key).shape == (num_dims,)
        assert params.x_opt.shape == (num_dims,)
        assert params.r.shape == (num_dims, num_dims)
        assert params.q.shape == (num_dims, num_dims)

        val, pen = problem._value(params.x_opt, params)
        assert float(val) + float(pen) == pytest.approx(0.0, abs=1e-9), (
            f"{name} at D={num_dims}: f(x_opt) = {float(val) + float(pen)!r}"
        )


def test_ill_conditioned_schaffers_is_the_conditioning_alone():
    """f18 is f17 with the conditioning raised, and nothing else."""
    assert issubclass(SchaffersF7IllConditioned, SchaffersF7)
    assert SchaffersF7.conditioning == 10.0
    assert SchaffersF7IllConditioned.conditioning == 1000.0

    # The math is written once: the subclass adds no _value of its own.
    assert "_value" not in vars(SchaffersF7IllConditioned)

    # And it is genuinely a different landscape.
    num_dims = 5
    key = jax.random.key(0)
    f17 = SchaffersF7(num_dims=num_dims)
    f18 = SchaffersF7IllConditioned(num_dims=num_dims)
    params = f17.sample(key)
    x = f17.sample_x(jax.random.key(1))
    assert float(f17._value(x, params)[0]) != float(f18._value(x, params)[0])


def test_place_x_opt_is_the_default_unless_overridden():
    """Exactly the six constrained functions override where the optimum may sit."""
    overriding = {
        name
        for name, problem_class in BBOB_PROBLEMS.items()
        if "_place_x_opt" in vars(problem_class)
    }
    assert overriding == {
        "bueche_rastrigin",
        "linear_slope",
        "rosenbrock",
        "schwefel",
        "gallagher_21_hi",
        "lunacek",
    }
