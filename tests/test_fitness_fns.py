"""Tests for BBOBax fitness functions."""

import jax
import jax.numpy as jnp
import pytest

from bbobax.fitness_fns import (
    bbob_fns,
    f_pen,
    lambda_alpha,
    transform_asy,
    transform_osz,
)


def test_lambda_alpha():
    """Test lambda_alpha helper function."""
    max_dims = 5
    num_dims = 3
    alpha = 10.0

    # Test output shape
    res = lambda_alpha(alpha, max_dims, num_dims)
    assert res.shape == (max_dims, max_dims)

    # Test diagonal property
    assert jnp.allclose(res, jnp.diag(jnp.diag(res)))

    # Test masking (elements beyond num_dims should be alpha^0.5 = 3.162...)
    # Wait, the implementation sets exp to 0.5 * mask, so masked elements
    # get exp=0 -> 1.0?
    # Let's check the code: exp = ... * mask. If mask is False (0), exp is 0.
    # alpha^0 = 1.
    # The code says: exp = (jnp.where(...) * mask).
    # So for masked dimensions, exp should be 0, so value should be 1.

    diag = jnp.diag(res)
    assert jnp.all(diag[num_dims:] == 1.0)


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
    num_dims = 3
    res = transform_asy(x, beta, num_dims)
    assert res.shape == x.shape
    # Check positive values are transformed
    assert not jnp.allclose(res, x)

    # Negative values should be identity (or handle gracefully depending on
    # implementation)
    # The implementation uses jnp.where(x > 0, ..., x), so negative should be identity
    x_neg = jnp.array([-1.0, -2.0])
    res_neg = transform_asy(x_neg, beta, num_dims)
    assert jnp.allclose(res_neg, x_neg)


def test_f_pen():
    """Test boundary penalty."""
    num_dims = 3

    # Within bounds [-5, 5] -> penalty 0
    x_in = jnp.array([4.0, -4.0, 0.0, 10.0, 10.0])  # Last two are masked out
    assert f_pen(x_in, num_dims) == 0.0

    # Out of bounds -> penalty > 0
    x_out = jnp.array([6.0, 0.0, 0.0, 0.0, 0.0])
    assert f_pen(x_out, num_dims) > 0.0


@pytest.mark.parametrize("fn_name", bbob_fns.keys())
def test_fitness_fn_shapes_and_optimum(fn_name, mock_state, mock_params):
    """Test each fitness function for shapes and optimum value."""
    fn = bbob_fns[fn_name]
    max_dims = 5
    num_dims = 3

    state = mock_state(max_dims)
    params = mock_params(num_dims, max_dims)

    # Test at optimum (x_opt = 0 for mock params)
    x = jnp.zeros(max_dims)
    val, pen = fn(x, state, params)

    # Output should be scalar arrays
    assert val.shape == ()
    assert pen.shape == ()

    # At optimum, value should be close to 0
    # (for most functions defined as f(x) = ... with min at 0)
    # Note: Some functions might have offsets or complex optima, but with x_opt=0
    # and standard setup they are designed to be minimized at x_opt with value 0.
    # Some functions like linear_slope might be different, but generally BBOB fns
    # are 0 at optimum.
    # We allow small tolerance.
    if fn_name not in [
        "linear_slope",
        "schwefel",
        "lunacek",
        "griewank_rosenbrock",
    ]:
        assert jnp.abs(val) < 1e-5

    if fn_name not in [
        "schwefel",
    ]:
        assert pen == 0.0


@pytest.mark.parametrize("fn_name", bbob_fns.keys())
def test_fitness_fn_jit_vmap(fn_name, mock_state, mock_params):
    """Test JAX transformations on fitness functions."""
    fn = bbob_fns[fn_name]
    max_dims = 5
    num_dims = 3
    batch_size = 10

    state = mock_state(max_dims)
    params = mock_params(num_dims, max_dims)

    # JIT test
    jitted_fn = jax.jit(fn)
    x = jnp.ones(max_dims)
    val, pen = jitted_fn(x, state, params)
    assert val.shape == ()

    # VMAP test
    vmapped_fn = jax.vmap(fn, in_axes=(0, None, None))
    x_batch = jnp.ones((batch_size, max_dims))
    val_batch, pen_batch = vmapped_fn(x_batch, state, params)

    assert val_batch.shape == (batch_size,)
    assert pen_batch.shape == (batch_size,)


def test_masking_behavior(mock_state, mock_params):
    """Test that dimensions beyond num_dims do not affect the result."""
    # We pick Sphere as a simple example
    fn = bbob_fns["sphere"]
    max_dims = 5
    num_dims = 2

    state = mock_state(max_dims)
    params = mock_params(num_dims, max_dims)

    # Base input
    x1 = jnp.array([1.0, 1.0, 0.0, 0.0, 0.0])
    # Modified in masked dimensions
    x2 = jnp.array([1.0, 1.0, 5.0, -5.0, 10.0])

    val1, _ = fn(x1, state, params)
    val2, _ = fn(x2, state, params)

    assert jnp.allclose(val1, val2)
