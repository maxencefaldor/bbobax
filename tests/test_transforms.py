"""Tests for the transformations BBOB builds its landscapes from."""

import jax.numpy as jnp
import pytest

from bbobax.transforms import f_pen, lambda_alpha, transform_asy, transform_osz


def test_lambda_alpha():
    """The conditioning vector is 10**(0.5 i/(D-1)) over exactly D coordinates."""
    num_dims, alpha = 5, 10.0

    res = lambda_alpha(alpha, num_dims)
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
