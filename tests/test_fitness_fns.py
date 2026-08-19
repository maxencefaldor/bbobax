"""Tests for BBOBax fitness functions."""

import jax
import jax.numpy as jnp
import pytest

from bbobax.bbob import BBOB
from bbobax.fitness_fns import (
    BBOB_FNS,
    _lambda_alpha_vector,
    f_pen,
    transform_asy,
    transform_osz,
)


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


@pytest.mark.parametrize("fn_name", BBOB_FNS.keys())
def test_fitness_fn_shapes_and_optimum(fn_name, mock_state, mock_params):
    """Each fitness function returns scalars and bottoms out at x_opt.

    The x_opt conventions are applied by ``BBOB.sample``, so a real task is the
    honest way to build params; the identity-rotation mock is kept for the
    shape checks.
    """
    num_dims = 5
    fn = BBOB_FNS[fn_name]

    # Shapes, on hand-built params with x_opt = 0 and r = q = I.
    val, pen = fn(jnp.zeros(num_dims), mock_state, mock_params(num_dims))
    assert val.shape == ()
    assert pen.shape == ()

    # Optimum, on a real instance: f(x_opt) == 0 before f_opt is added.
    task = BBOB(fn_name, num_dims=num_dims)
    params = task.sample(jax.random.key(0))
    state = task.init(params)
    val, pen = fn(params.x_opt, state, params)
    assert float(val) + float(pen) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("fn_name", BBOB_FNS.keys())
def test_fitness_fn_jit_vmap(fn_name, mock_state, mock_params):
    """Test JAX transformations on fitness functions."""
    fn = BBOB_FNS[fn_name]
    num_dims = 5
    batch_size = 10

    state = mock_state
    params = mock_params(num_dims)

    # JIT test
    jitted_fn = jax.jit(fn)
    x = jnp.ones(num_dims)
    val, pen = jitted_fn(x, state, params)
    assert val.shape == ()

    # VMAP test
    vmapped_fn = jax.vmap(fn, in_axes=(0, None, None))
    x_batch = jnp.ones((batch_size, num_dims))
    val_batch, pen_batch = vmapped_fn(x_batch, state, params)

    assert val_batch.shape == (batch_size,)
    assert pen_batch.shape == (batch_size,)


@pytest.mark.parametrize("num_dims", [2, 5, 10])
def test_task_dimension_is_exact(num_dims):
    """A task at num_dims=D works in exactly D dimensions -- no padding.

    Masking is gone: solutions and every instance array are (D,), and every one
    of the 24 functions still bottoms out at 0 on its true optimum, with
    rotations on. (Replaces the old masking test, which asserted that
    coordinates beyond num_dims were ignored.)
    """
    key = jax.random.key(7)

    for name, fn in BBOB_FNS.items():
        task = BBOB(name, num_dims=num_dims)
        assert task.sample_rotation is True

        params = task.sample(key)
        state = task.init(params)

        assert task.sample_x(key).shape == (num_dims,)
        assert params.x_opt.shape == (num_dims,)
        assert params.r.shape == (num_dims, num_dims)
        assert params.q.shape == (num_dims, num_dims)

        val, pen = fn(params.x_opt, state, params)
        assert float(val) + float(pen) == pytest.approx(0.0, abs=1e-9), (
            f"{name} at D={num_dims}: f(x_opt) = {float(val) + float(pen)!r}"
        )
