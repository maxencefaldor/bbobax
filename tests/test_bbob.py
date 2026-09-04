"""Tests for the bbob suite, the 24 standard noiseless functions."""

import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bbobax.bbob import (
    BBOB_PROBLEMS,
    DIMENSIONS,
    SchaffersF7,
    SchaffersF7IllConditioned,
)
from bbobax.problem import BBOBProblem


def test_dimensions_are_cocos():
    """DIMENSIONS is the bbob suite's own dimension set, from `suite_bbob.c`."""
    assert DIMENSIONS == (2, 3, 5, 10, 20, 40)


def test_registry_is_the_standard_24():
    """BBOB_PROBLEMS holds the 24, keyed by each class's own name."""
    assert len(BBOB_PROBLEMS) == 24

    for name, problem_class in BBOB_PROBLEMS.items():
        assert issubclass(problem_class, BBOBProblem)
        assert problem_class.name == name

    # f1 and f24 pin the canonical order the dict preserves.
    assert next(iter(BBOB_PROBLEMS)) == "sphere"
    assert list(BBOB_PROBLEMS)[-1] == "lunacek"


@pytest.mark.parametrize("name", BBOB_PROBLEMS.keys())
def test_value_shapes_and_optimum(name, mock_params):
    """Each function returns scalars and bottoms out at x_opt.

    The x_opt constraints are applied by `sample`, so a real instance is the
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
    val, _pen = jax.jit(problem._value)(jnp.ones(num_dims), params)
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
    assert SchaffersF7.condition == 10.0
    assert SchaffersF7IllConditioned.condition == 1000.0

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


def test_sample_x_opt_is_the_default_unless_overridden():
    """Exactly the six constrained functions draw their own optimum."""
    overriding = {
        name
        for name, problem_class in BBOB_PROBLEMS.items()
        if "_sample_x_opt" in vars(problem_class)
    }
    assert overriding == {
        "bueche_rastrigin",
        "linear_slope",
        "rosenbrock",
        "schwefel",
        "gallagher_21_hi",
        "lunacek",
    }


@pytest.mark.parametrize("name", BBOB_PROBLEMS.keys())
def test_sample_x_opt_lands_where_the_definition_allows(name):
    """The drawn optimum is in the set its function admits, at every dimension."""
    num_dims = 5
    problem = BBOB_PROBLEMS[name](num_dims=num_dims)
    keys = jax.random.split(jax.random.key(0), 32)
    x_opt = np.asarray(jax.vmap(problem._sample_x_opt)(keys))

    assert x_opt.shape == (32, num_dims)

    # The three sign-only functions draw a lattice, not a continuous point:
    # only the sign of each coordinate is an instance choice, so that is all
    # `_sample_x_opt` draws -- and both signs do occur.
    lattice = {"linear_slope": 5.0, "schwefel": 4.2096874633 / 2.0, "lunacek": 1.25}
    if name in lattice:
        assert set(np.unique(np.abs(x_opt))) == {lattice[name]}
        assert np.any(x_opt > 0) and np.any(x_opt < 0)
        return

    # Everything else draws continuously, inside the range its definition allows.
    bound = {"rosenbrock": 3.0, "gallagher_21_hi": 4.0 * 0.98}.get(name, 4.0)
    assert np.all(np.abs(x_opt) <= bound + 1e-12)
    assert len(np.unique(x_opt)) > 1

    # Bueche-Rastrigin forces its skewed (0-based even) coordinates non-negative.
    if name == "bueche_rastrigin":
        assert np.all(x_opt[:, ::2] >= 0.0)
        assert np.any(x_opt[:, 1::2] < 0.0)


def _stress_points(rng, num_dims, x_opt):
    """Points that stress every branch a function has.

    Inside the box, outside it (the boundary penalty), the corners (where the
    penalty and the linear-slope plateau are extremal), a ring around the
    optimum, and the optimum itself.
    """
    return np.vstack(
        [
            rng.uniform(-5.0, 5.0, size=(24, num_dims)),
            rng.uniform(-8.0, 8.0, size=(8, num_dims)),
            np.full((1, num_dims), 5.0),
            np.full((1, num_dims), -5.0),
            np.zeros((1, num_dims)),
            x_opt + rng.normal(scale=1e-8, size=(4, num_dims)),
            x_opt[None, :],
        ]
    )


@pytest.mark.parametrize("name", BBOB_PROBLEMS.keys())
def test_no_function_is_ever_non_finite(name):
    """Every function is finite everywhere it can be asked, at every dimension.

    Across all of COCO's `DIMENSIONS`, not a convenient few: the dimension
    enters `katsuura` as `10 / D**1.2`, `lunacek` as `sqrt(D + 20) - 4.1`, and
    `weierstrass` and `schwefel` divide by it, so D = 40 exercises expressions
    that D = 5 never reaches. Checked through `evaluate`, so the boundary
    penalty and `f_opt` are on the path too.
    """
    for num_dims in DIMENSIONS:
        problem = BBOB_PROBLEMS[name](num_dims=num_dims)
        params = problem.sample(jax.random.key(num_dims))

        rng = np.random.default_rng(1000 + num_dims)
        xs = jnp.asarray(_stress_points(rng, num_dims, np.asarray(params.x_opt)))
        keys = jax.random.split(jax.random.key(1), xs.shape[0])

        fitness = np.asarray(
            jax.vmap(problem.evaluate, in_axes=(0, 0, None))(keys, xs, params).fitness
        )

        assert np.all(np.isfinite(fitness)), (
            f"{name} at D={num_dims}: "
            f"{int((~np.isfinite(fitness)).sum())} non-finite of {fitness.size}"
        )
        # BBOB minimizes to f_opt, pinned at 0 by default: nothing may go below.
        assert np.all(fitness >= -1e-9), f"{name} at D={num_dims}: below the optimum"


@pytest.mark.parametrize("name", BBOB_PROBLEMS.keys())
def test_every_function_reaches_its_optimum_at_every_dimension(name):
    """f(x_opt) == 0 across all of COCO's dimensions, not just the small ones."""
    for num_dims in DIMENSIONS:
        problem = BBOB_PROBLEMS[name](num_dims=num_dims)
        params = problem.sample(jax.random.key(num_dims))

        value, penalty = problem._value(params.x_opt, params)
        assert float(value) + float(penalty) == pytest.approx(0.0, abs=1e-9), (
            f"{name} at D={num_dims}: f(x_opt) = {float(value) + float(penalty)!r}"
        )


# The suite runs in float64 (conftest), which is what the alignment tests need
# -- but float32 is JAX's default and therefore what bbobax actually runs in
# unless a user opts out. That cannot be checked in-process once x64 is on, so
# it is checked in a fresh interpreter.
_FLOAT32_SWEEP = """
import jax, jax.numpy as jnp, numpy as np
import bbobax

assert jnp.zeros(1).dtype == jnp.float32, "expected float32 by default"

bad = []
for name, problem_class in bbobax.BBOB_PROBLEMS.items():
    for num_dims in bbobax.DIMENSIONS:
        problem = problem_class(num_dims=num_dims)
        params = problem.sample(jax.random.key(num_dims))
        rng = np.random.default_rng(num_dims)
        xs = jnp.asarray(np.vstack([
            rng.uniform(-5.0, 5.0, (24, num_dims)),
            rng.uniform(-8.0, 8.0, (8, num_dims)),
            np.asarray(params.x_opt)[None, :],
        ]))
        keys = jax.random.split(jax.random.key(1), xs.shape[0])
        fitness = np.asarray(
            jax.vmap(problem.evaluate, in_axes=(0, 0, None))(keys, xs, params).fitness
        )
        if not np.all(np.isfinite(fitness)):
            bad.append((name, num_dims))
print("BAD:" + repr(bad))
"""


def test_no_function_overflows_in_float32():
    """Nothing overflows at the library's default precision, at any dimension.

    float32 is where this could plausibly break: `katsuura` multiplies D terms
    together before taking a fractional power, and `ellipsoidal` carries a 1e6
    conditioning, both of which grow with the dimension.
    """
    result = subprocess.run(
        [sys.executable, "-c", _FLOAT32_SWEEP],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "BAD:[]" in result.stdout, result.stdout
