"""Tests for the BBOBax noise models.

The three official models are checked *by formula* against the vendored
`bbobbenchmarks.py` (`fGauss`, `fUniform`, `fCauchy`), transcribed into numpy
here. Their random draws cannot be shared -- the official code has its own
seeded generator -- so the draws bbobax makes are reproduced from the same key
and fed to the numpy transcription, which pins the arithmetic exactly rather
than only in distribution.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bbobax.functions import Sphere
from bbobax.noise import (
    NOISE_MODELS,
    TARGET_VALUE,
    Additive,
    Cauchy,
    Gaussian,
    Noise,
    Noiseless,
    Uniform,
    stabilize,
)

# `bbobbenchmarks.py`: `tol = 1e-8`, and every noisy model ends with
# `fval += 1.01 * tol` then `fval[ftrue < tol] = ftrue[...]`.
TOL = 1e-8


def official_stabilize(ftrue, fval):
    """Apply the four closing lines shared by fGauss, fUniform and fCauchy."""
    return np.where(ftrue < TOL, ftrue, fval + 1.01 * TOL)


def test_registry_is_keyed_by_each_model_name():
    """NOISE_MODELS holds the five models, keyed by their own name."""
    assert set(NOISE_MODELS) == {
        "noiseless",
        "gaussian",
        "uniform",
        "cauchy",
        "additive",
    }
    for name, model_class in NOISE_MODELS.items():
        assert model_class.name == name


def test_every_model_satisfies_the_protocol():
    """Noise is a contract, not a base class: the models match it structurally."""
    for model_class in NOISE_MODELS.values():
        assert isinstance(model_class(), Noise)


def test_stabilize_matches_the_official_closing_lines():
    """Above the tolerance the offset is added; below it the value is untouched."""
    value = jnp.array([1e-9, 1e-8, 1.0, 100.0])
    noisy = jnp.array([5e-9, 3.0, 3.0, 3.0])

    np.testing.assert_allclose(
        np.asarray(stabilize(value, noisy)),
        official_stabilize(np.asarray(value), np.asarray(noisy)),
        rtol=1e-15,
    )
    assert TARGET_VALUE == TOL


def test_noiseless_returns_the_value_untouched():
    """Official's noise-free functions return ftrue: no offset, no stabilization."""
    model = Noiseless()
    params = model.sample(jax.random.key(0), 5)

    for value in (jnp.array(0.0), jnp.array(1e-12), jnp.array(1e6)):
        for k in range(4):
            assert model.apply(jax.random.key(k), value, params) == value


@pytest.mark.parametrize("value", [1e-12, 1e-9, 1e-6, 1.0, 1e3, 1e7])
def test_gaussian_matches_official_fgauss(value):
    """`ftrue * exp(beta * N(0,1))`, then the official stabilization."""
    model = Gaussian()
    key_sample, key_apply = jax.random.split(jax.random.key(7))
    params = model.sample(key_sample, 5)
    f = jnp.array(value)

    # The draw bbobax makes, reproduced from the same key.
    normal = np.asarray(jax.random.normal(key_apply, shape=f.shape))
    expected = official_stabilize(value, value * np.exp(float(params.beta) * normal))

    np.testing.assert_allclose(
        float(model.apply(key_apply, f, params)), expected, rtol=1e-12
    )


@pytest.mark.parametrize("value", [1e-12, 1e-9, 1e-6, 1.0, 1e3, 1e7])
def test_uniform_matches_official_funiform(value):
    """`U**beta * f * max(1, (1e9/(f+eps))**(alpha*U))`, then stabilization."""
    model = Uniform()
    key_sample, key_apply = jax.random.split(jax.random.key(11))
    params = model.sample(key_sample, 5)
    f = jnp.array(value)

    # Two independent uniform draws, as official; bbobax splits the key.
    key_beta, key_alpha = jax.random.split(key_apply)
    u_beta = float(jax.random.uniform(key_beta, shape=f.shape))
    u_alpha = float(jax.random.uniform(key_alpha, shape=f.shape))

    fval = (
        u_beta ** float(params.beta)
        * value
        * max(1.0, (1e9 / (value + 1e-99)) ** (float(params.alpha) * u_alpha))
    )
    expected = official_stabilize(value, fval)

    np.testing.assert_allclose(
        float(model.apply(key_apply, f, params)), expected, rtol=1e-12
    )


def test_uniform_alpha_carries_the_dimension_term():
    """The paper's alpha is `multiplier * (0.49 + 1/D)`, applied at sampling."""
    key = jax.random.key(3)
    # Same multiplier draw, two dimensions: the ratio is the (0.49 + 1/D) term.
    alpha_5 = float(Uniform().sample(key, 5).alpha)
    alpha_20 = float(Uniform().sample(key, 20).alpha)

    assert alpha_5 / (0.49 + 1 / 5) == pytest.approx(alpha_20 / (0.49 + 1 / 20))

    # Severe is the endpoint of the range at every dimension.
    severe = Uniform(alpha_range=(1.0, 1.0)).sample(key, 5)
    assert float(severe.alpha) == pytest.approx(0.49 + 1 / 5)


@pytest.mark.parametrize("value", [1e-12, 1e-9, 1e-6, 1.0, 1e3, 1e7])
def test_cauchy_matches_official_fcauchy(value):
    """`f + alpha * max(0, 1e3 + 1{U<p} * N/|N|)`, then stabilization."""
    model = Cauchy()
    key_sample, key_apply = jax.random.split(jax.random.key(13))
    params = model.sample(key_sample, 5)
    f = jnp.array(value)

    key_fire, key_num, key_den = jax.random.split(key_apply, 3)
    fires = float(jax.random.uniform(key_fire, shape=f.shape)) < float(params.p)
    # A standard Cauchy is the ratio of two independent normals, not N/|U|.
    num = float(jax.random.normal(key_num, shape=f.shape))
    den = float(jax.random.normal(key_den, shape=f.shape))
    cauchy = num / (abs(den) + 1e-199)

    fval = value + float(params.alpha) * max(0.0, 1e3 + fires * cauchy)
    expected = official_stabilize(value, fval)

    np.testing.assert_allclose(
        float(model.apply(key_apply, f, params)), expected, rtol=1e-12
    )


@pytest.mark.parametrize("model", [Gaussian(), Uniform(), Cauchy()])
def test_official_models_never_block_the_target(model):
    """Below the target precision the undisturbed value comes back, always.

    This is the property the stabilization exists for, and it is part of the
    model in official BBOB rather than an option -- so it holds here with no
    configuration at all.
    """
    params = model.sample(jax.random.key(0), 5)
    tiny = jnp.array(1e-12)

    for k in range(16):
        assert float(model.apply(jax.random.key(k), tiny, params)) == float(tiny)


def test_additive_is_a_bbobax_extension_and_is_not_stabilized():
    """`f + std * N(0,1)`, with no 1.01e-8 floor: it is not a BBOB model."""
    model = Additive(std_range=(0.1, 0.1))
    params = model.sample(jax.random.key(0), 5)
    key = jax.random.key(5)
    f = jnp.array(1e-12)

    normal = float(jax.random.normal(key, shape=f.shape))
    np.testing.assert_allclose(
        float(model.apply(key, f, params)), 1e-12 + 0.1 * normal, rtol=1e-12
    )


@pytest.mark.parametrize("name", sorted(NOISE_MODELS))
def test_every_model_composes_onto_a_problem(name):
    """A problem holds one model; evaluation stays jittable and vmappable."""
    problem = Sphere(num_dims=4, noise=NOISE_MODELS[name]())
    key = jax.random.key(0)
    params = problem.sample(key)

    batch = 8
    keys = jax.random.split(key, batch)
    xs = jax.vmap(problem.sample_x)(keys)
    results = jax.jit(jax.vmap(problem.evaluate, in_axes=(0, 0, None)))(
        keys, xs, params
    )

    assert results.fitness.shape == (batch,)
    assert jnp.all(jnp.isfinite(results.fitness))


@pytest.mark.parametrize("name", ["gaussian", "uniform", "cauchy", "additive"])
def test_noise_actually_disturbs(name):
    """A noisy problem is not deterministic in the evaluation key."""
    problem = Sphere(num_dims=4, noise=NOISE_MODELS[name]())
    params = problem.sample(jax.random.key(0))
    x = problem.sample_x(jax.random.key(1))

    values = {
        float(problem.evaluate(jax.random.key(k), x, params).fitness) for k in range(16)
    }
    assert len(values) > 1
