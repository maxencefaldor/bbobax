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

from bbobax.functions import DIMENSIONS, Sphere
from bbobax.noise import (
    NOISE_MODELS,
    TARGET_PRECISION,
    Cauchy,
    Gaussian,
    Mixture,
    Noiseless,
    NoiseModel,
    Uniform,
    _epsilon,
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
    }
    for name, model_class in NOISE_MODELS.items():
        assert model_class.name == name


def test_every_model_satisfies_the_protocol():
    """NoiseModel is a contract, not a base class: models match it structurally."""
    for model_class in NOISE_MODELS.values():
        assert isinstance(model_class(), NoiseModel)


def test_stabilize_matches_the_official_closing_lines():
    """Above the tolerance the offset is added; below it the value is untouched."""
    value = jnp.array([1e-9, 1e-8, 1.0, 100.0])
    noisy = jnp.array([5e-9, 3.0, 3.0, 3.0])

    np.testing.assert_allclose(
        np.asarray(stabilize(value, noisy)),
        official_stabilize(np.asarray(value), np.asarray(noisy)),
        rtol=1e-15,
    )
    assert TARGET_PRECISION == TOL


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


@pytest.mark.parametrize("name", sorted(NOISE_MODELS))
def test_every_model_composes_onto_a_problem(name):
    """A problem holds one model; evaluation stays jittable and vmappable."""
    problem = Sphere(num_dims=4, noise_model=NOISE_MODELS[name]())
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


@pytest.mark.parametrize("name", ["gaussian", "uniform", "cauchy"])
def test_noise_actually_disturbs(name):
    """A noisy problem is not deterministic in the evaluation key."""
    problem = Sphere(num_dims=4, noise_model=NOISE_MODELS[name]())
    params = problem.sample(jax.random.key(0))
    x = problem.sample_x(jax.random.key(1))

    values = {
        float(problem.evaluate(jax.random.key(k), x, params).fitness) for k in range(16)
    }
    assert len(values) > 1


def test_mixture_is_absent_from_the_registry():
    """`Mixture` is a combinator over models, not a model with a bare form."""
    assert "mixture" not in NOISE_MODELS
    assert isinstance(Mixture(Gaussian()), NoiseModel)

    with pytest.raises(ValueError, match="at least one"):
        Mixture()


def test_mixture_draws_a_model_per_instance():
    """The one thing a held model cannot do: instances that disagree on family.

    This is the meta-learning shape -- a batch of instances of one function
    carrying different noise families -- and the reason `Mixture` exists.
    """
    models = (Gaussian(), Uniform(), Cauchy())
    problem = Sphere(num_dims=4, noise_model=Mixture(*models))

    instances = 64
    keys = jax.random.split(jax.random.key(0), instances)
    params = jax.vmap(problem.sample)(keys)

    # Every family is represented across the batch, in one vmapped sample.
    drawn = set(np.asarray(params.noise_model.model_id).tolist())
    assert drawn == {0, 1, 2}

    # And the batch evaluates in one vmapped call.
    xs = jax.vmap(problem.sample_x)(keys)
    results = jax.jit(jax.vmap(problem.evaluate, in_axes=(0, 0, 0)))(keys, xs, params)
    assert results.fitness.shape == (instances,)
    assert jnp.all(jnp.isfinite(results.fitness))


def test_mixture_agrees_with_the_model_it_selected():
    """A mixture applies exactly the model the instance drew, unchanged."""
    models = (Gaussian(), Uniform(), Cauchy())
    mixture = Mixture(*models)

    for k in range(8):
        params = mixture.sample(jax.random.key(k), 5)
        chosen = models[int(params.model_id)]
        key, value = jax.random.key(100 + k), jnp.array(3.0)

        assert float(mixture.apply(key, value, params)) == pytest.approx(
            float(chosen.apply(key, value, params.models[int(params.model_id)]))
        )


# The values a noise model can actually be handed: BBOB functions bottom out at
# 0, so exactly-zero is reachable (at the optimum) and is the divide-by-zero
# case for `Uniform`; the large end is what `bent_cigar` or `ellipsoidal`
# produce far from their optimum at D = 40.
_EXTREME_VALUES = [0.0, 1e-300, 1e-30, 1e-12, 1e-8, 1.0, 1e6, 1e15, 1e30]


def _apply_many(model: NoiseModel, value: jax.Array, params) -> np.ndarray:
    """Apply `model` to one value under 64 keys.

    Takes the model as a `NoiseModel` rather than reading it off `NOISE_MODELS`
    inline: the registry's value type is a union of the concrete classes, and
    nothing correlates each one's `sample` output with its own `apply`. The
    protocol is what a caller actually holds, and is where `params` is open.
    """
    return np.asarray(
        jax.vmap(model.apply, in_axes=(0, None, None))(
            jax.random.split(jax.random.key(2), 64), value, params
        )
    )


@pytest.mark.parametrize("name", sorted(NOISE_MODELS))
def test_no_model_divides_by_zero_or_returns_nan(name):
    """No model produces a NaN or an infinity, including at f = 0 exactly.

    `Uniform` divides by the value and `Cauchy` by a normal draw, so both need
    their epsilon to be a real number at the working dtype -- which the paper's
    1e-99 and 1e-199 are not in float32.
    """
    model = NOISE_MODELS[name]()

    for num_dims in (2, 40):
        params = model.sample(jax.random.key(num_dims), num_dims)
        for value in _EXTREME_VALUES:
            noisy = _apply_many(model, jnp.array(value), params)
            assert np.all(np.isfinite(noisy)), (
                f"{name} at D={num_dims}, f={value}: "
                f"{int((~np.isfinite(noisy)).sum())} non-finite of {noisy.size}"
            )


@pytest.mark.parametrize("name", sorted(NOISE_MODELS))
def test_every_model_stays_finite_on_a_real_problem(name):
    """The same, through a problem, across every standard dimension."""
    for num_dims in DIMENSIONS:
        problem = Sphere(num_dims=num_dims, noise_model=NOISE_MODELS[name]())
        params = problem.sample(jax.random.key(num_dims))

        keys = jax.random.split(jax.random.key(3), 64)
        xs = jax.vmap(problem.sample_x)(keys)
        # The optimum is in the batch, so f = 0 reaches the model.
        xs = xs.at[0].set(params.x_opt)

        fitness = jax.vmap(problem.evaluate, in_axes=(0, 0, None))(keys, xs, params)
        assert jnp.all(jnp.isfinite(fitness.fitness)), f"{name} at D={num_dims}"


def test_the_epsilon_is_a_real_number_at_the_working_dtype():
    """The paper's literals round to zero in float32; the guard must not."""
    assert np.float32(1e-99) == 0.0
    assert np.float32(1e-199) == 0.0

    assert _epsilon(jnp.zeros((), dtype=jnp.float32)) > 0.0
    assert _epsilon(jnp.zeros((), dtype=jnp.float64)) > 0.0
    # Small enough that it cannot perturb any value the guard sits behind.
    assert _epsilon(jnp.zeros((), dtype=jnp.float64)) < 1e-300


@pytest.mark.parametrize(
    ("model_class", "moderate", "severe"),
    [
        (Gaussian, {"beta": 0.01}, {"beta": 1.0}),
        (Uniform, {"beta": 0.01}, {"beta": 1.0}),
        (Cauchy, {"alpha": 0.01, "p": 0.05}, {"alpha": 1.0, "p": 0.2}),
    ],
    ids=["gaussian", "uniform", "cauchy"],
)
def test_the_papers_two_severities_are_reachable_exactly(model_class, moderate, severe):
    """Continuous severity is bbobax's deviation; the paper's points still pin.

    Nothing here is comparable to a published f101-f130 result unless the
    severity is pinned, so the two official points are constructors rather than
    something a caller has to spell as a degenerate range.
    """
    for constructor, expected in (
        (model_class.moderate, moderate),
        (model_class.severe, severe),
    ):
        model = constructor()
        # Pinned means pinned: every key gives the same settings.
        for key_seed in range(4):
            params = model.sample(jax.random.key(key_seed), 5)
            for field, value in expected.items():
                assert float(getattr(params, field)) == pytest.approx(value)


def test_uniform_severity_carries_the_dimension_term_when_pinned():
    """The uniform model's alpha is `multiplier * (0.49 + 1/D)` at both severities."""
    for num_dims in (2, 5, 40):
        assert float(Uniform.severe().sample(jax.random.key(0), num_dims).alpha) == (
            pytest.approx(0.49 + 1 / num_dims)
        )
        assert float(Uniform.moderate().sample(jax.random.key(0), num_dims).alpha) == (
            pytest.approx(0.01 * (0.49 + 1 / num_dims))
        )


def test_the_default_range_spans_the_two_severities():
    """The continuous default runs from the moderate value to the severe one."""
    assert Gaussian().beta_range == (0.01, 1.0)
    assert Uniform().alpha_range == (0.01, 1.0)
    assert Uniform().beta_range == (0.01, 1.0)
    assert Cauchy().alpha_range == (0.01, 1.0)
    assert Cauchy().p_range == (0.05, 0.2)
