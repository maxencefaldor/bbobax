"""Tests for Many-Affine BBOB.

The combination is checked against an independent numpy transcription of
IOHexperimenter's `ManyAffine::evaluate`, evaluated on bbobax's own components
and weights -- the same structural approach `test_alignment.py` takes for the
Gallagher functions, and for the same reason: bbobax samples instances rather
than enumerating them, so a reference instance cannot be injected.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bbobax.bbob import BBOB_PROBLEMS, DIMENSIONS
from bbobax.many_affine import ManyAffine

# `many_affine.hpp`: `default_scales`, in canonical f1-f24 order.
OFFICIAL_SCALES = (
    11.0, 17.5, 12.3, 12.6, 11.5, 15.3, 12.1, 15.3,
    15.2, 17.4, 13.4, 20.4, 12.9, 10.4, 12.3, 10.3,
    9.8, 10.6, 10.0, 14.7, 10.7, 10.8, 9.0, 12.1,
)  # fmt: skip


def official_combine(component_values, weights, scales=OFFICIAL_SCALES):
    """`ManyAffine::evaluate`'s combination, transcribed into numpy.

    `f0 = clamp(f0, 1e-12, 1e20); f0 = (log10(f0) + 8) / scale; f0 *= w;`
    summed, then `pow(10, 10 * result - 8)`.
    """
    result = 0.0
    for value, weight, scale in zip(component_values, weights, scales, strict=True):
        f0 = min(max(float(value), 1e-12), 1e20)
        result += weight * (np.log10(f0) + 8.0) / scale
    return 10.0 ** (10.0 * result - 8.0)


def component_values_of(problem, x, params):
    """Reproduce what each of the 24 contributes, from the instance key alone."""
    _, key_components = jax.random.split(params.key)
    keys = jax.random.split(key_components, len(problem.components))

    values = []
    for component, key in zip(problem.components, keys, strict=True):
        component_params = component.sample(key)
        x0 = x + component_params.x_opt - params.x_opt
        value, penalty = component._value(x0, component_params)
        values.append(float(value) + float(penalty))
    return values


def weights_of(problem, params):
    """Reproduce the instance's weight vector."""
    key_weights, _ = jax.random.split(params.key)
    return np.asarray(problem._sample_weights(key_weights))


def test_the_scales_are_the_reference_constants():
    """The per-function scales are IOH's `default_scales`, unchanged."""
    assert ManyAffine.scales == OFFICIAL_SCALES
    assert len(ManyAffine.scales) == len(BBOB_PROBLEMS) == 24
    assert ManyAffine.weight_floor == 0.85
    assert ManyAffine.value_range == (1e-12, 1e20)


def test_it_is_built_from_all_24_in_order():
    """The components are the standard suite, at the problem's own dimension."""
    problem = ManyAffine(num_dims=6)

    assert [type(c).__name__ for c in problem.components] == [
        c.__name__ for c in BBOB_PROBLEMS.values()
    ]
    assert all(c.num_dims == 6 for c in problem.components)


@pytest.mark.parametrize("num_dims", [2, 10])
def test_the_weights_are_a_sparse_simplex(num_dims):
    """Non-negative, summing to 1, with at least two functions always combining.

    The reference raises the two largest draws to the 0.85 floor and zeroes
    everything below it, so a combination is never a single function wearing a
    disguise -- and is usually a handful, since P(U >= 0.85) = 0.15 over 24.
    """
    problem = ManyAffine(num_dims=num_dims)
    counts = []

    for seed in range(64):
        weights = weights_of(problem, problem.sample(jax.random.key(seed)))

        assert np.all(weights >= 0.0)
        assert float(np.sum(weights)) == pytest.approx(1.0)
        counts.append(int(np.sum(weights > 0.0)))
        assert counts[-1] >= 2

    # Sparse, but genuinely varying: never all 24, and not always the floor.
    assert max(counts) < 24
    assert len(set(counts)) > 1


@pytest.mark.parametrize("num_dims", [2, 5, 10])
def test_the_combination_matches_the_reference_formula(num_dims):
    """Match IOH's combination of bbobax's own component values.

    The optimum offset is added back, because that is the only place the two
    differ: the reference reports the raw combination and stores `f(x_opt)` as
    the optimum's value, while bbobax returns 0 at the optimum by convention.
    """
    problem = ManyAffine(num_dims=num_dims)
    params = problem.sample(jax.random.key(num_dims))
    weights = weights_of(problem, params)

    at_optimum = official_combine([0.0] * 24, weights)

    rng = np.random.default_rng(num_dims)
    points = np.vstack(
        [
            rng.uniform(-5.0, 5.0, size=(12, num_dims)),
            rng.uniform(-8.0, 8.0, size=(4, num_dims)),
            np.asarray(params.x_opt)[None, :],
        ]
    )

    for x in points:
        x = jnp.asarray(x)
        ours, penalty = problem._value(x, params)
        theirs = official_combine(component_values_of(problem, x, params), weights)

        assert float(penalty) == 0.0
        np.testing.assert_allclose(
            float(ours) + at_optimum, theirs, rtol=1e-10, atol=1e-12
        )


@pytest.mark.parametrize("num_dims", DIMENSIONS)
def test_the_optimum_is_zero_and_nothing_goes_below_it(num_dims):
    """Hold bbobax's optimum-at-zero invariant for ManyAffine, at every dimension."""
    problem = ManyAffine(num_dims=num_dims)
    params = problem.sample(jax.random.key(num_dims))

    value, penalty = problem._value(params.x_opt, params)
    assert float(value) + float(penalty) == pytest.approx(0.0, abs=1e-9)

    rng = np.random.default_rng(100 + num_dims)
    xs = jnp.asarray(
        np.vstack(
            [
                rng.uniform(-5.0, 5.0, size=(16, num_dims)),
                rng.uniform(-8.0, 8.0, size=(4, num_dims)),
            ]
        )
    )
    keys = jax.random.split(jax.random.key(1), xs.shape[0])
    fitness = np.asarray(
        jax.vmap(problem.evaluate, in_axes=(0, 0, None))(keys, xs, params).fitness
    )

    assert np.all(np.isfinite(fitness)), f"D={num_dims}: non-finite"
    assert np.all(fitness >= -1e-9), f"D={num_dims}: below the optimum"


def test_the_landscape_follows_the_instance_key():
    """Weights and components are instance data: same key same problem."""
    problem = ManyAffine(num_dims=4)
    xs = jnp.asarray(np.random.default_rng(0).uniform(-5.0, 5.0, size=(16, 4)))

    def values(seed):
        params = problem.sample(jax.random.key(seed))
        return np.asarray(jax.vmap(problem._value, in_axes=(0, None))(xs, params)[0])

    np.testing.assert_array_equal(values(1), values(1))
    assert not np.allclose(values(1), values(2))


def test_jit_vmap():
    """A batch of instances evaluates in one call -- the meta-learning shape."""
    num_dims, instances = 5, 8
    problem = ManyAffine(num_dims=num_dims)

    keys = jax.random.split(jax.random.key(0), instances)
    params = jax.vmap(problem.sample)(keys)
    xs = jax.vmap(problem.sample_x)(keys)

    fitness = jax.jit(jax.vmap(problem.evaluate, in_axes=(0, 0, 0)))(
        keys, xs, params
    ).fitness

    assert fitness.shape == (instances,)
    assert jnp.all(jnp.isfinite(fitness))
    # Different instances are different problems.
    assert len(set(np.asarray(fitness).tolist())) == instances
