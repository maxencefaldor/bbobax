"""Tests for the noisy BBOB suite, f101-f130.

The suite table is pinned against the vendored `bbobbenchmarks.py`, which is
where it was read from: each `F1xx` class names its base and its noise
settings, so the table can be checked entry by entry rather than trusted.
"""

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bbobax.bbob import DIMENSIONS, EllipsoidalRotated, GriewankRosenbrock
from bbobax.noisy import (
    NOISY_PROBLEMS,
    _Ellipsoid1e4,
    _GriewankRosenbrock1,
    noisy_suite,
)
from bbobax.problem import BBOBProblem
from bbobax.transforms import f_pen

# How the official class names map onto bbobax's. `_FRosenbrock` is F8's base,
# the unrotated one; `_FEllipsoid` is F10's, the rotated one.
OFFICIAL_BASES = {
    "_FSphere": "sphere",
    "_FRosenbrock": "rosenbrock",
    "_FStepEllipsoid": "step_ellipsoidal",
    "_FEllipsoid": "ellipsoidal_rotated_1e4",
    "_FDiffPow": "different_powers",
    "_FSchaffersF7": "schaffers_f7",
    "_F8F2": "griewank_rosenbrock_1",
    "_FGallagher": "gallagher_101_me",
}
OFFICIAL_NOISE = {"Gauss": "gaussian", "Uniform": "uniform", "Cauchy": "cauchy"}


def official_suite():
    """Read f101-f130 back out of the vendored reference, as (base, model)."""
    source = (
        __import__("pathlib")
        .Path(__file__)
        .parent.joinpath("_official/bbobbenchmarks.py")
        .read_text()
    )
    found = {}
    for match in re.finditer(
        r"class F(1[0-3][0-9])\((_F\w+), BBOB(\w+)Function\)", source
    ):
        fid, base, noise = match.groups()
        found[f"f{fid}"] = (OFFICIAL_BASES[base], OFFICIAL_NOISE[noise])
    return found


def test_the_suite_is_the_reference_table():
    """Every one of the thirty pairs the base and model the reference does."""
    official = official_suite()
    assert len(official) == 30, "expected f101-f130 in the vendored reference"

    for name, (base_name, noise_name) in official.items():
        problem_class, severity = NOISY_PROBLEMS[name]
        assert problem_class.name == base_name, name
        assert severity.startswith(noise_name), name

    assert list(NOISY_PROBLEMS) == [f"f{fid}" for fid in range(101, 131)]


def test_the_severities_are_the_papers_two():
    """f101-f106 are moderate; everything from f107 on is severe."""
    for name, (_, severity) in NOISY_PROBLEMS.items():
        fid = int(name[1:])
        expected = "moderate" if fid <= 106 else "severe"
        assert severity.endswith(expected), name


def test_the_reparameterized_bases_differ_from_their_noiseless_twins():
    """Two of the bases are not the noiseless function, and must not be."""
    suite = noisy_suite(["f116", "f125"], num_dims=5)

    # f116-f118 use an ellipsoid of conditioning 1e4, where f10 has 1e6.
    assert type(suite["f116"].problem) is _Ellipsoid1e4
    assert _Ellipsoid1e4.condition == 1e4
    assert EllipsoidalRotated.condition == 1e6

    # f125-f127 scale the composition by 1, where f19 uses 10.
    assert type(suite["f125"].problem) is _GriewankRosenbrock1
    assert _GriewankRosenbrock1.facftrue == 1.0
    assert GriewankRosenbrock.facftrue == 10.0


def test_boundary_handling_is_uniform_and_replaces_the_base():
    """Every noisy problem penalizes with factor 100, whatever its base did.

    The base's own penalty is discarded rather than added to -- including for
    the several noiseless functions that have none at all.
    """
    suite = noisy_suite(num_dims=4)
    key = jax.random.key(0)
    outside = jnp.array([7.0, -7.0, 0.0, 6.0])

    for name, problem in suite.items():
        assert problem.penalty_factor == 100.0, name
        params = problem.sample(key)

        _, penalty = problem._value(outside, params)
        np.testing.assert_allclose(
            float(penalty), 100.0 * float(f_pen(outside)), rtol=1e-12
        )
        # And it is the base's core value that is handed back, not its value
        # plus its own penalty.
        base_value, _ = problem.problem._value(outside, params)
        ours, _ = problem._value(outside, params)
        assert float(ours) == float(base_value), name


def test_it_is_a_problem_like_any_other():
    """`Noisy` satisfies the same contract, so nothing downstream cares."""
    problem = noisy_suite(["f107"], num_dims=5)["f107"]
    assert isinstance(problem, BBOBProblem)

    params = problem.sample(jax.random.key(0))
    x = problem.sample_x(jax.random.key(1))
    assert x.shape == (5,)
    assert problem.evaluate(jax.random.key(2), x, params).fitness.shape == ()


def test_the_optimum_stays_reachable_under_every_model():
    """Stabilization holds at the optimum, for all thirty.

    This is what the three official models' stabilization is *for*: noise must
    never stop an algorithm reaching the target precision, so `f(x_opt)` is the
    undisturbed value however many times it is asked.
    """
    suite = noisy_suite(num_dims=4)
    key = jax.random.key(0)

    for name, problem in suite.items():
        params = problem.sample(key)
        values = {
            float(problem.evaluate(jax.random.key(k), params.x_opt, params).fitness)
            for k in range(8)
        }
        assert values == {0.0}, f"{name}: {values}"


def test_noise_is_live_away_from_the_optimum():
    """Above the target precision every problem is genuinely stochastic.

    Enough draws to see a Cauchy outlier. `fCauchy` is
    `f + alpha * max(0, 1e3 + 1{U < p} * N/|N|)`, so between outliers it is
    *deterministic* -- a constant shift of `1000 * alpha` -- and only the
    indicator firing makes it vary. At f103's moderate `p = 0.05`, eight draws
    miss an outlier two times in three; 256 miss it with probability 2e-6.
    """
    suite = noisy_suite(num_dims=4)
    key = jax.random.key(0)

    for name, problem in suite.items():
        params = problem.sample(key)
        x = problem.sample_x(jax.random.key(1))
        values = {
            float(problem.evaluate(jax.random.key(k), x, params).fitness)
            for k in range(256)
        }
        assert len(values) > 1, f"{name} is deterministic"


def test_moderate_is_milder_than_severe():
    """The paired moderate/severe problems differ in the expected direction.

    f101/f107 are the same sphere under the same Gaussian model at the paper's
    two settings, so the severe one must scatter more.
    """
    suite = noisy_suite(["f101", "f107"], num_dims=5)
    key = jax.random.key(0)

    spread = {}
    for name, problem in suite.items():
        params = problem.sample(key)
        x = problem.sample_x(jax.random.key(1))
        values = np.array(
            [
                float(problem.evaluate(jax.random.key(k), x, params).fitness)
                for k in range(256)
            ]
        )
        spread[name] = float(np.std(np.log(values)))

    assert spread["f107"] > 10 * spread["f101"], spread


@pytest.mark.parametrize("num_dims", DIMENSIONS)
def test_nothing_is_non_finite_at_any_dimension(num_dims):
    """All thirty stay finite across the box and outside it, at every D."""
    suite = noisy_suite(num_dims=num_dims)
    rng = np.random.default_rng(num_dims)
    xs = jnp.asarray(
        np.vstack(
            [
                rng.uniform(-5.0, 5.0, size=(8, num_dims)),
                rng.uniform(-8.0, 8.0, size=(4, num_dims)),
            ]
        )
    )
    keys = jax.random.split(jax.random.key(1), xs.shape[0])

    for name, problem in suite.items():
        params = problem.sample(jax.random.key(num_dims))
        fitness = jax.vmap(problem.evaluate, in_axes=(0, 0, None))(keys, xs, params)
        assert jnp.all(jnp.isfinite(fitness.fitness)), f"{name} at D={num_dims}"


def test_the_severities_are_pinned_not_sampled():
    """A published f1xx number is at one of the paper's two points, exactly.

    Checked on what an instance actually *draws*, not on how its model was
    configured: pinning a range is only useful if the draw lands on the point.
    """
    names = ["f101", "f107", "f102", "f108", "f103", "f109"]
    suite = noisy_suite(names, num_dims=5)

    drawn = {
        name: suite[name].sample(jax.random.key(seed)).noise_model
        for seed in range(4)
        for name in names
    }

    # Gaussian: beta 0.01 moderate, 1 severe.
    assert float(drawn["f101"].beta) == pytest.approx(0.01)
    assert float(drawn["f107"].beta) == pytest.approx(1.0)
    # Uniform: beta 0.01 / 1, and alpha carries the (0.49 + 1/D) term.
    assert float(drawn["f102"].beta) == pytest.approx(0.01)
    assert float(drawn["f108"].beta) == pytest.approx(1.0)
    assert float(drawn["f108"].alpha) == pytest.approx(0.49 + 1 / 5)
    # Cauchy: (alpha, p) = (0.01, 0.05) moderate, (1, 0.2) severe.
    assert float(drawn["f103"].alpha) == pytest.approx(0.01)
    assert float(drawn["f103"].p) == pytest.approx(0.05)
    assert float(drawn["f109"].alpha) == pytest.approx(1.0)
    assert float(drawn["f109"].p) == pytest.approx(0.2)


def test_suite_rejects_unknown_names():
    """Only f101-f130 exist."""
    with pytest.raises(KeyError, match="f999"):
        noisy_suite(["f101", "f999"])
