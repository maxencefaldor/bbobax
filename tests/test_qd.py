"""Tests for the Quality-Diversity extension.

The descriptor families are tested against the two promises the module makes:
every family satisfies the same contract, and every family lands in
`[-1, 1]^k` exactly for solutions inside the search box. Each family's own
phenomenon -- irregularity, discontinuity, sensitivity, redundancy, alignment
-- is then pinned by the property that defines it.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bbobax.bbob import BBOB_PROBLEMS, Rastrigin, Sphere
from bbobax.qd import (
    AlignedProjection,
    Descriptor,
    FourierProjection,
    IrregularProjection,
    QDEval,
    QDParams,
    QDProblem,
    QuantizedProjection,
    RandomProjection,
    SubsetProjection,
    sphere_descriptor_optimum,
)

FAMILIES = [
    RandomProjection,
    IrregularProjection,
    QuantizedProjection,
    FourierProjection,
    SubsetProjection,
    AlignedProjection,
]


def make_instance(descriptor, num_dims=6, seed=0):
    """One problem instance and the descriptor's, drawn the way QDProblem does."""
    problem = Sphere(num_dims=num_dims)
    key_problem, key_descriptor = jax.random.split(jax.random.key(seed))
    problem_params = problem.sample(key_problem)
    return problem, descriptor.sample(key_descriptor, problem, problem_params)


# --- The contract, for every family ------------------------------------------


@pytest.mark.parametrize("family", FAMILIES)
def test_every_family_satisfies_the_protocol(family):
    """All six families are Descriptors, structurally."""
    assert isinstance(family(descriptor_size=2), Descriptor)


@pytest.mark.parametrize("family", FAMILIES)
def test_every_family_declares_the_box(family):
    """Descriptor space is [-1, 1]^k by construction, and says so."""
    assert family(descriptor_size=2).descriptor_range == (-1.0, 1.0)


@pytest.mark.parametrize("family", FAMILIES)
def test_every_family_lands_in_the_box(family):
    """For solutions inside the search box, the descriptor is in [-1, 1]^k."""
    descriptor = family(descriptor_size=2)
    problem, params = make_instance(descriptor)

    xs = jax.vmap(problem.sample_x)(jax.random.split(jax.random.key(1), 256))
    values = jax.vmap(descriptor.evaluate, in_axes=(None, 0, None))(
        jax.random.key(2), xs, params
    )

    assert values.shape == (256, 2)
    assert jnp.all(jnp.abs(values) <= 1.0 + 1e-6)


@pytest.mark.parametrize("family", FAMILIES)
def test_every_family_is_deterministic_in_the_key(family):
    """The shipped families measure exactly; the key is contract, not behavior."""
    descriptor = family(descriptor_size=2)
    problem, params = make_instance(descriptor)
    x = problem.sample_x(jax.random.key(1))

    values = {
        tuple(np.asarray(descriptor.evaluate(jax.random.key(k), x, params)))
        for k in range(4)
    }
    assert len(values) == 1


@pytest.mark.parametrize("family", FAMILIES)
def test_every_family_jit_vmaps(family):
    """Sampling and evaluation compile, and evaluation vmaps over solutions."""
    descriptor = family(descriptor_size=3)
    problem, params = make_instance(descriptor)

    xs = jax.vmap(problem.sample_x)(jax.random.split(jax.random.key(1), 8))
    batch = jax.jit(jax.vmap(descriptor.evaluate, in_axes=(None, 0, None)))(
        jax.random.key(2), xs, params
    )
    assert batch.shape == (8, 3)


@pytest.mark.parametrize("family", FAMILIES)
def test_descriptor_size_is_validated(family):
    """A descriptor space needs at least one dimension."""
    with pytest.raises(ValueError, match="descriptor_size"):
        family(descriptor_size=0)


# --- Each family's own phenomenon --------------------------------------------


def test_random_projection_bounds_are_tight():
    """The bound is achieved: the right box corner maps to exactly +-1."""
    descriptor = RandomProjection(descriptor_size=2)
    problem, params = make_instance(descriptor)

    for row in np.asarray(params):
        corner = 5.0 * np.sign(row)
        value = float(np.asarray(row) @ corner)
        assert value == pytest.approx(1.0)


def test_irregular_projection_stays_bounded_and_differs():
    """Osz and asy preserve [-1, 1] while bending the metric."""
    plain = RandomProjection(descriptor_size=2)
    irregular = IrregularProjection(descriptor_size=2, beta=0.5)
    problem, params = make_instance(plain)  # same sample: same matrix

    xs = jax.vmap(problem.sample_x)(jax.random.split(jax.random.key(1), 64))
    straight = jax.vmap(plain.evaluate, in_axes=(None, 0, None))(
        jax.random.key(2), xs, params
    )
    bent = jax.vmap(irregular.evaluate, in_axes=(None, 0, None))(
        jax.random.key(2), xs, params
    )

    assert jnp.all(jnp.abs(bent) <= 1.0 + 1e-6)
    assert not jnp.allclose(straight, bent)
    # The endpoints are fixed: osz(+-1) = +-1 and asy leaves them.
    np.testing.assert_allclose(
        np.asarray(irregular.evaluate(jax.random.key(0), jnp.zeros(6), params)),
        0.0,
        atol=1e-12,
    )


def test_irregular_projection_needs_two_components():
    """The asymmetry schedule spans components, exactly as BBOB needs D >= 2."""
    with pytest.raises(ValueError, match="descriptor_size"):
        IrregularProjection(descriptor_size=1)


def test_quantized_projection_values_are_level_centers():
    """Every value is one of the num_levels centers, and more than one occurs."""
    num_levels = 10
    descriptor = QuantizedProjection(descriptor_size=2, num_levels=num_levels)
    problem, params = make_instance(descriptor)

    xs = jax.vmap(problem.sample_x)(jax.random.split(jax.random.key(1), 256))
    values = jax.vmap(descriptor.evaluate, in_axes=(None, 0, None))(
        jax.random.key(2), xs, params
    )

    width = 2.0 / num_levels
    levels = (values + 1.0) / width - 0.5
    np.testing.assert_allclose(levels, np.round(levels), atol=1e-6)
    assert len(np.unique(np.asarray(values))) > 2


def test_quantized_projection_jumps():
    """Two nearby solutions on either side of a level boundary differ by a level.

    The discontinuity is the phenomenon: an arbitrarily small step in x moves
    the descriptor by a whole level width.
    """
    num_levels = 10
    descriptor = QuantizedProjection(descriptor_size=1, num_levels=num_levels)
    problem, params = make_instance(descriptor, num_dims=2)

    # March along a direction that moves the projection, in tiny steps.
    direction = jnp.asarray(np.sign(np.asarray(params))[0])
    steps = jnp.linspace(0.0, 4.0, 4096)
    xs = steps[:, None] * direction[None, :]
    values = jax.vmap(descriptor.evaluate, in_axes=(None, 0, None))(
        jax.random.key(0), xs, params
    )[:, 0]

    jumps = jnp.abs(jnp.diff(values))
    assert jnp.all((jumps == 0.0) | (jnp.abs(jumps - 2.0 / num_levels) < 1e-6))
    assert jnp.any(jumps > 0.0)


def test_fourier_projection_sensitivity_scales_with_bandwidth():
    """The Lipschitz constant is the dial: 10x bandwidth, ~10x the gradient."""
    problem = Sphere(num_dims=6)
    key_problem, key_descriptor = jax.random.split(jax.random.key(0))
    problem_params = problem.sample(key_problem)

    norms = {}
    for bandwidth in (1.0, 10.0):
        descriptor = FourierProjection(descriptor_size=2, bandwidth=bandwidth)
        params = descriptor.sample(key_descriptor, problem, problem_params)
        xs = jax.vmap(problem.sample_x)(jax.random.split(jax.random.key(1), 64))
        jacobians = jax.vmap(
            jax.jacobian(descriptor.evaluate, argnums=1), in_axes=(None, 0, None)
        )(jax.random.key(2), xs, params)
        norms[bandwidth] = float(jnp.mean(jnp.linalg.norm(jacobians, axis=(1, 2))))

    assert norms[10.0] > 5.0 * norms[1.0]


def test_fourier_projection_is_bounded_everywhere():
    """A cosine needs no box: bounded even far outside the search space."""
    descriptor = FourierProjection(descriptor_size=2)
    problem, params = make_instance(descriptor)

    far = 100.0 * jax.random.normal(jax.random.key(1), shape=(64, 6))
    values = jax.vmap(descriptor.evaluate, in_axes=(None, 0, None))(
        jax.random.key(2), far, params
    )
    assert jnp.all(jnp.abs(values) <= 1.0)


def test_subset_projection_ignores_the_rest():
    """Moving a coordinate outside the subset does not move the descriptor."""
    descriptor = SubsetProjection(descriptor_size=2, subset_size=2)
    problem, params = make_instance(descriptor)

    x = problem.sample_x(jax.random.key(1))
    before = descriptor.evaluate(jax.random.key(2), x, params)

    outside = jnp.setdiff1d(
        jnp.arange(problem.num_dims), params.subset, size=problem.num_dims - 2
    )
    moved = x.at[outside].add(1.0)
    after = descriptor.evaluate(jax.random.key(2), moved, params)
    np.testing.assert_allclose(np.asarray(before), np.asarray(after), rtol=1e-12)

    # And moving a subset coordinate does move it.
    changed = descriptor.evaluate(
        jax.random.key(2), x.at[params.subset[0]].add(1.0), params
    )
    assert not jnp.allclose(before, changed)


def test_subset_projection_rejects_oversized_subsets():
    """The subset cannot exceed the dimension it reads from."""
    descriptor = SubsetProjection(descriptor_size=2, subset_size=8)
    problem = Sphere(num_dims=4)
    params = problem.sample(jax.random.key(0))

    with pytest.raises(ValueError, match="subset_size"):
        descriptor.sample(jax.random.key(1), problem, params)


def test_aligned_projection_at_one_is_the_landscape_axes():
    """Alignment = 1 describes the solution in the instance's own rotation."""
    descriptor = AlignedProjection(descriptor_size=2, alignment=1.0)
    problem = Sphere(num_dims=6)
    key_problem, key_descriptor = jax.random.split(jax.random.key(0))
    problem_params = problem.sample(key_problem)
    params = descriptor.sample(key_descriptor, problem, problem_params)

    rows = np.asarray(problem_params.r)[:2]
    expected = rows / (np.sum(np.abs(rows), axis=-1, keepdims=True) * 5.0)
    np.testing.assert_allclose(np.asarray(params), expected, rtol=1e-6)


def test_aligned_projection_validates_its_dial():
    """Alignment lives in [0, 1]; the rotation has only num_dims rows."""
    with pytest.raises(ValueError, match="alignment"):
        AlignedProjection(descriptor_size=2, alignment=1.5)

    descriptor = AlignedProjection(descriptor_size=8)
    problem = Sphere(num_dims=4)
    with pytest.raises(ValueError, match="descriptor_size"):
        descriptor.sample(jax.random.key(0), problem, problem.sample(jax.random.key(1)))


# --- The composed problem -----------------------------------------------------


def test_qd_problem_workflow():
    """The QD contract: sample -> evaluate, fitness and descriptor together."""
    num_dims, descriptor_size = 5, 2
    problem = QDProblem(
        Sphere(num_dims=num_dims), RandomProjection(descriptor_size=descriptor_size)
    )
    key = jax.random.key(0)

    params = problem.sample(key)
    assert isinstance(params, QDParams)
    assert params.descriptor.shape == (descriptor_size, num_dims)
    # Composed, not inherited: the problem's own instance is nested whole.
    assert params.problem.x_opt.shape == (num_dims,)

    x = problem.sample_x(key)
    evaluation = problem.evaluate(key, x, params)

    assert isinstance(evaluation, QDEval)
    assert evaluation.fitness.shape == ()
    assert evaluation.descriptor.shape == (descriptor_size,)


def test_qd_problem_delegates_what_it_wraps():
    """A QD problem answers for the function it wraps, bounds included."""
    problem = QDProblem(Rastrigin(num_dims=7), RandomProjection(descriptor_size=3))

    assert problem.name == "rastrigin"
    assert problem.num_dims == 7
    assert problem.x_range == (-5.0, 5.0)
    assert problem.descriptor_size == 3
    # The ground truth an archive needs, straight from the problem.
    assert problem.descriptor_range == (-1.0, 1.0)


def test_qd_fitness_is_the_underlying_fitness():
    """Pairing a descriptor onto a function does not change the function."""
    num_dims = 6
    base = Rastrigin(num_dims=num_dims)
    problem = QDProblem(base, RandomProjection(descriptor_size=2))

    key = jax.random.key(0)
    params = problem.sample(key)
    x = problem.sample_x(jax.random.key(1))

    qd_evaluation = problem.evaluate(jax.random.key(2), x, params)
    base_evaluation = base.evaluate(jax.random.key(2), x, params.problem)

    assert float(qd_evaluation.fitness) == float(base_evaluation.fitness)
    # And the descriptor is exactly the projection of x.
    np.testing.assert_allclose(
        np.asarray(qd_evaluation.descriptor),
        np.asarray(params.descriptor) @ np.asarray(x),
        rtol=1e-12,
    )


@pytest.mark.parametrize("family", FAMILIES)
def test_any_function_pairs_with_any_family(family):
    """Composition is why this is not 24 subclasses: all 24 pair the same way.

    And the `[-1, 1]^k` promise is checked here for *every* function, not only
    the one the focused bounds tests use: the bound depends only on the box
    and the descriptor, so it must hold whatever landscape sits underneath --
    including the six functions that constrain their optimum, and the
    rotation-coupled family on every instance's own rotation.
    """
    num_dims, descriptor_size, batch = 4, 2, 32
    descriptor = family(descriptor_size=descriptor_size)
    key = jax.random.key(3)

    for name, problem_class in BBOB_PROBLEMS.items():
        problem = QDProblem(problem_class(num_dims=num_dims), descriptor)
        params = problem.sample(key)

        keys = jax.random.split(key, batch)
        xs = jax.vmap(problem.sample_x)(keys)
        evaluation = jax.vmap(problem.evaluate, in_axes=(0, 0, None))(keys, xs, params)

        assert evaluation.descriptor.shape == (batch, descriptor_size), name
        assert not jnp.any(jnp.isnan(evaluation.fitness)), name
        assert jnp.all(jnp.abs(evaluation.descriptor) <= 1.0 + 1e-6), name


# --- Exact ground truth, where it exists --------------------------------------


def test_sphere_descriptor_optimum_is_feasible_and_matches_evaluate():
    """The argmin achieves `d` exactly, and its fitness is what evaluate says."""
    problem = QDProblem(Sphere(num_dims=8), RandomProjection(descriptor_size=2))
    params = problem.sample(jax.random.key(0))
    d = jnp.array([0.3, -0.5])

    fitness, x = sphere_descriptor_optimum(d, params.descriptor, params.problem)

    np.testing.assert_allclose(
        np.asarray(params.descriptor @ x), np.asarray(d), atol=1e-6
    )
    evaluation = problem.evaluate(jax.random.key(1), x, params)
    assert float(evaluation.fitness) == pytest.approx(float(fitness), rel=1e-5)


def test_sphere_descriptor_optimum_is_the_optimum():
    """No solution achieving `d` beats it: the claim, attacked directly.

    Every `x` with `matrix @ x = d` is the argmin plus a null-space
    perturbation, so the whole feasible set is enumerable up to sampling --
    and everything in it must be at least as bad.
    """
    num_dims = 8
    problem = QDProblem(Sphere(num_dims=num_dims), RandomProjection(descriptor_size=2))
    params = problem.sample(jax.random.key(0))
    matrix = params.descriptor
    d = jnp.array([-0.2, 0.4])

    fitness, x = sphere_descriptor_optimum(d, matrix, params.problem)

    # Feasible perturbations: project random directions onto the null space.
    null = jnp.eye(num_dims) - jnp.linalg.pinv(matrix) @ matrix
    perturbations = jax.random.normal(jax.random.key(1), shape=(64, num_dims)) @ null.T
    rivals = x[None, :] + perturbations
    rival_fitness = jax.vmap(
        lambda r: problem.evaluate(jax.random.key(2), r, params).fitness
    )(rivals)

    np.testing.assert_allclose(
        np.asarray(matrix @ rivals.T).T, np.tile(np.asarray(d), (64, 1)), atol=1e-5
    )
    assert jnp.all(rival_fitness >= fitness - 1e-6)


def test_sphere_descriptor_optimum_supports_subset_by_scattering():
    """The documented scatter makes SubsetProjection's map a full-width matrix."""
    num_dims = 6
    problem = QDProblem(
        Sphere(num_dims=num_dims), SubsetProjection(descriptor_size=2, subset_size=3)
    )
    params = problem.sample(jax.random.key(0))
    sub = params.descriptor

    full = jnp.zeros((2, num_dims)).at[:, sub.subset].set(sub.matrix)
    d = jnp.array([0.1, 0.2])
    fitness, x = sphere_descriptor_optimum(d, full, params.problem)

    evaluation = problem.evaluate(jax.random.key(1), x, params)
    np.testing.assert_allclose(
        np.asarray(evaluation.descriptor), np.asarray(d), atol=1e-6
    )
    assert float(evaluation.fitness) == pytest.approx(float(fitness), rel=1e-5)
    # Coordinates the descriptor never reads stay at the optimum: no correction.
    outside = np.setdiff1d(np.arange(num_dims), np.asarray(sub.subset))
    np.testing.assert_allclose(
        np.asarray(x)[outside], np.asarray(params.problem.x_opt)[outside], rtol=1e-6
    )


def test_sphere_descriptor_optimum_at_the_optimums_own_descriptor():
    """At `d = matrix @ x_opt` the optimum is `x_opt` itself, worth `f_opt`."""
    problem = QDProblem(Sphere(num_dims=5), RandomProjection(descriptor_size=2))
    params = problem.sample(jax.random.key(0))

    d = params.descriptor @ params.problem.x_opt
    fitness, x = sphere_descriptor_optimum(d, params.descriptor, params.problem)

    np.testing.assert_allclose(
        np.asarray(x), np.asarray(params.problem.x_opt), atol=1e-6
    )
    assert float(fitness) == pytest.approx(float(params.problem.f_opt), abs=1e-9)


def test_qd_problem_jit_vmap():
    """Test JAX transformations on QD evaluation."""
    num_dims, descriptor_size, batch_size = 5, 2, 8
    problem = QDProblem(
        Sphere(num_dims=num_dims), RandomProjection(descriptor_size=descriptor_size)
    )
    key = jax.random.key(0)
    params = problem.sample(key)

    keys = jax.random.split(key, batch_size)
    xs = jax.vmap(problem.sample_x)(keys)
    results = jax.jit(jax.vmap(problem.evaluate, in_axes=(0, 0, None)))(
        keys, xs, params
    )

    assert results.fitness.shape == (batch_size,)
    assert results.descriptor.shape == (batch_size, descriptor_size)
