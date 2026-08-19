"""Quality-Diversity on top of a BBOB problem.

The Quality-Diversity extension is bbobax's own; COCO has no descriptor notion.

A descriptor is *orthogonal* to a function: any of the 24 can be paired with
any descriptor, so a QD problem is composed rather than subclassed --

    problem = QDProblem(Rastrigin(num_dims=10), RandomProjection(descriptor_size=2))

Subclassing would need one QD class per function, and the combinatorics get
worse with every descriptor family added. Composition also keeps each half
testable on its own, and lets a descriptor be reused across dimensions: the
descriptor draws its instance data from the problem it is handed, so the same
object serves a whole meta-learning loop over `DIMENSIONS`.

**Descriptor space is `[-1, 1]^k` by construction**, the way COCO's search
space is `[-5, 5]^D`: every descriptor bbobax ships lands in that box exactly,
for any solution inside the search box. The linear families achieve it by
dividing each projection row by its tight bound over the box; `transform_osz`
and `transform_asy` map `[-1, 1]` onto itself, quantization stays inside by
construction, and a cosine is born there. Solutions *outside* the search box
can exceed the bounds for the linear families -- the same status BBOB gives
the boundary penalty. The benchmark knowing the bounds does not mean an
algorithm must be told: withholding them is how a bounds-free method (novelty
search and its descendants) is honestly compared against a bounds-given one
(MAP-Elites) on identical problems -- an information regime, exactly as no
algorithm reads `params.x_opt`.

The families are organized the way BBOB organizes its functions: each isolates
one named phenomenon of real descriptor maps, built by composing the same
transformations the functions themselves are built from.

- `RandomProjection` -- smooth, linear, Lipschitz: the controlled baseline.
- `IrregularProjection` -- osz/asy on the projection: non-uniform sensitivity
  and asymmetry, BBOB's own irregularities applied to behavior.
- `QuantizedProjection` -- level sets on the projection: a small step in `x`
  crosses a level and the descriptor jumps, the contact-switch phenomenon.
- `FourierProjection` -- random Fourier features: smooth but arbitrarily
  sensitive, with the Lipschitz constant as an explicit dial.
- `SubsetProjection` -- the descriptor reads a few coordinates while fitness
  depends on all of them: behavior-space redundancy.
- `AlignedProjection` -- rows interpolated toward the instance's own rotation:
  the one *coupled* family, dialing how directly behavior trades against
  fitness structure.

Reading anything from `params` beyond shapes couples a descriptor to the
landscape; every family above except `AlignedProjection` deliberately does
not.

One corner of the space has **exact ground truth**: for `Sphere` under any
*linear* descriptor, the best fitness achievable at a descriptor value is a
closed-form projection -- `sphere_descriptor_optimum`. That is something no
other QD benchmark provides: the per-cell optimum is exactly what QD-score
normalization has always lacked (evaluate it at cell centers and the archive
has an absolute reference), and it exists here because the fidelity work made
`x_opt` the true argmin.
"""

from typing import Any, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
from flax.struct import dataclass

from .problem import BBOBParams, BBOBProblem
from .transforms import transform_asy, transform_osz


@dataclass
class QDParams:
    """One sampled instance of a Quality-Diversity problem.

    Composed, not inherited: a QD problem is a problem paired with a
    descriptor, and each half draws its own instance data. The two fields are
    named for the two halves of `QDProblem`, so the parameters read the same
    way the object does. `descriptor` is whatever that descriptor's `sample`
    returns -- a projection matrix for `RandomProjection`, something else for a
    descriptor that needs more.
    """

    problem: BBOBParams
    descriptor: Any


@dataclass
class QDEval:
    """What evaluating a solution on a Quality-Diversity problem yields."""

    fitness: jax.Array
    descriptor: jax.Array


@runtime_checkable
class Descriptor(Protocol):
    """Maps a solution to a point in descriptor space.

    The same shape of contract as a problem: `sample` draws the instance data
    the descriptor needs, `evaluate` scores **one** solution -- keyed, so a
    descriptor may be stochastic, exactly as a problem may be noisy -- and
    callers `vmap` it to cover a batch.

    `sample` receives the problem and its freshly drawn instance. Shapes and
    the search box come from the problem; reading anything from `params` beyond
    that couples the descriptor to the landscape, which is a declared design
    choice rather than the default.

    A protocol rather than a base class, for the reason `BBOBProblem` spells
    out: descriptors share a contract, not an implementation, and satisfying
    one should not require importing anything from bbobax.
    """

    # The descriptor's name, and its key in `DESCRIPTORS`.
    name: str
    # Dimensionality of the descriptor space.
    descriptor_size: int
    # Exact bounds of the descriptor, per component, for solutions inside the
    # search box. Every descriptor bbobax ships is (-1.0, 1.0) by construction.
    descriptor_range: tuple[float, float]

    def sample(self, key: jax.Array, problem: BBOBProblem, params: BBOBParams) -> Any:
        """Sample this descriptor's instance data.

        Args:
            key: JAX random key.
            problem: The problem being described; its `num_dims` and `x_range`
                set the shapes and the normalization.
            params: The problem's freshly drawn instance. Reading it couples
                the descriptor to the landscape.

        Returns:
            The instance's descriptor parameters.

        """
        ...

    def evaluate(self, key: jax.Array, x: jax.Array, params: Any) -> jax.Array:
        """Compute the descriptor of a solution.

        Args:
            key: JAX random key, for descriptors that measure stochastically.
            x: Input solution, shape `(num_dims,)`.
            params: Instance descriptor parameters.

        Returns:
            The descriptor, shape `(descriptor_size,)`.

        """
        ...


class _Projection:
    """What every projection family shares: the size, the box, the bounds.

    A base class rather than a protocol, by the package's own rule: this is
    shared *implementation* -- the validation and the exact normalization that
    makes `[-1, 1]^k` true by construction. Families whose instance parameters
    are a bare matrix subclass `RandomProjection`; families with richer
    parameters subclass this directly, so no override ever narrows a type.
    """

    descriptor_range: tuple[float, float] = (-1.0, 1.0)

    def __init__(self, descriptor_size: int = 2):
        """Initialize the descriptor.

        Args:
            descriptor_size: Dimensionality of the descriptor space.

        """
        if descriptor_size < 1:
            raise ValueError(f"descriptor_size must be >= 1, got {descriptor_size}")
        self.descriptor_size = descriptor_size

    def _normalize(self, matrix: jax.Array, problem: BBOBProblem) -> jax.Array:
        """Divide each row by its tight bound over the problem's search box."""
        half_width = float(max(abs(problem.x_range[0]), abs(problem.x_range[1])))
        bound = jnp.sum(jnp.abs(matrix), axis=-1, keepdims=True) * half_width
        return matrix / bound


class RandomProjection(_Projection):
    """A Gaussian random projection of the solution, normalized to the box.

    Rows are Gaussian directions, and each row is divided by its tight bound
    over the search box (`sum_i |P_ji| * max|x|`), so the descriptor lands in
    `[-1, 1]` exactly and the bound is achieved at a box corner.

    This descriptor is smooth and Lipschitz: a small change in `x` gives a
    small change in the descriptor, and the reachable set is a linear image of
    the box. Real-world QD is frequently neither, so treat it as the baseline
    rather than as representative -- the other families each break one of
    these properties deliberately.

    Args:
        descriptor_size: Dimensionality of the descriptor space.

    """

    name = "random_projection"

    def sample(
        self, key: jax.Array, problem: BBOBProblem, params: BBOBParams
    ) -> jax.Array:
        """Sample the instance's projection matrix, normalized to the box.

        Args:
            key: JAX random key.
            problem: The problem being described.
            params: The problem's instance; not read -- this family is
                deliberately uncoupled from the landscape.

        Returns:
            A `(descriptor_size, num_dims)` matrix.

        """
        matrix = jax.random.normal(key, shape=(self.descriptor_size, problem.num_dims))
        return self._normalize(matrix, problem)

    def evaluate(self, key: jax.Array, x: jax.Array, params: jax.Array) -> jax.Array:
        """Compute the descriptor of a solution: the projection of `x`.

        Args:
            key: JAX random key; unused, this descriptor is deterministic.
            x: Input solution, shape `(num_dims,)`.
            params: The instance's projection matrix.

        Returns:
            The descriptor, shape `(descriptor_size,)`.

        """
        return params @ x


class IrregularProjection(RandomProjection):
    """The projection under BBOB's own irregularities: osz, then asy.

    Both transformations map `[-1, 1]` onto itself -- osz fixes the endpoints
    and is strictly monotone, asy only shrinks positive components -- so the
    bounds survive the composition exactly. What changes is the *metric*: the
    descriptor's sensitivity to `x` varies over the space and is asymmetric
    around the origin, the same distortions the functions themselves apply.

    Args:
        descriptor_size: Dimensionality of the descriptor space; at least 2,
            because the asymmetry schedule spans the components exactly as it
            spans the dimensions in BBOB.
        beta: The asymmetry, the paper's beta.

    """

    name = "irregular_projection"

    # Asymmetry, the paper's beta at its most common setting.
    beta: float = 0.2

    def __init__(self, descriptor_size: int = 2, beta: float = 0.2):
        """Initialize the descriptor."""
        if descriptor_size < 2:
            raise ValueError(
                f"descriptor_size must be >= 2 for the asymmetry schedule, "
                f"got {descriptor_size}"
            )
        super().__init__(descriptor_size)
        self.beta = beta

    def evaluate(self, key: jax.Array, x: jax.Array, params: jax.Array) -> jax.Array:
        """Compute the descriptor: the projection, made irregular."""
        return transform_asy(transform_osz(params @ x), self.beta)


class QuantizedProjection(RandomProjection):
    """The projection snapped to level centers: a discontinuous descriptor.

    A small step in `x` that crosses a level boundary jumps the descriptor by
    a whole level -- the contact-switch phenomenon of real behavior maps,
    built the way `StepEllipsoidal` builds its plateaus. Values are the
    `num_levels` level centers, strictly inside `[-1, 1]`.

    Args:
        descriptor_size: Dimensionality of the descriptor space.
        num_levels: Levels per component; the discontinuity size is
            `2 / num_levels`.

    """

    name = "quantized_projection"

    # Levels per component.
    num_levels: int = 10

    def __init__(self, descriptor_size: int = 2, num_levels: int = 10):
        """Initialize the descriptor."""
        if num_levels < 2:
            raise ValueError(f"num_levels must be >= 2, got {num_levels}")
        super().__init__(descriptor_size)
        self.num_levels = num_levels

    def evaluate(self, key: jax.Array, x: jax.Array, params: jax.Array) -> jax.Array:
        """Compute the descriptor: the projection, quantized to level centers."""
        width = 2.0 / self.num_levels
        level = jnp.floor((params @ x + 1.0) / width)
        level = jnp.clip(level, 0, self.num_levels - 1)
        return (level + 0.5) * width - 1.0


@dataclass
class FourierParams:
    """One sampled instance of `FourierProjection`."""

    weights: jax.Array
    phase: jax.Array


class FourierProjection(_Projection):
    """Random Fourier features of the solution: smooth but arbitrarily sensitive.

    Each component is `cos(w_j . x + b_j)` with Gaussian frequencies scaled by
    `bandwidth`, so the descriptor is smooth everywhere yet its Lipschitz
    constant grows linearly with the dial: at `bandwidth = 1` the phase moves
    through roughly one oscillation across the box, at `bandwidth = 10` a
    small step in `x` can carry the descriptor anywhere in its range. This is
    the smooth version of "a small change in `x` is a large change in
    behavior". Bounded in `[-1, 1]` by construction -- everywhere, not only
    inside the box.

    Args:
        descriptor_size: Dimensionality of the descriptor space.
        bandwidth: The sensitivity; the Lipschitz constant scales linearly
            with it.

    """

    name = "fourier_projection"

    # The sensitivity dial.
    bandwidth: float = 1.0

    def __init__(self, descriptor_size: int = 2, bandwidth: float = 1.0):
        """Initialize the descriptor."""
        super().__init__(descriptor_size)
        self.bandwidth = bandwidth

    def sample(
        self, key: jax.Array, problem: BBOBProblem, params: BBOBParams
    ) -> FourierParams:
        """Sample the instance's frequencies and phases.

        Args:
            key: JAX random key.
            problem: The problem being described.
            params: The problem's instance; not read.

        Returns:
            The instance's frequencies and phases.

        """
        key_weights, key_phase = jax.random.split(key)
        half_width = float(max(abs(problem.x_range[0]), abs(problem.x_range[1])))
        scale = self.bandwidth * jnp.pi / (half_width * jnp.sqrt(problem.num_dims))
        return FourierParams(
            weights=scale
            * jax.random.normal(
                key_weights, shape=(self.descriptor_size, problem.num_dims)
            ),
            phase=jax.random.uniform(
                key_phase, shape=(self.descriptor_size,), maxval=2.0 * jnp.pi
            ),
        )

    def evaluate(
        self, key: jax.Array, x: jax.Array, params: FourierParams
    ) -> jax.Array:
        """Compute the descriptor of a solution.

        Args:
            key: JAX random key; unused, this descriptor is deterministic.
            x: Input solution, shape `(num_dims,)`.
            params: The instance's frequencies and phases.

        Returns:
            The descriptor, shape `(descriptor_size,)`.

        """
        return jnp.cos(params.weights @ x + params.phase)


@dataclass
class SubsetParams:
    """One sampled instance of `SubsetProjection`."""

    subset: jax.Array
    matrix: jax.Array


class SubsetProjection(_Projection):
    """A projection that reads only a few coordinates: redundant behavior.

    The descriptor depends on `subset_size` randomly chosen coordinates while
    the fitness depends on all of them, so most of the search space moves
    without moving the behavior -- the genotype-to-behavior collapse of real
    QD, in its simplest form. Moving any coordinate outside the subset leaves
    the descriptor exactly unchanged.

    Args:
        descriptor_size: Dimensionality of the descriptor space.
        subset_size: How many coordinates the descriptor reads; defaults to
            `descriptor_size`, the most redundant full-rank choice.

    """

    name = "subset_projection"

    def __init__(self, descriptor_size: int = 2, subset_size: int | None = None):
        """Initialize the descriptor."""
        super().__init__(descriptor_size)
        self.subset_size = descriptor_size if subset_size is None else subset_size
        if self.subset_size < 1:
            raise ValueError(f"subset_size must be >= 1, got {self.subset_size}")

    def sample(
        self, key: jax.Array, problem: BBOBProblem, params: BBOBParams
    ) -> SubsetParams:
        """Sample the instance's coordinate subset and its projection.

        Args:
            key: JAX random key.
            problem: The problem being described; `subset_size` must not
                exceed its `num_dims`.
            params: The problem's instance; not read.

        Returns:
            The instance's subset and projection matrix.

        """
        if self.subset_size > problem.num_dims:
            raise ValueError(
                f"subset_size {self.subset_size} exceeds num_dims {problem.num_dims}"
            )
        key_subset, key_matrix = jax.random.split(key)
        subset = jax.random.permutation(key_subset, problem.num_dims)[
            : self.subset_size
        ]
        matrix = jax.random.normal(
            key_matrix, shape=(self.descriptor_size, self.subset_size)
        )
        return SubsetParams(subset=subset, matrix=self._normalize(matrix, problem))

    def evaluate(self, key: jax.Array, x: jax.Array, params: SubsetParams) -> jax.Array:
        """Compute the descriptor: the projection of the subset's coordinates."""
        return params.matrix @ x[params.subset]


class AlignedProjection(RandomProjection):
    """A projection interpolated toward the landscape's own axes.

    The one *coupled* family: its rows are drawn between random unit
    directions and the first rows of the instance's rotation `params.r` -- the
    axes along which the function's conditioning acts. `alignment = 0` is a
    random projection; `alignment = 1` describes the solution in the
    landscape's own rotated coordinates, so moving through behavior space is
    moving directly along the function's structure. The dial sets how directly
    behavior trades against fitness.

    Only meaningful when the problem samples a rotation; with
    `sample_rotation=False` the landscape axes are the coordinate axes.
    Requires `descriptor_size <= num_dims`, since the rotation has only
    `num_dims` rows.

    Args:
        descriptor_size: Dimensionality of the descriptor space.
        alignment: In `[0, 1]`, from random directions to the landscape's own.

    """

    name = "aligned_projection"

    # The coupling dial.
    alignment: float = 1.0

    def __init__(self, descriptor_size: int = 2, alignment: float = 1.0):
        """Initialize the descriptor."""
        if not 0.0 <= alignment <= 1.0:
            raise ValueError(f"alignment must be in [0, 1], got {alignment}")
        super().__init__(descriptor_size)
        self.alignment = alignment

    def sample(
        self, key: jax.Array, problem: BBOBProblem, params: BBOBParams
    ) -> jax.Array:
        """Sample the instance's projection: random rows mixed toward `params.r`.

        Args:
            key: JAX random key.
            problem: The problem being described.
            params: The problem's instance; its rotation `r` is read -- this
                is the declared coupling.

        Returns:
            A `(descriptor_size, num_dims)` matrix.

        """
        if self.descriptor_size > problem.num_dims:
            raise ValueError(
                f"descriptor_size {self.descriptor_size} exceeds "
                f"num_dims {problem.num_dims}"
            )
        random_rows = jax.random.normal(
            key, shape=(self.descriptor_size, problem.num_dims)
        )
        # Unit rows, so the mix is between directions rather than magnitudes:
        # rotation rows are unit norm, Gaussian rows are ~sqrt(num_dims).
        random_rows = random_rows / jnp.linalg.norm(random_rows, axis=-1, keepdims=True)
        mixed = (1.0 - self.alignment) * random_rows + self.alignment * params.r[
            : self.descriptor_size
        ]
        return self._normalize(mixed, problem)


# The descriptor families, keyed by their own name -- one per phenomenon, in
# the module's order from baseline to coupled. The parallel of
# `BBOB_PROBLEMS` and `NOISE_MODELS`: what lets a config name a descriptor
# the way it names a function.
_DESCRIPTORS: tuple[type[Descriptor], ...] = (
    RandomProjection,
    IrregularProjection,
    QuantizedProjection,
    FourierProjection,
    SubsetProjection,
    AlignedProjection,
)

DESCRIPTORS: dict[str, type[Descriptor]] = {
    descriptor.name: descriptor for descriptor in _DESCRIPTORS
}


def sphere_descriptor_optimum(
    d: jax.Array, matrix: jax.Array, params: BBOBParams
) -> tuple[jax.Array, jax.Array]:
    """Compute the exact best fitness at `d`, for Sphere under a linear map.

    Ground truth in closed form: over `{x : matrix @ x = d}`, the sphere
    `|x - x_opt|^2 + f_opt` is minimized by the minimum-norm correction
    `x = x_opt + pinv(matrix) @ (d - matrix @ x_opt)`. Evaluated at a grid's
    cell centers, this is the absolute per-cell reference QD benchmarking has
    lacked -- targets per cell, not only per problem.

    Valid exactly for `Sphere` composed with a descriptor whose map is linear:
    `RandomProjection` and `AlignedProjection` (`matrix` is
    `params.descriptor`), and `SubsetProjection` after scattering its matrix
    to full width::

        full = jnp.zeros((k, num_dims)).at[:, sub.subset].set(sub.matrix)

    The nonlinear families have no closed form, and the quantized family's
    preimages are slabs rather than points; both are out of scope by
    construction, not omission.

    The minimum is taken over all of `R^D`, so the value is a lower bound for
    the box-constrained optimum and exact whenever the returned argmin lies
    inside the box -- which the caller can check, and which holds for every
    cell whose preimage meets the box interior.

    Args:
        d: A descriptor value, shape `(descriptor_size,)`.
        matrix: The descriptor's effective linear map, shape
            `(descriptor_size, num_dims)`.
        params: The sphere's instance.

    Returns:
        The optimal fitness at `d` (with `f_opt` added, comparable to
        `evaluate`), and its argmin, shape `(num_dims,)`.

    """
    correction = jnp.linalg.pinv(matrix) @ (d - matrix @ params.x_opt)
    fitness = jnp.sum(jnp.square(correction)) + params.f_opt
    return fitness, params.x_opt + correction


class QDProblem:
    """A BBOB problem paired with a descriptor.

    The pair is the problem: `sample` draws an instance of each half, and
    `evaluate` returns the fitness and the descriptor together.
    """

    def __init__(self, problem: BBOBProblem, descriptor: Descriptor):
        """Initialize the Quality-Diversity problem.

        Args:
            problem: The function being optimized.
            descriptor: What makes two equally-fit solutions different.

        """
        self.problem = problem
        self.descriptor = descriptor

    @property
    def name(self) -> str:
        """The underlying function's name."""
        return self.problem.name

    @property
    def num_dims(self) -> int:
        """The dimension of the search space."""
        return self.problem.num_dims

    @property
    def x_range(self) -> tuple[float, float]:
        """The range of input variables."""
        return self.problem.x_range

    @property
    def descriptor_size(self) -> int:
        """The dimension of the descriptor space."""
        return self.descriptor.descriptor_size

    @property
    def descriptor_range(self) -> tuple[float, float]:
        """Exact bounds of the descriptor, per component, inside the box.

        Ground truth the benchmark always has; whether an algorithm is told is
        the experiment's choice, exactly as no algorithm reads `x_opt`.
        """
        return self.descriptor.descriptor_range

    def sample(self, key: jax.Array) -> QDParams:
        """Sample an instance of this problem: the function's, and the descriptor's.

        The problem's instance is drawn first and handed to the descriptor,
        so a coupled descriptor can read the landscape it describes.

        Args:
            key: JAX random key.

        Returns:
            The instance's parameters.

        """
        key_problem, key_descriptor = jax.random.split(key)
        problem_params = self.problem.sample(key_problem)
        return QDParams(
            problem=problem_params,
            descriptor=self.descriptor.sample(
                key_descriptor, self.problem, problem_params
            ),
        )

    def evaluate(self, key: jax.Array, x: jax.Array, params: QDParams) -> QDEval:
        """Evaluate the fitness and the descriptor of a solution.

        Args:
            key: JAX random key, consumed by the noise model and by a
                stochastic descriptor.
            x: Input solution, shape `(num_dims,)`.
            params: Instance parameters.

        Returns:
            The evaluation results.

        """
        key_problem, key_descriptor = jax.random.split(key)
        evaluation = self.problem.evaluate(key_problem, x, params.problem)
        descriptor = self.descriptor.evaluate(key_descriptor, x, params.descriptor)
        return QDEval(fitness=evaluation.fitness, descriptor=descriptor)

    def sample_x(self, key: jax.Array) -> jax.Array:
        """Sample a random solution.

        Args:
            key: JAX random key.

        Returns:
            Random solution within `x_range`, shape `(num_dims,)`.

        """
        return self.problem.sample_x(key)
