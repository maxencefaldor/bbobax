"""Quality-Diversity on top of a BBOB problem.

The Quality-Diversity extension is bbobax's own; COCO has no descriptor notion.

A descriptor is *orthogonal* to a function: any of the 24 can be paired with
any descriptor, so a QD problem is composed rather than subclassed --

    problem = QDProblem(Rastrigin(num_dims=10), RandomProjection(descriptor_size=2))

Subclassing would need one QD class per function, and the combinatorics get
worse with every descriptor family added. Composition also keeps each half
testable on its own, and lets a descriptor be reused across dimensions: the
descriptor is told `num_dims` when it draws its instance data, so the same
object serves a whole meta-learning loop over `DIMENSIONS`.
"""

from typing import Any, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
from flax.struct import dataclass

from .problem import BBOBParams, BBOBProblem


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
    the descriptor needs, `evaluate` scores **one** solution and callers `vmap`
    it to cover a batch.

    A protocol rather than a base class, for the reason `BBOBProblem` spells
    out: descriptors share a contract, not an implementation, and satisfying
    one should not require importing anything from bbobax.
    """

    # Dimensionality of the descriptor space.
    descriptor_size: int

    def sample(self, key: jax.Array, num_dims: int) -> Any:
        """Sample this descriptor's instance data.

        Args:
            key: JAX random key.
            num_dims: Dimensionality of the problem being described.

        Returns:
            The instance's descriptor parameters.

        """
        ...

    def evaluate(self, x: jax.Array, params: Any) -> jax.Array:
        """Compute the descriptor of a solution.

        Args:
            x: Input solution, shape `(num_dims,)`.
            params: Instance descriptor parameters.

        Returns:
            The descriptor, shape `(descriptor_size,)`.

        """
        ...


class RandomProjection:
    """A Gaussian random projection of the solution.

    Entries are `N(0, 1) / sqrt(descriptor_size)`, so the projection preserves
    expected squared norms.

    This descriptor is smooth and Lipschitz: a small change in `x` gives a
    small change in the descriptor, and the reachable set is a linear image of
    the box. Real-world QD is frequently neither, so treat it as the baseline
    rather than as representative.
    """

    def __init__(self, descriptor_size: int = 2):
        """Initialize the descriptor.

        Args:
            descriptor_size: Dimensionality of the descriptor space.

        """
        self.descriptor_size = descriptor_size

    def sample(self, key: jax.Array, num_dims: int) -> jax.Array:
        """Sample the instance's projection matrix.

        Args:
            key: JAX random key.
            num_dims: Dimensionality of the problem being described.

        Returns:
            A `(descriptor_size, num_dims)` matrix.

        """
        return jax.random.normal(
            key, shape=(self.descriptor_size, num_dims)
        ) / jnp.sqrt(self.descriptor_size)

    def evaluate(self, x: jax.Array, params: jax.Array) -> jax.Array:
        """Compute the descriptor of a solution: the projection of `x`.

        Args:
            x: Input solution, shape `(num_dims,)`.
            params: The instance's projection matrix.

        Returns:
            The descriptor, shape `(descriptor_size,)`.

        """
        return params @ x


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

    def sample(self, key: jax.Array) -> QDParams:
        """Sample an instance of this problem: the function's, and the descriptor's.

        Args:
            key: JAX random key.

        Returns:
            The instance's parameters.

        """
        key_problem, key_descriptor = jax.random.split(key)
        return QDParams(
            problem=self.problem.sample(key_problem),
            descriptor=self.descriptor.sample(key_descriptor, self.num_dims),
        )

    def evaluate(self, key: jax.Array, x: jax.Array, params: QDParams) -> QDEval:
        """Evaluate the fitness and the descriptor of a solution.

        Args:
            key: JAX random key, consumed by the noise model.
            x: Input solution, shape `(num_dims,)`.
            params: Instance parameters.

        Returns:
            The evaluation results.

        """
        evaluation = self.problem.evaluate(key, x, params.problem)
        descriptor = self.descriptor.evaluate(x, params.descriptor)
        return QDEval(fitness=evaluation.fitness, descriptor=descriptor)

    def sample_x(self, key: jax.Array) -> jax.Array:
        """Sample a random solution.

        Args:
            key: JAX random key.

        Returns:
            Random solution within `x_range`, shape `(num_dims,)`.

        """
        return self.problem.sample_x(key)
