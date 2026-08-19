"""The noisy BBOB suite, f101-f130.

Thirty problems built from eight of the noiseless functions under the three
official noise models, at the paper's two severities. Transcribed from the
vendored `bbobbenchmarks.py`, whose `_evalfull` makes the composition explicit:

    fadd = f_opt + boundaryhandling(x)
    ftrue = raw(x - x_opt)          # the core, without the function's own penalty
    fval = noise(ftrue) + fadd

Two things in that are easy to get wrong, and neither is "the noiseless
function plus noise":

- **Boundary handling is uniform here.** Every noisy problem uses
  `defaultboundaryhandling(x, 100)`, replacing whatever factor its noiseless
  counterpart used -- including the several that have no penalty at all. Noise
  makes the outside of the box exploitable in a way it is not without it.
- **Three of the bases are reparameterized.** f116-f118 use an ellipsoid of
  conditioning 1e4, not the noiseless f10's 1e6; f125-f127 scale
  Griewank-Rosenbrock by 1 rather than 10. (The Gallagher variants look
  reparameterized too -- `highpeakcond = 1000 ** .5` where f22 has `1000` --
  but that is f21's own value in the reference's parameterization, which is the
  square root of the paper's. They are f21 unchanged.)

The suite is a table because it *is* a table: a base function, a noise model,
and a severity. `Noisy` composes those three, so nothing here restates a
function that already exists.
"""

import jax

from .bbob import (
    DifferentPowers,
    EllipsoidalRotated,
    Gallagher101Me,
    GriewankRosenbrock,
    Rosenbrock,
    SchaffersF7,
    Sphere,
    StepEllipsoidal,
)
from .noise import Cauchy, Gaussian, NoiseModel, Uniform
from .problem import BBOBParams, BBOBProblem
from .transforms import f_pen


class Noisy(BBOBProblem):
    """A noiseless BBOB function under one of the three noise models.

    A base class rather than a protocol, and a wrapper rather than 30
    subclasses: the suite varies only in which function, which model and which
    severity, so those are constructor arguments and the composition is written
    once.

    Args:
        problem: The noiseless function underneath, already at its dimension.
        noise_model: The model, at a pinned severity -- `Gaussian.severe()`
            and friends, since a published f101-f130 number is at one of the
            paper's two points.
        **kwargs: Passed to `BBOBProblem`.

    """

    name = "noisy"

    # Every noisy problem's boundary handling, replacing the base function's
    # own (`defaultboundaryhandling(x, 100.)` on all three noisy mixins).
    penalty_factor: float = 100.0

    def __init__(self, problem: BBOBProblem, noise_model: NoiseModel, **kwargs):
        """Initialize the noisy problem."""
        super().__init__(num_dims=problem.num_dims, noise_model=noise_model, **kwargs)
        self.problem = problem

    def _sample_x_opt(self, key: jax.Array) -> jax.Array:
        """Draw the optimum the way the underlying function requires."""
        return self.problem._sample_x_opt(key)

    def _value(self, x: jax.Array, params: BBOBParams) -> tuple[jax.Array, jax.Array]:
        # The base's own penalty is discarded, not added to: the noisy suite
        # replaces it with one factor for all thirty.
        value, _ = self.problem._value(x, params)
        return value, self.penalty_factor * f_pen(x)


class _Ellipsoid1e4(EllipsoidalRotated):
    """The ellipsoid f116-f118 use: conditioning 1e4, where f10 has 1e6."""

    name = "ellipsoidal_rotated_1e4"

    condition: float = 1e4


class _GriewankRosenbrock1(GriewankRosenbrock):
    """The composition f125-f127 use: scaled by 1, where f19 uses 10."""

    name = "griewank_rosenbrock_1"

    facftrue: float = 1.0


# f101-f130, from `bbobbenchmarks.py`. Each entry is the base function and the
# noise model at the severity that problem is defined at.
_SUITE: tuple[tuple[int, type[BBOBProblem], str], ...] = (
    (101, Sphere, "gaussian_moderate"),
    (102, Sphere, "uniform_moderate"),
    (103, Sphere, "cauchy_moderate"),
    (104, Rosenbrock, "gaussian_moderate"),
    (105, Rosenbrock, "uniform_moderate"),
    (106, Rosenbrock, "cauchy_moderate"),
    (107, Sphere, "gaussian_severe"),
    (108, Sphere, "uniform_severe"),
    (109, Sphere, "cauchy_severe"),
    (110, Rosenbrock, "gaussian_severe"),
    (111, Rosenbrock, "uniform_severe"),
    (112, Rosenbrock, "cauchy_severe"),
    (113, StepEllipsoidal, "gaussian_severe"),
    (114, StepEllipsoidal, "uniform_severe"),
    (115, StepEllipsoidal, "cauchy_severe"),
    (116, _Ellipsoid1e4, "gaussian_severe"),
    (117, _Ellipsoid1e4, "uniform_severe"),
    (118, _Ellipsoid1e4, "cauchy_severe"),
    (119, DifferentPowers, "gaussian_severe"),
    (120, DifferentPowers, "uniform_severe"),
    (121, DifferentPowers, "cauchy_severe"),
    (122, SchaffersF7, "gaussian_severe"),
    (123, SchaffersF7, "uniform_severe"),
    (124, SchaffersF7, "cauchy_severe"),
    (125, _GriewankRosenbrock1, "gaussian_severe"),
    (126, _GriewankRosenbrock1, "uniform_severe"),
    (127, _GriewankRosenbrock1, "cauchy_severe"),
    (128, Gallagher101Me, "gaussian_severe"),
    (129, Gallagher101Me, "uniform_severe"),
    (130, Gallagher101Me, "cauchy_severe"),
)

_NOISE_MODELS = {
    "gaussian_moderate": Gaussian.moderate,
    "gaussian_severe": Gaussian.severe,
    "uniform_moderate": Uniform.moderate,
    "uniform_severe": Uniform.severe,
    "cauchy_moderate": Cauchy.moderate,
    "cauchy_severe": Cauchy.severe,
}

# The suite, keyed as `f101` .. `f130`. Numbers rather than names because that
# is how the noisy suite is cited: a paper reports f107, not "severely
# Gaussian-noised sphere".
NOISY_PROBLEMS: dict[str, tuple[type[BBOBProblem], str]] = {
    f"f{fid}": (problem_class, severity) for fid, problem_class, severity in _SUITE
}


def noisy_suite(
    names: list[str] | None = None, num_dims: int = 10, **kwargs
) -> dict[str, Noisy]:
    """Build the noisy BBOB suite, f101-f130.

    Args:
        names: Which problems to include, as `f101` .. `f130`; defaults to all
            thirty in order.
        num_dims: The dimension every problem is built at.
        **kwargs: Passed to every problem.

    Returns:
        A mapping from `f101`-style name to problem.

    Raises:
        KeyError: If `names` contains something outside f101-f130.

    """
    names = list(NOISY_PROBLEMS) if names is None else names

    unknown = [name for name in names if name not in NOISY_PROBLEMS]
    if unknown:
        raise KeyError(
            f"not noisy BBOB problems: {unknown}; available: {sorted(NOISY_PROBLEMS)}"
        )

    suite = {}
    for name in names:
        problem_class, severity = NOISY_PROBLEMS[name]
        base = problem_class(num_dims=num_dims, **kwargs)
        suite[name] = Noisy(base, _NOISE_MODELS[severity](), **kwargs)
    return suite
