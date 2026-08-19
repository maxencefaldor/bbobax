"""Black-box Optimization Benchmarking in JAX."""

from .bbob import (
    BBOB_PROBLEMS,
    AttractiveSector,
    BentCigar,
    BuecheRastrigin,
    DifferentPowers,
    Discus,
    Ellipsoidal,
    EllipsoidalRotated,
    Gallagher21Hi,
    Gallagher101Me,
    GriewankRosenbrock,
    Katsuura,
    LinearSlope,
    Lunacek,
    Rastrigin,
    RastriginRotated,
    Rosenbrock,
    RosenbrockRotated,
    SchaffersF7,
    SchaffersF7IllConditioned,
    Schwefel,
    SharpRidge,
    Sphere,
    StepEllipsoidal,
    Weierstrass,
    suite,
)
from .many_affine import ManyAffine
from .noise import (
    NOISE_MODELS,
    Cauchy,
    Gaussian,
    Mixture,
    Noiseless,
    NoiseModel,
    Uniform,
)
from .noisy import NOISY_PROBLEMS, Noisy, noisy_suite
from .problem import DIMENSIONS, BBOBEval, BBOBParams, BBOBProblem
from .qd import Descriptor, QDEval, QDParams, QDProblem, RandomProjection
from .transforms import f_pen, lambda_alpha, transform_asy, transform_osz

__all__ = [
    # The contract, and the standard suite built on it
    "BBOBProblem",
    "BBOB_PROBLEMS",
    "DIMENSIONS",
    "suite",
    # The 24, in canonical f1-f24 order
    "Sphere",
    "Ellipsoidal",
    "Rastrigin",
    "BuecheRastrigin",
    "LinearSlope",
    "AttractiveSector",
    "StepEllipsoidal",
    "Rosenbrock",
    "RosenbrockRotated",
    "EllipsoidalRotated",
    "Discus",
    "BentCigar",
    "SharpRidge",
    "DifferentPowers",
    "RastriginRotated",
    "Weierstrass",
    "SchaffersF7",
    "SchaffersF7IllConditioned",
    "GriewankRosenbrock",
    "Schwefel",
    "Gallagher101Me",
    "Gallagher21Hi",
    "Katsuura",
    "Lunacek",
    # Suites built from the 24, rather than one of them
    "Noisy",
    "NOISY_PROBLEMS",
    "noisy_suite",
    "ManyAffine",
    # Quality-Diversity, composed onto a problem
    "QDProblem",
    "Descriptor",
    "RandomProjection",
    # The transformations functions are built from, for building your own
    "lambda_alpha",
    "transform_osz",
    "transform_asy",
    "f_pen",
    # Noise models, held by a problem rather than switched on
    "NoiseModel",
    "NOISE_MODELS",
    "Noiseless",
    "Gaussian",
    "Uniform",
    "Cauchy",
    "Mixture",
    # The data types
    "BBOBParams",
    "BBOBEval",
    "QDParams",
    "QDEval",
]
