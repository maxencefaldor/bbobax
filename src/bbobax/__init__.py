"""Black-box Optimization Benchmarking in JAX."""

from .functions import (
    BBOB_PROBLEMS,
    DIMENSIONS,
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
from .noise import NoiseModel
from .problem import BBOBProblem
from .qd import Descriptor, QDProblem, RandomProjection
from .types import BBOBEval, BBOBParams, NoiseParams, QDEval, QDParams

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
    # Quality-Diversity, composed onto a problem
    "QDProblem",
    "Descriptor",
    "RandomProjection",
    # Noise, and the data types
    "NoiseModel",
    "BBOBParams",
    "BBOBEval",
    "QDParams",
    "QDEval",
    "NoiseParams",
]
