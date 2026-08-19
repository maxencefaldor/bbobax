"""Black-box Optimization Benchmarking Noise Models.

The three BBOB noise models (Gaussian, uniform, Cauchy) follow Hansen, Finck,
Ros, Auger, "Real-Parameter Black-Box Optimization Benchmarking 2009: Noisy
Functions Definitions" (INRIA RR-6869) and are verified against the official
`bbobbenchmarks.py` formulas in `tests/test_noise.py`.

A noise model has the same shape of contract as a problem -- `sample` draws the
instance's settings, `apply` disturbs one value -- and a problem holds one,
statically. Like the 24 functions, a model is chosen by holding the object
rather than by switching on an index, so nothing evaluates the branches it did
not want. `Mixture` is the one exception, and pays for itself in the open.

Two things to know:

- Noise applies to the raw function value only; the boundary penalty and
  f_opt are added outside it (`BBOBProblem.evaluate`), as the paper prescribes.
- **Stabilization is part of the three official models, not an option.** Each
  of `fGauss`, `fUniform` and `fCauchy` ends by adding `1.01 * 1e-8` and
  returning the *undisturbed* value below that tolerance, unconditionally, so
  noise can never stop an algorithm reaching the target precision.
  `Noiseless` has none, as official's noise-free functions do not.

**Severity is continuous here, and that is a deviation.** The paper defines two
discrete severities per model (moderate/severe) and the noisy suite pins each
function to one of them; bbobax samples the settings continuously across that
span instead, so difficulty varies per instance for free and a fixed model is
not a fixed difficulty. That is what meta-learning wants and what published
noisy-BBOB numbers are *not*: nothing here is comparable to a published
f101-f130 result unless the severity is pinned. The paper's two points are
therefore first-class -- `Gaussian.severe()`, `Cauchy.moderate()` -- and pin
the settings exactly.

Only the model family stays fixed on a problem. When one batch genuinely has to
mix families, `Mixture` restores that and states its cost.
"""

from typing import Any, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
from flax.struct import dataclass

# The target precision BBOB measures against, and the floor the official noise
# models leave undisturbed (`bbobbenchmarks.py`: `tol = 1e-8`).
TARGET_PRECISION = 1e-8


def _epsilon(value: jax.Array) -> float:
    """Return the smallest positive number of `value`'s dtype.

    The uniform and Cauchy models divide by a quantity that can be zero, and
    the paper guards each with a literal (1e-99 and 1e-199). Both are exactly
    0.0 in float32 -- JAX's default -- so the literal guards nothing unless
    `jax_enable_x64` happens to be on. The smallest positive normal of the
    working dtype guards in either precision, and in float64 it is 2e-308
    against the paper's 1e-99: a difference no reachable value can resolve.
    """
    return float(jnp.finfo(value.dtype).tiny)


def stabilize(value: jax.Array, noisy: jax.Array) -> jax.Array:
    """Leave a value alone once it is within the target precision.

    The official models spell this as `fval += 1.01 * tol` followed by
    `fval[ftrue < tol] = ftrue[...]`; this is the same thing branchlessly.

    Args:
        value: The raw function value.
        noisy: The disturbed value.

    Returns:
        The stabilized value.

    """
    return jnp.where(value < TARGET_PRECISION, value, noisy + 1.01 * TARGET_PRECISION)


@runtime_checkable
class NoiseModel(Protocol):
    """How a raw function value is disturbed.

    A protocol rather than a base class, exactly like `Descriptor`: the models
    share no implementation, only a contract, and each owns its own parameter
    type. `sample` draws the instance's settings and `apply` disturbs one value.
    """

    # The model's name, and its key in `NOISE_MODELS`.
    name: str

    def sample(self, key: jax.Array, num_dims: int) -> Any:
        """Sample this model's settings for one instance.

        Args:
            key: JAX random key.
            num_dims: Dimensionality of the problem -- the uniform model's
                alpha is dimension-dependent.

        Returns:
            The instance's noise parameters.

        """
        ...

    def apply(self, key: jax.Array, value: jax.Array, params: Any) -> jax.Array:
        """Disturb one raw function value.

        Args:
            key: JAX random key.
            value: The raw function value.
            params: The instance's noise parameters.

        Returns:
            The disturbed value.

        """
        ...


# Each model owns its parameter type, holding only what it draws -- the same
# rule a descriptor follows for `QDParams.descriptor`.


@dataclass
class NoiselessParams:
    """`Noiseless` draws nothing."""


class Noiseless:
    """No noise: the value is returned untouched.

    Official BBOB's noise-free functions do exactly this, with no
    stabilization -- so the noiseless suite is bit-for-bit the raw functions.
    """

    name = "noiseless"

    def sample(self, key: jax.Array, num_dims: int) -> NoiselessParams:
        """Sample nothing."""
        return NoiselessParams()

    def apply(
        self, key: jax.Array, value: jax.Array, params: NoiselessParams
    ) -> jax.Array:
        """Return the value unchanged."""
        return value


@dataclass
class GaussianParams:
    """Settings of the Gaussian model."""

    beta: jax.Array


class Gaussian:
    """Multiplicative log-normal noise: `f * exp(beta * N(0, 1))`.

    Official `fGauss` (paper 3.1). Moderate is beta = 0.01, severe beta = 1.
    """

    name = "gaussian"

    def __init__(self, beta_range: tuple[float, float] = (0.01, 1.0)):
        """Initialize the model.

        Args:
            beta_range: Range beta is drawn from, moderate to severe.

        """
        self.beta_range = beta_range

    @classmethod
    def moderate(cls) -> "Gaussian":
        """Pin the paper's moderate severity: beta = 0.01."""
        return cls(beta_range=(0.01, 0.01))

    @classmethod
    def severe(cls) -> "Gaussian":
        """Pin the paper's severe severity: beta = 1."""
        return cls(beta_range=(1.0, 1.0))

    def sample(self, key: jax.Array, num_dims: int) -> GaussianParams:
        """Sample beta."""
        return GaussianParams(
            beta=jax.random.uniform(
                key, minval=self.beta_range[0], maxval=self.beta_range[1]
            )
        )

    def apply(
        self, key: jax.Array, value: jax.Array, params: GaussianParams
    ) -> jax.Array:
        """Disturb one value."""
        noisy = value * jnp.exp(params.beta * jax.random.normal(key, shape=value.shape))
        return stabilize(value, noisy)


@dataclass
class UniformParams:
    """Settings of the uniform model."""

    alpha: jax.Array
    beta: jax.Array


class Uniform:
    """Uniform-powered noise with a heavy low-value tail.

    Official `fUniform` (paper 3.2):
    `U^beta * f * max(1, (1e9 / (f + eps)) ** (alpha * U))`, with two
    independent uniform draws.

    Alpha is dimension-dependent -- the paper's `0.01 (0.49 + 1/D)` moderate
    and `(0.49 + 1/D)` severe -- so what is drawn here is the *multiplier* on
    `(0.49 + 1/D)`, which puts the paper's two severities at the endpoints of
    the range at every dimension.
    """

    name = "uniform"

    def __init__(
        self,
        alpha_range: tuple[float, float] = (0.01, 1.0),
        beta_range: tuple[float, float] = (0.01, 1.0),
    ):
        """Initialize the model.

        Args:
            alpha_range: Range the multiplier on `(0.49 + 1/D)` is drawn from.
            beta_range: Range beta is drawn from, moderate to severe.

        """
        self.alpha_range = alpha_range
        self.beta_range = beta_range

    @classmethod
    def moderate(cls) -> "Uniform":
        """Pin the paper's moderate severity: alpha = 0.01(0.49 + 1/D), beta = 0.01."""
        return cls(alpha_range=(0.01, 0.01), beta_range=(0.01, 0.01))

    @classmethod
    def severe(cls) -> "Uniform":
        """Pin the paper's severe severity: alpha = 0.49 + 1/D, beta = 1."""
        return cls(alpha_range=(1.0, 1.0), beta_range=(1.0, 1.0))

    def sample(self, key: jax.Array, num_dims: int) -> UniformParams:
        """Sample alpha and beta."""
        key_alpha, key_beta = jax.random.split(key)
        alpha = jax.random.uniform(
            key_alpha, minval=self.alpha_range[0], maxval=self.alpha_range[1]
        ) * (0.49 + 1.0 / num_dims)
        beta = jax.random.uniform(
            key_beta, minval=self.beta_range[0], maxval=self.beta_range[1]
        )
        return UniformParams(alpha=alpha, beta=beta)

    def apply(
        self, key: jax.Array, value: jax.Array, params: UniformParams
    ) -> jax.Array:
        """Disturb one value."""
        key_beta, key_alpha = jax.random.split(key)
        scale = jnp.power(jax.random.uniform(key_beta, shape=value.shape), params.beta)
        blowup = jnp.power(
            1e9 / (value + _epsilon(value)),
            params.alpha * jax.random.uniform(key_alpha, shape=value.shape),
        )
        noisy = value * scale * jnp.maximum(1.0, blowup)
        return stabilize(value, noisy)


@dataclass
class CauchyParams:
    """Settings of the Cauchy model."""

    alpha: jax.Array
    p: jax.Array


class Cauchy:
    """Additive Cauchy outliers, fired with probability `p`.

    Official `fCauchy` (paper 3.3):
    `f + alpha * max(0, 1e3 + 1{U < p} * N / |N|)`. Moderate is
    alpha = 0.01, p = 0.05; severe alpha = 1, p = 0.2.
    """

    name = "cauchy"

    def __init__(
        self,
        alpha_range: tuple[float, float] = (0.01, 1.0),
        p_range: tuple[float, float] = (0.05, 0.2),
    ):
        """Initialize the model.

        Args:
            alpha_range: Range alpha is drawn from, moderate to severe.
            p_range: Range the outlier probability is drawn from.

        """
        self.alpha_range = alpha_range
        self.p_range = p_range

    @classmethod
    def moderate(cls) -> "Cauchy":
        """Pin the paper's moderate severity: alpha = 0.01, p = 0.05."""
        return cls(alpha_range=(0.01, 0.01), p_range=(0.05, 0.05))

    @classmethod
    def severe(cls) -> "Cauchy":
        """Pin the paper's severe severity: alpha = 1, p = 0.2."""
        return cls(alpha_range=(1.0, 1.0), p_range=(0.2, 0.2))

    def sample(self, key: jax.Array, num_dims: int) -> CauchyParams:
        """Sample alpha and p."""
        key_alpha, key_p = jax.random.split(key)
        alpha = jax.random.uniform(
            key_alpha, minval=self.alpha_range[0], maxval=self.alpha_range[1]
        )
        p = jax.random.uniform(key_p, minval=self.p_range[0], maxval=self.p_range[1])
        return CauchyParams(alpha=alpha, p=p)

    def apply(
        self, key: jax.Array, value: jax.Array, params: CauchyParams
    ) -> jax.Array:
        """Disturb one value."""
        key_fire, key_num, key_den = jax.random.split(key, 3)
        fires = jax.random.uniform(key_fire, shape=value.shape) < params.p
        # A standard Cauchy draw is the ratio of two independent normals -- the
        # paper's N(0,1)/|N(0,1)|. (An earlier version divided by
        # |Uniform(0,1)|, which has ~30% heavier tails and the wrong law.)
        cauchy = jax.random.normal(key_num, shape=value.shape) / (
            jnp.abs(jax.random.normal(key_den, shape=value.shape)) + _epsilon(value)
        )
        noisy = value + params.alpha * jnp.maximum(0.0, 1000.0 + fires * cauchy)
        return stabilize(value, noisy)


@dataclass
class MixtureParams:
    """Settings of `Mixture`: which model fired, and every model's settings."""

    model_id: jax.Array
    models: tuple


class Mixture:
    """Draw a noise model per instance, alongside the instance itself.

    Every other model here is chosen by holding it, so nothing dispatches. That
    is the right default, but it makes the *model family* a property of the
    problem rather than of the instance -- and meta-learning sometimes wants a
    batch of instances that disagree about which noise they carry. This
    restores exactly that, and is the only place in bbobax that pays for it:

        models = Mixture(Gaussian(), Uniform(), Cauchy())
        problem = Sphere(num_dims=10, noise_model=models)

    The cost is real and worth stating. `sample` draws every model's settings
    and `apply` selects with `lax.switch`, which under `vmap` over a varying
    `model_id` evaluates *every* branch for every solution. With three models
    that is three noise models computed per evaluation. Prefer looping over
    models in Python -- as one loops over functions and dimensions -- and reach
    for this only when one batch genuinely has to mix them.

    Note that severity already varies continuously per instance inside a single
    model (`Gaussian` draws its beta across the paper's moderate-to-severe
    span), so a fixed model is not a fixed difficulty. Only the family is
    fixed, and that is what this changes.
    """

    name = "mixture"

    def __init__(self, *models: NoiseModel):
        """Initialize the mixture.

        Args:
            *models: The models to draw among, uniformly.

        Raises:
            ValueError: If no model is given.

        """
        if not models:
            raise ValueError("Mixture needs at least one model")
        self.models = models

    def sample(self, key: jax.Array, num_dims: int) -> MixtureParams:
        """Draw which model this instance carries, and every model's settings."""
        key_id, *keys = jax.random.split(key, len(self.models) + 1)
        return MixtureParams(
            model_id=jax.random.choice(key_id, len(self.models)),
            models=tuple(
                model.sample(k, num_dims) for model, k in zip(self.models, keys)
            ),
        )

    def apply(
        self, key: jax.Array, value: jax.Array, params: MixtureParams
    ) -> jax.Array:
        """Disturb one value with the model this instance drew."""
        # Default arguments bind per iteration; a bare closure over the loop
        # variable would give every branch the last model.
        branches = [
            lambda model=model, model_params=model_params: model.apply(
                key, value, model_params
            )
            for model, model_params in zip(self.models, params.models)
        ]
        return jax.lax.switch(params.model_id, branches)


# The noise models, keyed by their own name. `Mixture` is deliberately absent:
# it is a combinator over models rather than a model, and has no bare form.
#
# Left to inference rather than annotated `dict[str, type[NoiseModel]]`:
# `NoiseModel` is a protocol, and a protocol's *data* members (here `name`) are
# not reachable through `type[NoiseModel]`. Inference keeps each concrete class,
# so the registry stays introspectable and every entry satisfies the protocol.
_MODELS = (Noiseless, Gaussian, Uniform, Cauchy)

NOISE_MODELS = {model.name: model for model in _MODELS}
