"""Black-box Optimization Benchmarking Noise Models.

The three BBOB noise models (Gaussian, uniform, Cauchy) follow Hansen, Finck,
Ros, Auger, "Real-Parameter Black-Box Optimization Benchmarking 2009: Noisy
Functions Definitions" (INRIA RR-6869) and are verified against its formulas
and the official `bbobbenchmarks.py`. `additive` is a bbobax extension with
no COCO counterpart. Noise applies to the raw function value only; the boundary
penalty and f_opt are added outside it (`BBOB.evaluate`), as the paper
prescribes.

The paper defines two discrete severities per model (moderate/severe); bbobax
samples the parameters continuously across that span. The uniform model's
alpha is dimension-dependent in the paper (0.49 + 1/D at severe); bbobax
samples the *multiplier* on (0.49 + 1/D), so the paper's severities are the
endpoints at every dimension.
"""

import jax
import jax.numpy as jnp

from .types import IntScalar, NoiseParams


class NoiseModel:
    """Black-box Optimization Benchmarking Noise Models class."""

    # Parameter ranges spanning the paper's moderate..severe severities.
    # uniform_alpha is the multiplier on (0.49 + 1/D), so the paper's levels are
    # the endpoints at every dimension.
    DEFAULT_RANGES: dict[str, tuple[float, float]] = {
        "gaussian_beta": (0.01, 1.0),
        "uniform_alpha": (0.01, 1.0),
        "uniform_beta": (0.01, 1.0),
        "cauchy_alpha": (0.01, 1.0),
        "cauchy_p": (0.05, 0.2),
        "additive_std": (0.0, 0.1),
    }

    def __init__(
        self,
        noise_model_names: tuple[str, ...] = (
            "noiseless",
            "gaussian",
            "uniform",
            "cauchy",
            "additive",
        ),
        noise_ranges: dict[str, tuple[float, float]] | None = None,
        use_stabilization: bool = False,
    ):
        """Initialize the noise model.

        Args:
            noise_model_names: Names of the noise models to sample among.
            noise_ranges: Overrides for parameter ranges, merged over
                `DEFAULT_RANGES`; a partial dictionary is fine.
            use_stabilization: Whether to use noise stabilization.

        """
        unknown = set(noise_model_names) - set(NOISE_MODELS)
        if unknown:
            raise ValueError(
                f"unknown noise models {sorted(unknown)}; "
                f"available: {sorted(NOISE_MODELS)}"
            )
        unknown = set(noise_ranges or {}) - set(self.DEFAULT_RANGES)
        if unknown:
            raise ValueError(
                f"unknown noise ranges {sorted(unknown)}; "
                f"available: {sorted(self.DEFAULT_RANGES)}"
            )

        self.active_models = [
            model for name, model in NOISE_MODELS.items() if name in noise_model_names
        ]
        self.noise_ids = jnp.arange(len(self.active_models))
        self.noise_ranges = {**self.DEFAULT_RANGES, **(noise_ranges or {})}

        # Use noise stabilization close to optimal value
        self.use_stabilization = use_stabilization

    def sample(self, key: jax.Array, num_dims: IntScalar) -> NoiseParams:
        """Sample a noise model and its parameter settings.

        Args:
            key: JAX random key.
            num_dims: Dimensionality of the task instance -- the uniform
                model's alpha is dimension-dependent (0.49 + 1/D).

        """
        (
            key_id,
            key_gaussian,
            key_uniform_1,
            key_uniform_2,
            key_cauchy_1,
            key_cauchy_2,
            key_additive,
        ) = jax.random.split(key, 7)

        noise_id = jax.random.choice(key_id, self.noise_ids)

        # Sampled continuously across the paper's moderate..severe span
        gaussian_beta = jax.random.uniform(
            key_gaussian,
            minval=self.noise_ranges["gaussian_beta"][0],
            maxval=self.noise_ranges["gaussian_beta"][1],
        )

        # Paper: alpha = 0.01(0.49 + 1/D) moderate, (0.49 + 1/D) severe -- the
        # sampled value is the multiplier, the 1/D term is applied here.
        uniform_alpha = jax.random.uniform(
            key_uniform_1,
            minval=self.noise_ranges["uniform_alpha"][0],
            maxval=self.noise_ranges["uniform_alpha"][1],
        ) * (0.49 + 1.0 / num_dims)
        uniform_beta = jax.random.uniform(
            key_uniform_2,
            minval=self.noise_ranges["uniform_beta"][0],
            maxval=self.noise_ranges["uniform_beta"][1],
        )

        cauchy_alpha = jax.random.uniform(
            key_cauchy_1,
            minval=self.noise_ranges["cauchy_alpha"][0],
            maxval=self.noise_ranges["cauchy_alpha"][1],
        )
        cauchy_p = jax.random.uniform(
            key_cauchy_2,
            minval=self.noise_ranges["cauchy_p"][0],
            maxval=self.noise_ranges["cauchy_p"][1],
        )

        additive_std = jax.random.uniform(
            key_additive,
            minval=self.noise_ranges["additive_std"][0],
            maxval=self.noise_ranges["additive_std"][1],
        )

        return NoiseParams(
            noise_id,
            gaussian_beta,
            uniform_alpha,
            uniform_beta,
            cauchy_alpha,
            cauchy_p,
            additive_std,
        )

    def apply(
        self, key: jax.Array, fn_val: jax.Array, noise_params: NoiseParams
    ) -> jax.Array:
        """Apply a noise model given its parameter settings."""
        fn_noise = jax.lax.switch(
            noise_params.noise_id,
            self.active_models,
            key,
            fn_val,
            noise_params,
        )

        if self.use_stabilization:
            fn_noise = stabilize(fn_val, fn_noise)
        return fn_noise


def stabilize(
    fn_val: jax.Array, fn_noise: jax.Array, target_value: float = 1e-08
) -> jax.Array:
    """Leave the value alone once it is within the target precision.

    Below ``target_value`` the undisturbed value is returned, so noise cannot
    stop an algorithm from reaching the target; above it, the paper's
    ``1.01 * target_value`` offset is added.
    """
    return jnp.where(fn_val < target_value, fn_val, fn_noise + 1.01 * target_value)


def noiseless(
    key: jax.Array, fn_val: jax.Array, noise_params: NoiseParams
) -> jax.Array:
    """Return the value unchanged."""
    return fn_val


def gaussian(key: jax.Array, fn_val: jax.Array, noise_params: NoiseParams) -> jax.Array:
    """Apply Gaussian noise."""
    # Moderate noise: beta = 0.01
    # Severe noise: beta = 1
    return fn_val * jnp.exp(
        noise_params.gaussian_beta * jax.random.normal(key, shape=fn_val.shape)
    )


def uniform(key: jax.Array, fn_val: jax.Array, noise_params: NoiseParams) -> jax.Array:
    """Apply uniform noise."""
    # Moderate noise: alpha = 0.01 * (0.49 + 1/D), beta = 0.01
    # Severe noise: alpha = 0.49 + 1/D, beta = 1.0
    key_1, key_2 = jax.random.split(key)
    f_1 = jnp.power(
        jax.random.uniform(key_1, shape=fn_val.shape), noise_params.uniform_beta
    )
    # Epsilon guards division by zero only; the paper sets it to 1e-99 so it
    # never distorts the ratio near the 1e-8 target-precision band.
    f_2 = jnp.power(
        1e9 / (fn_val + 1e-99),
        noise_params.uniform_alpha * jax.random.uniform(key_2, shape=fn_val.shape),
    )
    return fn_val * f_1 * jnp.maximum(1.0, f_2)


def cauchy(key: jax.Array, fn_val: jax.Array, noise_params: NoiseParams) -> jax.Array:
    """Apply Cauchy noise."""
    # Moderate noise: alpha = 0.01, p = 0.05
    # Severe noise: alpha = 1, p = 0.2
    key_1, key_2, key_3 = jax.random.split(key, 3)
    indicator = jax.random.uniform(key_1, shape=fn_val.shape) < noise_params.cauchy_p
    # A standard Cauchy draw is the ratio of two independent normals -- the
    # paper's N(0,1)/|N(0,1)| with epsilon 1e-199 against division by zero.
    # (An earlier version divided by |Uniform(0,1)|, which has ~30% heavier
    # tails and the wrong central law.)
    cauchy = jax.random.normal(key_2, shape=fn_val.shape) / (
        jnp.abs(jax.random.normal(key_3, shape=fn_val.shape)) + 1e-199
    )
    return fn_val + noise_params.cauchy_alpha * jnp.maximum(
        0.0, 1000.0 + indicator * cauchy
    )


def additive(key: jax.Array, fn_val: jax.Array, noise_params: NoiseParams) -> jax.Array:
    """Apply additive noisification."""
    # Moderate noise: std = 0.01
    # Severe noise: std = 1
    return fn_val + noise_params.additive_std * jax.random.normal(
        key, shape=fn_val.shape
    )


# The registry a NoiseModel selects from. Keys match the function names.
NOISE_MODELS = {
    "noiseless": noiseless,
    "gaussian": gaussian,
    "uniform": uniform,
    "cauchy": cauchy,
    "additive": additive,
}
