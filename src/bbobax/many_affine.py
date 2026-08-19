"""Many-Affine BBOB: a problem built by combining all 24.

MA-BBOB (Vermetten, Ye, Back, Doerr) draws a sparse weight vector over the 24
functions and combines their values in log space, giving a continuous space of
problems rather than a set of 24. That is what makes it worth having here:
meta-learning over `bbob_suite()` samples 24 atoms, while meta-learning over
`ManyAffine` samples a space.

Transcribed from the reference implementation, IOHexperimenter's
`ManyAffine` (`include/ioh/problem/bbob/many_affine.hpp`), whose combination is

    for each of the 24:
        x0 = x + x_opt_i - x_opt              # each component sees its own optimum
        f0 = clip(f_i(x0) - f_opt_i, 1e-12, 1e20)
        result += w_i * (log10(f0) + 8) / scale_i
    value = 10 ** (10 * result - 8)

Two things follow from the shift. Every component is evaluated at *its own*
optimum when `x` is the combined problem's optimum, so the combination is
anchored there; and `x0` can leave the box even when `x` does not, so each
component's boundary penalty is live and is folded into the log-space sum.

`f(x_opt)` is therefore not 0 -- the reference evaluates it in its constructor
and stores it as the optimum's value. bbobax's convention is that `_value`
returns 0 at the optimum and `evaluate` adds `f_opt`, so that constant is
subtracted here. It is available in closed form: at the optimum every
component contributes exactly `clip(0, 1e-12, ...) -> 1e-12`, hence
`(log10(1e-12) + 8) / scale_i = -4 / scale_i`, needing no extra evaluation.
Add it back to compare against a reference value.
"""

from typing import Any

import jax
import jax.numpy as jnp

from .bbob import BBOB_PROBLEMS
from .problem import BBOBParams, BBOBProblem


class ManyAffine(BBOBProblem):
    """A weighted affine combination of all 24 BBOB functions, in log space.

    The instance is the weight vector and the 24 component instances, all drawn
    from `params.key` rather than stored -- the same choice the Gallagher
    functions make for their peak layouts, and for the same reason: it is
    instance structure that is cheaper to derive than to carry, and it keeps
    `BBOBParams` the one parameter type every problem here uses, with no
    subclass narrowing `_value`'s signature.

    Deriving it costs nothing at steady state, which was worth measuring rather
    than assuming: the derivation depends on `params` and not on `x`, so XLA
    hoists it out of a `vmap` over solutions. A batch through `ManyAffine`
    takes 0.92x the time of evaluating the 24 separately at D = 10, and 0.78x
    at D = 40 -- at or below the floor of "24 function evaluations", because
    the components fuse into one program.

    Args:
        num_dims: The problem dimension.
        **kwargs: Passed to `BBOBProblem` and to all 24 components.

    """

    name = "many_affine"

    # The reference's `default_scales`, in canonical f1-f24 order. They put the
    # 24 functions' value ranges on comparable footing before the weights are
    # applied; without them the combination is dominated by whichever component
    # happens to have the largest dynamic range.
    scales: tuple[float, ...] = (
        11.0, 17.5, 12.3, 12.6, 11.5, 15.3, 12.1, 15.3,
        15.2, 17.4, 13.4, 20.4, 12.9, 10.4, 12.3, 10.3,
        9.8, 10.6, 10.0, 14.7, 10.7, 10.8, 9.0, 12.1,
    )  # fmt: skip

    # Weights below this are dropped, and the two largest are raised to it, so
    # at least two functions always combine. The rest are set to zero and what
    # survives is normalized to sum to 1.
    weight_floor: float = 0.85

    # Component values are clipped into this range before the logarithm, which
    # is what makes `log10` safe at a component's own optimum.
    value_range: tuple[float, float] = (1e-12, 1e20)

    def __init__(self, num_dims: int = 10, **kwargs: Any):
        """Initialize the problem, and the 24 it is built from."""
        super().__init__(num_dims=num_dims, **kwargs)
        self.components = tuple(
            problem_class(num_dims=num_dims, **kwargs)
            for problem_class in BBOB_PROBLEMS.values()
        )

    def _sample_weights(self, key: jax.Array) -> jax.Array:
        """Draw the sparse weight vector over the 24 functions.

        Args:
            key: JAX random key.

        Returns:
            Non-negative weights summing to 1, shape `(24,)`, with at least two
            non-zero.

        """
        weights = jax.random.uniform(key, shape=(len(self.components),))

        # Raise the two largest to the floor, so two functions always survive.
        _, largest = jax.lax.top_k(weights, 2)
        weights = weights.at[largest].max(self.weight_floor)

        weights = jnp.where(weights >= self.weight_floor, weights, 0.0)
        return weights / jnp.sum(weights)

    def _combine(self, values: jax.Array, weights: jax.Array) -> jax.Array:
        """Combine the 24 component values, in log space.

        Args:
            values: Each component's value above its own optimum, shape `(24,)`.
            weights: The instance's weights, shape `(24,)`.

        Returns:
            The combined value.

        """
        low, high = self.value_range
        scaled = (jnp.log10(jnp.clip(values, low, high)) + 8.0) / jnp.asarray(
            self.scales
        )
        return jnp.power(10.0, 10.0 * jnp.sum(weights * scaled) - 8.0)

    def _value(self, x: jax.Array, params: BBOBParams) -> tuple[jax.Array, jax.Array]:
        key_weights, key_components = jax.random.split(params.key)
        weights = self._sample_weights(key_weights)
        keys = jax.random.split(key_components, len(self.components))

        values = []
        for component, key in zip(self.components, keys):
            component_params = component.sample(key)
            # Each component is asked at its own optimum when x is ours, which
            # is what anchors the combination. The shift can leave the box, so
            # the component's boundary penalty is part of its value here.
            x0 = x + component_params.x_opt - params.x_opt
            value, penalty = component._value(x0, component_params)
            values.append(value + penalty)

        # The reference's optimum value, in closed form: every component sits
        # at its own optimum, so each contributes the clipped floor.
        low, _ = self.value_range
        at_optimum = jnp.power(
            10.0,
            10.0 * jnp.sum(weights * (jnp.log10(low) + 8.0) / jnp.asarray(self.scales))
            - 8.0,
        )

        value = self._combine(jnp.stack(values), weights) - at_optimum
        # The boundary penalty is inside the combination rather than beside it:
        # the components carry their own, and the log-space sum is non-linear,
        # so there is nothing separable to hand back.
        return value, jnp.array(0.0)
