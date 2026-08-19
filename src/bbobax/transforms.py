"""The transformations BBOB builds its landscapes from.

These are the paper's raw ingredients (Hansen et al., 2010, section 0.2):
every function in every suite is a simple core -- a sphere, a sum of powers --
made hard by composing these transformations onto it. They know nothing about
problems or suites, which is why they live below both.
"""

import jax
import jax.numpy as jnp


def lambda_alpha(condition: float, num_dims: int) -> jax.Array:
    """Conditioning transformation function: `condition ** (0.5 i / (D - 1))`.

    The paper's Lambda^alpha, and COCO's `transform_vars_conditioning`, which
    spells the same thing as `pow(sqrt(condition), i / (D - 1))`.
    """
    exp = 0.5 * jnp.arange(num_dims) / (num_dims - 1)
    return jnp.power(condition, exp)


def transform_osz(element: jax.Array) -> jax.Array:
    """Oscillation transformation function."""
    # Avoid log(0) by substituting 0 with 1 (log(1) = 0), handling the 0 case
    # explicitly with where.
    safe_element = jnp.abs(element) + (element == 0.0)
    x_hat = jnp.where(element == 0.0, 0.0, jnp.log(safe_element))

    c_1 = jnp.where(element > 0.0, 10.0, 5.5)
    c_2 = jnp.where(element > 0.0, 7.9, 3.1)

    return jnp.sign(element) * jnp.exp(
        x_hat + 0.049 * (jnp.sin(c_1 * x_hat) + jnp.sin(c_2 * x_hat))
    )


def transform_asy(x: jax.Array, beta: float) -> jax.Array:
    """Asymmetry transformation function.

    The untaken branch is sanitized before `sqrt`/`power` so that `jax.grad`
    through negative coordinates yields zeros instead of NaN (the standard
    double-`where` guard); values are unchanged.
    """
    num_dims = x.shape[0]

    safe_x = jnp.where(x > 0.0, x, 0.0)
    exp = 1.0 + beta * (jnp.arange(num_dims) / (num_dims - 1)) * jnp.sqrt(safe_x)
    return jnp.where(x > 0.0, jnp.power(safe_x, exp), x)


def f_pen(x: jax.Array) -> jax.Array:
    """Boundary penalty."""
    out = jnp.abs(x) - 5.0
    return jnp.sum(jnp.square(jnp.maximum(0.0, out)))
