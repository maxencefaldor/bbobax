"""Black-box Optimization Benchmarking Functions Definitions."""

import jax
import jax.numpy as jnp

from .types import BBOBParams, BBOBState


def _lambda_alpha_vector(alpha: float, max_num_dims: int, num_dims: int) -> jax.Array:
    """Masked lambda alpha vector."""
    mask = jnp.arange(max_num_dims) < num_dims

    exp = (
        jnp.where(num_dims > 1, 0.5 * jnp.arange(max_num_dims) / (num_dims - 1), 0.5)
        * mask
    )
    return jnp.power(alpha, exp)


def lambda_alpha(alpha: float, max_num_dims: int, num_dims: int) -> jax.Array:
    """Masked lambda alpha matrix."""
    return jnp.diag(_lambda_alpha_vector(alpha, max_num_dims, num_dims))


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


def transform_asy(x: jax.Array, beta: float, num_dims: int) -> jax.Array:
    """Asymmetry transformation function."""
    max_num_dims = x.shape[0]
    mask = jnp.arange(max_num_dims) < num_dims

    exp = (
        1.0
        + beta
        * jnp.where(num_dims > 1, jnp.arange(max_num_dims) / (num_dims - 1), 1.0)
        * jnp.sqrt(x)
        * mask
    )
    return jnp.where(x > 0.0, jnp.power(x, exp), x)


def f_pen(x: jax.Array, num_dims: int) -> jax.Array:
    """Boundary penalty."""
    max_num_dims = x.shape[0]
    mask = jnp.arange(max_num_dims) < num_dims

    out = jnp.abs(x) - 5.0
    return jnp.sum(jnp.square(jnp.maximum(0.0, out * mask)))


def sphere(
    x: jax.Array, state: BBOBState, params: BBOBParams
) -> tuple[jax.Array, jax.Array]:
    """Sphere Function (Hansen et al., 2010, p. 5)."""
    max_num_dims = x.shape[0]
    mask = jnp.arange(max_num_dims) < params.num_dims

    z = x - params.x_opt

    out = jnp.square(z)
    return jnp.sum(out * mask), jnp.array(0.0)


def ellipsoidal(
    x: jax.Array, state: BBOBState, params: BBOBParams
) -> tuple[jax.Array, jax.Array]:
    """Ellipsoidal Function (Hansen et al., 2010, p. 10)."""
    max_num_dims = x.shape[0]
    mask = jnp.arange(max_num_dims) < params.num_dims

    z = transform_osz(x - params.x_opt)

    exp = (
        jnp.where(
            params.num_dims > 1,
            6.0 * jnp.arange(max_num_dims) / (params.num_dims - 1),
            6.0,
        )
        * mask
    )
    out = jnp.power(10.0, exp) * jnp.square(z)
    return jnp.sum(out * mask), jnp.array(0.0)


def rastrigin(
    x: jax.Array, state: BBOBState, params: BBOBParams
) -> tuple[jax.Array, jax.Array]:
    """Rastrigin Function (Hansen et al., 2010, p. 15)."""
    max_num_dims = x.shape[0]
    mask = jnp.arange(max_num_dims) < params.num_dims

    z = transform_asy(transform_osz(x - params.x_opt), 0.2, params.num_dims)
    z = _lambda_alpha_vector(10.0, max_num_dims, params.num_dims) * z

    out_1 = jnp.cos(2 * jnp.pi * z)
    out_2 = jnp.square(z)

    return 10 * (params.num_dims - jnp.sum(out_1 * mask)) + jnp.sum(
        out_2 * mask
    ), jnp.array(0.0)


def bueche_rastrigin(
    x: jax.Array, state: BBOBState, params: BBOBParams
) -> tuple[jax.Array, jax.Array]:
    """Bueche-Rastrigin Function (Hansen et al., 2010, p. 20)."""
    max_num_dims = x.shape[0]
    mask = jnp.arange(max_num_dims) < params.num_dims

    # TODO: in "Real-Parameter Black-Box Optimization Benchmarking 2009: Noiseless
    # Functions Definitions", circular definition between z and s.
    z = transform_osz(x - params.x_opt)

    exp = (
        jnp.where(
            params.num_dims > 1,
            0.5 * jnp.arange(max_num_dims) / (params.num_dims - 1),
            0.5,
        )
        * mask
    )
    cond = (z > 0.0) & (jnp.arange(max_num_dims) % 2 == 1)
    s = jnp.where(cond, jnp.power(10.0, exp + 1), jnp.power(10.0, exp))

    z = s * z

    out_1 = jnp.cos(2 * jnp.pi * z)
    out_2 = jnp.square(z)

    return 10 * (params.num_dims - jnp.sum(out_1 * mask)) + jnp.sum(
        out_2 * mask
    ), 100 * f_pen(x, params.num_dims)


def linear_slope(
    x: jax.Array, state: BBOBState, params: BBOBParams
) -> tuple[jax.Array, jax.Array]:
    """Linear Slope (Hansen et al., 2010, p. 25)."""
    max_num_dims = x.shape[0]
    mask = jnp.arange(max_num_dims) < params.num_dims

    # x_opt = 5.0 * (2 * jax.random.bernoulli(subkey, shape=(max_num_dims,)) - 1)
    x_opt = jnp.where(params.x_opt > 0.0, 5.0, -5.0)

    z = jnp.where(x * x_opt < 25.0, x, x_opt)

    exp = (
        jnp.where(
            params.num_dims > 1, jnp.arange(max_num_dims) / (params.num_dims - 1), 0.5
        )
        * mask
    )
    s = jnp.sign(x_opt) * jnp.power(10.0, exp)

    out = 5.0 * jnp.abs(s) - s * z
    return jnp.sum(out * mask), jnp.array(0.0)


def attractive_sector(
    x: jax.Array, state: BBOBState, params: BBOBParams
) -> tuple[jax.Array, jax.Array]:
    """Attractive Sector Function (Hansen et al., 2010, p. 30)."""
    max_num_dims = x.shape[0]
    mask = jnp.arange(max_num_dims) < params.num_dims

    z = state.r @ (x - params.x_opt)
    z = _lambda_alpha_vector(10.0, max_num_dims, params.num_dims) * z
    z = state.q @ z

    s = jnp.where(z * params.x_opt > 0.0, 100.0, 1.0)

    out = jnp.sum(jnp.square(s * z) * mask)
    return jnp.power(transform_osz(out), 0.9), jnp.array(0.0)


def step_ellipsoidal(
    x: jax.Array, state: BBOBState, params: BBOBParams
) -> tuple[jax.Array, jax.Array]:
    """Step Ellipsoidal Function (Hansen et al., 2010, p. 35)."""
    max_num_dims = x.shape[0]
    mask = jnp.arange(max_num_dims) < params.num_dims

    z_hat = state.r @ (x - params.x_opt)
    z_hat = _lambda_alpha_vector(10.0, max_num_dims, params.num_dims) * z_hat

    z_tilde = jnp.where(
        jnp.abs(z_hat) > 0.5,
        jnp.floor(0.5 + z_hat),
        jnp.floor(0.5 + 10.0 * z_hat) / 10.0,
    )

    z = state.q @ z_tilde

    exp = (
        jnp.where(
            params.num_dims > 1,
            2.0 * jnp.arange(max_num_dims) / (params.num_dims - 1.0),
            2.0,
        )
        * mask
    )
    out = jnp.sum(100.0 * jnp.power(10.0, exp) * jnp.square(z) * mask)
    return 0.1 * jnp.maximum(jnp.abs(z_hat[0]) / 1e4, out), f_pen(x, params.num_dims)


def rosenbrock(
    x: jax.Array, state: BBOBState, params: BBOBParams
) -> tuple[jax.Array, jax.Array]:
    """Rosenbrock Function, original (Hansen et al., 2010, p. 40)."""
    max_num_dims = x.shape[0]
    mask = jnp.arange(max_num_dims - 1) < (params.num_dims - 1)

    x_opt = params.x_opt * 3 / 4
    z = jnp.maximum(1.0, jnp.sqrt(params.num_dims) / 8.0) * (x - x_opt) + 1.0
    z_i = z[:-1]
    z_ip1 = jnp.roll(z, -1)[:-1]

    out = 100.0 * jnp.square(jnp.square(z_i) - z_ip1) + jnp.square(z_i - 1)
    return jnp.sum(out * mask), jnp.array(0.0)


def rosenbrock_rotated(
    x: jax.Array, state: BBOBState, params: BBOBParams
) -> tuple[jax.Array, jax.Array]:
    """Rosenbrock Function, rotated (Hansen et al., 2010, p. 45)."""
    max_num_dims = x.shape[0]
    mask = jnp.arange(max_num_dims - 1) < (params.num_dims - 1)

    z = (
        jnp.maximum(1.0, jnp.sqrt(params.num_dims) / 8.0)
        * (state.r @ (x - params.x_opt))
        + 1.0
    )  # TODO: check if correct
    z_i = z[:-1]
    z_ip1 = jnp.roll(z, -1)[:-1]

    out = 100.0 * jnp.square(jnp.square(z_i) - z_ip1) + jnp.square(z_i - 1)
    return jnp.sum(out * mask), jnp.array(0.0)


def ellipsoidal_rotated(
    x: jax.Array, state: BBOBState, params: BBOBParams
) -> tuple[jax.Array, jax.Array]:
    """Ellipsoidal Function (Hansen et al., 2010, p. 50)."""
    max_num_dims = x.shape[0]
    mask = jnp.arange(max_num_dims) < params.num_dims

    z = state.r @ (x - params.x_opt)
    z = transform_osz(z)

    exp = (
        jnp.where(
            params.num_dims > 1,
            6.0 * jnp.arange(max_num_dims) / (params.num_dims - 1),
            6.0,
        )
        * mask
    )
    out = jnp.power(10.0, exp) * jnp.square(z)
    return jnp.sum(out * mask), jnp.array(0.0)


def discus(
    x: jax.Array, state: BBOBState, params: BBOBParams
) -> tuple[jax.Array, jax.Array]:
    """Discus Function (Hansen et al., 2010, p. 55)."""
    max_num_dims = x.shape[0]
    mask = jnp.arange(max_num_dims) < params.num_dims

    z = state.r @ (x - params.x_opt)
    z = transform_osz(z)

    z_squared = jnp.square(z)
    out = z_squared.at[0].multiply(10**6)
    return jnp.sum(out * mask), jnp.array(0.0)


def bent_cigar(
    x: jax.Array, state: BBOBState, params: BBOBParams
) -> tuple[jax.Array, jax.Array]:
    """Bent Cigar Function (Hansen et al., 2010, p. 60)."""
    max_num_dims = x.shape[0]
    mask = jnp.arange(max_num_dims) < params.num_dims

    z = state.r @ (x - params.x_opt)
    z = transform_asy(z, 0.5, params.num_dims)
    z = state.r @ z

    z_squared = jnp.square(z)
    out = z_squared.at[1:].multiply(10**6)
    return jnp.sum(out * mask), jnp.array(0.0)


def sharp_ridge(
    x: jax.Array, state: BBOBState, params: BBOBParams
) -> tuple[jax.Array, jax.Array]:
    """Sharp Ridge Function (Hansen et al., 2010, p. 65)."""
    max_num_dims = x.shape[0]
    mask = jnp.arange(max_num_dims - 1) < (params.num_dims - 1)

    z = state.r @ (x - params.x_opt)
    z = _lambda_alpha_vector(10.0, max_num_dims, params.num_dims) * z
    z = state.q @ z

    z_squared = jnp.square(z)
    return z_squared[0] + 100 * jnp.sqrt(jnp.sum(z_squared[1:] * mask)), jnp.array(0.0)


def different_powers(
    x: jax.Array, state: BBOBState, params: BBOBParams
) -> tuple[jax.Array, jax.Array]:
    """Different Powers Function (Hansen et al., 2010, p. 70)."""
    max_num_dims = x.shape[0]
    mask = jnp.arange(max_num_dims) < params.num_dims

    z = state.r @ (x - params.x_opt)

    exp = (
        jnp.where(
            params.num_dims > 1,
            2.0 + 4.0 * jnp.arange(max_num_dims) / (params.num_dims - 1),
            6.0,
        )
        * mask
    )
    out = jnp.power(jnp.abs(z), exp)
    return jnp.sqrt(jnp.sum(out * mask)), jnp.array(0.0)


def rastrigin_rotated(
    x: jax.Array, state: BBOBState, params: BBOBParams
) -> tuple[jax.Array, jax.Array]:
    """Rastrigin Function (Hansen et al., 2010, p. 75)."""
    max_num_dims = x.shape[0]
    mask = jnp.arange(max_num_dims) < params.num_dims

    z = state.r @ (x - params.x_opt)
    z = transform_asy(transform_osz(z), 0.2, params.num_dims)
    z = state.q @ z
    z = _lambda_alpha_vector(10.0, max_num_dims, params.num_dims) * z
    z = state.r @ z

    out_1 = jnp.cos(2 * jnp.pi * z)
    out_2 = jnp.square(z)
    return 10 * (params.num_dims - jnp.sum(out_1 * mask)) + jnp.sum(
        out_2 * mask
    ), jnp.array(0.0)


def weierstrass(
    x: jax.Array, state: BBOBState, params: BBOBParams
) -> tuple[jax.Array, jax.Array]:
    """Weierstrass Function (Hansen et al., 2010, p. 80)."""
    max_num_dims = x.shape[0]
    mask = jnp.arange(max_num_dims) < params.num_dims

    z = state.r @ (x - params.x_opt)
    z = transform_osz(z)
    z = state.q @ z
    z = _lambda_alpha_vector(0.01, max_num_dims, params.num_dims) * z
    z = state.r @ z

    k_order = 12
    half_pow_k = jnp.power(0.5, jnp.arange(k_order))
    three_pow_k = jnp.power(3.0, jnp.arange(k_order))
    f_0 = jnp.sum(half_pow_k * jnp.cos(jnp.pi * three_pow_k))

    out = jnp.sum(
        half_pow_k
        * jnp.cos(2 * jnp.pi * three_pow_k * (z[:, None] + 0.5))
        * mask[:, None]
    )
    return 10 * (out / params.num_dims - f_0) ** 3, 10 * f_pen(
        x, params.num_dims
    ) / params.num_dims


def schaffers_f7(
    x: jax.Array, state: BBOBState, params: BBOBParams
) -> tuple[jax.Array, jax.Array]:
    """Schaffers F7 Function (Hansen et al., 2010, p. 85)."""
    max_num_dims = x.shape[0]
    mask = jnp.arange(max_num_dims - 1) < (params.num_dims - 1)

    if max_num_dims == 1:
        return jnp.array(0.0), jnp.array(0.0)

    z = state.r @ (x - params.x_opt)
    z = transform_asy(z, 0.5, params.num_dims)
    z = state.q @ z
    z = _lambda_alpha_vector(10.0, max_num_dims, params.num_dims) * z

    z_i = z[:-1]
    z_ip1 = jnp.roll(z, -1)[:-1]
    s = jnp.sqrt(jnp.square(z_i) + jnp.square(z_ip1))

    out = jnp.sum((jnp.sqrt(s) + jnp.sqrt(s) * jnp.sin(50 * s**0.2) ** 2) * mask)
    return (out / (params.num_dims - 1.0)) ** 2, 10 * f_pen(x, params.num_dims)


def schaffers_f7_ill_conditioned(
    x: jax.Array, state: BBOBState, params: BBOBParams
) -> tuple[jax.Array, jax.Array]:
    """Schaffers F7 Function, ill-conditioned (Hansen et al., 2010, p. 90)."""
    max_num_dims = x.shape[0]
    mask = jnp.arange(max_num_dims - 1) < (params.num_dims - 1)

    if max_num_dims == 1:
        return jnp.array(0.0), jnp.array(0.0)

    z = state.r @ (x - params.x_opt)
    z = transform_asy(z, 0.5, params.num_dims)
    z = state.q @ z
    z = _lambda_alpha_vector(1000.0, max_num_dims, params.num_dims) * z

    z_i = z[:-1]
    z_ip1 = jnp.roll(z, -1)[:-1]
    s = jnp.sqrt(jnp.square(z_i) + jnp.square(z_ip1))

    out = jnp.sum((jnp.sqrt(s) + jnp.sqrt(s) * jnp.sin(50 * s**0.2) ** 2) * mask)
    return (out / (params.num_dims - 1.0)) ** 2, 10 * f_pen(x, params.num_dims)


def griewank_rosenbrock(
    x: jax.Array, state: BBOBState, params: BBOBParams
) -> tuple[jax.Array, jax.Array]:
    """Composite Griewank-Rosenbrock Function F8F2 (Hansen et al., 2010, p. 95)."""
    max_num_dims = x.shape[0]
    mask = jnp.arange(max_num_dims - 1) < (params.num_dims - 1)

    z = (
        jnp.maximum(1.0, jnp.sqrt(params.num_dims) / 8.0)
        * (state.r @ (x - params.x_opt))
        + 1.0
    )  # TODO: check if correct
    # z = jnp.maximum(1.0, jnp.sqrt(num_dims) / 8.0) * jnp.matmul(r, x - x_opt) + 1.0
    z_i = z[:-1]
    z_ip1 = jnp.roll(z, -1)[:-1]

    s = 100.0 * jnp.square(jnp.square(z_i) - z_ip1) + jnp.square(z_i - 1)
    out = s / 4000.0 - jnp.cos(s)
    return 10.0 * jnp.sum(out * mask) / (params.num_dims - 1) + 10, jnp.array(0.0)


def schwefel(
    x: jax.Array, state: BBOBState, params: BBOBParams
) -> tuple[jax.Array, jax.Array]:
    """Schwefel Function (Hansen et al., 2010, p. 100)."""
    max_num_dims = x.shape[0]
    mask = jnp.arange(max_num_dims) < params.num_dims

    # x_opt = 4.2096874633 * (2 * jax.random.bernoulli(subkey,
    # shape=(max_num_dims,)) - 1) / 2
    bernoulli = jnp.where(params.x_opt > 0.0, 1.0, -1.0)
    x_opt = 4.2096874633 * bernoulli / 2.0

    x_hat = 2.0 * bernoulli * x
    x_hat_i = x_hat
    x_hat_im1 = jnp.roll(x_hat, 1).at[0].set(0.0)
    x_opt_im1 = jnp.roll(x_opt, 1).at[0].set(0.0)
    z_hat = x_hat_i + 0.25 * (x_hat_im1 - 2 * jnp.abs(x_opt_im1))
    z = 100 * (
        _lambda_alpha_vector(10.0, max_num_dims, params.num_dims)
        * (z_hat - 2 * jnp.abs(x_opt))
        + 2 * jnp.abs(x_opt)
    )

    out = z * jnp.sin(jnp.sqrt(jnp.abs(z)))
    return -(
        jnp.sum(out * mask) / (100.0 * params.num_dims)
    ) + 4.189828872724339, 100 * f_pen(z / 100, params.num_dims)


def gallagher_101_me(
    x: jax.Array, state: BBOBState, params: BBOBParams
) -> tuple[jax.Array, jax.Array]:
    """Gallagher's Gaussian 101-me Peaks Function (Hansen et al., 2010, p. 105)."""
    max_num_dims = x.shape[0]
    mask = jnp.arange(max_num_dims) < params.num_dims

    num_optima = 101
    key = jax.random.key(0)
    key = jax.random.fold_in(key, state.q[0, 0])

    w = jnp.where(
        jnp.arange(num_optima) == 0,
        10.0,
        1.1 + 8.0 * (jnp.arange(num_optima) - 1.0) / (num_optima - 2.0),
    )

    alpha = jnp.zeros(num_optima)
    alpha_set = jnp.power(1000.0, 2.0 * jnp.arange(num_optima - 1) / (num_optima - 2))
    key, subkey = jax.random.split(key)
    alpha_permuted = jax.random.permutation(subkey, alpha_set)
    alpha = alpha.at[0].set(1000.0)
    alpha = alpha.at[1:].set(alpha_permuted)
    c = jax.vmap(
        lambda alpha: lambda_alpha(alpha, max_num_dims, params.num_dims) / alpha**0.25
    )(alpha)

    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, num_optima)

    def permute_diag(c_i, key):
        perm = jax.random.choice(
            key, jnp.arange(max_num_dims), shape=(max_num_dims,), replace=False, p=mask
        )
        diag = jnp.diag(c_i)
        new_diag = diag[perm]
        return c_i.at[jnp.arange(max_num_dims), jnp.arange(max_num_dims)].set(new_diag)

    c = jax.vmap(permute_diag)(c, keys)

    key, subkey = jax.random.split(key)
    y = jax.random.uniform(
        subkey,
        shape=(
            num_optima,
            max_num_dims,
        ),
        minval=-5.0,
        maxval=5.0,
    )
    y = y.at[0].set(params.x_opt) * mask

    def f(c_i, y_i, w_i):
        out = state.r @ (x - y_i)
        out = c_i @ out
        out = state.r.T @ out
        out = jnp.dot(x - y_i, out * mask)
        return w_i * jnp.exp(-out / (2 * params.num_dims))

    out = jax.vmap(f)(c, y, w)
    return jnp.square(transform_osz(10.0 - jnp.max(out))), f_pen(x, params.num_dims)


def gallagher_21_hi(
    x: jax.Array, state: BBOBState, params: BBOBParams
) -> tuple[jax.Array, jax.Array]:
    """Gallagher's Gaussian 21-hi Peaks Function (Hansen et al., 2010, p. 110)."""
    max_num_dims = x.shape[0]
    mask = jnp.arange(max_num_dims) < params.num_dims

    x_opt = params.x_opt * 3.92 / 4.0

    num_optima = 21
    key = jax.random.key(0)
    key = jax.random.fold_in(key, state.q[0, 0])

    w = jnp.where(
        jnp.arange(num_optima) == 0,
        10.0,
        1.1 + 8.0 * (jnp.arange(num_optima) - 1.0) / (num_optima - 2.0),
    )

    alpha = jnp.zeros(num_optima)
    alpha_set = jnp.power(1000.0, 2.0 * jnp.arange(num_optima - 1) / (num_optima - 2))
    key, subkey = jax.random.split(key)
    alpha_permuted = jax.random.permutation(subkey, alpha_set)
    alpha = alpha.at[0].set(1000.0**2)
    alpha = alpha.at[1:].set(alpha_permuted)
    c = jax.vmap(
        lambda alpha: lambda_alpha(alpha, max_num_dims, params.num_dims) / alpha**0.25
    )(alpha)

    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, num_optima)

    def permute_diag(c_i, key):
        perm = jax.random.choice(
            key, jnp.arange(max_num_dims), shape=(max_num_dims,), replace=False, p=mask
        )
        diag = jnp.diag(c_i)
        new_diag = diag[perm]
        return c_i.at[jnp.arange(max_num_dims), jnp.arange(max_num_dims)].set(new_diag)

    c = jax.vmap(permute_diag)(c, keys)

    key, subkey = jax.random.split(key)
    y = jax.random.uniform(
        subkey,
        shape=(
            num_optima,
            max_num_dims,
        ),
        minval=-4.9,
        maxval=4.9,
    )
    y = y.at[0].set(x_opt) * mask

    def f(c_i, y_i, w_i):
        out = state.r @ (x - y_i)
        out = c_i @ out
        out = state.r.T @ out
        out = jnp.dot(x - y_i, out * mask)
        return w_i * jnp.exp(-out / (2 * params.num_dims))

    out = jax.vmap(f)(c, y, w)
    return jnp.square(transform_osz(10.0 - jnp.max(out))), f_pen(x, params.num_dims)


def katsuura(
    x: jax.Array, state: BBOBState, params: BBOBParams
) -> tuple[jax.Array, jax.Array]:
    """Katsuura Function (Hansen et al., 2010, p. 115)."""
    # NOTE: Overflow in float32 because of the terms 2**31 and 2**32 but works with
    # higher precision
    # float64 with jax.config.update("jax_enable_x64", True)
    max_num_dims = x.shape[0]
    mask = jnp.arange(max_num_dims) < params.num_dims

    # num_terms = 32 in Hansen et al., 2010, but set to 30 for numerical stability
    num_terms = 30

    z = state.r @ (x - params.x_opt)
    z = _lambda_alpha_vector(100.0, max_num_dims, params.num_dims) * z
    z = state.q @ z

    two_pow_j = jnp.power(2.0, jnp.arange(1, num_terms + 1))
    sum_term = jnp.sum(
        jnp.abs(two_pow_j * z[:, None] - jnp.round(two_pow_j * z[:, None])) / two_pow_j,
        axis=1,
    )
    prod = jnp.prod(1.0 + jnp.arange(1, max_num_dims + 1) * sum_term * mask)
    return (10.0 / params.num_dims**2) * (
        jnp.power(prod, 10.0 / params.num_dims**1.2) - 1.0
    ), f_pen(x, params.num_dims)


def lunacek(
    x: jax.Array, state: BBOBState, params: BBOBParams
) -> tuple[jax.Array, jax.Array]:
    """Lunacek bi-Rastrigin Function (Hansen et al., 2010, p. 120)."""
    max_num_dims = x.shape[0]
    mask = jnp.arange(max_num_dims) < params.num_dims

    mu_0 = 2.5
    d = 1.0
    s = 1.0 - 1.0 / (2.0 * jnp.sqrt(params.num_dims + 20.0) - 8.2)
    mu_1 = -jnp.sqrt((mu_0**2 - d) / s)

    x_opt = jnp.where(params.x_opt > 0.0, mu_0 / 2.0, -mu_0 / 2.0)
    x_hat = 2.0 * jnp.sign(x_opt) * x

    z = state.r @ (x_hat - mu_0)
    z = _lambda_alpha_vector(100.0, max_num_dims, params.num_dims) * z
    z = state.q @ z

    s_1 = jnp.sum(jnp.square(x_hat - mu_0) * mask)
    s_2 = jnp.sum(jnp.square(x_hat - mu_1) * mask)
    s_3 = jnp.sum(jnp.cos(2 * jnp.pi * z) * mask)
    return jnp.minimum(s_1, d * params.num_dims + s * s_2) + 10.0 * (
        params.num_dims - s_3
    ), 10.0**4 * f_pen(x, params.num_dims)


bbob_fns = {
    # Part 1: Separable functions
    "sphere": sphere,
    "ellipsoidal": ellipsoidal,
    "rastrigin": rastrigin,
    "bueche_rastrigin": bueche_rastrigin,
    "linear_slope": linear_slope,
    # Part 2: Functions with low or moderate conditions
    "attractive_sector": attractive_sector,
    "step_ellipsoidal": step_ellipsoidal,
    "rosenbrock": rosenbrock,
    "rosenbrock_rotated": rosenbrock_rotated,
    # Part 3: Functions with high conditioning and unimodal
    "ellipsoidal_rotated": ellipsoidal_rotated,
    "discus": discus,
    "bent_cigar": bent_cigar,
    "sharp_ridge": sharp_ridge,
    "different_powers": different_powers,
    # Part 4: Multi-modal functions with adequate global structure
    "rastrigin_rotated": rastrigin_rotated,
    "weierstrass": weierstrass,
    "schaffers_f7": schaffers_f7,
    "schaffers_f7_ill_conditioned": schaffers_f7_ill_conditioned,
    "griewank_rosenbrock": griewank_rosenbrock,
    # Part 5: Multi-modal functions with weak global structure
    "schwefel": schwefel,
    "gallagher_101_me": gallagher_101_me,
    "gallagher_21_hi": gallagher_21_hi,
    "katsuura": katsuura,
    "lunacek": lunacek,
}
