"""Black-box Optimization Benchmarking Functions Definitions.

Every function is verified numerically against the official 2009 BBOB
implementation (`bbobbenchmarks.py`) on identical instances in float64 -- see
`tests/test_alignment.py`. Where the paper (Hansen, Finck, Ros, Auger,
INRIA RR-6829) and the official code disagree, the code wins, with a comment at
the site: the code is what every published BBOB result actually ran.

Two conventions to know:

- Functions receive fully prepared `params`: `params.x_opt` is the true
  argmin of every function. The per-function x_opt conventions of the official
  suite (sign vectors, scalings, component sign-forcing) are applied at sampling
  time via `X_OPT_CONVENTIONS`, mirroring how COCO stores the post-convention
  optimum. Hand-built params should apply the matching convention first.
- Functions return `(value, penalty)` with the optimum at value 0; `f_opt`
  is added by `BBOB.evaluate`, and noise applies to the value only.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

from .types import BBOBParams, BBOBState, FitnessFn


def _lambda_alpha_vector(alpha: float, num_dims: int) -> jax.Array:
    """Return the conditioning vector `alpha ** (0.5 i / (D - 1))`."""
    exp = 0.5 * jnp.arange(num_dims) / (num_dims - 1)
    return jnp.power(alpha, exp)


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

    The untaken branch is sanitized before `sqrt`/`power` so that
    `jax.grad` through negative coordinates yields zeros instead of NaN
    (the standard double-`where` guard); values are unchanged.
    """
    num_dims = x.shape[0]

    safe_x = jnp.where(x > 0.0, x, 0.0)
    exp = 1.0 + beta * (jnp.arange(num_dims) / (num_dims - 1)) * jnp.sqrt(safe_x)
    return jnp.where(x > 0.0, jnp.power(safe_x, exp), x)


def f_pen(x: jax.Array) -> jax.Array:
    """Boundary penalty."""
    out = jnp.abs(x) - 5.0
    return jnp.sum(jnp.square(jnp.maximum(0.0, out)))


# Weierstrass is a truncated series with fixed coefficients: 12 summands, as
# official (`F_WEIERSTRASS_SUMMANDS`). Built once with numpy rather than jnp:
# a jnp constant would freeze its dtype at import time, silently staying
# float32 under `jax_enable_x64` and costing precision.
_WEIERSTRASS_K_ORDER = 12
_WEIERSTRASS_HALF_POW_K = np.power(0.5, np.arange(_WEIERSTRASS_K_ORDER))
_WEIERSTRASS_THREE_POW_K = np.power(3.0, np.arange(_WEIERSTRASS_K_ORDER))
_WEIERSTRASS_F_0 = np.sum(
    _WEIERSTRASS_HALF_POW_K * np.cos(np.pi * _WEIERSTRASS_THREE_POW_K)
)


def sphere(
    x: jax.Array, state: BBOBState, params: BBOBParams
) -> tuple[jax.Array, jax.Array]:
    """Sphere Function (Hansen et al., 2010, p. 5)."""
    z = x - params.x_opt

    out = jnp.square(z)
    return jnp.sum(out), jnp.array(0.0)


def ellipsoidal(
    x: jax.Array, state: BBOBState, params: BBOBParams
) -> tuple[jax.Array, jax.Array]:
    """Ellipsoidal Function (Hansen et al., 2010, p. 10)."""
    num_dims = x.shape[0]

    z = transform_osz(x - params.x_opt)

    exp = 6.0 * jnp.arange(num_dims) / (num_dims - 1)
    out = jnp.power(10.0, exp) * jnp.square(z)
    return jnp.sum(out), jnp.array(0.0)


def rastrigin(
    x: jax.Array, state: BBOBState, params: BBOBParams
) -> tuple[jax.Array, jax.Array]:
    """Rastrigin Function (Hansen et al., 2010, p. 15)."""
    num_dims = x.shape[0]

    z = transform_asy(transform_osz(x - params.x_opt), 0.2)
    z = _lambda_alpha_vector(10.0, num_dims) * z

    out_1 = jnp.cos(2 * jnp.pi * z)
    out_2 = jnp.square(z)

    return 10 * (num_dims - jnp.sum(out_1)) + jnp.sum(out_2), jnp.array(0.0)


def bueche_rastrigin(
    x: jax.Array, state: BBOBState, params: BBOBParams
) -> tuple[jax.Array, jax.Array]:
    """Bueche-Rastrigin Function (Hansen et al., 2010, p. 20)."""
    num_dims = x.shape[0]

    # The paper's "for i = 1, 3, 5, ..." is 1-based; the skewed coordinates are
    # the 0-based EVEN ones. Both officials agree (transform_vars_brs.c:46 with a
    # comment flagging exactly this trap; bbobbenchmarks applies it to x[::2]).
    # The paper's seemingly circular z/s definition resolves cleanly: s_i > 0
    # always, so conditioning on the post-osz sign (as the officials do) is exact.
    z = transform_osz(x - params.x_opt)

    exp = 0.5 * jnp.arange(num_dims) / (num_dims - 1)
    cond = (z > 0.0) & (jnp.arange(num_dims) % 2 == 0)
    s = jnp.where(cond, jnp.power(10.0, exp + 1), jnp.power(10.0, exp))

    z = s * z

    out_1 = jnp.cos(2 * jnp.pi * z)
    out_2 = jnp.square(z)

    return 10 * (num_dims - jnp.sum(out_1)) + jnp.sum(out_2), 100 * f_pen(x)


def linear_slope(
    x: jax.Array, state: BBOBState, params: BBOBParams
) -> tuple[jax.Array, jax.Array]:
    """Linear Slope (Hansen et al., 2010, p. 25)."""
    num_dims = x.shape[0]

    # params.x_opt is the true optimum: +-5 per coordinate, applied by the
    # sampling convention (see X_OPT_CONVENTIONS).
    x_opt = params.x_opt

    z = jnp.where(x * x_opt < 25.0, x, x_opt)

    exp = jnp.arange(num_dims) / (num_dims - 1)
    s = jnp.sign(x_opt) * jnp.power(10.0, exp)

    out = 5.0 * jnp.abs(s) - s * z
    return jnp.sum(out), jnp.array(0.0)


def attractive_sector(
    x: jax.Array, state: BBOBState, params: BBOBParams
) -> tuple[jax.Array, jax.Array]:
    """Attractive Sector Function (Hansen et al., 2010, p. 30)."""
    num_dims = x.shape[0]

    z = params.r @ (x - params.x_opt)
    z = _lambda_alpha_vector(10.0, num_dims) * z
    z = params.q @ z

    s = jnp.where(z * params.x_opt > 0.0, 100.0, 1.0)

    out = jnp.sum(jnp.square(s * z))
    return jnp.power(transform_osz(out), 0.9), jnp.array(0.0)


def step_ellipsoidal(
    x: jax.Array, state: BBOBState, params: BBOBParams
) -> tuple[jax.Array, jax.Array]:
    """Step Ellipsoidal Function (Hansen et al., 2010, p. 35)."""
    num_dims = x.shape[0]

    z_hat = params.r @ (x - params.x_opt)
    z_hat = _lambda_alpha_vector(10.0, num_dims) * z_hat

    # floor(0.5 + z) is round-half-up, matching COCO's C (`coco_double_round`).
    # The 2009 Python reference uses numpy's round-half-to-even instead, so the
    # two officials disagree on exact .5 ties; ties have measure zero and the C
    # is what the suite ships.
    z_tilde = jnp.where(
        jnp.abs(z_hat) > 0.5,
        jnp.floor(0.5 + z_hat),
        jnp.floor(0.5 + 10.0 * z_hat) / 10.0,
    )

    z = params.q @ z_tilde

    exp = 2.0 * jnp.arange(num_dims) / (num_dims - 1.0)
    # No leading coefficient: the conditioning lives entirely in the exponent.
    # (An earlier version carried an extra x100 here; paper 2.7, f_step_ellipsoid.c
    # and bbobbenchmarks all agree there is none.)
    out = jnp.sum(jnp.power(10.0, exp) * jnp.square(z))
    return 0.1 * jnp.maximum(jnp.abs(z_hat[0]) / 1e4, out), f_pen(x)


def rosenbrock(
    x: jax.Array, state: BBOBState, params: BBOBParams
) -> tuple[jax.Array, jax.Array]:
    """Rosenbrock Function, original (Hansen et al., 2010, p. 40)."""
    num_dims = x.shape[0]

    # params.x_opt is the true optimum, already scaled by 0.75 at sampling time
    # (COCO stores the scaled vector too: f_rosenbrock.c:86).
    z = jnp.maximum(1.0, jnp.sqrt(num_dims) / 8.0) * (x - params.x_opt) + 1.0
    z_i = z[:-1]
    z_ip1 = jnp.roll(z, -1)[:-1]

    out = 100.0 * jnp.square(jnp.square(z_i) - z_ip1) + jnp.square(z_i - 1)
    return jnp.sum(out), jnp.array(0.0)


def rosenbrock_rotated(
    x: jax.Array, state: BBOBState, params: BBOBParams
) -> tuple[jax.Array, jax.Array]:
    """Rosenbrock Function, rotated (Hansen et al., 2010, p. 45)."""
    num_dims = x.shape[0]

    # Official BBOB uses z = s*R*x + 1/2 with the optimum derived from R (norm
    # ~ sqrt(D)/2, always near the origin). This parameterization is verified
    # exactly equivalent: with x_opt = R^T (0.5/s) 1 it reproduces the official
    # function bit-for-bit, and any other x_opt is a pure translation of an
    # official landscape. Kept deliberately: the optimum is placeable, and
    # params.x_opt is the argmin like everywhere else.
    z = (
        jnp.maximum(1.0, jnp.sqrt(num_dims) / 8.0) * (params.r @ (x - params.x_opt))
        + 1.0
    )
    z_i = z[:-1]
    z_ip1 = jnp.roll(z, -1)[:-1]

    out = 100.0 * jnp.square(jnp.square(z_i) - z_ip1) + jnp.square(z_i - 1)
    return jnp.sum(out), jnp.array(0.0)


def ellipsoidal_rotated(
    x: jax.Array, state: BBOBState, params: BBOBParams
) -> tuple[jax.Array, jax.Array]:
    """Ellipsoidal Function (Hansen et al., 2010, p. 50)."""
    num_dims = x.shape[0]

    z = params.r @ (x - params.x_opt)
    z = transform_osz(z)

    exp = 6.0 * jnp.arange(num_dims) / (num_dims - 1)
    out = jnp.power(10.0, exp) * jnp.square(z)
    return jnp.sum(out), jnp.array(0.0)


def discus(
    x: jax.Array, state: BBOBState, params: BBOBParams
) -> tuple[jax.Array, jax.Array]:
    """Discus Function (Hansen et al., 2010, p. 55)."""
    z = params.r @ (x - params.x_opt)
    z = transform_osz(z)

    z_squared = jnp.square(z)
    out = z_squared.at[0].multiply(10**6)
    return jnp.sum(out), jnp.array(0.0)


def bent_cigar(
    x: jax.Array, state: BBOBState, params: BBOBParams
) -> tuple[jax.Array, jax.Array]:
    """Bent Cigar Function (Hansen et al., 2010, p. 60)."""
    z = params.r @ (x - params.x_opt)
    z = transform_asy(z, 0.5)
    z = params.r @ z

    z_squared = jnp.square(z)
    out = z_squared.at[1:].multiply(10**6)
    return jnp.sum(out), jnp.array(0.0)


def sharp_ridge(
    x: jax.Array, state: BBOBState, params: BBOBParams
) -> tuple[jax.Array, jax.Array]:
    """Sharp Ridge Function (Hansen et al., 2010, p. 65)."""
    num_dims = x.shape[0]

    z = params.r @ (x - params.x_opt)
    z = _lambda_alpha_vector(10.0, num_dims) * z
    z = params.q @ z

    z_squared = jnp.square(z)
    return z_squared[0] + 100 * jnp.sqrt(jnp.sum(z_squared[1:])), jnp.array(0.0)


def different_powers(
    x: jax.Array, state: BBOBState, params: BBOBParams
) -> tuple[jax.Array, jax.Array]:
    """Different Powers Function (Hansen et al., 2010, p. 70)."""
    num_dims = x.shape[0]

    z = params.r @ (x - params.x_opt)

    exp = 2.0 + 4.0 * jnp.arange(num_dims) / (num_dims - 1)
    out = jnp.power(jnp.abs(z), exp)
    return jnp.sqrt(jnp.sum(out)), jnp.array(0.0)


def rastrigin_rotated(
    x: jax.Array, state: BBOBState, params: BBOBParams
) -> tuple[jax.Array, jax.Array]:
    """Rastrigin Function (Hansen et al., 2010, p. 75)."""
    num_dims = x.shape[0]

    z = params.r @ (x - params.x_opt)
    z = transform_asy(transform_osz(z), 0.2)
    z = params.q @ z
    z = _lambda_alpha_vector(10.0, num_dims) * z
    z = params.r @ z

    out_1 = jnp.cos(2 * jnp.pi * z)
    out_2 = jnp.square(z)
    return 10 * (num_dims - jnp.sum(out_1)) + jnp.sum(out_2), jnp.array(0.0)


def weierstrass(
    x: jax.Array, state: BBOBState, params: BBOBParams
) -> tuple[jax.Array, jax.Array]:
    """Weierstrass Function (Hansen et al., 2010, p. 80)."""
    num_dims = x.shape[0]

    z = params.r @ (x - params.x_opt)
    z = transform_osz(z)
    z = params.q @ z
    z = _lambda_alpha_vector(0.01, num_dims) * z
    z = params.r @ z

    out = jnp.sum(
        _WEIERSTRASS_HALF_POW_K
        * jnp.cos(2 * jnp.pi * _WEIERSTRASS_THREE_POW_K * (z[:, None] + 0.5))[:, None]
    )
    return 10 * (out / num_dims - _WEIERSTRASS_F_0) ** 3, 10 * f_pen(x) / num_dims


def _schaffers(
    x: jax.Array, params: BBOBParams, conditioning: float
) -> tuple[jax.Array, jax.Array]:
    """Schaffers F7, shared by the two conditionings."""
    num_dims = x.shape[0]

    z = params.r @ (x - params.x_opt)
    z = transform_asy(z, 0.5)
    z = params.q @ z
    z = _lambda_alpha_vector(conditioning, num_dims) * z

    z_i = z[:-1]
    z_ip1 = jnp.roll(z, -1)[:-1]
    s = jnp.sqrt(jnp.square(z_i) + jnp.square(z_ip1))

    out = jnp.sum(jnp.sqrt(s) + jnp.sqrt(s) * jnp.sin(50 * s**0.2) ** 2)
    # The sum runs over D - 1 consecutive pairs; guard the degenerate D = 1.
    num_pairs = num_dims - 1.0
    return (out / num_pairs) ** 2, 10 * f_pen(x)


def schaffers_f7(
    x: jax.Array, state: BBOBState, params: BBOBParams
) -> tuple[jax.Array, jax.Array]:
    """Schaffers F7 Function (Hansen et al., 2010, p. 85)."""
    return _schaffers(x, params, conditioning=10.0)


def schaffers_f7_ill_conditioned(
    x: jax.Array, state: BBOBState, params: BBOBParams
) -> tuple[jax.Array, jax.Array]:
    """Schaffers F7 Function, ill-conditioned (Hansen et al., 2010, p. 90)."""
    return _schaffers(x, params, conditioning=1000.0)


def griewank_rosenbrock(
    x: jax.Array, state: BBOBState, params: BBOBParams
) -> tuple[jax.Array, jax.Array]:
    """Composite Griewank-Rosenbrock Function F8F2 (Hansen et al., 2010, p. 95)."""
    num_dims = x.shape[0]

    # Same deliberate parameterization as rosenbrock_rotated: official BBOB has
    # z = s*R*x + 1/2 with an R-derived optimum; this form is verified exactly
    # equivalent (x_opt = R^T (0.5/s) 1 reproduces it bit-for-bit) and makes
    # params.x_opt the argmin.
    z = (
        jnp.maximum(1.0, jnp.sqrt(num_dims) / 8.0) * (params.r @ (x - params.x_opt))
        + 1.0
    )
    z_i = z[:-1]
    z_ip1 = jnp.roll(z, -1)[:-1]

    s = 100.0 * jnp.square(jnp.square(z_i) - z_ip1) + jnp.square(z_i - 1)
    out = s / 4000.0 - jnp.cos(s)
    return 10.0 * jnp.sum(out) / (num_dims - 1) + 10, jnp.array(0.0)


def schwefel(
    x: jax.Array, state: BBOBState, params: BBOBParams
) -> tuple[jax.Array, jax.Array]:
    """Schwefel Function (Hansen et al., 2010, p. 100)."""
    num_dims = x.shape[0]

    # params.x_opt is the true optimum: +-4.2096874633/2 per coordinate, applied
    # by the sampling convention. (The C source carries ...4637 in one literal;
    # the paper, bbobbenchmarks and C's own best_parameter all say ...4633.)
    x_opt = params.x_opt
    bernoulli = jnp.where(x_opt > 0.0, 1.0, -1.0)

    x_hat = 2.0 * bernoulli * x
    x_hat_i = x_hat
    x_hat_im1 = jnp.roll(x_hat, 1).at[0].set(0.0)
    x_opt_im1 = jnp.roll(x_opt, 1).at[0].set(0.0)
    z_hat = x_hat_i + 0.25 * (x_hat_im1 - 2 * jnp.abs(x_opt_im1))
    z = 100 * (
        _lambda_alpha_vector(10.0, num_dims) * (z_hat - 2 * jnp.abs(x_opt))
        + 2 * jnp.abs(x_opt)
    )

    out = z * jnp.sin(jnp.sqrt(jnp.abs(z)))
    return -(jnp.sum(out) / (100.0 * num_dims)) + 4.189828872724339, 100 * f_pen(
        z / 100
    )


def _gallagher(
    x: jax.Array,
    params: BBOBParams,
    num_optima: int,
    first_alpha: float,
    y_range: float,
    fold: int,
) -> tuple[jax.Array, jax.Array]:
    """Gallagher's Gaussian peaks, shared by the 101-me and 21-hi variants.

    The peak layout -- which conditioning goes to which peak, how each one is
    permuted, and where the peaks sit -- is instance data, drawn from the
    instance's own key. (An earlier version derived it from
    ``fold_in(key(0), q[0, 0])``; ``fold_in`` int-casts the float and
    ``|q00| <= 1`` for any rotation, so every rotated instance shared one
    frozen layout.) ``fold`` separates the two variants' layouts.
    """
    num_dims = x.shape[0]
    key = jax.random.fold_in(params.key, fold)

    w = jnp.where(
        jnp.arange(num_optima) == 0,
        10.0,
        1.1 + 8.0 * (jnp.arange(num_optima) - 1.0) / (num_optima - 2.0),
    )

    alpha_set = jnp.power(1000.0, 2.0 * jnp.arange(num_optima - 1) / (num_optima - 2))
    key, subkey = jax.random.split(key)
    alpha = (
        jnp.zeros(num_optima)
        .at[0]
        .set(first_alpha)
        .at[1:]
        .set(jax.random.permutation(subkey, alpha_set))
    )
    # C_i is diagonal, so carry the diagonal alone: (num_optima, D) rather than
    # (num_optima, D, D).
    c = jax.vmap(lambda alpha: _lambda_alpha_vector(alpha, num_dims) / alpha**0.25)(
        alpha
    )

    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, num_optima)
    c = jax.vmap(lambda c_i, k: c_i[jax.random.permutation(k, num_dims)])(c, keys)

    key, subkey = jax.random.split(key)
    y = jax.random.uniform(
        subkey, shape=(num_optima, num_dims), minval=-y_range, maxval=y_range
    )
    y = y.at[0].set(params.x_opt)

    # (x - y_i)^T R^T C_i R (x - y_i) = z^T C_i z with z = R(x - y_i), and C_i
    # diagonal makes that sum(c_i * z**2). All peaks at once: one (num_optima, D)
    # by (D, D) matmul instead of three matrix-vector products per peak.
    z = (x - y) @ params.r.T
    out = w * jnp.exp(-jnp.sum(c * jnp.square(z), axis=-1) / (2 * num_dims))
    return jnp.square(transform_osz(10.0 - jnp.max(out))), f_pen(x)


def gallagher_101_me(
    x: jax.Array, state: BBOBState, params: BBOBParams
) -> tuple[jax.Array, jax.Array]:
    """Gallagher's Gaussian 101-me Peaks Function (Hansen et al., 2010, p. 105)."""
    return _gallagher(
        x, params, num_optima=101, first_alpha=1000.0, y_range=5.0, fold=21
    )


def gallagher_21_hi(
    x: jax.Array, state: BBOBState, params: BBOBParams
) -> tuple[jax.Array, jax.Array]:
    """Gallagher's Gaussian 21-hi Peaks Function (Hansen et al., 2010, p. 110).

    ``params.x_opt`` is the true optimum, already scaled by 0.98 at sampling.
    """
    return _gallagher(
        x, params, num_optima=21, first_alpha=1000.0**2, y_range=4.9, fold=22
    )


def katsuura(
    x: jax.Array, state: BBOBState, params: BBOBParams
) -> tuple[jax.Array, jax.Array]:
    """Katsuura Function (Hansen et al., 2010, p. 115)."""
    num_dims = x.shape[0]

    # 32 terms, exactly as official. The powers are computed in floating point
    # (2.0**j), so there is no integer overflow; in float32 the j > 24 terms are
    # below mantissa resolution and contribute harmless near-zero noise, while in
    # float64 all 32 terms are needed for exactness (J=30 costs ~1e-7 absolute).
    num_terms = 32

    z = params.r @ (x - params.x_opt)
    z = _lambda_alpha_vector(100.0, num_dims) * z
    z = params.q @ z

    two_pow_j = jnp.power(2.0, jnp.arange(1, num_terms + 1))
    sum_term = jnp.sum(
        jnp.abs(two_pow_j * z[:, None] - jnp.round(two_pow_j * z[:, None])) / two_pow_j,
        axis=1,
    )
    prod = jnp.prod(1.0 + jnp.arange(1, num_dims + 1) * sum_term)
    return (10.0 / num_dims**2) * (jnp.power(prod, 10.0 / num_dims**1.2) - 1.0), f_pen(
        x
    )


def lunacek(
    x: jax.Array, state: BBOBState, params: BBOBParams
) -> tuple[jax.Array, jax.Array]:
    """Lunacek bi-Rastrigin Function (Hansen et al., 2010, p. 120)."""
    num_dims = x.shape[0]

    mu_0 = 2.5
    d = 1.0
    s = 1.0 - 1.0 / (2.0 * jnp.sqrt(num_dims + 20.0) - 8.2)
    mu_1 = -jnp.sqrt((mu_0**2 - d) / s)

    # params.x_opt is the true optimum: +-mu_0/2 per coordinate, applied by the
    # sampling convention.
    x_opt = params.x_opt
    x_hat = 2.0 * jnp.sign(x_opt) * x

    z = params.r @ (x_hat - mu_0)
    z = _lambda_alpha_vector(100.0, num_dims) * z
    z = params.q @ z

    s_1 = jnp.sum(jnp.square(x_hat - mu_0))
    s_2 = jnp.sum(jnp.square(x_hat - mu_1))
    s_3 = jnp.sum(jnp.cos(2 * jnp.pi * z))
    return jnp.minimum(s_1, d * num_dims + s * s_2) + 10.0 * (
        num_dims - s_3
    ), 10.0**4 * f_pen(x)


def _x_opt_bueche_rastrigin(x_opt: jax.Array) -> jax.Array:
    # The officials force the skewed (0-based even) coordinates non-negative
    # (f_bueche_rastrigin.c:81-84, "in the legacy code but _not_ in the function
    # description"; bbobbenchmarks xopt[::2] = abs(...)). The paper omits it;
    # the code defines the real instances, so the code wins.
    is_even = jnp.arange(x_opt.shape[0]) % 2 == 0
    return jnp.where(is_even, jnp.abs(x_opt), x_opt)


def _x_opt_linear_slope(x_opt: jax.Array) -> jax.Array:
    # The optimum sits on the +-5 corner selected by the sign of the draw.
    return jnp.where(x_opt > 0.0, 5.0, -5.0)


def _x_opt_rosenbrock(x_opt: jax.Array) -> jax.Array:
    # Optimum in [-3, 3]^D (paper 2.8); COCO stores the scaled vector too.
    return 0.75 * x_opt


def _x_opt_schwefel(x_opt: jax.Array) -> jax.Array:
    return jnp.where(x_opt > 0.0, 4.2096874633 / 2.0, -4.2096874633 / 2.0)


def _x_opt_gallagher_21_hi(x_opt: jax.Array) -> jax.Array:
    # y_1 in [-3.92, 3.92]^D (paper 5.22; bbobbenchmarks fac = 0.98).
    return 0.98 * x_opt


def _x_opt_lunacek(x_opt: jax.Array) -> jax.Array:
    return jnp.where(x_opt > 0.0, 2.5 / 2.0, -2.5 / 2.0)


# Per-function preparation of a raw uniform x_opt draw, applied by
# `BBOB.sample` so that `params.x_opt` is always the function's true argmin
# -- the invariant COCO keeps by storing the post-convention optimum. Functions
# absent from this mapping use the draw unchanged.
X_OPT_CONVENTIONS: dict[str, Callable[[jax.Array], jax.Array]] = {
    "bueche_rastrigin": _x_opt_bueche_rastrigin,
    "linear_slope": _x_opt_linear_slope,
    "rosenbrock": _x_opt_rosenbrock,
    "schwefel": _x_opt_schwefel,
    "gallagher_21_hi": _x_opt_gallagher_21_hi,
    "lunacek": _x_opt_lunacek,
}


# The 24 standard BBOB functions, in canonical f1-f24 order.
BBOB_FNS: dict[str, FitnessFn] = {
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
