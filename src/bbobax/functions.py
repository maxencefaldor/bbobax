"""The 24 standard BBOB functions, one class each.

Every function is verified numerically against the official 2009 BBOB
implementation (`bbobbenchmarks.py`) on identical instances in float64 -- see
`tests/test_alignment.py`. Where the paper (Hansen, Finck, Ros, Auger,
INRIA RR-6829) and the official code disagree, the code wins, with a comment at
the site: the code is what every published BBOB result actually ran.

Three conventions to know:

- `_value` receives fully prepared `params`: `params.x_opt` is the true argmin
  of every function, drawn by `_sample_x_opt`, mirroring how COCO stores the
  optimum its own construction ends up at.
- `_value` returns `(value, penalty)` with the optimum at value 0; `f_opt` is
  added by `BBOBProblem.evaluate`, and noise applies to the value only.
- Every constant the paper or COCO gives a name to is a class attribute, so a
  function's settings are readable without reading its body. Two names recur
  and are spelled the same way throughout, whatever the COCO file happens to
  call them locally: `condition` is the conditioning, and `penalty_factor` is
  the multiplier on the boundary penalty. Constants that are part of a
  function's own formula and unnamed upstream (Rosenbrock's 100, Griewank's
  4000) stay inline, where the formula reads.

Where two of the 24 are the same landscape at different settings -- the two
Schaffers conditionings, the two Gallagher peak counts -- that relationship is
a subclass with different class attributes, so the math is written once.
"""

import jax
import jax.numpy as jnp
import numpy as np

from .problem import BBOBParams, BBOBProblem


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


# --- Part 1: Separable functions ----------------------------------------------


class Sphere(BBOBProblem):
    """Sphere Function (Hansen et al., 2010, p. 5)."""

    name = "sphere"

    def _value(self, x: jax.Array, params: BBOBParams) -> tuple[jax.Array, jax.Array]:
        z = x - params.x_opt

        value = jnp.sum(jnp.square(z))
        return value, jnp.array(0.0)


class Ellipsoidal(BBOBProblem):
    """Ellipsoidal Function (Hansen et al., 2010, p. 10)."""

    name = "ellipsoidal"

    # Conditioning (COCO: the `conditioning` f_ellipsoid is allocated with).
    condition: float = 1e6

    def _value(self, x: jax.Array, params: BBOBParams) -> tuple[jax.Array, jax.Array]:
        z = transform_osz(x - params.x_opt)

        exp = jnp.arange(self.num_dims) / (self.num_dims - 1)
        value = jnp.sum(jnp.power(self.condition, exp) * jnp.square(z))
        return value, jnp.array(0.0)


class Rastrigin(BBOBProblem):
    """Rastrigin Function (Hansen et al., 2010, p. 15)."""

    name = "rastrigin"

    # Conditioning.
    condition: float = 10.0
    # Asymmetry, the paper's beta.
    beta: float = 0.2

    def _value(self, x: jax.Array, params: BBOBParams) -> tuple[jax.Array, jax.Array]:
        z = transform_asy(transform_osz(x - params.x_opt), self.beta)
        z = lambda_alpha(self.condition, self.num_dims) * z

        oscillation = jnp.sum(jnp.cos(2 * jnp.pi * z))
        value = 10 * (self.num_dims - oscillation) + jnp.sum(jnp.square(z))
        return value, jnp.array(0.0)


class BuecheRastrigin(BBOBProblem):
    """Bueche-Rastrigin Function (Hansen et al., 2010, p. 20)."""

    name = "bueche_rastrigin"

    # Conditioning.
    condition: float = 10.0
    # Multiplier on the boundary penalty.
    penalty_factor: float = 100.0

    def _sample_x_opt(self, key: jax.Array) -> jax.Array:
        """Draw the optimum with its skewed (0-based even) coordinates non-negative.

        Both official implementations do this (`f_bueche_rastrigin.c:81-84`,
        whose comment notes it is "in the legacy code but _not_ in the function
        description"; `bbobbenchmarks` applies `xopt[::2] = abs(...)`). The
        paper omits it; the code defines the instances every published result
        ran on, so the code wins.
        """
        x_opt = super()._sample_x_opt(key)
        is_even = jnp.arange(self.num_dims) % 2 == 0
        return jnp.where(is_even, jnp.abs(x_opt), x_opt)

    def _value(self, x: jax.Array, params: BBOBParams) -> tuple[jax.Array, jax.Array]:
        # The paper's "for i = 1, 3, 5, ..." is 1-based; the skewed coordinates
        # are the 0-based EVEN ones. Both officials agree (transform_vars_brs.c:46
        # with a comment flagging exactly this trap; bbobbenchmarks applies it to
        # x[::2]). The paper's seemingly circular z/s definition resolves cleanly:
        # s_i > 0 always, so conditioning on the post-osz sign (as the officials
        # do) is exact.
        z = transform_osz(x - params.x_opt)

        conditioning = lambda_alpha(self.condition, self.num_dims)
        is_skewed = (z > 0.0) & (jnp.arange(self.num_dims) % 2 == 0)
        z = jnp.where(is_skewed, 10.0, 1.0) * conditioning * z

        oscillation = jnp.sum(jnp.cos(2 * jnp.pi * z))
        value = 10 * (self.num_dims - oscillation) + jnp.sum(jnp.square(z))
        return value, self.penalty_factor * f_pen(x)


class LinearSlope(BBOBProblem):
    """Linear Slope (Hansen et al., 2010, p. 25)."""

    name = "linear_slope"

    # Conditioning of the slope (COCO: `alpha` in f_linear_slope.c).
    condition: float = 100.0

    def _sample_x_opt(self, key: jax.Array) -> jax.Array:
        """Draw a corner of the box: a linear function has no interior minimum.

        Only the sign of each coordinate is an instance choice, so that is what
        is drawn -- `x_opt_range` does not apply here.
        """
        is_positive = jax.random.bernoulli(key, shape=(self.num_dims,))
        return jnp.where(is_positive, 5.0, -5.0)

    def _value(self, x: jax.Array, params: BBOBParams) -> tuple[jax.Array, jax.Array]:
        # params.x_opt is the true optimum: +-5 per coordinate, drawn by
        # _sample_x_opt.
        x_opt = params.x_opt

        z = jnp.where(x * x_opt < 25.0, x, x_opt)
        s = jnp.sign(x_opt) * lambda_alpha(self.condition, self.num_dims)

        value = jnp.sum(5.0 * jnp.abs(s) - s * z)
        return value, jnp.array(0.0)


# --- Part 2: Functions with low or moderate conditioning ----------------------


class AttractiveSector(BBOBProblem):
    """Attractive Sector Function (Hansen et al., 2010, p. 30)."""

    name = "attractive_sector"

    # Conditioning.
    condition: float = 10.0
    # Exponent of the final power transform (COCO: `transform_obj_power`).
    power: float = 0.9

    def _value(self, x: jax.Array, params: BBOBParams) -> tuple[jax.Array, jax.Array]:
        z = params.r @ (x - params.x_opt)
        z = lambda_alpha(self.condition, self.num_dims) * z
        z = params.q @ z

        s = jnp.where(z * params.x_opt > 0.0, 100.0, 1.0)

        value = jnp.power(transform_osz(jnp.sum(jnp.square(s * z))), self.power)
        return value, jnp.array(0.0)


class StepEllipsoidal(BBOBProblem):
    """Step Ellipsoidal Function (Hansen et al., 2010, p. 35)."""

    name = "step_ellipsoidal"

    # Conditioning of the ellipsoid.
    condition: float = 100.0
    # Conditioning applied before the step (COCO: `alpha` in f_step_ellipsoid.c).
    alpha: float = 10.0
    # Multiplier on the boundary penalty.
    penalty_factor: float = 1.0

    def _value(self, x: jax.Array, params: BBOBParams) -> tuple[jax.Array, jax.Array]:
        z_hat = params.r @ (x - params.x_opt)
        z_hat = lambda_alpha(self.alpha, self.num_dims) * z_hat

        # floor(0.5 + z) is round-half-up, matching COCO's C
        # (`coco_double_round`). The 2009 Python reference uses numpy's
        # round-half-to-even instead, so the two officials disagree on exact .5
        # ties; ties have measure zero and the C is what the suite ships.
        z_tilde = jnp.where(
            jnp.abs(z_hat) > 0.5,
            jnp.floor(0.5 + z_hat),
            jnp.floor(0.5 + 10.0 * z_hat) / 10.0,
        )

        z = params.q @ z_tilde

        exp = jnp.arange(self.num_dims) / (self.num_dims - 1.0)
        # No leading coefficient: the conditioning lives entirely in the
        # exponent. (An earlier version carried an extra x100 here; paper 2.7,
        # f_step_ellipsoid.c and bbobbenchmarks all agree there is none.)
        out = jnp.sum(jnp.power(self.condition, exp) * jnp.square(z))

        value = 0.1 * jnp.maximum(jnp.abs(z_hat[0]) / 1e4, out)
        return value, self.penalty_factor * f_pen(x)


class Rosenbrock(BBOBProblem):
    """Rosenbrock Function, original (Hansen et al., 2010, p. 40)."""

    name = "rosenbrock"

    def _sample_x_opt(self, key: jax.Array) -> jax.Array:
        """Draw from [-3, 3]^D: Rosenbrock's optimum lives there (paper 2.8)."""
        return 0.75 * super()._sample_x_opt(key)

    def _value(self, x: jax.Array, params: BBOBParams) -> tuple[jax.Array, jax.Array]:
        # params.x_opt is the true optimum, already scaled at sampling time
        # (COCO stores the scaled vector too: f_rosenbrock.c:86).
        scale = jnp.maximum(1.0, jnp.sqrt(self.num_dims) / 8.0)
        z = scale * (x - params.x_opt) + 1.0
        z_i = z[:-1]
        z_ip1 = jnp.roll(z, -1)[:-1]

        out = 100.0 * jnp.square(jnp.square(z_i) - z_ip1) + jnp.square(z_i - 1)

        value = jnp.sum(out)
        return value, jnp.array(0.0)


class RosenbrockRotated(BBOBProblem):
    """Rosenbrock Function, rotated (Hansen et al., 2010, p. 45)."""

    name = "rosenbrock_rotated"

    def _value(self, x: jax.Array, params: BBOBParams) -> tuple[jax.Array, jax.Array]:
        # Official BBOB uses z = s*R*x + 1/2 with the optimum derived from R
        # (norm ~ sqrt(D)/2, always near the origin). This parameterization is
        # verified exactly equivalent: with x_opt = R^T (0.5/s) 1 it reproduces
        # the official function bit-for-bit, and any other x_opt is a pure
        # translation of an official landscape. Kept deliberately: the optimum
        # is placeable, and params.x_opt is the argmin like everywhere else.
        scale = jnp.maximum(1.0, jnp.sqrt(self.num_dims) / 8.0)
        z = scale * (params.r @ (x - params.x_opt)) + 1.0
        z_i = z[:-1]
        z_ip1 = jnp.roll(z, -1)[:-1]

        out = 100.0 * jnp.square(jnp.square(z_i) - z_ip1) + jnp.square(z_i - 1)

        value = jnp.sum(out)
        return value, jnp.array(0.0)


# --- Part 3: Functions with high conditioning and unimodal --------------------


class EllipsoidalRotated(BBOBProblem):
    """Ellipsoidal Function, rotated (Hansen et al., 2010, p. 50).

    Not a subclass of `Ellipsoidal`: only the conditioning is shared, the
    coordinate pipeline differs, so subclassing would inherit one constant and
    override everything else. Subclassing is reserved here for the pairs that
    share their whole `_value` -- the two Schaffers and the two Gallaghers.
    """

    name = "ellipsoidal_rotated"

    # Conditioning.
    condition: float = 1e6

    def _value(self, x: jax.Array, params: BBOBParams) -> tuple[jax.Array, jax.Array]:
        z = transform_osz(params.r @ (x - params.x_opt))

        exp = jnp.arange(self.num_dims) / (self.num_dims - 1)
        value = jnp.sum(jnp.power(self.condition, exp) * jnp.square(z))
        return value, jnp.array(0.0)


class Discus(BBOBProblem):
    """Discus Function (Hansen et al., 2010, p. 55)."""

    name = "discus"

    # Conditioning of the single distinguished axis.
    condition: float = 1e6

    def _value(self, x: jax.Array, params: BBOBParams) -> tuple[jax.Array, jax.Array]:
        z = params.r @ (x - params.x_opt)
        z = transform_osz(z)

        z_squared = jnp.square(z)

        value = jnp.sum(z_squared.at[0].multiply(self.condition))
        return value, jnp.array(0.0)


class BentCigar(BBOBProblem):
    """Bent Cigar Function (Hansen et al., 2010, p. 60)."""

    name = "bent_cigar"

    # Conditioning of all but the first axis.
    condition: float = 1e6
    # Asymmetry, the paper's beta.
    beta: float = 0.5

    def _value(self, x: jax.Array, params: BBOBParams) -> tuple[jax.Array, jax.Array]:
        z = params.r @ (x - params.x_opt)
        z = transform_asy(z, self.beta)
        z = params.r @ z

        z_squared = jnp.square(z)

        value = jnp.sum(z_squared.at[1:].multiply(self.condition))
        return value, jnp.array(0.0)


class SharpRidge(BBOBProblem):
    """Sharp Ridge Function (Hansen et al., 2010, p. 65)."""

    name = "sharp_ridge"

    # Conditioning.
    condition: float = 10.0
    # Weight of the ridge (COCO: `alpha` in f_sharp_ridge.c).
    alpha: float = 100.0

    def _value(self, x: jax.Array, params: BBOBParams) -> tuple[jax.Array, jax.Array]:
        z = params.r @ (x - params.x_opt)
        z = lambda_alpha(self.condition, self.num_dims) * z
        z = params.q @ z

        z_squared = jnp.square(z)

        value = z_squared[0] + self.alpha * jnp.sqrt(jnp.sum(z_squared[1:]))
        return value, jnp.array(0.0)


class DifferentPowers(BBOBProblem):
    """Different Powers Function (Hansen et al., 2010, p. 70)."""

    name = "different_powers"

    def _value(self, x: jax.Array, params: BBOBParams) -> tuple[jax.Array, jax.Array]:
        z = params.r @ (x - params.x_opt)

        exp = 2.0 + 4.0 * jnp.arange(self.num_dims) / (self.num_dims - 1)

        value = jnp.sqrt(jnp.sum(jnp.power(jnp.abs(z), exp)))
        return value, jnp.array(0.0)


# --- Part 4: Multi-modal functions with adequate global structure -------------


class RastriginRotated(BBOBProblem):
    """Rastrigin Function, rotated (Hansen et al., 2010, p. 75)."""

    name = "rastrigin_rotated"

    # Conditioning.
    condition: float = 10.0
    # Asymmetry, the paper's beta.
    beta: float = 0.2

    def _value(self, x: jax.Array, params: BBOBParams) -> tuple[jax.Array, jax.Array]:
        z = params.r @ (x - params.x_opt)
        z = transform_asy(transform_osz(z), self.beta)
        z = params.q @ z
        z = lambda_alpha(self.condition, self.num_dims) * z
        z = params.r @ z

        oscillation = jnp.sum(jnp.cos(2 * jnp.pi * z))
        value = 10 * (self.num_dims - oscillation) + jnp.sum(jnp.square(z))
        return value, jnp.array(0.0)


class Weierstrass(BBOBProblem):
    """Weierstrass Function (Hansen et al., 2010, p. 80)."""

    name = "weierstrass"

    # Conditioning. The series flattens the landscape, so the conditioning
    # enters inverted: COCO spells it `base = 1 / sqrt(condition)`.
    condition: float = 100.0
    # Summands of the truncated series (COCO: `F_WEIERSTRASS_SUMMANDS`).
    k_order: int = 12
    # Multiplier on the boundary penalty, divided by the dimension.
    penalty_factor: float = 10.0

    # The series coefficients and f_0, built once. With numpy rather than jnp:
    # a jnp constant would freeze its dtype when the class is created, silently
    # staying float32 under `jax_enable_x64` and costing precision.
    _half_pow_k = np.power(0.5, np.arange(k_order))
    _three_pow_k = np.power(3.0, np.arange(k_order))
    _f_0 = np.sum(_half_pow_k * np.cos(np.pi * _three_pow_k))

    def _value(self, x: jax.Array, params: BBOBParams) -> tuple[jax.Array, jax.Array]:
        z = params.r @ (x - params.x_opt)
        z = transform_osz(z)
        z = params.q @ z
        z = lambda_alpha(1.0 / self.condition, self.num_dims) * z
        z = params.r @ z

        out = jnp.sum(
            self._half_pow_k
            * jnp.cos(2 * jnp.pi * self._three_pow_k * (z[:, None] + 0.5))[:, None]
        )

        value = 10 * (out / self.num_dims - self._f_0) ** 3
        return value, self.penalty_factor * f_pen(x) / self.num_dims


class SchaffersF7(BBOBProblem):
    """Schaffers F7 Function (Hansen et al., 2010, p. 85)."""

    name = "schaffers_f7"

    # Conditioning; the only thing f18 changes.
    condition: float = 10.0
    # Asymmetry, the paper's beta.
    beta: float = 0.5
    # Multiplier on the boundary penalty.
    penalty_factor: float = 10.0

    def _value(self, x: jax.Array, params: BBOBParams) -> tuple[jax.Array, jax.Array]:
        z = params.r @ (x - params.x_opt)
        z = transform_asy(z, self.beta)
        z = params.q @ z
        z = lambda_alpha(self.condition, self.num_dims) * z

        z_i = z[:-1]
        z_ip1 = jnp.roll(z, -1)[:-1]
        s = jnp.sqrt(jnp.square(z_i) + jnp.square(z_ip1))

        out = jnp.sum(jnp.sqrt(s) + jnp.sqrt(s) * jnp.sin(50 * s**0.2) ** 2)

        # The sum runs over D - 1 consecutive pairs.
        value = (out / (self.num_dims - 1.0)) ** 2
        return value, self.penalty_factor * f_pen(x)


class SchaffersF7IllConditioned(SchaffersF7):
    """Schaffers F7 Function, ill-conditioned (Hansen et al., 2010, p. 90).

    The same landscape as `SchaffersF7` with the conditioning raised from 10 to
    1000 -- which is exactly how the official suite defines f18 against f17.
    """

    name = "schaffers_f7_ill_conditioned"

    condition: float = 1000.0


class GriewankRosenbrock(BBOBProblem):
    """Composite Griewank-Rosenbrock Function F8F2 (Hansen et al., 2010, p. 95)."""

    name = "griewank_rosenbrock"

    def _value(self, x: jax.Array, params: BBOBParams) -> tuple[jax.Array, jax.Array]:
        # Same deliberate parameterization as RosenbrockRotated: official BBOB
        # has z = s*R*x + 1/2 with an R-derived optimum; this form is verified
        # exactly equivalent (x_opt = R^T (0.5/s) 1 reproduces it bit-for-bit)
        # and makes params.x_opt the argmin.
        scale = jnp.maximum(1.0, jnp.sqrt(self.num_dims) / 8.0)
        z = scale * (params.r @ (x - params.x_opt)) + 1.0
        z_i = z[:-1]
        z_ip1 = jnp.roll(z, -1)[:-1]

        s = 100.0 * jnp.square(jnp.square(z_i) - z_ip1) + jnp.square(z_i - 1)
        out = jnp.sum(s / 4000.0 - jnp.cos(s))

        value = 10.0 * out / (self.num_dims - 1) + 10
        return value, jnp.array(0.0)


# --- Part 5: Multi-modal functions with weak global structure -----------------


class Schwefel(BBOBProblem):
    """Schwefel Function (Hansen et al., 2010, p. 100)."""

    name = "schwefel"

    # Conditioning.
    condition: float = 10.0
    # The constant Schwefel is built around; the optimum sits at +-half of it.
    # (The C source carries ...4637 in one literal; the paper, bbobbenchmarks
    # and C's own best_parameter all say ...4633.)
    constant: float = 4.2096874633
    # Multiplier on the boundary penalty, which acts on the scaled z.
    penalty_factor: float = 100.0

    def _sample_x_opt(self, key: jax.Array) -> jax.Array:
        """Draw +-`constant`/2 per coordinate: only the sign is an instance choice."""
        is_positive = jax.random.bernoulli(key, shape=(self.num_dims,))
        return jnp.where(is_positive, self.constant / 2.0, -self.constant / 2.0)

    def _value(self, x: jax.Array, params: BBOBParams) -> tuple[jax.Array, jax.Array]:
        # params.x_opt is the true optimum, drawn by _sample_x_opt.
        x_opt = params.x_opt
        sign = jnp.where(x_opt > 0.0, 1.0, -1.0)

        x_hat = 2.0 * sign * x
        x_hat_im1 = jnp.roll(x_hat, 1).at[0].set(0.0)
        x_opt_im1 = jnp.roll(x_opt, 1).at[0].set(0.0)
        z_hat = x_hat + 0.25 * (x_hat_im1 - 2 * jnp.abs(x_opt_im1))
        z = 100 * (
            lambda_alpha(self.condition, self.num_dims) * (z_hat - 2 * jnp.abs(x_opt))
            + 2 * jnp.abs(x_opt)
        )

        out = jnp.sum(z * jnp.sin(jnp.sqrt(jnp.abs(z))))

        value = -(out / (100.0 * self.num_dims)) + 4.189828872724339
        return value, self.penalty_factor * f_pen(z / 100)


class _Gallagher(BBOBProblem):
    """Gallagher's Gaussian peaks, shared by the 101-me and 21-hi variants.

    The peak layout -- which conditioning goes to which peak, how each one is
    permuted, and where the peaks sit -- is instance data, drawn from the
    instance's own key. (An earlier version derived it from
    `fold_in(key(0), q[0, 0])`; `fold_in` int-casts the float and `|q00| <= 1`
    for any rotation, so every rotated instance shared one frozen layout.)
    `fold` separates the two variants' layouts.
    """

    # Number of Gaussian peaks.
    num_optima: int
    # Conditioning of the global (first) peak.
    first_condition: float
    # Half-width of the box the non-global peaks are drawn from.
    y_range: float
    # Tag folded into the instance key, so the two variants differ.
    fold: int
    # Largest conditioning among the local peaks (COCO: `maxcondition`).
    max_condition: float = 1000.0
    # Multiplier on the boundary penalty.
    penalty_factor: float = 1.0

    def _value(self, x: jax.Array, params: BBOBParams) -> tuple[jax.Array, jax.Array]:
        key = jax.random.fold_in(params.key, self.fold)
        peaks = jnp.arange(self.num_optima)

        w = jnp.where(
            peaks == 0,
            10.0,
            1.1 + 8.0 * (peaks - 1.0) / (self.num_optima - 2.0),
        )

        condition_set = jnp.power(
            self.max_condition,
            2.0 * jnp.arange(self.num_optima - 1) / (self.num_optima - 2),
        )
        key, subkey = jax.random.split(key)
        conditions = (
            jnp.zeros(self.num_optima)
            .at[0]
            .set(self.first_condition)
            .at[1:]
            .set(jax.random.permutation(subkey, condition_set))
        )
        # C_i is diagonal, so carry the diagonal alone: (num_optima, D) rather
        # than (num_optima, D, D).
        c = jax.vmap(lambda alpha: lambda_alpha(alpha, self.num_dims) / alpha**0.25)(
            conditions
        )

        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, self.num_optima)
        c = jax.vmap(lambda c_i, k: c_i[jax.random.permutation(k, self.num_dims)])(
            c, keys
        )

        key, subkey = jax.random.split(key)
        y = jax.random.uniform(
            subkey,
            shape=(self.num_optima, self.num_dims),
            minval=-self.y_range,
            maxval=self.y_range,
        )
        y = y.at[0].set(params.x_opt)

        # (x - y_i)^T R^T C_i R (x - y_i) = z^T C_i z with z = R(x - y_i), and
        # C_i diagonal makes that sum(c_i * z**2). All peaks at once: one
        # (num_optima, D) by (D, D) matmul instead of three matrix-vector
        # products per peak.
        z = (x - y) @ params.r.T
        out = w * jnp.exp(-jnp.sum(c * jnp.square(z), axis=-1) / (2 * self.num_dims))

        value = jnp.square(transform_osz(10.0 - jnp.max(out)))
        return value, self.penalty_factor * f_pen(x)


class Gallagher101Me(_Gallagher):
    """Gallagher's Gaussian 101-me Peaks Function (Hansen et al., 2010, p. 105)."""

    name = "gallagher_101_me"

    num_optima = 101
    first_condition = 1000.0
    y_range = 5.0
    fold = 21


class Gallagher21Hi(_Gallagher):
    """Gallagher's Gaussian 21-hi Peaks Function (Hansen et al., 2010, p. 110)."""

    name = "gallagher_21_hi"

    num_optima = 21
    first_condition = 1000.0**2
    y_range = 4.9
    fold = 22

    def _sample_x_opt(self, key: jax.Array) -> jax.Array:
        """Draw from [-3.92, 3.92]^D, the 21-hi global peak's range (paper 5.22)."""
        return 0.98 * super()._sample_x_opt(key)


class Katsuura(BBOBProblem):
    """Katsuura Function (Hansen et al., 2010, p. 115)."""

    name = "katsuura"

    # Conditioning.
    condition: float = 100.0
    # Terms of the inner series, exactly as official. The powers are computed
    # in floating point (2.0**j), so there is no integer overflow; in float32
    # the j > 24 terms are below mantissa resolution and contribute harmless
    # near-zero noise, while in float64 all 32 are needed for exactness (J=30
    # costs ~1e-7 absolute).
    num_terms: int = 32
    # Multiplier on the boundary penalty.
    penalty_factor: float = 1.0

    def _value(self, x: jax.Array, params: BBOBParams) -> tuple[jax.Array, jax.Array]:
        z = params.r @ (x - params.x_opt)
        z = lambda_alpha(self.condition, self.num_dims) * z
        z = params.q @ z

        two_pow_j = jnp.power(2.0, jnp.arange(1, self.num_terms + 1))
        out = jnp.sum(
            jnp.abs(two_pow_j * z[:, None] - jnp.round(two_pow_j * z[:, None]))
            / two_pow_j,
            axis=1,
        )
        prod = jnp.prod(1.0 + jnp.arange(1, self.num_dims + 1) * out)

        value = (10.0 / self.num_dims**2) * (
            jnp.power(prod, 10.0 / self.num_dims**1.2) - 1.0
        )
        return value, self.penalty_factor * f_pen(x)


class Lunacek(BBOBProblem):
    """Lunacek bi-Rastrigin Function (Hansen et al., 2010, p. 120)."""

    name = "lunacek"

    # Conditioning.
    condition: float = 100.0
    # Centre of the first sphere; the optimum sits at +-half of it.
    mu_0: float = 2.5
    # Depth of the second sphere (COCO: `d`).
    depth: float = 1.0
    # Multiplier on the boundary penalty.
    penalty_factor: float = 1e4

    def _sample_x_opt(self, key: jax.Array) -> jax.Array:
        """Draw +-`mu_0`/2 per coordinate: only the sign is an instance choice."""
        is_positive = jax.random.bernoulli(key, shape=(self.num_dims,))
        return jnp.where(is_positive, self.mu_0 / 2.0, -self.mu_0 / 2.0)

    def _value(self, x: jax.Array, params: BBOBParams) -> tuple[jax.Array, jax.Array]:
        # COCO spells s as `1 - 0.5 / (sqrt(D + 20) - 4.1)`, which is the same
        # number (f_lunacek_bi_rastrigin.c:41); this is the paper's form.
        s = 1.0 - 1.0 / (2.0 * jnp.sqrt(self.num_dims + 20.0) - 8.2)
        mu_1 = -jnp.sqrt((self.mu_0**2 - self.depth) / s)

        # params.x_opt is the true optimum, drawn by _sample_x_opt.
        x_hat = 2.0 * jnp.sign(params.x_opt) * x

        z = params.r @ (x_hat - self.mu_0)
        z = lambda_alpha(self.condition, self.num_dims) * z
        z = params.q @ z

        s_1 = jnp.sum(jnp.square(x_hat - self.mu_0))
        s_2 = jnp.sum(jnp.square(x_hat - mu_1))
        oscillation = jnp.sum(jnp.cos(2 * jnp.pi * z))

        value = jnp.minimum(s_1, self.depth * self.num_dims + s * s_2) + 10.0 * (
            self.num_dims - oscillation
        )
        return value, self.penalty_factor * f_pen(x)


# The 24 standard BBOB functions, in canonical f1-f24 order.
_PROBLEMS: tuple[type[BBOBProblem], ...] = (
    # Part 1: Separable functions
    Sphere,
    Ellipsoidal,
    Rastrigin,
    BuecheRastrigin,
    LinearSlope,
    # Part 2: Functions with low or moderate conditioning
    AttractiveSector,
    StepEllipsoidal,
    Rosenbrock,
    RosenbrockRotated,
    # Part 3: Functions with high conditioning and unimodal
    EllipsoidalRotated,
    Discus,
    BentCigar,
    SharpRidge,
    DifferentPowers,
    # Part 4: Multi-modal functions with adequate global structure
    RastriginRotated,
    Weierstrass,
    SchaffersF7,
    SchaffersF7IllConditioned,
    GriewankRosenbrock,
    # Part 5: Multi-modal functions with weak global structure
    Schwefel,
    Gallagher101Me,
    Gallagher21Hi,
    Katsuura,
    Lunacek,
)

BBOB_PROBLEMS: dict[str, type[BBOBProblem]] = {
    problem.name: problem for problem in _PROBLEMS
}

# The dimensions COCO's bbob suite enumerates, from `suite_bbob.c`:
# `const size_t dimensions[] = { 2, 3, 5, 10, 20, 40 };`.
#
# Array shapes are static in JAX, so a problem fixes its dimension and callers
# that want several loop over these in Python. For meta-learning that loop is
# an advantage over sampling a dimension: every meta-step sees every dimension,
# which is stratified rather than noisy, and there are only six compilations to
# cache. It does require the learned parameters to be dimension-independent.
DIMENSIONS: tuple[int, ...] = (2, 3, 5, 10, 20, 40)


def suite(names: list[str] | None = None, **kwargs) -> dict[str, BBOBProblem]:
    """Build the standard BBOB functions as individual problems.

    Args:
        names: Which functions to include; defaults to all 24, in the
            canonical f1-f24 order.
        **kwargs: Passed to every problem (`num_dims`, `noise`, ...).

    Returns:
        A mapping from function name to problem. Loop over it to cover the
        suite: each problem compiles separately, so nothing pays for dispatch.

    Raises:
        KeyError: If `names` contains something that is not a BBOB function.

    """
    names = list(BBOB_PROBLEMS) if names is None else names

    unknown = [name for name in names if name not in BBOB_PROBLEMS]
    if unknown:
        raise KeyError(
            f"not BBOB functions: {unknown}; available: {sorted(BBOB_PROBLEMS)}"
        )

    return {name: BBOB_PROBLEMS[name](**kwargs) for name in names}
