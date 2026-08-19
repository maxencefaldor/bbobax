"""The BBOB problem: one function at one dimension.

`BBOBProblem` is the contract every benchmark function in bbobax satisfies:

    problem = Sphere(num_dims=10)
    params = problem.sample(key)  # draw an instance
    evaluation = problem.evaluate(key, x, params)

Instance generation is *per function*, not merely per suite -- where the
optimum may sit, which rotations matter, what extra structure a landscape needs
-- so the function and the generation of its instances live in one object. A
subclass supplies `_value`, and overrides `_sample_x_opt` when its definition
constrains where the optimum can be.

There is deliberately no evaluation state. All 24 BBOB functions are
memoryless: the value at `x` does not depend on when `x` was asked. A dynamic
benchmark whose landscape moves with the evaluation count is a genuinely
different contract and would get its own, rather than a parameter these 24
carry and ignore.
"""

from typing import Any

import jax
import jax.numpy as jnp
from flax.struct import dataclass

from .noise import Noise, Noiseless


@dataclass
class BBOBParams:
    """One sampled instance of a problem.

    Everything that varies between instances lives here, drawn once by
    `BBOBProblem.sample` and never mutated. The function and the dimension are
    the *problem's*, not the instance's -- COCO enumerates those and draws only
    the instance, and so does bbobax.

    - `key` is the instance's own PRNG key, for instance structure that is
      cheaper to derive than to store -- the Gallagher functions draw their
      peak layouts from it.
    - `x_opt` is the function's true argmin, drawn by `_sample_x_opt`.
    - `r` and `q` are the instance's rotation matrices, the R and Q of the
      function definitions.
    - `noise_params` is whatever the problem's noise model's `sample` returns.
      A plugged-in component owns its own parameter type -- the same rule
      `QDParams.descriptor` follows for descriptors.
    """

    key: jax.Array
    x_opt: jax.Array
    f_opt: jax.Array
    r: jax.Array
    q: jax.Array
    noise_params: Any


@dataclass
class BBOBEval:
    """What evaluating a solution yields."""

    fitness: jax.Array


class BBOBProblem:
    """One BBOB function at one fixed dimension.

    That is COCO's own structure -- a suite enumerates function x dimension x
    instance, and only the instance is drawn. A problem here fixes the function
    and the dimension; `sample` draws an instance of it.

    To cover many functions or many dimensions, hold many problems and loop
    over them (`bbobax.suite` builds the standard 24, `bbobax.DIMENSIONS` lists
    the standard dimensions). Under `jit` that loop unrolls, so each problem
    keeps its own compiled code and nothing pays for dispatch -- unlike a
    single problem that switches over functions, which under `vmap` must
    evaluate every branch for every solution.

    Subclasses supply `_value`, and override `_sample_x_opt` when their own
    definition constrains where the optimum can be.
    """

    # The problem's name, and its key in `BBOB_PROBLEMS`.
    name: str = "bbob"

    def __init__(
        self,
        num_dims: int = 10,
        x_range: tuple[float, float] = (-5.0, 5.0),
        x_opt_range: tuple[float, float] = (-4.0, 4.0),
        f_opt_range: tuple[float, float] = (0.0, 0.0),
        clip_x: bool = False,
        sample_rotation: bool = True,
        noise: Noise | None = None,
    ):
        """Initialize the problem.

        Args:
            num_dims: The problem dimension, at least 2 as BBOB requires.
            x_range: Range of input variables.
            x_opt_range: Range the optimum is drawn from. BBOB uses [-4, 4].
                This is the *default* draw: a function whose definition pins its
                optimum elsewhere overrides `_sample_x_opt` and may ignore it.
            f_opt_range: Range of optimal fitness values. The default pins
                f_opt to 0; official BBOB draws a 2-decimal Cauchy clipped to
                +-1000 -- configure that explicitly if you want it.
            clip_x: Whether to clip input variables. Official BBOB never
                clips: boundary handling is the in-function penalty.
            sample_rotation: Whether to sample rotation matrices. BBOB always
                rotates; with False, R = Q = I and every rotated variant
                collapses onto its axis-aligned base function (measured:
                f10 becomes f2 exactly). Only disable this deliberately.
            noise: The noise model. The default is `Noiseless`, exactly
                COCO's noiseless BBOB; pass a model to opt into noise. The
                model is held, not switched on, so nothing evaluates a model
                this problem does not use.

        Raises:
            ValueError: If `num_dims` is below 2.

        """
        if num_dims < 2:
            raise ValueError(f"BBOB is defined for num_dims >= 2; got {num_dims}")
        self.num_dims = num_dims
        self.x_range = x_range
        self.x_opt_range = x_opt_range
        self.f_opt_range = f_opt_range
        self.clip_x = clip_x
        self.sample_rotation = sample_rotation

        # Default is the plain noiseless suite -- exactly COCO's noiseless
        # BBOB. Noise is opt-in.
        self.noise = Noiseless() if noise is None else noise

    def sample(self, key: jax.Array) -> BBOBParams:
        """Sample an instance of this problem.

        `params.x_opt` is always the true argmin, whatever `_sample_x_opt` had
        to do to draw it -- the invariant COCO keeps by storing the optimum its
        own construction ends up at.

        Args:
            key: JAX random key.

        Returns:
            The instance's parameters.

        """
        key_x, key_f, key_r, key_q, key_noise, key_instance = jax.random.split(key, 6)

        x_opt = self._sample_x_opt(key_x)
        f_opt = jax.random.uniform(
            key_f,
            minval=self.f_opt_range[0],
            maxval=self.f_opt_range[1],
        )

        # Rotation matrices: instance data like x_opt, drawn once and never
        # mutated. BBOB always rotates; with sample_rotation off both are the
        # identity and every rotated variant collapses onto its base function.
        if self.sample_rotation:
            r = self.generate_random_rotation(key_r, self.num_dims)
            q = self.generate_random_rotation(key_q, self.num_dims)
        else:
            r = jnp.eye(self.num_dims)
            q = jnp.eye(self.num_dims)

        noise_params = self.noise.sample(key_noise, self.num_dims)

        return BBOBParams(key_instance, x_opt, f_opt, r, q, noise_params)

    def evaluate(self, key: jax.Array, x: jax.Array, params: BBOBParams) -> BBOBEval:
        """Evaluate the fitness of a solution.

        Args:
            key: JAX random key, consumed by the noise model.
            x: Input solution, shape `(num_dims,)`.
            params: Instance parameters.

        Returns:
            The evaluation results.

        """
        if self.clip_x:
            x = jnp.clip(x, self.x_range[0], self.x_range[1])

        value, penalty = self._value(x, params)

        # Noise applies to the raw value alone; the boundary penalty and f_opt
        # are added outside it, as the noisy-functions paper prescribes.
        noisy = self.noise.apply(key, value, params.noise_params)
        return BBOBEval(fitness=noisy + penalty + params.f_opt)

    def sample_x(self, key: jax.Array) -> jax.Array:
        """Sample a random solution.

        Args:
            key: JAX random key.

        Returns:
            Random solution within `x_range`, shape `(num_dims,)`.

        """
        return jax.random.uniform(
            key,
            shape=(self.num_dims,),
            minval=self.x_range[0],
            maxval=self.x_range[1],
        )

    def _sample_x_opt(self, key: jax.Array) -> jax.Array:
        """Sample the instance's optimum.

        The default is a uniform draw from `x_opt_range`, which is what most of
        the 24 admit. A function whose definition constrains its optimum
        overrides this and draws what it actually needs -- `LinearSlope` draws a
        corner of the box, because a linear function has no interior minimum, so
        a continuous draw would be a coordinate it never uses.

        Args:
            key: JAX random key.

        Returns:
            The instance's true argmin, shape `(num_dims,)`.

        """
        return jax.random.uniform(
            key,
            shape=(self.num_dims,),
            minval=self.x_opt_range[0],
            maxval=self.x_opt_range[1],
        )

    def _value(self, x: jax.Array, params: BBOBParams) -> tuple[jax.Array, jax.Array]:
        """Return the raw function value and the boundary penalty at `x`.

        The value is 0 at the optimum: `f_opt` is added by `evaluate`, and
        noise applies to the value alone, so the two come back separately.

        Args:
            x: Input solution, shape `(num_dims,)`.
            params: Instance parameters.

        Returns:
            The value and the boundary penalty, both scalars.

        Raises:
            NotImplementedError: Always; a subclass supplies the function.

        """
        raise NotImplementedError(f"{type(self).__name__} defines no _value")

    @staticmethod
    def generate_random_rotation(key: jax.Array, num_dims: int) -> jax.Array:
        """Generate a random rotation matrix, Haar-uniform on SO(n).

        Args:
            key: JAX random key.
            num_dims: Size of the matrix.

        Returns:
            An orthogonal `(num_dims, num_dims)` matrix of determinant +1.

        """
        # QR of a Gaussian matrix with the sign correction that makes it Haar
        # (Mezzadri 2007). COCO orthonormalizes by Gram-Schmidt and lands on
        # O(n); forcing the determinant to +1 restricts this to SO(n), which
        # differs only in orientation.
        orthogonal_matrix, upper_triangular = jnp.linalg.qr(
            jax.random.normal(key, (num_dims, num_dims))
        )

        # Zero-safe: a zero diagonal entry has measure zero but would give NaN.
        diagonal = jnp.diag(upper_triangular)
        sign_correction = jnp.where(diagonal == 0.0, 1.0, jnp.sign(diagonal))
        rotation = orthogonal_matrix * sign_correction

        determinant = jnp.linalg.det(rotation)
        return rotation.at[0].multiply(determinant)
