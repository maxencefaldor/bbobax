"""Black-box Optimization Benchmarking Task."""

import dataclasses

import jax
import jax.numpy as jnp

from .fitness_fns import BBOB_FNS, X_OPT_CONVENTIONS
from .noise import NoiseModel
from .types import (
    BBOBEval,
    BBOBParams,
    BBOBState,
    DescriptorFn,
    FitnessFn,
    QDBBOBEval,
    QDBBOBParams,
)


class BBOB:
    """One BBOB problem: one function at one dimension.

    That is COCO's own structure -- a suite enumerates function x dimension x
    instance, and only the instance is drawn. A task here fixes the function
    and the dimension; `sample` draws an instance of it.

    To cover many functions or many dimensions, hold many tasks and loop over
    them (`bbobax.suite` builds the standard 24). Under `jit` that loop
    unrolls, so each task keeps its own compiled code and nothing pays for
    dispatch -- unlike a single task that switches over functions, which under
    `vmap` must evaluate every branch for every solution.
    """

    def __init__(
        self,
        fitness_fn: str | FitnessFn,
        num_dims: int = 10,
        x_range: tuple[float, float] = (-5.0, 5.0),
        x_opt_range: tuple[float, float] = (-4.0, 4.0),
        f_opt_range: tuple[float, float] = (0.0, 0.0),
        clip_x: bool = False,
        sample_rotation: bool = True,
        noise_config: dict | None = None,
    ):
        """Initialize the BBOB task.

        Args:
            fitness_fn: The name of a standard BBOB function (a key of
                `BBOB_FNS`), or a callable of your own. A name also selects
                the function's x_opt convention, so `params.x_opt` is the
                true argmin; a bare callable gets the raw draw.
            num_dims: The problem dimension, at least 2 as BBOB requires.
            x_range: Range of input variables.
            x_opt_range: Range the raw optimum is drawn from, before the
                function's convention reshapes it. BBOB uses [-4, 4].
            f_opt_range: Range of optimal fitness values. The default pins
                f_opt to 0; official BBOB draws a 2-decimal Cauchy clipped to
                +-1000 -- configure that explicitly if you want it.
            clip_x: Whether to clip input variables. Official BBOB never
                clips: boundary handling is the in-function penalty.
            sample_rotation: Whether to sample rotation matrices. BBOB always
                rotates; with False, R = Q = I and every rotated variant
                collapses onto its axis-aligned base function (measured:
                f10 becomes f2 exactly). Only disable this deliberately.
            noise_config: Configuration for noise models. The default is the
                plain noiseless suite with no stabilization, exactly COCO's
                noiseless BBOB; pass a config to opt into noise.

        Raises:
            KeyError: If `fitness_fn` names no standard BBOB function.
            ValueError: If `num_dims` is below 2.

        """
        if isinstance(fitness_fn, str):
            if fitness_fn not in BBOB_FNS:
                raise KeyError(
                    f"{fitness_fn!r} is not a BBOB function; "
                    f"available: {sorted(BBOB_FNS)}"
                )
            self.name = fitness_fn
            self.fitness_fn = BBOB_FNS[fitness_fn]
            self.x_opt_convention = X_OPT_CONVENTIONS.get(
                fitness_fn, lambda x_opt: x_opt
            )
        else:
            self.name = getattr(fitness_fn, "__name__", "custom")
            self.fitness_fn = fitness_fn
            self.x_opt_convention = lambda x_opt: x_opt

        if num_dims < 2:
            raise ValueError(f"BBOB is defined for num_dims >= 2; got {num_dims}")
        self.num_dims = num_dims
        self.x_range = x_range
        self.x_opt_range = x_opt_range
        self.f_opt_range = f_opt_range
        self.clip_x = clip_x
        self.sample_rotation = sample_rotation

        # Noise: default is the plain noiseless suite -- exactly COCO's
        # noiseless BBOB. Noise (and its stabilization) is opt-in.
        if noise_config is None:
            noise_config = {
                "noise_model_names": ("noiseless",),
                "use_stabilization": False,
            }
        self.noise_model = NoiseModel(**noise_config)

    def sample(self, key: jax.Array) -> BBOBParams:
        """Sample an instance of this problem.

        The raw uniform x_opt draw is reshaped by the function's own
        convention (sign vectors, scalings, sign-forcing -- see
        `X_OPT_CONVENTIONS`), so `params.x_opt` is the true argmin, the
        same invariant COCO keeps by storing the post-convention optimum.

        Args:
            key: JAX random key.

        Returns:
            The instance's parameters.

        """
        key_x, key_f, key_r, key_q, key_noise, key_instance = jax.random.split(key, 6)

        x_opt = self.x_opt_convention(
            jax.random.uniform(
                key_x,
                shape=(self.num_dims,),
                minval=self.x_opt_range[0],
                maxval=self.x_opt_range[1],
            )
        )
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

        noise_params = self.noise_model.sample(key_noise, self.num_dims)

        return BBOBParams(x_opt, f_opt, r, q, key_instance, noise_params)

    def init(self, params: BBOBParams) -> BBOBState:
        """Initialize the evaluation state for an instance.

        Args:
            params: Instance parameters.

        Returns:
            Initial task state.

        """
        return BBOBState(counter=0)

    def evaluate(
        self,
        key: jax.Array,
        x: jax.Array,
        state: BBOBState,
        params: BBOBParams,
    ) -> tuple[BBOBState, BBOBEval]:
        """Evaluate the fitness of a solution.

        Args:
            key: JAX random key.
            x: Input solution, shape `(num_dims,)`.
            state: Current task state.
            params: Instance parameters.

        Returns:
            Updated state and evaluation results.

        """
        if self.clip_x:
            x = jnp.clip(x, self.x_range[0], self.x_range[1])

        fn_val, fn_pen = self.fitness_fn(x, state, params)

        # Noise applies to the raw value alone; the boundary penalty and f_opt
        # are added outside it, as the noisy-functions paper prescribes.
        fn_noise = self.noise_model.apply(key, fn_val, params.noise_params)
        fitness = fn_noise + fn_pen + params.f_opt

        state = dataclasses.replace(state, counter=state.counter + 1)
        return state, BBOBEval(fitness=fitness)

    def sample_x(self, key: jax.Array) -> jax.Array:
        """Sample a random solution.

        Args:
            key: JAX random key.

        Returns:
            Random solution within the defined range, shape `(num_dims,)`.

        """
        return jax.random.uniform(
            key,
            shape=(self.num_dims,),
            minval=self.x_range[0],
            maxval=self.x_range[1],
        )

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


class QDBBOB(BBOB):
    """One QD-BBOB problem: a BBOB function paired with a descriptor.

    The Quality-Diversity extension is bbobax's own; COCO has no descriptor
    notion.
    """

    def __init__(
        self,
        fitness_fn: str | FitnessFn,
        descriptor_fn: DescriptorFn,
        descriptor_size: int = 2,
        **kwargs,
    ):
        """Initialize the QD-BBOB task.

        Args:
            fitness_fn: The name of a standard BBOB function, or a callable.
            descriptor_fn: The descriptor function.
            descriptor_size: Dimensionality of the descriptor.
            **kwargs: Additional arguments for BBOB.

        """
        super().__init__(fitness_fn=fitness_fn, **kwargs)
        self.descriptor_fn = descriptor_fn
        self.descriptor_size = descriptor_size

    def sample(self, key: jax.Array) -> QDBBOBParams:
        """Sample an instance, including its descriptor projection.

        Args:
            key: JAX random key.

        Returns:
            The instance's parameters.

        """
        key_base, key_descriptor = jax.random.split(key)
        base_params = super().sample(key_base)
        # Shallow field copy: `dataclasses.asdict` would recurse and turn the
        # nested NoiseParams into a plain dict.
        base = {
            field.name: getattr(base_params, field.name)
            for field in dataclasses.fields(base_params)
        }
        return QDBBOBParams(
            **base,
            descriptor_params=self.generate_gaussian_projection(key_descriptor),
        )

    def evaluate(
        self,
        key: jax.Array,
        x: jax.Array,
        state: BBOBState,
        params: QDBBOBParams,
    ) -> tuple[BBOBState, QDBBOBEval]:
        """Evaluate the fitness and descriptor of a solution.

        Args:
            key: JAX random key.
            x: Input solution, shape `(num_dims,)`.
            state: Current task state.
            params: Instance parameters.

        Returns:
            Updated state and evaluation results.

        """
        state, bbob_eval = super().evaluate(key, x, state, params)
        descriptor = self.descriptor_fn(x, state, params)
        return state, QDBBOBEval(fitness=bbob_eval.fitness, descriptor=descriptor)

    def generate_gaussian_projection(self, key: jax.Array) -> jax.Array:
        """Generate the instance's random Gaussian projection matrix.

        Entries are `N(0, 1) / sqrt(descriptor_size)`, so the projection
        preserves expected squared norms.

        Args:
            key: JAX random key.

        Returns:
            A `(descriptor_size, num_dims)` matrix.

        """
        return jax.random.normal(
            key, shape=(self.descriptor_size, self.num_dims)
        ) / jnp.sqrt(self.descriptor_size)


def suite(names: list[str] | None = None, **kwargs) -> dict[str, BBOB]:
    """Build the standard BBOB functions as individual tasks.

    Args:
        names: Which functions to include; defaults to all 24, in the
            canonical f1-f24 order.
        **kwargs: Passed to every task (`num_dims`, `noise_config`, ...).

    Returns:
        A mapping from function name to task. Loop over it to cover the suite:
        each task compiles separately, so nothing pays for dispatch.

    """
    return {name: BBOB(name, **kwargs) for name in (names or list(BBOB_FNS))}
