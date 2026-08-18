"""Black-box Optimization Benchmarking Task."""

import jax
import jax.numpy as jnp

from .fitness_fns import X_OPT_CONVENTIONS, bbob_fns
from .noise import NoiseModel
from .types import (
    BBOBEval,
    BBOBParams,
    BBOBState,
    DescriptorFn,
    FitnessFn,
    IntScalar,
    QDBBOBEval,
    QDBBOBParams,
)


class BBOB:
    """Black-box Optimization Benchmarking Task class (Single Objective)."""

    def __init__(
        self,
        fitness_fns: list[FitnessFn] | dict[str, FitnessFn],
        min_num_dims: int = 2,
        max_num_dims: int = 10,
        x_range: tuple[float, float] = (-5.0, 5.0),
        x_opt_range: tuple[float, float] = (-4.0, 4.0),
        f_opt_range: tuple[float, float] = (0.0, 0.0),
        clip_x: bool = False,
        sample_rotation: bool = True,
        noise_config: dict | None = None,
    ):
        """Initialize the BBOB task.

        Args:
            fitness_fns: Dictionary of fitness functions by name (canonical:
                ``bbob_fns`` or a subset), or a bare list. Names drive the
                per-function x_opt conventions, so ``params.x_opt`` is always
                the true argmin; a bare list gets no conventions.
            min_num_dims: Minimum number of dimensions (inclusive).
            max_num_dims: Maximum number of dimensions (inclusive).
            x_range: Range of input variables.
            x_opt_range: Range of optimal input variables. BBOB draws x_opt in
                [-4, 4]; per-function conventions then reshape the draw.
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

        """
        if isinstance(fitness_fns, dict):
            self.fitness_fn_names = list(fitness_fns.keys())
            self.fitness_fns = list(fitness_fns.values())
        else:
            self.fitness_fn_names = None
            self.fitness_fns = fitness_fns

        # Per-function preparation of the raw x_opt draw (identity where no
        # convention exists, and everywhere for unnamed function lists).
        if self.fitness_fn_names is not None:
            self._x_opt_conventions = [
                X_OPT_CONVENTIONS.get(name, lambda x_opt: x_opt)
                for name in self.fitness_fn_names
            ]
        else:
            self._x_opt_conventions = [lambda x_opt: x_opt] * len(self.fitness_fns)

        self.min_num_dims = min_num_dims
        self.max_num_dims = max_num_dims
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

        self.num_fns = len(self.fitness_fns)

    def sample(self, key: jax.Array) -> BBOBParams:
        """Sample BBOB task parameters.

        The raw uniform x_opt draw is reshaped by the sampled function's
        convention (sign vectors, scalings, sign-forcing -- see
        ``X_OPT_CONVENTIONS``), so ``params.x_opt`` is the true argmin, the
        same invariant COCO keeps by storing the post-convention optimum.
        """
        key_fn, key_d, key_x, key_f, key_noise, key_instance = jax.random.split(key, 6)

        fn_id = jax.random.randint(key_fn, (), minval=0, maxval=self.num_fns)
        # randint's maxval is exclusive; both bounds here are inclusive.
        num_dims = jax.random.randint(
            key_d, (), minval=self.min_num_dims, maxval=self.max_num_dims + 1
        )

        x_opt = jax.random.uniform(
            key_x,
            shape=(self.max_num_dims,),
            minval=self.x_opt_range[0],
            maxval=self.x_opt_range[1],
        )
        x_opt = jax.lax.switch(fn_id, self._x_opt_conventions, x_opt)
        f_opt = jax.random.uniform(
            key_f,
            minval=self.f_opt_range[0],
            maxval=self.f_opt_range[1],
        )

        # Sample noise model parameters
        noise_params = self.noise_model.sample(key_noise, num_dims)

        return BBOBParams(fn_id, num_dims, x_opt, f_opt, key_instance, noise_params)

    def init(self, key: jax.Array, params: BBOBParams) -> BBOBState:
        """Initialize the task state.

        Args:
            key: JAX random key.
            params: Task parameters.

        Returns:
            Initial task state.

        """
        if self.sample_rotation:
            key_r, key_q = jax.random.split(key)
            r = self.generate_random_rotation(key_r, self.max_num_dims, params.num_dims)
            q = self.generate_random_rotation(key_q, self.max_num_dims, params.num_dims)
        else:
            r = jnp.eye(self.max_num_dims)
            q = jnp.eye(self.max_num_dims)
        return BBOBState(counter=0, r=r, q=q)

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
            x: Input solution.
            state: Current task state.
            params: Task parameters.

        Returns:
            Updated state and evaluation results.

        """
        if self.clip_x:
            x = jnp.clip(x, self.x_range[0], self.x_range[1])

        # Evaluate fitness
        # Using switch to select the correct fitness function based on fn_id
        fn_val, fn_pen = jax.lax.switch(
            params.fn_id,
            self.fitness_fns,
            x,
            state,
            params,
        )

        # Apply noise
        fn_noise = self.noise_model.apply(key, fn_val, params.noise_params)

        # Add boundary handling penalty and optimal function value
        final_fitness = fn_noise + fn_pen + params.f_opt

        bbob_eval = BBOBEval(fitness=final_fitness)
        return state.replace(counter=state.counter + 1), bbob_eval

    def sample_x(self, key: jax.Array) -> jax.Array:
        """Sample a random solution.

        Args:
            key: JAX random key.

        Returns:
            Random solution within the defined range.

        """
        return jax.random.uniform(
            key,
            shape=(self.max_num_dims,),
            minval=self.x_range[0],
            maxval=self.x_range[1],
        )

    def generate_random_rotation(
        self, key: jax.Array, max_dims: int, num_dims: IntScalar
    ) -> jax.Array:
        """Generate a random (n, n) rotation matrix uniformly sampled from SO(n)."""
        # Generate fixed-size random normal matrix but mask based on num_dims
        random_matrix = jax.random.normal(key, (max_dims, max_dims))
        mask = (jnp.arange(max_dims)[:, None] < num_dims) & (
            jnp.arange(max_dims)[None, :] < num_dims
        )
        random_matrix = jnp.where(mask, random_matrix, 0.0)

        # Add identity matrix for masked region to ensure valid QR decomposition
        random_matrix = random_matrix + jnp.where(~mask, jnp.eye(max_dims), 0.0)

        # QR decomposition
        orthogonal_matrix, upper_triangular = jnp.linalg.qr(random_matrix)

        # Extract diagonal and create sign correction matrix (zero-safe: a zero
        # diagonal entry has measure zero but would otherwise produce NaN)
        diagonal = jnp.diag(upper_triangular)
        sign_correction = jnp.diag(jnp.where(diagonal == 0.0, 1.0, jnp.sign(diagonal)))

        # Apply sign correction
        rotation = orthogonal_matrix @ sign_correction

        # Ensure determinant is 1 by possibly flipping first row
        determinant = jnp.linalg.det(rotation)
        rotation = rotation.at[0].multiply(determinant)

        return rotation

    @classmethod
    def create_default(cls, **kwargs):
        """Create a BBOB task with all 24 standard functions."""
        return cls(fitness_fns=bbob_fns, **kwargs)


class QDBBOB(BBOB):
    """Quality-Diversity Black-box Optimization Benchmarking Task class."""

    def __init__(
        self,
        descriptor_fns: list[DescriptorFn] | dict[str, DescriptorFn],
        fitness_fns: list[FitnessFn] | dict[str, FitnessFn],
        descriptor_size: int = 2,
        **kwargs,
    ):
        """Initialize the QD-BBOB task.

        Args:
            descriptor_fns: List or dictionary of descriptor functions.
            fitness_fns: List or dictionary of fitness functions.
            descriptor_size: Size of the descriptor vector.
            **kwargs: Additional arguments for BBOB.

        """
        super().__init__(fitness_fns=fitness_fns, **kwargs)

        if isinstance(descriptor_fns, dict):
            self.descriptor_fns = list(descriptor_fns.values())
        else:
            self.descriptor_fns = descriptor_fns

        self.descriptor_size = descriptor_size
        self.num_descriptors = len(self.descriptor_fns)

    def sample(self, key: jax.Array) -> QDBBOBParams:
        """Sample BBOB task parameters including descriptor params."""
        key_base, key_desc_id, key_desc_params = jax.random.split(key, 3)

        base_params = super().sample(key_base)

        desc_id = jax.random.randint(
            key_desc_id, (), minval=0, maxval=self.num_descriptors
        )

        # Descriptor params
        descriptor_params = self.gaussian_random_projection(
            key_desc_params, base_params.num_dims
        )

        return QDBBOBParams(
            fn_id=base_params.fn_id,
            num_dims=base_params.num_dims,
            x_opt=base_params.x_opt,
            f_opt=base_params.f_opt,
            key=base_params.key,
            noise_params=base_params.noise_params,
            descriptor_params=descriptor_params,
            descriptor_id=desc_id,
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
            x: Input solution.
            state: Current task state.
            params: Task parameters.

        Returns:
            Updated state and evaluation results.

        """
        state, bbob_eval = super().evaluate(key, x, state, params)

        descriptor = jax.lax.switch(
            params.descriptor_id, self.descriptor_fns, x, state, params
        )

        bbob_eval = QDBBOBEval(fitness=bbob_eval.fitness, descriptor=descriptor)
        return state, bbob_eval

    def gaussian_random_projection(
        self, key: jax.Array, num_dims: IntScalar
    ) -> jax.Array:
        """Generate a random Gaussian projection matrix.

        Args:
            key: JAX random key.
            num_dims: Number of dimensions.

        Returns:
            Random projection matrix.

        """
        descriptor_params = jax.random.normal(
            key,
            shape=(self.descriptor_size, self.max_num_dims),
        ) / jnp.sqrt(self.descriptor_size)
        mask = jnp.arange(self.max_num_dims) < num_dims
        descriptor_params = jnp.where(mask, descriptor_params, 0)
        return descriptor_params

    @classmethod
    def create_default(cls, descriptor_fns, descriptor_size: int = 2, **kwargs):
        """Create a QD-BBOB task with all 24 standard functions.

        Args:
            descriptor_fns: List or dictionary of descriptor functions.
            descriptor_size: Size of the descriptor vector.
            **kwargs: Additional arguments for BBOB.

        """
        return cls(
            descriptor_fns=descriptor_fns,
            fitness_fns=bbob_fns,
            descriptor_size=descriptor_size,
            **kwargs,
        )
