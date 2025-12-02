"""Black-box Optimization Benchmarking Task."""

from collections.abc import Callable

import jax
import jax.numpy as jnp

from .functions import bbob_fns
from .noise import NoiseModel
from .types import BBOBEval, BBOBParams, BBOBState, QDBBOBEval, QDBBOBParams


class BBOB:
    """Black-box Optimization Benchmarking Task class (Single Objective)."""

    def __init__(
        self,
        fitness_fns: list[Callable[[jax.Array, BBOBState, BBOBParams], jax.Array]]
        | dict[str, Callable[[jax.Array, BBOBState, BBOBParams], jax.Array]],
        min_num_dims: int = 2,
        max_num_dims: int = 10,
        x_range: list[float] = [-5.0, 5.0],
        x_opt_range: list[float] = [-4.0, 4.0],
        f_opt_range: list[float] = [0.0, 0.0],
        clip_x: bool = False,
        sample_rotation: bool = False,
        noise_config: dict | None = None,
    ):
        """Initialize the BBOB task.

        Args:
            fitness_fns: List or dictionary of fitness functions.
            min_num_dims: Minimum number of dimensions.
            max_num_dims: Maximum number of dimensions.
            x_range: Range of input variables.
            x_opt_range: Range of optimal input variables.
            f_opt_range: Range of optimal fitness values.
            clip_x: Whether to clip input variables.
            sample_rotation: Whether to sample rotation matrices.
            noise_config: Configuration for noise models.

        """
        if isinstance(fitness_fns, dict):
            self.fitness_fns = list(fitness_fns.values())
        else:
            self.fitness_fns = fitness_fns

        self.min_num_dims = min_num_dims
        self.max_num_dims = max_num_dims
        self.x_range = x_range
        self.x_opt_range = x_opt_range
        self.f_opt_range = f_opt_range
        self.clip_x = clip_x
        self.sample_rotation = sample_rotation

        # Noise
        if noise_config is None:
            noise_config = {
                "noise_models": [
                    "noiseless",
                    "gaussian",
                    "uniform",
                    "cauchy",
                    "additive",
                ],
                "use_stabilization": True,
            }
        self.noise_model = NoiseModel(**noise_config)

        self.num_fns = len(self.fitness_fns)

        # Prepare vectorized fitness evaluation
        self._vmapped_fitness_fns = [
            jax.vmap(fn, in_axes=(0, None, None)) for fn in self.fitness_fns
        ]

    def sample(self, key: jax.Array) -> BBOBParams:
        """Sample BBOB task parameters."""
        key_fn, key_d, key_x, key_f, key_noise = jax.random.split(key, 5)

        fn_id = jax.random.randint(key_fn, (), minval=0, maxval=self.num_fns)
        num_dims = jax.random.randint(
            key_d, (), minval=self.min_num_dims, maxval=self.max_num_dims
        )

        x_opt = jax.random.uniform(
            key_x,
            shape=(self.max_num_dims,),
            minval=self.x_opt_range[0],
            maxval=self.x_opt_range[1],
        )
        f_opt = jax.random.uniform(
            key_f,
            minval=self.f_opt_range[0],
            maxval=self.f_opt_range[1],
        )

        # Sample noise model parameters
        noise_params = self.noise_model.sample(key_noise)

        return BBOBParams(fn_id, num_dims, x_opt, f_opt, noise_params)

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
            self._vmapped_fitness_fns,
            x,
            state,
            params,
        )

        # Apply noise
        fn_noise = self.noise_model.apply(key, fn_val, params.noise_params)

        # Add boundary handling penalty and optimal function value
        final_fitness = -(fn_noise + fn_pen + params.f_opt)

        task_eval = BBOBEval(fitness=final_fitness)
        return state.replace(counter=state.counter + 1), task_eval

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
        self, key: jax.Array, max_dims: int, num_dims: int
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

        # Extract diagonal and create sign correction matrix
        diagonal = jnp.diag(upper_triangular)
        sign_correction = jnp.diag(diagonal / jnp.abs(diagonal))

        # Apply sign correction
        rotation = orthogonal_matrix @ sign_correction

        # Ensure determinant is 1 by possibly flipping first row
        determinant = jnp.linalg.det(rotation)
        rotation = rotation.at[0].multiply(determinant)

        return rotation

    @classmethod
    def create_default(cls, **kwargs):
        """Create a BBOB task with all standard functions."""
        return cls(fitness_fns=bbob_fns, **kwargs)


class QDBBOB(BBOB):
    """Quality-Diversity Blackbox Optimization Benchmarking Task class."""

    def __init__(
        self,
        descriptor_fns: list[Callable[[jax.Array, BBOBState, QDBBOBParams], jax.Array]]
        | dict[str, Callable[[jax.Array, BBOBState, QDBBOBParams], jax.Array]],
        fitness_fns: list[Callable[[jax.Array, BBOBState, BBOBParams], jax.Array]]
        | dict[str, Callable[[jax.Array, BBOBState, BBOBParams], jax.Array]],
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

        # Vectorize descriptors
        self._vmapped_descriptor_fns = [
            jax.vmap(fn, in_axes=(0, None, None)) for fn in self.descriptor_fns
        ]

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
            params.descriptor_id, self._vmapped_descriptor_fns, x, state, params
        )

        task_eval = QDBBOBEval(fitness=bbob_eval.fitness, descriptor=descriptor)
        return state, task_eval

    def gaussian_random_projection(self, key: jax.Array, num_dims: int) -> jax.Array:
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
