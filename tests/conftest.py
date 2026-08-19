"""Pytest fixtures for BBOBax tests."""

import jax
import jax.numpy as jnp
import pytest

from bbobax.types import BBOBParams, BBOBState, NoiseParams, QDBBOBParams

# The whole suite runs in float64: the alignment tests (tests/test_alignment.py)
# compare against the official 2009 implementation at 1e-9 relative error, and
# the optimum tests assert f(x_opt) == 0 to 1e-9 -- neither is meaningful in
# float32. conftest is imported before any test module, so this applies
# everywhere.
jax.config.update("jax_enable_x64", True)


def zero_noise_params() -> NoiseParams:
    """All-zero noise parameters (noise_id 0 selects the first model)."""
    return NoiseParams(
        noise_id=jnp.array(0),
        gaussian_beta=jnp.array(0.0),
        uniform_alpha=jnp.array(0.0),
        uniform_beta=jnp.array(0.0),
        cauchy_alpha=jnp.array(0.0),
        cauchy_p=jnp.array(0.0),
        additive_std=jnp.array(0.0),
    )


@pytest.fixture
def mock_state():
    """Create a mock BBOB state factory."""

    def _get_mock_state(max_num_dims: int) -> BBOBState:
        return BBOBState(counter=0)

    return _get_mock_state


@pytest.fixture
def mock_params():
    """Create mock BBOB parameters factory."""

    def _get_mock_params(num_dims: int, max_num_dims: int) -> BBOBParams:
        return BBOBParams(
            fn_id=jnp.array(0),
            num_dims=jnp.array(num_dims),
            x_opt=jnp.zeros(max_num_dims),
            f_opt=jnp.array(0.0),
            r=jnp.eye(max_num_dims),
            q=jnp.eye(max_num_dims),
            key=jax.random.key(0),
            noise_params=zero_noise_params(),
        )

    return _get_mock_params


@pytest.fixture
def mock_qdbbob_params():
    """Create mock QD-BBOB parameters factory."""

    def _get_mock_qdbbob_params(
        num_dims: int, max_num_dims: int, descriptor_size: int
    ) -> QDBBOBParams:
        # Create random projection matrix
        key = jax.random.key(0)
        descriptor_params = jax.random.normal(key, (descriptor_size, max_num_dims))

        return QDBBOBParams(
            fn_id=jnp.array(0),
            num_dims=jnp.array(num_dims),
            x_opt=jnp.zeros(max_num_dims),
            f_opt=jnp.array(0.0),
            r=jnp.eye(max_num_dims),
            q=jnp.eye(max_num_dims),
            key=jax.random.key(0),
            noise_params=zero_noise_params(),
            descriptor_params=descriptor_params,
            descriptor_id=jnp.array(0),
        )

    return _get_mock_qdbbob_params
