"""Pytest fixtures for BBOBax tests."""

import jax
import jax.numpy as jnp
import pytest

from bbobax.types import BBOBParams, NoiseParams

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
def mock_params():
    """Create mock BBOB parameters factory.

    One problem = one function at one fixed dimension, so the instance carries
    no function id and no dimension: every array is exactly ``num_dims`` long.
    """

    def _get_mock_params(num_dims: int) -> BBOBParams:
        return BBOBParams(
            key=jax.random.key(0),
            x_opt=jnp.zeros(num_dims),
            f_opt=jnp.array(0.0),
            r=jnp.eye(num_dims),
            q=jnp.eye(num_dims),
            noise_params=zero_noise_params(),
        )

    return _get_mock_params
