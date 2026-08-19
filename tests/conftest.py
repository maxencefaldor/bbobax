"""Pytest fixtures for BBOBax tests."""

import jax
import jax.numpy as jnp
import pytest

from bbobax.noise import NoiselessParams
from bbobax.problem import BBOBParams

# The whole suite runs in float64: the alignment tests (tests/test_alignment.py)
# compare against the official 2009 implementation at 1e-9 relative error, and
# the optimum tests assert f(x_opt) == 0 to 1e-9 -- neither is meaningful in
# float32. conftest is imported before any test module, so this applies
# everywhere.
jax.config.update("jax_enable_x64", True)


def noiseless_params() -> NoiselessParams:
    """Return the default model's parameters; `Noiseless` draws nothing."""
    return NoiselessParams()


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
            noise_params=noiseless_params(),
        )

    return _get_mock_params
