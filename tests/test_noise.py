"""Tests for BBOBax noise models."""

import jax
import jax.numpy as jnp
import pytest

from bbobax.noise import NoiseModel, NoiseParams


def test_noise_model_initialization():
    """Test NoiseModel initialization with default and custom parameters."""
    # Default initialization
    model = NoiseModel()
    assert model.use_stabilization is False

    # Custom initialization
    custom_ranges = {
        "gaussian_beta": (0.1, 0.5),
        "uniform_alpha": None,
        "uniform_beta": None,
        "cauchy_alpha": None,
        "cauchy_p": None,
        "additive_std": None,
    }
    model_custom = NoiseModel(noise_ranges=custom_ranges, use_stabilization=True)
    assert model_custom.use_stabilization is True
    assert model_custom.noise_ranges["gaussian_beta"] == (0.1, 0.5)


def test_noise_model_sample():
    """Test sampling of noise parameters."""
    model = NoiseModel()
    key = jax.random.key(0)

    params = model.sample(key)

    assert isinstance(params, NoiseParams)
    assert params.noise_id.shape == ()
    assert params.gaussian_beta.shape == ()

    # Check ranges (basic check)
    assert params.gaussian_beta >= 0.01  # Default min
    assert params.gaussian_beta <= 1.0  # Default max


@pytest.mark.parametrize(
    "noise_type", ["noiseless", "gaussian", "uniform", "cauchy", "additive"]
)
def test_noise_model_apply(noise_type):
    """Test application of different noise models."""
    # Initialize model with only the specific noise type enabled to ensure we test it
    model = NoiseModel(noise_model_names=[noise_type])
    key = jax.random.key(0)

    # Sample parameters (noise_id should correspond to the single available model)
    params = model.sample(key)

    # Create dummy function value
    fn_val = jnp.array([10.0, 100.0])

    # Apply noise
    noisy_val = model.apply(key, fn_val, params)

    assert noisy_val.shape == fn_val.shape

    if noise_type == "noiseless":
        assert jnp.allclose(noisy_val, fn_val)
    else:
        # For other noise types, value should generally change
        # (unless noise parameters sampled are extremely small,
        # which is unlikely with default ranges)
        assert not jnp.allclose(noisy_val, fn_val)


def test_noise_stabilization():
    """Test noise stabilization."""
    # Directly test the stabilize function if accessible or check behavior through apply
    # Since stabilize is not exposed in public API of NoiseModel but used in apply
    # We rely on apply.

    # However, it's hard to control the random noise inside apply to ensure
    # specific condition.
    # We can check import stabilize from noise module if we want unit test
    from bbobax.noise import stabilize

    target_value = 1e-08

    # Case 1: fn_val >= target_value
    val_large = jnp.array(1e-7)
    noise_large = jnp.array(2e-7)
    # Result should be noise + 1.01 * target_value
    res_large = stabilize(val_large, noise_large, target_value)
    assert jnp.allclose(res_large, noise_large + 1.01 * target_value)

    # Case 2: fn_val < target_value
    val_small = jnp.array(1e-9)
    noise_small = jnp.array(2e-9)
    # Result should be fn_val (undisturbed)
    res_small = stabilize(val_small, noise_small, target_value)
    assert jnp.allclose(res_small, val_small)
