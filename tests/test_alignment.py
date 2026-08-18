"""Numerical alignment of bbobax against the official 2009 BBOB implementation.

Every numerically matchable function (all 24 except the two Gallagher
functions, whose peak layouts are instance data that cannot be injected) is
compared to the vendored ``tests/_official/bbobbenchmarks.py`` on identical
instances in float64: the official instance's ``xopt``/``fopt`` and rotation
matrices are injected into bbobax params/state, and values must agree to 1e-9
relative error. float64 is enabled globally in ``conftest.py``.

Conventions bridged here (see ``_INSTANCE_MAPPINGS``):

- ``bbobbenchmarks`` is row-vector (``x @ M``), bbobax column-vector
  (``M @ x``): official matrices enter transposed.
- ``params.x_opt = official.xopt`` directly for every function -- bbobax's
  per-function x_opt conventions mirror what the official code stores.

Every non-trivial rotation mapping is verified in-test by reconstructing the
official product matrix (``linearTF``) to <= 1e-10, so a regression in the
mapping itself fails loudly rather than producing a confusing value mismatch.

The Gallagher functions get structural tests instead: the value formula is
checked against an independent numpy implementation of the paper formula on
bbobax's own generated peak layout, and the layout's dependence on
``params.key`` is pinned (regression for the frozen-layout bug).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _official import bbobbenchmarks as bb

from bbobax.fitness_fns import bbob_fns
from bbobax.types import BBOBParams, BBOBState

# Tolerances: values must match to 1e-9 relative against |f - fopt|, with an
# absolute floor of 1e-8 for near-zero values; mapping reconstructions must
# match to 1e-10.
REL_TOL = 1e-9
ABS_FLOOR = 1e-8
RECON_ATOL = 1e-10


def _assert_orthogonal(m: np.ndarray) -> None:
    np.testing.assert_allclose(m @ m.T, np.eye(m.shape[0]), atol=RECON_ATOL)


def _map_identity(off, D):
    """No rotation: the official function is axis-aligned (r = q = I)."""
    return np.eye(D), np.eye(D)


def _map_rotation_only(off, D):
    """Single rotation (f10, f11, f12, f14): r = rotation^T, q unused."""
    rot = np.asarray(off.rotation)
    _assert_orthogonal(rot)
    return rot.T, np.eye(D)


def _map_f6(off, D):
    """f6 attractive_sector.

    Official: ``linearTF = R2 @ diag(scales) @ rotation`` with
    ``rotation = R(rseed + 1e6)``; column-effective z = rotation^T @
    diag(scales) @ R2^T @ (x - xopt). bbobax computes z = q @ (lambda * (r @
    (x - xopt))), so q = rotation^T and r = R2^T, with R2 reconstructed from
    linearTF.
    """
    rot = np.asarray(off.rotation)
    scales = np.asarray(off.scales)
    lin = np.asarray(off.linearTF)
    R2 = lin @ rot.T @ np.diag(1.0 / scales)
    _assert_orthogonal(R2)
    np.testing.assert_allclose(R2 @ np.diag(scales) @ rot, lin, atol=RECON_ATOL)
    return R2.T, rot.T


def _map_f7(off, D):
    """f7 step_ellipsoidal.

    Official: ``linearTF = R(rseed) @ diag(s)`` with s = sqrt(10)^linspace
    (note: NOT ``off.scales``, which holds the 100^linspace core weights), and
    the outer ``rotation = R(rseed + 1e6)``. bbobax z_hat = lambda * (r @ .),
    z = q @ z_tilde, so r comes from the linearTF side and q = rotation^T.
    """
    s = np.sqrt(10.0) ** np.linspace(0.0, 1.0, D)
    lin = np.asarray(off.linearTF)
    R = lin @ np.diag(1.0 / s)
    _assert_orthogonal(R)
    np.testing.assert_allclose(R @ np.diag(s), lin, atol=RECON_ATOL)
    rot = np.asarray(off.rotation)
    _assert_orthogonal(rot)
    return R.T, rot.T


def _map_scaled_rotation(off, D):
    """f9/f19: ``linearTF = scale * R(rseed)``; r = (linearTF / scale)^T.

    The official optimum is derived from the rotation
    (xopt = R (0.5 / scale) 1); bbobax's placeable-optimum parameterization is
    exactly equivalent with x_opt = official.xopt, verified here.
    """
    scale = max(1.0, np.sqrt(D) / 8.0)
    B = np.asarray(off.linearTF) / scale
    _assert_orthogonal(B)
    np.testing.assert_allclose(
        B @ (0.5 / scale * np.ones(D)), np.asarray(off.xopt), atol=RECON_ATOL
    )
    return B.T, np.eye(D)


def _map_f13(off, D):
    """f13 sharp_ridge -- the flipped sandwich.

    Official: ``linearTF = R(rseed) @ diag(scales) @ rotation`` with
    ``rotation = R(rseed + 1e6)``, applied as one product; bbobax applies
    r first and q last, so r = R(rseed)^T (inner) and q = rotation^T (outer).
    """
    R0 = np.asarray(bb.compute_rotation(off.rseed, D))
    rot = np.asarray(off.rotation)
    np.testing.assert_allclose(
        R0 @ np.diag(np.asarray(off.scales)) @ rot,
        np.asarray(off.linearTF),
        atol=RECON_ATOL,
    )
    return R0.T, rot.T


def _map_sandwich(off, D):
    """f15/f16: ``linearTF = R(rseed) @ diag(scales) @ rotation``.

    bbobax applies r = rotation^T first (and, for these functions, again
    last), q = R(rseed)^T in the middle -- the mirror of f13.
    """
    R0 = np.asarray(bb.compute_rotation(off.rseed, D))
    rot = np.asarray(off.rotation)
    np.testing.assert_allclose(
        R0 @ np.diag(np.asarray(off.scales)) @ rot,
        np.asarray(off.linearTF),
        atol=RECON_ATOL,
    )
    return rot.T, R0.T


def _map_sandwich_no_tail(off, D):
    """f17/f18: ``linearTF = R(rseed) @ diag(scales)`` (no trailing rotation).

    r = rotation^T (rotation = R(rseed + 1e6)), q = R(rseed)^T; the diagonal
    scaling is bbobax's own lambda, applied after q.
    """
    R0 = np.asarray(bb.compute_rotation(off.rseed, D))
    np.testing.assert_allclose(
        R0 @ np.diag(np.asarray(off.scales)),
        np.asarray(off.linearTF),
        atol=RECON_ATOL,
    )
    rot = np.asarray(off.rotation)
    _assert_orthogonal(rot)
    return rot.T, R0.T


def _map_inner_from_linear_tf(off, D):
    """f23/f24: ``linearTF = A @ diag(scales) @ rotation`` with A = R(rseed).

    A is reconstructed from linearTF; r = A^T (inner), q = rotation^T (outer).
    """
    B = np.asarray(off.rotation)
    scales = np.asarray(off.scales)
    lin = np.asarray(off.linearTF)
    A = lin @ B.T @ np.diag(1.0 / scales)
    _assert_orthogonal(A)
    np.testing.assert_allclose(A @ np.diag(scales) @ B, lin, atol=RECON_ATOL)
    return A.T, B.T


# fid -> (bbobax function name, (official, D) -> (r, q) with reconstruction
# asserts). x_opt is always official.xopt, injected directly.
_INSTANCE_MAPPINGS = {
    1: ("sphere", _map_identity),
    2: ("ellipsoidal", _map_identity),
    3: ("rastrigin", _map_identity),
    4: ("bueche_rastrigin", _map_identity),
    5: ("linear_slope", _map_identity),
    6: ("attractive_sector", _map_f6),
    7: ("step_ellipsoidal", _map_f7),
    8: ("rosenbrock", _map_identity),
    9: ("rosenbrock_rotated", _map_scaled_rotation),
    10: ("ellipsoidal_rotated", _map_rotation_only),
    11: ("discus", _map_rotation_only),
    12: ("bent_cigar", _map_rotation_only),
    13: ("sharp_ridge", _map_f13),
    14: ("different_powers", _map_rotation_only),
    15: ("rastrigin_rotated", _map_sandwich),
    16: ("weierstrass", _map_sandwich),
    17: ("schaffers_f7", _map_sandwich_no_tail),
    18: ("schaffers_f7_ill_conditioned", _map_sandwich_no_tail),
    19: ("griewank_rosenbrock", _map_scaled_rotation),
    20: ("schwefel", _map_identity),
    23: ("katsuura", _map_inner_from_linear_tf),
    24: ("lunacek", _map_inner_from_linear_tf),
}


def _make_params_state(D, x_opt, f_opt, r, q, key=None):
    params = BBOBParams(
        fn_id=jnp.array(0),
        num_dims=jnp.array(D),
        x_opt=jnp.asarray(x_opt, dtype=jnp.float64),
        f_opt=jnp.array(float(f_opt)),
        key=key if key is not None else jax.random.key(0),
        noise_params=None,
    )
    state = BBOBState(
        r=jnp.asarray(r, dtype=jnp.float64),
        q=jnp.asarray(q, dtype=jnp.float64),
        counter=0,
    )
    return params, state


def _test_points(rng, D, x_opt):
    """~60 points: bulk in-domain, penalty region, near-optimum, the optimum."""
    return np.vstack(
        [
            rng.uniform(-5.0, 5.0, size=(40, D)),
            rng.uniform(-7.0, 7.0, size=(15, D)),
            x_opt + rng.normal(scale=1e-3, size=(5, D)),
            x_opt[None, :],
        ]
    )


@pytest.mark.parametrize("instance", [1, 2])
@pytest.mark.parametrize("num_dims", [5, 8])
@pytest.mark.parametrize(
    "fid",
    sorted(_INSTANCE_MAPPINGS),
    ids=[f"f{fid:02d}-{_INSTANCE_MAPPINGS[fid][0]}" for fid in sorted(_INSTANCE_MAPPINGS)],
)
def test_alignment_with_official(fid, num_dims, instance):
    """bbobax == official 2009 implementation on an injected instance."""
    D = num_dims
    name, mapper = _INSTANCE_MAPPINGS[fid]
    off = getattr(bb, f"F{fid}")(instance)
    off(np.zeros(D))  # trigger dimension-dependent initialization
    r, q = mapper(off, D)
    x_opt = np.asarray(off.xopt, dtype=float)
    params, state = _make_params_state(D, x_opt, off.fopt, r, q)

    rng = np.random.default_rng(97 * fid + 10 * D + instance)
    X = _test_points(rng, D, x_opt)

    value, penalty = jax.vmap(bbob_fns[name], in_axes=(0, None, None))(
        jnp.asarray(X), state, params
    )
    ours = np.asarray(value) + np.asarray(penalty) + float(off.fopt)
    theirs = np.array([float(off(x)) for x in X])

    err = np.abs(ours - theirs)
    tol = np.maximum(REL_TOL * np.abs(theirs - off.fopt), ABS_FLOOR)
    worst = int(np.argmax(err - tol))
    assert np.all(err <= tol), (
        f"f{fid} ({name}) D={D} instance={instance}: worst point #{worst}: "
        f"ours={ours[worst]!r} official={theirs[worst]!r} err={err[worst]:.3e}"
    )

    # The last point is exactly the optimum: f(x_opt) == fopt.
    assert abs(ours[-1] - float(off.fopt)) <= 1e-9


# ---------------------------------------------------------------------------
# Gallagher (f21/f22): structural tests. The peak layout is drawn from
# params.key, so official instances cannot be injected; instead the value
# formula is verified against an independent numpy implementation of the
# paper formula evaluated on bbobax's own generated layout.
# ---------------------------------------------------------------------------

_GALLAGHER = {
    "gallagher_101_me": {"num_optima": 101, "first_alpha": 1000.0, "y_range": 5.0, "fold": 21},
    "gallagher_21_hi": {"num_optima": 21, "first_alpha": 1000.0**2, "y_range": 4.9, "fold": 22},
}


def _gallagher_layout(name, D, instance_key, x_opt):
    """Replicate the layout generation of the bbobax Gallagher functions.

    Mirrors the PRNG call sequence in ``fitness_fns.gallagher_*`` exactly
    (fold_in with the function's tag, then permutation / per-peak diagonal
    permutations / peak positions), at full dimensionality (num_dims == D).
    Returns numpy (w, diagonals, y).
    """
    spec = _GALLAGHER[name]
    num_optima = spec["num_optima"]
    key = jax.random.fold_in(instance_key, spec["fold"])

    w = np.where(
        np.arange(num_optima) == 0,
        10.0,
        1.1 + 8.0 * (np.arange(num_optima) - 1.0) / (num_optima - 2.0),
    )

    alpha_set = jnp.power(1000.0, 2.0 * jnp.arange(num_optima - 1) / (num_optima - 2))
    key, subkey = jax.random.split(key)
    alpha = np.zeros(num_optima)
    alpha[0] = spec["first_alpha"]
    alpha[1:] = np.asarray(jax.random.permutation(subkey, alpha_set))

    # Pre-permutation diagonal of C_i: alpha^(0.5 j / (D-1)) / alpha^(1/4).
    diags = alpha[:, None] ** (0.5 * np.arange(D) / (D - 1)) / alpha[:, None] ** 0.25

    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, num_optima)
    mask = jnp.arange(D) < D  # all True at full dimensionality, as in the fn
    perms = jax.vmap(
        lambda k: jax.random.choice(k, jnp.arange(D), shape=(D,), replace=False, p=mask)
    )(keys)
    diags = np.stack([d[p] for d, p in zip(diags, np.asarray(perms))])

    key, subkey = jax.random.split(key)
    # np.array (not asarray): arrays converted from JAX are read-only.
    y = np.array(
        jax.random.uniform(
            subkey,
            shape=(num_optima, D),
            minval=-spec["y_range"],
            maxval=spec["y_range"],
        )
    )
    y[0] = np.asarray(x_opt)
    return w, diags, y


def _monotone_tf_osc(f):
    """The official scalar oscillation transform (monotoneTFosc), numpy."""
    if f > 0.0:
        g = np.log(f) / 0.1
        return np.exp(g + 0.49 * (np.sin(g) + np.sin(0.79 * g))) ** 0.1
    if f < 0.0:
        g = np.log(-f) / 0.1
        return -(np.exp(g + 0.49 * (np.sin(0.55 * g) + np.sin(0.31 * g))) ** 0.1)
    return f


def _gallagher_reference(x, R, w, diags, y, D):
    """Paper formula (Hansen et al. 2010, f21/f22), independent numpy impl."""
    values = []
    for i in range(len(w)):
        d = R @ (x - y[i])
        values.append(w[i] * np.exp(-np.sum(diags[i] * d * d) / (2.0 * D)))
    value = _monotone_tf_osc(10.0 - max(values)) ** 2
    penalty = np.sum(np.maximum(0.0, np.abs(x) - 5.0) ** 2)
    return value + penalty


@pytest.mark.parametrize("num_dims", [5, 8])
@pytest.mark.parametrize("name", sorted(_GALLAGHER))
def test_gallagher_matches_paper_formula(name, num_dims):
    """bbobax Gallagher value == numpy paper formula on bbobax's own layout."""
    D = num_dims
    rng = np.random.default_rng(2100 + D)
    R = np.asarray(bb.compute_rotation(70 + D, D))  # any orthogonal matrix
    instance_key = jax.random.key(1234 + D)
    scale = 0.98 if name == "gallagher_21_hi" else 1.0  # x_opt convention
    x_opt = scale * rng.uniform(-4.0, 4.0, size=D)
    params, state = _make_params_state(D, x_opt, 0.0, R, np.eye(D), key=instance_key)
    w, diags, y = _gallagher_layout(name, D, instance_key, x_opt)

    X = _test_points(rng, D, x_opt)
    value, penalty = jax.vmap(bbob_fns[name], in_axes=(0, None, None))(
        jnp.asarray(X), state, params
    )
    ours = np.asarray(value) + np.asarray(penalty)
    theirs = np.array([_gallagher_reference(x, R, w, diags, y, D) for x in X])

    np.testing.assert_allclose(ours, theirs, rtol=1e-12, atol=1e-12)

    # (d) f(y_1) == 0: the first peak is the global optimum with value 0.
    v, p = bbob_fns[name](jnp.asarray(x_opt), state, params)
    assert abs(float(v) + float(p)) <= 1e-9


@pytest.mark.parametrize("name", sorted(_GALLAGHER))
def test_gallagher_layout_follows_key(name):
    """Different params.key -> different landscape; same key -> identical.

    Regression for the frozen-layout bug, where every rotated instance shared
    one peak layout because the layout entropy collapsed to int(|q00|) = 0.
    """
    D = 6
    rng = np.random.default_rng(9)
    R = np.asarray(bb.compute_rotation(80 + D, D))
    x_opt = rng.uniform(-3.9, 3.9, size=D)
    X = jnp.asarray(rng.uniform(-5.0, 5.0, size=(20, D)))

    def values(key):
        params, state = _make_params_state(D, x_opt, 0.0, R, np.eye(D), key=key)
        v, p = jax.vmap(bbob_fns[name], in_axes=(0, None, None))(X, state, params)
        return np.asarray(v) + np.asarray(p)

    v1 = values(jax.random.key(1))
    v1_again = values(jax.random.key(1))
    v2 = values(jax.random.key(2))

    np.testing.assert_array_equal(v1, v1_again)
    assert not np.allclose(v1, v2)


def test_gallagher_101_and_21_differ():
    """f21 and f22 with the same instance key are different landscapes."""
    D = 6
    rng = np.random.default_rng(10)
    R = np.asarray(bb.compute_rotation(90 + D, D))
    x_opt = rng.uniform(-3.9, 3.9, size=D)
    X = jnp.asarray(rng.uniform(-5.0, 5.0, size=(20, D)))
    params, state = _make_params_state(D, x_opt, 0.0, R, np.eye(D), key=jax.random.key(3))

    v21, p21 = jax.vmap(bbob_fns["gallagher_101_me"], in_axes=(0, None, None))(
        X, state, params
    )
    v22, p22 = jax.vmap(bbob_fns["gallagher_21_hi"], in_axes=(0, None, None))(
        X, state, params
    )
    assert not np.allclose(np.asarray(v21), np.asarray(v22))
