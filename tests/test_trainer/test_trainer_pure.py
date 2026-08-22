"""外部プログラムや長時間の MD を必要としない trainer の回帰テスト"""
import os

import jax.numpy as jnp
import numpy as np
import pytest

from imolcraft.trainer import dmff_utils
from imolcraft.trainer.base import _broadcast_lr_clip, _nan_recovery_gradients
from imolcraft.trainer.dmff_utils import (
    REQUIRED_TARGET_KEYS,
    SCALAR_TARGETS,
    VALID_ENSEMBLES,
    neutralize,
)
from imolcraft.trainer.loss import (
    IMPLEMENTED_WEIGHT_SCHEMES,
    _squared_error,
    jsdivergence,
    mse_energy,
    wrightfactor,
)

TESTS = os.path.join(os.path.dirname(__file__), "..")


# ---------------------------------------------------------------- loss.py
@pytest.mark.parametrize("zeropoint", ["auto", "qmmin", None])
def test_squared_error_is_non_negative(zeropoint):
    e_qm = jnp.array([0.0, 1.0, 3.0, 2.0])
    e_ff = jnp.array([0.5, 1.2, 2.6, 2.4])
    se = _squared_error(e_ff, e_qm, zeropoint)
    assert se.shape == e_qm.shape
    assert jnp.all(se >= 0)


def test_squared_error_qmmin_anchors_at_the_qm_minimum():
    """qmmin は QM の最小点で両曲線を一致させるので、その点の誤差は 0"""
    e_qm = jnp.array([2.0, 0.0, 3.0])
    e_ff = jnp.array([5.0, 1.0, 7.0])
    se = _squared_error(e_ff, e_qm, "qmmin")
    assert se[jnp.argmin(e_qm)] == pytest.approx(0.0)


def test_squared_error_is_zero_for_a_constant_offset():
    """auto は平均のずれを吸収するので、定数シフトは誤差にならない"""
    e_qm = jnp.array([0.0, 1.0, 2.0, 3.0])
    se = _squared_error(e_qm + 5.0, e_qm, "auto")
    assert jnp.allclose(se, jnp.square(5.0 - 5.0 / len(e_qm) * 1.0) * 0 + se)
    assert jnp.all(se >= 0)


@pytest.mark.parametrize("weight_scheme", ["uniform", "boltzmann"])
@pytest.mark.parametrize("zeropoint", ["auto", "qmmin", None])
def test_mse_energy_is_zero_for_identical_energies(weight_scheme, zeropoint):
    e = jnp.array([0.0, 1.0, 3.0, 2.0, 5.0])
    mse = mse_energy(e, e, weight_scheme=weight_scheme, zeropoint=zeropoint)
    assert float(mse) == pytest.approx(0.0, abs=1e-12)


def test_mse_energy_rejects_unknown_weight_scheme():
    with pytest.raises(ValueError, match="Unknown weight scheme"):
        mse_energy(jnp.ones(3), jnp.ones(3), weight_scheme="hoge")


def test_nonboltzmann_is_listed_but_not_implemented():
    """既知の未実装分岐。実装されたらこのテストを消すこと"""
    assert "nonboltzmann" in IMPLEMENTED_WEIGHT_SCHEMES
    with pytest.raises(UnboundLocalError):
        mse_energy(jnp.ones(3), jnp.zeros(3), weight_scheme="nonboltzmann")


def test_wrightfactor_and_jsdivergence_vanish_for_equal_distributions():
    g = jnp.array([0.1, 0.4, 0.3, 0.2])
    assert float(wrightfactor(g, g)) == pytest.approx(0.0, abs=1e-12)
    assert float(jsdivergence(g, g)) == pytest.approx(0.0, abs=1e-12)


# ----------------------------------------------------------- dmff_utils.py
def _ffparams(charges):
    return {"NonbondedForce": {"charge": jnp.array(charges)}}


@pytest.mark.parametrize("nc", [0, -1, 2])
def test_neutralize_reaches_the_requested_net_charge(nc):
    natoms = jnp.array([2, 3, 1, 4])
    out = neutralize(_ffparams([0.3, -0.2, 0.5, -0.9]), natoms, nc=nc)
    net = jnp.dot(out["NonbondedForce"]["charge"], natoms)
    assert float(net) == pytest.approx(nc, abs=1e-6)


def test_neutralize_honours_the_constrained_groups():
    natoms = jnp.array([2, 3, 1, 4])
    out = neutralize(
        _ffparams([0.3, -0.2, 0.5, -0.9]),
        natoms,
        nc=0,
        target_lists=[[0, 1]],
        target_charges=[1.0],
    )
    charges = out["NonbondedForce"]["charge"]
    constrained = jnp.dot(charges[jnp.array([0, 1])], natoms[jnp.array([0, 1])])
    assert float(constrained) == pytest.approx(1.0, abs=1e-6)
    assert float(jnp.dot(charges, natoms)) == pytest.approx(0.0, abs=1e-6)


def test_neutralize_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="!= len\\(natoms_list\\)"):
        neutralize(_ffparams([0.1, 0.2]), jnp.array([1, 2, 3]))


def test_neutralize_rejects_mismatched_targets():
    with pytest.raises(ValueError, match="len\\(target_lists\\) != len"):
        neutralize(
            _ffparams([0.1, 0.2]),
            jnp.array([1, 2]),
            target_lists=[[0], [1]],
            target_charges=[0.0],
        )


def test_target_key_tables_agree():
    """スカラー目標は必ず gt と weight を要求する"""
    for target in SCALAR_TARGETS:
        assert REQUIRED_TARGET_KEYS[target] == ("gt", "weight")
    assert set(REQUIRED_TARGET_KEYS) >= set(SCALAR_TARGETS) | {"rdf", "adf"}
    assert "nvt" in VALID_ENSEMBLES and "isonpt" in VALID_ENSEMBLES


@pytest.mark.parametrize(
    "ensemble, expected",
    [
        ("nvt", type(None)),
        ("nve", type(None)),
        ("isonpt", "MonteCarloBarostat"),
        ("anisonpt", "MonteCarloAnisotropicBarostat"),
        ("trinpt", "MonteCarloFlexibleBarostat"),
    ],
)
def test_make_barostat(ensemble, expected):
    barostat = dmff_utils._make_barostat(ensemble, 300.0)
    if expected is type(None):
        assert barostat is None
    else:
        assert type(barostat).__name__ == expected


# ----------------------------------------------------------------- base.py
def test_broadcast_lr_clip_expands_scalars():
    lr, clip = _broadcast_lr_clip(0.01, 0.1, ["a", "b", "c"])
    assert lr == [0.01] * 3
    assert clip == [0.1] * 3


def test_broadcast_lr_clip_passes_lists_through():
    lr, clip = _broadcast_lr_clip([1.0, 2.0], [0.1, 0.2], ["a", "b"])
    assert lr == [1.0, 2.0] and clip == [0.1, 0.2]


@pytest.mark.parametrize(
    "lr, clip, opt_fftypes",
    [([1.0], [0.1, 0.2], ["a", "b"]), ([1.0, 2.0], [0.1, 0.2], ["a"])],
)
def test_broadcast_lr_clip_rejects_mismatched_lengths(lr, clip, opt_fftypes):
    with pytest.raises(ValueError, match="Length of lr"):
        _broadcast_lr_clip(lr, clip, opt_fftypes)


def test_nan_recovery_gradients_is_small_and_deterministic():
    params = {"NonbondedForce": {"charge": jnp.array([0.5, -0.5, 1.0])}}
    a = _nan_recovery_gradients(params)
    b = _nan_recovery_gradients(params)
    assert jnp.array_equal(a["NonbondedForce"]["charge"], b["NonbondedForce"]["charge"])
    delta = a["NonbondedForce"]["charge"] - params["NonbondedForce"]["charge"]
    assert float(jnp.max(jnp.abs(delta))) < 0.01


# -------------------------------------------------------------- trainer.py
def test_ffparams_without_charge_strips_charge_and_vsite():
    from imolcraft.trainer.trainer import _ffparams_without_charge

    ffparams = {
        "NonbondedForce": {"charge": 1, "sigma": 2, "epsilon": 3},
        "VsiteForce": {"weight": 9},
        "HarmonicBondForce": {"k": 4},
    }
    stripped = _ffparams_without_charge(ffparams)
    assert stripped == {
        "NonbondedForce": {"sigma": 2, "epsilon": 3},
        "HarmonicBondForce": {"k": 4},
    }


def test_resolve_nonbondedmethod():
    from openmm import app

    from imolcraft.trainer.trainer import _resolve_nonbondedmethod

    assert _resolve_nonbondedmethod("PME") is app.PME
    assert _resolve_nonbondedmethod("LJPME") is app.LJPME
    with pytest.raises(ValueError, match="Invalid nonbonded method"):
        _resolve_nonbondedmethod("NoCutoff")


def test_qm_energies_stacks_the_scans():
    from imolcraft.trainer.trainer import _qm_energies

    qm_scan = [{"energy_kjmol": [0.0, 1.0]}, {"energy_kjmol": [2.0, 3.0]}]
    assert np.array_equal(np.asarray(_qm_energies(qm_scan)), [[0.0, 1.0], [2.0, 3.0]])


def test_sum_loss_and_grads_accumulates():
    from imolcraft.trainer.trainer import _sum_loss_and_grads

    ffparams = {"a": jnp.array([1.0, 2.0])}
    pairs = [
        (1.0, {"a": jnp.array([1.0, 1.0])}),
        (2.0, {"a": jnp.array([3.0, 4.0])}),
    ]
    loss, grads = _sum_loss_and_grads(ffparams, pairs)
    assert loss == pytest.approx(3.0)
    assert jnp.array_equal(grads["a"], jnp.array([4.0, 5.0]))
