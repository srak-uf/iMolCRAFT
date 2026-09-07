#!/usr/bin/env python
import os
from typing import Any, Optional

import jax.numpy as jnp
import psutil
from ase import units
from dmff.mbar import (
    MBAREstimator,
    TargetState,
    buildInputEnergyFunction,
    buildTrajEnergyFunction,
)
from jax import vmap
from jax.scipy.special import kl_div

from .properties import DISTRIBUTION_TARGETS, SCALAR_TARGETS

#: Weighting schemes accepted by :func:`mse_energy`.
IMPLEMENTED_WEIGHT_SCHEMES = ["uniform", "boltzmann", "nonboltzmann"]

#: Ensembles that MBAR treats as an NPT ensemble.
_NPT_ENSEMBLES = ("isonpt", "anisonpt", "trinpt")

#: No charge penalty.
_NO_CHARGE_PENALTY = {
    "index": [None],
    "weight": [0.0],
    "initial": [0.0],
    "valence": [1.0],
}


def _squared_error(e_ff, e_qm, zeropoint):
    """
    Squared deviation of the force field energies from the QM ones, after
    lining up their zero points.

    Parameters
    ----------
    e_ff, e_qm : jnp.ndarray
        Force field and QM energies.
    zeropoint : {"auto", "qmmin", None}
        "auto" shifts by the mean deviation, "qmmin" lines both curves up at
        the QM minimum, None compares the raw values.

    Returns
    -------
    jnp.ndarray
        Per-point squared error.
    """
    if zeropoint == "auto":
        delta_e = (e_ff - e_qm) / len(e_qm)
        return jnp.square(e_ff - e_qm - delta_e)
    if zeropoint == "qmmin":
        qm_argmin = jnp.argmin(e_qm)
        return jnp.square(e_ff - e_ff[qm_argmin] - (e_qm - e_qm[qm_argmin]))
    return jnp.square(e_ff - e_qm)


def mse_energy(
    e_ff,
    e_qm,
    weight_scheme="uniform",
    norm_var=True,
    zeropoint="auto",
    temperature=500,
):
    """
    Calculate the mean squared error between two energy arrays.

    Parameters
    ----------
    e_ff : np.ndarray
        Array of energies from the force field.
    e_qm : np.ndarray
        Array of energies from quantum mechanics.
    weight_scheme : str
        Optional weighting scheme for the energies.
        Options are "uniform", "boltzmann", and "nonboltzmann".
        Default is "uniform".
    norm_var : bool
        Optional normalization by variance of e_qm. Default is True.
    zeropoint : str
        Optional zero-point energy correction.
        Options are "auto", "qmmin", and None.
        If "auto", the zero-point energy is determined automatically.
        If "qmmin", the zero-point energy is set to the minimum QM energy.
        if None, the zero-point energy is not corrected.
        Default is "auto".
    temperature : float
        Temperature in Kelvin for the Boltzmann weighting scheme. Default is 500 K.

    Returns
    -------
    float
        The mean squared error between the two energy arrays.
    """
    if weight_scheme not in IMPLEMENTED_WEIGHT_SCHEMES:
        raise ValueError(f"Unknown weight scheme: {weight_scheme}")

    # Build [(e_mm[0] - e_qm[0] - delta_e)^2, (e_mm[1]-e_qm[1]-delta_e)^2, ..]
    se_array = _squared_error(e_ff, e_qm, zeropoint)
    var_qm = jnp.var(e_qm) if norm_var else 1.0

    if weight_scheme == "uniform":
        weight = jnp.ones_like(e_qm) / len(e_qm)
    elif weight_scheme == "boltzmann":
        kT = (
            units.kB * temperature * (units.kJ / units.mol) ** -1
        )  # 300 K = 2.494 kJ/mol
        weight = jnp.exp(-(e_qm - jnp.mean(e_qm)) / kT)
        weight = weight / jnp.sum(weight)
    elif weight_scheme == "nonboltzmann":
        # NOTE: not implemented; `weight` stays undefined and the line below
        # raises UnboundLocalError.
        pass

    return jnp.sum(se_array * weight) / var_qm


def wrightfactor(g_ff, g_gt):
    """Wright factor: squared deviation of two distributions, scaled by the reference."""
    return jnp.sum((g_ff - g_gt) ** 2) / jnp.sum(g_gt**2)


def jsdivergence(g_ff, g_gt):
    """Jensen-Shannon divergence between two distributions."""
    M = 0.5 * g_ff + 0.5 * g_gt
    return jnp.sum(0.5 * (kl_div(g_ff, M) + kl_div(g_gt, M)))


#: Loss functions comparing two distribution functions.
_DISTRIBUTION_LOSSES = {
    "wrightfactor": wrightfactor,
    "jsdivergence": jsdivergence,
}


def relative_error(pred, gt):
    """Signed relative deviation, negative when the prediction is too low."""
    return (pred - gt) / gt


def abs_relative_error(pred, gt):
    """Relative deviation without its sign."""
    return abs(pred - gt) / abs(gt)


def squared_relative_error(pred, gt):
    """Squared relative deviation, the form the scalar targets are fitted with."""
    return (pred - gt) ** 2 / gt**2


def absolute_error(pred, gt):
    """Signed deviation, in the unit of the property itself."""
    return pred - gt


#: Metrics comparing a single predicted number with its reference. They are
#: dimensionless but for ``diff``, and all of them are zero for a perfect
#: match; only the signed ones say in which direction the prediction is off.
_SCALAR_DEVIATIONS = {
    "relerr": relative_error,
    "absrelerr": abs_relative_error,
    "sqrelerr": squared_relative_error,
    "diff": absolute_error,
}


def loss_energy(
    ffparams,
    efunc,
    positions,
    pairs,
    y_gt,
    weight_scheme="uniform",
    norm_var=True,
    zeropoint="auto",
    temperature=500,
):
    """Mean squared error of the force field energies of a scan."""
    batched_efunc = vmap(lambda x: efunc(x, None, pairs[0], ffparams))
    return mse_energy(
        batched_efunc(positions),
        y_gt,
        weight_scheme=weight_scheme,
        norm_var=norm_var,
        zeropoint=zeropoint,
        temperature=temperature,
    )


def _mbar_weights(
    ffparams, efunc, cov_map, rc, ens, Temperature_K, estimator, pressure
):
    """
    MBAR weights of the sampled frames under the current parameters.

    Once the estimator holds its input energies, the cheaper direct route can
    be taken instead of recomputing them from the trajectory.
    """
    if estimator._input is None:
        energy_function = buildTrajEnergyFunction(
            # efunc, cov_map, rc, ensemble=ens, useFreud=True, pressure=pressure
            efunc, cov_map, rc, ensemble=ens, useFreud=False, useRS=True,
            pressure=pressure,
        )
        return estimator.estimate_weight(
            TargetState(Temperature_K, energy_function),
            parameters=ffparams,
            return_input=True,
        )

    energy_function = buildInputEnergyFunction(efunc, ensemble=ens, pressure=pressure)
    return estimator.estimate_weight(
        TargetState(Temperature_K, energy_function), parameters=ffparams, direct=True
    )


def _charge_penalty_loss(ffparams, charge_penalty):
    """Penalty pulling the optimized charges back towards their initial values."""
    charges = ffparams["NonbondedForce"]["charge"]
    return sum(
        charge_penalty["weight"][i]
        * jnp.abs(charges[idx] - charge_penalty["initial"][i])
        / jnp.abs(charge_penalty["valence"][i])
        for i, idx in enumerate(charge_penalty["index"])
    )


def loss_thermodynamicperturbation(
    ffparams: dict,
    efunc: Any,
    cov_map,
    rc: float,
    ensemble: str,
    Temperature_K: float,
    estimator: MBAREstimator,
    target_gt: dict,
    target_pred: dict,
    pressure: float = 1.0,
    losstype_distribfn: str = "wrightfactor",
    charge_penalty: Optional[dict] = None,
):
    """
    Reweighted deviation of the predicted thermodynamic targets from their
    reference values.

    Returns
    -------
    (loss, (utarget, weighted_results)) : (float, (jnp.ndarray, dict))
    """
    if charge_penalty is None:
        charge_penalty = _NO_CHARGE_PENALTY

    ens = "npt" if ensemble in _NPT_ENSEMBLES else ensemble
    weight, utarget = _mbar_weights(
        ffparams, efunc, cov_map, rc, ens, Temperature_K, estimator, pressure
    )

    process = psutil.Process(os.getpid())
    print(f"Get weight, Memory Usage: {process.memory_info().rss / 1024**2:.2f} MB")

    loss = 0.0
    weighted_results = {}
    for key in target_gt.keys():
        if key in SCALAR_TARGETS:
            # a single number per frame, so the reweighted average is compared
            pred = jnp.average(target_pred[key], weights=weight)
            weighted_results[key] = pred
            loss += (
                target_gt[key]["weight"]
                * (target_gt[key]["gt"] - pred) ** 2
                / target_gt[key]["gt"] ** 2
            )
        elif key in DISTRIBUTION_TARGETS:
            # a curve per frame, so the frames are reweighted bin by bin
            weighted_results[key] = {}
            for kind in target_gt[key].keys():
                pred = (target_pred[key][kind] * weight.reshape((-1, 1))).sum(axis=0)
                weighted_results[key][kind] = pred
                loss_fn = _DISTRIBUTION_LOSSES.get(losstype_distribfn)
                loss_tmp = 0.0 if loss_fn is None else loss_fn(
                    pred, target_gt[key][kind]["gt"]
                )
                loss += target_gt[key][kind]["weight"] * loss_tmp

    if not charge_penalty["index"] == [None]:
        loss += _charge_penalty_loss(ffparams, charge_penalty)

    print(
        "Finish loss calc Memory Usage: "
        f"{process.memory_info().rss / 1024**2:.2f} MB"
    )
    return loss, (utarget, weighted_results)
