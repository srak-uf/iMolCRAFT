#!/usr/bin/env python
import jax.numpy as jnp
from jax.scipy.special import kl_div
from ase import units
from jax import jit, vmap
from .dmff_utils import update_ffinfo_from_params
from openmm import app
import openmm.unit as unit
from dmff.mbar import TargetState, buildTrajEnergyFunction, MBAREstimator, buildInputEnergyFunction
from dmff import Hamiltonian, DMFFTopology
import psutil
import os


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
    implemented_weight_schemes = ["uniform", "boltzmann", "nonboltzmann"]
    if weight_scheme not in implemented_weight_schemes:
        raise ValueError(f"Unknown weight scheme: {weight_scheme}")

    delta_e = (e_ff - e_qm) / len(e_qm)
    if norm_var:
        var_qm = jnp.var(e_qm)
    else:
        var_qm = 1.0
    # [(e_mm[0] - e_qm[0] - delta_e)^2, (e_mm[1]-e_qm[1]-delta_e)^2,..] のarrayを作成
    if zeropoint == "auto":
        se_array = jnp.square(e_ff - e_qm - delta_e)
    elif zeropoint == "qmmin":
        qm_argmin = jnp.argmin(e_qm)
        e_qm_min = e_qm[qm_argmin]
        e_ff_min = e_ff[qm_argmin]
        se_array = jnp.square(e_ff - e_ff_min - (e_qm - e_qm_min))
    elif zeropoint is None:
        se_array = jnp.square(e_ff - e_qm)

    if weight_scheme == "uniform":
        weight = jnp.ones_like(e_qm) / len(e_qm)
    elif weight_scheme == "boltzmann":
        kT = (
            units.kB * temperature * (units.kJ / units.mol) ** -1
        )  # 300 K = 2.494 kJ/mol
        ave = jnp.mean(e_qm)
        weight = jnp.exp(-(e_qm - ave) / kT)
        weight = weight / jnp.sum(weight)
    elif weight_scheme == "nonboltzmann":
        pass
    mse = jnp.sum(se_array * weight) / var_qm
    return mse


def wrightfactor(g_ff, g_gt):
    return jnp.sum((g_ff - g_gt) ** 2) / jnp.sum(g_gt**2)


def jsdivergence(g_ff, g_gt):
    M = 0.5 * g_ff + 0.5 * g_gt
    js_div = 0.5 * (kl_div(g_ff, M) + kl_div(g_gt, M))
    js_div = jnp.sum(js_div)
    return js_div


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
    batched_efunc = vmap(lambda x: efunc(x, None, pairs[0], ffparams))
    e_ff = batched_efunc(positions)
    loss = mse_energy(
        e_ff,
        y_gt,
        weight_scheme=weight_scheme,
        norm_var=norm_var,
        zeropoint=zeropoint,
        temperature=temperature,
    )
    return loss


def loss_thermodynamicperturbation(
    ffparams: dict,
    efunc: any,
    cov_map,
    rc: float,
    ensemble: str,
    Temperature_K: float,
    estimator: MBAREstimator,
    target_gt: dict,
    target_pred: dict,
    pressure: float = 1.0,
    losstype_distribfn: str = "wrightfactor",
    charge_penalty: dict = {"index": [None], "weight": [0.0], "initial": [0.0], "valance": [1.0]},
):
    if ensemble in ["isonpt", "anisonpt", "trinpt"]:
        ens = "npt"
    else:
        ens = ensemble

    if estimator._input is None:
        target_energy_function = buildTrajEnergyFunction(
            # efunc, cov_map, rc, ensemble=ens, useFreud=True, pressure=pressure
            efunc, cov_map, rc, ensemble=ens, useFreud=False, useRS=True, pressure=pressure
        )
        target_state = TargetState(Temperature_K, target_energy_function)
        weight, utarget = estimator.estimate_weight(
            target_state, parameters=ffparams, return_input=True
        )
    else:
        input_energy_function = buildInputEnergyFunction(
            efunc, ensemble=ens, pressure=pressure
        )
        target_state = TargetState(Temperature_K, input_energy_function)
        weight, utarget = estimator.estimate_weight(
            target_state, parameters=ffparams, direct=True
        )

    process = psutil.Process(os.getpid())
    print(f"Get weight, Memory Usage: {process.memory_info().rss / 1024**2:.2f} MB")
    loss = 0.0
    weighted_results = {}
    for key in target_gt.keys():
        if key in ["density_gcm3", "La_A", "Lb_A", "Lc_A"]:
            density_pred = jnp.average(target_pred[key], weights=weight)
            weighted_results[key] = density_pred
            loss += (
                target_gt[key]["weight"]
                * (target_gt[key]["gt"] - density_pred) ** 2
                / target_gt[key]["gt"] ** 2
            )
        elif key in ["rdf", "adf"]:
            weighted_results[key] = {}
            for kind in target_gt[key].keys():
                rdf_pred = (target_pred[key][kind] * weight.reshape((-1, 1))).sum(
                    axis=0
                )
                weighted_results[key][kind] = rdf_pred
                loss_tmp = 0.0
                if losstype_distribfn == "wrightfactor":
                    loss_tmp = wrightfactor(rdf_pred, target_gt[key][kind]["gt"])
                elif losstype_distribfn == "jsdivergence":
                    loss_tmp = jsdivergence(rdf_pred, target_gt[key][kind]["gt"])
                loss += target_gt[key][kind]["weight"] * loss_tmp
    
    if charge_penalty["index"] is not [None]:
        for i_loop, idx in enumerate(charge_penalty["index"]):
            penalty = charge_penalty["weight"][i_loop] * jnp.abs(ffparams["NonbondedForce"]["charge"][idx] - charge_penalty["initial"][i_loop]) / charge_penalty["valence"][i_loop]
            loss += penalty

    print(f"Finish loss calc Memory Usage: {process.memory_info().rss / 1024**2:.2f} MB")
    return loss, (utarget, weighted_results)
