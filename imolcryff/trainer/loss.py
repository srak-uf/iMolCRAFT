#!/usr/bin/env python
import jax.numpy as jnp
from jax.scipy.special import kl_div
from ase import units
from jax import value_and_grad, jit, vmap
from .dmff_utils import update_ffinfo_from_params
from openmm import app
import openmm.unit as unit
from dmff.mbar import TargetState, buildTrajEnergyFunction
from dmff import Hamiltonian, DMFFTopology

def mse_energy(e_ff, e_qm, weight_scheme="uniform", norm_var=True, zeropoint="auto", temperature=500):
    """
    Calculate the mean squared error between two energy arrays.
    :param e_ff: Array of energies from the force field.
    :param e_qm: Array of energies from quantum mechanics.
    :param weight_scheme: Optional weighting scheme for the energies.
    :param zeropoint: Optional zero-point energy correction.
                        If "auto", the zero-point energy is determined automatically.
                        If "qmmin", the zero-point energy is set to the minimum QM energy.
                        if None, the zero-point energy is not corrected.
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
        kT = units.kB * temperature * (units.kJ / units.mol) ** -1  # 300 K = 2.494 kJ/mol
        ave = jnp.mean(e_qm)
        weight = jnp.exp(-(e_qm - ave) / kT)
        weight = weight / jnp.sum(weight)
    elif weight_scheme == "nonboltzmann":
        pass
    mse = jnp.sum(se_array * weight) / var_qm
    return mse

def wrightfactor(g_ff, g_gt):
    return jnp.sum((g_ff-g_gt)**2)/jnp.sum(g_gt**2)

def jsdivergence(g_ff, g_gt):
    M = 0.5 * g_ff + 0.5 * g_gt
    js_div = 0.5*(kl_div(g_ff, M) + kl_div(g_gt, M))
    js_div = jnp.sum(js_div)
    return js_div

def loss_energy(ffparams, ff, topology, positions, pairs, y_gt):
    ff_d = update_ffinfo_from_params(ff, ffparams)
    ffparams_wo_charge = {}
    for key in ffparams.keys():
        if key == "NonbondedForce":
            ffparams_wo_charge[key] = {}
            for key2 in ffparams[key].keys():
                if key2 != "charges":
                    ffparams_wo_charge[key][key2] = ffparams[key][key2]
        elif key == "VsiteForce":
            pass
        else:
            ffparams_wo_charge[key] = ffparams[key]

    pots = ff_d.createPotential(topology) # should be pdb topology wo vsites
    efunc = pots.getPotentialFunc()
    batched_efunc = vmap(lambda x: efunc(x, None, pairs[0], ffparams_wo_charge))
    e_ff = batched_efunc(positions)
    loss = mse_energy(e_ff, y_gt, weight_scheme="uniform", norm_var=True)
    return loss

def loss_thermodynamicperturbation(ffparams: dict,
                                   ff: Hamiltonian,
                                   topology: app.Topology | DMFFTopology, 
                                   cov_map,
                                   rc: float,
                                   ensemble: str,
                                   Temperature_K: float,
                                   estimator,
                                   target_gt: dict,
                                   target_pred: dict,
                                   pressure: float = 1.0,
                                   useDispersionCorrection: bool = False,
                                   losstype_distribfn: str = "wrightfactor"):
    ff = update_ffinfo_from_params(ff, ffparams)
    ffparams_wo_charge = {}
    for key in ffparams.keys():
        if key == "NonbondedForce":
            ffparams_wo_charge[key] = {}
            for key2 in ffparams[key].keys():
                if key2 != "charges":
                    ffparams_wo_charge[key][key2] = ffparams[key][key2]
        elif key == "VsiteForce":
            pass
        else:
            ffparams_wo_charge[key] = ffparams[key]
    pots = ff.createPotential(topology,
                              nonbondedMethod=app.PME,
                              nonbondedCutoff=rc*unit.nanometer,
                              useDispersionCorrection=useDispersionCorrection)
    efunc = jit(pots.getPotentialFunc())
    if ensemble in ["isonpt", "anisonpt", "trinpt"]:
        ens = "npt"
    else:
        ens = ensemble
    target_energy_function = buildTrajEnergyFunction(efunc,
                                                     cov_map,
                                                     rc,
                                                     ensemble=ens,
                                                     useFreud=True,
                                                     pressure=pressure)
    target_state = TargetState(Temperature_K, target_energy_function)
    weight, utarget = estimator.estimate_weight(target_state, parameters=ffparams_wo_charge)

    loss = 0.0
    weighted_results = {}
    for key in target_gt.keys():
        if key in ["density_gcm3", "La_A", "Lb_A", "Lc_A"]:
            density_pred = jnp.average(target_pred[key], weights=weight)
            weighted_results[key] = density_pred
            loss += target_gt[key]["weight"]*(target_gt[key]["gt"] \
                                - density_pred)**2 / target_gt[key]["gt"]**2
        elif key in ["rdf", "adf"]:
            weighted_results[key] = {}
            for kind in target_gt[key].keys():
                rdf_pred = (target_pred[key][kind] * weight.reshape((-1, 1))).sum(axis=0)
                weighted_results[key][kind] = rdf_pred
                loss_tmp = 0.0
                if losstype_distribfn == "wrightfactor":
                    loss_tmp = wrightfactor(rdf_pred, target_gt[key][kind]["gt"])
                elif losstype_distribfn == "jsdivergence":
                    loss_tmp = jsdivergence(rdf_pred, target_gt[key][kind]["gt"])
                loss += target_gt[key][kind]["weight"] * loss_tmp

    return loss, (utarget, weighted_results)

