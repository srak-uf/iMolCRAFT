#!/usr/bin/env python
import os
import sys
import yaml
import numpy as np
import openmm
from openmm import app
import openmm.unit as unit

import jax.numpy as jnp
import dmff
from dmff.operators.templatetype import TemplateATypeOperator
from dmff.operators.templatevsite import TemplateVSiteOperator

import MDAnalysis

import matplotlib.pyplot as plt
import math

from ..analyzer.analyzer import (
    calc_density_frame,
    calc_cellpar_frame,
    calc_rdf_frame,
    calc_adf_frame,
)


#: Targets given as a single reference number.
SCALAR_TARGETS = ("density_gcm3", "La_A", "Lb_A", "Lc_A")

#: Keys every target must provide. Scalar targets carry them directly, while
#: rdf / adf carry one such block per named pair or triplet.
REQUIRED_TARGET_KEYS = {
    "density_gcm3": ("gt", "weight"),
    "La_A": ("gt", "weight"),
    "Lb_A": ("gt", "weight"),
    "Lc_A": ("gt", "weight"),
    "rdf": ("elem1", "elem2", "rcut12_A"),
    "adf": ("elem1", "elem2", "elem3", "rcut12_A", "rcut23_A"),
}

#: Ensembles md_sample knows how to set up.
VALID_ENSEMBLES = ("nve", "nvt", "isonpt", "anisonpt", "trinpt")


def parser_dmffyaml(yaml_file):
    """
    Parse the YAML file for DMFF to extract relevant information.

    Parameters
    ----------
    yaml_file : str
        Path to the YAML file.
    Returns
    -------
    data : dict
        Parsed data from the YAML file.
    """
    default_params = {
        "sampling": {
            "dt_fs": 1.0,
            "ensemble": "nvt",
            "temperature_K": 300,
            "pressure_bar": 1,
            "anneal_steps": 1,
            "anneal_Tmax": 400,
            "anneal_totaltime": 0,
            "nstxout": 100,
            "rcut_nm": 1.2,
            "nonbondedmethod": "PME",
            "dispcorr": False
        }
    }
    necessarykeys_sampling = ["init_structure", "relax_steps", "prod_steps"]

    with open(yaml_file, "r") as file:
        data = yaml.safe_load(file)

    # Check for required keys and set defaults if not present
    if "sampling" not in data:
        data["sampling"] = default_params["sampling"]
    for key in default_params["sampling"]:
        if key not in data["sampling"]:
            data["sampling"][key] = default_params["sampling"][key]

    # Check for necessarykeys_sampling, necessarykeys_forcefield
    for key in necessarykeys_sampling:
        if key not in data["sampling"]:
            raise KeyError(f"Missing necessary key in sampling: {key}")

    # Check for file existence
    if not os.path.isfile(data["sampling"]["init_structure"]):
        if os.path.isfile(
            os.path.join(os.path.dirname(yaml_file), data["sampling"]["init_structure"])
        ):
            data["sampling"]["init_structure"] = os.path.join(
                os.path.dirname(yaml_file), data["sampling"]["init_structure"]
            )
        else:
            raise FileNotFoundError(
                (
                    f"Initial structure file {data['sampling']['init_structure']} "
                    "not found."
                )
            )

    # Check for valid ensemble
    if data["sampling"]["ensemble"] not in VALID_ENSEMBLES:
        raise ValueError(
            f"Invalid ensemble {data['sampling']['ensemble']}. Must be one of "
            f"{list(VALID_ENSEMBLES)}."
        )

    # check for valid target_types
    if "targets" not in data:
        raise KeyError("Missing necessary key in data: targets")
    for target_name in data["targets"]:
        if target_name not in REQUIRED_TARGET_KEYS:
            raise ValueError(
                f"Invalid target type {target_name}. Must be one of "
                f"{list(REQUIRED_TARGET_KEYS)}."
            )
        required = REQUIRED_TARGET_KEYS[target_name]
        if target_name in SCALAR_TARGETS:
            for key in required:
                if key not in data["targets"][target_name]:
                    raise KeyError(f"Missing {key} for target {target_name}.")
        else:
            for key in data["targets"][target_name].keys():
                for required_key in required:
                    if required_key not in data["targets"][target_name][key]:
                        raise KeyError(
                            f"Missing {required_key} for target {target_name}: {key}."
                        )

    # output parsed data as yaml, the filename is added with "_parsed"
    yaml_file = os.path.splitext(yaml_file)[0] + "_parsed.yaml"
    with open(yaml_file, "w") as file:
        yaml.dump(data, file)

    return data


def get_target_gt(target_params: dict):
    """
    Get the ground truth values for the targets from the parameters obtained by
    parser_dmffyaml().

    Parameters
    ----------
    target_params : dict
        Parameters for the target parameters.
        keys(target_params) = ['density_gcm3', 'La_A', 'Lb_A', 'Lc_A', 'rdf', 'adf']

    Returns
    -------
    target_gt : dict
        Dictionary containing the ground truth values for the targets.
    """
    target_gt = {}
    for target_name in target_params.keys():
        if target_name in SCALAR_TARGETS:
            target_gt[target_name] = {}
            target_gt[target_name]["gt"] = float(target_params[target_name]["gt"])
            target_gt[target_name]["weight"] = float(
                target_params[target_name]["weight"]
            )
        else:
            target_gt[target_name] = {}
            for key in target_params[target_name].keys():
                target_gt[target_name][key] = {}
                gt_file = target_params[target_name][key]["gt"]
                weight = target_params[target_name][key]["weight"]
                target_gt[target_name][key]["gt"] = np.loadtxt(gt_file).T[1]
                target_gt[target_name][key]["weight"] = float(weight)
    return target_gt


def neutralize(ffparams, natoms_list, nc=0, target_lists=None, target_charges=None):
    """
    Neutralize the system by adjusting the charges.

    Parameters
    ----------
    ffparams : dict
        Force field parameters.
    natoms_list : jnp.ndarray
        List of number of atoms in the system.
    nc : float, optional
        The net charge of the system.
    target_lists : list, optional
        List of targets to make charge the target_charges value. If None, all atoms are
        neutralized. ex: [[0, 1], [2, 3]] means that the first two atoms are neutralized
        to target_charges[0] and the next two atoms are neutralized to
        target_charges[1]. For example: [[0, 1], [2, 3]] means that the total charge of
        first two atoms based on natoms_list are neutralized to target_charges[0]. The
        next two atoms are neutralized to target_charges[1].
    target_charges : list, optional
        List of target charges for the atoms. If None, the charges are neutralized to
        zero. ex: [0.0, 2.0] means that the first two atoms are neutralized to 0.0 and
        the next two atoms are neutralized to 2.0.

    Returns
    -------
    ffparams : dict
        Updated force field parameters with neutralized charges.
    """

    if len(ffparams["NonbondedForce"]["charge"]) != len(natoms_list):
        raise ValueError(
            "len(ffparams['NonbondedForce']['charge']) != len(natoms_list): "
            f"{len(ffparams['NonbondedForce']['charge'])} != {len(natoms_list)}"
        )

    if target_lists is not None and target_charges is not None:
        if len(target_lists) != len(target_charges):
            raise ValueError(
                "len(target_lists) != len(target_charges): "
                f"{len(target_lists)} != {len(target_charges)}"
            )
        for i, target_list in enumerate(target_lists):
            net_q = jnp.dot(
                ffparams["NonbondedForce"]["charge"][jnp.array(target_list)],
                natoms_list[jnp.array(target_list)],
            )
            desired_q = target_charges[i]
            natoms_list_sum = natoms_list[jnp.array(target_list)].sum()
            charges_mod = (
                ffparams["NonbondedForce"]["charge"][jnp.array(target_list)]
                + (desired_q - net_q) / natoms_list_sum
            )
            ffparams["NonbondedForce"]["charge"] = (
                ffparams["NonbondedForce"]["charge"]
                .at[jnp.array(target_list)]
                .set(charges_mod)
            )

        if nc is not None:
            # Constrained charge index
            target_all = list(set([i for sublist in target_lists for i in sublist]))

            # all index
            nottarget_list = jnp.array(
                [natoms_list[i] for i in range(len(natoms_list)) if i not in target_all]
            )
            nottarget_idx = jnp.array(
                [i for i in range(len(natoms_list)) if i not in target_all]
            )

            if nottarget_list.sum() <= 0:
                raise ValueError("All atoms are constrained. Cannot neutralize.")

            # Update charges for non-targeted atoms
            net_q = jnp.dot(ffparams["NonbondedForce"]["charge"], natoms_list)
            ffparams["NonbondedForce"]["charge"] = (
                ffparams["NonbondedForce"]["charge"]
                .at[nottarget_idx]
                .set(
                    ffparams["NonbondedForce"]["charge"][nottarget_idx]
                    - (net_q - nc) / nottarget_list.sum()
                )
            )
    elif nc is not None:
        net_q = jnp.dot(ffparams["NonbondedForce"]["charge"], natoms_list)
        ffparams["NonbondedForce"]["charge"] = (
            ffparams["NonbondedForce"]["charge"] - (net_q - nc) / natoms_list.sum()
        )

    return ffparams


def _get_charges_types(topdata: app.Topology, ff, gen_dmfftop=False):
    """
    Get the charges and types of the atoms in the topology data.
    Parameters
    ----------
    topdata : app.Topology
        Topology data.
    ff : Hamiltonian
        Force field object.
    gen_dmfftop : bool, optional
        Whether to generate a DMFF topology. Default is False.
    Returns
    -------
    charges : list
        List of charges for each atom.
    types : list
        List of types for each atom.
    topdata : DMFFTopology, optional
        DMFF topology data if gen_dmfftop is True.
    """
    vsite = TemplateVSiteOperator(ff.ffinfo)
    template = TemplateATypeOperator(ff.ffinfo)
    topdata = dmff.api.DMFFTopology(from_top=topdata)
    topdata = vsite(topdata)
    topdata = template(topdata)
    charges = [a.meta["charge"] for a in topdata.atoms()]
    types = [a.meta["type"] for a in topdata.atoms()]
    if gen_dmfftop:
        return charges, types, topdata
    else:
        return charges, types


def update_ffinfo_from_rescharges(ff, rescharges):
    """
    Update the force field information with the residue charges.
    Parameters
    ----------
    ff : Hamiltonian
        Force field object.
    rescharges : list
        List of residue charges dictionary.
    Returns
    -------
    ff : Hamiltonian
        Updated force field object.
    """
    for i_res in range(len(ff.ffinfo["Residues"])):
        for key in rescharges[i_res]:
            for key2 in rescharges[i_res][key]:
                res_idx = rescharges[i_res][key][key2]["index"]
                for i in res_idx:
                    ff.ffinfo["Residues"][i_res]["particles"][i]["charge"] = rescharges[
                        i_res
                    ][key][key2]["value"]
    return ff


def update_ffinfo_from_params(ff, params):
    """
    Update the force field information with the parameters.
    Parameters
    ----------
    ff : Hamiltonian
        Force field object.
    params : dict
        Parameters for the force field.
    Returns
    -------
    ff : Hamiltonian
        Updated force field object.
    """
    idx = 0
    for i_res in range(len(ff.ffinfo["Residues"])):
        for i, _ in enumerate(ff.ffinfo["Residues"][i_res]["particles"]):
            ff.ffinfo["Residues"][i_res]["particles"][i]["charge"] = params[
                "NonbondedForce"
            ]["charge"][idx]
            idx += 1

    idx = 0
    if "VirtualSite" in params:
        vs = params["VirtualSite"]
        w2_ave2 = vs["vsite_w2_type_2"] if "vsite_w2_type_2" in vs else []
        if len(w2_ave2) > 0:
            w1_ave2 = jnp.ones(w2_ave2.shape) - w2_ave2
            ave2_idx = 0
        w2_ave3 = vs["vsite_w2_type_3"] if "vsite_w2_type_3" in vs else []
        w3_ave3 = vs["vsite_w3_type_3"] if "vsite_w3_type_3" in vs else []
        if len(w2_ave3) > 0 or len(w3_ave3) > 0:
            w1_ave3 = jnp.ones(w2_ave3.shape) - w2_ave3 - w3_ave3
            ave3_idx = 0

        # "vsite_w2_type_2, vsite_w2_type_3, vsite_w3_type_3"以外のkeyがあればエラー
        for key in params["VirtualSite"].keys():
            if key not in ["vsite_w2_type_2", "vsite_w2_type_3", "vsite_w3_type_3"]:
                raise ValueError(f"Unknown key in VirtualSite params: {key}")

        for i_res in range(len(ff.ffinfo["Residues"])):
            for i, _ in enumerate(ff.ffinfo["Residues"][i_res]["vsites"]):
                if ff.ffinfo["Residues"][i_res]["vsites"][i]["type"] == "average2":
                    ff.ffinfo["Residues"][i_res]["vsites"][i]["weight1"] = float(
                        w1_ave2[ave2_idx]
                    )
                    ff.ffinfo["Residues"][i_res]["vsites"][i]["weight2"] = float(
                        w2_ave2[ave2_idx]
                    )
                    ave2_idx += 1
                elif ff.ffinfo["Residues"][i_res]["vsites"][i]["type"] == "average3":
                    ff.ffinfo["Residues"][i_res]["vsites"][i]["weight1"] = float(
                        w1_ave3[ave3_idx]
                    )
                    ff.ffinfo["Residues"][i_res]["vsites"][i]["weight2"] = float(
                        w2_ave3[ave3_idx]
                    )
                    ff.ffinfo["Residues"][i_res]["vsites"][i]["weight3"] = float(
                        w3_ave3[ave3_idx]
                    )
                    ave3_idx += 1
                # n_weights = len(
                #     [
                #         key
                #         for key in ff.ffinfo["Residues"][i_res]["vsites"][i].keys()
                #         if key.startswith("weight")
                #     ]
                # )
                # for i_weight in range(n_weights):
                #     ff.ffinfo["Residues"][i_res]["vsites"][i][f"weight{i_weight+1}"] = (
                #         params["VirtualSite"]["weight"][idx]
                #     )
                #     idx += 1

    return ff


def get_chgparams_from_rescharges(params, rescharges):
    natoms = np.array(
        [
            len(rescharges[i_res][key][key2]["index"])
            for i_res in range(len(rescharges))
            for key in rescharges[i_res]
            for key2 in rescharges[i_res][key]
        ]
    ).sum()
    params["NonbondedForce"]["charge"] = jnp.zeros(natoms)
    ishift = 0
    for res in rescharges:
        natoms = 0
        for t in res:
            for key in res[t]:
                for idx in res[t][key]["index"]:
                    params["NonbondedForce"]["charge"] = (
                        params["NonbondedForce"]["charge"]
                        .at[idx + ishift]
                        .set(res[t][key]["value"])
                    )
                    natoms += 1
        ishift += natoms
    return params


def vsiteinfo_to_params(ff, params):
    """
    Convert vsite information from ff.ffinfo to params.
    """
    # vsiteinfo = ff.ffinfo["Residues"][0]["vsites"]
    weights = []
    for i_res in range(len(ff.ffinfo["Residues"])):
        for i_vs, _ in enumerate(ff.ffinfo["Residues"][i_res]["vsites"]):
            weights_tmp = [
                value
                for key, value in ff.ffinfo["Residues"][i_res]["vsites"][i_vs].items()
                if key.startswith("weight")
            ]
            # weights_tmpのすべての要素をweightsに追加
            weights.extend(weights_tmp)

    params["VirtualSite"] = {}
    params["VirtualSite"]["weight"] = jnp.zeros(len(weights))
    params["VirtualSite"]["scale"] = jnp.zeros(len(weights))
    for i in range(len(weights)):
        params["VirtualSite"]["weight"] = (
            params["VirtualSite"]["weight"].at[i].set(weights[i])
        )

    return params


def update_rescharges_from_params(rescharges, params):
    ishift = 0
    for res in rescharges:
        natoms = 0
        for t in res:
            for key in res[t]:
                charge_forave = []
                for idx in res[t][key]["index"]:
                    natoms += 1
                    charge_forave.append(
                        params["NonbondedForce"]["charge"][idx + ishift]
                    )
                res[t][key]["value"] = np.mean(charge_forave)
                # res[t][key]["value"] = params["NonbondedForce"]["charge"][idx+ishift]
        ishift += natoms
    return rescharges


def get_rescharges_from_residues(ff, ratio=None):
    """
    Get the charges and types of the residues in the force field.
    Parameters
    ----------
    ff : Hamiltonian
        Force field object.
    ratio : list, optional
        List of ratios for each residue. Default is None.

    Returns
    -------
    rescharges : list
        List of residue charges.
    natoms_list : jnp.ndarray, optional
        List of number of atoms in the system if ratio is provided.
    """
    residues = ff.ffinfo["Residues"]
    rescharges = []

    if ratio is not None:
        if len(residues) != len(ratio):
            raise ValueError(
                f"len(residues) != len(ratio): {len(residues)} != {len(ratio)}"
            )
        natoms_list = []

    for i_res in range(len(residues)):
        residue = residues[i_res]
        chargedict = {}
        for i, p in enumerate(residue["particles"]):
            t = p["type"]
            c = p["charge"]
            if ratio is not None:
                natoms_list.append(ratio[i_res])

            if t not in chargedict:
                chargedict[t] = {}

            if len(chargedict[t]) == 0:
                chargedict[t][f"{t}_0"] = {"value": c, "index": [i]}
            elif len(chargedict[t]) > 0:
                sameTYPEflag = False
                for c_idx, key in enumerate(chargedict[t]):
                    if np.isclose(float(chargedict[t][key]["value"]), c, atol=1e-3):
                        chargedict[t][key]["index"].append(i)
                        sameTYPEflag = True
                if not sameTYPEflag:
                    chargedict[t][f"{t}_{c_idx+1}"] = {"value": c, "index": [i]}
        rescharges.append(chargedict)

    if ratio is not None:
        return rescharges, jnp.array(natoms_list)
    else:
        return rescharges


def _make_barostat(ensemble, T):
    """
    Barostat matching the requested NPT flavour, or None for a fixed volume.

    The three NPT ensembles differ in how much of the box shape they let move:
    isotropic scaling, independent axes, or a fully flexible triclinic cell.
    """
    if ensemble == "isonpt":
        print("Isotropic pressure control")
        return openmm.MonteCarloBarostat(1.0 * unit.bar, T * unit.kelvin)
    if ensemble == "anisonpt":
        print("Anisotropic pressure control")
        return openmm.MonteCarloAnisotropicBarostat(
            [1.0 * unit.bar] * 3, T * unit.kelvin
        )
    if ensemble == "trinpt":
        return openmm.MonteCarloFlexibleBarostat(1.0 * unit.bar, T * unit.kelvin)
    return None


def md_sample(
    initialpdb,
    ffxml,
    trajectory,
    rc=1.2,
    T=300,
    anneal_Tmax=300,
    anneal_steps=0,
    anneal_totalsteps=0,
    dt=1.0,
    nstxout=1000,
    relax_steps=100000,
    prod_steps=2000000,
    ensemble="nvt",
    nonbondedmethod="PME",
    useDispersionCorrection=False,
    useHbondConstraint=True,
    rigidWater=False,
    device="CPU"
):
    """
    Run MD simulation with OpenMM

    Parameters
    ----------
    initialpdb : str
        Path to the initial PDB file
    ffxml : str
        Path to the force field XML file
    trajectory : str
        Path to the output trajectory file
    sampling_params : dict
        Dictionary containing sampling parameters such as temperature,
        annealing steps, etc.
    useDispersionCorrection : bool, optional
        Whether to use dispersion correction in the nonbonded force. Default is False.

    Returns
    -------
    state_init : dict
        Dictionary containing the initial state of the system
    """
    pdb = app.PDBFile(initialpdb)
    forcefield = app.ForceField(ffxml)

    modeller = app.Modeller(pdb.topology, pdb.getPositions())
    modeller.addExtraParticles(forcefield)
    pos = modeller.getPositions()
    topology = modeller.topology
    # modellerをpdbに書き出す
    # app.PDBFile.writeFile(topology, pos, open("modeller.pdb", "w"))

    if nonbondedmethod == "PME":
        nonbondedmethod = app.PME
    elif nonbondedmethod == "LJPME":
        nonbondedmethod = app.LJPME

    constraints = {"constraints": app.HBonds} if useHbondConstraint else {}
    system = forcefield.createSystem(
        topology,
        nonbondedMethod=nonbondedmethod,
        nonbondedCutoff=rc * unit.nanometer,
        rigidWater=rigidWater,
        **constraints,
    )

    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            force.setUseDispersionCorrection(useDispersionCorrection)

    print(f"Using {ensemble} ensemble")
    barostat = _make_barostat(ensemble, T)
    if barostat is not None:
        system.addForce(barostat)

    integrator = openmm.LangevinIntegrator(
        T * unit.kelvin, 1 / unit.picosecond, dt * unit.femtosecond
    )

    platform = openmm.Platform.getPlatformByName(device)

    simulation = app.Simulation(topology, system, integrator, platform)
    xtcfile = os.path.join("xtcfiles", trajectory)
    try:
        os.remove(xtcfile)
    except Exception:
        pass
    simulation.context.setPositions(pos)
    print("== Energy minimization ==")
    simulation.minimizeEnergy()
    simulation.context.setVelocitiesToTemperature(T * unit.kelvin)

    simulation.reporters.append(
        app.StateDataReporter(
            sys.stdout,
            nstxout,
            potentialEnergy=True,
            density=True,
            step=True,
            remainingTime=True,
            speed=True,
            totalSteps=relax_steps + prod_steps,
        )
    )

    # relaxation run
    # SA
    if anneal_totalsteps > 0:
        print("== Start Simulated Annealing ==")
        deltaT = (T - anneal_Tmax) / anneal_steps
        step_pertemp = int(anneal_totalsteps / anneal_steps)
        for i in range(anneal_steps):
            integrator.setTemperature((anneal_Tmax + deltaT * i) * unit.kelvin)
            simulation.step(step_pertemp)
    # relax at desired temperature
    print("== Start Relaxation ==")
    integrator.setTemperature(T * unit.kelvin)
    simulation.step(relax_steps)
    # production run
    print("== Start Production ==")
    os.makedirs("xtcfiles", exist_ok=True)
    simulation.reporters.append(app.XTCReporter(xtcfile, nstxout))
    simulation.step(prod_steps)
    return xtcfile


def get_target_pred_frame(xtcfile, pdbfile, target_params: dict):
    """
    Get the predicted values for the targets from the parameters obtained by
    parser_dmffyaml().

    Parameters
    ----------
    xtcfile : str
        Path to the XTC file.
    pdbfile : str
        Path to the PDB file.
    target_params : dict
        Parameters for the target parameters.
        keys(target_params) = ['density_gcm3', 'La_A', 'Lb_A', 'Lc_A', 'rdf', 'adf']

    Returns
    -------
    target_pred : dict
        Dictionary containing the predicted values for the targets.
    """
    target_pred = {}
    u = MDAnalysis.Universe(pdbfile, xtcfile)

    for target_name in target_params.keys():
        if target_name == "density_gcm3":
            target_pred[target_name] = calc_density_frame(u)
        elif target_name in ["La_A", "Lb_A", "Lc_A"]:
            target_pred[target_name] = calc_cellpar_frame(u, target_name)
        elif target_name == "rdf":
            target_pred[target_name] = {}
            for key in target_params[target_name].keys():
                elem1 = target_params[target_name][key]["elem1"]
                elem2 = target_params[target_name][key]["elem2"]
                rcut12_A = target_params[target_name][key]["rcut12_A"]
                dr_A = target_params[target_name][key].get("dr_A", 0.01)
                inter_molecular_flag = target_params[target_name][key].get(
                    "intermolecular", False
                )
                target_pred[target_name][key] = calc_rdf_frame(
                    u, elem1, elem2, rmax=rcut12_A, dr=dr_A,
                    only_intermolecular=inter_molecular_flag,
                )
        elif target_name == "adf":
            target_pred[target_name] = {}
            for key in target_params[target_name].keys():
                elem1 = target_params[target_name][key]["elem1"]
                elem2 = target_params[target_name][key]["elem2"]
                elem3 = target_params[target_name][key]["elem3"]
                rcut12_A = target_params[target_name][key]["rcut12_A"]
                rcut23_A = target_params[target_name][key]["rcut23_A"]
                target_pred[target_name][key] = calc_adf_frame(
                    xtcfile,
                    pdbfile,
                    elem1,
                    elem2,
                    elem3,
                    rcut12=rcut12_A,
                    rcut23=rcut23_A,
                )
    return target_pred


def plot_compare(target_gt, target_pred_frame, label="sample"):
    num_plots = 0
    for key in target_gt.keys():
        if key in SCALAR_TARGETS:
            num_plots += 1
        elif key in ["rdf", "adf"]:
            for kind in target_gt[key].keys():
                num_plots += 1
    num_raw = np.max([math.ceil(num_plots / 2), 2])
    fig, ax = plt.subplots(num_raw, 2, figsize=(6.5, 2.5 * num_raw))
    i_plot = 0
    for key in target_gt.keys():
        if key in SCALAR_TARGETS:
            axis = ax[i_plot // 2, i_plot % 2]
            axis.set_title(key)
            x = ["GT", "FF"]
            y = [target_gt[key]["gt"], target_pred_frame[key].mean()]
            axis.bar(x, y, width=0.35)
            # barごとに値を表示
            for i, v in enumerate(y):
                axis.text(i, v + 0.01, str(round(v, 3)), ha="center", va="bottom")
            axis.set_ylim(0, y[0] * 1.2)
            i_plot += 1
        elif key in ["rdf", "adf"]:
            for kind in target_gt[key].keys():
                axis = ax[i_plot // 2, i_plot % 2]
                axis.set_title(key + "_" + kind)
                axis.plot(np.mean(target_pred_frame[key][kind], axis=0), alpha=0.5)
                axis.plot(target_gt[key][kind]["gt"], label="gt")
                i_plot += 1
    plt.tight_layout()
    fig.savefig(f"{label}.png", bbox_inches="tight")
    plt.close(fig)
