#!/usr/bin/env python
import os
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

from ..calculator.md import VALID_ENSEMBLES
from ..analyzer.analyzer import (
    calc_density_frame,
    calc_cellpar_frame,
    calc_rdf_frame,
    calc_adf_frame,
    calc_dself,
)
from .loss import _DISTRIBUTION_LOSSES, _SCALAR_DEVIATIONS
from .properties import (
    PROPERTY_KINDS,
    REQUIRED_TARGET_KEYS,
    REQUIRED_VALIDATION_KEYS,
    SCALAR_TARGETS,
    VALIDATION_ONLY_PROPERTIES,
)


#: Metric comparing a validated distribution with its reference.
DEFAULT_DISTRIBUTION_METRIC = "wrightfactor"

#: Metric comparing a validated scalar with its reference. Signed, so that the
#: record says whether the property comes out too high or too low.
DEFAULT_SCALAR_METRIC = "relerr"


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

    # LJPME sums the long-range dispersion itself, so a dispersion correction
    # on top of it counts the same energy twice. OpenMM silently ignores the
    # flag under LJPME while DMFF honours it, which would leave the sampling
    # and the target state with different energies for the same configuration
    # and eat most of the effective samples before the fit even starts.
    if data["sampling"]["nonbondedmethod"] == "LJPME" and data["sampling"]["dispcorr"]:
        raise ValueError(
            "dispcorr must be False with the LJPME nonbonded method, which "
            "already accounts for the long-range dispersion."
        )

    # check for valid target_types
    if "targets" not in data:
        raise KeyError("Missing necessary key in data: targets")
    for target_name in data["targets"]:
        if target_name in VALIDATION_ONLY_PROPERTIES:
            raise ValueError(
                f"Target {target_name} is only available in the validation "
                "section: thermodynamic perturbation reweights stored "
                "configurations, which cannot give a dynamical property."
            )
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

    # validation targets are optional, an empty section keeps the callers from
    # having to special-case a missing key
    data["validation"] = _check_validation(data.get("validation"))

    # output parsed data as yaml, the filename is added with "_parsed"
    yaml_file = os.path.splitext(yaml_file)[0] + "_parsed.yaml"
    with open(yaml_file, "w") as file:
        yaml.dump(data, file)

    return data


def _check_validation(validation_params):
    """
    Check the ``validation`` section, keyed by a user chosen name::

        validation:
            dself_Li:
                property: dself_cm2s
                select: element Li
                gt: 1.0e-6

    Parameters
    ----------
    validation_params : dict or None
        Section read from the YAML.

    Returns
    -------
    validation_params : dict
        The same section, or an empty dict when it is absent, so that the
        callers never have to special-case a missing key.
    """
    if validation_params is None:
        return {}

    for name, block in validation_params.items():
        if "property" not in block:
            raise KeyError(f"Missing property for validation target {name}.")
        prop = block["property"]
        if prop not in REQUIRED_VALIDATION_KEYS:
            raise ValueError(
                f"Invalid validation property {prop} for {name}. Must be one of "
                f"{list(REQUIRED_VALIDATION_KEYS)}."
            )
        for key in REQUIRED_VALIDATION_KEYS[prop]:
            if key not in block:
                raise KeyError(f"Missing {key} for validation target {name}.")
        is_distribution = PROPERTY_KINDS[prop] == "distribution"
        metrics = _DISTRIBUTION_LOSSES if is_distribution else _SCALAR_DEVIATIONS
        default = (
            DEFAULT_DISTRIBUTION_METRIC if is_distribution else DEFAULT_SCALAR_METRIC
        )
        metric = block.get("metric", default)
        if metric not in metrics:
            raise ValueError(
                f"Invalid metric {metric} for validation target {name}. Must be "
                f"one of {list(metrics)}."
            )
        # a metric is a distance to a reference, so asking for one without
        # giving the reference is a mistake rather than something to ignore
        if not is_distribution and "metric" in block and block.get("gt") is None:
            raise KeyError(
                f"Missing gt for validation target {name}: metric {metric} "
                "measures the deviation from a reference value."
            )

    return validation_params


def _score_scalar(value, block):
    """
    Deviation of a computed scalar from its reference value.

    Parameters
    ----------
    value : float
        Value computed from the trajectory.
    block : dict
        Validation block, whose ``metric`` picks how the deviation is measured
        and whose ``gt`` is the reference it is measured against.

    Returns
    -------
    float or None
        The deviation, or None when the block declares no reference: a scalar
        is worth watching even without one, and then only its value is kept.
    """
    if block.get("gt") is None:
        return None
    metric = _SCALAR_DEVIATIONS[block.get("metric", DEFAULT_SCALAR_METRIC)]
    return float(metric(float(value), float(block["gt"])))


def _score_distribution(pred, block):
    """
    Distance between a computed distribution and its reference, with the two
    curves behind it.

    A whole curve cannot be followed epoch by epoch, so what is recorded is the
    same metric the loss uses on the fitted distributions: zero for a perfect
    match, larger the further apart the two curves are. The reference file is
    the one the targets use, ``x`` in the first column and the distribution in
    the second.
    """
    x, gt = np.loadtxt(block["gt"]).T[:2]
    if len(pred) != len(gt):
        raise ValueError(
            f"The computed distribution has {len(pred)} bins while the "
            f"reference {block['gt']} has {len(gt)}. Check the cutoff and the "
            "bin width against the ones the reference was made with."
        )
    metric = _DISTRIBUTION_LOSSES[block.get("metric", DEFAULT_DISTRIBUTION_METRIC)]
    return float(metric(jnp.array(pred), jnp.array(gt))), np.column_stack([x, pred, gt])


def get_validation_gt(validation_params: dict):
    """
    Get the reference values of the validation targets that declare one.

    Parameters
    ----------
    validation_params : dict
        Validation section as returned by :func:`parser_dmffyaml`.

    Returns
    -------
    validation_gt : dict
        Reference value of every scalar block carrying a ``gt`` key. The
        distributions are left out: their ``gt`` is a file, and what they
        record is already the distance to it.
    """
    return {
        name: float(block["gt"])
        for name, block in validation_params.items()
        if block.get("gt") is not None
        and PROPERTY_KINDS[block["property"]] == "scalar"
    }


def get_validation_pred(xtcfile, pdbfile, validation_params: dict):
    """
    Get the validation values of one trajectory from the parameters obtained by
    parser_dmffyaml().

    Unlike the targets these are one number per trajectory rather than
    per-frame arrays: they are scores watched along the optimization, not
    quantities reweighted by MBAR.

    Parameters
    ----------
    xtcfile : str
        Path to the XTC file.
    pdbfile : str
        Path to the PDB file, with the virtual sites of the trajectory.
    validation_params : dict
        Parameters for the validation targets.
        keys(validation_params) = names chosen by the user

    Returns
    -------
    validation_pred : dict
        Value of every scalar block, keyed by its name. A distribution has no
        single value to record, so it does not appear here.
    validation_dev : dict
        Deviation of every block that has a reference from it, keyed by its
        name and measured with the ``metric`` of the block. Zero for a perfect
        match; the deviations are kept apart from one another rather than
        summed into a score.
    validation_curves : dict
        Curve behind the value, for the blocks that have one: the MSD as
        ``(lagtime_ps, msd_A2)`` columns, a distribution as ``(x, pred, gt)``
        columns.
    """
    validation_pred = {}
    validation_dev = {}
    validation_curves = {}

    for name, block in validation_params.items():
        prop = block["property"]
        if prop not in REQUIRED_VALIDATION_KEYS:
            raise ValueError(
                f"Invalid validation property {prop} for {name}. Must be one of "
                f"{list(REQUIRED_VALIDATION_KEYS)}."
            )
        # unwrapping edits the trajectory in place, so each block reads its own
        # Universe rather than inheriting the transformations of the previous one
        u = MDAnalysis.Universe(pdbfile, xtcfile)

        if prop == "dself_cm2s":
            # the MSD comes back with the coefficient so that the fit can be
            # judged afterwards: one fitted outside the diffusive regime looks
            # just as reasonable as a good one until the curve is seen
            value, lagtime_ps, msd_A2 = calc_dself(
                u,
                select=block["select"],
                msd_type=block.get("msd_type", "xyz"),
                fit_range=tuple(block.get("fit_range", (0.1, 0.5))),
                start=block.get("start"),
                stop=block.get("stop"),
                step=block.get("step"),
                return_msd=True,
            )
            validation_curves[name] = np.column_stack([lagtime_ps, msd_A2])
        elif prop == "density_gcm3":
            value = np.mean(calc_density_frame(u))
        elif prop in ["La_A", "Lb_A", "Lc_A"]:
            value = np.mean(calc_cellpar_frame(u, prop))
        elif prop == "rdf":
            pred = calc_rdf_frame(
                u,
                block["elem1"],
                block["elem2"],
                rmax=block["rcut12_A"],
                dr=block.get("dr_A", 0.01),
                only_intermolecular=block.get("intermolecular", False),
            ).mean(axis=0)
            validation_dev[name], validation_curves[name] = _score_distribution(
                pred, block
            )
            continue
        elif prop == "adf":
            pred = calc_adf_frame(
                xtcfile,
                pdbfile,
                block["elem1"],
                block["elem2"],
                block["elem3"],
                rcut12=block["rcut12_A"],
                rcut23=block["rcut23_A"],
            ).mean(axis=0)
            validation_dev[name], validation_curves[name] = _score_distribution(
                pred, block
            )
            continue

        validation_pred[name] = float(value)
        deviation = _score_scalar(value, block)
        if deviation is not None:
            validation_dev[name] = deviation

    return validation_pred, validation_dev, validation_curves


def plot_validation(history: list, gt: dict = None, label="validation",
                    baseline: float = None):
    """
    Plot every validation value against the epoch.

    The same figure serves the values and their deviations: the values are
    read against their reference, the deviations against zero.

    Parameters
    ----------
    history : list of dict
        Records written by the trainer, each with an ``epoch`` and an ``ffxml``
        key describing the force field measured, and one key per value.
    gt : dict, optional
        Reference values, keyed like the records. A dashed line is drawn for
        the values that have one.
    label : str, optional
        Stem of the figure written, ``{label}.png``.
    baseline : float, optional
        Value a dashed line is drawn at on every panel, 0 for a plot of
        deviations. Use it instead of `gt` when every panel shares one
        reference.
    """
    # a replica resampled for the first time adds its keys mid-history, so the
    # columns are collected over every record rather than from the first one.
    # epoch and ffxml say which force field the record describes, they are not
    # values to plot
    keys = []
    for record in history:
        keys.extend(
            key for key in record
            if key not in ("epoch", "ffxml") and key not in keys
        )
    if len(keys) == 0:
        return

    gt = {} if gt is None else gt
    epochs = [record["epoch"] for record in history]
    fig, ax = plt.subplots(len(keys), 1, figsize=(5, 2.5 * len(keys)), squeeze=False)
    for axis, key in zip(ax[:, 0], keys):
        axis.set_title(key)
        axis.plot(epochs, [record.get(key, np.nan) for record in history], marker="o")
        if key in gt:
            axis.axhline(gt[key], linestyle="--", color="k", label="gt")
            axis.legend()
        elif baseline is not None:
            axis.axhline(baseline, linestyle="--", color="k")
        axis.set_xlabel("Epoch")
    plt.tight_layout()
    fig.savefig(f"{label}.png", bbox_inches="tight")
    plt.close(fig)


def plot_validation_curves(curves: dict, validation_params: dict,
                           label="validation_curves"):
    """
    Plot the curves the validation values of one replica were derived from.

    They say why a value is what it is: whether the MSD is straight over the
    fitted window, or where a distribution departs from its reference.

    Parameters
    ----------
    curves : dict
        Curves as returned by :func:`get_validation_pred`.
    validation_params : dict
        Validation section of the replica, for the axis labels.
    label : str, optional
        Stem of the figure written, ``{label}.png``.
    """
    if len(curves) == 0:
        return

    fig, ax = plt.subplots(len(curves), 1, figsize=(5, 2.5 * len(curves)),
                           squeeze=False)
    for axis, (name, curve) in zip(ax[:, 0], curves.items()):
        prop = validation_params[name]["property"]
        axis.set_title(name)
        if PROPERTY_KINDS[prop] == "distribution":
            axis.plot(curve[:, 0], curve[:, 1], alpha=0.5, label="pred")
            axis.plot(curve[:, 0], curve[:, 2], label="gt")
            axis.legend()
            axis.set_xlabel("r ($\\mathrm{\\AA}$)" if prop == "rdf"
                            else "angle (deg)")
        else:
            axis.plot(curve[:, 0], curve[:, 1])
            axis.set_xlabel("Time (ps)")
            axis.set_ylabel("MSD ($\\mathrm{\\AA}^2$)")
    plt.tight_layout()
    fig.savefig(f"{label}.png", bbox_inches="tight")
    plt.close(fig)


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
