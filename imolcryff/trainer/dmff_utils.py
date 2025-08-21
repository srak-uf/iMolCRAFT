#!/usr/bin/env python
from dmff import Hamiltonian
import os, sys, copy
import yaml, pickle
import numpy as np
import mdtraj as md
import openmm
from openmm import app
import openmm.unit as unit

import jax.numpy as jnp
from jax import value_and_grad, jit
import dmff
from dmff.operators.templatetype import TemplateATypeOperator
from dmff.operators.templatevsite import TemplateVSiteOperator
from dmff import Hamiltonian, DMFFTopology
from dmff.mbar import TargetState, buildTrajEnergyFunction

sys.path.append(os.path.abspath('.'))
from ..analyzer.analyzer import calc_density_frame, calc_cellpar_frame, calc_rdf_frame, calc_adf_frame
import MDAnalysis

import matplotlib.pyplot as plt
import math

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
        "sampling": 
            {
                "dt_fs": 1.0,
                "ensemble": "nvt", 
                "temperature_K": 300,
                "pressure_bar": 1,
                "anneal_steps": 1,
                "anneal_Tmax": 400,
                "anneal_totaltime": 0,
                "nstxout": 100
            }
    }
    necessarykeys_sampling = ["init_structure", "relax_steps", "prod_steps"]
    target_types = ['density_gcm3', 'La_A', 'Lb_A', 'Lc_A', 'rdf', 'adf']

    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)
    
    # Check for required keys and set defaults if not present
    if 'sampling' not in data:
        data['sampling'] = default_params['opt_sample']
    for key in default_params['sampling']:
        if key not in data['sampling']:
            data['sampling'][key] = default_params['sampling'][key]

    # Check for necessarykeys_sampling, necessarykeys_forcefield
    for key in necessarykeys_sampling:
        if key not in data['sampling']:
            raise KeyError(f"Missing necessary key in sampling: {key}")
        
    # Check for file existence
    if not os.path.isfile(data['sampling']['init_structure']):
        raise FileNotFoundError(f"Initial structure file {data['sampling']['init_structure']} not found.")
        
    # Check for valid ensemble
    valid_ensembles = ["nve", "nvt", "isonpt", "anisonpt", "trinpt"]
    if data['sampling']['ensemble'] not in valid_ensembles:
        raise ValueError(f"Invalid ensemble {data['sampling']['ensemble']}. Must be one of {valid_ensembles}.")

    # check for valid target_types
    valid_targets = ["density_gcm3", "La_A", "Lb_A", "Lc_A", "rdf", "adf"]
    if 'targets' not in data:
        raise KeyError("Missing necessary key in data: targets")
    for target_name in data['targets']:
        if target_name not in valid_targets:
            raise ValueError(f"Invalid target type {target_name}. Must be one of {valid_targets}.")
        if target_name in ["density_gcm3", "La_A", "Lb_A", "Lc_A"]:
            if "gt" not in data['targets'][target_name]:
                raise KeyError(f"Missing ground truth for target {target_name}.")
            if "weight" not in data['targets'][target_name]:
                raise KeyError(f"Missing weight for target {target_name}.")
        if target_name in ["rdf"]:
            for key in data["targets"][target_name].keys():
                if "elem1" not in data['targets'][target_name][key]:
                    raise KeyError(f"Missing elem1 for target {target_name}: {key}.")
                if "elem2" not in data['targets'][target_name][key]:
                    raise KeyError(f"Missing elem2 for target {target_name}: {key}.")
                if "rcut12_A" not in data['targets'][target_name][key]:
                    raise KeyError(f"Missing rcut12_A for target {target_name}: {key}.")
        if target_name in ["adf"]:
            for key in data["targets"][target_name].keys():
                if "elem1" not in data["targets"][target_name][key]:
                    raise KeyError(f"Missing elem1 for target {target_name}: {key}.")
                if "elem2" not in data["targets"][target_name][key]:
                    raise KeyError(f"Missing elem2 for target {target_name}: {key}.")
                if "elem3" not in data["targets"][target_name][key]:
                    raise KeyError(f"Missing elem3 for target {target_name}: {key}.")
                if "rcut12_A" not in data["targets"][target_name][key]:
                    raise KeyError(f"Missing rcut12_A for target {target_name}: {key}.")
                if "rcut23_A" not in data["targets"][target_name][key]:
                    raise KeyError(f"Missing rcut23_A for target {target_name}: {key}.")

    # output parsed data as yaml, the filename is added with "_parsed"
    yaml_file = os.path.splitext(yaml_file)[0] + "_parsed.yaml"
    with open(yaml_file, 'w') as file:
        yaml.dump(data, file)
    
    return data

def get_target_gt(target_params: dict):
    """
    Get the ground truth values for the targets from the parameters obtained by parser_dmffyaml().

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
        if target_name not in ["rdf", "adf"]:
            target_gt[target_name] = {}
            target_gt[target_name]["gt"] = float(target_params[target_name]["gt"])
            target_gt[target_name]["weight"] = float(target_params[target_name]["weight"])
        else:
            target_gt[target_name] = {}
            for key in target_params[target_name].keys():
                target_gt[target_name][key] = {}
                gt_file = target_params[target_name][key]["gt"]
                weight = target_params[target_name][key]["weight"]
                target_gt[target_name][key]["gt"] = np.loadtxt(gt_file).T[1]
                target_gt[target_name][key]["weight"] = float(weight)
    return target_gt

def merge_xml(ffxml_list, outxml):
    """
    Merge multiple XML files into one.
    Parameters
    ----------
    ffxml_list : list
        List of XML files to be merged.
    outxml : str
        Name of the output XML file.
    """
    ff = Hamiltonian(*ffxml_list)
    del_idx = []
    attribfromres_flag = False
    # ffinfo_nb = ff.ffinfo["Forces"]["NonbondedForce"]["node"]
    for i, f in enumerate(ff.ffinfo["Forces"]["NonbondedForce"]["node"]):
        if "name" in f and "attrib" in f:
            if f["name"] == "UseAttributeFromResidue" and f["attrib"]["name"] == "charge" and attribfromres_flag == False:
                attribfromres_flag = True
            elif f["name"] == "UseAttributeFromResidue" and f["attrib"]["name"] == "charge" and attribfromres_flag == True:
                del_idx.append(i)

    n_del = 0
    for i in del_idx:
        ff.ffinfo["Forces"]["NonbondedForce"]["node"].pop(i-n_del)
        n_del += 1
    
    os.makedirs("xmlfiles", exist_ok=True)
    ff.renderXML(os.path.join("xmlfiles",outxml))

    return os.path.join("xmlfiles",outxml)

import jax.numpy as jnp


def neutralize(ffparams, natoms_list, nc=0, target_lists=None, target_charges=None):
    """
    Neutralize the system by adjusting the charges.
    Parameters
    ----------
    ffparams : dict
        Force field parameters.
    natoms_list : jnp.ndarray
        List of number of atoms in the system.
    target_lists : list, optional
        List of targets to make charge the target_charges value. If None, all atoms are neutralized.
        ex: [[0, 1], [2, 3]] means that the first two atoms are neutralized to target_charges[0] and the next two atoms are neutralized to target_charges[1].
    target_charges : list, optional
        List of target charges for the atoms. If None, the charges are neutralized to zero.
        ex: [0.0, 2.0] means that the first two atoms are neutralized to 0.0 and the next two atoms are neutralized to 2.0.
    Returns
    -------
    ffparams : dict
        Updated force field parameters with neutralized charges.
    """
    if target_lists is not None and target_charges is not None:
        assert len(target_lists) == len(target_charges), "len(target_lists) != len(target_charges)"
        for i, target_list in enumerate(target_lists):
            net_q = jnp.dot(ffparams['NonbondedForce']['charges'][jnp.array(target_list)],
                            natoms_list[jnp.array(target_list)])
            desired_q = target_charges[i]
            natoms_list_sum = natoms_list[jnp.array(target_list)].sum()
            charges_mod = ffparams['NonbondedForce']['charges'][jnp.array(target_list)] + \
                             (desired_q - net_q) / natoms_list_sum
            ffparams['NonbondedForce']['charges'] = \
                ffparams['NonbondedForce']['charges'].at[jnp.array(target_list)].set(charges_mod)
        
        if nc is not None:
            # Constrained charge index
            target_all = list(set([i for sublist in target_lists for i in sublist]))
            
            # all index
            nottarget_list = jnp.array([natoms_list[i] for i in range(len(natoms_list)) if i not in target_all])
            nottarget_idx = jnp.array([i for i in range(len(natoms_list)) if i not in target_all])
            
            # Update charges for non-targeted atoms
            net_q = jnp.dot(ffparams['NonbondedForce']['charges'], natoms_list)
            ffparams['NonbondedForce']['charges'] = \
                ffparams['NonbondedForce']['charges'].at[nottarget_idx].set(ffparams['NonbondedForce']['charges'][nottarget_idx] - (net_q-nc) / nottarget_list.sum())
    elif nc is not None:
        net_q = jnp.dot(ffparams['NonbondedForce']['charges'], natoms_list)
        ffparams['NonbondedForce']['charges'] = ffparams['NonbondedForce']['charges'] - (net_q-nc) / natoms_list.sum()

    return ffparams

def get_charges_types(topdata: app.Topology, ff, gen_dmfftop=False):
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
    types = [a.meta['type'] for a in topdata.atoms()]
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
    for i_res in range(len(ff.ffinfo['Residues'])):
        for key in rescharges[i_res]:
            for key2 in rescharges[i_res][key]:
                res_idx = rescharges[i_res][key][key2]["index"]
                for i in res_idx:
                    ff.ffinfo['Residues'][i_res]["particles"][i]["charge"] = rescharges[i_res][key][key2]["value"]
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
    for i_res in range(len(ff.ffinfo['Residues'])):
        for i,_ in enumerate(ff.ffinfo['Residues'][i_res]["particles"]):
            ff.ffinfo['Residues'][i_res]["particles"][i]["charge"] = params["NonbondedForce"]["charges"][idx]
            idx += 1
    
    idx = 0
    for i_res in range(len(ff.ffinfo['Residues'])):
        for i,_ in enumerate(ff.ffinfo['Residues'][i_res]["vsites"]):
            n_weights = len([key for key in ff.ffinfo['Residues'][i_res]["vsites"][i].keys() \
                             if key.startswith("weight")])
            for i_weight in range(n_weights):
                ff.ffinfo['Residues'][i_res]["vsites"][i][f"weight{i_weight+1}"] = params["VsiteForce"]["weight"][idx]
                idx += 1

    return ff

def get_chgparams_from_rescharges(params, rescharges):
    natoms = np.array([len(rescharges[i_res][key][key2]["index"]) for i_res in range(len(rescharges)) for key in rescharges[i_res] for key2 in rescharges[i_res][key]]).sum()
    params["NonbondedForce"]["charges"] = jnp.zeros(natoms)
    ishift = 0
    for res in rescharges:
        natoms = 0
        for t in res:
            for key in res[t]:
                for idx in res[t][key]["index"]:
                    params["NonbondedForce"]["charges"] = params["NonbondedForce"]["charges"].at[idx+ishift].set(res[t][key]["value"])
                    natoms += 1
        ishift += natoms
    return params

def vsiteinfo_to_params(ff, params):
    """
    Convert vsite information from ff.ffinfo to params.
    """
    # vsiteinfo = ff.ffinfo["Residues"][0]["vsites"]
    weights = []
    for i_res in range(len(ff.ffinfo['Residues'])):
        for i_vs, _ in enumerate(ff.ffinfo['Residues'][i_res]["vsites"]):
            weights_tmp = [value for key, value in ff.ffinfo["Residues"][i_res]["vsites"][i_vs].items() if key.startswith("weight")]
            # weights_tmpのすべての要素をweightsに追加
            weights.extend(weights_tmp)

    params["VsiteForce"] = {}
    params["VsiteForce"]["weight"] = jnp.zeros(len(weights))
    params["VsiteForce"]["scale"] = jnp.zeros(len(weights))
    for i in range(len(weights)):
        params["VsiteForce"]["weight"] = params["VsiteForce"]["weight"].at[i].set(weights[i])

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
                    charge_forave.append(params["NonbondedForce"]["charges"][idx+ishift])
                res[t][key]["value"] = np.mean(charge_forave)
                # res[t][key]["value"] = params["NonbondedForce"]["charges"][idx+ishift]
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
    residues = ff.ffinfo['Residues']
    rescharges = []
    
    if ratio is not None:
        assert len(residues) == len(ratio), "len(residues) != len(ratio)"
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
                chargedict[t][f"{t}_0"] = {"value": c, "index":[i]}
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

def md_sample(initialpdb,
              ffxml,
              trajectory,
              rc,
              T,
              anneal_Tmax,
              anneal_steps,
              anneal_totalsteps,
              dt,
              nstxout,
              relax_steps,
              prod_steps,
              ensemble,
              useDispersionCorrection=False,
              useHbondConstraint=True,
              rigidWater=False):
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
        Dictionary containing sampling parameters such as temperature, annealing steps, etc.
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

    if useHbondConstraint:
        system = forcefield.createSystem(topology, 
                                        nonbondedMethod=app.PME,
                                        nonbondedCutoff=rc*unit.nanometer,
                                        constraints=app.HBonds,
                                        rigidWater=rigidWater)
    else:
        system = forcefield.createSystem(topology, 
                                        nonbondedMethod=app.PME,
                                        nonbondedCutoff=rc*unit.nanometer,
                                        rigidWater=rigidWater)
        
    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            if useDispersionCorrection:
                force.setUseDispersionCorrection(True)
            else:
                force.setUseDispersionCorrection(False)

    print(f"Using {ensemble} ensemble")
    if ensemble == "isonpt":
        print("Isotropic pressure control")
        system.addForce(openmm.MonteCarloBarostat(1.0*unit.bar, T*unit.kelvin))
    elif ensemble == "anisonpt":
        print("Anisotropic pressure control")
        system.addForce(openmm.MonteCarloAnisotropicBarostat([1.0*unit.bar] * 3, T*unit.kelvin))
    elif ensemble == "trinpt":
        system.addForce(openmm.MonteCarloFlexibleBarostat(1.0*unit.bar, T*unit.kelvin))
    
    integrator = openmm.LangevinIntegrator(T*unit.kelvin, 5/unit.picosecond, dt*unit.femtosecond)
    simulation = app.Simulation(topology, system, integrator)
    xtcfile = os.path.join("xtcfiles", trajectory)
    try:
        os.remove(xtcfile)
    except:
        pass
    simulation.context.setPositions(pos)
    print("== Energy minimization ==")
    simulation.minimizeEnergy()
    simulation.context.setVelocitiesToTemperature(T*unit.kelvin)

    simulation.reporters.append(app.StateDataReporter(sys.stdout, nstxout, density=True, step=True, remainingTime=True, speed=True,totalSteps=relax_steps + prod_steps))

    # relaxation run
    ## SA
    if anneal_totalsteps > 0:
        print("== Start Simulated Annealing ==")
        deltaT = (T-anneal_Tmax)/anneal_steps
        step_pertemp = int(anneal_totalsteps/anneal_steps)
        for i in range(anneal_steps):
            integrator.setTemperature((anneal_Tmax+deltaT*i)*unit.kelvin)
            simulation.step(step_pertemp)
    ## relax at desired temperature
    print("== Start Relaxation ==")
    integrator.setTemperature(T*unit.kelvin)
    simulation.step(relax_steps)
    # production run
    print("== Start Production ==")
    os.makedirs("xtcfiles", exist_ok=True)
    simulation.reporters.append(app.XTCReporter(xtcfile, nstxout))
    simulation.step(prod_steps)
    return xtcfile

def get_target_pred_frame(xtcfile, pdbfile, target_params: dict):
    """
    Get the predicted values for the targets from the parameters obtained by parser_dmffyaml().
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
                target_pred[target_name][key] = calc_rdf_frame(u, elem1, elem2, rmax=rcut12_A)
        elif target_name == "adf":
            target_pred[target_name] = {}
            for key in target_params[target_name].keys():
                elem1 = target_params[target_name][key]["elem1"]
                elem2 = target_params[target_name][key]["elem2"]
                elem3 = target_params[target_name][key]["elem3"]
                rcut12_A = target_params[target_name][key]["rcut12_A"]
                rcut23_A = target_params[target_name][key]["rcut23_A"]
                target_pred[target_name][key] = calc_adf_frame(xtcfile,
                                                               pdbfile,
                                                               elem1,
                                                               elem2,
                                                               elem3,
                                                               rcut12=rcut12_A,
                                                               rcut23=rcut23_A)
    return target_pred

def plot_compare(target_gt, target_pred_frame):
    num_plots = 0
    for key in target_gt.keys():
        if key in ["density_gcm3", "La_A", "Lb_A", "Lc_A"]:
            num_plots += 1
        elif key in ["rdf", "adf"]:
            for kind in target_gt[key].keys():
                num_plots += 1

    num_raw = np.max([math.ceil(num_plots/2),2])
    fig, ax = plt.subplots(num_raw, 2, figsize=(6.5,2.5*num_raw))
    i_plot = 0
    for key in target_gt.keys():
        if key in ["density_gcm3", "La_A", "Lb_A", "Lc_A"]:
            ax[i_plot//2, i_plot%2].set_title(key)
            x = ["GT", "FF"]
            y = [target_gt[key]["gt"], target_pred_frame[key].mean()]
            ax[i_plot//2, i_plot%2].bar(x,y, width=0.35)
            # barごとに値を表示
            for i, v in enumerate(y):
                ax[i_plot//2, i_plot%2].text(i, v + 0.01, str(round(v, 3)), ha='center', va='bottom')
            ax[i_plot//2, i_plot%2].set_ylim(0, y[0]*1.2)
            i_plot += 1
        elif key in ["rdf", "adf"]:
            for kind in target_gt[key].keys():
                ax[i_plot//2, i_plot%2].set_title(key+"_"+kind)
                spectra = np.mean(target_pred_frame[key][kind], axis=0)
                ax[i_plot//2, i_plot%2].plot(spectra, alpha=0.5)
                ax[i_plot//2, i_plot%2].plot(target_gt[key][kind]["gt"], label="gt")
                i_plot += 1

    plt.tight_layout()
    fig.savefig('sample.png')
    plt.close(fig)

class saver_wresults:
    def __init__(self, target_gt):
        self.results_dict = {"epoch":[], "loss": []} 
        self.target_gt = target_gt
        for key in target_gt.keys():
            if key in ["density_gcm3", "La_A", "Lb_A", "Lc_A"]:
                self.results_dict[key] = []
            elif key in ["rdf", "adf"]:
                self.results_dict[key] = {}
                for k in target_gt[key].keys():
                    self.results_dict[key][k] = []
    
    def append(self, i_epoch, loss, wresults):
        if jnp.isnan(loss) == False:
            self.results_dict["loss"].append(float(loss))
            self.results_dict["epoch"].append(i_epoch)
            for key in self.target_gt.keys():
                if key in ["density_gcm3", "La_A", "Lb_A", "Lc_A"]:
                    self.results_dict[key].append(float(wresults[key]))
                elif key in ["rdf", "adf"]:
                    for k in self.target_gt[key].keys():
                        self.results_dict[key][k].append(np.array(wresults[key][k]).tolist())
    
    def save(self, basename="results"):
        # self.results_dictから深さが1の要素を抽出
        shallow_dict = {k: v for k, v in self.results_dict.items() if isinstance(v, (int, float, list))}
        # # 深さが2以上の要素を抽出
        # deep_dict = {k: v for k, v in self.results_dict.items() if isinstance(v, dict)}
        with open(basename+".yml", 'w') as file:
            yaml.dump(shallow_dict, file)
        with open(basename+".pkl", "wb") as f:
            pickle.dump(self.results_dict, f)
    
    def plot_learningcurve(self, filename="learning_curve.png"):
        fig, ax = plt.subplots(1, 1, figsize=(3.25,2.5))
        ax.plot(self.results_dict["epoch"], self.results_dict["loss"])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        plt.tight_layout()
        fig.savefig(filename)
        plt.close(fig)
 
