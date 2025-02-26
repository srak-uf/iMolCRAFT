#!/usr/bin/env python
import jax
import jax.numpy as jnp
from jax import value_and_grad, jit, vmap
from jax import jit, value_and_grad, vmap, pmap, grad, random, tree_util
 
import openmm
from openmm import app
import openmm.unit as unit
from openmm.app.pdbfile import PDBFile

from dmff import Hamiltonian, NeighborList
from dmff.api.xmlio import XMLIO
from dmff.api.paramset import ParamSet
from dmff.operators.templatetype import TemplateATypeOperator
import dmff
from dmff.generators.classical import PeriodicTorsionGenerator, NonbondedGenerator
from dmff.mbar import MBAREstimator, TargetState, Sample, OpenMMSampleState, buildTrajEnergyFunction
from dmff.optimize import MultiTransform, genOptimizer

import MDAnalysis
import MDAnalysis.analysis.rdf as mda
import mdtraj as md

import copy
import matplotlib.pyplot as plt
import numpy as np
import sys
import os, json, shutil
from tqdm import tqdm
import optax



try:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-lr", nargs=3)

    args = parser.parse_args()
    lr = args.lr
    # float
    lr = [float(i) for i in lr]
except:
    lr = [0.001, 0.001, 0.001]


seed = 0
init_stru = "merged_supercell_bonds.pdb"
rc = 1.2
T = 300.0
relax_step = 20 * 1000
prod_step = 100 * 1000 # 100 * 1000
nstxout = 1000
n_epochs = 400
ffxml_list = ["gaffxml_0_best.xml", "gaffxml_1.xml", "ionsff99_tip3p.xml"]
res_ratio = [1, 2, 1] # stoichiometric ratio of residues of ffxml_list
ensemble = "npt" # nvt or npt
dt = 1.0 # fs
ref_density = 1362.5137 / 1000 # exp
ref_Lx = 40.456 / 10   # nm
ref_Lz = 32.952 / 10   # nm
ref_rdfLiN = "matlantis_LiN.txt"
ref_rdfLiO = "matlantis_LiO.txt"
state_name = 'c2c2'


def merge_xml(ffxml_list, outxml):
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


def calc_rdf(xtcfile, pdbfile, elem1, elem2, rmax=8.0, dr=0.01):
    u = MDAnalysis.Universe(pdbfile, xtcfile)
    u_select1 = u.select_atoms(f"element {elem1}")
    u_select2 = u.select_atoms(f"element {elem2}")
    rdf = mda.InterRDF(u_select1, u_select2, range=(0,rmax), nbins=int(rmax/dr))
    rdf.run()
    r = rdf.results.bins
    g = rdf.results.rdf
    return r, g



def calc_rdf_frame(xtcfile, pdbfile, elem1, elem2, rmax=8.0, dr=0.01):
    u = MDAnalysis.Universe(pdbfile, xtcfile)
    u_select1 = u.select_atoms(f"element {elem1}")
    u_select2 = u.select_atoms(f"element {elem2}")
    rdf = mda.InterRDF(u_select1, u_select2, range=(0,rmax), nbins=int(rmax/dr))
    rdf_list = []    
    for i_frame in range(len(u.trajectory)):
        rdf.run(frames=[i_frame])
        g = rdf.results.rdf
        rdf_list.append(g)
    return np.array(rdf_list)
    

# OpenMM sampler, for resampling during optimization
def npt_sample(initialpdb, ffxml, trajectory):
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

    Returns
    -------
    None    
    """
    pdb = app.PDBFile(initialpdb)
    forcefield = app.ForceField(ffxml)
    system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.PME, nonbondedCutoff=rc*unit.nanometer, constraints=app.HBonds)
    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            force.setUseDispersionCorrection(False)
    # system.addForce(openmm.MonteCarloBarostat(1.0*unit.bar, T*unit.kelvin, 20))
    system.addForce(openmm.MonteCarloAnisotropicBarostat([1.0*unit.bar] * 3, T*unit.kelvin))
    
    integrator = openmm.LangevinIntegrator(T*unit.kelvin, 5/unit.picosecond, dt*unit.femtosecond)
    simulation = app.Simulation(pdb.topology, system, integrator)
    
    try:
        os.remove(trajectory)
    except:
        pass
    simulation.context.setPositions(pdb.getPositions())
    simulation.minimizeEnergy()
    simulation.context.setVelocitiesToTemperature(T*unit.kelvin)

    simulation.reporters.append(app.StateDataReporter(sys.stdout, 1000, density=True, step=True, remainingTime=True, speed=True,totalSteps=relax_step + prod_step))

    # relaxation run
    simulation.step(relax_step)
    # production run
    os.makedirs("xtcfiles", exist_ok=True)
    simulation.reporters.append(app.XTCReporter(f"xtcfiles/{trajectory}",nstxout))
    simulation.step(prod_step)


    u = md.load_xtc(f"xtcfiles/{trajectory}", top = initialpdb)
    positions = jnp.array(u.xyz)
    state_init = {}
    state_init['pos'] = positions
    # key, subkey = random.split(key)
    return state_init #, key


def neutralize(params, num_elems):
    net_q = jnp.dot(params['NonbondedForce']['charges'], num_elems)
    params['NonbondedForce']['charges'] = params['NonbondedForce']['charges'] - net_q / num_elems.sum()
    return params


def plot_loss(epoch_list, loss_list):
    plt.plot(epoch_list, loss_list)
    plt.xlabel("steps")
    plt.ylabel("loss")
    plt.tight_layout()
    plt.savefig("loss.png")
    plt.close()

def get_charges_types(topdata: app.Topology, ff, gen_dmfftop=False):
    template = TemplateATypeOperator(ff.ffinfo)
    topdata = dmff.api.DMFFTopology(from_top=topdata)
    topdata = template(topdata)
    charges = [a.meta["charge"] for a in topdata.atoms()]
    types = [a.meta['type'] for a in topdata.atoms()]
    if gen_dmfftop:
        return charges, types, topdata
    else:
        return charges, types


merge_xml(ffxml_list, "epoch-0.xml")
initial_xml = os.path.join("xmlfiles","epoch-0.xml")
ff = Hamiltonian(initial_xml) # "gaffxml_0.xml"
pdb = PDBFile(init_stru)
pots = ff.createPotential(pdb.topology, nonbondedMethod=app.PME, nonbondedCutoff=rc*unit.nanometer)
cov_map = pots.meta['cov_map']
params = ff.getParameters().parameters
pos = jnp.array(pdb.getPositions()._value)
box = jnp.array(pdb.topology.getPeriodicBoxVectors()._value)
efunc = jit(pots.getPotentialFunc())


def get_rescharges_from_residues(ff, ratio=None):
    residues = ff.ffinfo['Residues']
    rescharges = []
    
    if ratio is not None:
        assert len(residues) == len(ratio), "len(residues) != len(ratio)"
        num_elems = []

    for i_res in range(len(residues)):
        residue = residues[i_res]
        chargedict = {}
        for i, p in enumerate(residue["particles"]):
            t = p["type"]
            c = p["charge"]
            if ratio is not None:
                num_elems.append(ratio[i_res])

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
        return rescharges, jnp.array(num_elems)
    else:
        return rescharges

def update_ffinfo_from_rescharges(ff, rescharges):
    for i_res in range(len(ff.ffinfo['Residues'])):
        for key in rescharges[i_res]:
            for key2 in rescharges[i_res][key]:
                res_idx = rescharges[i_res][key][key2]["index"]
                for i in res_idx:
                    ff.ffinfo['Residues'][i_res]["particles"][i]["charge"] = rescharges[i_res][key][key2]["value"]
    return ff

def update_ffinfo_from_params(ff, params):
    idx = 0
    for i_res in range(len(ff.ffinfo['Residues'])):
        for i,_ in enumerate(ff.ffinfo['Residues'][i_res]["particles"]):
            ff.ffinfo['Residues'][i_res]["particles"][i]["charge"] = params["NonbondedForce"]["charges"][idx]
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




def Loss(params, density, rdf_1, rdf_2, Lx, Lz):
    ff_d = update_ffinfo_from_params(ff, params)
    params_wo_charge = copy.deepcopy(params)
    del params_wo_charge["NonbondedForce"]["charges"]

    pots = ff_d.createPotential(pdb.topology, nonbondedMethod=app.PME, nonbondedCutoff=rc*unit.nanometer)
    efunc = jit(pots.getPotentialFunc())

    target_energy_function = buildTrajEnergyFunction(efunc, cov_map, rc, ensemble=ensemble, useFreud=True)
    target_state = TargetState(T, target_energy_function)
    weight, utarget = estimator.estimate_weight(target_state, parameters=params_wo_charge)

    density = jnp.average(density, weights=weight)
    loss_den = (density - ref_density)**2

    Lx = jnp.average(Lx, weights=weight)
    Lz = jnp.average(Lz, weights=weight)
    loss_Lx = (Lx - ref_Lx)**2
    loss_Lz = (Lz - ref_Lz)**2

    print(f"Den: {density.item()}, Lx: {Lx.item()}, Lz: {Lz.item()}")
    print(f"Reference Den: {ref_density}, Lx: {ref_Lx}, Lz: {ref_Lz}")

    rdf_pert1 = (rdf_1 * weight.reshape((-1, 1))).sum(axis=0)
    loss_rdf1 = jnp.log(jnp.power(rdf_pert1 - ref_rdf1, 2).mean())

    rdf_pert2 = (rdf_2 * weight.reshape((-1, 1))).sum(axis=0)
    loss_rdf2 = jnp.log(jnp.power(rdf_pert2 - ref_rdf2, 2).mean())
    loss_rdf = loss_rdf1 + loss_rdf2

    return loss_den*0.5 + loss_rdf + loss_Lx +  loss_Lz, (density.item(), Lx.item(), Lz.item(), utarget)


charges, types, topdata = get_charges_types(pdb.topology, ff, gen_dmfftop=True)
rescharges, num_elems = get_rescharges_from_residues(ff, ratio=res_ratio)
params = get_chgparams_from_rescharges(params, rescharges)
rescharges = update_rescharges_from_params(rescharges, params)
state_init = npt_sample(init_stru, initial_xml, "loop-0.xtc")

estimator = MBAREstimator()
state = OpenMMSampleState(state_name, initial_xml, init_stru, temperature=T, pressure=1.0,
                            nonbondedMethod=app.PME, nonbondedCutoff=rc*unit.nanometer)
traj = md.load('xtcfiles/loop-0.xtc', top=init_stru)
sample = Sample(traj, state_name)
estimator.add_state(state)
estimator.add_sample(sample)
estimator.optimize_mbar()



lin_ref = np.loadtxt(ref_rdfLiN)
lio_ref = np.loadtxt(ref_rdfLiO)
ref_rdf1 = lio_ref.T[1]
ref_rdf2 = lin_ref.T[1]
print(md.density(traj).mean()/1000, ref_density)
print(traj.unitcell_lengths.T[0].mean(), ref_Lx)
print(traj.unitcell_lengths.T[2].mean(), ref_Lz)

g_LiO_frame = calc_rdf_frame(f"xtcfiles/loop-0.xtc", init_stru, "Li", "O")
g_LiN_frame = calc_rdf_frame(f"xtcfiles/loop-0.xtc", init_stru, "Li", "N")
r_LiN,g_LiN = calc_rdf(f"xtcfiles/loop-0.xtc", init_stru, "Li", "N")
r_LiO,g_LiO = calc_rdf(f"xtcfiles/loop-0.xtc", init_stru, "Li", "O")


multiTrans = MultiTransform(params)
multiTrans["NonbondedForce/sigma"]   = genOptimizer(learning_rate=lr[0], clip=0.001,nonzero=False)
multiTrans["NonbondedForce/epsilon"] = genOptimizer(learning_rate=lr[1], clip=0.001,nonzero=False)
multiTrans["NonbondedForce/charges"] = genOptimizer(learning_rate=lr[2], clip=0.001,nonzero=False)
multiTrans.finalize()
grad_transform = optax.multi_transform(multiTrans.transforms, multiTrans.labels)
mask = jax.tree_util.tree_map(lambda x: x.dtype != jnp.int32 and x.dtype != int, params)
grad_transform = optax.masked(grad_transform, mask)
opt_state = grad_transform.init(params)


results_dict = {"epoch":[], "loss": [], "density": [], "Lx": [], "Lz": []}
for i_epoch in range(1, n_epochs+1):
    density_frame = md.density(traj) / 1000 # density (g cm-3) by frame
    Lx_frame = traj.unitcell_lengths.T[0]
    Lz_frame = traj.unitcell_lengths.T[2]
    (loss, (density, lx, lz, utarget)), gradient = value_and_grad(Loss, argnums=(0), has_aux=True)(params,
                                                                                                density_frame,
                                                                                                g_LiO_frame,
                                                                                                g_LiN_frame,
                                                                                                Lx_frame,
                                                                                                Lz_frame)

    if jnp.isnan(loss) == False:
        print("Loss:", loss)
        results_dict["loss"].append(float(loss))
        results_dict["epoch"].append(i_epoch-1)
        results_dict["density"].append(float(density))
        results_dict["Lx"].append(float(lx))
        results_dict["Lz"].append(float(lz))
        plot_loss(results_dict["epoch"], results_dict["loss"])
        if os.path.isfile(f"xmlfiles/epoch-{i_epoch-1}.xml") == False:
            shutil.copy(f"xmlfiles/epoch-{i_epoch-2}.xml", f"xmlfiles/epoch-{i_epoch-1}.xml")

        updates, opt_state = grad_transform.update(gradient, opt_state)
        params = optax.apply_updates(params, updates)
        
        params = neutralize(params, num_elems)
        rescharges = update_rescharges_from_params(rescharges, params)
        ff = update_ffinfo_from_rescharges(ff, rescharges)
        params_wo_charge = copy.deepcopy(params)
        del params_wo_charge["NonbondedForce"]["charges"]

        ff.getParameters().parameters = params_wo_charge
        os.makedirs("xmlfiles", exist_ok=True)
        ff.renderXML(f"xmlfiles/epoch-{i_epoch}.xml")
        parameter = f'xmlfiles/epoch-{i_epoch}.xml'

        # checkout the effective size of each sample in the estimator
        print('Effective sample sizes:')
        try:
            ieff = estimator.estimate_effective_sample(utarget, decompose=True)
            for k, v in ieff.items():
                print(f'{k}: {v}')
            
            for k, v in ieff.items():
                if v < 5 and k != "Total":
                    estimator.remove_state(k)
        except:
            estimator.states = []
            estimator.samples = []

        nan_flag = False
    else:
        nan_flag = True
        estimator.states = []
        estimator.samples = []
    
    if len(estimator.states) < 1 or nan_flag == True:
        print("Add", f"loop-{i_epoch}")
        # get new sample using the current state
        state_init = npt_sample(init_stru, parameter, f"loop-{i_epoch}.xtc")
        traj = md.load(f"xtcfiles/loop-{i_epoch}.xtc", top=init_stru)
        state = OpenMMSampleState(f"loop-{i_epoch}", parameter, init_stru, temperature=T, pressure=1.0,
                                  nonbondedMethod=app.CutoffPeriodic, nonbondedCutoff=rc*unit.nanometer)
        sample = Sample(traj, f"loop-{i_epoch}")
        estimator.add_state(state)
        estimator.add_sample(sample)
        # estimator need to be reconverged whenenver new samples or states are added
        estimator.optimize_mbar()
        
        g_LiO_frame = calc_rdf_frame(f"xtcfiles/loop-{i_epoch}.xtc", init_stru, "Li", "O")
        g_LiN_frame = calc_rdf_frame(f"xtcfiles/loop-{i_epoch}.xtc", init_stru, "Li", "N")

        fig, ax = plt.subplots(1, 2, figsize=(5,2.5))
        ax[0].set_xticks(np.arange(0, 8, 2))
        ax[1].set_xticks(np.arange(0, 8, 2))

        ax[0].plot(r_LiN, np.mean(g_LiN_frame, axis=0))
        ax[0].plot(lin_ref.T[0], lin_ref.T[1], linestyle="--")
        ax[0].set_xlabel("r (nm)")
        ax[0].set_ylabel("g(r)")

        ax[1].plot(r_LiO, np.mean(g_LiO_frame, axis=0))
        ax[1].plot(lio_ref.T[0], lio_ref.T[1], linestyle="--")
        ax[1].set_xlabel("r (nm)")
        ax[1].set_ylabel("g(r)")

        plt.tight_layout()
        fig.savefig('sample.png')
        plt.close(fig)
    
    with open("results.json", "w") as f:
        json.dump(results_dict, f)


