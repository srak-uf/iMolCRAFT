#!/usr/bin/env python
import jax
import jax.numpy as jnp
from jax import value_and_grad, jit
 
import openmm
from openmm import app
import openmm.unit as unit
from openmm.app.pdbfile import PDBFile

from dmff import Hamiltonian
from dmff.mbar import MBAREstimator, TargetState, Sample, OpenMMSampleState, buildTrajEnergyFunction
from dmff.optimize import MultiTransform, genOptimizer

import mdtraj as md

import copy
import matplotlib.pyplot as plt
import numpy as np
import os, json, shutil, sys
import optax

from dmff_utils import *
from analyzer   import *

try:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-lr", nargs=3)

    args = parser.parse_args()
    lr = args.lr
    # float
    lr = [float(i) for i in lr]
except:
    lr = [0.0005, 0.0005, 0.0010, 0.0001, 0.0001]


def plot_loss(epoch_list, loss_list):
    plt.plot(epoch_list, loss_list)
    plt.xlabel("steps")
    plt.ylabel("loss")
    plt.tight_layout()
    plt.savefig("loss.png")
    plt.close()


def Loss(params, density, rdf_1, rdf_2, rdf_3, Lx, Lz):
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

    rdf_pert3 = (rdf_3 * weight.reshape((-1, 1))).sum(axis=0)
    loss_rdf3 = jnp.log(jnp.power(rdf_pert3 - ref_adf1, 2).mean())

    loss_rdf = loss_rdf1 + loss_rdf2 + loss_rdf3*10

    return loss_den*0.5 + loss_rdf + loss_Lx +  loss_Lz, (density, utarget)


seed = 0
init_stru = "merged_supercell_bonds.pdb"
rc = 1.2
T = 233.15
anneal_steps = 1
anneal_Tmax = 400.0
anneal_totaltime = 0 * 1000
relax_step = 20 * 1000
prod_step = 100 * 1000 # 100 * 1000
nstxout = 1000
n_epochs = 400
ffxml_list = ["Li.xml", "FSA.xml", "SN.xml"]
res_ratio = [1, 1, 2] # stoichiometric ratio of residues of ffxml_list
ensemble = "npt" # nvt or npt
dt = 1.0 # fs
ref_density = 1.572 # 1.568 experiment Lx = 36.282 Lz = 40.2152   1.55205 300k matlantis 
ref_Lx = 36.282 / 10
ref_Lz = 40.2152 / 10
ref_rdfLiO = "reference_rdf_Li_O.txt"
ref_rdfLiN = "reference_rdf_Li_N.txt"
ref_adfCNLi  = "reference_adf_CNLi.txt"
state_name = 'lipf6dmc'


merge_xml(ffxml_list, "epoch-0.xml")
parameter = "xmlfiles/epoch-0.xml"
initial_xml = os.path.join("xmlfiles", "epoch-0.xml")
ff = Hamiltonian(initial_xml) # "gaffxml_0.xml"
pdb = PDBFile(init_stru)
pots = ff.createPotential(pdb.topology, nonbondedMethod=app.PME, nonbondedCutoff=rc*unit.nanometer)
cov_map = pots.meta['cov_map']
params = ff.getParameters().parameters
pos = jnp.array(pdb.getPositions()._value)
box = jnp.array(pdb.topology.getPeriodicBoxVectors()._value)
efunc = jit(pots.getPotentialFunc())

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

    simulation.reporters.append(app.StateDataReporter(sys.stdout, nstxout, density=True, step=True, remainingTime=True, speed=True,totalSteps=relax_step + prod_step))

    # relaxation run
    ## SA
    deltaT = (T-anneal_Tmax)/anneal_steps
    step_pertemp = int(anneal_totaltime/anneal_steps)
    for i in range(anneal_steps):
        integrator.setTemperature((anneal_Tmax+deltaT*i)*unit.kelvin)
        simulation.step(step_pertemp)
    ## relax at desired temperature
    integrator.setTemperature(T*unit.kelvin)
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
cnli_ref = np.loadtxt(ref_adfCNLi)
ref_rdf1 = lio_ref.T[1]
ref_rdf2 = lin_ref.T[1]
ref_adf1 = cnli_ref.T[1]
print(md.density(traj).mean()/1000, ref_density)

g_LiO_frame = calc_rdf_frame(f"xtcfiles/loop-0.xtc", init_stru, "Li", "O")
g_LiN_frame = calc_rdf_frame(f"xtcfiles/loop-0.xtc", init_stru, "Li", "N")
g_CNLi_frame = calc_adf_frame(f"xtcfiles/loop-0.xtc", init_stru, "C", "N", "Li", rcut12=1.4, rcut23=2.9)
r_LiN,g_LiN = calc_rdf(f"xtcfiles/loop-0.xtc", init_stru, "Li", "N")
r_LiO,g_LiO = calc_rdf(f"xtcfiles/loop-0.xtc", init_stru, "Li", "O")
deg_CNLi,g_CNLi = calc_adf(f"xtcfiles/loop-0.xtc", init_stru, "C", "N", "Li", rcut12=1.4, rcut23=2.9)


multiTrans = MultiTransform(params)
multiTrans["NonbondedForce/sigma"]   = genOptimizer(learning_rate=lr[0], clip=0.001,nonzero=False)
multiTrans["NonbondedForce/epsilon"] = genOptimizer(learning_rate=lr[1], clip=0.001,nonzero=False)
multiTrans["NonbondedForce/charges"] = genOptimizer(learning_rate=lr[2], clip=0.001,nonzero=False)
multiTrans.finalize()
grad_transform = optax.multi_transform(multiTrans.transforms, multiTrans.labels)
mask = jax.tree_util.tree_map(lambda x: x.dtype != jnp.int32 and x.dtype != int, params)
grad_transform = optax.masked(grad_transform, mask)
opt_state = grad_transform.init(params)


results_dict = {"epoch":[], "loss": [], "density": []}
for i_epoch in range(1, n_epochs+1):
    density_frame = md.density(traj) / 1000 # density (g cm-3) by frame
    Lx_frame = traj.unitcell_lengths.T[0]
    Lz_frame = traj.unitcell_lengths.T[2]
    (loss, (density, utarget)), gradient = \
            value_and_grad(Loss, argnums=(0), has_aux=True)(params,
                                                            density_frame,
                                                            g_LiO_frame,
                                                            g_LiN_frame,
                                                            g_CNLi_frame,
                                                            Lx_frame,
                                                            Lz_frame)

    if jnp.isnan(loss) == False:
        print("Loss:", loss)
        results_dict["loss"].append(float(loss))
        results_dict["epoch"].append(i_epoch-1)
        results_dict["density"].append(float(density))
        plot_loss(results_dict["epoch"], results_dict["loss"])
        if os.path.isfile(f"xmlfiles/epoch-{i_epoch-1}.xml") == False:
            # "xmlfiles/epoch-*.xmlのうち、引数が最新のものをコピーして次のエポックのxmlファイルとする"
            for i in range(i_epoch, 0, -1):
                if os.path.isfile(f"xmlfiles/epoch-{i-2}.xml"):
                    shutil.copy(f"xmlfiles/epoch-{i-2}.xml", f"xmlfiles/epoch-{i_epoch-1}.xml")
                    break


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
                if v < 30 and k != "Total":
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
                                  nonbondedMethod=app.PME, nonbondedCutoff=rc*unit.nanometer)
        sample = Sample(traj, f"loop-{i_epoch}")
        estimator.add_state(state)
        estimator.add_sample(sample)
        # estimator need to be reconverged whenenver new samples or states are added
        estimator.optimize_mbar()
        
        g_LiO_frame = calc_rdf_frame(f"xtcfiles/loop-{i_epoch}.xtc", init_stru, "Li", "O")
        g_LiN_frame = calc_rdf_frame(f"xtcfiles/loop-{i_epoch}.xtc", init_stru, "Li", "N")
        g_CNLi_frame = calc_adf_frame(f"xtcfiles/loop-{i_epoch}.xtc", init_stru, "C", "N", "Li", rcut12=1.4, rcut23=2.9)

        fig, ax = plt.subplots(1, 3, figsize=(7.5,5))
        ax[0].set_xticks(np.arange(0, 8, 2))
        ax[1].set_xticks(np.arange(0, 8, 2))
        ax[2].set_xticks(np.arange(0, 181, 60))

        ax[0].plot(r_LiN, np.mean(g_LiN_frame, axis=0))
        ax[0].plot(lin_ref.T[0], lin_ref.T[1], linestyle="--")
        ax[0].set_xlabel("r (nm)")
        ax[0].set_ylabel("g(r)")

        ax[1].plot(r_LiO, np.mean(g_LiO_frame, axis=0))
        ax[1].plot(lio_ref.T[0], lio_ref.T[1], linestyle="--")
        ax[1].set_xlabel("r (nm)")
        ax[1].set_ylabel("g(r)")

        ax[2].plot(deg_CNLi, np.mean(g_CNLi_frame, axis=0))
        ax[2].plot(cnli_ref.T[0], cnli_ref.T[1], linestyle="--")
        ax[2].set_xlabel("Angle (deg)")
        ax[2].set_ylabel("Prob")

        plt.tight_layout()
        fig.savefig('sample.png')
        plt.close(fig)
    
    with open("results.json", "w") as f:
        json.dump(results_dict, f)


