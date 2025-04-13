#!/usr/bin/env python
import jax
import jax.numpy as jnp
from openmm import app
import openmm.unit as unit
from openmm.app.pdbfile import PDBFile
from dmff import Hamiltonian
from dmff.mbar import MBAREstimator, Sample, OpenMMSampleState
from dmff.optimize import MultiTransform, genOptimizer
import optax
import mdtraj as md
import os, shutil, copy
from imolcryff.dmff_utils import *
from imolcryff.analyzer   import *

dmff_params = parser_dmffyaml("dmff.yml")
init_stru = dmff_params["sampling"]["init_structure"]
T_K = float(dmff_params["sampling"]["temperature_K"])
P_bar = float(dmff_params["sampling"]["pressure_bar"])
ensemble = dmff_params["sampling"]["ensemble"]
ffxml_list = dmff_params["forcefield"]["xml_list"]
res_ratio = dmff_params["forcefield"]["residue_ratio"]
rc = float(dmff_params["forcefield"]["rcut_nm"])
n_epochs = int(dmff_params["opt_scheme"]["n_epochs"])
lr = dmff_params["opt_scheme"]["lr"]
neff = int(dmff_params["opt_scheme"]["neff"])

ffxml = merge_xml(ffxml_list, "epoch-0.xml")
initial_xml = os.path.join("xmlfiles", "epoch-0.xml")
ff = Hamiltonian(initial_xml)
pdb = PDBFile(init_stru)
pots = ff.createPotential(pdb.topology, nonbondedMethod=app.PME, nonbondedCutoff=rc*unit.nanometer)
cov_map = pots.meta['cov_map']
ffparams = ff.getParameters().parameters
efunc = jax.jit(pots.getPotentialFunc())

charges, types, topdata = get_charges_types(pdb.topology, ff, gen_dmfftop=True)
rescharges, num_elems = get_rescharges_from_residues(ff, ratio=res_ratio)
ffparams = get_chgparams_from_rescharges(ffparams, rescharges)
rescharges = update_rescharges_from_params(rescharges, ffparams)
state_init = md_sample(init_stru, initial_xml, "loop-0.xtc", dmff_params["sampling"], dmff_params["forcefield"])

state_name = "loop-0"
estimator = MBAREstimator()
state = OpenMMSampleState(state_name, initial_xml, init_stru, temperature=T_K, pressure=P_bar,
                          nonbondedMethod=app.PME, nonbondedCutoff=rc*unit.nanometer)
traj = md.load('xtcfiles/loop-0.xtc', top=init_stru)
sample = Sample(traj, state_name)
estimator.add_state(state)
estimator.add_sample(sample)
estimator.optimize_mbar()


multiTrans = MultiTransform(ffparams)
if isinstance(lr, list) == True:
    multiTrans["NonbondedForce/sigma"]   = genOptimizer(learning_rate=lr[0], clip=0.001,nonzero=False)
    multiTrans["NonbondedForce/epsilon"] = genOptimizer(learning_rate=lr[1], clip=0.001,nonzero=False)
    multiTrans["NonbondedForce/charges"] = genOptimizer(learning_rate=lr[2], clip=0.001,nonzero=False)
else:
    multiTrans["NonbondedForce/sigma"]   = genOptimizer(learning_rate=lr, clip=0.001,nonzero=False)
    multiTrans["NonbondedForce/epsilon"] = genOptimizer(learning_rate=lr, clip=0.001,nonzero=False)
    multiTrans["NonbondedForce/charges"] = genOptimizer(learning_rate=lr, clip=0.001,nonzero=False)

multiTrans.finalize()
grad_transform = optax.multi_transform(multiTrans.transforms, multiTrans.labels)
mask = jax.tree_util.tree_map(lambda x: x.dtype != jnp.int32 and x.dtype != int, ffparams)
grad_transform = optax.masked(grad_transform, mask)
opt_state = grad_transform.init(ffparams)

target_gt = get_target_gt(dmff_params)
target_pred_frame = get_target_pred_frame("xtcfiles/loop-0.xtc", init_stru, dmff_params)
saver = saver_wresults(target_gt)
for i_epoch in range(1, n_epochs+1):
    (loss, (utarget, wresults)), gradient = get_loss_autograd(
                                            ffparams,
                                            ff,
                                            pdb.topology,
                                            cov_map,
                                            rc,
                                            ensemble,
                                            T_K,
                                            estimator,
                                            target_gt,
                                            target_pred_frame,
                                            pressure=P_bar
                                            )

    if jnp.isnan(loss) == False:
        nan_flag = False
        print("Loss:", loss)
        saver.append(i_epoch-1, float(loss), wresults)
        if os.path.isfile(f"xmlfiles/epoch-{i_epoch-1}.xml") == False:
            for i in range(i_epoch, 0, -1): # "xmlfiles/epoch-*.xmlのうち、引数が最新のものをコピー
                if os.path.isfile(f"xmlfiles/epoch-{i-2}.xml"):
                    shutil.copy(f"xmlfiles/epoch-{i-2}.xml", f"xmlfiles/epoch-{i_epoch-1}.xml")
                    break

        updates, opt_state = grad_transform.update(gradient, opt_state)
        ffparams = optax.apply_updates(ffparams, updates)
        ffparams = neutralize(ffparams, num_elems)
        rescharges = update_rescharges_from_params(rescharges, ffparams)
        ff = update_ffinfo_from_rescharges(ff, rescharges)
        params_wo_charge = copy.deepcopy(ffparams)
        del params_wo_charge["NonbondedForce"]["charges"]

        ff.getParameters().parameters = params_wo_charge
        os.makedirs("xmlfiles", exist_ok=True)
        ff.renderXML(f"xmlfiles/epoch-{i_epoch}.xml")
        ffxml = f'xmlfiles/epoch-{i_epoch}.xml'

        # checkout the effective size of each sample in the estimator
        print('Effective sample sizes:')
        try:
            ieff = estimator.estimate_effective_sample(utarget, decompose=True)
            for k, v in ieff.items():
                print(f'{k}: {v}')
            for k, v in ieff.items():
                if v < neff and k != "Total":
                    estimator.remove_state(k)
        except:
            estimator.states = []
            estimator.samples = []
    else:
        nan_flag = True
        estimator.states = []
        estimator.samples = []
    
    if len(estimator.states) < 1 or nan_flag == True:
        print("Add", f"loop-{i_epoch}")
        # get new sample using the current state
        state_init = md_sample(init_stru,
                               ffxml,
                               f"loop-{i_epoch}.xtc",
                               dmff_params["sampling"],
                               dmff_params["forcefield"]
                               )
        traj = md.load(f"xtcfiles/loop-{i_epoch}.xtc", top=init_stru)
        state = OpenMMSampleState(f"loop-{i_epoch}", ffxml, init_stru, temperature=T_K, pressure=P_bar,
                                  nonbondedMethod=app.PME, nonbondedCutoff=rc*unit.nanometer)
        sample = Sample(traj, f"loop-{i_epoch}")
        estimator.add_state(state)
        estimator.add_sample(sample)
        estimator.optimize_mbar()
        
        target_pred_frame = get_target_pred_frame(f"xtcfiles/loop-{i_epoch}.xtc", init_stru, dmff_params)
        plot_compare(target_gt, target_pred_frame)
    
    saver.save(basename="results")
    saver.plot_learningcurve()
