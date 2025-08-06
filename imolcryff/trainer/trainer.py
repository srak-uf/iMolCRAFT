from dmff import Hamiltonian, DMFFTopology
from dmff.common.nblist import NoCutoffNeighborList
from dmff.generators.classical import PeriodicTorsionGenerator
from dmff.api.paramset import ParamSet
from dmff.api.xmlio import XMLIO
from dmff.api.hamiltonian import Potential
from dmff.optimize import MultiTransform, genOptimizer
from openmm import app
import copy, shutil
import numpy as np
import jax
import jax.numpy as jnp
import os
from jax import value_and_grad, jit
import optax
from tqdm import tqdm
from typing import NamedTuple, Callable
from ..calculator import DihedCalculator
from ..trainer.dmff_utils import merge_xml, neutralize, update_rescharges_from_params, update_ffinfo_from_rescharges, get_chgparams_from_rescharges, get_rescharges_from_residues, update_ffinfo_from_params


class DistanceTrainState(NamedTuple):
    ffparams: dict
    ff: Hamiltonian
    opt_state: dict
    loss: float = 0.0

class _DistanceTrainer(NamedTuple):
    init_step: Callable
    train_step: Callable

def make_distance_trainer(
        optimizer: optax.GradientTransformation,
) -> _DistanceTrainer:

    def init_step(ffparams, ff):
        opt_state = optimizer.init(ffparams)
        return DistanceTrainState(ffparams=ffparams,
                                  ff=ff,
                                  opt_state=opt_state)

    def train_step(train_state, inputs, gt_scans, natoms_list):
        inputs = inputs
        gt_scans = gt_scans
        ffparams = train_state.ffparams

        def add_grad(grad1:ParamSet, grad2:ParamSet):
            for k in grad1.parameters.keys():
                for t in grad1.parameters[k].keys():
                    grad1.parameters[k][t] += grad2.parameters[k][t]
            return ParamSet(grad1.parameters, grad1.mask)
        
        def compute_loss(ffparams: dict, 
                         topology: app.Topology | DMFFTopology, 
                         positions, pairs, y_gt):
            ff_d = update_ffinfo_from_params(ff, ffparams)
            ffparams_wo_charge = copy.deepcopy(ffparams)
            del ffparams_wo_charge["NonbondedForce"]["charges"]

            pots = ff_d.createPotential(topology)
            efunc = jit(pots.getPotentialFunc())

            E_dmff = jnp.zeros((positions.shape[0],), dtype=jnp.float64)
            for ipt in range(len(positions)):
                E_dmff = E_dmff.at[ipt].set(efunc(positions[ipt], 
                                                  None,
                                                  pairs[ipt],
                                                  ffparams_wo_charge))
            E_min = jnp.min(E_dmff)
            E_dmff = E_dmff - E_min
            kT = 2.494 * 5/3 # 300 K = 2.494 kJ/mol
            weights_pts = jnp.piecewise(y_gt, [y_gt<25, y_gt>=25],
                                        [lambda x: jnp.array(1.0),
                                         lambda x: jnp.exp(-(x-25)/kT)])
            dE = E_dmff - y_gt
            mse = dE**2 * weights_pts / jnp.sum(weights_pts)
            mse = jnp.sum(mse)
            # mse = jnp.sum((E_dmff - y_gt)**2)
            return mse

        lossgrad_fn = jit(value_and_grad(compute_loss, argnums=(0)))
        for i_dihed in range(len(inputs["positions"])):
            loss_tmp, grads_tmp = lossgrad_fn(train_state.ffparams,
                                              inputs["positions"][i_dihed],
                                              inputs["pairs"][i_dihed],
                                              gt_scans[i_dihed])
            if i_dihed == 0:
                loss = loss_tmp
                grads = grads_tmp
            else:
                loss += loss_tmp
                grads = add_grad(grads, grads_tmp)

        # train_state.loss.append(loss)
        updates, new_opt_state = optimizer.update(grads, train_state.opt_state)
        ffparams = optax.apply_updates(ffparams, updates)
        ffparams = neutralize(ffparams, natoms_list)
        rescharges = update_rescharges_from_params(rescharges, ffparams)
        ff = update_ffinfo_from_rescharges(train_state.ff, rescharges)
        params_wo_charge = copy.deepcopy(ffparams)
        del params_wo_charge["NonbondedForce"]["charges"]

        return DistanceTrainState(ffparams=ffparams,
                                  ff=ff,
                                  opt_state=new_opt_state,
                                  loss=loss)

    return _DistanceTrainer(init_step, train_step)

class DistanceTrainer:
    # xml, pdbfile with topology, DihedCalculator
    def __init__(self,
                 ffxml_list,
                 nums_ffxml,
                 pdb,
                 calculator,
                 optparam_types=["NonbondedForce/charges", "NonbondedForce/sigma", "NonbondedForce/epsilon"],
                 batch_size=1,
                 optimizer_algo="adam",
                 lr=0.1):
        if isinstance(ffxml_list, str):
            ffxml_list = [ffxml_list]
        
        assert len(ffxml_list) == len(nums_ffxml), "Length of ffxml_list and nums_ffxml must be the same."

        xmlfile = merge_xml(ffxml_list, "merge.xml")
        self.ffxml = xmlfile
        self.ff = Hamiltonian(self.ffxml)
        self.pdb = app.PDBFile(pdb)
        self.potentials = self.ff.createPotential(self.pdb.topology)
        self.calculator = calculator
        self.optparam_types = optparam_types

        ffparams = self.ff.getParameters().parameters
        self.rescharges, self.natoms_list = get_rescharges_from_residues(self.ff, ratio=nums_ffxml)
        self.ffparams = get_chgparams_from_rescharges(ffparams, self.rescharges)

        multiTrans = MultiTransform(self.ffparams)
        for optparam_type in self.optparam_types:
            multiTrans[optparam_type] = genOptimizer(learning_rate=lr, clip=0.001, nonzero=False)
        multiTrans.finalize()
        grad_transform = optax.multi_transform(multiTrans.transforms, multiTrans.labels)
        mask = jax.tree_util.tree_map(lambda x: x.dtype != jnp.int32 and x.dtype != int, self.ffparams)
        self.optimizer = optax.masked(grad_transform, mask)

        self.inputs = {"positions": [],
                       "pairs": [],
                       "dihed_index": []}
        self.ytrue = []
        self.loss = []
        self.epoch = []

    def setup(self):
        shutil.copyfile(self.ffxml, "loop-0.xml")
        self.calculator.do_ffscan(f"loop-0.xml", ini_geom="QM")
        positions_list = []
        jnp_pairs_list = []
        GT_scans = []
        for i in range(len(self.calculator.dihedral_list)):
            self.calculator.do_ffscan(f"loop-0.xml", ini_geom="QM")
            positions = [atoms.positions for atoms in self.calculator.ff_distancescan[i]["atoms"]]
            positions_list.append(positions)

            jnp_pairs= []
            for j in range(len(self.calculator.ff_distancescan[0]["atoms"])):
                nbList = NoCutoffNeighborList(cov_map=self.potentials.meta["cov_map"])
                nbList.allocate(positions_list[0][j])
                jnp_pairs.append(nbList.pairs)
            jnp_pairs_list.append(jnp_pairs)

            GT_scan = self.calculator.qm_distancescan[i]["energies_kjmol"]
            GT_scans.append(GT_scan)

        self.GT_scans = GT_scans
        self.ytrue = jnp.array(self.ytrue)
        self.inputs["positions"] = jnp.array(positions_list, dtype=jnp.float32)/10 # nm
        self.inputs["pairs"] = jnp.array(jnp_pairs_list)

    def fit(self, steps=1000, relax_steps=20):
        trainer = make_distance_trainer(self.optimizer)
        p_train_step = jax.jit(trainer.train_step)
        train_state = trainer.init_step(self.ffparams, self.ff)
        for epoch in range(steps+1):
            train_state = p_train_step(train_state,
                                       self.inputs,
                                       self.GT_scans,
                                       self.natoms_list)
            if epoch % relax_steps==0:
                print(f"epoch {epoch}")
                self.loss.append(train_state.loss)
                print(f"loss {train_state.loss}")
                self.epoch.append(epoch)
                train_state.ff.renderXML(f"loop-{epoch}.xml")
                self.calculator.do_ffscan(f"loop-{epoch+1}.xml",
                                            ini_geom="FF")
 
                positions_list = []
                for i in range(len(self.calculator.ff_distancescan)):
                    positions = [atoms.positions for atoms in self.calculator.ff_distancescan[i]["atoms"]]
                    positions_list.append(positions)
                positions_list = jnp.array(positions_list)/10 # Angstrom to nm
                self.inputs["positions"] = positions_list

class DihedTrainState(NamedTuple):
    paramset_dihed: ParamSet
    paramset_all: ParamSet
    opt_state: dict
    loss: float = 0.0
    # loss: list = []

class _DihedTrainer(NamedTuple):
    init_step: Callable
    train_step: Callable

def make_dihedtrainer(
    potentials: Potential,
    optimizer: optax.GradientTransformation,
) -> _DihedTrainer:

    def init_step(paramset_dihed_init, paramset_all_init):
        opt_state = optimizer.init(paramset_dihed_init)
        return DihedTrainState(paramset_dihed=paramset_dihed_init,
                              paramset_all=paramset_all_init,
                              opt_state=opt_state)

    def train_step(train_state, inputs, gt_scans):
        inputs = inputs
        gt_scans = gt_scans
        paramset_all = train_state.paramset_all

        def add_grad(grad1:ParamSet, grad2:ParamSet):
            for k in grad1.parameters.keys():
                for t in grad1.parameters[k].keys():
                    grad1.parameters[k][t] += grad2.parameters[k][t]
            return ParamSet(grad1.parameters, grad1.mask)
        
        def compute_loss(paramset_dihed, positions, pairs, y_gt):
            efunc = potentials.getPotentialFunc()
            params_dihed = paramset_dihed.parameters
            params_all = copy.deepcopy(paramset_all)
            params_all['PeriodicTorsionForce']["proper_phase"] = \
                    params_dihed["PeriodicTorsionForce"]["proper_phase"]
            params_all['PeriodicTorsionForce']["proper_k"] = \
                    params_dihed["PeriodicTorsionForce"]["proper_k"]
            E_dmff = jnp.zeros((positions.shape[0],), dtype=jnp.float64)
            for ipt in range(len(positions)):
                E_dmff = E_dmff.at[ipt].set(efunc(positions[ipt], 
                                                  None,
                                                  pairs[ipt],
                                                  params_all))
            E_min = jnp.min(E_dmff)
            E_dmff = E_dmff - E_min
            kT = 2.494 * 5/3 # 300 K = 2.494 kJ/mol
            weights_pts = jnp.piecewise(y_gt, [y_gt<25, y_gt>=25],
                                        [lambda x: jnp.array(1.0),
                                         lambda x: jnp.exp(-(x-25)/kT)])
            dE = E_dmff - y_gt
            mse = dE**2 * weights_pts / jnp.sum(weights_pts)
            mse = jnp.sum(mse)
            # mse = jnp.sum((E_dmff - y_gt)**2)
            return mse

        lossgrad_fn = jit(value_and_grad(compute_loss, argnums=(0)))
        for i_dihed in range(len(inputs["positions"])):
            loss_tmp, grads_tmp = lossgrad_fn(train_state.paramset_dihed,
                                      inputs["positions"][i_dihed],
                                      inputs["pairs"][i_dihed],
                                      gt_scans[i_dihed])
            if i_dihed == 0:
                loss = loss_tmp
                grads = grads_tmp
            else:
                loss += loss_tmp
                grads = add_grad(grads, grads_tmp)

        # train_state.loss.append(loss)
        updates, new_opt_state = optimizer.update(grads, train_state.opt_state)
        new_paramset_dihed = optax.apply_updates(train_state.paramset_dihed, updates)
        return DihedTrainState(paramset_dihed=new_paramset_dihed,
                               paramset_all=train_state.paramset_all,
                               opt_state=new_opt_state,
                            #    loss=train_state.loss)
                               loss=loss)
    
    return _DihedTrainer(init_step, train_step)


class DihedTrainer:
    # xml, pdbfile with topology, DihedCalculator
    def __init__(self, ffxml, pdb, calculator, batch_size=1,
                 optimizer=optax.adam(learning_rate=0.1)):
        self.ffxml = ffxml
        self.ff = Hamiltonian(ffxml)
        self.pdb = app.PDBFile(pdb)
        self.potentials = self.ff.createPotential(self.pdb.topology)
        self.calculator = calculator
        self.optimizer = optimizer
        self.inputs = {"positions": [],
                       "pairs": [],
                       "dihed_index": []}
        self.ytrue = []
        self.loss = []
        self.epoch = []

    def setup(self):
        shutil.copyfile(self.ffxml, "loop-0.xml")
        self.calculator.get_dihedral_ff(f"loop-0.xml", angles=None, ini_geom="FF")
        positions_list = []
        jnp_pairs_list = []
        GT_scans = []
        for i in range(len(self.calculator.dihedral_list)):
            self.calculator.get_dihedral_ff(f"loop-0.xml", i, angles=None, ini_geom="FF")
            positions = [atoms.positions for atoms in self.calculator.ff_dihedscan[i]["atoms"]]
            positions_list.append(positions)

            jnp_pairs= []
            for i_dihed in range(len(self.calculator.ff_dihedscan[0]["atoms"])):
                nbList = NoCutoffNeighborList(cov_map=self.potentials.meta["cov_map"])
                nbList.allocate(positions_list[0][i_dihed])
                jnp_pairs.append(nbList.pairs)
            jnp_pairs_list.append(jnp_pairs)

            GT_scan = self.calculator.qm_dihedscan[i]["energies_kjmol"]
            GT_scans.append(GT_scan)

        self.GT_scans = GT_scans
        self.ytrue = jnp.array(self.ytrue)
        self.inputs["positions"] = jnp.array(positions_list, dtype=jnp.float32)/10 # nm
        self.inputs["pairs"] = jnp.array(jnp_pairs_list)

        # Parameters
        self.paramset_dihed = ParamSet()
        self.torsion_gen = PeriodicTorsionGenerator(self.ff.ffinfo, self.paramset_dihed)
        paramset_all_init = self.ff.getParameters()
        paramset_all_init.to_jax()
        self.paramset_all = copy.deepcopy(paramset_all_init)

    def fit(self, steps=1000, relax_steps=20):
        trainer = make_dihedtrainer(self.potentials, self.optimizer)
        p_train_step = jax.jit(trainer.train_step)
        train_state = trainer.init_step(self.paramset_dihed, self.paramset_all)
        for epoch in range(steps+1):
            train_state = p_train_step(train_state,
                                       self.inputs,
                                       self.GT_scans)
            if epoch % relax_steps==0:
                print(f"epoch {epoch}")
                self.loss.append(train_state.loss)
                print(f"loss {train_state.loss}")
                self.epoch.append(epoch)
                io = XMLIO()
                io.writeXML(f"loop-{epoch}.xml", self.torsion_gen.ffinfo)
                self.torsion_gen.overwrite(train_state.paramset_dihed)
                io2 = XMLIO()
                io2.writeXML(f"loop-{epoch+1}.xml", self.torsion_gen.ffinfo)
                self.calculator.get_dihedral_ff(f"loop-{epoch+1}.xml",
                                                angles=None,
                                                ini_geom="FF")
                positions_list = []
                for i in range(len(self.calculator.dihedral_list)):
                    positions = [atoms.positions for atoms in self.calculator.ff_dihedscan[i]["atoms"]]
                    positions_list.append(positions)
                positions_list = jnp.array(positions_list)/10 # Angstrom to nm
                self.inputs["positions"] = positions_list


class ThermodynamicTrainer:
    def __init__(
            self,
            xml,
            loss_fn,
            optimizer,
            epochs):
        self.hamiltonian = Hamiltonian(xml)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.best_loss = float('inf')

    def train(self):
        # Initialize the optimizer
        opt_state = self.optimizer.init(self.hamiltonian.parameters)
