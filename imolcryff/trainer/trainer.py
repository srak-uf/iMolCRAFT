from dmff import Hamiltonian, DMFFTopology
from dmff.common.nblist import NoCutoffNeighborList
from dmff.generators.classical import PeriodicTorsionGenerator
from dmff.api.paramset import ParamSet
from dmff.api.xmlio import XMLIO
from dmff.api.hamiltonian import Potential
from dmff.optimize import MultiTransform, genOptimizer
from dmff.mbar import MBAREstimator, Sample, OpenMMSampleState
import pickle
from openmm import app
from openmm.app import NoCutoff, Simulation, PDBFile, ForceField, Modeller
import copy, shutil
import numpy as np
import jax
import jax.numpy as jnp
import os
import mdtraj as md
from jax import value_and_grad, jit, vmap
from jax.tree_util import tree_map
import optax
from tqdm import tqdm
from typing import NamedTuple, Callable
from ..calculator import DihedCalculator
from ..crafter.ffxml import check_vsite, delvsite_pdb
from ..trainer.dmff_utils import get_loss_autograd, merge_xml, neutralize, \
    update_rescharges_from_params, update_ffinfo_from_rescharges, \
    get_chgparams_from_rescharges, get_rescharges_from_residues, \
    update_ffinfo_from_params, vsiteinfo_to_params, md_sample, \
    saver_wresults, get_target_pred_frame, get_target_gt, plot_compare
import time

class DistanceTrainState(NamedTuple):
    ffparams: dict
    ff: Hamiltonian
    rescharges: dict
    opt_state: dict
    loss: float = 0.0

class _DistanceTrainer(NamedTuple):
    init_step: Callable
    train_step: Callable

def make_distance_trainer(
        optimizer: optax.GradientTransformation,
) -> _DistanceTrainer:

    def init_step(ffparams, ff, rescharges):
        opt_state = optimizer.init(ffparams)
        return DistanceTrainState(ffparams=ffparams,
                                  ff=ff,
                                  rescharges=rescharges,
                                  opt_state=opt_state)

    def train_step(train_state,
                   inputs,
                   topology,
                   gt_scans,
                   add_mask_fn=None,
                   modify_ffparams=None):
        """
        train_state: DistanceTrainState
        inputs: dict with keys "positions", "pairs", "dihed_index"
        topology: OpenMM Topology object
        gt_scans: ground truth scans (jnp.array)
        natoms_list: list of number of atoms in each residue
        charge_params, sigma_params, epsilon_params: dicts with target lists for parameters
        """ 
        def Loss(ffparams, ff, topology, positions, pairs, y_gt):
            # ここでefuncを再定義することでchargesやvsiteの微分を可能にしている
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

            # print(f"ffparams_wo_charge: {ffparams_wo_charge}")
            pots = ff_d.createPotential(topology) # should be pdb topology wo vsites
            efunc = pots.getPotentialFunc()
            # batched_efunc = jit(vmap(lambda x: efunc(x, None, pairs[0], ffparams_wo_charge)))
            batched_efunc = vmap(lambda x: efunc(x, None, pairs[0], ffparams_wo_charge))
            E_dmff = batched_efunc(positions)
            E_min = jnp.min(E_dmff)
            E_dmff = E_dmff - E_min
            kT = 2.494 * 5/3 # 300 K = 2.494 kJ/mol
            weights_pts = jnp.piecewise(y_gt, [y_gt<100, y_gt>=100],
                                        [lambda x: jnp.array(1.0),
                                         lambda x: jnp.exp(-(x-100)/kT)])
            dE = E_dmff - y_gt
            mse = dE**2 * weights_pts / jnp.sum(weights_pts)
            mse = jnp.sum(mse)
            return mse

        ###########################

        # initialize gradient
        grads = tree_map(lambda x: x*0.0, train_state.ffparams)
        loss = 0.0

        for i_dihed in range(len(inputs["positions"])):
            ## for debug
            # grads_tmp = tree_map(lambda x: x*0.0, train_state.ffparams)
            # loss_tmp = 0.0
            ###############
            loss_tmp, grads_tmp = value_and_grad(Loss, argnums=0)(train_state.ffparams,
                                                                  train_state.ff,
                                                                  topology,
                                                                  inputs["positions"][i_dihed],
                                                                  inputs["pairs"][i_dihed],
                                                                  gt_scans[i_dihed])

            loss += loss_tmp
            grads = tree_map(lambda x, y: x + y, grads, grads_tmp)

        if add_mask_fn is not None:
            grads = add_mask_fn(grads)

        updates, new_opt_state = optimizer.update(grads, train_state.opt_state)
        # print(f"updates: {updates}")
        ffparams = optax.apply_updates(train_state.ffparams, updates)

        if modify_ffparams is not None:
            ffparams = modify_ffparams(ffparams)

        ff = update_ffinfo_from_params(train_state.ff, ffparams)
        rescharges = update_rescharges_from_params(train_state.rescharges, ffparams)
        ff = update_ffinfo_from_rescharges(ff, rescharges)
        ffparams_wo_charge = {}
        for key in train_state.ffparams.keys():
            if key == "NonbondedForce":
                ffparams_wo_charge[key] = {}
                for key2 in train_state.ffparams[key].keys():
                    if key2 != "charges":
                        ffparams_wo_charge[key][key2] = train_state.ffparams[key][key2]
            elif key == "VsiteForce":
                pass
            else:
                ffparams_wo_charge[key] = train_state.ffparams[key]

        ff.getParameters().parameters = ffparams_wo_charge

        return DistanceTrainState(ffparams=ffparams,
                                  ff=ff,
                                  rescharges=rescharges,
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
                 lr=0.01):
        if isinstance(ffxml_list, str):
            ffxml_list = [ffxml_list]
        
        assert len(ffxml_list) == len(nums_ffxml), "Length of ffxml_list and nums_ffxml must be the same."

        xmlfile = merge_xml(ffxml_list, "merge.xml")
        self.ffxml = xmlfile
        self.ff = Hamiltonian(self.ffxml)
        self.num_vsites = check_vsite(self.ffxml)
        self._epoch = 0

        self.pdb = app.PDBFile(pdb)
        self.potentials = self.ff.createPotential(self.pdb.topology)
        self.calculator = calculator
        if self.num_vsites > 0:
            modeller = Modeller(self.pdb.topology, self.pdb.positions)
            modeller.addExtraParticles(ForceField(self.ffxml))
            pos = modeller.getPositions()
            self.topology = modeller.topology
        else:
            self.topology = self.pdb.topology
        self.optparam_types = optparam_types

        ffparams = self.ff.getParameters().parameters
        self.rescharges, self.natoms_list = get_rescharges_from_residues(self.ff, ratio=nums_ffxml)
        if self.num_vsites > 0:
            ffparams = vsiteinfo_to_params(self.ff, ffparams)
        self.ffparams = get_chgparams_from_rescharges(ffparams, self.rescharges)

        multiTrans = MultiTransform(self.ffparams)
        for optparam_type in self.optparam_types:
            # multiTrans[optparam_type] = genOptimizer(learning_rate=lr, clip=0.001, nonzero=False)
            if optparam_type == "NonbondedForce/charges":
                multiTrans[optparam_type] = genOptimizer(learning_rate=lr, nonzero=False)
            else:
                multiTrans[optparam_type] = genOptimizer(learning_rate=lr,
                                                         nonzero=False) # Should be True
        multiTrans.finalize()
        self.grad_transform = optax.multi_transform(multiTrans.transforms, multiTrans.labels)
        mask = jax.tree_util.tree_map(lambda x: x.dtype != jnp.int32 and x.dtype != int, self.ffparams)
        self.optimizer = optax.masked(self.grad_transform, mask)

        self.inputs = {"positions": [],
                       "pairs": [],
                       "dihed_index": []}
        self.loss = []
        self.epoch = []
        self.all_loss = []
        self.all_epoch = []

    def setup(self, trainer_checkpoint=None):
        shutil.copyfile(self.ffxml, "loop-0.xml")
        self.calculator.do_ffscan(f"loop-0.xml", ini_geom="QM")

        GT_scans = []
        for i in range(len(self.calculator.qm_distancescan)):
            GT_scan = self.calculator.qm_distancescan[i]["energy_kjmol"] \
                        - np.array(self.calculator.qm_distancescan[i]["energy_kjmol"]).min()
            GT_scans.append(GT_scan)
        self.GT_scans = jnp.array(GT_scans)
        
        if trainer_checkpoint is not None:
            with open(trainer_checkpoint, "rb") as f:
                dump_dict = pickle.load(f)
            opt_state = dump_dict["opt_state"]
            self.ffparams = dump_dict["ffparams"]
            self.ff.ffinfo = dump_dict["ffinfo"]
            self.rescharges = dump_dict["rescharges"]
            self.trainer = make_distance_trainer(self.optimizer)
            self.train_state = DistanceTrainState(ffparams=self.ffparams,
                                                  ff=self.ff,
                                                  rescharges=self.rescharges,
                                                  opt_state=opt_state)
            self.inputs["positions"] = dump_dict["positions"]
            self.inputs["pairs"] = dump_dict["pairs"]
            self._epoch = dump_dict["epoch"] + 1
        else:
            self.trainer = make_distance_trainer(self.optimizer)
            self.train_state = self.trainer.init_step(self.ffparams, self.ff, self.rescharges)
            positions_list = []
            jnp_pairs_list = []
            for i in range(len(self.calculator.qm_distancescan)):
                self.calculator.do_ffscan(f"loop-0.xml", ini_geom="QM")
                # position unit is nanometer
                positions = [jnp.array(atoms.positions, dtype=jnp.float64) / 10 \
                                for atoms in self.calculator.ff_distancescan[i]["atoms"]]
                
                if self.num_vsites > 0:
                    positions = [self.potentials.topology.addVSiteToPos(p) for p in positions]

                positions_list.append(positions)

                jnp_pairs= []
                for j in range(len(self.calculator.ff_distancescan[0]["atoms"])):
                    nbList = NoCutoffNeighborList(cov_map=self.potentials.meta["cov_map"])
                    nbList.allocate(positions_list[0][j])
                    jnp_pairs.append(nbList.pairs)
                jnp_pairs_list.append(jnp_pairs)
            
            self.inputs["positions"] = jnp.array(positions_list, dtype=jnp.float64) 
            self.inputs["pairs"] = jnp.array(jnp_pairs_list)

    def fit(self, steps=1000, relax_steps=20, add_mask_fn=None, ffparams_modify=None):
        p_train_step = self.trainer.train_step
        for epoch in range(self._epoch, steps+1):
            print(f"epoch {epoch}")
            start_time = time.time()
            self.train_state = p_train_step(self.train_state,
                                            self.inputs,
                                            self.pdb.topology,
                                            self.GT_scans,
                                            add_mask_fn=add_mask_fn,
                                            modify_ffparams=ffparams_modify,
                                            )
            end_time = time.time()
            print(f"Time taken for epoch {epoch}: {end_time - start_time:.2f} seconds")
            self.all_loss.append(self.train_state.loss)
            self.all_epoch.append(epoch)
            self._epoch = epoch + 1
            
            if epoch % relax_steps==0:
                # self.trainerをpickleで保存
                with open(f"train_state.pkl", "wb") as f:
                    dump_dict = {
                        "ffparams": self.train_state.ffparams,
                        "opt_state": self.train_state.opt_state,
                        "ffinfo": self.train_state.ff.ffinfo,
                        "loss": self.train_state.loss,
                        "rescharges": self.train_state.rescharges,
                        "positions": self.inputs["positions"],
                        "pairs": self.inputs["pairs"],
                        "epoch": epoch,
                    }
                    pickle.dump(dump_dict, f)

                print(f"epoch {epoch}")
                self.loss.append(self.train_state.loss)
                print(f"loss {self.train_state.loss}")
                self.epoch.append(epoch)
                self.train_state.ff.renderXML(f"loop-{epoch+1}.xml")
                self.calculator.do_ffscan(f"loop-{epoch+1}.xml",
                                          ini_geom="QM")
 
                positions_list = []
                for i in range(len(self.calculator.ff_distancescan)):
                    positions = [jnp.array(atoms.positions, dtype=jnp.float64) / 10 \
                            for atoms in self.calculator.ff_distancescan[i]["atoms"]]
                    if self.num_vsites > 0:
                        positions = [self.potentials.topology.addVSiteToPos(p) for p in positions]
                    positions_list.append(positions)
                positions_list = jnp.array(positions_list)
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


from openmm import unit

class ThermodynamicTrainer:
    def __init__(
            self,
            dmff_params,
            label,
            chkfile=None):
        
        ffxml_list = dmff_params["forcefield"]["xml_list"]
        self.initial_xml = merge_xml(ffxml_list, "epoch-0.xml")
        self.ff = Hamiltonian(ffxml_list)
        self.pdbfile = dmff_params["sampling"]["init_structure"]
        self.pdbfile_wovs = self.pdbfile
        self.pdb = PDBFile(self.pdbfile)
        self.pdb_wovs = PDBFile(self.pdbfile)

        # Virtual sites
        modeller = app.Modeller(self.pdb.topology, self.pdb.getPositions())
        modeller.addExtraParticles(app.ForceField(self.initial_xml))
        self.pos = modeller.getPositions()
        self.topology = modeller.topology
        num_vsites = 0
        for residue in self.ff.ffinfo["Residues"]:
            if "vsites" in residue.keys():
                num_vsites += len(residue["vsites"])
        if num_vsites > 0:
            basename = os.path.splitext(self.pdbfile)[0]
            self.pdbfile = basename + "_vs.pdb"
            app.PDBFile.writeFile(self.topology, self.pos, open(self.pdbfile, "w"))

        # MD + Energy function setup
        self.rc = float(dmff_params["forcefield"]["rcut_nm"])
        pots = self.ff.createPotential(self.pdb.topology,
                                       nonbondedMethod=app.PME,
                                       nonbondedCutoff=self.rc*unit.nanometer)
        self.cov_map = pots.meta['cov_map']
        self.ffparams = self.ff.getParameters().parameters
        res_ratio = dmff_params["forcefield"]["residue_ratio"]
        self.rescharges, self.natoms_list = get_rescharges_from_residues(self.ff, ratio=res_ratio)
        self.T_K = float(dmff_params["sampling"]["temperature_K"])
        self.P_bar = float(dmff_params["sampling"]["pressure_bar"])
        self.ensemble = dmff_params["sampling"]["ensemble"]

        self.neff = dmff_params["opt_scheme"]["neff"]

        # params
        self.sampling_params = dmff_params["sampling"]
        self.forcefield_params = dmff_params["forcefield"]
        self.optscheme_params = dmff_params["opt_scheme"]

    def setup(self):
        state_name = "loop-0"
        self.state_init = md_sample(self.pdbfile,
                               self.initial_xml,
                               f"{state_name}.xtc",
                               self.sampling_params,
                               self.forcefield_params)
        self.saver = saver_wresults(self.optscheme_params)
        self.estimator = MBAREstimator()
        state = OpenMMSampleState(state_name,
                                  self.initial_xml,
                                  self.pdbfile_wovs, # without virtual sites
                                  temperature=self.T_K,
                                  pressure=self.P_bar,
                                  nonbondedMethod=app.PME,
                                  nonbondedCutoff=self.rc*unit.nanometer)
        traj = md.load(f'xtcfiles/{state_name}.xtc', top=self.pdbfile)
        sample = Sample(traj, state_name)
        self.estimator.add_state(state)
        self.estimator.add_sample(sample)
        self.estimator.optimize_mbar()

        self.target_gt = get_target_gt(self.optscheme_params)
        self.target_pred_frame = get_target_pred_frame("xtcfiles/loop-0.xtc",
                                                      self.pdbfile,
                                                      self.optscheme_params)

    def fit(self, steps=500, add_mask_fn=None, modify_ffparams=None):
        for i_epoch in range(1, steps+1):
            (loss, (utarget, wresults)), gradient = get_loss_autograd(
                                                    ffparams,
                                                    ff,
                                                    self.pdb_wovs.topology, # wo virtual sites
                                                    self.cov_map,
                                                    self.rc,
                                                    self.ensemble,
                                                    self.T_K,
                                                    self.estimator,
                                                    self.target_gt,
                                                    self.target_pred_frame,
                                                    pressure=self.P_bar
                                                    )

            if jnp.isnan(loss) == False:
                nan_flag = False
                print("Loss:", loss)
                self.saver.append(i_epoch-1, float(loss), wresults)
                if os.path.isfile(f"xmlfiles/epoch-{i_epoch-1}.xml") == False:
                    for i in range(i_epoch, 0, -1): # "xmlfiles/epoch-*.xmlのうち、引数が最新のものをコピー
                        if os.path.isfile(f"xmlfiles/epoch-{i-2}.xml"):
                            shutil.copy(f"xmlfiles/epoch-{i-2}.xml", f"xmlfiles/epoch-{i_epoch-1}.xml")
                            break
                
                gradient = add_mask_fn(gradient) if add_mask_fn is not None else gradient
                updates, opt_state = self.grad_transform.update(gradient, opt_state)
                ffparams = optax.apply_updates(ffparams, updates)
                if modify_ffparams is not None:
                    ffparams = modify_ffparams(ffparams)

                rescharges = update_rescharges_from_params(rescharges, ffparams)
                ff = update_ffinfo_from_rescharges(ff, rescharges)
                params_wo_charge = copy.deepcopy(ffparams)
                if "NonbondedForce" in params_wo_charge:
                    if "charges" in params_wo_charge["NonbondedForce"]:
                        del params_wo_charge["NonbondedForce"]["charges"]
                if "VsiteForce" in params_wo_charge:
                    del params_wo_charge["VsiteForce"]

                ff.getParameters().parameters = params_wo_charge
                os.makedirs("xmlfiles", exist_ok=True)
                ff.renderXML(f"xmlfiles/epoch-{i_epoch}.xml")
                ffxml = f'xmlfiles/epoch-{i_epoch}.xml'

                # checkout the effective size of each sample in the estimator
                print('Effective sample sizes:')
                try:
                    ieff = self.estimator.estimate_effective_sample(utarget, decompose=True)
                    for k, v in ieff.items():
                        print(f'{k}: {v}')
                    for k, v in ieff.items():
                        if v < self.neff and k != "Total":
                            self.estimator.remove_state(k)
                except:
                    self.estimator.states = []
                    self.estimator.samples = []
            else:
                nan_flag = True
                self.estimator.states = []
                self.estimator.samples = []

            if len(self.estimator.states) < 1 or nan_flag == True:
                print("Add", f"loop-{i_epoch}")
                # get new sample using the current state
                self.state_init = md_sample(self.pdbfile_wovs,
                                       ffxml,
                                       f"loop-{i_epoch}.xtc",
                                       self.sampling_params,
                                       self.forcefield_params
                                       )
                traj = md.load(f"xtcfiles/loop-{i_epoch}.xtc", top=self.pdbfile)
                state = OpenMMSampleState(f"loop-{i_epoch}",
                                          ffxml,
                                          self.pdbfile_wovs, # without virtual sites
                                          temperature=self.T_K,
                                          pressure=self.P_bar,
                                          nonbondedMethod=app.PME,
                                          nonbondedCutoff=self.rc*unit.nanometer)
                sample = Sample(traj, f"loop-{i_epoch}")
                self.estimator.add_state(state)
                self.estimator.add_sample(sample)
                self.estimator.optimize_mbar()
                
                self.target_pred_frame = get_target_pred_frame(f"xtcfiles/loop-{i_epoch}.xtc",
                                                               self.pdbfile,
                                                               self.optscheme_params)
                plot_compare(self.target_gt, self.target_pred_frame)

            self.saver.save(basename="results")
            self.saver.plot_learningcurve()
