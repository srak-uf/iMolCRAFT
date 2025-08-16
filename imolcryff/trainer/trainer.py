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
from ..trainer.dmff_utils import get_loss_autograd, merge_xml, neutralize, \
    update_rescharges_from_params, update_ffinfo_from_rescharges, \
    get_chgparams_from_rescharges, get_rescharges_from_residues, \
    update_ffinfo_from_params, vsiteinfo_to_params, md_sample, \
    saver_wresults, get_target_pred_frame, get_target_gt, plot_compare
from ..trainer.base import BaseTrainer
from openmm import unit

def mse_energy(e_ff, e_qm, weight_scheme="uniform", norm_var=True):
    """
    Calculate the mean squared error between two energy arrays.
    :param e_ff: Array of energies from the force field.
    :param e_qm: Array of energies from quantum mechanics.
    :param weight_scheme: Optional weighting scheme for the energies.
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
    se_array = jnp.square(e_ff - e_qm - delta_e)
    if weight_scheme == "uniform":
        weight = jnp.ones_like(e_qm) / len(e_qm)
    elif weight_scheme == "boltzmann":
        kT = 2.494 * 5/3  # 300 K = 2.494 kJ/mol
        ave = jnp.mean(e_qm)
        weight = jnp.exp(-(e_qm - ave) / kT)
        weight = weight / jnp.sum(weight)
    elif weight_scheme == "nonboltzmann":
        pass
    mse = jnp.sum(se_array * weight) / var_qm

    return mse

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


class DistanceTrainer(BaseTrainer):
    def __init__(self,
                 ffxml_list,
                 nums_ffxml,
                 pdbfile,
                 calculator,
                 loss_fn,
                 relax_steps=20,
                 opt_fftypes=["NonbondedForce/charges",
                              "NonbondedForce/sigma",
                              "NonbondedForce/epsilon"],
                 label=None,
                 batch_size=1,
                 optimizer_algo="adam",
                 lr=0.01,
                 clip=0.1):
        
        super().__init__(ffxml_list=ffxml_list,
                         nums_ffxml=nums_ffxml,
                         pdbfile=pdbfile,
                         loss_fn=loss_fn,
                         opt_fftypes=opt_fftypes,
                         batch_size=batch_size,
                         optimizer_algo=optimizer_algo,
                         label=label,
                         lr=lr,
                         clip=clip)
        self.relax_steps = relax_steps
        self.inputs = {"positions": [],
                       "pairs": [],
                       "dihed_index": []}
        self.calculator = calculator

    def setup(self):
        self.calculator.do_ffscan(self.ffxml, ini_geom="QM")
        GT_scans = []
        for i in range(len(self.calculator.qm_distancescan)):
            GT_scans.append(self.calculator.qm_distancescan[i]["energy_kjmol"])
        self.GT_scans = jnp.array(GT_scans)

        positions_list = []
        jnp_pairs_list = []
        for i in range(len(self.calculator.qm_distancescan)):
            self.calculator.do_ffscan(self.ffxml, ini_geom="QM")
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
        self.opt_state = self.optimizer.init(self.ffparams)

    def get_loss_gradients(self):
        grads = tree_map(lambda x: x*0.0, self.ffparams)
        loss = 0.0
        for i_dihed in range(len(self.inputs["positions"])):
            loss_tmp, grads_tmp = value_and_grad(self.loss_fn, argnums=0)(self.ffparams,
                                                                self.ff,
                                                                self.pdb.topology,
                                                                self.inputs["positions"][i_dihed],
                                                                self.inputs["pairs"][i_dihed],
                                                                self.GT_scans[i_dihed])
            loss += loss_tmp
            grads = tree_map(lambda x, y: x + y, grads, grads_tmp)
        return loss, grads

    def after_step(self):
        self.ff = update_ffinfo_from_params(self.ff, self.ffparams)
        self.rescharges = update_rescharges_from_params(self.rescharges, self.ffparams)
        self.ff = update_ffinfo_from_rescharges(self.ff, self.rescharges)
        ffparams_wo_charge = {}
        for key in self.ffparams.keys():
            if key == "NonbondedForce":
                ffparams_wo_charge[key] = {}
                for key2 in self.ffparams[key].keys():
                    if key2 != "charges":
                        ffparams_wo_charge[key][key2] = self.ffparams[key][key2]
            elif key == "VsiteForce":
                pass
            else:
                ffparams_wo_charge[key] = self.ffparams[key]
        self.ff.getParameters().parameters = ffparams_wo_charge

        epoch = self._epoch + 1
        if epoch % self.relax_steps == 0:
            self.ff.renderXML(f"loop-{epoch}.xml")
            self.calculator.do_ffscan(f"loop-{epoch}.xml",
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

    def write_checkpoint(self, checkpoint_frequency):
        if self._epoch % checkpoint_frequency == 0:
            with open(f"train_state.pkl", "wb") as f:
                dump_dict = {
                    "ffparams": self.ffparams,
                    "opt_state": self.opt_state,
                    "ffinfo": self.ff.ffinfo,
                    "rescharges": self.rescharges,
                    "positions": self.inputs["positions"],
                    "pairs": self.inputs["pairs"],
                    "calculator": self.calculator,
                    "epoch": self._epoch,
                    "losses": self.losses,
                    "epochs": self.epochs,
                    "label": self.label,
                    "optimizer_algo": self.optimizer_algo,
                    "opt_fftypes": self.opt_fftypes,
                    "lr": self.lr,
                    "clip": self.clip
                }
                pickle.dump(dump_dict, f)

    @classmethod
    def from_checkpoint(cls,
                        trainer_checkpoint,
                        ffxml_list,
                        nums_ffxml,
                        pdbfile,
                        loss_fn = None,
                        opt_fftypes=None,
                        optimizer_algo=None,
                        lr=None,
                        clip=None):

        with open(trainer_checkpoint, "rb") as f:
            dump_dict = pickle.load(f)

        if lr is None:
            lr = dump_dict["lr"]
        if clip is None:
            clip = dump_dict["clip"]
        if optimizer_algo is None:
            optimizer_algo = dump_dict["optimizer_algo"]
        if opt_fftypes is None:
            opt_fftypes = dump_dict["opt_fftypes"]

        trainer = cls(ffxml_list=ffxml_list,
                      nums_ffxml=nums_ffxml,
                      pdbfile=pdbfile,
                      calculator=dump_dict["calculator"],
                      loss_fn=loss_fn,
                      opt_fftypes=opt_fftypes,
                      optimizer_algo=optimizer_algo,
                      label=dump_dict["label"],
                      lr=lr,
                      clip=clip)
        
        GT_scans = []
        for i in range(len(trainer.calculator.qm_distancescan)):
            GT_scans.append(trainer.calculator.qm_distancescan[i]["energy_kjmol"])
        trainer.GT_scans = jnp.array(GT_scans)
        
        trainer.opt_state = dump_dict["opt_state"]
        trainer.ffparams = dump_dict["ffparams"]
        trainer.ff.ffinfo = dump_dict["ffinfo"]
        trainer.rescharges = dump_dict["rescharges"]
        trainer.inputs["positions"] = dump_dict["positions"]
        trainer.inputs["pairs"] = dump_dict["pairs"]
        trainer._epoch = dump_dict["epoch"]
        trainer.losses = dump_dict["losses"]
        trainer.epochs = dump_dict["epochs"]

        return trainer

class DihedralTrainer(BaseTrainer):
    # xml, pdbfile with topology, DihedralCalculator
    def __init__(self,
                 ffxml,
                 pdbfile,
                 calculator,
                 loss_fn,
                 relax_steps=20,
                 opt_fftypes=["PeriodicTorsionForce/proper_phase", 
                              "PeriodicTorsionForce/proper_k"],
                 label=None,
                 batch_size=1,
                 optimizer_algo="adam",
                 lr=0.01,
                 clip=0.1):
        
        super().__init__(ffxml_list=[ffxml],
                         nums_ffxml=[1],
                         pdbfile=pdbfile,
                         loss_fn=loss_fn,
                         opt_fftypes=opt_fftypes,
                         batch_size=batch_size,
                         optimizer_algo=optimizer_algo,
                         label=label,
                         lr=lr,
                         clip=clip)
        
        self.relax_steps = relax_steps
        self.inputs = {"positions": [],
                       "pairs": [],
                       "dihed_index": []}
        self.calculator = calculator

    def setup(self):
        GT_scans = []
        for i in range(len(self.calculator.qm_scan)):
            GT_scans.append(self.calculator.qm_scan[i]["energy_kjmol"])
        self.GT_scans = jnp.array(GT_scans)

        positions_list = []
        jnp_pairs_list = []
        for i in range(len(self.calculator.ff_scan)):
            self.calculator.do_ffscan(self.ffxml, i, angles=None)
            positions = [atoms.positions for atoms in self.calculator.ff_scan[i]["atoms"]]
            positions_list.append(positions)
            jnp_pairs= []
            for i_dihed in range(len(self.calculator.ff_scan[0]["atoms"])):
                nbList = NoCutoffNeighborList(cov_map=self.potentials.meta["cov_map"])
                nbList.allocate(positions_list[0][i_dihed])
                jnp_pairs.append(nbList.pairs)
            jnp_pairs_list.append(jnp_pairs)
        self.inputs["positions"] = jnp.array(positions_list, dtype=jnp.float32)/10 # nm
        self.inputs["pairs"] = jnp.array(jnp_pairs_list)
        self.opt_state = self.optimizer.init(self.ffparams)
    
    def get_loss_gradients(self):
        grads = tree_map(lambda x: x*0.0, self.ffparams)
        loss = 0.0
        for i_dihed in range(len(self.inputs["positions"])):
            loss_tmp, grads_tmp = value_and_grad(self.loss_fn, argnums=0)(self.ffparams,
                                                                self.ff,
                                                                self.pdb.topology,
                                                                self.inputs["positions"][i_dihed],
                                                                self.inputs["pairs"][i_dihed],
                                                                self.GT_scans[i_dihed])
            loss += loss_tmp
            grads = tree_map(lambda x, y: x + y, grads, grads_tmp)
        return loss, grads

    def after_step(self):
        self.ff = update_ffinfo_from_params(self.ff, self.ffparams)
        self.rescharges = update_rescharges_from_params(self.rescharges, self.ffparams)
        self.ff = update_ffinfo_from_rescharges(self.ff, self.rescharges)
        ffparams_wo_charge = {}
        for key in self.ffparams.keys():
            if key == "NonbondedForce":
                ffparams_wo_charge[key] = {}
                for key2 in self.ffparams[key].keys():
                    if key2 != "charges":
                        ffparams_wo_charge[key][key2] = self.ffparams[key][key2]
            elif key == "VsiteForce":
                pass
            else:
                ffparams_wo_charge[key] = self.ffparams[key]
        self.ff.getParameters().parameters = ffparams_wo_charge

        epoch = self._epoch + 1
        if epoch % self.relax_steps == 0:
            self.ff.renderXML(f"loop-{epoch}.xml")
            self.calculator.do_ffscan(f"loop-{epoch}.xml",
                                      angles=None,
                                      ini_geom="QM")
            positions_list = []
            for i in range(len(self.calculator.ff_scan)):
                positions = [jnp.array(atoms.positions, dtype=jnp.float64) / 10 \
                        for atoms in self.calculator.ff_scan[i]["atoms"]]
                if self.num_vsites > 0:
                    positions = [self.potentials.topology.addVSiteToPos(p) for p in positions]
                positions_list.append(positions)
            positions_list = jnp.array(positions_list)
            self.inputs["positions"] = positions_list


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

    def setup(self, checkpoint=None):
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
