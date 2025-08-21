from sympy import ff
from dmff import Hamiltonian, DMFFTopology
from dmff.common.nblist import NoCutoffNeighborList
from dmff.api.xmlio import XMLIO
from dmff.api.hamiltonian import Potential
from dmff.optimize import MultiTransform, genOptimizer
from dmff.mbar import MBAREstimator, Sample, OpenMMSampleState
import pickle
from openmm import app
from openmm.app import NoCutoff, Simulation, PDBFile, ForceField, Modeller
import jax.numpy as jnp
import os
import mdtraj as md
from jax import value_and_grad
from jax.tree_util import tree_map
from ..trainer.dmff_utils import  \
    update_rescharges_from_params, update_ffinfo_from_rescharges, \
    update_ffinfo_from_params, md_sample, get_target_pred_frame, get_target_gt
from .base import BaseTrainer
from ..calculator import DihedralCalculator, DistanceCalculator
from openmm import unit

class DistanceTrainer(BaseTrainer):
    def __init__(self,
                 ffxml_list,
                 nums_ffxml,
                 pdbfile,
                 calculator: DistanceCalculator,
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
                 calculator: DihedralCalculator,
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
            self.calculator.do_ffscan(self.ffxml, i, angles="QM", ini_geom="FF")
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
                                      angles="QM",
                                      ini_geom="FF")
            positions_list = []
            for i in range(len(self.calculator.ff_scan)):
                positions = [jnp.array(atoms.positions, dtype=jnp.float64) / 10 \
                        for atoms in self.calculator.ff_scan[i]["atoms"]]
                if self.num_vsites > 0:
                    positions = [self.potentials.topology.addVSiteToPos(p) for p in positions]
                positions_list.append(positions)
            positions_list = jnp.array(positions_list)
            self.inputs["positions"] = positions_list


class ThermodynamicTrainer(BaseTrainer):
    def __init__(
            self,
            ffxml_list,
            nums_ffxml,
            pdbfile,
            loss_fn,
            sampling_params,
            target_params,
            opt_fftypes=["NonbondedForce/charges",
                         "NonbondedForce/sigma",
                         "NonbondedForce/epsilon"],
            label=None,
            optimizer_algo="adam",
            lr=0.0001,
            clip=0.1
            ):
        
        # params
        self.sampling_params = sampling_params
        self.target_params = target_params

        super().__init__(ffxml_list=ffxml_list,
                         nums_ffxml=nums_ffxml,
                         pdbfile=pdbfile,
                         loss_fn=loss_fn,
                         opt_fftypes=opt_fftypes,
                         optimizer_algo=optimizer_algo,
                         label=label,
                         lr=lr,
                         clip=clip)
        
        # MD + Energy function setup
        if isinstance(self.sampling_params, dict):
            self.pdb = [self.pdb]
            self.pdbfile = [self.pdbfile]
            self.potentials = [self.potentials]
            self.topology = [self.topology]
            self.sampling_params = [self.sampling_params]
        elif isinstance(self.sampling_params, list):
            self.pdbfile = []
            self.pdb = []
            self.potentials = []
            self.topology = []
            self.T_K = []
            for i, sampling_param in enumerate(self.sampling_params):
                self.pdbfile.append(sampling_param["init_structure"])
                self.pdb.append(app.PDBFile(self.pdbfile[i]))
                self.potentials.append(self.ff.createPotential(self.pdb[i].topology,
                                                               nonbondedMethod=app.PME,
                                                               nonbondedCutoff=self.rc*unit.nanometer))
                self.topology.append(self.pdb[i].topology)
                if self.num_vsites > 0:
                    modeller = Modeller(self.pdb[i].topology, self.pdb[i].positions)
                    modeller.addExtraParticles(ForceField(self.ffxml))
                    pos = modeller.getPositions()
                    self.topology[i] = modeller.topology
                else:
                    self.topology[i] = self.pdb[i].topology
        else:
            raise AssertionError("Invalid sampling parameters")

        self.T_K = []
        self.P_bar = []
        self.anneal_steps = []
        self.anneal_Tmax = []
        self.anneal_steps = []
        self.anneal_totalsteps = []
        self.relax_steps = []
        self.rc_nm = []
        self.prod_steps = []
        self.nstxout = []
        self.neff = []
        self.dt_fs = []
        self.ensemble = []
        for i, sampling_param in enumerate(self.sampling_params):
            self.T_K.append(float(sampling_param["temperature_K"]))
            self.P_bar.append(float(sampling_param["pressure_bar"]))
            self.anneal_steps.append(int(sampling_param["anneal_steps"]))
            self.anneal_Tmax.append(float(sampling_param["anneal_Tmax"]))
            self.anneal_totalsteps.append(int(sampling_param["anneal_totalsteps"]))
            self.relax_steps.append(int(sampling_param["relax_steps"]))
            self.rc_nm.append(float(sampling_param["rcut_nm"]))
            self.prod_steps.append(float(sampling_param["prod_steps"]))
            self.nstxout.append(int(sampling_param["nstxout"]))
            self.neff.append(int(sampling_param["neff"]))
            self.dt_fs.append(float(sampling_param["dt_fs"]))
            self.ensemble.append(sampling_param["ensemble"])

        # target
        if isinstance(self.target_params, dict):
            self.target_params = [self.target_params]
        assert len(self.target_params) == len(self.sampling_params), \
            "Options scheme parameters and sampling parameters must have the same length"
        self.target_gt = []
        self.target_pred_frame = []
        self.utarget = []
        self.resample = [False for i in range(len(self.sampling_params))]

        # loss function
        if not isinstance(self.loss_fn, list):
            self.loss_fn = [self.loss_fn for _ in range(len(self.sampling_params))]

    def setup(self):
        self.estimator = MBAREstimator()
        for i in range(len(self.sampling_params)):
            state_name = f"sample_{i}"
            xtcfile = md_sample(self.pdbfile[i],
                                self.ffxml,
                                f"{state_name}.xtc",
                                self.rc_nm[i],
                                self.T_K[i],
                                self.anneal_Tmax[i],
                                self.anneal_steps[i],
                                self.anneal_totalsteps[i],
                                self.dt_fs[i],
                                self.nstxout[i],
                                self.relax_steps[i],
                                self.prod_steps[i],
                                self.ensemble[i]
                                )
            state = OpenMMSampleState(state_name,
                                      self.ffxml,
                                      self.pdbfile[i], # without virtual sites
                                      temperature=self.T_K[i],
                                      pressure=self.P_bar[i],
                                      nonbondedMethod=app.PME,
                                      nonbondedCutoff=self.rc_nm[i]*unit.nanometer)
            traj = md.load(xtcfile, top=self.pdbfile[i])
            sample = Sample(traj, state_name)
            self.estimator.add_state(state)
            self.estimator.add_sample(sample)

            self.target_gt.append(get_target_gt(self.target_params[i]))
            self.target_pred_frame.append(get_target_pred_frame(xtcfile,
                                                                self.pdbfile[i],
                                                                self.target_params[i]))
        self.estimator.optimize_mbar()
        self.opt_state = self.optimizer.init(self.ffparams)

    def get_loss_gradients(self):
        grads = tree_map(lambda x: x*0.0, self.ffparams)
        loss = 0.0
        for i in range(len(self.sampling_params)):
            (loss_tmp, (utarget, wresults)), grads_tmp = value_and_grad(self.loss_fn[i], argnums=0, has_aux=True)(
                                                             self.ffparams,
                                                             self.ff,
                                                             self.pdb[i].topology, # wo virtual sites
                                                             self.potentials[i].meta['cov_map'],
                                                             self.rc_nm[i],
                                                             self.ensemble[i],
                                                             self.T_K[i],
                                                             self.estimator,
                                                             self.target_gt[i],
                                                             self.target_pred_frame[i],
                                                             pressure=self.P_bar[i]
                                                             )
            loss += loss_tmp
            grads = tree_map(lambda x, y: x + y, grads, grads_tmp)
            self.utarget.append(utarget)
        if jnp.isnan(loss) == True:
            self.resample = [True for i in range(len(self.sampling_params))]

        return loss, grads

    def _resample(self):
        if len(self.estimator.states) > 0:
            removedstatename = [self.estimator.states[i].name for i, flag in enumerate(self.resample) if flag]
            removedstateidx = [i for i, flag in enumerate(self.resample) if flag]
        else:
            removedstatename = [self.estimator.states[i].name for i in self.sampling_params]
            removedstateidx = [i for i in self.sampling_params]

        for idx in removedstateidx:
            if len(removedstatename) > 0:
                self.estimator.remove_state(removedstatename[idx])
                # self.estimator.remove_sample(removedstatename[idx])
                state_name = removedstatename[idx]
            elif len(removedstateidx) > 0:
                state_name = f"sample_{idx}"
            xtcfile = md_sample(self.pdbfile[idx],
                                self.ffxml,
                                f"{state_name}.xtc",
                                self.rc_nm[idx],
                                self.T_K[idx],
                                self.anneal_Tmax[idx],
                                self.anneal_steps[idx],
                                self.anneal_totalsteps[idx],
                                self.dt_fs[idx],
                                self.nstxout[idx],
                                self.relax_steps[idx],
                                self.prod_steps[idx],
                                self.ensemble[idx])
            traj = md.load(f"{xtcfile}", top=self.pdbfile[idx])
            state = OpenMMSampleState(state_name,
                                    self.ffxml,
                                    self.pdbfile[idx], # without virtual sites
                                    temperature=self.T_K[idx],
                                    pressure=self.P_bar[idx],
                                    nonbondedMethod=app.PME,
                                    nonbondedCutoff=self.rc_nm[idx]*unit.nanometer)
            sample = Sample(traj, state_name)
            self.estimator.add_state(state)
            self.estimator.add_sample(sample)
        self.estimator.optimize_mbar()

    def after_step(self):
        if True in self.resample:
            self._resample()
            self.resample = [False for i in range(len(self.sampling_params))]
        else:
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
            os.makedirs("xmlfiles", exist_ok=True)
            self.ff.renderXML(f"xmlfiles/epoch-{self._epoch}.xml")
            self.ffxml = f'xmlfiles/epoch-{self._epoch}.xml'

            print('Effective sample sizes:')
            for ii in range(len(self.sampling_params)):
                try:
                    ieff = self.estimator.estimate_effective_sample(self.utarget[ii], decompose=True)
                    for k, v in ieff.items():
                        print(f'{k}: {v}')
                    for i, (k, v) in enumerate(ieff.items()):
                        if v < self.neff and k != "Total" and ii == i:
                            self.resample[i] = True
                            print(f"{i} -> Resample")
                except:
                    print(f"Warning: Error in estimating effective sample size")
                    self.estimator.states = []
                    self.estimator.samples = []
                    self.resample = [True for i in range(len(self.sampling_params))]

            if True in self.resample:
                self._resample()
                self.resample = [False for i in range(len(self.sampling_params))]
