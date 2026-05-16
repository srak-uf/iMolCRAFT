from dmff.common.nblist import NoCutoffNeighborList
from typing import List, Union, Optional, Callable, Any, Tuple
from dmff.mbar import MBAREstimator, Sample, OpenMMSampleState
import pickle
from openmm import app
from openmm.app import ForceField, Modeller
import jax.numpy as jnp
import os
import mdtraj as md
from jax import value_and_grad, jit
from jax.tree_util import tree_map
from ..trainer.dmff_utils import (
    update_rescharges_from_params,
    update_ffinfo_from_rescharges,
    update_ffinfo_from_params,
    md_sample,
    get_target_pred_frame,
    get_target_gt,
    plot_compare,
)
from .base import BaseTrainer
from ..calculator import DihedralCalculator, DistanceCalculator
from openmm import unit
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt


class DistanceTrainer(BaseTrainer):
    """
    Trainer for optimizing force field parameters using distance scan data.

    This class handles the setup, optimization, and checkpointing for distance-based
    parameter fitting tasks, using JAX and differentiable force fields.
    """
    def __init__(
        self,
        ffxml_list: Union[str, List[str]],
        nums_ffxml: List[int],
        pdbfile: str,
        calculator: DistanceCalculator,
        loss_fn: Callable[..., float],
        relax_steps: int = 20,
        opt_fftypes: List[str] = [
            "NonbondedForce/charge",
            "NonbondedForce/sigma",
            "NonbondedForce/epsilon",
        ],
        label: Optional[str] = None,
        batch_size: int = 1,
        optimizer_algo: str = "adam",
        lr: Union[float, List[float]] = 0.01,
        clip: Union[float, List[float]] = 0.1,
    ) -> None:

        super().__init__(
            ffxml_list=ffxml_list,
            nums_ffxml=nums_ffxml,
            pdbfile=pdbfile,
            loss_fn=loss_fn,
            opt_fftypes=opt_fftypes,
            batch_size=batch_size,
            optimizer_algo=optimizer_algo,
            label=label,
            lr=lr,
            clip=clip,
        )
        self.relax_steps = relax_steps
        self.inputs = {"positions": [], "pairs": [], "dihed_index": []}
        self.calculator = calculator
        self.efunc = jit(self.potentials.getPotentialFunc())

    def setup(self) -> None:
        """
        Set up the trainer by running force field scans and preparing input arrays.
        """
        self.calculator.do_ffscan(self.ffxml, ini_geom="QM")
        GT_scans = []
        for i in range(len(self.calculator.qm_scan)):
            GT_scans.append(self.calculator.qm_scan[i]["energy_kjmol"])
        self.GT_scans = jnp.array(GT_scans)

        positions_list = []
        jnp_pairs_list = []
        for i in range(len(self.calculator.qm_scan)):
            self.calculator.do_ffscan(self.ffxml, ini_geom="QM")
            # position unit is nanometer
            positions = [
                jnp.array(atoms.positions, dtype=jnp.float64) / 10
                for atoms in self.calculator.ff_scan[i]["atoms"]
            ]
            if self.num_vsites > 0:
                positions = [
                    self.potentials.topology.addVSiteToPos(p) for p in positions
                ]
            positions_list.append(positions)
            jnp_pairs = []
            for j in range(len(self.calculator.ff_scan[i]["atoms"])):
                nbList = NoCutoffNeighborList(cov_map=self.potentials.meta["cov_map"])
                nbList.allocate(positions_list[i][j])
                jnp_pairs.append(nbList.pairs)
            jnp_pairs_list.append(jnp_pairs)
        self.inputs["positions"] = jnp.array(positions_list, dtype=jnp.float64)
        self.inputs["pairs"] = jnp.array(jnp_pairs_list)
        self.opt_state = self.optimizer.init(self.ffparams)

    def get_loss_gradients(self) -> Tuple[Any, Any]:
        """
        Compute the loss and its gradients for the current parameters.

        Returns
        -------
        loss : float
            The computed loss value.
        grads : Any
            Gradients of the loss with respect to the parameters.
        """
        grads = tree_map(lambda x: x * 0.0, self.ffparams)
        loss = 0.0
        for i_dihed in range(len(self.inputs["positions"])):
            loss_tmp, grads_tmp = value_and_grad(self.loss_fn, argnums=0)(
                self.ffparams,
                self.efunc,
                self.inputs["positions"][i_dihed],
                self.inputs["pairs"][i_dihed],
                self.GT_scans[i_dihed],
            )
            loss += loss_tmp
            grads = tree_map(lambda x, y: x + y, grads, grads_tmp)
        return loss, grads

    def after_step(self) -> None:
        """
        Update force field and input arrays after each optimization step.
        Handles periodic relaxation and XML output.
        """
        self.ff = update_ffinfo_from_params(self.ff, self.ffparams)
        self.rescharges = update_rescharges_from_params(self.rescharges, self.ffparams)
        self.ff = update_ffinfo_from_rescharges(self.ff, self.rescharges)
        ffparams_wo_charge = {}
        for key in self.ffparams.keys():
            if key == "NonbondedForce":
                ffparams_wo_charge[key] = {}
                for key2 in self.ffparams[key].keys():
                    if key2 != "charge":
                        ffparams_wo_charge[key][key2] = self.ffparams[key][key2]
            elif key == "VsiteForce":
                pass
            else:
                ffparams_wo_charge[key] = self.ffparams[key]
        self.ff.getParameters().parameters = ffparams_wo_charge

        epoch = self._epoch + 1
        if epoch % self.relax_steps == 0:
            self.ff.renderXML(f"loop-{epoch}.xml")
            self.calculator.do_ffscan(f"loop-{epoch}.xml", ini_geom="QM")
            positions_list = []
            for i in range(len(self.calculator.ff_scan)):
                positions = [
                    jnp.array(atoms.positions, dtype=jnp.float64) / 10
                    for atoms in self.calculator.ff_scan[i]["atoms"]
                ]
                if self.num_vsites > 0:
                    positions = [
                        self.potentials.topology.addVSiteToPos(p) for p in positions
                    ]
                positions_list.append(positions)
            positions_list = jnp.array(positions_list)
            self.inputs["positions"] = positions_list

    def write_checkpoint(self, checkpoint_frequency: int) -> None:
        """
        Save the current training state to a checkpoint file.

        Parameters
        ----------
        checkpoint_frequency : int
            Frequency (in epochs) to write checkpoints.
        """
        if self._epoch % checkpoint_frequency == 0:
            with open("train_state.pkl", "wb") as f:
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
                    "clip": self.clip,
                }
                pickle.dump(dump_dict, f)

    @classmethod
    def from_checkpoint(
        cls,
        trainer_checkpoint: str,
        ffxml_list: Union[str, List[str]],
        nums_ffxml: List[int],
        pdbfile: str,
        loss_fn: Optional[Callable[..., float]] = None,
        opt_fftypes: Optional[List[str]] = None,
        optimizer_algo: Optional[str] = None,
        lr: Optional[Union[float, List[float]]] = None,
        clip: Optional[Union[float, List[float]]] = None,
    ) -> "DistanceTrainer":

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

        trainer = cls(
            ffxml_list=ffxml_list,
            nums_ffxml=nums_ffxml,
            pdbfile=pdbfile,
            calculator=dump_dict["calculator"],
            loss_fn=loss_fn,
            opt_fftypes=opt_fftypes,
            optimizer_algo=optimizer_algo,
            label=dump_dict["label"],
            lr=lr,
            clip=clip,
        )

        GT_scans = []
        for i in range(len(trainer.calculator.qm_scan)):
            GT_scans.append(trainer.calculator.qm_scan[i]["energy_kjmol"])
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
    """
    Trainer for optimizing force field parameters using dihedral scan data.

    This class handles the setup, optimization, and checkpointing for dihedral-based
    parameter fitting tasks, using JAX and differentiable force fields.
    """
    def __init__(
        self,
        ffxml: str,
        pdbfile: str,
        calculator: DihedralCalculator,
        loss_fn: Callable[..., float],
        relax_steps: int = 20,
        opt_fftypes: List[str] = [
            "PeriodicTorsionForce/proper_phase",
            "PeriodicTorsionForce/proper_k",
        ],
        label: Optional[str] = None,
        batch_size: int = 1,
        optimizer_algo: str = "adam",
        lr: Union[float, List[float]] = 0.01,
        clip: Union[float, List[float]] = 0.1,
    ) -> None:
        """
        Initialize the DihedralTrainer.

        Parameters
        ----------
        ffxml_list : str or list of str
            List of force field XML file paths or a single path.
        nums_ffxml : list of int
            Number of residues for each ffxml file.
        pdbfile : str
            Path to the PDB file.
        loss_fn : callable
            Loss function to be minimized.
        opt_fftypes : list of str
            List of force field parameter types to optimize.
        label : str, optional
            Label for the training run and output files.
        batch_size : int, optional
            Batch size for training (default: 1).
        optimizer_algo : str, optional
            Optimizer algorithm name (default: 'adam').
        lr : float or list of float, optional
            Learning rate(s) for optimizer (default: 0.0001).
        clip : float or list of float, optional
            Gradient clipping value(s) (default: 0.1).
        """
        super().__init__(
            ffxml_list=[ffxml],
            nums_ffxml=[1],
            pdbfile=pdbfile,
            loss_fn=loss_fn,
            opt_fftypes=opt_fftypes,
            batch_size=batch_size,
            optimizer_algo=optimizer_algo,
            label=label,
            lr=lr,
            clip=clip,
        )

        self.relax_steps = relax_steps
        self.inputs = {"positions": [], "pairs": [], "dihed_index": []}
        self.calculator = calculator
        self.efunc = jit(self.potentials.getPotentialFunc())

    def setup(self) -> None:
        """
        Set up the trainer by running force field scans and preparing input arrays.
        """
        GT_scans = []
        for i in range(len(self.calculator.qm_scan)):
            GT_scans.append(self.calculator.qm_scan[i]["energy_kjmol"])
        self.GT_scans = jnp.array(GT_scans)

        positions_list = []
        jnp_pairs_list = []
        for i in range(len(self.calculator.ff_scan)):
            self.calculator.do_ffscan(self.ffxml, i, angles="QM", ini_geom="FF")
            positions = [
                atoms.positions for atoms in self.calculator.ff_scan[i]["atoms"]
            ]
            positions_list.append(positions)
            jnp_pairs = []
            for i_dihed in range(len(self.calculator.ff_scan[0]["atoms"])):
                nbList = NoCutoffNeighborList(cov_map=self.potentials.meta["cov_map"])
                nbList.allocate(positions_list[0][i_dihed])
                jnp_pairs.append(nbList.pairs)
            jnp_pairs_list.append(jnp_pairs)
        self.inputs["positions"] = (
            jnp.array(positions_list, dtype=jnp.float32) / 10
        )  # nm
        self.inputs["pairs"] = jnp.array(jnp_pairs_list)
        self.opt_state = self.optimizer.init(self.ffparams)

    def get_loss_gradients(self) -> Tuple[Any, Any]:
        """
        Compute the loss and its gradients for the current parameters.

        Returns
        -------
        loss : float
            The computed loss value.
        grads : Any
            Gradients of the loss with respect to the parameters.
        """
        grads = tree_map(lambda x: x * 0.0, self.ffparams)
        loss = 0.0
        for i_dihed in range(len(self.inputs["positions"])):
            loss_tmp, grads_tmp = value_and_grad(self.loss_fn, argnums=0)(
                self.ffparams,
                self.efunc,
                self.inputs["positions"][i_dihed],
                self.inputs["pairs"][i_dihed],
                self.GT_scans[i_dihed],
            )
            loss += loss_tmp
            grads = tree_map(lambda x, y: x + y, grads, grads_tmp)
        return loss, grads

    def after_step(self) -> None:
        """
        Update force field and input arrays after each optimization step.
        Handles periodic relaxation and XML output.
        """
        self.ff = update_ffinfo_from_params(self.ff, self.ffparams)
        self.rescharges = update_rescharges_from_params(self.rescharges, self.ffparams)
        self.ff = update_ffinfo_from_rescharges(self.ff, self.rescharges)
        ffparams_wo_charge = {}
        for key in self.ffparams.keys():
            if key == "NonbondedForce":
                ffparams_wo_charge[key] = {}
                for key2 in self.ffparams[key].keys():
                    if key2 != "charge":
                        ffparams_wo_charge[key][key2] = self.ffparams[key][key2]
            elif key == "VsiteForce":
                pass
            else:
                ffparams_wo_charge[key] = self.ffparams[key]
        self.ff.getParameters().parameters = ffparams_wo_charge

        epoch = self._epoch + 1
        if epoch % self.relax_steps == 0:
            self.ff.renderXML(f"loop-{epoch}.xml")
            self.calculator.do_ffscan(f"loop-{epoch}.xml", angles="QM", ini_geom="FF")
            positions_list = []
            for i in range(len(self.calculator.ff_scan)):
                positions = [
                    jnp.array(atoms.positions, dtype=jnp.float64) / 10
                    for atoms in self.calculator.ff_scan[i]["atoms"]
                ]
                if self.num_vsites > 0:
                    positions = [
                        self.potentials.topology.addVSiteToPos(p) for p in positions
                    ]
                positions_list.append(positions)
            positions_list = jnp.array(positions_list)
            self.inputs["positions"] = positions_list


class ThermodynamicTrainer(BaseTrainer):
    """
    Trainer for optimizing force field parameters using thermodynamic property data.

    This class handles the setup, optimization, and checkpointing for thermodynamic
    property fitting tasks, using JAX and differentiable force fields.
    """
    def __init__(
        self,
        ffxml_list: Union[str, List[str]],
        nums_ffxml: List[int],
        pdbfile: str,
        loss_fn: Callable[..., float],
        sampling_params: List[Any],
        target_params: List[Any],
        opt_fftypes: List[str] = [
            "NonbondedForce/charge",
            "NonbondedForce/sigma",
            "NonbondedForce/epsilon",
        ],
        label: Optional[str] = None,
        optimizer_algo: str = "adam",
        lr: Union[float, List[float]] = 0.0001,
        clip: Union[float, List[float]] = 0.1,
        resample_freq: int = 50,
        restart_xml: str = None,
        device: str = "CPU"
    ) -> None:
        """
        Initialize the ThermodynamicTrainer.

        Parameters
        ----------
        ffxml_list : str or list of str
            List of force field XML file paths or a single path.
        nums_ffxml : list of int
            Number of residues for each ffxml file.
        pdbfile : str
            Path to the PDB file.
        loss_fn : callable
            Loss function to be minimized.
        sampling_params : list of dict
            List of dictionaries containing sampling parameters for each replica.
        target_params : list of dict
            List of dictionaries containing target parameters for each replica.
        opt_fftypes : list of str
            List of force field parameter types to optimize.
        label : str, optional
            Label for the training run and output files.
        optimizer_algo : str, optional
            Optimizer algorithm name (default: 'adam').
        lr : float or list of float, optional
            Learning rate(s) for optimizer (default: 0.0001).
        clip : float or list of float, optional
            Gradient clipping value(s) (default: 0.1).
        restart_xml : str, optional
            Path to the XML file for restarting the training.
        """
        # params
        self.sampling_params = sampling_params
        self.target_params = target_params
        self.resample_freq = resample_freq
        self.resample_counter = 0

        self.device = device

        super().__init__(
            ffxml_list=ffxml_list,
            nums_ffxml=nums_ffxml,
            pdbfile=pdbfile,
            loss_fn=loss_fn,
            opt_fftypes=opt_fftypes,
            optimizer_algo=optimizer_algo,
            label=label,
            lr=lr,
            clip=clip,
            restart_xml=restart_xml
        )

        # MD + Energy function setup
        if isinstance(self.sampling_params, dict):
            self.pdb = [self.pdb]
            self.pdbfile = [self.pdbfile]
            self.pdbfile_vsite = [self.pdbfile_vsite]
            self.sampling_params = [self.sampling_params]
            if self.sampling_params[0]["nonbondedmethod"] == "PME":
                nonbondedmethod = app.PME
            elif self.sampling_params[0]["nonbondedmethod"] == "LJPME":
                nonbondedmethod = app.LJPME
            else:
                raise AssertionError("Invalid nonbonded method")

            pots = self.ff.createPotential(
                    self.pdb[0].topology,
                    nonbondedMethod=nonbondedmethod,
                    nonbondedCutoff=self.sampling_params[0]["rcut_nm"]
                    * unit.nanometer,
                    useDispersionCorrection=self.sampling_params[0]["dispcorr"],
                    )
            self.potentials = [pots]
            self.topology = [self.topology]
            # self.efuncs = [jit(self.potentials[0].getPotentialFunc())]
            self.efuncs = [self.potentials[0].getPotentialFunc()]
        elif isinstance(self.sampling_params, list):
            self.pdbfile = []
            self.pdb = []
            self.pdbfile_vsite = []
            self.potentials = []
            self.efuncs = []
            self.topology = []
            self.T_K = []
            for i, sampling_param in enumerate(self.sampling_params):
                self.pdbfile.append(sampling_param["init_structure"])
                self.pdb.append(app.PDBFile(self.pdbfile[i]))
                if self.sampling_params[i]["nonbondedmethod"] == "PME":
                    nonbondedmethod = app.PME
                elif self.sampling_params[i]["nonbondedmethod"] == "LJPME":
                    nonbondedmethod = app.LJPME
                else:
                    raise AssertionError("Invalid nonbonded method")
                pots = self.ff.createPotential(
                        self.pdb[i].topology,
                        nonbondedMethod=nonbondedmethod,
                        nonbondedCutoff=self.sampling_params[i]["rcut_nm"]
                        * unit.nanometer,
                        useDispersionCorrection=self.sampling_params[i]["dispcorr"],
                    )
                self.potentials.append(pots)
                self.efuncs.append(jit(pots.getPotentialFunc()))
                self.topology.append(self.pdb[i].topology)
                if self.num_vsites > 0:
                    modeller = Modeller(self.pdb[i].topology, self.pdb[i].positions)
                    modeller.addExtraParticles(ForceField(self.ffxml))
                    pos = modeller.getPositions()
                    self.topology[i] = modeller.topology
                else:
                    self.topology[i] = self.pdb[i].topology
                    pos = self.pdb[i].positions
                # self.pdbfile[i]のbasenameにvs_をつけて保存
                vs_pdbfile = f"vs_{os.path.basename(self.pdbfile[i])}"
                app.PDBFile.writeFile(self.topology[i], pos, open(vs_pdbfile, "w"))
                self.pdbfile_vsite.append(vs_pdbfile)
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
        self.nonbondedmethod = []
        self.dispcorr = []
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
            self.dispcorr.append(bool(sampling_param["dispcorr"]))
            self.prod_steps.append(float(sampling_param["prod_steps"]))
            self.nstxout.append(int(sampling_param["nstxout"]))
            self.neff.append(int(sampling_param["neff"]))
            self.dt_fs.append(float(sampling_param["dt_fs"]))
            self.ensemble.append(sampling_param["ensemble"])
            self.nonbondedmethod.append(sampling_param["nonbondedmethod"])

        for i in range(len(self.nonbondedmethod)):
            if self.nonbondedmethod[i] == "PME":
                self.nonbondedmethod[i] = app.PME
            elif self.nonbondedmethod[i] == "LJPME":
                self.nonbondedmethod[i] = app.LJPME

        # target
        if isinstance(self.target_params, dict):
            self.target_params = [self.target_params]
        assert len(self.target_params) == len(
            self.sampling_params
        ), "Options scheme parameters and sampling parameters must have the same length"
        self.target_gt = []
        self.target_pred_frame = []
        self.utarget = []
        self.resample = [False for i in range(len(self.sampling_params))]

        # loss function
        if not isinstance(self.loss_fn, list):
            self.loss_fn = [self.loss_fn for _ in range(len(self.sampling_params))]

    def setup(self) -> None:
        """
        Set up the trainer by running MD simulations and preparing MBAR estimator.
        """
        self.estimator = MBAREstimator()
        if len(self.target_gt) == 0:
            has_target_gt = False
        else:
            has_target_gt = True

        for i in range(len(self.sampling_params)):
            state_name = f"sample_{i}"
            xtcfile = md_sample(
                initialpdb=self.pdbfile[i],
                ffxml=self.ffxml,
                trajectory=f"{state_name}.xtc",
                rc=self.rc_nm[i],
                T=self.T_K[i],
                anneal_Tmax=self.anneal_Tmax[i],
                anneal_steps=self.anneal_steps[i],
                anneal_totalsteps=self.anneal_totalsteps[i],
                dt=self.dt_fs[i],
                nstxout=self.nstxout[i],
                relax_steps=self.relax_steps[i],
                prod_steps=self.prod_steps[i],
                ensemble=self.ensemble[i],
                nonbondedmethod=self.nonbondedmethod[i],
                useDispersionCorrection=self.dispcorr[i],
                device=self.device
            )
            state = OpenMMSampleState(
                state_name,
                self.ffxml,
                self.pdbfile[i],  # without virtual sites
                temperature=self.T_K[i],
                pressure=self.P_bar[i],
                nonbondedMethod=self.nonbondedmethod[i],
                nonbondedCutoff=self.rc_nm[i] * unit.nanometer,
                useDispersionCorrection=self.dispcorr[i],
                platform=self.device

            )
            traj = md.load(xtcfile, top=self.pdbfile_vsite[i])
            sample = Sample(traj, state_name)
            self.estimator.add_state(state)
            self.estimator.add_sample(sample)

            if has_target_gt is False:
                self.target_gt.append(get_target_gt(self.target_params[i]))
                self.target_pred_frame.append(
                    get_target_pred_frame(
                        xtcfile, self.pdbfile_vsite[i], self.target_params[i]
                    )
                )
        self.estimator.optimize_mbar()
        self.opt_state = self.optimizer.init(self.ffparams)

    def get_loss_gradients(self) -> Tuple[Any, Any]:
        """
        Compute the loss and its gradients for the current parameters.

        Returns
        -------
        loss : float
            The computed loss value.
        grads : Any
            Gradients of the loss with respect to the parameters.
        """
        grads = tree_map(lambda x: x * 0.0, self.ffparams)
        loss = 0.0
        self.utarget = []
        self.wresults = []
        for i in range(len(self.sampling_params)):
            (loss_tmp, (utarget, wresults)), grads_tmp = value_and_grad(
                self.loss_fn[i], argnums=0, has_aux=True
            )(
                self.ffparams,
                self.efuncs[i],
                self.potentials[i].meta["cov_map"],
                self.rc_nm[i],
                self.ensemble[i],
                self.T_K[i],
                self.estimator,
                self.target_gt[i],
                self.target_pred_frame[i],
                pressure=self.P_bar[i],
            )
            loss += loss_tmp
            grads = tree_map(lambda x, y: x + y, grads, grads_tmp)
            self.utarget.append(utarget)
            self.wresults.append(wresults)
        if jnp.isnan(loss) is True:
            self.resample = [True for i in range(len(self.sampling_params))]

        return loss, grads

    def _resample(self) -> None:
        """
        Resample MD trajectories and update MBAR estimator if needed.
        """
        self.resample_counter = 0
        if len(self.estimator.states) > 0:
            removedstatename = [
                self.estimator.states[i].name
                for i, flag in enumerate(self.resample)
                if flag
            ]
            removedstateidx = [i for i, flag in enumerate(self.resample) if flag]
        else:
            removedstatename = []
            removedstateidx = [i for i in range(len(self.sampling_params))]

        for idx in removedstateidx:
            if len(removedstatename) > 0:
                self.estimator.remove_state(removedstatename[idx])
                # self.estimator.remove_sample(removedstatename[idx])
                state_name = removedstatename[idx]
            elif len(removedstateidx) > 0:
                state_name = f"sample_{idx}"
            print(f"Resampling {state_name}... by {self.ffxml}")
            xtcfile = md_sample(
                initialpdb=self.pdbfile[idx],
                ffxml=self.ffxml,
                trajectory=f"{state_name}.xtc",
                rc=self.rc_nm[idx],
                T=self.T_K[idx],
                anneal_Tmax=self.anneal_Tmax[idx],
                anneal_steps=self.anneal_steps[idx],
                anneal_totalsteps=self.anneal_totalsteps[idx],
                dt=self.dt_fs[idx],
                nstxout=self.nstxout[idx],
                relax_steps=self.relax_steps[idx],
                prod_steps=self.prod_steps[idx],
                ensemble=self.ensemble[idx],
                nonbondedmethod=self.nonbondedmethod[idx],
                useDispersionCorrection=self.dispcorr[idx],
                platform=self.device
            )
            traj = md.load(f"{xtcfile}", top=self.pdbfile_vsite[idx])
            state = OpenMMSampleState(
                state_name,
                self.ffxml,
                self.pdbfile[idx],  # without virtual sites
                temperature=self.T_K[idx],
                pressure=self.P_bar[idx],
                nonbondedMethod=self.nonbondedmethod[idx],
                nonbondedCutoff=self.rc_nm[idx] * unit.nanometer,
                useDispersionCorrection=self.dispcorr[idx],
                platform=self.device
            )
            sample = Sample(traj, state_name)
            self.target_pred_frame[idx] = get_target_pred_frame(
                xtcfile, self.pdbfile_vsite[idx], self.target_params[idx]
            )
            self.estimator.add_state(state)
            self.estimator.add_sample(sample)
        self.estimator.optimize_mbar()

    def after_step(self) -> None:
        """
        Update force field, input arrays, and resample if necessary after each
        optimization step.
        Handles periodic XML output and effective sample size checks.
        """
        self.resample_counter += 1
        if self.resample_counter >= self.resample_freq:
            self.resample = [True for i in range(len(self.sampling_params))]

        if True in self.resample:  # i.e., loss is nan
            self._resample()
            self.resample = [False for i in range(len(self.sampling_params))]
        else:
            self.ff = update_ffinfo_from_params(self.ff, self.ffparams)
            self.rescharges = update_rescharges_from_params(
                self.rescharges, self.ffparams
            )
            self.ff = update_ffinfo_from_rescharges(self.ff, self.rescharges)
            self.ff.getParameters().parameters = self.ffparams
            os.makedirs("xmlfiles", exist_ok=True)
            self.ff.renderXML(f"xmlfiles/epoch_{self.label}-{self._epoch+1}.xml")
            self.ffxml = f"xmlfiles/epoch_{self.label}-{self._epoch+1}.xml"

            print("Effective sample sizes:")
            for ii in range(len(self.sampling_params)):
                try:
                    ieff = self.estimator.estimate_effective_sample(
                        self.utarget[ii], decompose=True
                    )
                    for k, v in ieff.items():
                        print(f"  {k}: {v}")
                    for i, (k, v) in enumerate(ieff.items()):
                        if v < self.neff[ii] and k != "Total" and ii == i:
                            self.resample[i] = True
                            print(f"  {i} -> Resample")
                        else:  # Vsiteのposition update
                            # self.estimator._input***
                            pass
                except Exception:
                    print("Warning: Error in estimating effective sample size")
                    self.estimator.states = []
                    self.estimator.samples = []
                    self.resample = [True for i in range(len(self.sampling_params))]

            if True in self.resample:
                self._resample()
                self.resample = [False for i in range(len(self.sampling_params))]

    def write_checkpoint(self, checkpoint_frequency: int) -> None:
        """
        Save the current training state and plots to a checkpoint file.

        Parameters
        ----------
        checkpoint_frequency : int
            Frequency (in epochs) to write checkpoints.
        """
        if self._epoch % checkpoint_frequency == 0:
            self.ff.renderXML(f"chkpoint_{self.label}.xml")
            with open(f"train_state_{self.label}.pkl", "wb") as f:
                dump_dict = {
                    "ffparams": self.ffparams,
                    "opt_state": self.opt_state,
                    "ffinfo": self.ff.ffinfo,
                    "rescharges": self.rescharges,
                    "pdb": self.pdb,
                    "topology": self.topology,
                    "T_K": self.T_K,
                    "P_bar": self.P_bar,
                    "anneal_steps": self.anneal_steps,
                    "anneal_Tmax": self.anneal_Tmax,
                    "anneal_totalsteps": self.anneal_totalsteps,
                    "relax_steps": self.relax_steps,
                    "rc_nm": self.rc_nm,
                    "prod_steps": self.prod_steps,
                    "nstxout": self.nstxout,
                    "neff": self.neff,
                    "dt_fs": self.dt_fs,
                    "ensemble": self.ensemble,
                    "target_params": self.target_params,
                    "sampling_params": self.sampling_params,
                    "epoch": self._epoch,
                    "losses": self.losses,
                    "epochs": self.epochs,
                    "label": self.label,
                    "optimizer_algo": self.optimizer_algo,
                    "opt_fftypes": self.opt_fftypes,
                    "lr": self.lr,
                    "clip": self.clip,
                    "target_gt": self.target_gt,
                    "target_pred_frame": self.target_pred_frame,
                }
                pickle.dump(dump_dict, f)

            # plotter
            for i in range(len(self.target_gt)):
                plot_compare(
                    self.target_gt[i], self.target_pred_frame[i],
                    label=f"sample_{self.label}_{i}"
                )

            fig, ax = plt.subplots(1, 1, figsize=(3.25, 2.5))
            ax.plot(self.epochs, self.losses)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            plt.tight_layout()
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            fig.savefig(f"{self.label}_learning_curve.png")
            plt.close(fig)

            fig, ax = plt.subplots(1, 1, figsize=(3.25, 2.5))
            ax.set_yscale("log")
            ax.plot(self.epochs, self.losses)
            plt.tight_layout()
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            fig.savefig(f"logy_{self.label}_learning_curve.png")
            plt.close(fig)

            fig, ax = plt.subplots(1, 1, figsize=(3.25, 2.5))
            ax.set_yscale("log")
            ax.set_xscale("log")
            ax.plot(self.epochs, self.losses)
            plt.tight_layout()
            fig.savefig(f"logylogx_{self.label}_learning_curve.png")
            plt.close(fig)

    @classmethod
    def from_checkpoint(
        cls,
        trainer_checkpoint: str,
        ffxml_list: Union[str, List[str]],
        nums_ffxml: List[int],
        pdbfile: str,
        initial_ffxml: str,
        loss_fn: Optional[Callable[..., float]] = None,
        sampling_params: Optional[List[Any]] = None,
        target_params: Optional[List[Any]] = None,
        opt_fftypes: Optional[List[str]] = None,
        optimizer_algo: Optional[str] = None,
        lr: Optional[Union[float, List[float]]] = None,
        clip: Optional[Union[float, List[float]]] = None,
    ) -> "ThermodynamicTrainer":

        with open(trainer_checkpoint, "rb") as f:
            dump_dict = pickle.load(f)

        if lr is None:
            lr = dump_dict["lr"]
            del dump_dict["lr"]
        if clip is None:
            clip = dump_dict["clip"]
            del dump_dict["clip"]
        if optimizer_algo is None:
            optimizer_algo = dump_dict["optimizer_algo"]
            del dump_dict["optimizer_algo"]
        if opt_fftypes is None:
            opt_fftypes = dump_dict["opt_fftypes"]
            del dump_dict["opt_fftypes"]
        if sampling_params is None:
            sampling_params = dump_dict["sampling_params"]
            del dump_dict["sampling_params"]
        if target_params is None:
            target_params = dump_dict["target_params"]
            del dump_dict["target_params"]
        
        trainer = cls(
            ffxml_list=ffxml_list,
            nums_ffxml=nums_ffxml,
            pdbfile=pdbfile,
            loss_fn=loss_fn,
            sampling_params=sampling_params,
            target_params=target_params,
            opt_fftypes=opt_fftypes,
            optimizer_algo=optimizer_algo,
            label=dump_dict["label"],
            lr=lr,
            clip=clip,
            restart_xml=initial_ffxml,
        )

        attr_lists = ["epoch", "epochs", "losses", "ff_info"]
        for key, value in dump_dict.items():
            if key in attr_lists:
                setattr(trainer, key, value)

        # order is important
        trainer.ffxml = initial_ffxml
        trainer.opt_state = dump_dict["opt_state"]
        trainer._epoch = dump_dict["epoch"]

        return trainer
