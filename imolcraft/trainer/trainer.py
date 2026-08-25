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
    get_validation_gt,
    get_validation_pred,
    plot_compare,
    plot_validation,
    plot_validation_curves,
    get_chgparams_from_rescharges
)
from .base import BaseTrainer, plot_learning_curve
from ..calculator import DihedralCalculator, DistanceCalculator
from openmm import unit


#: Per-replica sampling settings, as (attribute name, type to cast to).
_SAMPLING_FIELDS = (
    ("T_K", float),
    ("P_bar", float),
    ("anneal_steps", int),
    ("anneal_Tmax", float),
    ("anneal_totalsteps", int),
    ("relax_steps", int),
    ("rc_nm", float),
    ("dispcorr", bool),
    ("prod_steps", float),
    ("nstxout", int),
    ("neff", int),
    ("dt_fs", float),
    ("ensemble", str),
    ("nonbondedmethod", str),
)

#: Attribute names differing from the key used in the sampling parameters.
_SAMPLING_KEYS = {
    "T_K": "temperature_K",
    "P_bar": "pressure_bar",
    "rc_nm": "rcut_nm",
}

#: Nonbonded methods accepted in the sampling parameters.
_NONBONDED_METHODS = {"PME": app.PME, "LJPME": app.LJPME}


def _state_name(idx: int) -> str:
    """Name of the MBAR state of one replica. Must match everywhere it is used."""
    return f"sample_{idx}"


def _validation_record(values_per_replica: List[dict], epoch: int) -> dict:
    """
    One history record out of the per-replica validation values, keyed by the
    replica the value belongs to so that the entries of two replicas sharing a
    name stay apart.
    """
    record = {"epoch": epoch}
    for idx, values in enumerate(values_per_replica):
        for name, value in values.items():
            record[f"{_state_name(idx)}/{name}"] = value
    return record


def _resolve_nonbondedmethod(name):
    """Translate the nonbonded method name of the YAML into an OpenMM constant."""
    if name not in _NONBONDED_METHODS:
        raise ValueError(
            f"Invalid nonbonded method: {name}. Must be one of "
            f"{list(_NONBONDED_METHODS)}."
        )
    return _NONBONDED_METHODS[name]


def _scan_positions_nm(ff_scan, potentials, num_vsites, dtype=jnp.float64):
    """
    Coordinates of every scan point in nanometre, with the virtual sites
    inserted when the force field has any.

    ASE keeps angstrom, DMFF expects nanometre, hence the division by ten.
    """
    positions_list = []
    for scan in ff_scan:
        positions = [
            jnp.array(atoms.positions, dtype=dtype) / 10 for atoms in scan["atoms"]
        ]
        if num_vsites > 0:
            positions = [potentials.topology.addVSiteToPos(p) for p in positions]
        positions_list.append(positions)
    return positions_list


def _neighbour_pairs(positions, potentials):
    """Neighbour list pairs of every geometry, for the cutoff-free scan energies."""
    pairs = []
    for pos in positions:
        nbList = NoCutoffNeighborList(cov_map=potentials.meta["cov_map"])
        nbList.allocate(pos)
        pairs.append(nbList.pairs)
    return pairs


def _qm_energies(qm_scan):
    """Reference energies of every scan, in kJ/mol."""
    return jnp.array([scan["energy_kjmol"] for scan in qm_scan])


def _ffparams_without_charge(ffparams):
    """
    The force field parameters minus the charges and the virtual site force.

    The charges live in the residue templates rather than in the parameter
    tree, so handing them back to the Hamiltonian would duplicate them.
    """
    stripped = {}
    for key in ffparams.keys():
        if key == "NonbondedForce":
            stripped[key] = {
                key2: value
                for key2, value in ffparams[key].items()
                if key2 != "charge"
            }
        elif key == "VsiteForce":
            pass
        else:
            stripped[key] = ffparams[key]
    return stripped


def _sum_loss_and_grads(ffparams, loss_and_grads):
    """Accumulate the losses and gradients of every scan into one pair."""
    grads = tree_map(lambda x: x * 0.0, ffparams)
    loss = 0.0
    for loss_tmp, grads_tmp in loss_and_grads:
        loss += loss_tmp
        grads = tree_map(lambda x, y: x + y, grads, grads_tmp)
    return loss, grads


class _ScanTrainerMixin:
    """Shared behaviour of the trainers fitting a relaxed scan."""

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
        return _sum_loss_and_grads(
            self.ffparams,
            (
                value_and_grad(self.loss_fn, argnums=0)(
                    self.ffparams,
                    self.efunc,
                    self.inputs["positions"][i],
                    self.inputs["pairs"][i],
                    self.GT_scans[i],
                )
                for i in range(len(self.inputs["positions"]))
            ),
        )

    def _render_loop_xml(self, epoch: int) -> str:
        """
        Write the current force field for the periodic re-relaxation and return
        its path.

        The label is part of the name so that two trainers combined by
        SumTrainer do not overwrite each other's files.
        """
        os.makedirs("xmlfiles", exist_ok=True)
        path = f"xmlfiles/loop_{self.label}-{epoch}.xml"
        self.ff.renderXML(path)
        return path

    def _push_params_to_ff(self) -> None:
        """Write the optimized parameters back into the Hamiltonian."""
        self.ff = update_ffinfo_from_params(self.ff, self.ffparams)
        self.rescharges = update_rescharges_from_params(self.rescharges, self.ffparams)
        self.ff = update_ffinfo_from_rescharges(self.ff, self.rescharges)
        self.ff.getParameters().parameters = _ffparams_without_charge(self.ffparams)


class DistanceTrainer(_ScanTrainerMixin, BaseTrainer):
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
        self.GT_scans = _qm_energies(self.calculator.qm_scan)

        positions_list = []
        jnp_pairs_list = []
        for i in range(len(self.calculator.qm_scan)):
            self.calculator.do_ffscan(self.ffxml, ini_geom="QM")
            # position unit is nanometer
            positions_list = _scan_positions_nm(
                self.calculator.ff_scan[: i + 1], self.potentials, self.num_vsites
            )
            jnp_pairs_list.append(
                _neighbour_pairs(positions_list[i], self.potentials)
            )
        self.inputs["positions"] = jnp.array(positions_list, dtype=jnp.float64)
        self.inputs["pairs"] = jnp.array(jnp_pairs_list)
        self.opt_state = self.optimizer.init(self.ffparams)

    def after_step(self) -> None:
        """
        Update force field and input arrays after each optimization step.
        Handles periodic relaxation and XML output.
        """
        self._push_params_to_ff()

        epoch = self._epoch + 1
        if epoch % self.relax_steps == 0:
            self.calculator.do_ffscan(self._render_loop_xml(epoch), ini_geom="QM")
            self.inputs["positions"] = jnp.array(
                _scan_positions_nm(
                    self.calculator.ff_scan, self.potentials, self.num_vsites
                )
            )

    def write_checkpoint(self, checkpoint_frequency: int) -> None:
        """
        Save the current training state to a checkpoint file.

        The file name carries the label so that two trainers combined by
        SumTrainer do not overwrite each other.

        Parameters
        ----------
        checkpoint_frequency : int
            Frequency (in epochs) to write checkpoints.
        """
        if self._epoch % checkpoint_frequency == 0:
            with open(f"train_state_{self.label}.pkl", "wb") as f:
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
                    **self._best_checkpoint_fields(),
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

        trainer.GT_scans = _qm_energies(trainer.calculator.qm_scan)

        trainer.opt_state = dump_dict["opt_state"]
        trainer.ffparams = dump_dict["ffparams"]
        trainer.ff.ffinfo = dump_dict["ffinfo"]
        trainer.rescharges = dump_dict["rescharges"]
        trainer.inputs["positions"] = dump_dict["positions"]
        trainer.inputs["pairs"] = dump_dict["pairs"]
        trainer._epoch = dump_dict["epoch"]
        trainer.losses = dump_dict["losses"]
        trainer.epochs = dump_dict["epochs"]
        trainer.validation_history = dump_dict.get("validation_history", [])
        trainer.validation_dev_history = dump_dict.get("validation_dev_history", [])
        trainer.validation_pred = dump_dict.get(
            "validation_pred", trainer.validation_pred
        )
        trainer.validation_dev = dump_dict.get(
            "validation_dev", trainer.validation_dev
        )
        trainer.validation_curves = dump_dict.get(
            "validation_curves", trainer.validation_curves
        )
        trainer._restore_best(dump_dict)

        return trainer


class DihedralTrainer(_ScanTrainerMixin, BaseTrainer):
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
        self.GT_scans = _qm_energies(self.calculator.qm_scan)

        positions_list = []
        jnp_pairs_list = []
        for i in range(len(self.calculator.ff_scan)):
            self.calculator.do_ffscan(self.ffxml, i, angles="QM", ini_geom="FF")
            positions = [
                atoms.positions for atoms in self.calculator.ff_scan[i]["atoms"]
            ]
            positions_list.append(positions)
            jnp_pairs_list.append(
                _neighbour_pairs(positions_list[0], self.potentials)
            )
        self.inputs["positions"] = (
            jnp.array(positions_list, dtype=jnp.float32) / 10
        )  # nm
        self.inputs["pairs"] = jnp.array(jnp_pairs_list)
        self.opt_state = self.optimizer.init(self.ffparams)

    def after_step(self) -> None:
        """
        Update force field and input arrays after each optimization step.
        Handles periodic relaxation and XML output.
        """
        self._push_params_to_ff()

        epoch = self._epoch + 1
        if epoch % self.relax_steps == 0:
            self.calculator.do_ffscan(
                self._render_loop_xml(epoch), angles="QM", ini_geom="FF"
            )
            self.inputs["positions"] = jnp.array(
                _scan_positions_nm(
                    self.calculator.ff_scan, self.potentials, self.num_vsites
                )
            )


    def write_checkpoint(self, checkpoint_frequency: int) -> None:
        """
        Save the current training state to a checkpoint file.

        The file name carries the label so that two trainers combined by
        SumTrainer do not overwrite each other.

        Parameters
        ----------
        checkpoint_frequency : int
            Frequency (in epochs) to write checkpoints.
        """
        if self._epoch % checkpoint_frequency == 0:
            with open(f"train_state_{self.label}.pkl", "wb") as f:
                pickle.dump(
                    {
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
                        **self._best_checkpoint_fields(),
                    },
                    f,
                )

    @classmethod
    def from_checkpoint(
        cls,
        trainer_checkpoint: str,
        ffxml: str,
        pdbfile: str,
        loss_fn: Optional[Callable[..., float]] = None,
        opt_fftypes: Optional[List[str]] = None,
        optimizer_algo: Optional[str] = None,
        lr: Optional[Union[float, List[float]]] = None,
        clip: Optional[Union[float, List[float]]] = None,
    ) -> "DihedralTrainer":
        """
        Rebuild a DihedralTrainer from a checkpoint.

        The scan geometries and the neighbour lists come from the checkpoint,
        so :meth:`setup` does not have to run the force field scan again.

        Parameters
        ----------
        trainer_checkpoint : str
            Path of the pickle written by :meth:`write_checkpoint`.
        ffxml : str
            Force field XML to resume from.
        pdbfile : str
            Path to the PDB file.
        loss_fn : callable, optional
            Loss function; the stored one cannot be pickled, so it is passed in.
        opt_fftypes, optimizer_algo, lr, clip : optional
            Override the values stored in the checkpoint.

        Returns
        -------
        DihedralTrainer
        """
        with open(trainer_checkpoint, "rb") as f:
            dump_dict = pickle.load(f)

        trainer = cls(
            ffxml=ffxml,
            pdbfile=pdbfile,
            calculator=dump_dict["calculator"],
            loss_fn=loss_fn,
            opt_fftypes=(
                dump_dict["opt_fftypes"] if opt_fftypes is None else opt_fftypes
            ),
            optimizer_algo=(
                dump_dict["optimizer_algo"]
                if optimizer_algo is None
                else optimizer_algo
            ),
            label=dump_dict["label"],
            lr=dump_dict["lr"] if lr is None else lr,
            clip=dump_dict["clip"] if clip is None else clip,
        )

        trainer.GT_scans = _qm_energies(trainer.calculator.qm_scan)
        trainer.opt_state = dump_dict["opt_state"]
        trainer.ffparams = dump_dict["ffparams"]
        trainer.ff.ffinfo = dump_dict["ffinfo"]
        trainer.rescharges = dump_dict["rescharges"]
        trainer.inputs["positions"] = dump_dict["positions"]
        trainer.inputs["pairs"] = dump_dict["pairs"]
        trainer._epoch = dump_dict["epoch"]
        trainer.losses = dump_dict["losses"]
        trainer.epochs = dump_dict["epochs"]
        trainer._restore_best(dump_dict)

        return trainer


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
        validation_params: Optional[List[Any]] = None,
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
        validation_params : list of dict, optional
            List of dictionaries containing validation parameters for each
            replica, as parsed from the ``validation`` section of the YAML.
            These properties are computed on every resampled trajectory and
            recorded, but never enter the loss. Default is None, i.e. nothing
            is monitored.
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
        self.validation_params = validation_params
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
            nonbondedmethod = _resolve_nonbondedmethod(
                self.sampling_params[0]["nonbondedmethod"]
            )

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
                nonbondedmethod = _resolve_nonbondedmethod(
                    self.sampling_params[i]["nonbondedmethod"]
                )
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
                with open(vs_pdbfile, "w") as f:
                    app.PDBFile.writeFile(self.topology[i], pos, f)
                self.pdbfile_vsite.append(vs_pdbfile)
        else:
            raise ValueError(
                "sampling_params must be a dict or a list of dicts, got "
                f"{type(self.sampling_params).__name__}"
            )

        # spread the per-replica sampling settings into parallel lists
        for name, cast in _SAMPLING_FIELDS:
            setattr(
                self,
                name,
                [cast(p[_SAMPLING_KEYS.get(name, name)]) for p in self.sampling_params],
            )

        self.nonbondedmethod = [
            _resolve_nonbondedmethod(name) for name in self.nonbondedmethod
        ]

        # target
        if isinstance(self.target_params, dict):
            self.target_params = [self.target_params]
        if len(self.target_params) != len(self.sampling_params):
            raise ValueError(
                "target_params and sampling_params must have the same length: "
                f"{len(self.target_params)} != {len(self.sampling_params)}"
            )
        self.target_gt = []
        self.target_pred_frame = []
        self.utarget = []
        self.resample = [False for i in range(len(self.sampling_params))]

        # validation, monitored only: one block dict per replica, empty when
        # nothing is asked for
        if self.validation_params is None:
            self.validation_params = [{} for _ in self.sampling_params]
        if isinstance(self.validation_params, dict):
            self.validation_params = [self.validation_params]
        if len(self.validation_params) != len(self.sampling_params):
            raise ValueError(
                "validation_params and sampling_params must have the same length: "
                f"{len(self.validation_params)} != {len(self.sampling_params)}"
            )
        self.validation_gt = {}
        for i, params in enumerate(self.validation_params):
            for name, gt in get_validation_gt(params).items():
                self.validation_gt[f"{_state_name(i)}/{name}"] = gt
        self.validation_pred = [{} for _ in self.sampling_params]
        self.validation_dev = [{} for _ in self.sampling_params]
        self.validation_curves = [{} for _ in self.sampling_params]
        self.validation_history = []
        self.validation_dev_history = []

        # loss function
        if not isinstance(self.loss_fn, list):
            self.loss_fn = [self.loss_fn for _ in range(len(self.sampling_params))]

    def _run_md(self, idx: int, state_name: str) -> str:
        """Run the MD of one replica with the current force field."""
        return md_sample(
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
            device=self.device,
        )

    def _add_sample(self, idx: int, state_name: str, xtcfile: str) -> None:
        """
        Register a trajectory and the state it was sampled in with the MBAR
        estimator.

        The state is described by the PDB without virtual sites, while the
        trajectory carries them, hence the two different topologies.
        """
        state = OpenMMSampleState(
            state_name,
            self.ffxml,
            self.pdbfile[idx],  # without virtual sites
            temperature=self.T_K[idx],
            pressure=self.P_bar[idx],
            nonbondedMethod=self.nonbondedmethod[idx],
            nonbondedCutoff=self.rc_nm[idx] * unit.nanometer,
            useDispersionCorrection=self.dispcorr[idx],
            platform=self.device,
        )
        traj = md.load(xtcfile, top=self.pdbfile_vsite[idx])
        self.estimator.add_state(state)
        self.estimator.add_sample(Sample(traj, state_name))

    def _update_validation(self, idx: int, xtcfile: str) -> None:
        """
        Recompute the validation properties of one replica from its fresh
        trajectory.

        They are read from the trajectory that has just been sampled, so they
        describe the force field of this epoch without any MBAR reweighting:
        a diffusion coefficient is a dynamical quantity, which reweighting
        static configurations cannot give.
        """
        if not self.validation_params[idx]:
            return
        (
            self.validation_pred[idx],
            self.validation_dev[idx],
            self.validation_curves[idx],
        ) = get_validation_pred(
            xtcfile, self.pdbfile_vsite[idx], self.validation_params[idx]
        )
        for name, value in self.validation_pred[idx].items():
            print(f"Validation {_state_name(idx)}/{name}: {value:.6e}")
        for name, deviation in self.validation_dev[idx].items():
            print(f"Validation {_state_name(idx)}/{name} deviation: {deviation:.6e}")

    def _record_validation(self) -> None:
        """
        Append the current validation values, and their deviations from the
        references, to their histories.

        The two are kept apart because they are read differently: a value is
        read against the reference line of its own property, a deviation
        against zero. The deviations are not combined with one another either;
        what a run is judged by stays the user's call.

        Only the replicas resampled in this round have fresh values; the others
        keep the ones of their last sampling, so a record always describes every
        replica. The histories ride in the checkpoint, next to the losses.
        """
        if not any(self.validation_params):
            return
        self.validation_history.append(
            _validation_record(self.validation_pred, self._epoch)
        )
        self.validation_dev_history.append(
            _validation_record(self.validation_dev, self._epoch)
        )

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
            state_name = _state_name(i)
            xtcfile = self._run_md(i, state_name)
            self._add_sample(i, state_name, xtcfile)
            self._update_validation(i, xtcfile)

            if has_target_gt is False:
                self.target_gt.append(get_target_gt(self.target_params[i]))
                self.target_pred_frame.append(
                    get_target_pred_frame(
                        xtcfile, self.pdbfile_vsite[i], self.target_params[i]
                    )
                )
        self._record_validation()
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

    def _resample_indices(self) -> List[int]:
        """
        Replica indices whose trajectory has to be sampled again.

        With nothing registered in the estimator yet, every replica has to be
        sampled; otherwise only the ones flagged by :meth:`after_step`.
        """
        if len(self.estimator.states) > 0:
            return [i for i, flag in enumerate(self.resample) if flag]
        return list(range(len(self.sampling_params)))

    def _resample(self) -> None:
        """
        Resample MD trajectories and update MBAR estimator if needed.

        States are addressed by name rather than by their position in
        ``estimator.states``: removing and re-adding a state moves it to the
        end of that list, so the positions stop matching the replica indices
        after the first resampling.
        """
        self.resample_counter = 0
        registered = {state.name for state in self.estimator.states}

        for idx in self._resample_indices():
            state_name = _state_name(idx)
            if state_name in registered:
                self.estimator.remove_state(state_name)
                # self.estimator.remove_sample(state_name)
            print(f"Resampling {state_name}... by {self.ffxml}")
            xtcfile = self._run_md(idx, state_name)
            self.target_pred_frame[idx] = get_target_pred_frame(
                xtcfile, self.pdbfile_vsite[idx], self.target_params[idx]
            )
            self._add_sample(idx, state_name, xtcfile)
            self._update_validation(idx, xtcfile)
        self._record_validation()
        self.estimator.optimize_mbar()

    def _needs_resample(self, ii: int, ieff: dict) -> bool:
        """
        Whether replica ``ii`` no longer contributes enough effective samples.

        ``ieff`` is keyed by state name, so the entry of this replica is looked
        up by name. Taking the ``ii``-th entry instead would break as soon as
        :meth:`_resample` reorders ``estimator.states``.
        """
        own = ieff.get(_state_name(ii))
        return own is not None and own < self.neff[ii]

    def after_step(self) -> None:
        """
        Update force field, input arrays, and resample if necessary after each
        optimization step.
        Handles periodic XML output and effective sample size checks.
        """
        self.resample_counter += 1
        if self.resample_counter >= self.resample_freq:
            self.resample = [True for i in range(len(self.sampling_params))]
        loss_value = getattr(self, "loss", None)
        loss_is_invalid = loss_value is not None and (
            bool(jnp.isnan(loss_value)) or bool(jnp.isinf(loss_value))
        )
        if loss_is_invalid:
            print(
                "Warning: Loss is NaN or Inf. "
                "Resampling with the last valid force field XML."
            )
            self.resample = [True for i in range(len(self.sampling_params))]

        self.ff = update_ffinfo_from_params(self.ff, self.ffparams)
        self.rescharges = update_rescharges_from_params(self.rescharges, self.ffparams)
        self.ff = update_ffinfo_from_rescharges(self.ff, self.rescharges)
        self.ff.getParameters().parameters = self.ffparams
        os.makedirs("xmlfiles", exist_ok=True)
        self.ff.renderXML(f"xmlfiles/epoch_{self.label}-{self._epoch+1}.xml")
        self.ffxml = f"xmlfiles/epoch_{self.label}-{self._epoch+1}.xml"
        self.ffparams = get_chgparams_from_rescharges(self.ffparams, self.rescharges)

        print("Effective sample sizes:")
        for ii in range(len(self.sampling_params)):
            try:
                ieff = self.estimator.estimate_effective_sample(
                    self.utarget[ii], decompose=True
                )
                for k, v in ieff.items():
                    print(f"  {k}: {v}")
                if self._needs_resample(ii, ieff):
                    self.resample[ii] = True
                    print(f"  {ii} -> Resample")
                # TODO: Vsiteのposition update (self.estimator._input***)
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
                    "validation_params": self.validation_params,
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
                    "validation_pred": self.validation_pred,
                    "validation_dev": self.validation_dev,
                    "validation_curves": self.validation_curves,
                    "validation_history": self.validation_history,
                    "validation_dev_history": self.validation_dev_history,
                    **self._best_checkpoint_fields(),
                }
                pickle.dump(dump_dict, f)

            # plotter
            for i in range(len(self.target_gt)):
                plot_compare(
                    self.target_gt[i], self.target_pred_frame[i],
                    label=f"sample_{self.label}_{i}"
                )

            plot_validation(
                self.validation_history,
                gt=self.validation_gt,
                label=f"validation_{self.label}",
            )
            plot_validation(
                self.validation_dev_history,
                baseline=0.0,
                label=f"validation_dev_{self.label}",
            )
            for i, curves in enumerate(self.validation_curves):
                plot_validation_curves(
                    curves,
                    self.validation_params[i],
                    label=f"validation_curves_{self.label}_{i}",
                )

            plot_learning_curve(self.epochs, self.losses, self.label)

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
        validation_params: Optional[List[Any]] = None,
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
        # checkpoints written before the validation section have no such key
        if validation_params is None:
            validation_params = dump_dict.pop("validation_params", None)
        
        trainer = cls(
            ffxml_list=ffxml_list,
            nums_ffxml=nums_ffxml,
            pdbfile=pdbfile,
            loss_fn=loss_fn,
            sampling_params=sampling_params,
            target_params=target_params,
            validation_params=validation_params,
            opt_fftypes=opt_fftypes,
            optimizer_algo=optimizer_algo,
            label=dump_dict["label"],
            lr=lr,
            clip=clip,
            restart_xml=initial_ffxml,
        )

        # the force field itself comes from initial_ffxml, which BaseTrainer
        # already loaded, so only the history is taken from the checkpoint
        trainer.losses = dump_dict["losses"]
        trainer.epochs = dump_dict["epochs"]

        # order is important
        trainer.ffxml = initial_ffxml
        trainer.opt_state = dump_dict["opt_state"]
        trainer._epoch = dump_dict["epoch"]
        trainer._restore_best(dump_dict)

        return trainer
