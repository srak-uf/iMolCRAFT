from dmff import Hamiltonian
from dmff.optimize import MultiTransform, genOptimizer
from openmm import app
from openmm.app import ForceField, Modeller
import jax
import jax.numpy as jnp
import os
import pickle
import optax
import time
from typing import List, Callable, Optional, Any, Union, Tuple
from imolcraft.crafter.ffxml import check_vsite
from imolcraft.crafter.ffxml import merge_xml
from imolcraft.provenance import provenance_fields
from imolcraft.trainer.dmff_utils import (
    get_chgparams_from_rescharges,
    get_rescharges_from_residues,
    update_ffinfo_from_params,
    update_ffinfo_from_rescharges,
    update_rescharges_from_params,
)
import psutil
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


#: Relative size of the random nudge applied when the loss turns NaN or Inf.
_NAN_RECOVERY_SCALE = 0.0001


def _print_memory(tag: str) -> None:
    """Report the resident set size of this process, for tracking JAX leaks."""
    process = psutil.Process(os.getpid())
    print(f"{tag} Memory Usage: {process.memory_info().rss / 1024**2:.2f} MB")


def _broadcast_lr_clip(lr, clip, opt_fftypes):
    """
    Give every optimized parameter type its own learning rate and clip value.

    A single number is shared by all of them; a list has to line up with
    ``opt_fftypes``.
    """
    if isinstance(lr, list):
        if len(lr) != len(opt_fftypes):
            raise ValueError("Length of lr must match length of opt_fftypes.")
        if len(lr) != len(clip):
            raise ValueError("Length of lr must match length of clip.")
        return lr, clip
    return [lr] * len(opt_fftypes), [clip] * len(opt_fftypes)


def _build_optimizer(ffparams, opt_fftypes, optimizer_algo, lr, clip):
    """
    Build the masked optimizer updating only the requested parameter types.

    Integer parameters (periodicities and the like) are masked out, since they
    are not continuous and must not be moved by the gradient step.
    """
    multiTrans = MultiTransform(ffparams)
    for i, opt_fftype in enumerate(opt_fftypes):
        multiTrans[opt_fftype] = genOptimizer(
            optimizer=optimizer_algo,
            learning_rate=lr[i],
            clip=clip[i],
            nonzero=False,  # Should be True
        )
    multiTrans.finalize()

    grad_transform = optax.multi_transform(multiTrans.transforms, multiTrans.labels)
    mask = jax.tree_util.tree_map(
        lambda x: x.dtype != jnp.int32 and x.dtype != int, ffparams
    )
    return grad_transform, optax.masked(grad_transform, mask)


def _nan_recovery_gradients(ffparams):
    """
    Replace the gradients by a small random nudge.

    Used when the loss came out NaN or Inf: rather than stopping, the
    parameters are jittered so the next step starts from a different point.
    """
    return jax.tree_util.tree_map(
        lambda x: x
        + _NAN_RECOVERY_SCALE * jax.random.normal(jax.random.PRNGKey(1), shape=x.shape),
        ffparams,
    )


def plot_learning_curve(epochs, losses, label: str) -> None:
    """Write the loss history on linear, log-y and log-log axes."""
    for prefix, xscale, yscale in (
        ("", "linear", "linear"),
        ("logy_", "linear", "log"),
        ("logylogx_", "log", "log"),
    ):
        fig, ax = plt.subplots(1, 1, figsize=(3.25, 2.5))
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.plot(epochs, losses)
        if prefix == "":
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
        plt.tight_layout()
        if xscale == "linear":
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        fig.savefig(f"{prefix}{label}_learning_curve.png", bbox_inches="tight")
        plt.close(fig)


class BaseTrainer:
    """
    Base class for force field parameter optimization trainers.

    This class manages the setup, optimization, and checkpointing of force field
    parameters using differentiable molecular force fields and JAX-based optimizers.
    """
    def __init__(
        self,
        ffxml_list: Union[str, List[str]],
        nums_ffxml: List[int],
        pdbfile: str,
        loss_fn: Callable[..., float],
        opt_fftypes: List[str],
        label: Optional[str] = None,
        batch_size: int = 1,
        optimizer_algo: str = "adam",
        lr: Union[float, List[float]] = 0.0001,
        clip: Union[float, List[float]] = 0.1,
        restart_xml: Optional[str] = None
    ) -> None:

        """
        Initialize the BaseTrainer.

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
        restart_xml : str, optional
            Path to the XML file for restarting the training.
        """
        if isinstance(ffxml_list, str):
            self.ffxml_list = [ffxml_list]
        else:
            self.ffxml_list = ffxml_list

        self.nums_ffxml = nums_ffxml
        if len(self.ffxml_list) != len(self.nums_ffxml):
            raise ValueError("Length of ffxml_list and nums_ffxml must be the same.")

        if label is None:
            label = [
                f"{os.path.splitext(os.path.basename(ffxml))[0]}{num}"
                for ffxml, num in zip(self.ffxml_list, self.nums_ffxml)
            ]
            self.label = "_".join(label)
        else:
            self.label = label

        xmlfile = merge_xml(self.ffxml_list, f"{self.label}.xml")

        if restart_xml is not None:
            print(f"Restarting training from {restart_xml}...")
            xmlfile = restart_xml
        self.ffxml = xmlfile
        self.ff = Hamiltonian(self.ffxml)
        self.num_vsites = check_vsite(self.ffxml)
        self._epoch = 0

        self.pdb = app.PDBFile(pdbfile)
        self.pdbfile = pdbfile
        self.potentials = self.ff.createPotential(self.pdb.topology)
        if self.num_vsites > 0:
            modeller = Modeller(self.pdb.topology, self.pdb.positions)
            modeller.addExtraParticles(ForceField(self.ffxml))
            pos = modeller.getPositions()
            self.topology = modeller.topology
        else:
            self.topology = self.pdb.topology
            pos = self.pdb.positions

        vs_pdbfile = f"vs_{os.path.basename(self.pdbfile)}"
        with open(vs_pdbfile, "w") as f:
            app.PDBFile.writeFile(self.topology, pos, f)
        self.pdbfile_vsite = vs_pdbfile
        self.opt_fftypes = opt_fftypes

        ffparams = self.ff.getParameters().parameters
        self.rescharges, self.natoms_list = get_rescharges_from_residues(
            self.ff, ratio=self.nums_ffxml
        )
        self.ffparams = get_chgparams_from_rescharges(ffparams, self.rescharges)

        self.loss_fn = loss_fn
        self.lr, self.clip = _broadcast_lr_clip(lr, clip, opt_fftypes)
        self.optimizer_algo = optimizer_algo
        self.grad_transform, self.optimizer = _build_optimizer(
            self.ffparams, self.opt_fftypes, optimizer_algo, self.lr, self.clip
        )

        self.losses = []
        self.epochs = []
        # best-so-far snapshot, filled in by fit()
        self.best_params = None
        self.best_epoch = None
        self.best_loss = None
        self._modifyfns = {}
        # self._modifyfns["after_grad"] returns the given grads unchanged
        self._modifyfns["after_grad"] = lambda grads: grads
        self._modifyfns["after_update"] = lambda ffparams: ffparams

    def _best_checkpoint_fields(self) -> dict:
        """The best-so-far snapshot, for inclusion in a checkpoint."""
        return {
            "best_params": self.best_params,
            "best_epoch": self.best_epoch,
            "best_loss": self.best_loss,
        }

    def _restore_best(self, dump_dict: dict) -> None:
        """
        Restore the best-so-far snapshot from a checkpoint.

        Without it a restarted run forgets the best model of the previous one:
        ``fit`` compares against ``min(self.losses)``, which is restored, but
        ``best_params`` would stay None until the historical minimum is beaten
        again. Checkpoints written before this was stored simply carry None.
        """
        self.best_params = dump_dict.get("best_params")
        self.best_epoch = dump_dict.get("best_epoch")
        self.best_loss = dump_dict.get("best_loss")

    def add_modifyfn(self, type_fn: str, fn: Callable[[Any], Any]) -> None:
        """
        Register a hook function to modify gradients or parameters.

        The hook takes the value and **returns** the modified one; the caller
        assigns that return value. It must not rely on modifying its argument
        in place, and it must not return None.

        Parameters
        ----------
        type_fn : {"after_grad", "after_update"}
            Type of hook to register. "after_grad" receives the gradients
            before the optimizer step, "after_update" the parameters after it.
        fn : callable
            Function taking the value and returning the modified value.
        """
        types_modifyfn = [
            "after_grad",
            "after_update",
        ]
        if type_fn not in types_modifyfn:
            raise ValueError(f"Unknown hook type: {type_fn}")
        self._modifyfns[type_fn] = fn

    def _do_modify(self, type_fn: str, *args: Any, **kwargs: Any) -> Any:
        """
        Call the registered hook function if available.

        Parameters
        ----------
        type_fn : str
            Hook type.
        *args, **kwargs :
            Arguments to pass to the hook function.

        Returns
        -------
        Any
            Result of the hook function. With no hook registered the first
            argument is handed back unchanged, so that callers can always
            assign the result.
        """
        if type_fn in self._modifyfns:
            return self._modifyfns[type_fn](*args, **kwargs)
        return args[0] if args else None

    def get_loss_gradients(self) -> Tuple[Any, Any]:
        """
        Compute the loss and its gradients.

        Returns
        -------
        Any
            Loss value and gradients. (To be implemented in subclass)
        """
        pass

    def training_step(self) -> None:
        """
        Perform a single training step: compute loss, gradients, and update parameters.
        Handles NaN/Inf loss by perturbing parameters.
        """
        self.loss, grads = self.get_loss_gradients()
        _print_memory("grad obtained....")
        if jnp.isnan(self.loss) or jnp.isinf(self.loss):
            print("Warning: Loss is NaN or Inf. Skipping this step.")
            # Randomly perturb self.ffparams by 0.01%
            grads = _nan_recovery_gradients(self.ffparams)

        grads = self._do_modify("after_grad", grads)
        _print_memory("grad modify....")
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.ffparams = optax.apply_updates(self.ffparams, updates)
        self.ffparams = self._do_modify("after_update", self.ffparams)
        _print_memory("update finished....")

    def before_step(self) -> None:
        """
        This method is called before each training step.
        It can be overridden in subclasses to implement custom behavior.
        """
        pass

    def after_step(self) -> None:
        """
        This method is called after each training step.
        It can be overridden in subclasses to implement custom behavior.
        """
        pass

    def write_checkpoint(self, checkpoint_frequency: int) -> None:
        """
        This method writes the current state of the trainer to a checkpoint file.
        It can be overridden in subclasses to implement custom behavior.
        """
        pass

    def fit(self, steps: int, checkpoint_frequency: int = 100) -> None:
        """
        Run the training loop for a given number of steps.

        Calling it again continues from where the previous call stopped, so
        ``fit(10)`` twice runs the same 20 epochs as ``fit(20)`` once.

        Parameters
        ----------
        steps : int
            Number of training steps (epochs).
        checkpoint_frequency : int, optional
            Frequency (in epochs) to write checkpoints (default: 100).
        """
        self.checkpoint_frequency = checkpoint_frequency
        start_epoch = self._epoch
        end_epoch = start_epoch + steps
        for i_epoch in range(start_epoch, end_epoch):
            start_time = time.time()
            self.before_step()
            self.training_step()
            self.after_step()
            if len(self.losses) == 0 or self.loss < min(self.losses):
                self.best_params = self.ffparams
                self.best_epoch = self._epoch
                self.best_loss = self.loss
                self.ff.renderXML(f"{self.label}_best.xml")
            self.epochs.append(self._epoch)
            self.losses.append(self.loss)
            if i_epoch % checkpoint_frequency == 0:
                self.write_checkpoint(checkpoint_frequency)
                _print_memory(f"Epoch {i_epoch}, Loss: {self.loss},")
            self._epoch += 1
            end_time = time.time()
            print("Loss: ", self.loss)
            print(
                "Best Loss: ",
                min(self.losses),
                "at epoch ",
                self.epochs[self.losses.index(min(self.losses))],
            )
            print(f"Epoch {i_epoch} completed in {end_time - start_time:.2f} seconds.")
            print("----")

class SumTrainer(BaseTrainer):
    def __init__(self, trainer1, trainer2,
                 opt_fftypes: List[str],
                 weight: List[float] = [1.0, 1.0],
                 optimizer_algo: str = "adam",
                 lr: Union[float, List[float]] = 0.0001,
                 clip: Union[float, List[float]] = 0.1,
                 restart_xml: Optional[str] = None
                 ):
        self.trainer1 = trainer1
        self.trainer2 = trainer2
        self.weight = weight
        self.ffparams = jax.tree_util.tree_map(
                            lambda x, y: jnp.concatenate((x, y), axis=0),
                            self.trainer1.ffparams,
                            self.trainer2.ffparams,
                        )

        self.ffparams_mapping = jax.tree_util.tree_map(
            lambda x, y: jnp.concatenate(
                (jnp.zeros_like(x, dtype=int), jnp.ones_like(y, dtype=int)), axis=0
            ),
            self.trainer1.ffparams,
            self.trainer2.ffparams,
        )
        self.ffxml_list = self.trainer1.ffxml_list + self.trainer2.ffxml_list
        self.nums_ffxml = self.trainer1.nums_ffxml + self.trainer2.nums_ffxml

        self.label = f"{self.trainer1.label}_{self.trainer2.label}"

        xmlfile = merge_xml(self.ffxml_list, f"{self.label}.xml")
        if restart_xml is not None:
            print(f"Restarting training from {restart_xml}...")
            xmlfile = restart_xml
        self.ffxml = xmlfile
        self.ff = Hamiltonian(self.ffxml)

        ffparams = self.ff.getParameters().parameters
        self.rescharges, self.natoms_list = get_rescharges_from_residues(
            self.ff, ratio=self.nums_ffxml
        )
        self.ffparams = get_chgparams_from_rescharges(ffparams, self.rescharges)
        
        self._epoch = 0
        self.losses = []
        self.epochs = []
        self.best_params = None
        self.best_epoch = None
        self.best_loss = None
        self._modifyfns = {}
        self._modifyfns["after_grad"] = lambda grads: grads
        self._modifyfns["after_update"] = lambda ffparams: ffparams

        self.opt_fftypes = opt_fftypes
        self.lr, self.clip = _broadcast_lr_clip(lr, clip, opt_fftypes)
        self.optimizer_algo = optimizer_algo
        self.grad_transform, self.optimizer = _build_optimizer(
            self.ffparams, self.opt_fftypes, optimizer_algo, self.lr, self.clip
        )

    def setup(self) -> None:
        """
        Set up both sub-trainers and the joint optimizer.

        An optimizer state restored by :meth:`from_checkpoint` is kept, so that
        the momenta of the previous run survive a restart.
        """
        self.trainer1.setup()
        self.trainer2.setup()
        if getattr(self, "opt_state", None) is None:
            self.opt_state = self.optimizer.init(self.ffparams)
    
    @staticmethod
    def _substep(trainer, name):
        """
        Loss and gradients of one sub-trainer, with its own "after_grad" hook
        applied.

        The sub-trainer's own optimizer is deliberately left alone: the
        parameters are updated by :meth:`training_step` through the joint
        optimizer, from the concatenated and weighted gradients. Stepping the
        sub optimizer here would leave it holding momenta for updates that were
        never applied, computed from gradients that had not been weighted yet,
        and those momenta would then be written to the sub-trainer checkpoint.
        """
        loss, grads = trainer.get_loss_gradients()
        print(f"{name}: {loss}")
        if jnp.isnan(loss) or jnp.isinf(loss):
            print("Warning: Loss is NaN or Inf. Skipping this step.")
            grads = _nan_recovery_gradients(trainer.ffparams)

        return loss, trainer._do_modify("after_grad", grads)

    def get_loss_gradients(self):
        self.loss1, grads1 = self._substep(self.trainer1, "Loss1")
        self.loss2, grads2 = self._substep(self.trainer2, "Loss2")

        # merge trainer1 and trainer2 ffparams
        loss = self.weight[0] * self.loss1 + self.weight[1] * self.loss2
        grads1 = jax.tree_util.tree_map(lambda x: x * self.weight[0], grads1)
        grads2 = jax.tree_util.tree_map(lambda x: x * self.weight[1], grads2)
        grad = jax.tree_util.tree_map(
            lambda x, y: jnp.concatenate((x, y), axis=0), grads1, grads2
        )
        return loss, grad
    
    def before_step(self):
        self.trainer1.before_step()
        self.trainer2.before_step()

    def training_step(self):
        """
        Perform a single training step: compute loss, gradients, and update parameters.
        Handles NaN/Inf loss by perturbing parameters.
        """
        self.loss, grads = self.get_loss_gradients()
        if jnp.isnan(self.loss) or jnp.isinf(self.loss):
            print("Warning: Loss is NaN or Inf. Skipping this step.")
            # Randomly perturb self.ffparams by 0.01%
            grads = _nan_recovery_gradients(self.ffparams)

        grads = self._do_modify("after_grad", grads)
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.ffparams = optax.apply_updates(self.ffparams, updates)

        self._scatter_to_subtrainers()
        # the sub hooks run on their own half first, then the joint hook sees
        # the whole vector; both results have to be assigned to take effect
        self.trainer1.ffparams = self.trainer1._do_modify(
            "after_update", self.trainer1.ffparams
        )
        self.trainer2.ffparams = self.trainer2._do_modify(
            "after_update", self.trainer2.ffparams
        )
        self.ffparams = self._gather_from_subtrainers()

        self.ffparams = self._do_modify("after_update", self.ffparams)
        self._scatter_to_subtrainers()

        # save xml files for trainer1 and trainer2
        self.trainer1.after_step()
        self.trainer2.after_step()

        self.trainer1.epochs.append(self._epoch)
        self.trainer2.epochs.append(self._epoch)
        self.trainer1.losses.append(self.loss1)
        self.trainer2.losses.append(self.loss2)
        self.trainer1._epoch += 1
        self.trainer2._epoch += 1

    def _scatter_to_subtrainers(self) -> None:
        """Split the joined parameter vector back into the two sub-trainers."""
        for value, trainer in ((0, self.trainer1), (1, self.trainer2)):
            trainer.ffparams = jax.tree_util.tree_map(
                lambda params, mapping, v=value: params[mapping == v],
                self.ffparams,
                self.ffparams_mapping,
            )

    def _gather_from_subtrainers(self):
        """Join the parameters of the two sub-trainers into one vector."""
        return jax.tree_util.tree_map(
            lambda x, y: jnp.concatenate((x, y), axis=0),
            self.trainer1.ffparams,
            self.trainer2.ffparams,
        )

    def after_step(self) -> None:
        self.ff = update_ffinfo_from_params(self.ff, self.ffparams)
        self.rescharges = update_rescharges_from_params(
            self.rescharges, self.ffparams
        )
        self.ff = update_ffinfo_from_rescharges(self.ff, self.rescharges)
        self.ff.getParameters().parameters = self.ffparams
        os.makedirs("xmlfiles", exist_ok=True)
        self.ff.renderXML(f"xmlfiles/epoch_{self.label}-{self._epoch+1}.xml")
        self.ffxml = f"xmlfiles/epoch_{self.label}-{self._epoch+1}.xml"
    
    @classmethod
    def from_checkpoint(
        cls,
        trainer_checkpoint: str,
        trainer1,
        trainer2,
        opt_fftypes: Optional[List[str]] = None,
        weight: Optional[List[float]] = None,
        optimizer_algo: Optional[str] = None,
        lr: Optional[Union[float, List[float]]] = None,
        clip: Optional[Union[float, List[float]]] = None,
        restart_xml: Optional[str] = None,
    ) -> "SumTrainer":
        """
        Rebuild a SumTrainer from its own checkpoint.

        The two sub-trainers have to be restored first and passed in: they hold
        the calculators, inputs and estimators that this class does not
        duplicate. What only lives here is the joint optimizer state, which is
        why reading the two sub-checkpoints alone is not enough to continue a
        run: the momenta of the joint optimizer would restart from zero.

        Parameters
        ----------
        trainer_checkpoint : str
            Path of the pickle written by :meth:`write_checkpoint`.
        trainer1, trainer2 : BaseTrainer
            The sub-trainers, already restored from their own checkpoints.
        opt_fftypes, weight, optimizer_algo, lr, clip : optional
            Override the values stored in the checkpoint.
        restart_xml : str, optional
            Force field XML to resume from, instead of merging the original ones.

        Returns
        -------
        SumTrainer
        """
        with open(trainer_checkpoint, "rb") as f:
            dump_dict = pickle.load(f)

        trainer = cls(
            trainer1,
            trainer2,
            opt_fftypes=(
                dump_dict["opt_fftypes"] if opt_fftypes is None else opt_fftypes
            ),
            weight=dump_dict.get("weight", [1.0, 1.0]) if weight is None else weight,
            optimizer_algo=(
                dump_dict["optimizer_algo"]
                if optimizer_algo is None
                else optimizer_algo
            ),
            lr=dump_dict["lr"] if lr is None else lr,
            clip=dump_dict["clip"] if clip is None else clip,
            restart_xml=restart_xml,
        )

        trainer.ffparams = dump_dict["ffparams"]
        trainer.opt_state = dump_dict["opt_state"]
        trainer.ff.ffinfo = dump_dict["ffinfo"]
        trainer._epoch = dump_dict["epoch"]
        trainer.losses = dump_dict["losses"]
        trainer.epochs = dump_dict["epochs"]
        trainer._restore_best(dump_dict)
        # hand the restored parameters down, so the sub-trainers agree with the
        # joint vector before the first step
        trainer._scatter_to_subtrainers()
        return trainer

    def write_checkpoint(self, checkpoint_frequency: int) -> None:
        """
        Save the current training state and plots to a checkpoint file.

        The sub-trainers write their own checkpoints as well; both are needed
        to resume, see :meth:`from_checkpoint`.

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
                    "epoch": self._epoch,
                    "losses": self.losses,
                    "epochs": self.epochs,
                    "label": self.label,
                    "optimizer_algo": self.optimizer_algo,
                    "opt_fftypes": self.opt_fftypes,
                    "lr": self.lr,
                    "clip": self.clip,
                    "weight": self.weight,
                    **self._best_checkpoint_fields(),
                    **provenance_fields(),
                }
                pickle.dump(dump_dict, f)

            
            plot_learning_curve(self.epochs, self.losses, self.label)

            self.trainer1.write_checkpoint(checkpoint_frequency)
            self.trainer2.write_checkpoint(checkpoint_frequency)

