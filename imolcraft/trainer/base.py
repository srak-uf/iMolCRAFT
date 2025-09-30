from dmff import Hamiltonian
from dmff.optimize import MultiTransform, genOptimizer
from openmm import app
from openmm.app import ForceField, Modeller
import jax
import jax.numpy as jnp
import os
import optax
import time
from typing import List, Callable, Optional, Any, Union, Tuple
from imolcraft.crafter.ffxml import check_vsite
from imolcraft.crafter.ffxml import merge_xml
from imolcraft.trainer.dmff_utils import (
    get_chgparams_from_rescharges,
    get_rescharges_from_residues,
    vsiteinfo_to_params,
)


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
        early_stopping: bool = False,
        patience: int = 10,
        min_delta: float = 1e-6,
        convergence_window: int = 5,
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
        early_stopping : bool, optional
            Enable early stopping based on loss convergence (default: False).
        patience : int, optional
            Number of epochs to wait before stopping after convergence criteria is met (default: 10).
        min_delta : float, optional
            Minimum change in loss to qualify as improvement (default: 1e-6).
        convergence_window : int, optional
            Number of recent epochs to use for convergence analysis (default: 5).
        """
        if isinstance(ffxml_list, str):
            ffxml_list = [ffxml_list]

        assert len(ffxml_list) == len(
            nums_ffxml
        ), "Length of ffxml_list and nums_ffxml must be the same."

        if label is None:
            label = [
                f"{os.path.splitext(os.path.basename(ffxml))[0]}{num}"
                for ffxml, num in zip(ffxml_list, nums_ffxml)
            ]
            self.label = "_".join(label)
        else:
            self.label = label

        xmlfile = merge_xml(ffxml_list, f"{self.label}.xml")
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
        app.PDBFile.writeFile(self.topology, pos, open(vs_pdbfile, "w"))
        self.pdbfile_vsite = vs_pdbfile
        self.opt_fftypes = opt_fftypes

        ffparams = self.ff.getParameters().parameters
        self.rescharges, self.natoms_list = get_rescharges_from_residues(
            self.ff, ratio=nums_ffxml
        )
        if self.num_vsites > 0:
            ffparams = vsiteinfo_to_params(self.ff, ffparams)
        self.ffparams = get_chgparams_from_rescharges(ffparams, self.rescharges)

        self.loss_fn = loss_fn
        if isinstance(lr, list):
            assert len(lr) == len(
                opt_fftypes
            ), "Length of lr must match length of opt_fftypes."
            assert len(lr) == len(clip), "Length of lr must match length of clip."
        else:
            lr = [lr] * len(opt_fftypes)
            clip = [clip] * len(opt_fftypes)

        self.lr = lr
        self.clip = clip
        multiTrans = MultiTransform(self.ffparams)
        self.optimizer_algo = optimizer_algo
        for i, opt_fftype in enumerate(self.opt_fftypes):
            # multiTrans[opt_fftype] = genOptimizer(
            #     learning_rate=lr, clip=0.001, nonzero=False
            # )
            if opt_fftype == "NonbondedForce/charges":
                multiTrans[opt_fftype] = genOptimizer(
                    optimizer=self.optimizer_algo,
                    learning_rate=lr[i],
                    clip=self.clip[i],
                    nonzero=False,
                )
            else:
                multiTrans[opt_fftype] = genOptimizer(
                    optimizer=self.optimizer_algo,
                    learning_rate=lr[i],
                    clip=self.clip[i],
                    nonzero=False,
                )  # Should be True

        multiTrans.finalize()
        self.grad_transform = optax.multi_transform(
            multiTrans.transforms, multiTrans.labels
        )
        mask = jax.tree_util.tree_map(
            lambda x: x.dtype != jnp.int32 and x.dtype != int, self.ffparams
        )
        self.optimizer = optax.masked(self.grad_transform, mask)

        self.losses = []
        self.epochs = []
        self._modifyfns = {}
        # self._modifyfns["after_grad"]は与えられたgradをそのまま返す
        self._modifyfns["after_grad"] = lambda grads: grads
        self._modifyfns["after_update"] = lambda ffparams: ffparams

        # Early stopping parameters
        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta
        self.convergence_window = convergence_window
        self.best_loss_early_stopping = float('inf')  # Separate variable for early stopping
        self.patience_counter = 0
        self.converged = False

    def add_modifyfn(self, type_fn: str, fn: Callable[[Any], Any]) -> None:
        """
        Register a hook function to modify gradients or parameters.

        Parameters
        ----------
        type_fn : {"after_grad", "after_update"}
            Type of hook to register.
        fn : callable
            Function to be called after gradients or parameter update.
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
            Result of the hook function, or None if not registered.
        """
        if type_fn in self._modifyfns:
            return self._modifyfns[type_fn](*args, **kwargs)
        else:
            return

    def _check_convergence(self) -> bool:
        """
        Check if training has converged based on loss gradient and patience.
        
        Returns
        -------
        bool
            True if convergence criteria are met, False otherwise.
        """
        if not self.early_stopping or len(self.losses) < self.convergence_window:
            return False
            
        # Get recent losses for gradient calculation
        recent_losses = self.losses[-self.convergence_window:]
        
        # Calculate the slope (derivative) of loss over recent epochs
        # Using linear regression slope: (n*Σxy - ΣxΣy) / (n*Σx² - (Σx)²)
        n = len(recent_losses)
        x = list(range(n))  # epoch indices
        y = recent_losses   # loss values
        
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi * xi for xi in x)
        
        # Calculate slope (derivative)
        denominator = n * sum_x2 - sum_x * sum_x
        if abs(denominator) < 1e-12:  # Avoid division by zero
            slope = 0.0
        else:
            slope = (n * sum_xy - sum_x * sum_y) / denominator
        
        # Check if current loss is better than best loss
        current_loss = self.losses[-1]
        
        # Initialize best_loss_early_stopping if it's still infinity
        if self.best_loss_early_stopping == float('inf'):
            self.best_loss_early_stopping = current_loss
            self.patience_counter = 0
            return False
        
        if current_loss < self.best_loss_early_stopping - self.min_delta:
            self.best_loss_early_stopping = current_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
        
        # Check convergence criteria:
        # 1. Loss gradient (slope) magnitude is very small
        # 2. Patience counter has reached the limit
        slope_magnitude = abs(slope)
        gradient_converged = slope_magnitude < self.min_delta
        patience_exceeded = self.patience_counter >= self.patience
        
        if gradient_converged and patience_exceeded:
            print(f"Training converged: Loss gradient magnitude ({slope_magnitude:.2e}) < {self.min_delta} for {self.patience} epochs")
            return True
        elif patience_exceeded:
            print(f"Early stopping: No improvement for {self.patience} epochs")
            return True
            
        return False

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
        if jnp.isnan(self.loss) or jnp.isinf(self.loss):
            print("Warning: Loss is NaN or Inf. Skipping this step.")
            # self.ffparamsを0.01%ランダムにずらす
            self.ffparams = jax.tree_util.tree_map(
                lambda x: x
                + 0.0001 * jax.random.normal(jax.random.PRNGKey(1), shape=x.shape),
                self.ffparams,
            )
        else:
            grads = self._do_modify("after_grad", grads)
            updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
            # print("Updates: ", updates)
            self.ffparams = optax.apply_updates(self.ffparams, updates)
            self.ffparams = self._do_modify("after_update", self.ffparams)

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

        Parameters
        ----------
        steps : int
            Number of training steps (epochs).
        checkpoint_frequency : int, optional
            Frequency (in epochs) to write checkpoints (default: 100).
        """
        self.checkpoint_frequency = checkpoint_frequency
        start_epoch = self._epoch
        end_epoch = start_epoch + steps + 1
        
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
                
            self._epoch += 1
            end_time = time.time()
            
            print(f"Epoch {i_epoch} completed in {end_time - start_time:.2f} seconds.")
            print("Loss: ", self.loss)
            print(
                "Best Loss: ",
                min(self.losses),
                "at epoch ",
                self.epochs[self.losses.index(min(self.losses))],
            )
            
            # Check for convergence if early stopping is enabled
            if self._check_convergence():
                self.converged = True
                print(f"Training stopped early at epoch {i_epoch} due to convergence.")
                break
