import pickle
from dmff import Hamiltonian
from dmff.optimize import MultiTransform, genOptimizer
from openmm import app
from openmm.app import  ForceField, Modeller
import jax
import jax.numpy as jnp
import os
import optax
import time
from ..crafter.ffxml import check_vsite
from ..trainer.dmff_utils import merge_xml, get_chgparams_from_rescharges, get_rescharges_from_residues, \
    vsiteinfo_to_params


class BaseExtension(object):
    pass


class BaseTrainer:
    def __init__(self,
                 ffxml_list,
                 nums_ffxml,
                 pdbfile,
                 loss_fn,
                 opt_fftypes,
                 label=None,
                 batch_size=1,
                 optimizer_algo="adam",
                 lr=0.0001,
                 clip=0.1
                 ):
        
        if isinstance(ffxml_list, str):
            ffxml_list = [ffxml_list]

        assert len(ffxml_list) == len(nums_ffxml), "Length of ffxml_list and nums_ffxml must be the same."

        if label is None:
            label = [f"{os.path.splitext(os.path.basename(ffxml))[0]}{num}" for ffxml, num in zip(ffxml_list, nums_ffxml)]
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
        self.opt_fftypes = opt_fftypes

        ffparams = self.ff.getParameters().parameters
        self.rescharges, self.natoms_list = get_rescharges_from_residues(self.ff, ratio=nums_ffxml)
        if self.num_vsites > 0:
            ffparams = vsiteinfo_to_params(self.ff, ffparams)
        self.ffparams = get_chgparams_from_rescharges(ffparams, self.rescharges)

        self.loss_fn = loss_fn
        if isinstance(lr, list):
            assert len(lr) == len(opt_fftypes), "Length of lr must match length of opt_fftypes."
            assert len(lr) == len(clip), "Length of lr must match length of clip."
        else:
            lr = [lr] * len(opt_fftypes)
            clip = [clip] * len(opt_fftypes)

        self.lr = lr
        self.clip = clip
        multiTrans = MultiTransform(self.ffparams)
        self.optimizer_algo = optimizer_algo
        for i, opt_fftype in enumerate(self.opt_fftypes):
            # multiTrans[opt_fftype] = genOptimizer(learning_rate=lr, clip=0.001, nonzero=False)
            if opt_fftype == "NonbondedForce/charges":
                multiTrans[opt_fftype] = genOptimizer(optimizer=self.optimizer_algo,
                                                      learning_rate=lr[i],
                                                      clip=self.clip[i],
                                                      nonzero=False)
            else:
                multiTrans[opt_fftype] = genOptimizer(optimizer=self.optimizer_algo,
                                                      learning_rate=lr[i],
                                                      clip=self.clip[i],
                                                      nonzero=False)  # Should be True

        multiTrans.finalize()
        self.grad_transform = optax.multi_transform(multiTrans.transforms, multiTrans.labels)
        mask = jax.tree_util.tree_map(lambda x: x.dtype != jnp.int32 and x.dtype != int, self.ffparams)
        self.optimizer = optax.masked(self.grad_transform, mask)

        self.losses = []
        self.epochs = []
        self._modifyfns = {}
        # self._modifyfns["after_grad"]は与えられたgradをそのまま返す
        self._modifyfns["after_grad"] = lambda grads: grads
        self._modifyfns["after_update"] = lambda ffparams: ffparams

    def add_modifyfn(self, type_fn, fn):
        types_modifyfn = [
            'after_grad',
            'after_update',
        ]
        if type_fn not in types_modifyfn:
            raise ValueError(f"Unknown hook type: {type_fn}")
        self._modifyfns[type_fn] = fn

    def _do_modify(self, type_fn, *args, **kwargs):
        if type_fn in self._modifyfns:
            return self._modifyfns[type_fn](*args, **kwargs)
        else:
            return

    def get_loss_gradients(self):
        pass

    def training_step(self):
        self.loss, grads = self.get_loss_gradients()
        grads = self._do_modify("after_grad", grads)
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        # print("Updates: ", updates)
        self.ffparams = optax.apply_updates(self.ffparams, updates)
        self.ffparams = self._do_modify("after_update", self.ffparams)

    def before_step(self):
        """
        This method is called before each training step.
        It can be overridden in subclasses to implement custom behavior.
        """
        pass
    
    def after_step(self):
        """
        This method is called after each training step.
        It can be overridden in subclasses to implement custom behavior.
        """
        pass

    def write_checkpoint(self, checkpoint_frequency):
        """
        This method writes the current state of the trainer to a checkpoint file.
        It can be overridden in subclasses to implement custom behavior.
        """
        pass
       
    def fit(self, steps, checkpoint_frequency=100):
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
            print("Best Loss: ", min(self.losses), "at epoch ", self.epochs[self.losses.index(min(self.losses))])

