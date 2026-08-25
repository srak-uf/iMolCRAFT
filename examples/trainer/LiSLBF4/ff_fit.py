#!/usr/bin/env python
from imolcraft.trainer import ThermodynamicTrainer
from imolcraft.trainer.dmff_utils import parser_dmffyaml
from imolcraft.trainer.loss import loss_thermodynamicperturbation
from functools import partial
from imolcraft.trainer.dmff_utils import neutralize


def grad_modify(grads):
    grads["NonbondedForce"]["charge"] = grads["NonbondedForce"]["charge"].at[20].set(0.0)
    return grads


def ffparams_modify(ffparams):
    ffparams = neutralize(
        ffparams, trainer.natoms_list,
        target_lists=[[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                       18, 19],
                      [0, 1, 2, 3, 4]],
        target_charges=[0.0, -0.8])
    return ffparams


params = parser_dmffyaml("dmff.yml")
lossfn = partial(loss_thermodynamicperturbation, losstype_distribfn="wrightfactor")

trainer = ThermodynamicTrainer(
    ffxml_list=["BF4_bond.xml", "SL.xml", "Li.xml"],
    nums_ffxml=[1, 1, 1],
    pdbfile="supercell_bonds.pdb",
    loss_fn=lossfn,
    sampling_params=params["sampling"],
    target_params=params["targets"],
    validation_params=params["validation"],
    opt_fftypes=["NonbondedForce/charge",
                 "NonbondedForce/epsilon",
                 "NonbondedForce/sigma"],
    label="lbs_opt",
)

trainer.add_modifyfn("after_grad", grad_modify)
trainer.add_modifyfn("after_update", ffparams_modify)

trainer.setup()
trainer.fit(500, 2)
