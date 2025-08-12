from dmff import Hamiltonian, DMFFTopology
from dmff.common.nblist import NoCutoffNeighborList
from dmff.generators.classical import PeriodicTorsionGenerator
from dmff.api.paramset import ParamSet
from dmff.api.xmlio import XMLIO
from dmff.api.hamiltonian import Potential
from dmff.optimize import MultiTransform, genOptimizer
from dmff.mbar import MBAREstimator, Sample, OpenMMSampleState
from openmm import app
from openmm.app import NoCutoff, Simulation, PDBFile, ForceField, Modeller
import copy, shutil
import numpy as np
import jax
import jax.numpy as jnp
import os
import mdtraj as md
from jax import value_and_grad, jit
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


