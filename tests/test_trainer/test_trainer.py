from imolcryff.trainer import DistanceTrainer, DihedralTrainer, ThermodynamicTrainer
from imolcryff.calculator import DistanceCalculator, DihedralCalculator
from imolcryff.trainer.loss import loss_energy
from imolcryff.crafter.asemol import aseatoms2pdb, asemol_wrapper, merge_asemols
from functools import partial
from ase import Atoms
from openmm.app import PDBFile
import tempfile
import os
import pytest


@pytest.mark.distance
class TestDistanceTrainer:
    @pytest.fixture
    def setup(self):
        atoms = Atoms("CH3CH3",
                      positions=[[-2.79452060, 1.06849313, 0.00000000],
                                 [-2.43786617, 0.05968313, 0.00000000],
                                 [-2.43784776, 1.57289132, -0.87365150],
                                 [-3.86452060, 1.06850632, 0.00000000],
                                 [-2.28117838, 1.79444941, 1.25740497],
                                 [-2.63623472, 2.80382250, 1.25642745],
                                 [-2.63944749, 1.29118098, 2.13105486],
                                 [-1.21118019, 1.79274272, 1.25838372]])
        aw = asemol_wrapper(atoms)
        molatoms, molecule_list, G_list = aw.get_ase_molecules(out_nX=True)
        atoms = merge_asemols(molatoms)
        bonds = aw.get_bonds()
        with tempfile.TemporaryDirectory() as td:
            temppdb = os.path.join(td, "temp.pdb")
            aseatoms2pdb(temppdb, atoms)
            pdb_omm = PDBFile(temppdb)
            atomlist_openmm = [a for a in pdb_omm.topology.atoms()]
            for b in bonds:
                a1 = atomlist_openmm[b[0]]
                a2 = atomlist_openmm[b[1]]
                pdb_omm.topology.addBond(a1, a2)
            PDBFile.writeFile(
                pdb_omm.topology, pdb_omm.positions, open(temppdb, "w")
            )
            self.pdbfile = temppdb
            self.calculator = DistanceCalculator(
                atoms,
                0,
                label="test",
                scan_idx=[[0, 1]],
                scan_ranges=[[1.06, 1.07, 1.08]],
                directory="test",
                qmparams={
                    "method": "hf",
                    "basis": "6-31g",
                    "opt": "modredundant"
                }
            )
            self.calculator.do_qmscan()
            self.ffxml = os.path.join(
                os.path.dirname(__file__),
                "..",
                "data",
                "ethane.xml"
            )
            self.calculator.do_ffscan(self.ffxml)
            yield

    @pytest.mark.parametrize("weight_scheme", ["uniform", "boltzmann"])  # , "nonboltzmann"])
    def test_fit(self, weight_scheme, setup):
        lossfn = partial(loss_energy, weight_scheme=weight_scheme)
        trainer = DistanceTrainer(
            ffxml_list=[self.ffxml],
            nums_ffxml=[1],
            pdbfile=self.pdbfile,
            calculator=self.calculator,
            loss_fn=lossfn,
            opt_fftypes=["HarmonicBondForce/k",
                         "HarmonicBondForce/length"],
            relax_steps=5,
            lr=0.002
        )
        trainer.setup()
        trainer.fit(steps=15, checkpoint_frequency=5)
        assert trainer.losses[-1] < trainer.losses[0], "Training did not reduce loss"

    def test_save_load(self, setup):
        lossfn = partial(loss_energy, weight_scheme="uniform")
        trainer = DistanceTrainer(
            ffxml_list=[self.ffxml],
            nums_ffxml=[1],
            pdbfile=self.pdbfile,
            calculator=self.calculator,
            loss_fn=lossfn,
            opt_fftypes=["HarmonicBondForce/k",
                         "HarmonicBondForce/length"],
            relax_steps=1,
            lr=0.002
        )
        trainer.setup()
        trainer.fit(steps=2, checkpoint_frequency=1)
        trainer = DistanceTrainer.from_checkpoint(
            trainer_checkpoint="train_state.pkl",
            ffxml_list=[self.ffxml],
            nums_ffxml=[1],
            pdbfile=self.pdbfile,
            loss_fn=lossfn,
            opt_fftypes=["HarmonicBondForce/k",
                         "HarmonicBondForce/length"],
        )
        trainer.fit(steps=2, checkpoint_frequency=1)


@pytest.mark.dihedral
class TestDihedralTrainer:
    @pytest.fixture
    def setup(self):
        atoms = Atoms("CH3CH3",
                      positions=[[-2.79452060, 1.06849313, 0.00000000],
                                 [-2.43786617, 0.05968313, 0.00000000],
                                 [-2.43784776, 1.57289132, -0.87365150],
                                 [-3.86452060, 1.06850632, 0.00000000],
                                 [-2.28117838, 1.79444941, 1.25740497],
                                 [-2.63623472, 2.80382250, 1.25642745],
                                 [-2.63944749, 1.29118098, 2.13105486],
                                 [-1.21118019, 1.79274272, 1.25838372]])
        aw = asemol_wrapper(atoms)
        molatoms, molecule_list, G_list = aw.get_ase_molecules(out_nX=True)
        atoms = merge_asemols(molatoms)
        bonds = aw.get_bonds()
        with tempfile.TemporaryDirectory() as td:
            temppdb = os.path.join(td, "temp.pdb")
            aseatoms2pdb(temppdb, atoms)
            pdb_omm = PDBFile(temppdb)
            atomlist_openmm = [a for a in pdb_omm.topology.atoms()]
            for b in bonds:
                a1 = atomlist_openmm[b[0]]
                a2 = atomlist_openmm[b[1]]
                pdb_omm.topology.addBond(a1, a2)
            PDBFile.writeFile(
                pdb_omm.topology, pdb_omm.positions, open(temppdb, "w")
            )
            self.pdbfile = temppdb
            self.calculator = DihedralCalculator(
                atoms,
                label="test",
                directory="test",
                qmparams={
                    "method": "hf",
                    "basis": "6-31g",
                    "opt": "modredundant"
                }
            )
            self.calculator.do_qmscan()
            self.ffxml = os.path.join(
                os.path.dirname(__file__),
                "..",
                "data",
                "ethane.xml"
            )
            self.calculator.do_ffscan(self.ffxml)
            yield

    @pytest.mark.parametrize("weight_scheme", ["uniform"])  # , "nonboltzmann"])
    def test_fit(self, weight_scheme, setup):
        lossfn = partial(loss_energy, weight_scheme=weight_scheme)
        trainer = DihedralTrainer(
            ffxml=self.ffxml,
            pdbfile=self.pdbfile,
            calculator=self.calculator,
            loss_fn=lossfn,
            opt_fftypes=["PeriodicTorsionForce/proper_k"],
            relax_steps=5,
            lr=0.01
        )
        trainer.setup()
        trainer.fit(steps=15, checkpoint_frequency=5)
        assert trainer.losses[-1] < trainer.losses[0], "Training did not reduce loss"
