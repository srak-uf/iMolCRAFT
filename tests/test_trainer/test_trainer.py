from imolcraft.trainer import DistanceTrainer, DihedralTrainer, ThermodynamicTrainer
from imolcraft.calculator import DistanceCalculator, DihedralCalculator
from imolcraft.trainer.loss import loss_energy, loss_thermodynamicperturbation
from imolcraft.trainer import dmff_utils
from imolcraft.crafter.asemol import aseatoms2pdb, asemol_wrapper, merge_asemols
from imolcraft.crafter import Crafter
from functools import partial
from ase import Atoms
from ase.io import write
from openmm.app import PDBFile
import tempfile
import os
import pickle
import pytest


@pytest.mark.g16
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
                directory=td,
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

@pytest.mark.g16
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


@pytest.mark.tp
class TestThermodynamicTrainer:
    @pytest.fixture
    def setup(self):
        atoms = Atoms("NCCCCNHHHH",
                      positions=[
                          [7.465626200000001, 6.932280800000000, 3.863675340000000],
                          [8.141680800000000, 7.499489400000000, 3.125223730000001],
                          [8.961653999999999, 8.221501200000001, 2.180166530000001],
                          [9.919498800000000, 9.179346000000001, 2.846733470000001],
                          [10.641510600000000, 9.999319200000000, 1.901676270000001],
                          [11.208719200000001, 10.675373799999999, 1.163224660000000],
                          [9.413715630000000, 9.759253299999999, 3.452771510000000],
                          [10.556731660000001, 8.665000270000000, 3.383053430000000],
                          [8.381746700000001, 8.727284370000000, 1.574128489999999],
                          [9.475999730000000, 7.584268339999999, 1.643846570000000]
                      ],
                      cell=[40, 40, 40],
                      pbc=True)
        write("sn.cif", atoms)
        cr = Crafter()
        cr.params_geoopt = {}
        cr.params_geoopt['basis'] = '6-31g'
        cr.params_geoopt['method'] = 'hf'
        cr.params_geoopt['software'] = 'psi4'
        cr.params_charge = {}
        cr.params_charge['type'] = 'resp'
        cr.params_charge['basis'] = '6-31g'
        cr.params_charge['method'] = 'hf'
        cr.params_charge['software'] = 'psi4'
        cr.params_ff = {}
        cr.params_ff['fftype'] = 'gaff-2.11'
        cr.structure = {}
        cr.structure['type'] = 'crystal'
        cr.structure['repeat'] = [1, 1, 1]
        cr.structure['cif'] = "sn.cif"
        cr.prep()
        cr.build()

        self.ffxml = os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "data",
                    "vsite_average2.xml"
                )

    def test_setup(self, setup):
        lossfn = partial(
            loss_thermodynamicperturbation, losstype_distribfn="wrightfactor"
        )
        self.trainer = ThermodynamicTrainer(
                            ffxml_list=[self.ffxml],
                            nums_ffxml=[1],
                            pdbfile="supercell_bonds.pdb",
                            loss_fn=lossfn,
                            sampling_params={'init_structure': 'supercell_bonds.pdb',
                                             'ensemble': 'nvt',
                                             'dt_fs': 1.0,
                                             'rcut_nm': 1.2,
                                             'temperature_K': 233.15,
                                             'pressure_bar': 1.0,
                                             'anneal_T': None,
                                             'anneal_steps': None,
                                             'anneal_interval': 100,
                                             'relax_steps': 100,
                                             'prod_steps': 100,
                                             'nstxout': 20,
                                             'neff': 2,
                                             'dispcorr': False,
                                             'nonbondedmethod': 'LJPME'},
                            target_params={'density_gcm3': {'weight': 1.0, 'gt': 0.4},
                                           'La_A': {'weight': 1.0, 'gt': 41},
                                           'Lb_A': {'weight': 1.0, 'gt': 41},
                                           'Lc_A': {'weight': 1.0, 'gt': 41}},
                            validation_params=dmff_utils._check_validation(
                                {'dself_C': {'property': 'dself_cm2s',
                                             'select': 'element C',
                                             'gt': 5.0e-6},
                                 'rho': {'property': 'density_gcm3',
                                         'gt': 0.4}}
                            ),
                            opt_fftypes=["NonbondedForce/charge", 'VirtualSite/weight'],
                            label="test_tp",
                        )
        self.trainer.setup()
        self.trainer.fit(10, 2)
        assert len(self.trainer.losses) > 1
        # validation gains one record per resampling
        assert len(self.trainer.validation_history) > 0
        assert all(
            'sample_0/dself_C' in record and 'sample_0/rho' in record
            for record in self.trainer.validation_history
        )
        # An item with a gt keeps its deviation as well, split per item
        assert len(self.trainer.validation_dev_history) == len(
            self.trainer.validation_history
        )
        assert all(
            'sample_0/dself_C' in record and 'sample_0/rho' in record
            for record in self.trainer.validation_dev_history
        )
        # The recorded epoch is the xml number of the force field the value came from.
        # Only the setup record is 0 (initial ff); the rest are unique and in range
        epochs = [record['epoch'] for record in self.trainer.validation_history]
        assert epochs[0] == 0
        assert epochs == sorted(set(epochs))
        assert epochs[-1] <= self.trainer._epoch
        assert all(
            os.path.exists(f"xmlfiles/epoch_test_tp-{epoch}.xml")
            for epoch in epochs[1:]
        )
        # The force field itself is recorded too, so it matches up with the number
        assert all(
            record['ffxml'] == f"xmlfiles/epoch_test_tp-{record['epoch']}.xml"
            for record in self.trainer.validation_history[1:]
        )
        # The deviation is the default relerr, i.e. (pred - gt) / gt
        assert self.trainer.validation_dev[0]['rho'] == pytest.approx(
            (self.trainer.validation_pred[0]['rho'] - 0.4) / 0.4
        )
        # The MSD curve stays in the pkl, so the fit range can be checked later
        assert self.trainer.validation_curves[0]['dself_C'].shape[1] == 2
        # The target history has one record per epoch, on the ff of that epoch,
        # and rides in the checkpoint
        history = self.trainer.target_history
        assert [record['epoch'] for record in history] == list(range(10))
        assert all(
            record['ffxml'] == f"xmlfiles/epoch_test_tp-{record['epoch']}.xml"
            for record in history[1:]
        )
        assert [float(record['loss']) for record in history] == pytest.approx(
            [float(loss) for loss in self.trainer.losses]
        )
        assert history[0]['sample_0/resampled'] is True
        assert 'sample_0/density_gcm3' in history[-1]
        with open("train_state_test_tp.pkl", "rb") as f:
            dump = pickle.load(f)
        assert dump['target_log'] == 'medium'
        assert [record['epoch'] for record in dump['target_history']] == list(range(9))
