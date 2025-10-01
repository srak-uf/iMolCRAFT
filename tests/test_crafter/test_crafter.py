from imolcraft.crafter.molinfo import Crafter
import os
import numpy as np
from ase import Atoms
from ase.io import write
import pytest


class TestCrafter_crystal:
    @pytest.fixture
    def init_crafter(self):
        self.crafter = Crafter()

    def test_loadyaml(self, init_crafter):
        self.crafter.from_yaml(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "data",
                "craft_params.yml"
            )
        )
        ffparams = self.crafter.params_geoopt
        params_charge = self.crafter.params_charge
        params_ff = self.crafter.params_ff
        structure = self.crafter.structure
        assert ffparams['basis'] == '6-311+g(2d,p)'
        assert ffparams['method'] == 'wb97xd'
        assert ffparams['software'] == 'psi4'
        assert params_charge['type'] == 'resp'
        assert params_charge['basis'] == '6-31g(d)'
        assert params_charge['method'] == 'hf'
        assert params_charge['software'] == 'psi4'
        assert params_ff['fftype'] == 'gaff-2.11'
        assert params_ff['iontype'] == 'amber/ions/ionsff99_tip3p.xml'
        assert params_ff['charge_scale_ion'] == 0.8
        assert params_ff['charge_scale_neutral'] == 1.0
        assert structure['type'] == 'crystal'
        assert structure['repeat'] == [1, 1, 2]

    def test_prep_woqm(self, init_crafter):
        import tempfile
        import shutil
        
        # Create a temporary directory for this test
        test_dir = tempfile.mkdtemp()
        original_dir = os.getcwd()
        
        try:
            os.chdir(test_dir)
            
            self.crafter.from_yaml(
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "data",
                    "craft_params.yml"
                )
            )
            self.crafter.prep(do_opt=False, do_charge=False)
            assert len(self.crafter.molatoms) == 8
            assert self.crafter.mol_info["MOL_0"]["netcharge"] == -1  # BH4-
            assert self.crafter.mol_info["MOL_1"]["netcharge"] == 1  # Li+
            
            # Check that PDB files with bonds were created
            assert os.path.exists("MOL_0_bonds.pdb"), "MOL_0_bonds.pdb was not created"
            assert os.path.exists("MOL_1_bonds.pdb"), "MOL_1_bonds.pdb was not created"
            
            # Check that the files contain CONECT records
            with open("MOL_0_bonds.pdb", 'r') as f:
                content = f.read()
                assert 'CONECT' in content, "MOL_0_bonds.pdb does not contain CONECT records"
                
        finally:
            os.chdir(original_dir)
            shutil.rmtree(test_dir)

    @pytest.mark.qm
    def test_prep_and_build(self, init_crafter):
        self.crafter.from_yaml(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "data",
                "craft_params.yml"
            )
        )
        self.crafter.prep(do_opt=True, do_charge=True)
        assert len(self.crafter.mol_info["MOL_0"]["partial_charges"]) == 5
        MOL0_charges = self.crafter.mol_info["MOL_0"]["partial_charges"]
        assert np.isclose(np.sum(MOL0_charges), -0.8)
        assert np.isclose(MOL0_charges[0], 0.1108411888503414, atol=1e-4)
        assert np.isclose(MOL0_charges[1], MOL0_charges[2], atol=1e-4)
        assert np.isclose(MOL0_charges[1], MOL0_charges[3], atol=1e-4)
        assert np.isclose(MOL0_charges[1], MOL0_charges[4], atol=1e-4)

        self.crafter.build()


class TestCrafter_liquid:
    @pytest.fixture
    def init_crafter(self):
        self.crafter = Crafter()
        self.crafter.from_yaml(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "data",
                "craft_params.yml"
            )
        )
        self.crafter.structure = {}
        self.crafter.structure["type"] = "liquid"
        self.crafter.structure["molecules"] = ["CH4.pdb"]
        atoms = Atoms('CH4', [(-1.397, 1.740, 0.000),
                              (-1.041, 0.731, 0.000),
                              (-1.041, 2.244, 0.874),
                              (-1.041, 2.244, -0.874),
                              (-2.467, 1.740, 0.000)])
        write("CH4.pdb", atoms)

    @pytest.mark.qm
    def test_prep_liq(self, init_crafter):
        self.crafter.structure["fixed_property"] = "num_mols"
        self.crafter.structure["priority_property"] = "cell"
        self.crafter.structure["nmols"] = [5]
        self.crafter.structure["density_kgm3"] = 0.5
        self.crafter.structure["cell_A"] = [10.0, 10.0, 10.0]
        self.crafter.prep(do_opt=False, do_charge=True)
        self.crafter.build()


class TestCrafter_function():
    @pytest.fixture
    def init_crafter(self):
        self.crafter = Crafter()
        self.atoms = Atoms("CH3CH3",
                           positions=[[-2.79452060, 1.06849313, 0.00000000],
                                      [-2.43786617, 0.05968313, 0.00000000],
                                      [-2.43784776, 1.57289132, -0.87365150],
                                      [-3.86452060, 1.06850632, 0.00000000],
                                      [-2.28117838, 1.79444941, 1.25740497],
                                      [-2.63623472, 2.80382250, 1.25642745],
                                      [-2.63944749, 1.29118098, 2.13105486],
                                      [-1.21118019, 1.79274272, 1.25838372]])

    def test_append_fromAtomsList(self, init_crafter):
        self.crafter.append_fromAtomsList([self.atoms], "ethane")
        assert "ethane" in self.crafter.mol_info
        assert self.crafter.mol_info["ethane"]["aseatoms_list"][0] == self.atoms

    def test_get_partial_charges(self, init_crafter):
        self.crafter.append_fromAtomsList([self.atoms], "ethane")
        self.crafter.get_partial_charges(software="psi4")
        assert len(self.crafter.mol_info["ethane"]["partial_charges"]) == 8
        assert np.isclose(
            np.sum(self.crafter.mol_info["ethane"]["partial_charges"]), 0.0, atol=1e-4)

    def test_get_ffxml(self, init_crafter):
        self.crafter.append_fromAtomsList([self.atoms], "ethane")
        self.crafter.params_ff = {
            "fftype": "gaff-2.11"
        }
        self.crafter.get_partial_charges(software="psi4")
        self.crafter.get_ffxml()

    def test_save_molecule_pdbs_with_bonds(self, init_crafter):
        """Test that PDB files with bond information are created"""
        import tempfile
        import shutil
        
        # Create a temporary directory for this test
        test_dir = tempfile.mkdtemp()
        original_dir = os.getcwd()
        
        try:
            os.chdir(test_dir)
            
            # Add molecule to crafter
            self.crafter.append_fromAtomsList([self.atoms], "ethane")
            
            # Save PDB with bonds
            self.crafter.save_molecule_pdbs_with_bonds()
            
            # Check that the file was created
            expected_file = "ethane_bonds.pdb"
            assert os.path.exists(expected_file), f"{expected_file} was not created"
            
            # Check that the file contains CONECT records
            with open(expected_file, 'r') as f:
                content = f.read()
                assert 'CONECT' in content, f"{expected_file} does not contain CONECT records"
                
                # Count CONECT records - should have bonds for the molecule
                conect_lines = [line for line in content.split('\n') if line.startswith('CONECT')]
                assert len(conect_lines) > 0, "No CONECT records found"
                
        finally:
            os.chdir(original_dir)
            shutil.rmtree(test_dir)
