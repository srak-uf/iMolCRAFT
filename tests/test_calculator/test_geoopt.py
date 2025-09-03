from imolcryff.calculator import Psi4GeoOptimizer
from ase import Atoms
import numpy as np
import pytest


class TestPsi4GeoOptimizer:
    @pytest.fixture
    def setup(self):
        # Create a test molecule
        atoms = Atoms("CH3CH3",
                      positions=[[-2.79452060, 1.06849313, 0.00000000],
                                 [-2.43786617, 0.05968313, 0.00000000],
                                 [-2.43784776, 1.57289132, -0.87365150],
                                 [-3.86452060, 1.06850632, 0.00000000],
                                 [-2.28117838, 1.79444941, 1.25740497],
                                 [-2.63623472, 2.80382250, 1.25642745],
                                 [-2.63944749, 1.29118098, 2.13105486],
                                 [-1.21118019, 1.79274272, 1.25838372]])
        self.calculator = Psi4GeoOptimizer(atoms,
                                           method="hf",
                                           basis_set="6-31g",
                                           charge=0)

    def test_calculate(self, setup):
        atoms_prev = self.calculator.atoms.copy()
        self.calculator.calculate()
        assert not np.array_equal(atoms_prev.positions, self.calculator.atoms.positions)
