from imolcryff.calculator import ChargeCalculator, Psi4ChargeCalculator
from ase import Atoms
import tempfile
import numpy as np
import pytest
import os


@pytest.mark.g16
class TestChargeCalculator:
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
        with tempfile.TemporaryDirectory() as tmpdir:
            self.calculator = ChargeCalculator(atoms,
                                               "resp",
                                               0,
                                               label="test",
                                               directory=tmpdir
                                               )
            yield
        
        if os.path.exists("fort.7"):
            os.remove("fort.7")

        if os.path.exists("timer.dat"):
            os.remove("timer.dat")

    def test_calculate_charge(self, setup):
        self.calculator.get_partialcharges()
        assert (
            np.isclose(np.sum(self.calculator.partial_charges), 0.0, atol=1e-5)
        )


class TestPsi4ChargeCalculator:
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
        with tempfile.TemporaryDirectory() as tmpdir:
            self.calculator = Psi4ChargeCalculator(atoms,
                                                   "resp",
                                                   0,
                                                   label="test",
                                                   directory=tmpdir
                                                   )
            yield

    def test_calculate_charge(self, setup):
        self.calculator.get_partialcharges()
        assert (
            np.isclose(np.sum(self.calculator.partial_charges), 0.0, atol=1e-5)
        )
