from imolcryff.calculator import DihedralCalculator
from imolcryff.io.rdkit import atoms2rdkit
from ase import Atoms
import numpy as np
import os
import pytest


class TestDihedralCalculator:
    @pytest.fixture
    def setup(self):
        # Create a test molecule
        self.atoms = Atoms("CH3CH3",
                           positions=[[-2.79452060, 1.06849313, 0.00000000],
                                      [-2.43786617, 0.05968313, 0.00000000],
                                      [-2.43784776, 1.57289132, -0.87365150],
                                      [-3.86452060, 1.06850632, 0.00000000],
                                      [-2.28117838, 1.79444941, 1.25740497],
                                      [-2.63623472, 2.80382250, 1.25642745],
                                      [-2.63944749, 1.29118098, 2.13105486],
                                      [-1.21118019, 1.79274272, 1.25838372]])
        self.mol, _, _ = atoms2rdkit(self.atoms)
        self.dc = DihedralCalculator(self.atoms,
                                     label="test",
                                     directory="test",
                                     qmparams={
                                         "method": "hf",
                                         "basis": "6-31g",
                                         "opt": "modredundant"
                                     })
        # cr = Crafter()
        # cr.append_fromAtomsList([atoms], "ethane")
        # cr.get_partial_charges()
        # cr.params_ff = {
        #     "fftype": "gaff-2.11",
        # }
        # cr.get_ffxml()

    def test_calculate_dihedral_ff(self, setup):
        ffxml = os.path.join(
                os.path.dirname(__file__),
                "..",
                "data",
                "ethane.xml"
            )
        self.dc.do_ffscan(ffxml, ini_geom="FF")

    @pytest.mark.qm
    def test_calculate_dihedral_qm(self, setup):
        self.dc.do_qmscan()
        assert len(self.dc.qm_scan) == 1
        minidx = np.argmin(self.dc.qm_scan[0]["energy_kjmol"])
        minangle = self.dc.qm_scan[0]["angle_deg"][minidx]
        assert (
            np.isclose(np.abs(minangle), 60, atol=1.0)
            or np.isclose(np.abs(minangle), 180, atol=1.0)
        )
        maxidx = np.argmax(self.dc.qm_scan[0]["energy_kjmol"])
        maxangle = self.dc.qm_scan[0]["angle_deg"][maxidx]
        assert (
            np.isclose(np.abs(maxangle), 120, atol=1.0)
        )
