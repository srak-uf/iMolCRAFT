from imolcryff.calculator import DistanceCalculator
from ase import Atoms
import numpy as np
import os
import tempfile
import pytest


@pytest.mark.g16
class TestDistanceCalculator:
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
        with tempfile.TemporaryDirectory() as tmpdir:
            self.dc = DistanceCalculator(self.atoms,
                                         0,
                                         label="test",
                                         scan_idx=[[0, 1]],
                                         scan_ranges=[[1.06, 1.07, 1.08, 1.09]],
                                         directory=tmpdir,
                                         qmparams={
                                             "method": "hf",
                                             "basis": "6-31g",
                                             "opt": "modredundant"
                                         })
            yield

        if os.path.exists("fort.7"):
            os.remove("fort.7")

        if os.path.exists("timer.dat"):
            os.remove("timer.dat")

    def test_calculate_distance_ff(self, setup):
        ffxml = os.path.join(
                os.path.dirname(__file__),
                "..",
                "data",
                "ethane.xml"
            )
        self.dc.do_ffscan(ffxml, ini_geom="FF")
        assert len(self.dc.ff_scan) == 1
        minidx = np.argmin(self.dc.ff_scan[0]["energy_kjmol"])
        mindistance = self.dc.ff_scan[0]["distance_A"][minidx]
        assert (
            np.isclose(np.abs(mindistance), 1.09)
        )
        maxidx = np.argmax(self.dc.ff_scan[0]["energy_kjmol"])
        maxdistance = self.dc.ff_scan[0]["distance_A"][maxidx]
        assert (
            np.isclose(np.abs(maxdistance), 1.06)
        )

    @pytest.mark.qm
    def test_calculate_distance_qm(self, setup):
        self.dc.do_qmscan()
        assert len(self.dc.qm_scan) == 1
        minidx = np.argmin(self.dc.qm_scan[0]["energy_kjmol"])
        mindistance = self.dc.qm_scan[0]["distance_A"][minidx]
        assert (
            np.isclose(np.abs(mindistance), 1.08)
        )

        maxidx = np.argmax(self.dc.qm_scan[0]["energy_kjmol"])
        maxdistance = self.dc.qm_scan[0]["distance_A"][maxidx]
        assert (
            np.isclose(np.abs(maxdistance), 1.06)
        )
