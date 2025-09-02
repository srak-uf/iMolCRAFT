from imolcryff.crafter.asemol import (
    ase_atoms_to_nx,
    asemol_wrapper,
    pdb2packmol
)
from imolcryff.io.mol2 import mol2_to_aseatoms
import os
import pytest
import numpy as np
import ase


@pytest.mark.parametrize(
    "mol2file, n_edge, n_node",
    [
        (os.path.join(
            os.path.dirname(__file__),
            "..",
            "data",
            "fsa_resp.mol2"), 8, 9),
        (os.path.join(
            os.path.dirname(__file__),
            "..",
            "data",
            "pf6_resp.mol2"), 6, 7)
    ]
)
def test_ase_atoms_to_nx(mol2file, n_edge, n_node):
    atoms = mol2_to_aseatoms(mol2file)
    g = ase_atoms_to_nx(atoms)
    assert g.number_of_edges() == n_edge
    assert g.number_of_nodes() == n_node


@pytest.mark.parametrize(
    "fixed_property, priority_property",
    [
        ("num_mols", "cell"),
        ("num_mols", "density"),
        ("density", "cell"),
        ("cell", "density")
    ]
)
def test_pdb2packmol(fixed_property, priority_property):
    atoms = ase.Atoms('CH4', [(-1.397, 1.740, 0.000),
                      (-1.041, 0.731, 0.000),
                      (-1.041, 2.244, 0.874),
                      (-1.041, 2.244, -0.874),
                      (-2.467, 1.740, 0.000)])
    ase.io.write("CH4.pdb", atoms)
    pdb2packmol(
        ["CH4.pdb"],
        fixed_property=fixed_property,
        priority_property=priority_property,
        num_mols=[6],
        cell=[10, 10, 10],
        density=500,
        outfile=f"CH4_{fixed_property}_{priority_property}.pdb"
    )


class TestAsemol_Wrapper:
    @pytest.fixture
    def init_asemol_wrapper(self):
        atoms = mol2_to_aseatoms(os.path.join(
            os.path.dirname(__file__),
            "..",
            "data",
            "fsa_resp.mol2"))
        atoms.cell = [[10, 0, 0], [0, 10, 0], [0, 0, 10]]
        atoms.pbc = True
        atoms = atoms.repeat((2, 2, 2))
        atoms = atoms[:-2]
        self.aw = asemol_wrapper(atoms)

    def test_get_ase_molecules(self, init_asemol_wrapper):
        molatoms, _ = self.aw.get_ase_molecules()
        assert len(molatoms) == 8
        assert len(molatoms[0]) == 9
        assert len(molatoms[-1]) == 7
        assert "residuenumbers" in molatoms[0].arrays

    def test_get_bonds(self, init_asemol_wrapper):
        bonds = self.aw.get_bonds()
        assert len(bonds) == 62

    def test_get_molecules(self, init_asemol_wrapper):
        molecules = self.aw.get_molecules()
        assert len(molecules) == 8
        assert molecules[0] == [i for i in range(len(molecules[0]))]

    def unwrap_molecules(self, init_asemol_wrapper):
        atoms_shift = self.aw.atoms.copy()
        for a in atoms_shift:
            a.position -= 2
        atoms_shift.wrap()
        self.aw.atoms = atoms_shift
        unwrap_atoms = self.aw.unwrap_molecules()
        # unwrap_atoms.positionsの一部がセルの外に出ていることを確認
        assert np.any(unwrap_atoms.positions < 0)
        assert np.any(unwrap_atoms.positions > 10)
