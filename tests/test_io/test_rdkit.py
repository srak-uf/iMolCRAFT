import numpy as np
import pytest
from ase import Atoms
from rdkit import Chem

from imolcraft.io.rdkit import _il_assign, _monatomic_smiles, atoms2rdkit


@pytest.mark.parametrize(
    "smiles, nc, expected_keys",
    [
        ("[F][P-](F)(F)(F)(F)F", -1, {"PF6_P", "PF6_F"}),
        ("[O-][Cl+3]([O-])([O-])[O-]", -1, {"ClO4_Cl", "ClO4_O"}),
        ("[O]=[S](=[O])([F])[N-][S](=[O])(=[O])[F]", -1, {"FSA_N", "FSA_S", "FSA_O"}),
    ],
)
def test_il_assign_detects_known_ions(smiles, nc, expected_keys):
    il_dict = _il_assign(Chem.MolFromSmiles(smiles), nc)
    assert il_dict is not None
    assert set(il_dict) == expected_keys
    assert all(len(v) > 0 for v in il_dict.values())


@pytest.mark.parametrize(
    "smiles, nc",
    [
        ("CC", 0),
        ("CC", -1),
        ("[F][P-](F)(F)(F)(F)F", 0),  # an anion is not classified when nc >= 0
        ("[O-]S(=O)(=O)C", -1),  # has S, but not the N-S backbone of FSA
    ],
)
def test_il_assign_returns_none_for_others(smiles, nc):
    assert _il_assign(Chem.MolFromSmiles(smiles), nc) is None


@pytest.mark.parametrize(
    "smiles, nc",
    [
        ("CC", -1),
        # Anions containing P but not PF6. Rewriting before the check puts -1 on P,
        # and RDKit grows implicit hydrogens to fix the valence, changing the formula
        ("[O-]P(=O)(O)O", -1),          # dihydrogen phosphate ion
        ("[O-]P(=O)(F)F", -1),          # difluorophosphate ion
        ("[O-]P(=O)([O-])[O-]", -3),    # phosphate ion
        # Four O on Cl, but one is an ester oxygen, so this is not ClO4
        ("CO[Cl+3]([O-])([O-])[O-]", -1),
    ],
)
def test_il_assign_leaves_unmatched_molecule_untouched(smiles, nc):
    """A molecule matching none of the ions is not rewritten at all"""
    mol = Chem.MolFromSmiles(smiles)
    before_charges = [a.GetFormalCharge() for a in mol.GetAtoms()]
    before_bonds = [str(b.GetBondType()) for b in mol.GetBonds()]
    before_smiles = Chem.MolToSmiles(mol)

    assert _il_assign(mol, nc) is None

    assert [a.GetFormalCharge() for a in mol.GetAtoms()] == before_charges
    assert [str(b.GetBondType()) for b in mol.GetBonds()] == before_bonds
    assert Chem.MolToSmiles(mol) == before_smiles


@pytest.mark.parametrize(
    "smiles, nc, expected",
    [
        ("[F][P-](F)(F)(F)(F)F", -1, {"PF6_P": [1], "PF6_F": [0, 2, 3, 4, 5, 6]}),
        ("[O-][Cl+3]([O-])([O-])[O-]", -1,
         {"ClO4_Cl": [1], "ClO4_O": [0, 2, 3, 4]}),
    ],
)
def test_il_assign_index_layout(smiles, nc, expected):
    """The central atom and ligand indices come back in ascending order"""
    assert _il_assign(Chem.MolFromSmiles(smiles), nc) == expected


def test_central_ion_indices_requires_terminal_ligands():
    from imolcraft.io.rdkit import _central_ion_indices

    bare = Chem.MolFromSmiles("[O-][Cl+3]([O-])([O-])[O-]")
    assert _central_ion_indices(bare.GetAtoms(), "Cl", "O", 4) == (1, [0, 2, 3, 4])

    # The ester oxygen is bonded to more than Cl, so it is not terminal
    ester = Chem.MolFromSmiles("CO[Cl+3]([O-])([O-])[O-]")
    assert _central_ion_indices(ester.GetAtoms(), "Cl", "O", 4) is None


@pytest.mark.parametrize(
    "symbol, nc, expected",
    [("Li", 1, "[Li+1]"), ("Cl", -1, "[Cl-1]"), ("Na", 0, "[Na]"),
     ("Mg", 2, "[Mg+2]")],
)
def test_monatomic_smiles(symbol, nc, expected):
    assert _monatomic_smiles(symbol, nc) == expected
    assert Chem.MolFromSmiles(expected) is not None


@pytest.mark.parametrize("symbol, nc", [("Li", 1), ("Cl", -1), ("Mg", 2)])
def test_atoms2rdkit_single_atom(symbol, nc):
    mol, mol2d, il_dict = atoms2rdkit(Atoms(symbol, positions=[[0, 0, 0]]), nc=nc)
    assert mol.GetNumAtoms() == 1
    assert mol.GetAtomWithIdx(0).GetFormalCharge() == nc
    assert mol2d is mol
    assert il_dict is None


@pytest.mark.parametrize(
    "nc", [1, np.int64(1), np.int32(1), np.int8(1), np.uint8(1)]
)
def test_atoms2rdkit_accepts_numpy_integers(nc):
    """numpy integers work too: the RDKit bindings reject them, so cast to int"""
    mol, _, _ = atoms2rdkit(Atoms("Li", positions=[[0, 0, 0]]), nc=nc)
    assert mol.GetAtomWithIdx(0).GetFormalCharge() == 1


def test_atoms2rdkit_accepts_numpy_integers_for_polyatomic():
    from ase.build import molecule as ase_molecule

    atoms = ase_molecule("CH4")
    ref, _, _ = atoms2rdkit(atoms.copy(), nc=0)
    mol, _, _ = atoms2rdkit(atoms.copy(), nc=np.int64(0))
    assert Chem.MolToSmiles(mol) == Chem.MolToSmiles(ref)


@pytest.mark.parametrize("nc", [1.0, np.float64(1.0), "1", None])
def test_atoms2rdkit_rejects_non_int_charge(nc):
    with pytest.raises(TypeError, match="nc must be an integer"):
        atoms2rdkit(Atoms("Li", positions=[[0, 0, 0]]), nc=nc)
