import os

import numpy as np
import pytest

from imolcraft.io.mol2 import (
    _element_from_atom_name,
    mol2_to_aseatoms,
    read_mol2,
    write_mol2,
)

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
FSA = os.path.join(DATA, "fsa_resp.mol2")
GAFF = os.path.join(DATA, "fsa_resp.mol2.gaff")
PF6 = os.path.join(DATA, "pf6_resp.mol2")


@pytest.mark.parametrize("mol2file", [FSA, PF6])
def test_read_mol2_sections(mol2file):
    mol2_dict = read_mol2(mol2file)
    assert "@<TRIPOS>MOLECULE" in mol2_dict
    assert "@<TRIPOS>ATOM" in mol2_dict
    assert "@<TRIPOS>BOND" in mol2_dict
    # Each row becomes a whitespace-separated token list
    assert all(isinstance(row, list) for row in mol2_dict["@<TRIPOS>ATOM"])
    # Blank lines are skipped
    assert all(row for row in mol2_dict["@<TRIPOS>ATOM"])


@pytest.mark.parametrize("mol2file", [FSA, PF6])
def test_write_mol2_dict_roundtrip(mol2file, tmp_path):
    """Writing a dict back and reading it again gives the same dict"""
    original = read_mol2(mol2file)
    out = tmp_path / "out.mol2"
    write_mol2(str(out), original)
    assert read_mol2(str(out)) == original


def test_write_mol2_accepts_data_as_keyword(tmp_path):
    """The public API keyword is named data"""
    out = tmp_path / "kw.mol2"
    original = read_mol2(FSA)
    write_mol2(str(out), data=original)
    assert read_mol2(str(out)) == original


def test_write_mol2_rejects_other_types(tmp_path):
    out = tmp_path / "never.mol2"
    with pytest.raises(ValueError, match="dictionary or a molecule object"):
        write_mol2(str(out), 42)
    # Validation happens before open, so no empty file is created
    assert not out.exists()


@pytest.mark.parametrize(
    "mol2file, n_atoms, symbols",
    [(FSA, 9, {"S", "O", "N", "F"}), (PF6, 7, {"P", "F"})],
)
def test_mol2_to_aseatoms(mol2file, n_atoms, symbols):
    atoms = mol2_to_aseatoms(mol2file)
    assert len(atoms) == n_atoms
    assert set(atoms.get_chemical_symbols()) == symbols
    assert atoms.positions.shape == (n_atoms, 3)
    assert np.isfinite(atoms.positions).all()


@pytest.mark.parametrize(
    "atom_name, element",
    [
        # The element symbol itself
        ("S", "S"), ("F", "F"), ("Li", "Li"),
        # Element symbol + sequence number (antechamber naming)
        ("O1", "O"), ("S1", "S"), ("C12", "C"), ("Cl1", "Cl"), ("CL1", "Cl"),
        # Form with a leading digit
        ("1HB", "H"),
    ],
)
def test_element_from_atom_name(atom_name, element):
    assert _element_from_atom_name(atom_name) == element


@pytest.mark.parametrize("atom_name", ["XX", "X", "", "Du", "1"])
def test_element_from_atom_name_rejects_unknown(atom_name):
    """An unresolvable name raises; it must not match ase's dummy element "X" either"""
    with pytest.raises(ValueError, match="Cannot determine the element"):
        _element_from_atom_name(atom_name)


def test_mol2_to_aseatoms_reads_gaff_typed_file():
    """Elements resolve even for antechamber output with atom names like O1 / S1"""
    rows = read_mol2(GAFF)["@<TRIPOS>ATOM"]
    assert "O1" in [row[1] for row in rows]  # the name is not the element symbol
    atoms = mol2_to_aseatoms(GAFF)
    assert atoms.get_chemical_symbols() == [
        "S", "F", "O", "O", "N", "S", "F", "O", "O"
    ]


def test_read_mol2_skips_comments(tmp_path):
    """Comment lines starting with # are skipped, before a header or in a section"""
    path = tmp_path / "commented.mol2"
    path.write_text(
        "# Created by SomeTool\n"
        "@<TRIPOS>MOLECULE\n"
        "MOL\n"
        "# a comment inside a section\n"
        "2 1 0 0 0\n"
        "  # indented comment\n"
        "@<TRIPOS>ATOM\n"
        "1 C 0.0 0.0 0.0 C 1 MOL 0.0\n"
    )
    assert read_mol2(str(path)) == {
        "@<TRIPOS>MOLECULE": [["MOL"], ["2", "1", "0", "0", "0"]],
        "@<TRIPOS>ATOM": [["1", "C", "0.0", "0.0", "0.0", "C", "1", "MOL", "0.0"]],
    }


def test_read_mol2_rejects_content_before_first_section(tmp_path):
    """Content before a header fails in a way that names the offending line"""
    bad = tmp_path / "bad.mol2"
    bad.write_text("\n\nnot a mol2 file\n@<TRIPOS>MOLECULE\nMOL\n")
    with pytest.raises(ValueError, match=r"bad\.mol2:3: content before the first"):
        read_mol2(str(bad))


def test_read_mol2_accepts_leading_blank_lines(tmp_path):
    ok = tmp_path / "ok.mol2"
    ok.write_text("\n \n@<TRIPOS>MOLECULE\nMOL\n")
    assert read_mol2(str(ok)) == {"@<TRIPOS>MOLECULE": [["MOL"]]}


def test_read_mol2_on_empty_file(tmp_path):
    empty = tmp_path / "empty.mol2"
    empty.write_text("")
    assert read_mol2(str(empty)) == {}
