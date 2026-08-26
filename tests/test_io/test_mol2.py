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
    # 各行は空白区切りのトークン列になる
    assert all(isinstance(row, list) for row in mol2_dict["@<TRIPOS>ATOM"])
    # 空行は読み飛ばされる
    assert all(row for row in mol2_dict["@<TRIPOS>ATOM"])


@pytest.mark.parametrize("mol2file", [FSA, PF6])
def test_write_mol2_dict_roundtrip(mol2file, tmp_path):
    """dict を書き戻して読み直すと同じ dict になる"""
    original = read_mol2(mol2file)
    out = tmp_path / "out.mol2"
    write_mol2(str(out), original)
    assert read_mol2(str(out)) == original


def test_write_mol2_accepts_data_as_keyword(tmp_path):
    """公開 API のキーワード名は data"""
    out = tmp_path / "kw.mol2"
    original = read_mol2(FSA)
    write_mol2(str(out), data=original)
    assert read_mol2(str(out)) == original


def test_write_mol2_rejects_other_types(tmp_path):
    out = tmp_path / "never.mol2"
    with pytest.raises(ValueError, match="dictionary or a molecule object"):
        write_mol2(str(out), 42)
    # 検証は open より先なので空ファイルは作られない
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
        # 元素記号そのもの
        ("S", "S"), ("F", "F"), ("Li", "Li"),
        # 元素記号 + 連番（antechamber の命名）
        ("O1", "O"), ("S1", "S"), ("C12", "C"), ("Cl1", "Cl"), ("CL1", "Cl"),
        # 先頭に数字が付く形式
        ("1HB", "H"),
    ],
)
def test_element_from_atom_name(atom_name, element):
    assert _element_from_atom_name(atom_name) == element


@pytest.mark.parametrize("atom_name", ["XX", "X", "", "Du", "1"])
def test_element_from_atom_name_rejects_unknown(atom_name):
    """解決できない名前は例外。ase のダミー元素 "X" にも一致させない"""
    with pytest.raises(ValueError, match="Cannot determine the element"):
        _element_from_atom_name(atom_name)


def test_mol2_to_aseatoms_reads_gaff_typed_file():
    """原子名が O1 / S1 のような antechamber 出力でも元素を解決できる"""
    rows = read_mol2(GAFF)["@<TRIPOS>ATOM"]
    assert "O1" in [row[1] for row in rows]  # 名前は元素記号そのものではない
    atoms = mol2_to_aseatoms(GAFF)
    assert atoms.get_chemical_symbols() == [
        "S", "F", "O", "O", "N", "S", "F", "O", "O"
    ]


def test_read_mol2_skips_comments(tmp_path):
    """# 始まりの注釈行は、見出しの前でもセクション内でも読み飛ばされる"""
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
    """見出しより前に中身がある場合は、どの行かが分かる形で失敗する"""
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
