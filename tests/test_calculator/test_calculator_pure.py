"""外部プログラム（g16 / psi4 / antechamber）を必要としない部分の回帰テスト"""
import copy
import os

import numpy as np
import pytest
from ase import Atoms
from rdkit import Chem

from imolcraft.calculator.dihedral import (
    _wrap_deg,
    get_rotatable_dihedral,
    load_g16scan,
    rotate_dihedral,
)
from imolcraft.calculator.distance import change_distance
from imolcraft.calculator.psi4geoopt import Psi4GeoOptimizer

TESTS = os.path.join(os.path.dirname(__file__), "..")

ETHANE = Atoms(
    "CH3CH3",
    positions=[
        [-2.79452060, 1.06849313, 0.0],
        [-2.43786617, 0.05968313, 0.0],
        [-2.43784776, 1.57289132, -0.87365150],
        [-3.86452060, 1.06850632, 0.0],
        [-2.28117838, 1.79444941, 1.25740497],
        [-2.63623472, 2.80382250, 1.25642745],
        [-2.63944749, 1.29118098, 2.13105486],
        [-1.21118019, 1.79274272, 1.25838372],
    ],
)


@pytest.mark.parametrize(
    "angle, expected",
    [(0, 0), (180, -180), (181, -179), (-180, -180), (360, 0), (-190, 170),
     (-181, 179)],
)
def test_wrap_deg(angle, expected):
    assert _wrap_deg(angle) == pytest.approx(expected)


@pytest.mark.parametrize(
    "smiles, n_dihedrals",
    [("C", 0), ("CC", 1), ("CCCC", 3), ("CCO", 2), ("CC(=O)OC", 3)],
)
def test_get_rotatable_dihedral(smiles, n_dihedrals):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    dihedrals, elements = get_rotatable_dihedral(mol)
    assert len(dihedrals) == n_dihedrals
    assert len(elements) == len(dihedrals)
    for dihedral, elems in zip(dihedrals, elements):
        assert len(dihedral) == 4
        assert len(set(dihedral)) == 4  # 4 原子はすべて異なる
        assert elems == [mol.GetAtomWithIdx(i).GetSymbol() for i in dihedral]


@pytest.mark.parametrize("target", [-180.0, -90.0, 0.0, 37.5, 120.0, 180.0])
def test_rotate_dihedral_reaches_target(target):
    dihedral = [1, 0, 4, 5]
    rotated = rotate_dihedral(ETHANE.copy(), dihedral, target)
    assert rotated.get_chemical_symbols() == ETHANE.get_chemical_symbols()
    assert _wrap_deg(rotated.get_dihedral(*dihedral)) == pytest.approx(
        _wrap_deg(target), abs=1e-6
    )


def test_rotate_dihedral_keeps_bond_lengths():
    """二面角の回転で結合長は変わらない"""
    before = ETHANE.get_all_distances()
    rotated = rotate_dihedral(ETHANE.copy(), [1, 0, 4, 5], 90.0)
    # 回転軸まわりの剛体回転なので C-C とメチル内の距離は保たれる
    assert rotated.get_distance(0, 4) == pytest.approx(before[0, 4], abs=1e-9)
    assert rotated.get_distance(0, 1) == pytest.approx(before[0, 1], abs=1e-9)
    assert rotated.get_distance(4, 5) == pytest.approx(before[4, 5], abs=1e-9)


@pytest.mark.parametrize("pair", [[0, 4], [0, 1], [4, 7]])
@pytest.mark.parametrize("target", [0.9, 1.5, 3.0, 6.0])
def test_change_distance_reaches_target(pair, target):
    moved = change_distance(ETHANE.copy(), pair, target)
    assert moved.get_distance(*pair) == pytest.approx(target, abs=1e-9)
    assert moved.get_chemical_symbols() == ETHANE.get_chemical_symbols()


def test_change_distance_translates_only_one_fragment():
    """切った結合の反対側の断片は動かない"""
    moved = change_distance(ETHANE.copy(), [0, 4], 3.0)
    # 原子 0 側のメチル (0,1,2,3) は静止
    assert np.allclose(moved.positions[:4], ETHANE.positions[:4])
    # 原子 4 側のメチル (4..7) は剛体移動
    shift = moved.positions[4] - ETHANE.positions[4]
    assert np.allclose(moved.positions[4:] - ETHANE.positions[4:], shift)


def test_load_g16scan(tmp_path):
    log = tmp_path / "scan.log"
    log.write_bytes(open(os.path.join(TESTS, "data", "test_dihed_0.log"), "rb").read())

    angle, energy, atoms_list = load_g16scan(str(log))
    assert len(angle) == len(energy) == len(atoms_list)
    assert np.all(np.diff(angle) > 0)          # 昇順にソートされる
    assert energy.min() == pytest.approx(0.0)  # 最小値が 0 になるようシフト
    assert np.all(energy >= 0)
    for a in atoms_list:
        assert len(a) == len(atoms_list[0])


@pytest.mark.parametrize(
    "atoms, task",
    [(ETHANE, "optimize("), (Atoms("Li", positions=[[0, 0, 0]]), "energy(")],
)
def test_psi4_input_generation(atoms, task, tmp_path):
    """多原子は構造最適化、単原子は一点計算になる"""
    out = tmp_path / "in.psi4in"
    Psi4GeoOptimizer(
        atoms, "wb97x-d", "6-31g", charge=-1, multiplicity=1, label="x"
    ).generate_input(str(out))

    text = out.read_text()
    assert task in text
    assert "basis 6-31g" in text
    assert "-1 1" in text                      # charge multiplicity
    assert text.count("molecule {") == 1
    assert len([ln for ln in text.splitlines() if "\t" in ln]) == len(atoms)


def test_am1bcc_declares_pdb_input_format(tmp_path, monkeypatch):
    """入力は PDB で書き出しているので antechamber にも -fi pdb と伝える"""
    from imolcraft.calculator import charge as charge_module

    calc = charge_module.ChargeCalculator(
        ETHANE, "am1bcc", 0, "t", directory=str(tmp_path)
    )

    captured = {}

    def fake_getoutput(cmd):
        captured["cmd"] = cmd
        # antechamber の代わりに最低限の mol2 を書き出す
        out = cmd.split("-o ")[1].split()[0]
        with open(out, "w") as f:
            f.write("@<TRIPOS>ATOM\n")
            for i in range(len(ETHANE)):
                f.write(f"{i+1} C 0.0 0.0 0.0 C 1 MOL -0.125\n")
        return ""

    monkeypatch.setattr(charge_module.subprocess, "getoutput", fake_getoutput)
    calc.get_partialcharges()

    assert " -fi pdb " in captured["cmd"]
    assert "gout" not in captured["cmd"]
    assert " -c bcc " in captured["cmd"]
    assert calc.partial_charges == [-0.125] * len(ETHANE)
    assert os.path.basename(calc.mol2file) == "t_am1bcc.mol2"


def _succinonitrile():
    """仮想サイト付き力場に対応する実原子だけの構造"""
    from ase import Atoms
    from openmm.app import PDBFile
    from openmm.unit import angstrom as omm_angstrom

    pdb = PDBFile(os.path.join(TESTS, "data", "vs_supercell_bonds.pdb"))
    coords = pdb.getPositions().value_in_unit(omm_angstrom)
    real = [
        (a.element.symbol, coords[i])
        for i, a in enumerate(pdb.topology.atoms())
        if a.element is not None
    ]
    return Atoms(
        symbols=[s for s, _ in real],
        positions=[[c.x, c.y, c.z] for _, c in real],
    )


@pytest.mark.parametrize("ffxml", ["vsite_average2.xml", "vsite_average3.xml"])
def test_scan_ff_dihedral_with_virtual_sites(ffxml, tmp_path, monkeypatch):
    """仮想サイトを含む力場でも粒子数が食い違わずスキャンできる"""
    pytest.importorskip("openmm")
    from imolcraft.calculator.dihedral import scan_ff_dihedral

    monkeypatch.chdir(tmp_path)
    atoms = _succinonitrile()
    angles = np.array([-60.0, 0.0, 60.0])

    scanned, energies, geometries = scan_ff_dihedral(
        os.path.join(TESTS, "data", ffxml),
        angles.copy(),
        [0, 2, 3, 4],
        geoopt_atoms=atoms,
    )

    assert np.array_equal(np.asarray(scanned), angles)
    assert min(energies) == pytest.approx(0.0)
    # 仮想サイトは結果の構造に残らない
    for geometry in geometries:
        assert len(geometry) == len(atoms)
        assert all(geometry.get_chemical_symbols())


def test_scan_ff_dihedral_requires_a_geometry_source():
    from imolcraft.calculator.dihedral import scan_ff_dihedral

    with pytest.raises(ValueError, match="Please provide one of them"):
        scan_ff_dihedral("unused.xml", [0.0], [0, 1, 2, 3])


def test_scan_ff_dihedral_checks_angle_count():
    from imolcraft.calculator.dihedral import scan_ff_dihedral

    with pytest.raises(ValueError, match="must be the same: 2 != 1"):
        scan_ff_dihedral(
            "unused.xml", [0.0, 10.0], [0, 1, 2, 3], atoms_list=[ETHANE]
        )


@pytest.mark.parametrize("ffxml", ["vsite_average2.xml", "vsite_average3.xml"])
def test_scan_ff_distance_with_virtual_sites(ffxml, tmp_path, monkeypatch):
    """仮想サイトを含む力場でも、次の点へ渡す座標が実原子だけになる"""
    pytest.importorskip("openmm")
    from imolcraft.calculator.distance import scan_ff_distance

    monkeypatch.chdir(tmp_path)
    atoms = _succinonitrile()
    targets = [1.5, 1.6, 1.7]

    scanned, energies, geometries = scan_ff_distance(
        os.path.join(TESTS, "data", ffxml), list(targets), [2, 3], atoms=atoms
    )

    assert list(scanned) == targets
    assert len(energies) == len(targets)
    # 仮想サイトは結果の構造に残らない
    for geometry in geometries:
        assert len(geometry) == len(atoms)
        assert all(geometry.get_chemical_symbols())


def test_psi4_charge_calculator_accepts_directory_none(tmp_path, monkeypatch):
    """directory=None は基底クラスと同じくカレントディレクトリを指す"""
    from imolcraft.calculator import Psi4ChargeCalculator

    monkeypatch.chdir(tmp_path)
    calc = Psi4ChargeCalculator(ETHANE, "resp", 0, "t")

    assert calc.directory == str(tmp_path)
    assert (tmp_path / "t.sdf").exists()
    assert calc.molecule.n_atoms == len(ETHANE)


def test_psi4_charge_calculator_creates_missing_directory(tmp_path):
    from imolcraft.calculator import Psi4ChargeCalculator

    target = tmp_path / "deep" / "nested"
    calc = Psi4ChargeCalculator(ETHANE, "resp", 0, "t", directory=str(target))

    assert calc.directory == str(target)
    assert (target / "t.sdf").exists()


def test_charge_calculator_does_not_mutate_module_defaults(tmp_path):
    """独自 params を渡してもモジュールレベルの既定値は変わらない"""
    from imolcraft.calculator import charge as charge_module

    before = copy.deepcopy(charge_module.resp_params)
    calc = charge_module.ChargeCalculator(
        ETHANE, "resp", 0, "A", directory=str(tmp_path), params={"method": "b3lyp"}
    )

    assert calc.params["method"] == "b3lyp"
    assert charge_module.resp_params == before
    # 既定値のうち上書きしなかったものは残る
    assert calc.params["basis"] == before["basis"]


def test_charge_calculator_instances_do_not_share_params(tmp_path):
    from imolcraft.calculator import charge as charge_module

    a = charge_module.ChargeCalculator(
        ETHANE, "resp", 0, "A", directory=str(tmp_path), params={"method": "b3lyp"}
    )
    b = charge_module.ChargeCalculator(
        ETHANE, "resp", 0, "B", directory=str(tmp_path)
    )

    # B は params を渡していないので既定値のまま
    assert b.params["method"] == charge_module.resp_params["method"] == "hf"
    assert a.params is not b.params
    assert a.params is not charge_module.resp_params
    assert b.params is not charge_module.resp_params


def test_charge_calculator_params_are_deep_copies(tmp_path):
    """入れ子のリストを書き換えても他へ波及しない"""
    from imolcraft.calculator import charge as charge_module

    before = copy.deepcopy(charge_module.resp_params)
    a = charge_module.ChargeCalculator(
        ETHANE, "resp", 0, "A", directory=str(tmp_path)
    )
    b = charge_module.ChargeCalculator(
        ETHANE, "resp", 0, "B", directory=str(tmp_path)
    )

    a.params["ioplist"].append("6/50=1")
    a.params["basis"] = "sto-3g"

    assert b.params == before
    assert charge_module.resp_params == before


def _dihedral_calculator(tmp_path):
    from imolcraft.calculator import DihedralCalculator

    return DihedralCalculator(ETHANE, "t", directory=str(tmp_path))


@pytest.mark.parametrize("dihed_idx", [0, "0", [0]])
def test_do_ffscan_normalises_a_bare_index(dihed_idx, tmp_path, monkeypatch):
    """int でも str でも 1 本分のスキャンとして扱われる"""
    from imolcraft.calculator import dihedral as dihedral_module

    calc = _dihedral_calculator(tmp_path)
    calls = []

    def fake_scan(ffxml, angles, dihed_atidx, **kwargs):
        calls.append(dihed_atidx)
        return (), (), ()

    monkeypatch.setattr(dihedral_module, "scan_ff_dihedral", fake_scan)
    calc.do_ffscan("unused.xml", dihed_idx=dihed_idx, ini_geom="FF")

    assert calls == [calc.scan_list[0]]


def test_resolve_angles_uses_the_default_grid(tmp_path):
    calc = _dihedral_calculator(tmp_path)
    angles = calc._resolve_angles(0, None)
    assert list(angles) == list(np.arange(-180, 181, 10))


def test_resolve_angles_accepts_a_sequence(tmp_path):
    calc = _dihedral_calculator(tmp_path)
    angles = calc._resolve_angles(0, [-30, 0, 30])
    assert isinstance(angles, np.ndarray)
    assert list(angles) == [-30, 0, 30]


def test_resolve_angles_accepts_qm(tmp_path):
    calc = _dihedral_calculator(tmp_path)
    calc.qm_scan[0]["angle_deg"] = [-10.0, 0.0, 10.0]
    assert calc._resolve_angles(0, "QM") == [-10.0, 0.0, 10.0]


@pytest.mark.parametrize("angles", ["FF", "qm", ""])
def test_resolve_angles_rejects_other_strings(angles, tmp_path):
    """"QM" 以外の文字列がそのまま角度列として流れないこと"""
    calc = _dihedral_calculator(tmp_path)
    with pytest.raises(ValueError, match='angles must be "QM"'):
        calc._resolve_angles(0, angles)


def test_pick_outer_atom_rejects_a_terminal_atom():
    """相手以外に隣接原子が無ければ二面角は定義できない"""
    from imolcraft.calculator.dihedral import _pick_outer_atom

    # 暗黙水素のエタンは各炭素の明示的な隣接原子が相手だけ
    mol = Chem.MolFromSmiles("CC")
    with pytest.raises(ValueError, match="has no other neighbour"):
        _pick_outer_atom(mol.GetAtomWithIdx(0), 0, 1)


@pytest.mark.parametrize(
    "smiles",
    ["CC", "CCC", "CCCC", "CCO", "CC(C)(C)C", "c1ccccc1CC", "CC(=O)OC",
     "CCCCCCCC", "C1CCCCC1CC", "FC(F)(F)CC", "N#CCC#N",
     "O=S(=O)(F)NS(=O)(=O)F"],
)
@pytest.mark.parametrize("add_hs", [True, False])
def test_get_rotatable_dihedral_never_yields_none(smiles, add_hs):
    """回転可能結合の SMARTS が両端の次数 >= 2 を保証するので None は出ない"""
    mol = Chem.MolFromSmiles(smiles)
    if add_hs:
        mol = Chem.AddHs(mol)

    dihedrals, elements = get_rotatable_dihedral(mol)
    for dihedral, elems in zip(dihedrals, elements):
        assert all(isinstance(i, int) for i in dihedral)
        assert len(set(dihedral)) == 4
        assert all(isinstance(e, str) and e for e in elems)
