import os

import numpy
import pytest

from imolcraft.io import exporter_lmp as exporter_lmp_module
from imolcraft.io._exporter import exporter
from imolcraft.io.exporter_lmp import to_lammps_non_rectangular

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
PDB = os.path.join(DATA, "supercell_bonds.pdb")
SYSTEM = os.path.join(DATA, "system.xml")


def test_exporter_rejects_unknown_format(tmp_path):
    with pytest.raises(ValueError, match="Unsupported format"):
        exporter(PDB, SYSTEM, str(tmp_path / "out"), "namd")


def test_exporter_accepts_fmt_as_keyword(tmp_path, monkeypatch):
    """公開 API のキーワード名は fmt（ノートブックがこの名前で呼ぶ）"""
    pytest.importorskip("openff.interchange")
    monkeypatch.chdir(tmp_path)
    exporter(pdb=PDB, system=SYSTEM, filename="kw", fmt="lmp")
    assert (tmp_path / "kw.data").exists()


@pytest.mark.parametrize(
    "fmt, suffixes", [("gmx", [".top", ".gro"]), ("lmp", [".data"])]
)
def test_exporter_writes_files(fmt, suffixes, tmp_path, monkeypatch):
    pytest.importorskip("openff.interchange")
    monkeypatch.chdir(tmp_path)
    exporter(PDB, SYSTEM, "out", fmt)
    for suffix in suffixes:
        out = tmp_path / f"out{suffix}"
        assert out.exists() and out.stat().st_size > 0


def test_exporter_lmp_header(tmp_path, monkeypatch):
    """LAMMPS data ファイルの見出しが期待どおりの並びで書かれる"""
    pytest.importorskip("openff.interchange")
    monkeypatch.chdir(tmp_path)
    exporter(PDB, SYSTEM, "out", "lmp")
    lines = (tmp_path / "out.data").read_text().splitlines()
    assert lines[0] == "Title"
    counts = lines[2:7]
    assert [line.split()[1] for line in counts] == [
        "atoms", "bonds", "angles", "dihedrals", "impropers"
    ]
    assert any(line.endswith("xy xz yz") for line in lines)


@pytest.fixture
def interchange(tmp_path, monkeypatch):
    """exporter_lmp が組み立てた Interchange を横取りして取り出す"""
    pytest.importorskip("openff.interchange")
    captured = {}
    real = exporter_lmp_module.to_lammps_non_rectangular

    def capture(interchange, file_path):
        captured["interchange"] = interchange
        return real(interchange, file_path)

    monkeypatch.setattr(exporter_lmp_module, "to_lammps_non_rectangular", capture)
    monkeypatch.chdir(tmp_path)
    exporter_lmp_module.exporter_lmp(PDB, SYSTEM, "captured")
    return captured["interchange"]


def _box_lines(path):
    return [
        line for line in path.read_text().splitlines()
        if line.endswith(("xlo xhi", "ylo yhi", "zlo zhi", "xy xz yz"))
    ]


def test_to_lammps_non_rectangular_without_box(interchange, tmp_path):
    """box が無いと 100 A の立方セルが書かれる（傾きはゼロ）"""
    interchange.box = None
    out = tmp_path / "nobox.data"
    to_lammps_non_rectangular(interchange, str(out))
    lines = _box_lines(out)
    assert lines[-1] == "0.0 0.0 0.0 xy xz yz"
    lo, hi = (float(v) for v in lines[0].split()[:2])
    assert numpy.isclose(hi - lo, 100.0)


def test_to_lammps_non_rectangular_orthogonal(interchange, tmp_path):
    from openff.toolkit.topology.molecule import unit

    interchange.box = numpy.diag([20.0, 30.0, 40.0]) * unit.angstrom
    out = tmp_path / "ortho.data"
    to_lammps_non_rectangular(interchange, str(out))
    lines = _box_lines(out)
    assert lines[-1] == "0.0 0.0 0.0 xy xz yz"
    edges = [
        float(line.split()[1]) - float(line.split()[0]) for line in lines[:3]
    ]
    assert numpy.allclose(edges, [20.0, 30.0, 40.0])


def test_to_lammps_non_rectangular_triclinic(interchange, tmp_path):
    """三斜晶は LAMMPS の傾き因子 xy / xz / yz に変換される"""
    from openff.toolkit.topology.molecule import unit

    interchange.box = numpy.array(
        [[20.0, 0.0, 0.0], [6.0, 19.0, 0.0], [4.0, 3.0, 18.0]]
    ) * unit.angstrom
    out = tmp_path / "triclinic.data"
    to_lammps_non_rectangular(interchange, str(out))
    lines = _box_lines(out)
    assert lines[-1] == "6 4 3 xy xz yz"
    edges = [
        float(line.split()[1]) - float(line.split()[0]) for line in lines[:3]
    ]
    assert numpy.allclose(edges, [20.0, 19.0, 18.0])
