import os

import numpy
import pytest
from openmm import XmlSerializer, app, unit

from imolcraft.io._exporter import exporter
from imolcraft.io.exporter_lmp import write_lammps

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
PDB = os.path.join(DATA, "supercell_bonds.pdb")
SYSTEM = os.path.join(DATA, "system.xml")


def test_exporter_rejects_unknown_format(tmp_path):
    with pytest.raises(ValueError, match="Unsupported format"):
        exporter(PDB, SYSTEM, str(tmp_path / "out"), "namd")


def test_exporter_accepts_fmt_as_keyword(tmp_path, monkeypatch):
    """The public API keyword is named fmt (the notebooks call it by this name)"""
    monkeypatch.chdir(tmp_path)
    exporter(pdb=PDB, system=SYSTEM, filename="kw", fmt="lmp")
    assert (tmp_path / "kw.data").exists()


@pytest.mark.parametrize(
    "fmt, suffixes", [("gmx", [".top", ".gro"]), ("lmp", [".data"])]
)
def test_exporter_writes_files(fmt, suffixes, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    exporter(PDB, SYSTEM, "out", fmt)
    for suffix in suffixes:
        out = tmp_path / f"out{suffix}"
        assert out.exists() and out.stat().st_size > 0


def test_exporter_lmp_header(tmp_path, monkeypatch):
    """The LAMMPS data file headers are written in the expected order"""
    monkeypatch.chdir(tmp_path)
    exporter(PDB, SYSTEM, "out", "lmp")
    lines = (tmp_path / "out.data").read_text().splitlines()
    assert lines[0].startswith("LAMMPS data file")
    counts = lines[2:7]
    assert [line.split()[1] for line in counts] == [
        "atoms", "bonds", "angles", "dihedrals", "impropers"
    ]
    assert any(line.endswith("zlo zhi") for line in lines)


@pytest.fixture
def inputs():
    """Topology, system and positions of the test molecule"""
    pdb = app.PDBFile(PDB)
    with open(SYSTEM) as f:
        system = XmlSerializer.deserialize(f.read())
    return pdb.topology, system, pdb.positions


def _box_lines(path):
    return [
        line for line in path.read_text().splitlines()
        if line.endswith(("xlo xhi", "ylo yhi", "zlo zhi", "xy xz yz"))
    ]


def test_write_lammps_without_box(inputs, tmp_path):
    """Without a box the default box of the system (40 A) is written"""
    topology, system, positions = inputs
    topology.setPeriodicBoxVectors(None)
    lines = _box_lines(write_lammps(topology, system, positions, tmp_path / "nobox"))
    assert not lines[-1].endswith("xy xz yz")
    lo, hi = (float(v) for v in lines[0].split()[:2])
    assert numpy.isclose(hi - lo, 40.0)


def test_write_lammps_orthogonal(inputs, tmp_path):
    topology, system, positions = inputs
    topology.setPeriodicBoxVectors(numpy.diag([2.0, 3.0, 4.0]) * unit.nanometer)
    lines = _box_lines(write_lammps(topology, system, positions, tmp_path / "ortho"))
    assert not lines[-1].endswith("xy xz yz")
    edges = [
        float(line.split()[1]) - float(line.split()[0]) for line in lines[:3]
    ]
    assert numpy.allclose(edges, [20.0, 30.0, 40.0])


def test_write_lammps_triclinic(inputs, tmp_path):
    """A triclinic cell is converted into the LAMMPS tilt factors xy / xz / yz"""
    topology, system, positions = inputs
    topology.setPeriodicBoxVectors(
        numpy.array([[2.0, 0.0, 0.0], [0.6, 1.9, 0.0], [0.4, 0.3, 1.8]])
        * unit.nanometer
    )
    lines = _box_lines(write_lammps(topology, system, positions, tmp_path / "triclinic"))
    assert lines[-1] == "6 4 3 xy xz yz"
    edges = [
        float(line.split()[1]) - float(line.split()[0]) for line in lines[:3]
    ]
    assert numpy.allclose(edges, [20.0, 19.0, 18.0])
