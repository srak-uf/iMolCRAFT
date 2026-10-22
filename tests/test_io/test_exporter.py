import math
import os

import numpy
import openmm
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


#: A small branched molecule C0-C1(-H4)-O2-H3: (element, mass, sigma [nm],
#: epsilon [kJ/mol]) of every atom, its bonds and its angles
_ATOMS = [
    ("C", 12.01, 0.34, 0.36),
    ("C", 12.01, 0.34, 0.36),
    ("O", 16.00, 0.30, 0.71),
    ("H", 1.008, 0.0, 0.0),
    ("H", 1.008, 0.26, 0.07),
]
_BONDS = [(0, 1), (1, 2), (2, 3), (1, 4)]
_ANGLES = [(0, 1, 2), (1, 2, 3), (0, 1, 4), (2, 1, 4)]


def _build(orders, n_ions=0, improper_phase=180.0):
    """
    Topology, system and positions of copies of the molecule in _ATOMS.

    Every entry of ``orders`` adds one copy with its atoms in that order (a
    permutation of the template). ``n_ions`` adds lone atoms with the
    parameters of the bonded O. Torsions: the chain 0-1-2-3 (n = 3, plus a
    zero-barrier term) and 0-2-1-4 around C1 (n = 2, ``improper_phase``).
    """
    topology, system = app.Topology(), openmm.System()
    nonbonded = openmm.NonbondedForce()
    bond_force = openmm.HarmonicBondForce()
    angle_force = openmm.HarmonicAngleForce()
    torsion_force = openmm.PeriodicTorsionForce()
    positions, bonds = [], []

    def add_atom(t, residue):
        element, mass, sigma, epsilon = _ATOMS[t]
        topology.addAtom(f"{element}{t}", app.element.get_by_symbol(element), residue)
        positions.append(openmm.Vec3(0.5 * len(positions), 0.1 * t, 0.0))
        nonbonded.addParticle(0.1 * (-1) ** t, sigma, epsilon)
        return system.addParticle(mass)

    for order in orders:
        residue = topology.addResidue("MOL", topology.addChain())
        index = {t: add_atom(t, residue) for t in order}  # template -> system
        for b, (t1, t2) in enumerate(_BONDS):
            bond_force.addBond(index[t1], index[t2], 0.10 + 0.01 * b, 2.0e5)
            bonds.append((index[t1], index[t2]))
        for t1, t2, t3 in _ANGLES:
            angle_force.addAngle(index[t1], index[t2], index[t3], 1.9, 300.0)
        chain = [index[t] for t in (0, 1, 2, 3)]
        torsion_force.addTorsion(*chain, 3, 0.0, 1.0)
        torsion_force.addTorsion(*chain, 1, 0.0, 0.0)
        torsion_force.addTorsion(
            *(index[t] for t in (0, 2, 1, 4)), 2, math.radians(improper_phase), 2.0
        )
    for _ in range(n_ions):
        add_atom(2, topology.addResidue("ION", topology.addChain()))
    nonbonded.createExceptionsFromBonds(bonds, 0.8333, 0.5)
    for force in (nonbonded, bond_force, angle_force, torsion_force):
        system.addForce(force)
    box = [openmm.Vec3(3.0, 0, 0), openmm.Vec3(0, 3.0, 0), openmm.Vec3(0, 0, 3.0)]
    system.setDefaultPeriodicBoxVectors(*box)
    return topology, system, positions * unit.nanometer


def _count(path, word):
    """A count of the header, e.g. _count(path, "bond types")"""
    for line in path.read_text().splitlines():
        parts = line.split()
        if parts[1:] == word.split():
            return int(parts[0])
    raise KeyError(word)


def _section(path, name):
    """Lines of a data file section, [] if it is not written"""
    lines = path.read_text().splitlines()
    heads = [i for i, line in enumerate(lines) if line.split("#")[0].strip() == name]
    if not heads:
        return []
    body = lines[heads[0] + 2:]
    return body[: body.index("")] if "" in body else body


def _nonbonded(system):
    return next(f for f in system.getForces() if isinstance(f, openmm.NonbondedForce))


def test_write_lammps_shares_types_across_molecules(tmp_path):
    """More copies of a molecule add atoms and molecules, not types"""
    one = write_lammps(*_build([range(5)]), tmp_path / "one")
    three = write_lammps(*_build([range(5)] * 3), tmp_path / "three")
    for kind in ("atom", "bond", "angle", "dihedral", "improper"):
        assert _count(three, f"{kind} types") == _count(one, f"{kind} types")
    assert _count(one, "atom types") == 4  # the two C share a type
    assert _count(three, "atoms") == 15
    assert _count(three, "bonds") == 12
    molecule_ids = [int(line.split()[1]) for line in _section(three, "Atoms")]
    assert molecule_ids == [1] * 5 + [2] * 5 + [3] * 5


def test_write_lammps_atom_types_follow_species(tmp_path):
    """Atom types depend on the species, not on the atom order"""
    order = [4, 2, 0, 3, 1]
    path = write_lammps(*_build([range(5), order], n_ions=1), tmp_path / "species")
    types = [int(line.split()[2]) for line in _section(path, "Atoms")]
    # the same molecule in another atom order gets the same types ...
    assert types[5:10] == [types[t] for t in order]
    # ... and the lone O, with the parameters of the bonded O, a new one
    assert types[10] not in types[:10]
    assert _count(path, "atom types") == 5


def test_write_lammps_splits_torsions(tmp_path):
    path = write_lammps(*_build([range(5)]), tmp_path / "torsions")
    # chain 0-1-2-3: a fourier dihedral, the zero-barrier term dropped
    assert _section(path, "Dihedral Coeffs") == ["1 1 0.2390057361 3 0"]
    assert _section(path, "Dihedrals") == ["1 1 1 2 3 4"]
    # 0-2-1-4 is not a chain: a cvff improper K d n, d = -1 for phase 180
    assert _section(path, "Improper Coeffs") == ["1 0.4780114723 -1 2"]
    assert _section(path, "Impropers") == ["1 1 1 3 2 5"]


def test_write_lammps_falls_back_to_fourier(tmp_path):
    """A torsion cvff cannot express (phase 90) is written as a dihedral"""
    path = write_lammps(*_build([range(5)], improper_phase=90.0), tmp_path / "fb")
    assert _count(path, "impropers") == 0
    assert _section(path, "Improper Coeffs") == []
    assert _section(path, "Dihedral Coeffs")[1] == "2 1 0.4780114723 2 90"
    assert _section(path, "Dihedrals")[1] == "2 2 1 3 2 5"


def _add_parameter_offset(system):
    nonbonded = _nonbonded(system)
    nonbonded.addGlobalParameter("lambda", 1.0)
    nonbonded.addParticleParameterOffset("lambda", 0, 0.1, 0.0, 0.0)


@pytest.mark.parametrize(
    "modify, message",
    [
        (lambda s: s.addForce(openmm.CustomBondForce("0")), "CustomBondForce"),
        (_add_parameter_offset, "parameter offsets"),
        (
            lambda s: s.setVirtualSite(4, openmm.TwoParticleAverageSite(0, 1, 0.5, 0.5)),
            "Virtual sites",
        ),
        (lambda s: s.addConstraint(0, 2, 0.25), "no bond term"),
        (lambda s: _nonbonded(s).addException(0, 5, 0.0, 0.1, 0.0), "three bonds"),
    ],
    ids=["force", "offset", "virtual-site", "constraint", "exception"],
)
def test_write_lammps_refuses_unsupported(modify, message, tmp_path):
    topology, system, positions = _build([range(5)] * 2)
    modify(system)
    with pytest.raises(NotImplementedError, match=message):
        write_lammps(topology, system, positions, tmp_path / "refused")


def test_write_lammps_rejects_size_mismatch(tmp_path):
    topology, system, positions = _build([range(5)])
    with pytest.raises(ValueError, match="number of atoms"):
        write_lammps(topology, system, positions[:-1], tmp_path / "short")
