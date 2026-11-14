"""
Tests of the Poltype2 / Tinker -> OpenMM XML converter.

The reference is ``openmm.app.TinkerFiles``, OpenMM's own reader of the
Tinker files, and, for one geometry, the energies Tinker's ``analyze``
printed during the Poltype2 run. TinkerFiles is handed a single file holding
header and key together,
since in OpenMM 8.5 each file it reads resets the global settings the
previous one made (``torsionunit`` among them).
"""
import os
import re
import shutil

import pytest
from openmm import Context, Platform, Vec3, VerletIntegrator, app, unit

from imolcraft.crafter.amoeba import (
    merge_amoeba_xml,
    read_tinker_parameters,
    read_tinker_xyz,
    tinker2xml,
)

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
KEY = os.path.join(DATA, "fsa_amoeba.key")
XYZ = os.path.join(DATA, "fsa_amoeba.xyz")
HEADER = os.path.join(DATA, "amoeba_bio18_settings.prm")
#: a point of the Poltype2 torsion scan, the geometry of the Tinker energies
SCAN_XYZ = os.path.join(DATA, "fsa_amoeba_scan034.xyz")
#: ethylene carbonate, typed 501-505 by Poltype2 (atmidx=501), QM with Psi4
EC_KEY = os.path.join(DATA, "ec_amoeba.key")
EC_XYZ = os.path.join(DATA, "ec_amoeba.xyz")
#: the structure Tinker's minimize ended at, and the energy it printed
EC_MIN_XYZ = os.path.join(DATA, "ec_amoeba_min.xyz")
EC_MIN_ENERGY = -0.04242140

pytestmark = pytest.mark.skipif(
    not hasattr(app, "TinkerFiles"), reason="needs openmm.app.TinkerFiles (OpenMM >= 8.2)"
)


def _combined_key(tmp_path, keys=(KEY,), extra=""):
    """Header, keys and ``extra`` in one file, the way TinkerFiles needs them."""
    path = tmp_path / "combined.key"
    text = open(HEADER).read()
    for key in keys:
        with open(key) as f:
            text += "".join(line for line in f if not line.startswith("parameters"))
    path.write_text(text + extra)
    return str(path)


def _two_molecules(tmp_path, shift=(0.9, 0.2, 0.0)):
    """Tinker xyz of two FSI- anions, the second moved by ``shift`` nm."""
    with open(XYZ) as f:
        natoms = int(f.readline().split()[0])
        rows = [f.readline().split() for _ in range(natoms)]
    lines = [f"{2 * natoms}  FSI pair"]
    for copy, (dx, dy, dz) in enumerate(((0, 0, 0), [10 * x for x in shift])):
        for row in rows:
            bonded = " ".join(str(int(b) + copy * natoms) for b in row[6:])
            lines.append(
                f"{int(row[0]) + copy * natoms:6d}  {row[1]:2s} "
                f"{float(row[2]) + dx:12.6f} {float(row[3]) + dy:12.6f} "
                f"{float(row[4]) + dz:12.6f} {row[5]:>6s} {bonded}"
            )
    path = tmp_path / "pair.xyz"
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def _energies(system, positions):
    """Potential energy of every force of ``system``, in kcal/mol."""
    for i, force in enumerate(system.getForces()):
        force.setForceGroup(i)
    context = Context(system, VerletIntegrator(0.001), Platform.getPlatformByName("Reference"))
    context.setPositions(positions)
    energies = {}
    for i, force in enumerate(system.getForces()):
        energy = context.getState(getEnergy=True, groups={i}).getPotentialEnergy()
        name = force.getName()
        energies[name] = energies.get(name, 0.0) + energy.value_in_unit(unit.kilocalorie_per_mole)
    return energies


def _compare(reference, converted):
    """Every force of the reference is reproduced; extra forces are empty."""
    for name, energy in reference.items():
        assert converted[name] == pytest.approx(energy, abs=1e-8), name
    for name in set(converted) - set(reference):
        assert converted[name] == pytest.approx(0.0, abs=1e-12), name


def test_read_tinker_parameters():
    params = read_tinker_parameters([HEADER, KEY])
    assert params["scalars"]["torsionunit"] == "0.5"
    assert sorted(params["atom"]) == ["401", "402", "403", "404"]
    assert len(params["multipole"]) == 4
    # Poltype2 writes the fitted torsions in both directions
    assert len(params["torsion"]) == 4
    # the bisector frame of the nitrogen keeps its negative axis types
    nitrogen = [m for m in params["multipole"] if m[0][0] == "404"][0]
    assert nitrogen[0] == ["404", "-402", "-402", "-0.73612"]


def test_read_tinker_xyz():
    atoms = read_tinker_xyz(XYZ)
    assert len(atoms) == 9
    assert atoms[4] == {"index": 5, "symbol": "N", "type": "404", "bonded": [2, 6]}


def test_single_molecule_matches_tinker(tmp_path):
    """
    One anion without cutoff, at a geometry of the Poltype2 torsion scan for
    which Tinker's ``analyze`` printed the energy. Poltype2 put the whole
    anion in one polarization group, so an isolated molecule has no induced
    dipoles (the mutual solver cannot even converge) and direct is used.
    """
    xml = tinker2xml(KEY, XYZ, str(tmp_path / "fsa.xml"), resname="FSA", header=HEADER)
    tinker = app.TinkerFiles(SCAN_XYZ, _combined_key(tmp_path))
    reference = _energies(
        tinker.createSystem(nonbondedMethod=app.NoCutoff, polarization="direct"),
        tinker.positions,
    )
    converted = _energies(
        app.ForceField(xml).createSystem(
            tinker.topology, nonbondedMethod=app.NoCutoff, polarization="direct"
        ),
        tinker.positions,
    )
    _compare(reference, converted)
    # Tinker analyze, kcal/mol, without the torsion restraint of the scan
    tinker_analyze = {
        "AmoebaBondForce": 0.11265237,
        "AmoebaAngleForce": 3.88493310,
        "AmoebaStretchBendForce": -0.10730662,
        "PeriodicTorsionForce": -1.88210407,
        "AmoebaVdwForce": -0.57259937,
        "AmoebaMultipoleForce": -11.83690539,
    }
    for name, energy in tinker_analyze.items():
        assert converted[name] == pytest.approx(energy, abs=1e-7), name


def test_pair_pme_matches_tinkerfiles(tmp_path):
    """
    Two anions under PME with mutual polarization, the topology read back
    from a PDB with CONECT records as the rest of iMolCRAFT does it.
    """
    xml = tinker2xml(KEY, XYZ, str(tmp_path / "fsa.xml"), resname="FSA", header=HEADER)
    box = [Vec3(3, 0, 0), Vec3(0, 3, 0), Vec3(0, 0, 3)] * unit.nanometer
    settings = dict(
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.2 * unit.nanometer,
        polarization="mutual",
        mutualInducedTargetEpsilon=1e-6,
        # the ForceField and TinkerFiles defaults differ (1e-4 and 5e-4)
        ewaldErrorTolerance=1e-5,
    )
    tinker = app.TinkerFiles(_two_molecules(tmp_path), _combined_key(tmp_path),
                             periodicBoxVectors=box)
    reference = tinker.createSystem(**settings)

    for atom in tinker.topology.atoms():
        atom.name = f"{atom.element.symbol}{atom.index % 9 + 1}"
    pdbfile = str(tmp_path / "pair.pdb")
    with open(pdbfile, "w") as f:
        app.PDBFile.writeFile(tinker.topology, tinker.positions, f)
    pdb = app.PDBFile(pdbfile)
    pdb.topology.setPeriodicBoxVectors(box)
    converted = app.ForceField(xml).createSystem(pdb.topology, **settings)
    for system in (reference, converted):
        system.setDefaultPeriodicBoxVectors(*box)

    # the PDB rounds coordinates to 1e-3 A, so both are given the exact ones
    _compare(_energies(reference, tinker.positions), _energies(converted, tinker.positions))


def test_merge_renumbers_clashing_types(tmp_path, capsys):
    """
    Two copies of FSI-, both typed 401-404 by Poltype2: the second moves to
    501-504 (types, classes, multipole frames and polarization groups alike),
    and either copy gives the energy of the original.
    """
    first = tinker2xml(KEY, XYZ, str(tmp_path / "a.xml"), resname="FS1", header=HEADER)
    second = tinker2xml(KEY, XYZ, str(tmp_path / "b.xml"), resname="FS2", header=HEADER)
    merged = merge_amoeba_xml([first, second], str(tmp_path / "merged.xml"))
    assert "move up by 100 (401 -> 501)" in capsys.readouterr().out

    text = open(merged).read()
    for shifted in ('name="501" class="501"', '<Atom name="N5" type="504"',
                    'kz="-502" kx="-502"', 'pgrp1="501" pgrp2="503" pgrp3="504"',
                    'class1="502" class2="504" class3="502" class4="501"'):
        assert shifted in text, shifted
    assert 'name="401" class="401"' in text

    forcefield = app.ForceField(merged)
    tinker = app.TinkerFiles(XYZ, _combined_key(tmp_path))
    residue = next(tinker.topology.residues())
    reference = _energies(
        app.ForceField(first).createSystem(tinker.topology, polarization="direct"),
        tinker.positions,
    )
    for template in ("FS1", "FS2"):
        system = forcefield.createSystem(
            tinker.topology, polarization="direct", residueTemplates={residue: template}
        )
        _compare(reference, _energies(system, tinker.positions))


#: A monatomic cation in Tinker format. The numbers are made up for the
#: test, not taken from an AMOEBA parameter file. The "Li+" label is not an
#: element symbol, which is the point: the element comes from the Z field.
_CATION_KEY = """\
atom        351   92    Li+   "test cation"    3     6.941    0
vdw          92               2.1000     0.0500
polarize    351          0.0300     0.3900
multipole   351    0    0               1.00000
                                        0.00000    0.00000    0.00000
                                        0.00000
                                        0.00000    0.00000
                                        0.00000    0.00000    0.00000
"""


def test_mixture_with_monatomic_ion(tmp_path):
    """
    Two cations and two FSI- converted separately, merged, and compared
    with TinkerFiles reading all parameters at once (PME, mutual).
    """
    ion_key = tmp_path / "ion.key"
    ion_key.write_text(_CATION_KEY)
    ion_xyz = tmp_path / "ion.xyz"
    ion_xyz.write_text("     1  ion\n     1  Li+   0.0  0.0  0.0   351\n")
    ion_xml = tinker2xml(str(ion_key), str(ion_xyz), str(tmp_path / "ion.xml"),
                         resname="LI", header=HEADER)
    assert 'element="Li"' in open(ion_xml).read()
    fsa_xml = tinker2xml(KEY, XYZ, str(tmp_path / "fsa.xml"), resname="FSA", header=HEADER)
    merged = merge_amoeba_xml([fsa_xml, ion_xml], str(tmp_path / "electrolyte.xml"))
    # no clash, so both keep the numbers TinkerFiles knows them by
    assert 'name="351" class="92"' in open(merged).read()

    pair = open(_two_molecules(tmp_path, shift=(1.2, 0.3, 0.2))).read().splitlines()
    pair[0] = "    20  LiFSI"
    pair += ["    19  Li+   0.5   4.0   0.5   351", "    20  Li+  12.0  -1.0   4.5   351"]
    xyz = tmp_path / "box.xyz"
    xyz.write_text("\n".join(pair) + "\n")
    combined = tmp_path / "all.key"
    combined.write_text(open(_combined_key(tmp_path)).read() + _CATION_KEY)

    box = [Vec3(3, 0, 0), Vec3(0, 3, 0), Vec3(0, 0, 3)] * unit.nanometer
    settings = dict(nonbondedMethod=app.PME, nonbondedCutoff=1.2 * unit.nanometer,
                    polarization="mutual", mutualInducedTargetEpsilon=1e-6,
                    ewaldErrorTolerance=1e-5)
    tinker = app.TinkerFiles(str(xyz), str(combined), periodicBoxVectors=box)
    reference = tinker.createSystem(**settings)
    converted = app.ForceField(merged).createSystem(tinker.topology, **settings)
    for system in (reference, converted):
        system.setDefaultPeriodicBoxVectors(*box)
    _compare(_energies(reference, tinker.positions), _energies(converted, tinker.positions))


def test_ec_matches_tinker(tmp_path):
    """
    Ethylene carbonate brings the terms FSI- lacks: in-plane angles and
    out-of-plane bends at the carbonyl carbon, a vdW reduction factor on the
    hydrogens, and several polarization groups, so a lone molecule is
    polarized (mutual) by itself.
    """
    xml = tinker2xml(EC_KEY, EC_XYZ, str(tmp_path / "ec.xml"), resname="EC", header=HEADER)
    tinker = app.TinkerFiles(EC_MIN_XYZ, _combined_key(tmp_path, keys=(EC_KEY,)))
    settings = dict(nonbondedMethod=app.NoCutoff, polarization="mutual",
                    mutualInducedTargetEpsilon=1e-6)
    reference = _energies(tinker.createSystem(**settings), tinker.positions)
    converted = _energies(
        app.ForceField(xml).createSystem(tinker.topology, **settings), tinker.positions
    )
    _compare(reference, converted)
    for name in ("AmoebaInPlaneAngleForce", "AmoebaOutOfPlaneBendForce"):
        assert reference[name] != 0.0, name
    # Tinker's own value; the xyz holds 8 decimals and the dipoles converge
    # to Tinker's polar-eps, hence the looser tolerance
    assert sum(converted.values()) == pytest.approx(EC_MIN_ENERGY, abs=1e-5)


def _stack_xyz(path, placements):
    """One Tinker xyz holding the molecules of ``placements`` (file, shift in A)."""
    lines, offset = [], 0
    for xyzfile, (dx, dy, dz) in placements:
        with open(xyzfile) as f:
            natoms = int(f.readline().split()[0])
            rows = [f.readline().split() for _ in range(natoms)]
        for row in rows:
            bonded = " ".join(str(int(b) + offset) for b in row[6:])
            lines.append(
                f"{int(row[0]) + offset:6d}  {row[1]:3s} {float(row[2]) + dx:12.6f} "
                f"{float(row[3]) + dy:12.6f} {float(row[4]) + dz:12.6f} {row[5]:>6s} {bonded}"
            )
        offset += natoms
    path.write_text(f"{offset}  stacked\n" + "\n".join(lines) + "\n")
    return str(path)


@pytest.mark.parametrize("ec_types", ["501", "401"])
def test_electrolyte_matches_tinkerfiles(tmp_path, ec_types):
    """
    A cation, FSI- and three EC converted separately and merged, under PME
    with mutual polarization, through a PDB topology. With ``"401"`` the EC
    files are retyped 401-405 first, as Poltype2 types a molecule by default,
    so they clash with FSI- and the merge has to move them back to 501-505.
    """
    ec_key, ec_xyz = EC_KEY, EC_XYZ
    if ec_types == "401":
        ec_key, ec_xyz = str(tmp_path / "ec401.key"), str(tmp_path / "ec401.xyz")
        for source, target in ((EC_KEY, ec_key), (EC_XYZ, ec_xyz)):
            text = re.sub(r"(?<![\d.\-])(-?)50([1-5])(?![\d.])", r"\g<1>40\2",
                          open(source).read())
            open(target, "w").write(text)
    ion_key = tmp_path / "ion.key"
    ion_key.write_text(_CATION_KEY)
    ion_xyz = tmp_path / "ion.xyz"
    ion_xyz.write_text("     1  ion\n     1  Li+   0.0  0.0  0.0   351\n")

    xmls = [
        tinker2xml(KEY, XYZ, str(tmp_path / "fsi.xml"), resname="FSI", header=HEADER),
        tinker2xml(ec_key, ec_xyz, str(tmp_path / "ec.xml"), resname="EC", header=HEADER),
        tinker2xml(str(ion_key), str(ion_xyz), str(tmp_path / "ion.xml"), resname="LI",
                   header=HEADER),
    ]
    merged = merge_amoeba_xml(xmls, str(tmp_path / "electrolyte.xml"))
    assert 'name="501" class="501" element="C"' in open(merged).read()

    # the reference always reads EC as 501-505, the numbers merging must restore
    box_xyz = _stack_xyz(tmp_path / "box.xyz", [
        (XYZ, (0, 0, 0)), (EC_XYZ, (8, 0, 0)), (EC_XYZ, (0, 8, 1)),
        (EC_XYZ, (-7, -6, 2)), (str(ion_xyz), (4, 4, -2)),
    ])
    box = [Vec3(3, 0, 0), Vec3(0, 3, 0), Vec3(0, 0, 3)] * unit.nanometer
    settings = dict(nonbondedMethod=app.PME, nonbondedCutoff=1.2 * unit.nanometer,
                    polarization="mutual", mutualInducedTargetEpsilon=1e-6,
                    ewaldErrorTolerance=1e-5)
    tinker = app.TinkerFiles(
        box_xyz, _combined_key(tmp_path, keys=(KEY, EC_KEY), extra=_CATION_KEY),
        periodicBoxVectors=box,
    )
    reference = tinker.createSystem(**settings)

    counts = {}
    for atom in tinker.topology.atoms():
        counts[atom.residue] = counts.get(atom.residue, 0) + 1
        atom.name = f"{atom.element.symbol}{counts[atom.residue]}"
    pdbfile = str(tmp_path / "box.pdb")
    with open(pdbfile, "w") as f:
        app.PDBFile.writeFile(tinker.topology, tinker.positions, f)
    pdb = app.PDBFile(pdbfile)
    pdb.topology.setPeriodicBoxVectors(box)
    converted = app.ForceField(merged).createSystem(pdb.topology, **settings)
    for system in (reference, converted):
        system.setDefaultPeriodicBoxVectors(*box)
    _compare(_energies(reference, tinker.positions), _energies(converted, tinker.positions))


def test_merge_refuses_shared_resname(tmp_path):
    first = tinker2xml(KEY, XYZ, str(tmp_path / "a.xml"), resname="FS1", header=HEADER)
    same_name = tinker2xml(KEY, XYZ, str(tmp_path / "b.xml"), resname="FS1", header=HEADER)
    with pytest.raises(ValueError, match="residue FS1"):
        merge_amoeba_xml([first, same_name], str(tmp_path / "m.xml"))


def test_header_found_next_to_key(tmp_path):
    """The cluster path of the parameters line falls back to the key's folder."""
    shutil.copy(KEY, tmp_path / "final.key")
    with pytest.raises(FileNotFoundError, match="amoebabio18_header.prm"):
        tinker2xml(str(tmp_path / "final.key"), XYZ, str(tmp_path / "a.xml"))
    shutil.copy(HEADER, tmp_path / "amoebabio18_header.prm")
    tinker2xml(str(tmp_path / "final.key"), XYZ, str(tmp_path / "a.xml"))


def test_unsupported_terms_are_refused(tmp_path):
    key = tmp_path / "extra.key"
    key.write_text(open(KEY).read() + "\ntortors 401 402 404 402 401 2 2\n")
    with pytest.raises(NotImplementedError, match="tortors"):
        tinker2xml(str(key), XYZ, str(tmp_path / "a.xml"), header=HEADER)

    key.write_text(open(KEY).read() + "\nchgpen 401 1.0 2.0\n")
    with pytest.raises(NotImplementedError, match="AMOEBA\\+"):
        tinker2xml(str(key), XYZ, str(tmp_path / "a.xml"), header=HEADER)


def test_multipole_scales_openmm_cannot_hold(tmp_path):
    key = tmp_path / "scaled.key"
    key.write_text(open(KEY).read() + "\nmpole-14-scale 0.5\n")
    with pytest.raises(ValueError, match="mpole-14-scale"):
        tinker2xml(str(key), XYZ, str(tmp_path / "a.xml"), header=HEADER)
