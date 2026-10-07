"""
LAMMPS exporter.

The data file is written for an input script with ``units real``,
``atom_style full``, ``bond_style harmonic``, ``angle_style harmonic``,
``dihedral_style fourier`` and ``improper_style cvff``. Terms with equal
parameters share one type; atom types are also split by molecular species.
"""

from pathlib import Path

import numpy
import openmm
from openmm import XmlSerializer, unit
from openmm.app import PDBFile

import imolcraft

#: Periodicities allowed by ``improper_style cvff``.
_CVFF_PERIODICITIES = (0, 1, 2, 3, 4, 6)

_KCAL = unit.kilocalorie_per_mole


def exporter_lmp(pdb, system, filename):
    """
    Export a system to LAMMPS format.

    Parameters
    ----------
    pdb : str
        The path to the PDB file.
        The pdb file should contain the topology information.
    system : str
        The path to the system xml file of openmm.
    filename : str
        The filename of the output file (filename.data).
    """
    pdb_omm = PDBFile(pdb)
    with open(system) as f:
        system_omm = XmlSerializer.deserialize(f.read())
    write_lammps(pdb_omm.topology, system_omm, pdb_omm.positions, filename)


def write_lammps(topology, system, positions, filename):
    """
    Write an OpenMM topology / system / positions to ``filename.data``.

    Raises NotImplementedError for systems the data file cannot represent.
    Returns the path of the data file.
    """
    _check_supported(system)
    nonbonded = _forces(system, openmm.NonbondedForce)[0]

    bond_types, bonds = {}, []
    for force in _forces(system, openmm.HarmonicBondForce):
        for i in range(force.getNumBonds()):
            p1, p2, r0, k = force.getBondParameters(i)
            # OpenMM: k/2 (r - r0)^2, LAMMPS harmonic: K (r - r0)^2
            key = _fmt(
                k.value_in_unit(_KCAL / unit.angstrom**2) / 2,
                r0.value_in_unit(unit.angstrom),
            )
            bonds.append((_type_id(bond_types, key), p1, p2))

    angle_types, angles = {}, []
    for force in _forces(system, openmm.HarmonicAngleForce):
        for i in range(force.getNumAngles()):
            p1, p2, p3, theta0, k = force.getAngleParameters(i)
            key = _fmt(
                k.value_in_unit(_KCAL / unit.radian**2) / 2,
                theta0.value_in_unit(unit.degree),
            )
            angles.append((_type_id(angle_types, key), p1, p2, p3))

    dihedral_types, dihedrals, improper_types, impropers = _torsions(system, bonds)

    atoms = list(topology.atoms())
    neighbors = [set() for _ in atoms]
    for _, p1, p2 in bonds:
        neighbors[p1].add(p2)
        neighbors[p2].add(p1)
    molecule_of, species_of = _molecules(neighbors, atoms)

    atom_types, atom_labels, atom_type_of, charges = {}, {}, [], []
    for i, atom in enumerate(atoms):
        q, sigma, epsilon = nonbonded.getParticleParameters(i)
        charges.append(q.value_in_unit(unit.elementary_charge))
        element = atom.element.symbol if atom.element is not None else "X"
        mass = _fmt(system.getParticleMass(i).value_in_unit(unit.dalton))
        coeffs = _fmt(epsilon.value_in_unit(_KCAL), sigma.value_in_unit(unit.angstrom))
        # Different species get different types even with equal parameters,
        # so that e.g. an anion O and a solvent O can be told apart.
        type_id = _type_id(atom_types, (mass, coeffs, element, species_of[i]))
        atom_labels.setdefault(type_id, f"{element} {atom.residue.name}:{atom.name}")
        atom_type_of.append(type_id)

    vectors = topology.getPeriodicBoxVectors()
    if vectors is None:
        vectors = system.getDefaultPeriodicBoxVectors()
    # OpenMM's reduced box is the LAMMPS restricted triclinic cell.
    (ax, _, _), (bx, by, _), (cx, cy, cz) = (
        v.value_in_unit(unit.angstrom) for v in vectors
    )

    sections = [
        ("bond", "harmonic", bonds, bond_types),
        ("angle", "harmonic", angles, angle_types),
        ("dihedral", "fourier", dihedrals, dihedral_types),
        ("improper", "cvff", impropers, improper_types),
    ]
    data_path = Path(f"{filename}.data")
    with open(data_path, "w") as f:
        f.write(f"LAMMPS data file written by iMolCRAFT {imolcraft.__version__}\n\n")
        f.write(f"{len(atoms)} atoms\n")
        for name, _, entries, _ in sections:
            f.write(f"{len(entries)} {name}s\n")
        f.write(f"\n{len(atom_types)} atom types\n")
        for name, _, _, types in sections:
            f.write(f"{len(types)} {name} types\n")

        f.write(f"\n0 {ax:.10g} xlo xhi\n0 {by:.10g} ylo yhi\n0 {cz:.10g} zlo zhi\n")
        if (bx, cx, cy) != (0.0, 0.0, 0.0):
            f.write(f"{bx:.10g} {cx:.10g} {cy:.10g} xy xz yz\n")

        f.write("\nMasses\n\n")
        for (mass, _, _, _), idx in atom_types.items():
            f.write(f"{idx} {mass}  # {atom_labels[idx]}\n")
        f.write("\nPair Coeffs\n\n")
        for (_, coeffs, _, _), idx in atom_types.items():
            f.write(f"{idx} {coeffs}  # {atom_labels[idx]}\n")
        for name, style, _, types in sections:
            if types:
                f.write(f"\n{name.capitalize()} Coeffs # {style}\n\n")
                for coeffs, idx in types.items():
                    f.write(f"{idx} {coeffs}\n")

        f.write("\nAtoms # full\n\n")
        for i, (x, y, z) in enumerate(positions.value_in_unit(unit.angstrom)):
            f.write(
                f"{i + 1} {molecule_of[i]} {atom_type_of[i]} "
                f"{charges[i]:.10g} {x:.10g} {y:.10g} {z:.10g}\n"
            )
        for name, _, entries, _ in sections:
            if entries:
                f.write(f"\n{name.capitalize()}s\n\n")
                for n, (type_id, *members) in enumerate(entries, start=1):
                    ids = " ".join(str(p + 1) for p in members)
                    f.write(f"{n} {type_id} {ids}\n")
    return data_path


def _fmt(*values):
    """Format floats compactly; the strings double as type keys."""
    return " ".join(f"{v:.10g}" for v in values)


def _type_id(types, key):
    """1-based type ID of a parameter key, in first-seen order."""
    return types.setdefault(key, len(types) + 1)


def _forces(system, cls):
    return [f for f in system.getForces() if isinstance(f, cls)]


def _check_supported(system):
    """Refuse systems the data file cannot represent."""
    supported = (
        openmm.NonbondedForce,
        openmm.HarmonicBondForce,
        openmm.HarmonicAngleForce,
        openmm.PeriodicTorsionForce,
        openmm.CMMotionRemover,
    )
    for force in system.getForces():
        if not isinstance(force, supported):
            raise NotImplementedError(
                f"{type(force).__name__} cannot be exported to LAMMPS."
            )
    nonbonded = _forces(system, openmm.NonbondedForce)
    if len(nonbonded) != 1 or (
        nonbonded[0].getNumParticleParameterOffsets()
        or nonbonded[0].getNumExceptionParameterOffsets()
    ):
        raise NotImplementedError(
            "The system needs exactly one NonbondedForce without parameter offsets."
        )
    if any(system.isVirtualSite(i) for i in range(system.getNumParticles())):
        raise NotImplementedError("Virtual sites cannot be exported to LAMMPS.")
    bonded = {
        frozenset(force.getBondParameters(i)[:2])
        for force in _forces(system, openmm.HarmonicBondForce)
        for i in range(force.getNumBonds())
    }
    if any(
        frozenset(system.getConstraintParameters(i)[:2]) not in bonded
        for i in range(system.getNumConstraints())
    ):
        raise NotImplementedError(
            "Some constraints have no bond term; create the system with "
            "constraints=None and rigidWater=False."
        )


def _torsions(system, bonds):
    """
    Split the periodic torsions into ``fourier`` dihedrals and ``cvff`` impropers.

    A torsion along a bonded chain i-j-k-l is a dihedral, with all its terms in
    one type. Any other torsion term k (1 + cos(n phi - phi0)) is a cvff
    improper K [1 + d cos(n phi)], d = +1 / -1 for phi0 = 0 / 180 degrees;
    terms cvff cannot express are written as fourier dihedrals.
    """
    bonded = {frozenset((p1, p2)) for _, p1, p2 in bonds}
    proper_terms = {}
    improper_types, impropers = {}, []
    for force in _forces(system, openmm.PeriodicTorsionForce):
        for i in range(force.getNumTorsions()):
            p1, p2, p3, p4, n, phase, k = force.getTorsionParameters(i)
            k = k.value_in_unit(_KCAL)
            if k == 0.0:
                continue
            phase = phase.value_in_unit(unit.degree)
            chain = all(
                frozenset(pair) in bonded for pair in ((p1, p2), (p2, p3), (p3, p4))
            )
            sign = _cvff_sign(phase)
            if chain or sign is None or n not in _CVFF_PERIODICITIES:
                quartet = min((p1, p2, p3, p4), (p4, p3, p2, p1))
                proper_terms.setdefault(quartet, []).append((n, phase, k))
            else:
                key = f"{_fmt(k)} {sign} {n}"
                impropers.append((_type_id(improper_types, key), p1, p2, p3, p4))

    dihedral_types, dihedrals = {}, []
    for quartet, terms in proper_terms.items():
        terms.sort()
        key = f"{len(terms)} " + " ".join(_fmt(k, n, phase) for n, phase, k in terms)
        dihedrals.append((_type_id(dihedral_types, key), *quartet))
    return dihedral_types, dihedrals, improper_types, impropers


def _cvff_sign(phase):
    """The cvff ``d`` (+1 / -1) of a phase in degrees, None if neither fits."""
    for d, ref in ((1, 0.0), (-1, 180.0)):
        if numpy.isclose(numpy.cos(numpy.deg2rad(phase - ref)), 1.0, atol=1e-9):
            return d
    return None


def _molecules(neighbors, atoms):
    """
    Molecule ID and species ID (both 1-based) of every atom.

    Molecules are the connected components of the bond graph. Two molecules
    are the same species when their element-labelled bond graphs match under
    Weisfeiler-Lehman refinement, independent of residue names and atom order.
    """
    n_atoms = len(neighbors)
    molecule_of = [0] * n_atoms
    members = []
    for start in range(n_atoms):
        if molecule_of[start]:
            continue
        members.append([])
        molecule_of[start] = len(members)
        stack = [start]
        while stack:
            i = stack.pop()
            members[-1].append(i)
            for j in neighbors[i]:
                if not molecule_of[j]:
                    molecule_of[j] = len(members)
                    stack.append(j)

    labels = [
        atom.element.symbol if atom.element is not None else atom.name
        for atom in atoms
    ]
    for _ in range(max(map(len, members), default=0)):
        new = [
            (labels[i], tuple(sorted(labels[j] for j in neighbors[i])))
            for i in range(n_atoms)
        ]
        compact = {label: k for k, label in enumerate(sorted(set(new), key=repr))}
        new = [compact[label] for label in new]
        if len(set(new)) == len(set(labels)):
            break
        labels = new

    species_ids = {}
    species_of_molecule = [
        _type_id(species_ids, tuple(sorted(repr(labels[i]) for i in m)))
        for m in members
    ]
    species_of = [species_of_molecule[m - 1] for m in molecule_of]
    return molecule_of, species_of
