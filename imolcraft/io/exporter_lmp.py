"""
LAMMPS exporter.

Translates an OpenMM System (``system.xml``) and its structure (PDB) into a
LAMMPS data file, term by term:

========================  ==============================================
OpenMM                    LAMMPS data file
========================  ==============================================
NonbondedForce            Pair Coeffs (epsilon, sigma), charges in Atoms
HarmonicBondForce         Bond Coeffs (``harmonic``)
HarmonicAngleForce        Angle Coeffs (``harmonic``)
PeriodicTorsionForce      Dihedral Coeffs (``fourier``) and
                          Improper Coeffs (``cvff``)
========================  ==============================================

The data file is written for an input script with ``units real``,
``atom_style full``, ``bond_style harmonic``, ``angle_style harmonic``,
``dihedral_style fourier`` and ``improper_style cvff``. The pair style,
``special_bonds`` (1-4 scaling) and kspace are left to the input script.

Terms with equal parameters share one type, so the number of types does not
grow with the number of molecules; atom types are also split by molecular
species.
"""

from pathlib import Path

import numpy
import openmm
from openmm import XmlSerializer, unit
from openmm.app import PDBFile

import imolcraft

#: Periodicities allowed by ``improper_style cvff``.
_CVFF_PERIODICITIES = (0, 1, 2, 3, 4, 6)

#: Energy unit of LAMMPS ``units real``; lengths are in Angstrom.
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

    # Every term below gets a type ID from a dict {parameter string: type ID},
    # so terms with equal parameters share one type (see _type_id).

    # --- Bonds: OpenMM k/2 (r - r0)^2, LAMMPS harmonic K (r - r0)^2 -> K = k/2
    bond_types, bonds = {}, []  # bonds: (type ID, atom1, atom2)
    for force in _forces(system, openmm.HarmonicBondForce):
        for i in range(force.getNumBonds()):
            p1, p2, r0, k = force.getBondParameters(i)
            key = _fmt(
                k.value_in_unit(_KCAL / unit.angstrom**2) / 2,
                r0.value_in_unit(unit.angstrom),
            )
            bonds.append((_type_id(bond_types, key), p1, p2))

    # --- Angles: same k/2 convention as the bonds; theta0 in degrees
    angle_types, angles = {}, []  # angles: (type ID, atom1, atom2, atom3)
    for force in _forces(system, openmm.HarmonicAngleForce):
        for i in range(force.getNumAngles()):
            p1, p2, p3, theta0, k = force.getAngleParameters(i)
            key = _fmt(
                k.value_in_unit(_KCAL / unit.radian**2) / 2,
                theta0.value_in_unit(unit.degree),
            )
            angles.append((_type_id(angle_types, key), p1, p2, p3))

    # --- Torsions: split into proper dihedrals and impropers
    dihedral_types, dihedrals, improper_types, impropers = _torsions(system, bonds)

    # --- Molecules: bonded fragments, and which of them are the same species
    atoms = list(topology.atoms())
    neighbors = [set() for _ in atoms]  # bonded neighbours of every atom
    for _, p1, p2 in bonds:
        neighbors[p1].add(p2)
        neighbors[p2].add(p1)
    molecule_of, species_of = _molecules(neighbors, atoms)

    # --- Atom types: equal mass, LJ parameters, element and species.
    # Charges are per atom (Atoms section), so they are not part of the type.
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
        # Label of the first atom of each type, written as a comment
        atom_labels.setdefault(type_id, f"{element} {atom.residue.name}:{atom.name}")
        atom_type_of.append(type_id)

    # --- Box: the PDB CRYST1 record, or else the default box of the system
    vectors = topology.getPeriodicBoxVectors()
    if vectors is None:
        vectors = system.getDefaultPeriodicBoxVectors()
    # OpenMM's reduced box (a along x, b in the xy plane) is the LAMMPS
    # restricted triclinic cell: lx = ax, ly = by, lz = cz, xy = bx, xz = cx,
    # yz = cy.
    (ax, _, _), (bx, by, _), (cx, cy, cz) = (
        v.value_in_unit(unit.angstrom) for v in vectors
    )

    # --- Write the data file
    # (name, LAMMPS style, terms, types) of the four bonded sections
    sections = [
        ("bond", "harmonic", bonds, bond_types),
        ("angle", "harmonic", angles, angle_types),
        ("dihedral", "fourier", dihedrals, dihedral_types),
        ("improper", "cvff", impropers, improper_types),
    ]
    data_path = Path(f"{filename}.data")
    with open(data_path, "w") as f:
        # Header: counts, type counts and the box
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

        # Parameters, one line per type. The "# harmonic" etc. after a section
        # name is a style hint LAMMPS checks against the input script.
        f.write("\nMasses\n\n")
        for (mass, _, _, _), idx in atom_types.items():
            f.write(f"{idx} {mass}  # {atom_labels[idx]}\n")
        f.write("\nPair Coeffs\n\n")
        for (_, coeffs, _, _), idx in atom_types.items():
            f.write(f"{idx} {coeffs}  # {atom_labels[idx]}\n")
        for name, style, _, types in sections:
            if types:  # an empty section is not written
                f.write(f"\n{name.capitalize()} Coeffs # {style}\n\n")
                for coeffs, idx in types.items():
                    f.write(f"{idx} {coeffs}\n")

        # Topology. LAMMPS IDs are 1-based, OpenMM indices 0-based.
        # Atoms line (atom_style full): atom ID, molecule ID, type, charge, x y z
        f.write("\nAtoms # full\n\n")
        for i, (x, y, z) in enumerate(positions.value_in_unit(unit.angstrom)):
            f.write(
                f"{i + 1} {molecule_of[i]} {atom_type_of[i]} "
                f"{charges[i]:.10g} {x:.10g} {y:.10g} {z:.10g}\n"
            )
        # Bonds / Angles / ... line: term ID, type, atom IDs
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
    """
    1-based type ID of a parameter key, in first-seen order.

    A new key gets the next ID; a key seen before gets its existing ID.
    """
    return types.setdefault(key, len(types) + 1)


def _forces(system, cls):
    """All forces of the system of the given class."""
    return [f for f in system.getForces() if isinstance(f, cls)]


def _check_supported(system):
    """
    Refuse systems the data file cannot represent.

    Writing such a system anyway would silently give LAMMPS a different force
    field, so these raise NotImplementedError instead.
    """
    # Only these forces have a data-file counterpart; CMMotionRemover adds no
    # energy and is dropped.
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
    # Parameter offsets (used e.g. for free-energy setups) change the
    # parameters at run time, which a data file cannot store.
    nonbonded = _forces(system, openmm.NonbondedForce)
    if len(nonbonded) != 1 or (
        nonbonded[0].getNumParticleParameterOffsets()
        or nonbonded[0].getNumExceptionParameterOffsets()
    ):
        raise NotImplementedError(
            "The system needs exactly one NonbondedForce without parameter offsets."
        )
    # Massless virtual sites (e.g. the M site of TIP4P water)
    if any(system.isVirtualSite(i) for i in range(system.getNumParticles())):
        raise NotImplementedError("Virtual sites cannot be exported to LAMMPS.")
    # A constrained bond (constraints=HBonds, rigid water) has no bond term in
    # OpenMM, so it would be missing from the Bonds section.
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

    OpenMM keeps proper and improper torsions in one PeriodicTorsionForce;
    LAMMPS needs them in separate sections.

    A torsion along a bonded chain i-j-k-l is a dihedral, with all its terms in
    one type. Any other torsion term k (1 + cos(n phi - phi0)) is a cvff
    improper K [1 + d cos(n phi)], d = +1 / -1 for phi0 = 0 / 180 degrees;
    terms cvff cannot express are written as fourier dihedrals, which LAMMPS
    evaluates the same way whether or not the atoms form a chain.
    """
    bonded = {frozenset((p1, p2)) for _, p1, p2 in bonds}
    proper_terms = {}  # {atom quartet: [(n, phase, k), ...]}
    improper_types, impropers = {}, []
    for force in _forces(system, openmm.PeriodicTorsionForce):
        for i in range(force.getNumTorsions()):
            p1, p2, p3, p4, n, phase, k = force.getTorsionParameters(i)
            k = k.value_in_unit(_KCAL)
            if k == 0.0:  # no energy; GAFF has some zero-barrier terms
                continue
            phase = phase.value_in_unit(unit.degree)
            chain = all(
                frozenset(pair) in bonded for pair in ((p1, p2), (p2, p3), (p3, p4))
            )
            sign = _cvff_sign(phase)
            if chain or sign is None or n not in _CVFF_PERIODICITIES:
                # i-j-k-l and l-k-j-i are the same dihedral; store it once
                quartet = min((p1, p2, p3, p4), (p4, p3, p2, p1))
                proper_terms.setdefault(quartet, []).append((n, phase, k))
            else:
                # cvff Improper Coeffs: K d n
                key = f"{_fmt(k)} {sign} {n}"
                impropers.append((_type_id(improper_types, key), p1, p2, p3, p4))

    # fourier Dihedral Coeffs: m, then K n phase for each of the m terms
    dihedral_types, dihedrals = {}, []
    for quartet, terms in proper_terms.items():
        terms.sort()  # same terms in another order give the same type
        key = f"{len(terms)} " + " ".join(_fmt(k, n, phase) for n, phase, k in terms)
        dihedrals.append((_type_id(dihedral_types, key), *quartet))
    return dihedral_types, dihedrals, improper_types, impropers


def _cvff_sign(phase):
    """The cvff ``d`` (+1 / -1) of a phase in degrees, None if neither fits."""
    for d, ref in ((1, 0.0), (-1, 180.0)):
        # compare on the circle, so that e.g. 180 and -180 both give d = -1
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
    # Molecule IDs: walk the bond graph from every atom not yet assigned
    n_atoms = len(neighbors)
    molecule_of = [0] * n_atoms  # 0 = not assigned yet
    members = []  # atom indices of every molecule
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

    # Species: start from element labels and repeatedly fold the labels of
    # the bonded neighbours into each atom's label, so that a label comes to
    # describe the atom's bonded environment. Stop once no label splits any
    # further (at most as many rounds as atoms in the largest molecule).
    labels = [
        atom.element.symbol if atom.element is not None else atom.name
        for atom in atoms
    ]
    for _ in range(max(map(len, members), default=0)):
        new = [
            (labels[i], tuple(sorted(labels[j] for j in neighbors[i])))
            for i in range(n_atoms)
        ]
        # renumber the labels as small integers
        compact = {label: k for k, label in enumerate(sorted(set(new), key=repr))}
        new = [compact[label] for label in new]
        if len(set(new)) == len(set(labels)):
            break
        labels = new

    # Molecules with the same multiset of atom labels are the same species
    species_ids = {}
    species_of_molecule = [
        _type_id(species_ids, tuple(sorted(repr(labels[i]) for i in m)))
        for m in members
    ]
    species_of = [species_of_molecule[m - 1] for m in molecule_of]
    return molecule_of, species_of
