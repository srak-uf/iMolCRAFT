import re

from ase import Atoms
from ase.data import chemical_symbols
from openff.toolkit.topology import Molecule

#: Real element symbols. ``ase.data.chemical_symbols`` starts with the dummy
#: entry "X", which must not be matched.
_ELEMENTS = frozenset(chemical_symbols) - {"X"}


def read_mol2(filename):
    """
    Read a mol2 file and return a dictionary with the contents.
    The dictionary will contain the keys "@<TRIPOS>MOLECULE", "@<TRIPOS>ATOM",
    "@<TRIPOS>BOND", and "@<TRIPOS>SUBSTRUCTURE".
    Each key will contain a list of lists, where each inner list represents a line
    in the corresponding section of the mol2 file.
    The first element of each inner list is the line number, and the rest are the
    corresponding values.

    Blank lines and ``#`` comments are ignored, so they are not preserved by a
    read / write round trip.

    Parameters
    ----------
    filename : str
        The name of the mol2 file.

    Returns
    -------
    dict
        A dictionary containing the contents of the mol2 file.

    Raises
    ------
    ValueError
        When a non-empty line appears before the first ``@<TRIPOS>`` header,
        i.e. the file is not a mol2 file or its beginning is truncated.
    """
    mol2_dict = {}
    key_name = None
    with open(filename) as f:
        for lineno, line in enumerate(f, start=1):
            stripped = line.strip()
            # blank lines and "#" comments carry no data and may appear anywhere
            if stripped == "" or stripped.startswith("#"):
                continue
            if line.startswith("@<TRIPOS>"):
                key_name = stripped
                mol2_dict[key_name] = []
                continue
            if key_name is None:
                raise ValueError(
                    f"{filename}:{lineno}: content before the first "
                    f"@<TRIPOS> section header: {stripped!r}. "
                    "Is this a mol2 file?"
                )
            mol2_dict[key_name].append(stripped.split())
    return mol2_dict


def _write_mol2_dict(f, mol2_dict):
    """Write back the section/row structure produced by :func:`read_mol2`."""
    for key, rows in mol2_dict.items():
        f.write(key + "\n")
        for row in rows:
            f.write(" ".join(str(x) for x in row) + "\n")


def _write_mol2_molecule(f, molecule):
    """
    Write an OpenFF molecule, taking the coordinates from its first conformer
    and the charges from its partial charges.
    """
    positions = molecule.conformers[0].magnitude

    f.write("@<TRIPOS>MOLECULE\n")
    f.write("MOL\n")
    f.write(f"{len(molecule.atoms)} {len(molecule.bonds)} 0 0 0\n")
    f.write("SMALL\n")
    f.write("CHARGES\n")
    f.write("\n")
    f.write("\n")

    f.write("@<TRIPOS>ATOM\n")
    for i, atom in enumerate(molecule.atoms):
        xx, yy, zz = positions[i]
        charge = molecule.partial_charges[i].magnitude
        f.write(
            f"{i+1} {atom.symbol} {xx} {yy} {zz} {atom.symbol}  1  "
            f"MOL {charge}\n"
        )

    f.write("@<TRIPOS>BOND\n")
    for i, bond in enumerate(molecule.bonds):
        f.write(
            f"{i+1} {bond.atom1_index+1} "
            f"{bond.atom2_index+1} {bond.bond_order}\n"
        )

    f.write("@<TRIPOS>SUBSTRUCTURE\n")
    f.write("1 MOL 1 TEMP              0 ****  ****    0 ROOT\n")


def write_mol2(filename, data):
    """
    Write a mol2 file from a dictionary or a molecule object.

    Parameters
    ----------
    filename : str
        The name of the file to write to.
    data : dict or openff.toolkit.topology.Molecule
        The data to write. If a dictionary, it should contain the keys
        "@<TRIPOS>MOLECULE", "@<TRIPOS>ATOM", "@<TRIPOS>BOND", and
        "@<TRIPOS>SUBSTRUCTURE".
        If a molecule object, it will be written in the mol2 format.
    """
    if isinstance(data, dict):
        writer = _write_mol2_dict
    elif isinstance(data, Molecule):
        writer = _write_mol2_molecule
    else:
        raise ValueError("Input must be a dictionary or a molecule object.")

    with open(filename, mode="w") as f:
        writer(f, data)


def _element_from_atom_name(atom_name):
    """
    Element symbol behind a mol2 atom name.

    Both writers feeding this package name their atoms after the element and
    append an index to tell copies apart: ``write_mol2`` emits ``S``, ``O``,
    and antechamber emits ``S1``, ``O1``, ``O2``. Dropping the digits and
    taking the longest leading element symbol recovers the element.

    The atom type column is deliberately not consulted. It holds a force field
    atom type whose vocabulary varies (SYBYL ``C.3``, GAFF ``s6``, plain
    element symbols), so mapping it back to an element needs a table of
    special cases, and a wrong type would silently yield a wrong element.

    Names that do not follow the element-first convention are not supported.
    In particular PDB style names such as ``CA`` (alpha carbon) or ``NE2``
    would be read as calcium and neon; no mol2 this package reads uses them.

    Parameters
    ----------
    atom_name : str
        The atom name field of a mol2 ATOM record.

    Returns
    -------
    str
        The element symbol.

    Raises
    ------
    ValueError
        When the name does not start with an element symbol.
    """
    base = re.sub(r"\d", "", atom_name)
    for length in (2, 1):
        candidate = base[:length].capitalize()
        if len(base) >= length and candidate in _ELEMENTS:
            return candidate
    raise ValueError(
        f"Cannot determine the element of the mol2 atom name {atom_name!r}"
    )


def mol2_to_aseatoms(mol2file):
    """
    Read the atom section of a mol2 file into an ``ase.Atoms``.

    The elements come from the atom name; see :func:`_element_from_atom_name`.

    Parameters
    ----------
    mol2file : str
        The name of the mol2 file.

    Returns
    -------
    ase.Atoms
        Atoms object holding the symbols and coordinates of the mol2 file.
    """
    tripos_atom = read_mol2(mol2file)["@<TRIPOS>ATOM"]
    return Atoms(
        symbols=[_element_from_atom_name(atom[1]) for atom in tripos_atom],
        positions=[list(map(float, atom[2:5])) for atom in tripos_atom],
    )
