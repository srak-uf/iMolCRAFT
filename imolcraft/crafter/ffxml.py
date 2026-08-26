import os
import shutil
from typing import Any, List, Optional

import numpy as np
import openmmforcefields
from dmff import Hamiltonian
from openff.toolkit import Molecule, Quantity
from openff.units import unit

import imolcraft
from imolcraft.io.mol2 import read_mol2

#: SMILES of the anion handled by the dedicated PF6 template.
_PF6_SMILES = "F[P-](F)(F)(F)(F)F"

#: OpenMM writes this in the element column of the ATOM records of extra
#: particles (``PDBFile.writeFile`` argument ``extraParticleIdentifier``).
_EXTRA_PARTICLE_IDENTIFIER = "EP"

#: Ion parameter library used unless ``iontype`` says otherwise, and the
#: fallback for every element a named ion force field does not cover.
#: Read relative to the ``ffxml`` directory of openmmforcefields.
DEFAULT_ION_FFXML = "amber/ions/ionsff99_tip3p.xml"

#: Ion force fields shipped with iMolCRAFT, selectable by name through the
#: ``iontype`` key of the input YAML.  The names are matched case
#: insensitively and each maps the elements it covers to a file name under
#: ``imolcraft/data``.  An element that is absent here falls back to
#: ``DEFAULT_ION_FFXML``, so ``iontype: Madrid`` reparameterizes lithium
#: alone and leaves every other ion on the Amber library.
#: The Li+ parameters come from https://doi.org/10.1021/acs.jpcb.3c05591
BUNDLED_ION_FFXML = {
    "gmanr": {"Li": "Gmanr_Li.xml"},
    "madrid": {"Li": "Madrid_Li.xml"},
    "wu-wick": {"Li": "Wu-Wick_Li.xml"},
    "smm": {"Li": "SMM_Li.xml"},
}


def resolve_ion_ffxml(ion_ffxml: Optional[str], symbol: str) -> str:
    """
    Turn the ``iontype`` setting into the path of the XML file to use for the
    ion ``symbol``.

    ``ion_ffxml`` may be ``None``, in which case ``DEFAULT_ION_FFXML`` is
    used, the name of an ion force field bundled with iMolCRAFT
    (``BUNDLED_ION_FFXML``, e.g. ``Madrid``), or a path. A named force field
    is consulted for ``symbol`` alone: it applies to the elements it covers
    and every other element falls back to ``DEFAULT_ION_FFXML``. A path is
    used as it stands when it exists, and is otherwise read relative to the
    ``ffxml`` directory of openmmforcefields, as before.
    """
    if ion_ffxml is None:
        ion_ffxml = DEFAULT_ION_FFXML

    bundled = BUNDLED_ION_FFXML.get(ion_ffxml.lower())
    if bundled is not None:
        filename = bundled.get(symbol)
        if filename is not None:
            return os.path.join(imolcraft.__path__[0], "data", filename)
        # the named force field says nothing about this element
        ion_ffxml = DEFAULT_ION_FFXML

    if os.path.exists(ion_ffxml):
        return ion_ffxml
    return os.path.join(
        os.path.dirname(openmmforcefields.__file__), "ffxml", ion_ffxml
    )


def get_element_fromtype(ptype, ff):
    for at in ff.ffinfo["AtomTypes"]:
        if at["class"] == ptype:
            return at["element"]


def check_vsite(ffxml: str) -> int:
    """
    Check if the force field XML file contains virtual sites.
    """
    ff = Hamiltonian(ffxml)
    return sum(
        len(residue["vsites"])
        for residue in ff.ffinfo["Residues"]
        if "vsites" in residue.keys()
    )


def _is_vsite_record(line: str) -> bool:
    """
    True for a PDB record describing a virtual site.

    Virtual sites are recognised by the ``EP`` element written by OpenMM, which
    is the last whitespace-separated field of the record. Lines carrying no
    field at all (blank lines) are never virtual sites.
    """
    fields = line.split()
    return bool(fields) and fields[-1] == _EXTRA_PARTICLE_IDENTIFIER


def delvsite_pdb(pdbfile: str) -> None:
    """
    Delete virtual sites from the PDB file.
    """
    with open(pdbfile, "r") as f:
        lines = f.readlines()
    lines = [line for line in lines if not _is_vsite_record(line)]
    with open(pdbfile, "w") as f:
        f.writelines(lines)


def _molecule2aseatoms(molecule):
    """Convert an OpenFF molecule (with a conformer) to an ``ase.Atoms``."""
    from ase import Atoms
    from ase.data import chemical_symbols

    positions = molecule.conformers[0].magnitude
    symbols = [chemical_symbols[atom.atomic_number] for atom in molecule.atoms]
    return Atoms(symbols=symbols, positions=positions)


def _get_element_angles(atoms, elem1, elem2, elem3, rcut=2.0):
    """
    Get the elem1-elem2-elem3 angles of the given elements in the molecule.

    Only triplets whose two bond distances are both shorter than ``rcut``
    are returned.
    """
    indices = [
        [i for i, atom in enumerate(atoms) if atom.symbol == elem]
        for elem in (elem1, elem2, elem3)
    ]

    angles = []
    angles_idx = []
    for i in indices[0]:
        for j in indices[1]:
            for k in indices[2]:
                if i == j or j == k or i >= k:
                    continue
                # Calculate the distance between the atoms
                if (
                    atoms.get_distance(i, j) < rcut
                    and atoms.get_distance(j, k) < rcut
                ):
                    angles.append(atoms.get_angle(i, j, k, mic=True))
                    angles_idx.append((i, j, k))
    return np.array(angles), np.array(angles_idx)


def _write_pf6_xml(molecule_off: Molecule, ffxml: str) -> None:
    """
    Write a PF6 force field XML with charges taken from ``molecule_off``.

    The bundled PF6 template distinguishes the F-P-F angles by whether they
    are linear (~180 deg) or square (~90 deg), so the angle keys are rewritten
    to match the atom ordering of the given molecule.
    """
    pf6xml = os.path.join(imolcraft.__path__[0], "data", "PF6_gaff.xml")
    ff = Hamiltonian(pf6xml)
    atoms = _molecule2aseatoms(molecule_off)
    angles, angles_idx = _get_element_angles(atoms, "F", "P", "F")
    angles_180_idx = np.where(np.abs(angles - 180) < 10)[0]

    P_idx = [i for i, atom in enumerate(molecule_off.atoms) if atom.atomic_number == 15]
    F_idx = [i for i, atom in enumerate(molecule_off.atoms) if atom.atomic_number == 9]

    particles = ff.ffinfo["Residues"][0]["particles"]
    # P charge
    particles[0]["charge"] = molecule_off.partial_charges[P_idx[0]].magnitude
    # F charge
    for i, charge in enumerate(molecule_off.partial_charges[F_idx].magnitude):
        particles[i + 1]["charge"] = charge

    angle_keys = ff.generators["HarmonicAngleForce"].angle_keys
    angles_90 = np.delete(angles_idx, angles_180_idx, axis=0)
    # the first three keys are the linear angles, the rest are the square ones
    for offset, group in ((0, angles_idx[angles_180_idx]), (3, angles_90)):
        for i, ag in enumerate(group, offset):
            ii = F_idx.index(ag[0])
            kk = F_idx.index(ag[2])
            angle_keys[i] = (f"f{ii+1}_pf6", angle_keys[i][1], f"f{kk+1}_pf6")

    ff.renderXML(ffxml)


def _write_ion_xml(molecule: Molecule, ion_ffxml: str, outxml: str) -> None:
    """
    Write a force field XML for a single-atom ion.

    The bundled ion library covers many ions at once, so it is copied and then
    stripped down to the single atom type matching ``molecule``, whose charge
    is replaced by the calculated partial charge.
    """
    shutil.copy(ion_ffxml, outxml)
    ff = Hamiltonian(outxml)
    symbol = molecule.atoms[0].symbol

    target_ptype = None
    for res in ff.ffinfo["Residues"]:
        if len(res["particles"]) == 1:
            ptype = res["particles"][0]["type"]
            if get_element_fromtype(ptype, ff) == symbol:
                res["particles"][0]["charge"] = float(
                    molecule.partial_charges[0].magnitude
                )
                target_ptype = ptype

    if target_ptype is None:
        raise ValueError(
            f"No parameters for the ion {symbol} in {ion_ffxml}. "
            "Please choose an ion force field that covers this element."
        )

    for ii, at in enumerate(ff.ffinfo["AtomTypes"]):
        if at["class"] == target_ptype:
            ff.ffinfo["AtomTypes"] = [ff.ffinfo["AtomTypes"][ii]]
    for ii, res in enumerate(ff.ffinfo["Residues"]):
        if res["particles"][0]["type"] == target_ptype:
            ff.ffinfo["Residues"] = [ff.ffinfo["Residues"][ii]]

    nb_nodes = ff.ffinfo["Forces"]["NonbondedForce"]["node"]
    for nb in nb_nodes:
        if nb["name"] == "UseAttributeFromResidue":
            atrib_def = nb
            break
    for nb in nb_nodes:
        if nb["name"] == "Atom" and nb["attrib"]["type"] == target_ptype:
            ff.ffinfo["Forces"]["NonbondedForce"]["node"] = [atrib_def, nb]
            break

    # ``renderXML`` writes the Lennard-Jones parameters back positionally,
    # walking ``atom_keys`` against the parameter arrays, so both have to be
    # narrowed to the target entry. Truncating ``atom_keys`` alone would
    # write the first entry of the library, whatever ion that happens to be.
    nb_generator = ff.generators["NonbondedForce"]
    target_index = nb_generator.atom_keys.index(target_ptype)
    for holder in (ff.paramset.parameters, ff.paramset.mask):
        entry = holder["NonbondedForce"]
        for name in ("sigma", "epsilon"):
            entry[name] = entry[name][target_index:target_index + 1]
    nb_generator.atom_keys = [target_ptype]
    ff.renderXML(outxml)


def _fill_charges_and_conformer(molecule: Molecule) -> None:
    """Fill in partial charges / conformer from the molecule's mol2 file."""
    mol2_dict = read_mol2(molecule.mol2file)
    if molecule.partial_charges is None:
        partial_charges = [atominfo[8] for atominfo in mol2_dict["@<TRIPOS>ATOM"]]
        molecule.partial_charges = Quantity(partial_charges, unit.elementary_charge)
    if molecule.conformers is None:
        positions = [atom[2:5] for atom in mol2_dict["@<TRIPOS>ATOM"]]
        molecule.add_conformer(Quantity(positions, unit.angstrom))


def gafftemplate2xml(
        mmm: List[Molecule],
        fftemplate_gen: Any,
        ion_ffxml: Optional[str] = None
        ) -> List[str]:
    """
    Generate GAFF XML files for a list of molecules.

    Parameters
    ----------
    mmm : list
        List of molecules to generate XML files.
    fftemplate_gen : openmmforcefields.GAFFTemplateGenerator
        GAFFTemplateGenerator object.
    ion_ffxml : str, optional
        Name of a bundled ion force field (``BUNDLED_ION_FFXML``), which is
        applied to the elements it covers only, or the path to an ion XML
        file, either existing or relative to the ``ffxml`` directory of
        openmmforcefields. If None, ``DEFAULT_ION_FFXML`` is used.

    Returns
    -------
    list
        List of paths to the generated XML files.
    """
    ffxmlfiles = []
    for i, molecule in enumerate(mmm):
        outxml = f"./gaffxml_{i}.xml"
        _fill_charges_and_conformer(molecule)

        if molecule == Molecule.from_smiles(_PF6_SMILES):
            _write_pf6_xml(molecule, f"gaffxml_{i}.xml")
        elif molecule.n_atoms > 1 or (
            molecule.total_charge == 0 and molecule.n_atoms == 1
        ):
            with open(f"gaffxml_{i}.xml", "w") as f:
                f.write(fftemplate_gen.generate_residue_template(molecule))
        else:  # single-atom ion
            _write_ion_xml(
                molecule,
                resolve_ion_ffxml(ion_ffxml, molecule.atoms[0].symbol),
                outxml,
            )

        ffxmlfiles.append(outxml)

    return ffxmlfiles


def merge_xml(ffxml_list, outxml):
    """
    Merge multiple XML files into one.

    Parameters
    ----------
    ffxml_list : list
        List of XML files to be merged.
    outxml : str
        Name of the output XML file.
    """
    ff = Hamiltonian(*ffxml_list)

    def _is_charge_attrib(node):
        return (
            "name" in node
            and "attrib" in node
            and node["name"] == "UseAttributeFromResidue"
            and node["attrib"]["name"] == "charge"
        )

    # keep only the first ``UseAttributeFromResidue name="charge"`` node
    kept = []
    seen_charge_attrib = False
    for node in ff.ffinfo["Forces"]["NonbondedForce"]["node"]:
        if _is_charge_attrib(node):
            if seen_charge_attrib:
                continue
            seen_charge_attrib = True
        kept.append(node)
    ff.ffinfo["Forces"]["NonbondedForce"]["node"] = kept

    os.makedirs("xmlfiles", exist_ok=True)
    ff.renderXML(os.path.join("xmlfiles", outxml))

    return os.path.join("xmlfiles", outxml)
