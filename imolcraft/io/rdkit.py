import copy

import numpy as np
from ase import Atoms
from rdkit import Chem
from rdkit.Chem import rdDepictor, rdDetermineBonds


def atoms2rdkit(atoms: Atoms, nc: int = 0, il_assign: bool = True):
    """
    Convert a list of atoms to an RDKit molecule.

    Parameters
    ----------
    atoms : ase.Atoms
        The atoms to convert.
    nc : int
        Net charge of the molecule. Python and numpy integers are both
        accepted; anything else raises TypeError.
    il_assign : bool
        If True, fix the formal charges and bond orders of the ionic liquid
        ions RDKit perceives incorrectly.

    Returns
    -------
    mol : rdkit.Chem.rdchem.Mol
        An RDKit molecule object.
    mol2d : rdkit.Chem.rdchem.Mol
        An RDKit 2D molecule object.
    il_dict : dict or None
        Atom indices of the recognised ion, or None. See :func:`_il_assign`.
    """
    if not isinstance(nc, (int, np.integer)):
        raise TypeError(f"nc must be an integer, but got {nc!r}")
    # RDKit's bindings reject numpy integers, so normalise once here rather
    # than at every call site downstream
    nc = int(nc)

    element_symbols = atoms.get_chemical_symbols()

    if len(atoms) > 1:
        mol = _molecule_from_positions(atoms, element_symbols)
        mol = _determine_bonds(mol, nc)
        mol2d = copy.deepcopy(mol)
        rdDepictor.Compute2DCoords(mol2d)
    else:
        # a lone atom carries no connectivity, so build it straight from SMILES
        mol = Chem.MolFromSmiles(_monatomic_smiles(element_symbols[0], nc))
        mol2d = mol

    il_dict = None
    if il_assign:
        il_dict = _il_assign(mol, nc, mol2d)
    return mol, mol2d, il_dict


def _molecule_from_positions(atoms: Atoms, element_symbols):
    """Build a bondless RWMol holding the elements and coordinates of ``atoms``."""
    mol = Chem.RWMol()
    conf = Chem.Conformer(len(atoms))
    for i in range(len(atoms)):
        atom_idx = mol.AddAtom(Chem.Atom(element_symbols[i]))
        conf.SetAtomPosition(atom_idx, atoms.positions[i])
    mol.AddConformer(conf)
    return mol


def _determine_bonds(mol, nc):
    """
    Perceive bonds and bond orders from the geometry.

    When the bond orders cannot be assigned consistently with the net charge,
    fall back to connectivity only and warn: the resulting molecule still has
    the right topology, but its SMILES may be wrong.
    """
    mol_tmp = copy.deepcopy(mol)
    try:
        rdDetermineBonds.DetermineBonds(mol_tmp, charge=nc)
        return mol_tmp
    except Exception:
        from rdkit.Chem import Draw

        mol_tmp = copy.deepcopy(mol)
        rdDetermineBonds.DetermineConnectivity(mol_tmp, charge=nc)
        print("Warning: The bonds may be assigned incorrectly. Please check.")
        print("Warning: This may give wrong SMILES")
        Draw.MolToImage(mol_tmp)  # Necessary
        return mol_tmp


def _monatomic_smiles(symbol, nc):
    """SMILES of a single atom carrying the net charge ``nc``."""
    if nc > 0:
        return f"[{symbol}+{nc}]"
    if nc < 0:
        return f"[{symbol}-{-nc}]"
    return f"[{symbol}]"


def _set_formal_charge(atom, atoms2d, index, charge):
    """Set a formal charge on the 3D atom and on its 2D counterpart."""
    atom.SetFormalCharge(charge)
    if atoms2d is not None:
        atoms2d[index].SetFormalCharge(charge)


def _bond_end_with_symbol(bond, symbol):
    """
    Index of the bond end whose element is ``symbol``, or None if neither is.

    The end atom is tested before the begin atom, so a bond between two atoms
    of that same element resolves to its end atom.
    """
    if bond.GetEndAtom().GetSymbol() == symbol:
        return bond.GetEndAtomIdx()
    if bond.GetBeginAtom().GetSymbol() == symbol:
        return bond.GetBeginAtomIdx()
    return None


def _central_ion_indices(atoms, center_symbol, ligand_symbol, n_ligands):
    """
    Locate the centre of an isolated ``center(ligand)n`` ion such as PF6 or
    ClO4. This only inspects the molecule; nothing is modified.

    The molecule must hold exactly one atom of ``center_symbol`` and all of
    its neighbours must be terminal ``ligand_symbol`` atoms. Requiring the
    ligands to be terminal is what tells the bare ion apart from an ester or
    another substituted derivative, whose bridging atom carries a further
    substituent (methyl perchlorate has four oxygens on its chlorine too, but
    one of them also bonds a carbon).

    Parameters
    ----------
    atoms : sequence of rdkit.Chem.rdchem.Atom
        The atoms of the molecule.
    center_symbol : str
        Element of the central atom.
    ligand_symbol : str
        Element of the surrounding atoms.
    n_ligands : int
        How many ligands the ion has.

    Returns
    -------
    (center_index, ligand_indices) : (int, list) or None
        None when the molecule is not that ion.
    """
    centers = [i for i, atom in enumerate(atoms) if atom.GetSymbol() == center_symbol]
    if len(centers) != 1:
        return None

    neighbors = list(atoms[centers[0]].GetNeighbors())
    if len(neighbors) != n_ligands:
        return None
    if not all(
        neighbor.GetSymbol() == ligand_symbol and neighbor.GetDegree() == 1
        for neighbor in neighbors
    ):
        return None
    return centers[0], sorted(neighbor.GetIdx() for neighbor in neighbors)


def _assign_pf6(mol, nc, mol2d=None):
    """
    Check if the molecule is PF6 and, if so, set the formal charges RDKit
    cannot infer: the phosphorus carries the charge and the fluorines stay
    neutral.

    The molecule is left untouched unless it really is PF6.
    """
    if nc >= 0:
        return None

    atoms = mol.GetAtoms()
    found = _central_ion_indices(atoms, "P", "F", 6)
    if found is None:
        return None

    p_index, f_index = found
    atoms2d = mol2d.GetAtoms() if mol2d is not None else None
    for i in f_index:
        _set_formal_charge(atoms[i], atoms2d, i, 0)
    _set_formal_charge(atoms[p_index], atoms2d, p_index, -1)
    return {"PF6_P": [p_index], "PF6_F": f_index}


def _assign_clo4(mol, nc, mol2d=None):
    """
    Check if the molecule is ClO4 and, if so, set the formal charges of the
    hypervalent chlorine and of its four oxygens.

    The molecule is left untouched unless it really is ClO4.
    """
    if nc >= 0:
        return None

    atoms = mol.GetAtoms()
    found = _central_ion_indices(atoms, "Cl", "O", 4)
    if found is None:
        return None

    cl_index, o_index = found
    atoms2d = mol2d.GetAtoms() if mol2d is not None else None
    for i in o_index:
        _set_formal_charge(atoms[i], atoms2d, i, -1)
    _set_formal_charge(atoms[cl_index], atoms2d, cl_index, 3)
    return {"ClO4_Cl": [cl_index], "ClO4_O": o_index}


def _find_fsalike_n_s(atoms):
    """
    Locate the FSA-like motif: a two-coordinated nitrogen bonded to a
    hexavalent sulfur.

    Returns
    -------
    (n_index, s_index) : (list, list)
    """
    fsalike_Nindex = []
    fsalike_Sindex = []
    for i, atom in enumerate(atoms):
        if len(atom.GetBonds()) != 2 or atom.GetSymbol() != "N":
            continue
        for bond in atom.GetBonds():
            s_index = _bond_end_with_symbol(bond, "S")
            if s_index is not None and atoms[s_index].GetTotalValence() == 6:
                fsalike_Nindex.append(i)
                fsalike_Sindex.append(s_index)
    return list(set(fsalike_Nindex)), list(set(fsalike_Sindex))


def _assign_fsalike(mol, nc, mol2d=None):
    """
    Check if the molecule is FSA-like and, if so, redraw the resonance
    structure RDKit picked: the charge goes on the nitrogen, its bonds become
    single, and the S-O bonds become double so the oxygens stay neutral.
    """
    if nc >= 0:
        return None

    atoms = mol.GetAtoms()
    atoms2d = mol2d.GetAtoms() if mol2d is not None else None

    fsalike_Nindex, fsalike_Sindex = _find_fsalike_n_s(atoms)
    if len(fsalike_Nindex) == 0 or len(fsalike_Sindex) == 0:
        return None

    # Sに結合しているO原子のindexをfsalike_Oindexに追加
    fsalike_Oindex = []
    for i in fsalike_Sindex:
        for bond in atoms[i].GetBonds():
            o_index = _bond_end_with_symbol(bond, "O")
            if o_index is not None:
                fsalike_Oindex.append(o_index)
    fsalike_Oindex = list(set(fsalike_Oindex))

    # the nitrogen carries the negative charge and only single bonds
    for i in fsalike_Nindex:
        _set_formal_charge(atoms[i], atoms2d, i, -1)
        for b_idx, bond in enumerate(atoms[i].GetBonds()):
            if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                _set_bond_type(bond, atoms2d, i, b_idx, Chem.rdchem.BondType.SINGLE)

    # every single S-O bond becomes a double bond to a neutral oxygen
    for i in fsalike_Sindex:
        for b_idx, bond in enumerate(atoms[i].GetBonds()):
            if bond.GetBondType() != Chem.rdchem.BondType.SINGLE:
                continue
            oxygen_idx = _bond_end_with_symbol(bond, "O")
            if oxygen_idx is None:
                continue
            _set_bond_type(bond, atoms2d, i, b_idx, Chem.rdchem.BondType.DOUBLE)
            _set_formal_charge(atoms[oxygen_idx], atoms2d, oxygen_idx, 0)

    return {
        "FSA_N": fsalike_Nindex,
        "FSA_S": fsalike_Sindex,
        "FSA_O": fsalike_Oindex,
    }


def _set_bond_type(bond, atoms2d, atom_index, bond_index, bond_type):
    """Set a bond type on the 3D molecule and on its 2D counterpart."""
    bond.SetBondType(bond_type)
    if atoms2d is not None:
        atoms2d[atom_index].GetBonds()[bond_index].SetBondType(bond_type)


#: Ionic liquid ions recognised by :func:`_il_assign`, in detection order,
#: with the message printed when one matches.
_IL_ASSIGNERS = (
    (_assign_pf6, "PF6-like molecule detected during rdkit conversion."),
    (_assign_clo4, "ClO4-like molecule detected during rdkit conversion."),
    (_assign_fsalike, "FSA-like molecule detected during rdkit conversion."),
)


def _il_assign(mol, nc, mol2d=None):
    """
    Fix the formal charges and bond orders of the ionic liquid ions RDKit
    perceives incorrectly from the geometry alone.

    Returns
    -------
    dict or None
        Mapping of ``<ion>_<element>`` to the atom indices carrying that role,
        or None when the molecule matches none of the known ions.
    """
    for assign, message in _IL_ASSIGNERS:
        il_dict = assign(mol, nc, mol2d)
        if il_dict is not None:
            print(message)
            return il_dict
    return None
