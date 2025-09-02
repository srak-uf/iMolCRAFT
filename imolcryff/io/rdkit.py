from rdkit import Chem
from rdkit.Chem import rdDepictor, rdDetermineBonds
import copy


def atoms2rdkit(atoms, nc=0, il_assign=True):
    """
    Convert a list of atoms to an RDKit molecule.

    Parameters
    ----------
    atoms : list
        A list of atoms, where each atom is represented as a dictionary with keys
        'symbol', 'x', 'y', and 'z'.

    Returns
    -------
    mol : rdkit.Chem.rdchem.Mol
        An RDKit molecule object.
    mol2d : rdkit.Chem.rdchem.Mol
        An RDKit 2D molecule object.
    """
    mol = Chem.RWMol()
    element_symbols = atoms.get_chemical_symbols()
    positions = atoms.positions

    # Create a conformer first
    conf = Chem.Conformer(len(atoms))
    for i, atom in enumerate(atoms):
        atom_idx = mol.AddAtom(Chem.Atom(element_symbols[i]))
        conf.SetAtomPosition(atom_idx, positions[i])

    # Add the conformer to the molecule
    mol.AddConformer(conf)

    natoms = len(atoms)
    if natoms > 1:
        try:
            mol_tmp = copy.deepcopy(mol)
            rdDetermineBonds.DetermineBonds(mol_tmp, charge=nc)
            mol = mol_tmp
        except Exception:
            from rdkit.Chem import Draw

            mol_tmp = copy.deepcopy(mol)
            rdDetermineBonds.DetermineConnectivity(mol_tmp, charge=nc)
            mol = copy.deepcopy(mol_tmp)
            print(
                "Warning: The bonds may be assigned incorrectly. Please check."
            )
            print("Warning: This may give wrong SMILES")
            Draw.MolToImage(mol)  # Necessary
        mol2d = copy.deepcopy(mol)
        rdDepictor.Compute2DCoords(mol2d)
    else:
        symbol = element_symbols[0]
        if nc > 0:
            mol = Chem.MolFromSmiles(f"[{symbol}+{abs(nc)}]")
        elif nc < 0:
            mol = Chem.MolFromSmiles(f"[{symbol}-{abs(nc)}]")
        else:
            mol = Chem.MolFromSmiles(f"[{symbol}]")
        mol2d = mol

    il_dict = None
    if il_assign:
        il_dict = _il_assign(mol, int(nc), mol2d)
    return mol, mol2d, il_dict


def _il_assign(mol, nc, mol2d=None):
    def assign_pf6(mol, nc, mol2d=None):
        """
        Check if the molecule is PF6
        """
        atoms = mol.GetAtoms()
        if mol2d is not None:
            atoms2d = mol2d.GetAtoms()
        pf6like_Pindex = []
        pf6like_Findex = []
        il_dict = {}

        PF6_flag = False
        P_flag = False

        if nc < 0:
            for atom in atoms:
                if atom.GetSymbol() == "P":
                    P_flag = True

            if P_flag:
                for i, atom in enumerate(atoms):
                    if atom.GetSymbol() == "F":
                        atom.SetFormalCharge(0)
                        if mol2d is not None:
                            atoms2d[i].SetFormalCharge(0)
                        pf6like_Findex.append(i)
                    elif atom.GetSymbol() == "P":
                        atom.SetFormalCharge(-1)
                        if mol2d is not None:
                            atoms2d[i].SetFormalCharge(-1)
                        pf6like_Pindex.append(i)

            if len(pf6like_Findex) == 6 and len(pf6like_Pindex) == 1:
                PF6_flag = True
            if not PF6_flag:
                return None
            else:
                il_dict["PF6_P"] = pf6like_Pindex
                il_dict["PF6_F"] = pf6like_Findex
                return il_dict

    def assign_fsalike(mol, nc, mol2d):
        """
        Check if the molecule is FSA-like
        """
        fsalike_Nindex = []
        fsalike_Sindex = []
        fsalike_Oindex = []
        il_dict = {}
        if nc < 0:
            atoms = mol.GetAtoms()
            if mol2d is not None:
                atoms2d = mol2d.GetAtoms()

            # Check FSA-like N, S, and O atoms
            for i in range(len(atoms)):
                if len(atoms[i].GetBonds()) == 2 and atoms[i].GetSymbol() == "N":
                    for bond in atoms[i].GetBonds():
                        if bond.GetEndAtom().GetSymbol() == "S":
                            s_index = bond.GetEndAtomIdx()
                            if atoms[s_index].GetTotalValence() == 6:
                                fsalike_Nindex.append(i)
                                fsalike_Sindex.append(s_index)
                        elif bond.GetBeginAtom().GetSymbol() == "S":
                            s_index = bond.GetBeginAtomIdx()
                            if atoms[s_index].GetTotalValence() == 6:
                                fsalike_Nindex.append(i)
                                fsalike_Sindex.append(s_index)

            fsalike_Nindex = list(set(fsalike_Nindex))
            fsalike_Sindex = list(set(fsalike_Sindex))

            # Sに結合しているO原子のindexをfsalike_Oindexに追加
            for i in fsalike_Sindex:
                for bond in atoms[i].GetBonds():
                    if bond.GetEndAtom().GetSymbol() == "O":
                        fsalike_Oindex.append(bond.GetEndAtomIdx())
                    elif bond.GetBeginAtom().GetSymbol() == "O":
                        fsalike_Oindex.append(bond.GetBeginAtomIdx())
            fsalike_Oindex = list(set(fsalike_Oindex))

            for i in fsalike_Nindex:
                atoms[i].SetFormalCharge(-1)
                if mol2d is not None:
                    atoms2d[i].SetFormalCharge(-1)
                for b_idx, bond in enumerate(atoms[i].GetBonds()):
                    if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                        # 3d rdkitmol
                        bond.SetBondType(Chem.rdchem.BondType.SINGLE)
                        # 2d rdkitmol
                        if mol2d is not None:
                            bond2d = atoms2d[i].GetBonds()[b_idx]
                            bond2d.SetBondType(Chem.rdchem.BondType.SINGLE)

            for i in fsalike_Sindex:
                for b_idx, bond in enumerate(atoms[i].GetBonds()):
                    bonded_element = [
                        bond.GetBeginAtom().GetSymbol(),
                        bond.GetEndAtom().GetSymbol(),
                    ]
                    if (
                        bond.GetBondType() == Chem.rdchem.BondType.SINGLE
                        and "O" in bonded_element
                    ):
                        oxygen_idx = (
                            bond.GetBeginAtomIdx()
                            if bonded_element[0] == "O"
                            else bond.GetEndAtomIdx()
                        )
                        # 3d rdkitmol
                        bond.SetBondType(Chem.rdchem.BondType.DOUBLE)
                        atoms[oxygen_idx].SetFormalCharge(0)

                        # 2d rdkitmol
                        if mol2d is not None:
                            bond2d = atoms2d[i].GetBonds()[b_idx]
                            bond2d.SetBondType(Chem.rdchem.BondType.DOUBLE)
                            atoms2d[oxygen_idx].SetFormalCharge(0)

            il_dict["FSA_N"] = fsalike_Nindex
            il_dict["FSA_S"] = fsalike_Sindex
            il_dict["FSA_O"] = fsalike_Oindex

        if len(fsalike_Nindex) == 0 or len(fsalike_Sindex) == 0:
            return None
        else:
            return il_dict

    il_dict = assign_pf6(mol, nc, mol2d)
    if il_dict is not None:
        print("PF6-like molecule detected during rdkit conversion.")
        return il_dict

    il_dict = assign_fsalike(mol, nc, mol2d)
    if il_dict is not None:
        print("FSA-like molecule detected during rdkit conversion.")
        return il_dict
