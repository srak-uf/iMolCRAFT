from rdkit import Chem
from rdkit.Chem import rdDepictor, rdDetermineBonds
from rdkit.Chem import AllChem
import copy

def atoms2rdkit(atoms, nc=0, il_assign=True):
    """
    Convert a list of atoms to an RDKit molecule.
    
    Parameters
    ----------
    atoms : list
        A list of atoms, where each atom is represented as a dictionary with keys 'symbol', 'x', 'y', and 'z'.
    
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
            rdDetermineBonds.DetermineBonds(mol_tmp,charge=nc)
            mol = mol_tmp
        except:
            from rdkit.Chem import Draw
            mol_tmp = copy.deepcopy(mol)
            rdDetermineBonds.DetermineConnectivity(mol_tmp, charge=nc)
            mol = copy.deepcopy(mol_tmp)
            print("Warning: The bonds may be assigned incorrectly. Please check carefully.")
            print("Warning: This may give wrong SMILES")
            Draw.MolToImage(mol) # Necessary
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
        il_dict = _il_assign(mol,
                            mol2d,
                            int(nc))
    return mol, mol2d, il_dict

def _il_assign(mol, mol2d, nc):
    fsalike_Nindex = []
    fsalike_Sindex = []
    fsalike_Oindex = []
    il_dict = {}
    if nc < 0:
        atoms = mol.GetAtoms()
        atoms2d = mol2d.GetAtoms()

        # Check FSA-like N, S, and O atoms
        for i in range(len(atoms)):
            if atoms[i].GetTotalValence() == 3 and atoms[i].GetSymbol() == "N":
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
            atoms2d[i].SetFormalCharge(-1)
            for b_idx, bond in enumerate(atoms[i].GetBonds()):
                if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                    ## 3d rdkitmol
                    bond.SetBondType(Chem.rdchem.BondType.SINGLE)
                    ## 2d rdkitmol
                    bond2d = atoms2d[i].GetBonds()[b_idx]
                    bond2d.SetBondType(Chem.rdchem.BondType.SINGLE)

        for i in fsalike_Sindex:
            for b_idx, bond in enumerate(atoms[i].GetBonds()):
                bonded_element = [bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol()]
                if bond.GetBondType() == Chem.rdchem.BondType.SINGLE \
                    and "O" in bonded_element:
                    oxygen_idx = bond.GetBeginAtomIdx() if bonded_element[0] == "O" else bond.GetEndAtomIdx()
                    ## 3d rdkitmol
                    bond.SetBondType(Chem.rdchem.BondType.DOUBLE)
                    atoms[oxygen_idx].SetFormalCharge(0)

                    ## 2d rdkitmol
                    bond2d = atoms2d[i].GetBonds()[b_idx]
                    bond2d.SetBondType(Chem.rdchem.BondType.DOUBLE)
                    atoms2d[oxygen_idx].SetFormalCharge(0)

        il_dict["FSA_N"] = fsalike_Nindex
        il_dict["FSA_S"] = fsalike_Sindex
        il_dict["FSA_O"] = fsalike_Oindex
    return il_dict
