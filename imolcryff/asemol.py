from ase.io import read, write
import ase
from ase import units
from ase.geometry import get_distances
from ase.data import chemical_symbols

import networkx as nx

import pandas as pd
import numpy as np
from collections import defaultdict

import os
import random

##### Future implementation #####
# Distinguish the cis and trans isomers
#################################

class asemol_wrapper:
    def __init__(self, atoms: ase.Atoms, bond_def_file=None, chemical_bonds=None):
        self.atoms = atoms
        self.filename = bond_def_file
        self.chemical_bonds = chemical_bonds
        self.bonds = None
        self.molecules = None

        if chemical_bonds is  None:
            if bond_def_file is None:
                self.bond_def_file = os.path.join(os.path.dirname(__file__), "bond_def.ini")
            
            self.chemical_bonds = pd.DataFrame(np.zeros((len(chemical_symbols), len(chemical_symbols))), index=chemical_symbols, columns=chemical_symbols)
            data = pd.read_csv(self.bond_def_file , sep='\s+',header=None)
            for i,elem_i in enumerate(data.iloc[:,0].values):
                elem_j = data.iloc[:,1].values[i]
                self.chemical_bonds.loc[elem_i, elem_j] = data.iloc[:,2].values[i]
                self.chemical_bonds.loc[elem_j, elem_i] = data.iloc[:,2].values[i]

    def get_bonds(self) -> list:
        """ase.Atomsから結合リストを生成"""
        atoms = self.atoms
        geo_matrx = get_distances(atoms.positions,cell=atoms.cell,pbc=True)[1]
        bonds = []
        for i in range(len(atoms)):
            for j in range(i+1, len(atoms)):
                if geo_matrx[i,j] <= self.chemical_bonds.loc[atoms[i].symbol, atoms[j].symbol]:
                    bonds.append((i, j))
        self.bonds = bonds
        return bonds
    
    def get_molecules(self):
        """結合リストから化合物ごとにatom indexのリストを作りそのリストを生成する関数"""
        atoms = self.atoms
        compounds = []
        graph = defaultdict(list)

        if self.bonds is None:
            self.bonds = self.get_bonds()
        
        # グラフの構築
        for bond in self.bonds:
            atom1, atom2 = bond
            graph[atom1].append(atom2)
            graph[atom2].append(atom1)
        
        visited = set()
        def dfs(atom, compound):
            # 深さ優先探索で化合物を構築
            if atom not in visited:
                visited.add(atom)
                compound.append(atom)
                for neighbor in graph[atom]:
                    dfs(neighbor, compound)
            return compound
        
        # グラフの各ノードに対してDFSを実行し、化合物を構築
        for atom in graph:
            if atom not in visited:
                compound = dfs(atom, [])
                compounds.append(compound)
        
        # 生成された化合物の原子リストを作成
        moleculed_atoms = []
        for molecule in compounds:
            moleculed_atoms.extend(molecule)

        # 結合を作っていない原子を単体として追加
        singleatoms = [[i] for i in range(len(atoms)) if i not in moleculed_atoms]
        compounds.extend(singleatoms)
        
        return compounds
    
    def get_ase_molecules(self, out_nX=False):
        asemols = []
        asenX = []

        if self.molecules is None:
            self.molecules = self.get_molecules()
        self.atoms = self.unwrap_molecules()

        for i_mol in range(len(self.molecules)):
            asemols.append(self.atoms[self.molecules[i_mol]])
        
        molecule_list = []
        for i_mol in range(len(asemols)):
            tmp2 = [t for t in molecule_list for t in t]
            if i_mol not in tmp2:
                molecule_list.append([i_mol])
                for j_mol in range(i_mol+1, len(asemols)):
                    if is_same_molecule(asemols[i_mol], asemols[j_mol], self.chemical_bonds):
                        molecule_list[-1].append(j_mol)
        
        ref_mols = [asemols[mol[0]] for mol in molecule_list]
        asenX = [None for _ in asemols]
        res_number = 0
        for i_mol, ref_mol in enumerate(ref_mols):
            ref_mol.arrays["residuenames"] = np.array([ f"M{i_mol}" for _ in range(len(ref_mol) )])
            for i, mol_i in enumerate(asemols):
                try:
                    mol_i_rorder = reorder_atoms(ref_mol, mol_i, self.chemical_bonds)
                    asemols[i] = mol_i_rorder
                    asenX[i] = ase_atoms_to_nx(asemols[i], self.chemical_bonds)
                    asemols[i].arrays["residuenumbers"] = np.array([res_number+i+1 for _ in range(len(mol_i_rorder))])
                except:
                    pass

        if out_nX == True:
            return asemols, molecule_list, asenX
        else:
            return asemols, molecule_list

    def unwrap_molecules(self) -> ase.Atoms:
        """分子構造を保って原子座標をunwrapする関数"""
        atoms = self.atoms
        atoms_unwrap = atoms.copy()
        if self.bonds is None:
            self.bonds = self.get_bonds()
        if self.molecules is None:
            self.molecules = self.get_molecules()

        shift_mic = get_distances(atoms.positions,cell=atoms.cell,pbc=True)[0]
        for i_mol in range(len(self.molecules)):
            mol = self.molecules[i_mol]
            ref_init = [mol[0]]
            next_ref = []
            ref_done = [mol[0]]
            while len(ref_done) != len(mol):
                next_ref = []
                for ref_i in ref_init:
                    for bond in self.bonds:
                        if ref_i in bond:
                            if ref_i == bond[0] and bond[1] not in ref_done:
                                next_ref.append(bond[1])
                                atoms_unwrap[next_ref[-1]].position = atoms_unwrap[ref_i].position + shift_mic[ref_i, next_ref[-1]]
                            elif ref_i == bond[1] and bond[0] not in ref_done:
                                next_ref.append(bond[0])
                                atoms_unwrap[next_ref[-1]].position = atoms_unwrap[ref_i].position + shift_mic[ref_i, next_ref[-1]]
                ref_init = next_ref
                ref_done.extend(next_ref)
                ref_done = list(set(ref_done))
        return atoms_unwrap

def ase_atoms_to_nx(atoms: ase.Atoms, chemical_bonds):
    G = nx.Graph()
    asemol_wrap = asemol_wrapper(atoms, chemical_bonds=chemical_bonds)
    bonds = asemol_wrap.get_bonds()
    for i, at in enumerate(atoms):
        G.add_node(i, element=at.symbol, xyz=at.position)
        # G.add_node(i, element=at.symbol)
    for bond in bonds:
        G.add_edge(bond[0], bond[1])
    return G

def is_same_molecule(mol1: ase.Atoms, mol2: ase.Atoms, chemical_bonds):
    """分子1(ase.atoms)と分子2(ase.atoms)が同じ分子かどうかを判定する関数"""
    G1 = ase_atoms_to_nx(mol1, chemical_bonds)
    G2 = ase_atoms_to_nx(mol2, chemical_bonds)
    return nx.isomorphism.GraphMatcher(G1, G2, node_match=lambda x, y: x['element'] == y['element']).is_isomorphic()

def reorder_atoms(atoms1, atoms2, chemical_bonds):
    """分子1の原子の並びに基づいて分子2の原子を並べ替える関数, atoms2 を atoms1 に合わせる"""
    # グラフが同型かどうかをチェック
    G1 = ase_atoms_to_nx(atoms1, chemical_bonds)
    G2 = ase_atoms_to_nx(atoms2, chemical_bonds)
    GM = nx.isomorphism.GraphMatcher(G1, G2, node_match=lambda x, y: x['element'] == y['element'])
    if not GM.is_isomorphic():
        raise ValueError("分子1と分子2は同型ではありません。")

    # 同型の場合、対応するノードのマッピングを取得
    mapping = GM.mapping
    
    # 分子2の原子を分子1の原子の順序に従って並べ替え
    new_order = [mapping[i] for i in range(len(atoms1))]
    reordered_atoms2 = atoms2[new_order]
    if 'residuenames' in atoms1.arrays:
        reordered_atoms2.arrays['residuenames'] = atoms1.arrays['residuenames']
    if 'atomtypes' in atoms1.arrays:
        reordered_atoms2.arrays['atomtypes'] = atoms1.arrays['atomtypes']
    return reordered_atoms2


def kabsch_algorithm(P, Q):
    """
    Calculation of optimal rotation matrix and translation vector by Kabsch algorithm

    Parameters:
    P : numpy.ndarray
        Coordinates of the first molecule (N x 3)
    Q : numpy.ndarray
        Coordinates of the second molecule (N x 3)
    Returns:
    R : numpy.ndarray
        Rotation matrix (3 x 3)
    t : numpy.ndarray
        Translation vector (3 x 1)
    """
    # Centroid
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)

    # Centering
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q

    # Covariance matrix
    H = P_centered.T @ Q_centered

    # SVD
    U, S, Vt = np.linalg.svd(H)
    V = Vt.T

    # Reflection 
    d = np.sign(np.linalg.det(V @ U.T))
    if d < 0:
        V[:, -1] *= -1

    # Rotational matrix
    R = V @ U.T

    # Translation vector
    t = centroid_Q - (R @ centroid_P)
    
    return R, t

def cast_molecules(G1, G2):
    """
    Cast molecules using Kabsch algorithm

    Parameters:
    G1 : networkx.Graph
        First molecule graph
    G2 : networkx.Graph
        Second molecule graph
    Returns:
    rmsd: float
        Lowest RMSD among mappings
    positions : np.ndarray
        Aligned positions of the first molecule by the lowest-rmsd conversion
    """
    GM = nx.isomorphism.GraphMatcher(G1, G2, node_match=lambda n1, n2: n1['element'] == n2['element'])
    best_portions = None
    if not GM.is_isomorphic():
        assert False, "Graph isomorphism failed"
    
    # Get the mappings
    mapping = list(GM.subgraph_isomorphisms_iter())

    best_rmsd = float('inf')
    for map in mapping:
        P = np.array([G1.nodes[i]['xyz'] for i in map.keys()])
        Q = np.array([G2.nodes[j]['xyz'] for j in map.values()])
        R, t = kabsch_algorithm(P, Q)
        P_aligned = (R @ P.T).T + t

        rmsd = np.sqrt(np.mean(np.sum((P_aligned - Q) ** 2, axis=1)))
        if rmsd < best_rmsd:
            best_rmsd = rmsd
            best_portions = P_aligned
    return best_rmsd, best_portions


def aseatoms2pdb(filename, atoms):
    # ATOM      1    1 MOL     1       2.155   3.338  13.788  1.00  0.00           S  
    # pdb_atom_format = '{:6s}{:5d} {:^4s}{:1s}{:3s}{:1s} {:4d}{:1s}{:3s}{:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}{:10s}{:>2s}'
    pdb_atom_format = '{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}{:2s}'
    atomname = 0
    with open(f"{filename}", "w") as f:
        Lx = atoms.cell.cellpar()[0]
        Ly = atoms.cell.cellpar()[1]
        Lz = atoms.cell.cellpar()[2]
        alpha = atoms.cell.cellpar()[3]
        beta  = atoms.cell.cellpar()[4]
        gamma = atoms.cell.cellpar()[5]

        cryst_line = "CRYST1{:9.3f}{:9.3f}{:9.3f}{:7.2f}{:7.2f}{:7.2f} P 1  \n".format(Lx, Ly, Lz, alpha, beta, gamma)
        f.write(cryst_line)
        f.write("MODEL     1\n")
        for i, atom in enumerate(atoms):
            if i >0 and atoms.arrays["residuenumbers"][i-1] != atoms.arrays["residuenumbers"][i]:
                atomname = 0
            # single atom residue case
            if np.count_nonzero(atoms.arrays["residuenumbers"] == atoms.arrays["residuenumbers"][i]) == 1:
                atomname = atoms.get_chemical_symbols()[i]
                atomline = pdb_atom_format.format('ATOM', i+1, str(atomname), ' ', f'{atoms.arrays["residuenames"][i]}', ' ',int(f'{atoms.arrays["residuenumbers"][i]}'), ' ',  atom.position[0], atom.position[1], atom.position[2], 1.0, 0.0, ' ',atoms.get_chemical_symbols()[i])
                atomname = 0
            else:
                symbol = atoms.get_chemical_symbols()[i]
                atomline = pdb_atom_format.format('ATOM', i+1, symbol+str(atomname), ' ', f'{atoms.arrays["residuenames"][i]}', ' ',int(f'{atoms.arrays["residuenumbers"][i]}'), ' ',  atom.position[0], atom.position[1], atom.position[2], 1.0, 0.0, ' ',atoms.get_chemical_symbols()[i])
                atomname += 1
            f.write(atomline+"\n")
        f.write("ENDMDL\n")


def merge_asemols(asemols):
    """分子（ase.Atoms class）のリストを受け取り、それらを結合したase.Atoms classを返す関数"""
    merge_asemols = asemols[0].copy()
    for i, mol in enumerate(asemols):
        if i > 0:
            merge_asemols.extend(mol)
    return merge_asemols

import math
def expand_cell(atoms, length=30):
    La = atoms.cell.cellpar()[0]
    Lb = atoms.cell.cellpar()[1]    
    Lc = atoms.cell.cellpar()[2]
    a_dup = math.ceil(length/La)
    b_dup = math.ceil(length/Lb)
    c_dup = math.ceil(length/Lc)
    print(a_dup, b_dup, c_dup)
    atoms = atoms.repeat((a_dup, b_dup, c_dup))
    return atoms


def pdb2packmol(pdbfiles, num_mols=None, cell=None, desired_density=None, outfile="packmol_tmp.xyz"):
    """
    pdbfiles: list of pdb files
    num_mols: list of number (or ratio) of molecules of each pdb file
    cell: cell size
    desired_density: desired density (kg/m^3)
    outfile: output file

    return: bonds_top, atomslist_mols, molecule_list
            bonds_top: list of bonds
            atomslist_mols: list of ase atoms objects
            molecule_list: list of molecule indices
    """
    if num_mols == None:
        num_mols = [1 for _ in  pdbfiles]
    if num_mols != None and cell != None and desired_density != None:
        M = 0.0
        for i, pdb in enumerate(pdbfiles):
            atoms = read(pdb)
            M += atoms.get_masses().sum() * num_mols[i]
        Nset = int(desired_density / (M / (cell[0]*cell[1]*cell[2])  * units.m**3 / units.kg ) )
        num_mols = [ n * Nset  for n in num_mols]
        print(f"num_mols: {num_mols}")
        
    if cell == None:
        cell = [1000,1000,1000]
        if desired_density != None:
            M = 0.0
            for i, pdb in enumerate(pdbfiles):
                atoms = read(pdb)
                M += atoms.get_masses().sum() * num_mols[i]
            rho = M / (cell[0]*cell[1]*cell[2])
            rho *= units.m**3 / units.kg
            scale = (rho / desired_density)**(1/3)
            cell = [cell[0]*scale, cell[1]*scale, cell[2]*scale]

    with open("pack_tmp.inp",mode='w') as f:
        f.write("seed  "+str(random.randint(1, 10000))+"\n")
        f.write("tolerance 2 \n")
        f.write("filetype pdb \n")
        f.write("output  packmol_tmp.pdb  \n")
        f.write(f"pbc {cell[0]} {cell[1]} {cell[2]} \n")
        ntot_atoms = 0
        bonds_top = []
        atomslist_mols = []
        molecule_list = []
        for i in range(len(pdbfiles)):
            atoms_pdb = read(pdbfiles[i])
            n_atoms = len(atoms_pdb)
            atoms_pdb.arrays["atomtypes"] = [atoms_pdb.arrays["atomtypes"][i]+str(i+1) for i in range(len(atoms_pdb.arrays["atomtypes"]))]
            atoms_pdb.arrays["residuenames"] = ["M"+str(i+1) for _ in range(len(atoms_pdb.arrays["residuenames"]))]
            write(f"atoms_{i}.pdb",atoms_pdb)
            atomslist_mols.append(atoms_pdb)
            molecule_list.append([i])
            for _ in range(num_mols[i]):
                bonds  = asemol_wrapper(read(pdbfiles[i])).get_bonds()
                bonds  = [(b[0] + ntot_atoms , b[1]+ ntot_atoms )  for b in bonds]
                ntot_atoms += n_atoms
                for b in bonds:
                    bonds_top.append(b)
            f.write(f"structure  atoms_{i}.pdb \n")
            f.write(f"  number  {num_mols[i]} \n")
            f.write("end structure \n")
    _ = os.system("packmol < "+"pack_tmp.inp")
    atoms_packtmp = read("packmol_tmp.pdb") 
    atoms_packtmp.cell = cell
    atoms_packtmp.pbc  = True
    write(f"{outfile}", atoms_packtmp)
    return bonds_top, atomslist_mols, molecule_list

    atoms_packtmp.pbc  = True
    write(f"{outfile}", atoms_packtmp)
    return bonds_top, atomslist_mols, molecule_list
