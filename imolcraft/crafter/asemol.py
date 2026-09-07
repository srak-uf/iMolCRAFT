import math
import os
import random
from collections import defaultdict

import ase
import networkx as nx
import numpy as np
import pandas as pd
from ase import units
from ase.data import chemical_symbols
from ase.geometry import get_distances
from ase.io import read, write
from openmm.app import PDBFile

import imolcraft

# -------------------------------
# Future implementation
# Distinguish the cis and trans isomers
# -------------------------------

#: PDB ATOM record layout used by :func:`aseatoms2pdb`.
# ATOM      1    1 MOL     1       2.155   3.338  13.788  1.00  0.00           S
_PDB_ATOM_FORMAT = (
    "{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   "
    "{:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}{:2s}"
)

#: Number of atoms below which the supercell bond shortcut is not worth it.
_SUPERCELL_FAST_MIN_ATOMS = 1000


def _bond_thresholds(chemical_bonds, symbols_a, symbols_b) -> np.ndarray:
    """
    Build the (len(symbols_a), len(symbols_b)) matrix of maximum bond lengths.

    This is a vectorised replacement for repeated ``chemical_bonds.loc[si, sj]``
    lookups, which dominate the cost of the distance-based bond search.
    """
    idx_a = chemical_bonds.index.get_indexer(symbols_a)
    idx_b = chemical_bonds.columns.get_indexer(symbols_b)
    return chemical_bonds.values[np.ix_(idx_a, idx_b)]


class asemol_wrapper:
    """
    Wrapper class for ASE Atoms object to handle molecular operations.

    Parameters
    ----------
    atoms : ase.Atoms
        The atoms object to be wrapped.
    bond_def_file : str, optional
        Path to the bond definition file. If None, a default file is used.
    chemical_bonds : pd.DataFrame, optional
        DataFrame containing the chemical bond definitions. If None, it is read
        from the bond_def_file.
    """

    def __init__(self, atoms: ase.Atoms, bond_def_file=None, chemical_bonds=None):
        self.atoms = atoms
        self.filename = bond_def_file
        self.chemical_bonds = chemical_bonds
        self.bonds = None
        self.molecules = None

        if chemical_bonds is None:
            if bond_def_file is None:
                path = imolcraft.__path__[0]
                self.bond_def_file = os.path.join(path, "data", "bond_def.ini")
            self.chemical_bonds = self._read_bond_definitions(self.bond_def_file)

    @staticmethod
    def _read_bond_definitions(bond_def_file: str) -> pd.DataFrame:
        """
        Read ``elem_i elem_j max_bond_length`` triplets into a symmetric
        element-by-element DataFrame of maximum bond lengths.
        """
        chemical_bonds = pd.DataFrame(
            np.zeros((len(chemical_symbols), len(chemical_symbols))),
            index=chemical_symbols,
            columns=chemical_symbols,
        )
        data = pd.read_csv(bond_def_file, sep=r"\s+", header=None)
        for elem_i, elem_j, length in data.itertuples(index=False, name=None):
            chemical_bonds.loc[elem_i, elem_j] = length
            chemical_bonds.loc[elem_j, elem_i] = length
        return chemical_bonds

    def get_bonds(
        self, use_cache=True, unit_cell_atoms=None, repeat_factors=None
    ) -> list:
        """
        Returns a list of bonds (pairs of atom indices) based on the distance matrix and
        the chemical bond definitions. The bonds are determined by checking if the
        distance between atoms is less than or equal to the defined bond length.

        Parameters
        ----------
        use_cache : bool, optional
            Whether to use cached bonds if available. Default is True.
        unit_cell_atoms : ase.Atoms, optional
            If provided along with repeat_factors, will use optimized supercell
            calculation
        repeat_factors : tuple or list, optional
            The repeat factors (nx, ny, nz) used to create the supercell

        Returns
        -------
        bonds : list
            List of tuples, where each tuple contains the indices of the two atoms
            that are bonded.
        """
        if use_cache and self.bonds is not None:
            return self.bonds

        # Use optimized calculation for supercells if unit cell info is provided
        if unit_cell_atoms is not None and repeat_factors is not None:
            self.bonds = self._calculate_supercell_bonds_fast(
                unit_cell_atoms, repeat_factors
            )
        else:
            self.bonds = self._calculate_bonds()
        return self.bonds

    def _calculate_bonds(self) -> list:
        """
        Calculate bonds using the standard O(n^2) distance-based method.
        """
        atoms = self.atoms
        symbols = atoms.get_chemical_symbols()
        geo_matrx = get_distances(atoms.positions, cell=atoms.cell, pbc=True)[1]
        thresholds = _bond_thresholds(self.chemical_bonds, symbols, symbols)

        i_idx, j_idx = np.triu_indices(len(atoms), k=1)
        bonded = geo_matrx[i_idx, j_idx] <= thresholds[i_idx, j_idx]
        return [(int(i), int(j)) for i, j in zip(i_idx[bonded], j_idx[bonded])]

    def _calculate_supercell_bonds_fast(self, unit_cell_atoms, repeat_factors):
        """
        Fast bond calculation for supercells created by repeating a unit cell.

        This method provides significant performance improvement by calculating bonds
        for the smaller unit cell and replicating the pattern.

        Parameters
        ----------
        unit_cell_atoms : ase.Atoms
            The original unit cell
        repeat_factors : tuple or list
            The repeat factors (nx, ny, nz)

        Returns
        -------
        bonds : list
            List of bond tuples
        """
        n_unit = len(unit_cell_atoms)
        total_cells = int(np.prod(repeat_factors))

        # Only use optimization for large systems where benefit is clear
        if len(self.atoms) < _SUPERCELL_FAST_MIN_ATOMS:
            return self._calculate_bonds()

        # Calculate bonds for unit cell (much smaller, so fast)
        unit_wrapper = asemol_wrapper(
            unit_cell_atoms, chemical_bonds=self.chemical_bonds
        )
        unit_bonds = unit_wrapper._calculate_bonds()

        # Replicate intra-cell bonds for each copy of the unit cell
        all_bonds = [
            (bond[0] + cell_idx * n_unit, bond[1] + cell_idx * n_unit)
            for cell_idx in range(total_cells)
            for bond in unit_bonds
        ]

        # For inter-cell bonds, use the most efficient approach available
        # For very large systems, we can accept some approximation for speed
        all_bonds.extend(
            self._find_intercell_bonds_simple(unit_cell_atoms, repeat_factors)
        )
        return all_bonds

    def _find_boundary_atoms(self, unit_cell_atoms) -> list:
        """
        Indices of the unit-cell atoms lying within one maximum bond length of
        any cell face; only these can bond across a cell boundary.
        """
        max_bond_length = self.chemical_bonds.values.max()
        unit_cell = unit_cell_atoms.cell

        boundary_atoms = []
        for i, pos in enumerate(unit_cell_atoms.positions):
            # Check if atom is near any cell face
            for dim in range(3):
                cell_dim = unit_cell[dim, dim]
                if pos[dim] < max_bond_length or pos[dim] > cell_dim - max_bond_length:
                    boundary_atoms.append(i)
                    break
        return boundary_atoms

    def _find_intercell_bonds_simple(self, unit_cell_atoms, repeat_factors):
        """
        Simple but efficient inter-cell bond calculation.

        For very large systems, we prioritize speed over perfect accuracy.
        """
        n_x, n_y, n_z = repeat_factors
        n_unit = len(unit_cell_atoms)

        # For large systems, use a very targeted approach
        # Only check bonds between atoms that are likely to be at cell boundaries
        boundary_atoms = self._find_boundary_atoms(unit_cell_atoms)
        if not boundary_atoms:
            return []

        positions = self.atoms.positions
        symbols = self.atoms.get_chemical_symbols()
        inter_bonds = []

        # For each boundary atom in each cell, check limited neighbors
        for cell_idx in range(n_x * n_y * n_z):
            ix = cell_idx // (n_y * n_z)
            iy = (cell_idx % (n_y * n_z)) // n_z
            iz = cell_idx % n_z
            cell_offset = cell_idx * n_unit

            # Only check +x, +y, +z neighbors to avoid double counting
            for dx, dy, dz in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                neighbor_cell_idx = (
                    ((ix + dx) % n_x) * n_y * n_z
                    + ((iy + dy) % n_y) * n_z
                    + ((iz + dz) % n_z)
                )
                neighbor_offset = neighbor_cell_idx * n_unit

                atoms_i = np.array([cell_offset + b for b in boundary_atoms])
                atoms_j = np.arange(neighbor_offset, neighbor_offset + n_unit)

                # Check bonds between boundary atoms and all atoms in neighbor cell
                dists = get_distances(
                    positions[atoms_i],
                    positions[atoms_j],
                    cell=self.atoms.cell,
                    pbc=True,
                )[1]
                thresholds = _bond_thresholds(
                    self.chemical_bonds,
                    [symbols[a] for a in atoms_i],
                    [symbols[a] for a in atoms_j],
                )
                # Avoid double counting
                bonded = (dists <= thresholds) & (
                    atoms_i[:, None] < atoms_j[None, :]
                )
                for a, b in zip(*np.nonzero(bonded)):
                    inter_bonds.append((int(atoms_i[a]), int(atoms_j[b])))

        return inter_bonds

    def get_molecules(self):
        """
        Returns a list of molecules (list of atom indices) from the bonds.
        The molecules are identified by connected components in the bond graph.

        Returns
        -------
        compounds : list
            List of molecules, where each molecule is represented as a list of
            atom indices.
        """
        atoms = self.atoms

        if self.bonds is None:
            self.bonds = self.get_bonds()

        # Construct a graph from the bonds
        graph = defaultdict(list)
        for atom1, atom2 in self.bonds:
            graph[atom1].append(atom2)
            graph[atom2].append(atom1)

        # Find all connected components (molecules) in the graph.
        # Iterative pre-order DFS: neighbours are pushed in reverse so that
        # they are popped in insertion order.
        compounds = []
        visited = set()
        for atom in graph:
            if atom in visited:
                continue
            compound = []
            stack = [atom]
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                compound.append(current)
                stack.extend(reversed(graph[current]))
            compounds.append(compound)

        # Add single atoms (not part of any bond) as separate molecules
        compounds.extend([[i] for i in range(len(atoms)) if i not in visited])

        return compounds

    def get_ase_molecules(self, out_nX=False):
        """
        Returns a list of molecules (ase.Atoms) and a list of molecule indices.
        The molecules are unwrapped and the indices are used to identify the molecules.

        Parameters
        ----------
        out_nX : bool
            If True, returns a list of networkx graphs of the molecules.

        Returns
        -------
        asemols : list
            List of ase.Atoms objects of the molecules.
        molecule_list : list
            List of molecule indices.
        asenX : list
            If out_nX is True, returns a list of networkx graphs of the molecules.
            List of networkx graphs of the molecules.
        """
        if self.molecules is None:
            self.molecules = self.get_molecules()
        self.atoms = self.unwrap_molecules()

        self.molecules = [sorted(mol) for mol in self.molecules]
        asemols = [self.atoms[mol] for mol in self.molecules]

        # Group the molecules by isomorphism; molecule_list[i] holds the
        # indices of every molecule identical to its first entry.
        molecule_list = []
        grouped = set()
        for i_mol in range(len(asemols)):
            if i_mol in grouped:
                continue
            group = [i_mol]
            grouped.add(i_mol)
            for j_mol in range(i_mol + 1, len(asemols)):
                if is_same_molecule(
                    asemols[i_mol], asemols[j_mol], self.chemical_bonds
                ):
                    group.append(j_mol)
                    grouped.add(j_mol)
            molecule_list.append(group)

        # Reorder every molecule to match the atom order of its group
        # representative, so that identical molecules share an atom ordering.
        ref_mols = [asemols[mol[0]] for mol in molecule_list]
        asenX = [None for _ in asemols]
        res_number = 0
        for i_mol, ref_mol in enumerate(ref_mols):
            ref_mol.arrays["residuenames"] = np.array(
                [f"M{i_mol}" for _ in range(len(ref_mol))]
            )
            for i, mol_i in enumerate(asemols):
                try:
                    mol_i_rorder = reorder_atoms(ref_mol, mol_i, self.chemical_bonds)
                    asemols[i] = mol_i_rorder
                    asenX[i] = ase_atoms_to_nx(asemols[i], self.chemical_bonds)
                    asemols[i].arrays["residuenumbers"] = np.array(
                        [res_number + i + 1 for _ in range(len(mol_i_rorder))]
                    )
                except Exception:
                    pass

        if out_nX is True:
            return asemols, molecule_list, asenX
        else:
            return asemols, molecule_list

    def unwrap_molecules(self) -> ase.Atoms:
        """
        Unwraps the positions of atoms in a molecule to their original positions in the
        unit cell. This is useful for visualizing the molecule in its original
        orientation.

        Returns
        -------
        atoms_unwrap : ase.Atoms
            Unwrapped atoms object.
        """
        atoms = self.atoms
        atoms_unwrap = atoms.copy()
        if self.bonds is None:
            self.bonds = self.get_bonds()
        if self.molecules is None:
            self.molecules = self.get_molecules()

        neighbors = defaultdict(list)
        for atom1, atom2 in self.bonds:
            neighbors[atom1].append(atom2)
            neighbors[atom2].append(atom1)

        shift_mic = get_distances(atoms.positions, cell=atoms.cell, pbc=True)[0]
        for mol in self.molecules:
            # Breadth-first walk from the first atom of the molecule, placing
            # each atom next to the neighbour it was reached from.
            ref_init = [mol[0]]
            ref_done = {mol[0]}
            while len(ref_done) != len(mol):
                next_ref = []
                for ref_i in ref_init:
                    for ref_j in neighbors[ref_i]:
                        if ref_j in ref_done:
                            continue
                        next_ref.append(ref_j)
                        atoms_unwrap[ref_j].position = (
                            atoms_unwrap[ref_i].position + shift_mic[ref_i, ref_j]
                        )
                ref_init = next_ref
                ref_done.update(next_ref)
        return atoms_unwrap


def ase_atoms_to_nx(atoms: ase.Atoms, chemical_bonds=None):
    """
    Convert ASE Atoms object to a NetworkX graph representation.

    Parameters
    ----------
    atoms : ase.Atoms
        The atoms object to be converted.
    chemical_bonds : pd.DataFrame, optional
        DataFrame containing the chemical bond definitions. If None, it is read from
        the bond_def_file.

    Returns
    -------
    G : networkx.Graph
        The NetworkX graph representation of the atoms object.
    """
    G = nx.Graph()
    asemol_wrap = asemol_wrapper(atoms, chemical_bonds=chemical_bonds)
    for i, at in enumerate(atoms):
        G.add_node(i, element=at.symbol, xyz=at.position)
    G.add_edges_from(asemol_wrap.get_bonds())
    return G


def _element_graph_matcher(mol1: ase.Atoms, mol2: ase.Atoms, chemical_bonds):
    """Build a ``GraphMatcher`` between two molecules, matching elements."""
    G1 = ase_atoms_to_nx(mol1, chemical_bonds)
    G2 = ase_atoms_to_nx(mol2, chemical_bonds)
    return nx.isomorphism.GraphMatcher(
        G1, G2, node_match=lambda x, y: x["element"] == y["element"]
    )


def is_same_molecule(mol1: ase.Atoms, mol2: ase.Atoms, chemical_bonds):
    """
    Check if two molecules (ASE Atoms objects) are the same based on their
    chemical bonds.

    Parameters
    ----------
    mol1 : ase.Atoms
        The first molecule to compare.
    mol2 : ase.Atoms
        The second molecule to compare.
    chemical_bonds : pd.DataFrame
        DataFrame containing the chemical bond definitions.

    Returns
    -------
    isomorphic : bool
        True if the two molecules are isomorphic (same structure), False otherwise.
    """
    return _element_graph_matcher(mol1, mol2, chemical_bonds).is_isomorphic()


def reorder_atoms(atoms1, atoms2, chemical_bonds):
    """
    Reorder the atoms in atoms2 to match the order of atoms in atoms1 based on their
    chemical bonds.
    This is useful for comparing two molecules with the same structure but
    different atom order.

    Parameters
    ----------
    atoms1 : ase.Atoms
        The first molecule (reference) to compare.
    atoms2 : ase.Atoms
        The second molecule to reorder.
    chemical_bonds : pd.DataFrame
        DataFrame containing the chemical bond definitions.

    Returns
    -------
    reordered_atoms2 : ase.Atoms
        The reordered atoms2 object.
    """
    # Check whether the two graphs are isomorphic
    GM = _element_graph_matcher(atoms1, atoms2, chemical_bonds)
    if not GM.is_isomorphic():
        raise ValueError("molecule 1 and molecule 2 are not isomorphic.")

    # If isomorphic, get the mapping of the corresponding nodes and reorder
    # the atoms of molecule 2 to follow the atom order of molecule 1
    mapping = GM.mapping
    reordered_atoms2 = atoms2[[mapping[i] for i in range(len(atoms1))]]

    for key in ("residuenames", "atomtypes"):
        if key in atoms1.arrays:
            reordered_atoms2.arrays[key] = atoms1.arrays[key]
    return reordered_atoms2


def kabsch_algorithm(P, Q):
    """
    Calculation of optimal rotation matrix and translation vector by Kabsch algorithm

    Parameters
    ----------
    P : numpy.ndarray
        Coordinates of the first molecule (N x 3)
    Q : numpy.ndarray
        Coordinates of the second molecule (N x 3)

    Returns
    -------
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
    if np.sign(np.linalg.det(V @ U.T)) < 0:
        V[:, -1] *= -1

    # Rotational matrix
    R = V @ U.T

    # Translation vector
    t = centroid_Q - (R @ centroid_P)

    return R, t


def cast_molecules(G1, G2):
    """
    Cast molecules using Kabsch algorithm

    Parameters
    ----------
    G1 : networkx.Graph
        First molecule graph
    G2 : networkx.Graph
        Second molecule graph

    Returns
    -------
    rmsd: float
        Lowest RMSD among mappings
    positions : np.ndarray
        Aligned positions of the first molecule by the lowest-rmsd conversion
    """
    GM = nx.isomorphism.GraphMatcher(
        G1, G2, node_match=lambda n1, n2: n1["element"] == n2["element"]
    )
    if not GM.is_isomorphic():
        raise ValueError("Graph isomorphism failed")

    # Try every isomorphism mapping and keep the best superposition
    best_rmsd = float("inf")
    best_portions = None
    for map in GM.subgraph_isomorphisms_iter():
        P = np.array([G1.nodes[i]["xyz"] for i in map.keys()])
        Q = np.array([G2.nodes[j]["xyz"] for j in map.values()])
        R, t = kabsch_algorithm(P, Q)
        P_aligned = (R @ P.T).T + t

        rmsd = np.sqrt(np.mean(np.sum((P_aligned - Q) ** 2, axis=1)))
        if rmsd < best_rmsd:
            best_rmsd = rmsd
            best_portions = P_aligned
    return best_rmsd, best_portions


def aseatoms2pdb(filename, atoms):
    """
    Write ASE Atoms object to PDB file.

    Parameters
    ----------
    filename : str
        Name of the output PDB file.
    atoms : ase.Atoms
        The atoms object to be written to the PDB file.
    """
    if "residuenumbers" not in atoms.arrays.keys():
        atoms.arrays["residuenumbers"] = np.array([1 for _ in range(len(atoms))])
        atoms.arrays["residuenames"] = np.array(["M1" for _ in range(len(atoms))])

    resnumbers = atoms.arrays["residuenumbers"]
    resnames = atoms.arrays["residuenames"]
    symbols = atoms.get_chemical_symbols()

    with open(f"{filename}", "w") as f:
        Lx, Ly, Lz, alpha, beta, gamma = atoms.cell.cellpar()
        f.write(
            "CRYST1{:9.3f}{:9.3f}{:9.3f}{:7.2f}{:7.2f}{:7.2f} P 1  \n".format(
                Lx, Ly, Lz, alpha, beta, gamma
            )
        )
        f.write("MODEL     1\n")

        atomname = 0
        for i, atom in enumerate(atoms):
            # the per-atom counter restarts on every new residue
            if i > 0 and resnumbers[i - 1] != resnumbers[i]:
                atomname = 0

            if np.count_nonzero(resnumbers == resnumbers[i]) == 1:
                # single atom residue case: the element alone is the atom name
                atom_label = symbols[i]
                atomname = 0
            else:
                atom_label = symbols[i] + str(atomname)
                atomname += 1

            f.write(
                _PDB_ATOM_FORMAT.format(
                    "ATOM",
                    i + 1,
                    atom_label,
                    " ",
                    f"{resnames[i]}",
                    " ",
                    int(f"{resnumbers[i]}"),
                    " ",
                    atom.position[0],
                    atom.position[1],
                    atom.position[2],
                    1.0,
                    0.0,
                    " ",
                    symbols[i],
                )
                + "\n"
            )
        f.write("ENDMDL\n")


def aseatoms2pdb_bonded(filename, atoms):
    aw = asemol_wrapper(atoms)
    molatoms, _, _ = aw.get_ase_molecules(out_nX=True)
    unit_cell_atoms = merge_asemols(molatoms)
    aseatoms2pdb(filename, unit_cell_atoms)

    aw = asemol_wrapper(read(filename))
    _, _, _ = aw.get_ase_molecules(out_nX=True)
    bonds = aw.get_bonds()

    pdb_omm = PDBFile(filename)
    atomlist_openmm = [a for a in pdb_omm.topology.atoms()]
    for b in bonds:
        pdb_omm.topology.addBond(atomlist_openmm[b[0]], atomlist_openmm[b[1]])
    PDBFile.writeFile(pdb_omm.topology, pdb_omm.positions, open(filename, "w"))


def merge_asemols(asemols):
    """
    Merge multiple asemol objects into a single one.

    Parameters
    ----------
    asemols : list
        List of asemol objects to be merged.

    Returns
    -------
    merged : ase.Atoms
        Merged ase.Atoms object.
    """
    merged = asemols[0].copy()
    for mol in asemols[1:]:
        merged.extend(mol)
    return merged


def expand_cell(atoms, length=30):
    """
    Expand the cell of the atoms object to fit the specified length.

    Parameters
    ----------
    atoms : ase.Atoms
        The atoms object to be expanded.
    length : float
        The desired length of the cell in Angstroms.

    Returns
    -------
    atoms : ase.Atoms
        The expanded atoms object.
    """
    La, Lb, Lc = atoms.cell.cellpar()[:3]
    duplication = (
        math.ceil(length / La),
        math.ceil(length / Lb),
        math.ceil(length / Lc),
    )
    print(*duplication)
    return atoms.repeat(duplication)


def _total_mass(pdbfiles, nmols) -> float:
    """Total mass (amu) of ``nmols[i]`` copies of each molecule in ``pdbfiles``."""
    return sum(
        read(pdb).get_masses().sum() * nmols[i] for i, pdb in enumerate(pdbfiles)
    )


def _cubic_cell_length(mass: float, density: float) -> float:
    """Edge length of the cubic cell holding ``mass`` at the given density."""
    return (mass / (density / (units.m**3 / units.kg))) ** (1 / 3)


def _nmols_from_density(pdbfiles, nmols, cell, density) -> list:
    """
    Scale the molecule ratio up to the largest multiple that fits ``cell``
    at the requested density.
    """
    gcd_value = math.gcd(*nmols)
    nmols = [n // gcd_value for n in nmols]
    mass = _total_mass(pdbfiles, nmols)
    Nset = int(
        density / (mass / (cell[0] * cell[1] * cell[2]) * units.m**3 / units.kg)
    )
    return [n * Nset for n in nmols]


def pdb2packmol(
    pdbfiles,
    fixed_property,
    priority_property,
    nmols,
    cell=None,
    density=None,
    outfile="packmol_tmp.xyz",
    cleanup=True,
):
    """
    Liquid packing using packmol.

    Parameters
    ----------
    pdbfiles : list
        List of PDB files to be packed.
    fixed_property: str
        Property to be fixed during packing: nmols, cell, or density.
    priority_property: str
        Property to be prioritized during packing: cell, or density.
    nmols : list
        List of number / ratio of molecules for PDB files.
        If nmols is not specified as a fixed_property, it is treated as a ratio.
    cell : list, optional
        List of cell dimensions [a, b, c]. If None, a default cell size is used.
    density : float, optional
        Desired density of the system. If None, the density is calculated based on
        the number of molecules and cell size.
    outfile : str, optional
        Output file name for the packed system. Default is "packmol_tmp.xyz".

    Returns
    -------
    bonds_top : list
        List of bonds in the packed system.
    atomslist_mols : list
        List of ASE Atoms objects for each molecule in the packed system.
    molecule_list : list
        List of molecule indices in the packed system.
    """
    valid_fixed_property = ["nmols", "cell", "density"]
    valid_priority_property = ["cell", "density"]

    if fixed_property not in valid_fixed_property:
        raise ValueError(
            f"Invalid fixed_property: {fixed_property}. "
            f"Must be one of {valid_fixed_property}."
        )
    if priority_property not in valid_priority_property:
        raise ValueError(
            f"Invalid priority_property: {priority_property}. "
            f"Must be one of {valid_priority_property}."
        )

    if fixed_property == "nmols" and priority_property == "cell":
        # both the counts and the cell are given: nothing to derive
        pass
    elif fixed_property == "nmols" and priority_property == "density":
        # derive the cell from the fixed counts
        length = _cubic_cell_length(_total_mass(pdbfiles, nmols), density)
        cell = [length, length, length]
    elif fixed_property == "cell" and priority_property == "density":
        # derive the counts from the fixed cell
        nmols = _nmols_from_density(pdbfiles, nmols, cell, density)
    elif fixed_property == "density" and priority_property == "cell":
        # derive the counts from the current cell, then rescale the cell so
        # that the density is exactly reproduced
        nmols = _nmols_from_density(pdbfiles, nmols, cell, density)
        length = _cubic_cell_length(_total_mass(pdbfiles, nmols), density)
        cell[0] = cell[1] = cell[2] = length
        print(cell[0])

    with open("pack_tmp.inp", mode="w") as f:
        f.write("seed  " + str(random.randint(1, 10000)) + "\n")
        f.write("tolerance 2 \n")
        f.write("filetype pdb \n")
        f.write("output  packmol_tmp.pdb  \n")
        f.write(f"pbc {cell[0]} {cell[1]} {cell[2]} \n")
        ntot_atoms = 0
        bonds_top = []
        atomslist_mols = []
        molecule_list = []
        for i, pdbfile in enumerate(pdbfiles):
            atoms_pdb = read(pdbfile)
            n_atoms = len(atoms_pdb)
            # make the atom types unique and give every species its own residue
            atoms_pdb.arrays["atomtypes"] = [
                atomtype + str(j + 1)
                for j, atomtype in enumerate(atoms_pdb.arrays["atomtypes"])
            ]
            atoms_pdb.arrays["residuenames"] = [
                "M" + str(i + 1) for _ in range(len(atoms_pdb.arrays["residuenames"]))
            ]
            write(f"atoms_{i}.pdb", atoms_pdb)
            atomslist_mols.append(atoms_pdb)
            molecule_list.append([i])

            # replicate the intramolecular bonds for every copy that packmol
            # will place, shifting the atom indices of each copy
            mol_bonds = asemol_wrapper(read(pdbfile)).get_bonds()
            for _ in range(nmols[i]):
                bonds_top.extend(
                    (b[0] + ntot_atoms, b[1] + ntot_atoms) for b in mol_bonds
                )
                ntot_atoms += n_atoms

            f.write(f"structure  atoms_{i}.pdb \n")
            f.write(f"  number  {nmols[i]} \n")
            f.write("end structure \n")

    _ = os.system("packmol < " + "pack_tmp.inp")
    atoms_packtmp = read("packmol_tmp.pdb")
    atoms_packtmp.cell = cell
    atoms_packtmp.pbc = True
    write(f"{outfile}", atoms_packtmp)

    if cleanup is True:
        os.remove("pack_tmp.inp")
        os.remove("packmol_tmp.pdb")
        for i in range(len(pdbfiles)):
            os.remove(f"atoms_{i}.pdb")
    return bonds_top, atomslist_mols, molecule_list
