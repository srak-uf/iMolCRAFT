from openmm import HarmonicBondForce, LangevinMiddleIntegrator
from openmm.unit import (
    kelvin,
    picosecond,
    picoseconds,
    kilojoules_per_mole,
    angstrom,
)
from openmm.app import NoCutoff, Simulation, PDBFile, ForceField, Modeller
from openmm.openmm import XmlSerializer
import numpy as np
from ase.io import read
from ase.calculators.gaussian import Gaussian
from ase import units
import os
import tempfile
import subprocess
import networkx as nx


class DistanceCalculator:
    """
    Class for distance calculations using quantum mechanical methods and force fields.
    This class allows for the calculation of distances and their corresponding energies
    using both quantum mechanical methods (Gaussian) and force fields.

    Parameters
    ----------
    atoms: ase.Atoms
        ASE Atoms object of the molecule.
    rdkitmol: rdkit.Chem.rdchem.Mol
        RDKit molecule object.
    label: str
        Label for the calculation.
    directory: str
        Directory to save the calculation files.
    qmparams: dict
        Parameters for the quantum mechanical calculation.

    Attributes
    ----------
    atoms: ase.Atoms
        ASE Atoms object of the molecule.
    rdmol: rdkit.Chem.rdchem.Mol
        RDKit molecule object.
    nc: int
        Net charge of the molecule.
    label: str
        Label for the calculation.
    directory: str
        Directory to save the calculation files.
    qmparams: dict
        Parameters for the quantum mechanical calculation.
    scan_idx: list
        List of pairs of atom indices for distance scans.
    qm_calculators: list
        List of quantum mechanical calculators for each distance.
    ff_calculators: list
        List of force field calculators for each distance.
    qm_scan: list
        List of dictionaries containing the results of the quantum mechanical
        distance scans.
    ff_scan: list
        List of dictionaries containing the results of the force field distance scans.
    """

    def __init__(
        self,
        atoms,
        nc,
        label,
        scan_idx,
        scan_ranges=None,
        directory=None,
        qmparams=None,
    ):
        """
        Parameters
        ----------
        atoms: ase.Atoms
            ASE Atoms object of the molecule.
        nc: int
            Net charge of the molecule.
        label: str
            Label for the calculation.
        scan_idx: list of list of int
            List of pairs of atom indices for distance scans.
        scan_ranges: list of np.ndarray
            List of ranges for the distances to scan.
        directory: str
            Directory to save the calculation files.
        qmparams: dict
            Parameters for the quantum mechanical calculation.
        """
        self.atoms = atoms.copy()
        self.atoms.pbc = False
        self.atoms.cell = None
        self.nc = nc
        self.label = label
        if qmparams is None:
            self.qmparams = {
                "method": "wb97xd",
                "basis": "6-311+g(2d,p)",
                "opt": "modredundant",
            }
        else:
            self.qmparams = qmparams

        if directory is None:
            self.directory = os.getcwd()
        else:
            self.directory = directory
            if not os.path.exists(self.directory):
                os.makedirs(self.directory)

        # scan_idx: [[d1_1, d1_2], [d2_1, d2_2],...]
        self.scan_idx = scan_idx

        self.rmin = []
        self.dr = []
        for d in self.scan_idx:
            rmin = self.atoms.get_distance(d[0], d[1]) * 0.8
            rmin = float("%.1f" % rmin)
            self.rmin.append(rmin)
            dr = self.atoms.get_distance(d[0], d[1]) * 0.1
            dr = float("%.1f" % dr)
            self.dr.append(dr)

        if scan_ranges is None:
            self.scan_ranges = [
                np.concatenate([np.arange(self.rmin[i], 6.0, self.dr[i])])
                for i in range(len(scan_idx))
            ]
        else:
            self.scan_ranges = scan_ranges

        self.qm_calculators = [None for _ in range(len(self.scan_idx))]
        self.ff_calculators = [None for _ in range(len(self.scan_idx))]
        self.qm_scan = [
            {"distance_A": [], "energy_kjmol": [], "atoms": []}
            for _ in range(len(self.scan_idx))
        ]
        self.ff_scan = [
            {"distance_A": [], "energy_kjmol": [], "atoms": []}
            for _ in range(len(self.scan_idx))
        ]

    def do_qmscan(self, dist_idx=None, do_calc=True):
        """
        Calculate the distance using quantum mechanical methods.

        Parameters
        ----------
        dist_idx: int, optional
           List of indices of the distances to calculate. If None, calculate all
           self.scan_idx.
        do_calc: bool, optional
            Whether to perform the calculation or just return the calculator.
        """
        if dist_idx is None:
            dist_idx = range(len(self.scan_idx))

        for di in dist_idx:
            for distance in self.scan_ranges[di]:
                label = f"{self.label}_dist_{di}__{distance:.2f}"
                d0 = self.scan_idx[di][0] + 1  # +1 for 0-indexing
                d1 = self.scan_idx[di][1] + 1
                g16_addsec = f"B {d0} {d1} F"
                params = self.qmparams.copy()
                params["addsec"] = g16_addsec

                g16 = Gaussian(label=label, charge=int(self.nc), **params)
                g16.directory = os.path.abspath(self.directory)

                if len(self.qm_scan[di]["atoms"]) > 0:
                    atoms = self.qm_scan[di]["atoms"][-1].copy()
                else:
                    atoms = self.atoms.copy()

                atoms = change_distance(atoms, self.scan_idx[di], distance)
                g16.write_input(atoms, system_changes=0)
                comfile = os.path.join(self.directory, f"{label}.com")
                logfile = os.path.join(self.directory, f"{label}.log")

                with open(comfile) as f:
                    lines = f.readlines()
                del lines[-3]  # to cope with bug of ase
                with open(comfile, mode="w") as f:
                    f.writelines(lines)

                self.qm_calculators[di] = g16
                if do_calc:
                    cmd = f"g16 < {comfile}  > {logfile}"
                    output = subprocess.getoutput(cmd)
                    print(f"g16distance -- {g16.label}")
                    print(cmd)
                    print(output)

                    atoms = read(logfile)
                    self.qm_scan[di]["distance_A"].append(distance)
                    self.qm_scan[di]["atoms"].append(atoms)
                    energy = read(logfile).get_potential_energy() / (
                        units.kJ / units.mol
                    )
                    self.qm_scan[di]["energy_kjmol"].append(energy)

    def do_ffscan(self, ffxml, dist_idx=None, ini_geom="QM"):
        """
        Perform distance scan calculations using force fields.

        Parameters
        ----------
        ffxml: str
            Path to the force field XML file.
        dist_idx: list of int
            Indices of the distances to calculate. If None, calculate all distances.
        ini_geom: str
            Initial geometry to use for the force field calculation.
            QM: Use the relaxed scan geometry from the quantum mechanical calculation.
            FF: Use the initial geometry from the force field calculation.
        """
        if dist_idx is None:
            dist_idx = range(len(self.scan_idx))

        if isinstance(dist_idx, (int, str)):
            dist_idx = [int(dist_idx)]

        for di in dist_idx:
            label = f"{self.label}_dihed_{di}"
            d1 = self.scan_idx[di][0]
            d2 = self.scan_idx[di][1]
            if ini_geom == "QM":
                (
                    self.ff_scan[di]["distance_A"],
                    self.ff_scan[di]["energy_kjmol"],
                    self.ff_scan[di]["atoms"],
                ) = scan_ff_distance(
                    ffxml,
                    self.qm_scan[di]["distance_A"],
                    self.scan_idx[di],
                    atoms=self.atoms,
                    atoms_list=self.qm_scan[di]["atoms"],
                )
            else:
                (
                    self.ff_scan[di]["distance_A"],
                    self.ff_scan[di]["energy_kjmol"],
                    self.ff_scan[di]["atoms"],
                ) = scan_ff_distance(
                    ffxml, self.scan_ranges[di], self.scan_idx[di], atoms=self.atoms
                )


def scan_ff_distance(ffxml, distances, dist_atidx, atoms, atoms_list=None, bonds=None):
    """
    Relaxed distance scan using OpenMM

    Parameters
    ----------
    ffxml: str
        Path to the force field XML file.
    distances: list of float
        List of distances to scan.
    dist_atidx: list of int
        Indices of the atoms involved in the distance calculation.
    atoms: ase.Atoms
        ASE Atoms object of the molecule.
    atoms_list: list of ase.Atoms, optional
        List of ASE Atoms objects for each distance. If None, use the initial geometry.
    bonds: list of list of int, optional
        List of bonds to consider during the scan.
        If None, use the bonds from the ASE Atoms object.

    Returns
    -------
    ff_pot_kjmol: numpy.ndarray
        Array of potential energies in kJ/mol for each distance.
    ff_distanceatoms: list of ase.Atoms
        List of ASE Atoms objects for each distance.
    """
    from ..crafter.asemol import aseatoms2pdb, asemol_wrapper, merge_asemols
    from ..crafter.ffxml import check_vsite, delvsite_pdb

    distance_ff_energy = []
    ff_distance_atoms = []
    d1 = dist_atidx[0]
    d2 = dist_atidx[1]

    if bonds is None:
        aw = asemol_wrapper(atoms)
        molatoms, molecule_list, G_list = aw.get_ase_molecules(out_nX=True)
        atoms = merge_asemols(molatoms)
        if bonds is None:
            bonds = aw.get_bonds()

    pos_prev = atoms.get_positions()

    with tempfile.TemporaryDirectory() as td:
        for i in range(len(distances)):
            if atoms_list is not None:
                pdb_ase = atoms_list[i].copy()
                pdb_ase.arrays["residuenames"] = atoms.arrays["residuenames"]
                pdb_ase.arrays["residuenumbers"] = atoms.arrays["residuenumbers"]
                pdb_ase.arrays["atomtypes"] = [i for i in range(len(pdb_ase))]
                temppdb = os.path.join(td, f"temp_dist_{i}.pdb")
                aseatoms2pdb(temppdb, pdb_ase)
                # aseatoms2pdb("wovsite.pdb", pdb_ase)

            else:
                # Set the positions for the distance calculation
                temppdb = os.path.join(td, f"temp_distance_{i}.pdb")
                atoms.cell = None
                atoms.pbc = False
                atoms.positions = pos_prev
                atoms_desireddistance = change_distance(atoms, dist_atidx, distances[i])
                aseatoms2pdb(temppdb, atoms_desireddistance)
                # aseatoms2pdb("wovsite.pdb", atoms_desireddistance)

            pdb_omm = PDBFile(temppdb)
            atomlist_openmm = [a for a in pdb_omm.topology.atoms()]
            for b in bonds:
                a1 = atomlist_openmm[b[0]]
                a2 = atomlist_openmm[b[1]]
                pdb_omm.topology.addBond(a1, a2)
            pos = pdb_omm.getPositions()
            topology = pdb_omm.topology
            num_vsites = check_vsite(ffxml)

            if num_vsites > 0:
                modeller = Modeller(pdb_omm.topology, pdb_omm.positions)
                modeller.addExtraParticles(ForceField(ffxml))
                pos = modeller.getPositions()
                topology = modeller.topology

            PDBFile.writeFile(topology, pos, open(temppdb, "w"))

            # relaxed scan
            forcefield = ForceField(ffxml)
            system = forcefield.createSystem(topology, nonbondedMethod=NoCutoff)
            tempsysxml = os.path.join(td, "system.xml")
            with open(tempsysxml, "w") as output:
                output.write(XmlSerializer.serialize(system))
            restraint = HarmonicBondForce()
            map_idx_wovsite2vsite = [
                i for i, atom in enumerate(topology.atoms()) if atom.element is not None
            ]
            d1_wv = map_idx_wovsite2vsite[d1]
            d2_wv = map_idx_wovsite2vsite[d2]
            restraint.addBond(
                d1_wv,
                d2_wv,
                distances[i] * angstrom,
                100000000 * kilojoules_per_mole / angstrom**2,
            )
            system.addForce(restraint)
            integrator = LangevinMiddleIntegrator(
                300 * kelvin, 1 / picosecond, 0.004 * picoseconds
            )
            simulation = Simulation(topology, system, integrator)
            simulation.context.setPositions(pos)
            simulation.minimizeEnergy()
            state = simulation.context.getState(getPositions=True, getEnergy=True)
            crd = simulation.context.getState(getPositions=True).getPositions()
            PDBFile.writeFile(topology, crd, open(temppdb, "w"))
            PDBFile.writeFile(topology, crd, open("hoge.pdb", "w"))

            pos_prev = []
            for p in state.getPositions():
                p = np.array(p)
                xx = p[0].value_in_unit(angstrom)
                yy = p[1].value_in_unit(angstrom)
                zz = p[2].value_in_unit(angstrom)
                pos_prev.append([xx, yy, zz])

            pdb_omm = PDBFile(temppdb)
            system = forcefield.createSystem(pdb_omm.topology, nonbondedMethod=NoCutoff)
            for j, f in enumerate(system.getForces()):
                f.setForceGroup(j)
                integrator = LangevinMiddleIntegrator(
                    300 * kelvin, 1 / picosecond, 0.004 * picoseconds
                )
                simulation = Simulation(pdb_omm.topology, system, integrator)
                simulation.context.setPositions(pdb_omm.positions)

            potential_energies = []
            for gi, f in enumerate(system.getForces()):
                state = simulation.context.getState(getEnergy=True, groups={gi})
                potential_energies.append(state.getPotentialEnergy().real)

            distance_ff_energy.append(sum(potential_energies))

            if num_vsites > 0:
                # Remove virtual sites from the PDB file
                delvsite_pdb(temppdb)
            atoms = read(temppdb)
            atoms.cell = None
            atoms.pbc = False
            ff_distance_atoms.append(atoms)

    ff_pot = np.array(distance_ff_energy)
    ff_pot_kjmol = ff_pot  # (ff_pot - ff_pot.min())
    # distancesを小さい順にソート
    zip_lists = zip(distances, ff_pot_kjmol, ff_distance_atoms)
    # 昇順でソート
    zip_sort = sorted(zip_lists)
    # zipを解除
    distances, ff_pot_kjmol, ff_distance_atoms = zip(*zip_sort)

    return distances, ff_pot_kjmol, ff_distance_atoms


def change_distance(atoms, dist_atidx, desired_distance):
    """
    Change the distance between two atoms in an ASE Atoms object.

    Parameters
    ----------
    atoms: ase.Atoms
        ASE Atoms object of the molecule.
    dist_atidx: list of int
        Indices of the atoms involved in the distance calculation.
    distance: float
        Desired distance between the two atoms.

    Returns
    -------
    atoms: ase.Atoms
        ASE Atoms object with the updated distance.
    """
    from ..crafter.asemol import ase_atoms_to_nx

    d1 = dist_atidx[0]
    d2 = dist_atidx[1]
    pos1 = atoms.positions[d1]
    pos2 = atoms.positions[d2]
    bondvec = pos2 - pos1
    abs_bondvec = np.linalg.norm(bondvec)
    unit_bondvec = bondvec / np.linalg.norm(bondvec)

    # aw = asemol_wrapper(atoms)
    # try:
    #     [atoms1, atoms2], _, [G1, G2] = aw.get_ase_molecules(out_nX=True)
    # except:
    #     write("error.pdb", atoms1)
    #     assert False, "Failed to get ASE molecules. Please check the input."
    G = ase_atoms_to_nx(atoms)

    for e in G.edges():
        if (e[0] == d1 and e[1] == d2) or (e[0] == d2 and e[1] == d1):
            G.remove_edge(e[0], e[1])
            break
    S_list = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    for i, s in enumerate(S_list):
        # d2が属するかどうか
        if d2 in s.nodes():
            # d2が属するグラフのidを取得
            id = i
            break

    d2Graph = S_list[id]
    elong_idx = list(d2Graph.nodes.keys())
    pos = atoms.positions
    for idx in elong_idx:
        # diff_vec = pos[idx] - pos2
        pos[idx] += unit_bondvec * (desired_distance - abs_bondvec)

    atoms_desireddistance = atoms.copy()
    atoms_desireddistance.positions = pos
    return atoms_desireddistance
