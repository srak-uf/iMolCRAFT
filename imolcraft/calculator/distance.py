import os
import subprocess
import tempfile

import networkx as nx
import numpy as np
from ase import units
from ase.calculators.gaussian import Gaussian
from ase.io import read
from openmm import HarmonicBondForce, LangevinMiddleIntegrator
from openmm.app import ForceField, Modeller, NoCutoff, PDBFile, Simulation
from openmm.openmm import XmlSerializer
from openmm.unit import (
    angstrom,
    kelvin,
    kilojoules_per_mole,
    picosecond,
    picoseconds,
)

#: Default Gaussian settings of a relaxed distance scan.
_DEFAULT_QMPARAMS = {
    "method": "wb97xd",
    "basis": "6-311+g(2d,p)",
    "opt": "modredundant",
}

#: Force constant restraining the scanned distance during the FF minimization.
_RESTRAINT_K = 100000000 * kilojoules_per_mole / angstrom**2

#: Largest distance of the automatically generated scan range, in angstrom.
_SCAN_RMAX_A = 6.0


def _make_integrator():
    return LangevinMiddleIntegrator(300 * kelvin, 1 / picosecond, 0.004 * picoseconds)


def _write_pdb(topology, positions, path):
    """Write an OpenMM topology to a PDB file, closing it afterwards."""
    with open(path, "w") as f:
        PDBFile.writeFile(topology, positions, f)


def _add_bonds(pdb_omm, bonds):
    """Add the given (index1, index2) bonds to a PDB topology."""
    atomlist_openmm = [a for a in pdb_omm.topology.atoms()]
    for bond in bonds:
        pdb_omm.topology.addBond(atomlist_openmm[bond[0]], atomlist_openmm[bond[1]])


def _real_atom_positions_in_angstrom(topology, state):
    """
    Positions of the atoms carrying an element, in angstrom.

    Virtual sites are left out so that the result can be assigned back to an
    ``ase.Atoms``, which only holds the real atoms.
    """
    positions = state.getPositions()
    return [
        [float(c.value_in_unit(angstrom)) for c in np.array(positions[i])]
        for i, atom in enumerate(topology.atoms())
        if atom.element is not None
    ]


def _energies_by_force_group(topology, system, positions):
    """
    Potential energy of every force of the system, evaluated separately.

    Each force is moved to its own group so that its contribution can be read
    back on its own.
    """
    for i, force in enumerate(system.getForces()):
        force.setForceGroup(i)

    simulation = Simulation(topology, system, _make_integrator())
    simulation.context.setPositions(positions)
    return [
        simulation.context.getState(getEnergy=True, groups={i})
        .getPotentialEnergy()
        .real
        for i in range(system.getNumForces())
    ]


class DistanceCalculator:
    """
    Class for distance calculations using quantum mechanical methods and force fields.
    This class allows for the calculation of distances and their corresponding energies
    using both quantum mechanical methods (Gaussian) and force fields.

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

    Attributes
    ----------
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
        self.atoms = atoms.copy()
        self.atoms.pbc = False
        self.atoms.cell = None
        self.nc = nc
        self.label = label
        self.qmparams = dict(_DEFAULT_QMPARAMS) if qmparams is None else qmparams

        if directory is None:
            self.directory = os.getcwd()
        else:
            self.directory = directory
            if not os.path.exists(self.directory):
                os.makedirs(self.directory)

        # scan_idx: [[d1_1, d1_2], [d2_1, d2_2],...]
        self.scan_idx = scan_idx

        # start the scan at 80% of the current distance and step by 10% of it
        distances = [self.atoms.get_distance(d[0], d[1]) for d in self.scan_idx]
        self.rmin = [float("%.1f" % (d * 0.8)) for d in distances]
        self.dr = [float("%.1f" % (d * 0.1)) for d in distances]

        if scan_ranges is None:
            self.scan_ranges = [
                np.arange(rmin, _SCAN_RMAX_A, dr)
                for rmin, dr in zip(self.rmin, self.dr)
            ]
        else:
            self.scan_ranges = scan_ranges

        n_scans = len(self.scan_idx)
        self.qm_calculators = [None for _ in range(n_scans)]
        self.ff_calculators = [None for _ in range(n_scans)]
        self.qm_scan = [
            {"distance_A": [], "energy_kjmol": [], "atoms": []} for _ in range(n_scans)
        ]
        self.ff_scan = [
            {"distance_A": [], "energy_kjmol": [], "atoms": []} for _ in range(n_scans)
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
                g16 = self._write_g16_input(di, distance, label)
                self.qm_calculators[di] = g16

                if not do_calc:
                    continue

                comfile = os.path.join(self.directory, f"{label}.com")
                logfile = os.path.join(self.directory, f"{label}.log")
                cmd = f"g16 < {comfile}  > {logfile}"
                output = subprocess.getoutput(cmd)
                print(f"g16distance -- {g16.label}")
                print(cmd)
                print(output)

                self.qm_scan[di]["distance_A"].append(distance)
                self.qm_scan[di]["atoms"].append(read(logfile))
                self.qm_scan[di]["energy_kjmol"].append(
                    read(logfile).get_potential_energy() / (units.kJ / units.mol)
                )

    def _write_g16_input(self, di, distance, label):
        """
        Write the Gaussian input of one scan point, restraining the scanned
        distance, and return the calculator.
        """
        # +1 because Gaussian numbers atoms from one
        d0 = self.scan_idx[di][0] + 1  # +1 for 0-indexing
        d1 = self.scan_idx[di][1] + 1
        params = self.qmparams.copy()
        params["addsec"] = f"B {d0} {d1} F"

        g16 = Gaussian(label=label, charge=int(self.nc), **params)
        g16.directory = os.path.abspath(self.directory)

        # continue from the previous scan point when there is one
        if len(self.qm_scan[di]["atoms"]) > 0:
            atoms = self.qm_scan[di]["atoms"][-1].copy()
        else:
            atoms = self.atoms.copy()
        g16.write_input(change_distance(atoms, self.scan_idx[di], distance),
                        system_changes=0)

        comfile = os.path.join(self.directory, f"{label}.com")
        with open(comfile) as f:
            lines = f.readlines()
        del lines[-3]  # to cope with bug of ase
        with open(comfile, mode="w") as f:
            f.writelines(lines)
        return g16

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
            if ini_geom == "QM":
                # walk the geometries the QM scan relaxed into
                distances = self.qm_scan[di]["distance_A"]
                atoms_list = self.qm_scan[di]["atoms"]
            else:
                distances = self.scan_ranges[di]
                atoms_list = None

            (
                self.ff_scan[di]["distance_A"],
                self.ff_scan[di]["energy_kjmol"],
                self.ff_scan[di]["atoms"],
            ) = scan_ff_distance(
                ffxml,
                distances,
                self.scan_idx[di],
                atoms=self.atoms,
                atoms_list=atoms_list,
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
    distances: tuple of float
        Scanned distances, sorted ascending.
    ff_pot_kjmol: tuple of float
        Potential energies in kJ/mol for each distance.
    ff_distanceatoms: tuple of ase.Atoms
        ASE Atoms objects for each distance.
    """
    from ..crafter.asemol import aseatoms2pdb, asemol_wrapper, merge_asemols
    from ..crafter.ffxml import check_vsite, delvsite_pdb

    distance_ff_energy = []
    ff_distance_atoms = []
    d1, d2 = dist_atidx

    if bonds is None:
        aw = asemol_wrapper(atoms)
        molatoms, _, _ = aw.get_ase_molecules(out_nX=True)
        atoms = merge_asemols(molatoms)
        bonds = aw.get_bonds()

    pos_prev = atoms.get_positions()
    num_vsites = check_vsite(ffxml)
    forcefield = ForceField(ffxml)

    with tempfile.TemporaryDirectory() as td:
        for i in range(len(distances)):
            if atoms_list is not None:
                pdb_ase = atoms_list[i].copy()
                pdb_ase.arrays["residuenames"] = atoms.arrays["residuenames"]
                pdb_ase.arrays["residuenumbers"] = atoms.arrays["residuenumbers"]
                pdb_ase.arrays["atomtypes"] = [i for i in range(len(pdb_ase))]
                temppdb = os.path.join(td, f"temp_dist_{i}.pdb")
                aseatoms2pdb(temppdb, pdb_ase)
            else:
                # Set the positions for the distance calculation
                temppdb = os.path.join(td, f"temp_distance_{i}.pdb")
                atoms.cell = None
                atoms.pbc = False
                atoms.positions = pos_prev
                aseatoms2pdb(
                    temppdb, change_distance(atoms, dist_atidx, distances[i])
                )

            pdb_omm = PDBFile(temppdb)
            _add_bonds(pdb_omm, bonds)
            pos = pdb_omm.getPositions()
            topology = pdb_omm.topology

            if num_vsites > 0:
                modeller = Modeller(pdb_omm.topology, pdb_omm.positions)
                modeller.addExtraParticles(ForceField(ffxml))
                pos = modeller.getPositions()
                topology = modeller.topology

            _write_pdb(topology, pos, temppdb)

            # relaxed scan
            system = forcefield.createSystem(topology, nonbondedMethod=NoCutoff)
            tempsysxml = os.path.join(td, "system.xml")
            with open(tempsysxml, "w") as output:
                output.write(XmlSerializer.serialize(system))

            system.addForce(
                _distance_restraint(topology, d1, d2, distances[i])
            )
            simulation = Simulation(topology, system, _make_integrator())
            simulation.context.setPositions(pos)
            simulation.minimizeEnergy()
            state = simulation.context.getState(getPositions=True, getEnergy=True)
            _write_pdb(topology, state.getPositions(), temppdb)

            pos_prev = _real_atom_positions_in_angstrom(topology, state)

            # energy of the relaxed geometry without the restraint
            pdb_omm = PDBFile(temppdb)
            distance_ff_energy.append(
                sum(
                    _energies_by_force_group(
                        pdb_omm.topology,
                        forcefield.createSystem(
                            pdb_omm.topology, nonbondedMethod=NoCutoff
                        ),
                        pdb_omm.positions,
                    )
                )
            )

            if num_vsites > 0:
                # Remove virtual sites from the PDB file
                delvsite_pdb(temppdb)
            atoms = read(temppdb)
            atoms.cell = None
            atoms.pbc = False
            ff_distance_atoms.append(atoms)

    ff_pot_kjmol = np.array(distance_ff_energy)  # (ff_pot - ff_pot.min())
    # Sort by ascending distance
    zip_sort = sorted(zip(distances, ff_pot_kjmol, ff_distance_atoms))
    distances, ff_pot_kjmol, ff_distance_atoms = zip(*zip_sort)

    return distances, ff_pot_kjmol, ff_distance_atoms


def _distance_restraint(topology, d1, d2, distance):
    """
    Harmonic restraint holding one atom pair at ``distance``.

    Virtual sites shift the atom numbering of the topology, so the indices are
    mapped through the atoms that carry an element.
    """
    map_idx_wovsite2vsite = [
        i for i, atom in enumerate(topology.atoms()) if atom.element is not None
    ]
    restraint = HarmonicBondForce()
    restraint.addBond(
        map_idx_wovsite2vsite[d1],
        map_idx_wovsite2vsite[d2],
        distance * angstrom,
        _RESTRAINT_K,
    )
    return restraint


def change_distance(atoms, dist_atidx, desired_distance):
    """
    Change the distance between two atoms in an ASE Atoms object.

    The bond between the two atoms is cut in the connectivity graph and the
    fragment carrying the second atom is translated along the bond, so that the
    rest of the molecule keeps its internal geometry.

    Parameters
    ----------
    atoms: ase.Atoms
        ASE Atoms object of the molecule.
    dist_atidx: list of int
        Indices of the atoms involved in the distance calculation.
    desired_distance: float
        Desired distance between the two atoms.

    Returns
    -------
    atoms: ase.Atoms
        ASE Atoms object with the updated distance.
    """
    from ..crafter.asemol import ase_atoms_to_nx

    d1, d2 = dist_atidx
    bondvec = atoms.positions[d2] - atoms.positions[d1]
    abs_bondvec = np.linalg.norm(bondvec)
    unit_bondvec = bondvec / abs_bondvec

    graph = ase_atoms_to_nx(atoms)
    # when the two atoms are not bonded the graph stays connected and the whole
    # molecule is translated, which is what the original code did as well
    if graph.has_edge(d1, d2):
        graph.remove_edge(d1, d2)
    moved = next(c for c in nx.connected_components(graph) if d2 in c)

    pos = atoms.positions
    for idx in moved:
        pos[idx] += unit_bondvec * (desired_distance - abs_bondvec)

    atoms_desireddistance = atoms.copy()
    atoms_desireddistance.positions = pos
    return atoms_desireddistance
