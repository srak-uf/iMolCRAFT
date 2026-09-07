import copy
import os
import subprocess
import tempfile

import cclib
import networkx as nx
import numpy as np
from ase import units
from ase.calculators.gaussian import Gaussian
from ase.io import read, write
from openmm import LangevinMiddleIntegrator, PeriodicTorsionForce
from openmm.app import ForceField, Modeller, NoCutoff, PDBFile, Simulation
from openmm.openmm import XmlSerializer
from openmm.unit import (
    angstrom,
    degree,
    kelvin,
    kilojoules_per_mole,
    picosecond,
    picoseconds,
)
from rdkit import Chem

from imolcraft.io.rdkit import atoms2rdkit

#: Default Gaussian settings of a relaxed dihedral scan.
_DEFAULT_QMPARAMS = {
    "method": "wb97xd",
    "basis": "6-311+g(2d,p)",
    "opt": "modredundant",
}

#: Gaussian scan specification: 35 steps of 10 degrees.
_G16_SCAN_STEPS = "S 35 10.0"

#: Force constant restraining the scanned dihedral during the FF minimization.
_RESTRAINT_K = 10000 * kilojoules_per_mole

#: Angles used when neither a QM scan nor an explicit list is available.
_DEFAULT_ANGLES_DEG = np.arange(-180, 181, 10)

#: SMARTS of a rotatable bond.
#: https://sourceforge.net/p/rdkit/mailman/message/34360982/
_ROTATABLE_BOND_SMARTS = "[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]"


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


def _wrap_deg(angle):
    """Fold an angle into the [-180, 180) range, so that 180 becomes -180."""
    return ((angle + 180) % 360) - 180


class DihedralCalculator:
    """
    Class for dihedral angle calculations using quantum mechanical methods and
    force fields.
    This class allows for the calculation of dihedral angles and their
    corresponding energies using both quantum mechanical methods (Gaussian)
    and force fields.

    Parameters
    ----------
    atoms: ase.Atoms
        ASE Atoms object of the molecule.
    label: str
        Label for the calculation.
    nc: int
        Net charge of the molecule.
    directory: str
        Directory to save the calculation files.
    qmparams: dict
        Parameters for the quantum mechanical calculation.

    Attributes
    ----------
    scan_list: list
        List of rotatable dihedrals, each given by four atom indices.
    scan_elem_list: list
        Elements involved in each dihedral.
    qm_calculators: list
        List of quantum mechanical calculators for each dihedral angle.
    ff_calculators: list
        List of force field calculators for each dihedral angle.
    qm_scan: list
        List of dictionaries containing the results of the quantum mechanical
        dihedral scans.
    ff_scan: list
        List of dictionaries containing the results of the force field dihedral scans.
    """

    def __init__(self, atoms, label, nc=int(0), directory=None, qmparams=None):
        self.atoms = atoms.copy()
        self.atoms.pbc = False
        self.atoms.cell = None
        self.rdmol, _, _ = atoms2rdkit(atoms, nc=nc)
        self.nc = nc  # sum([atom.GetFormalCharge() for atom in self.rdmol.GetAtoms()])
        self.label = label
        self.qmparams = dict(_DEFAULT_QMPARAMS) if qmparams is None else qmparams
        self.directory = os.getcwd() if directory is None else directory

        # scan_list: [[d1_1, d1_2, d1_3, d1_4], [d2_1, d2_2, d2_3, d2_4],...]
        # scan_elem_list: [[H, C, C, H], [C, C, C, C],...]
        self.scan_list, self.scan_elem_list = get_rotatable_dihedral(self.rdmol)

        n_scans = len(self.scan_list)
        self.qm_calculators = [None for _ in range(n_scans)]
        self.ff_calculators = [None for _ in range(n_scans)]
        self.qm_scan = [
            {"angle_deg": [], "energy_kjmol": [], "atoms": []} for _ in range(n_scans)
        ]
        self.ff_scan = [
            {"angle_deg": [], "energy_kjmol": [], "atoms": []} for _ in range(n_scans)
        ]

    def do_qmscan(self, dihed_idx=None, do_calc=True):
        """
        Perform dihedral angle calculations using quantum mechanical methods.

        Parameters
        ----------
        dihed_idx: list of int
            List of indices of the dihedral angles to be calculated.
        do_calc: bool
            If True, perform the calculations. If False, only prepare the input files.
        """
        if dihed_idx is None:
            dihed_idx = [i for i in range(len(self.scan_list))]

        for di in dihed_idx:
            label = f"{self.label}_dihed_{di}"  # f"{key}_dihed_{di}"
            g16 = self._write_g16_input(di, label)
            self.qm_calculators[di] = g16

            if not do_calc:
                continue

            comfile = os.path.join(self.directory, f"{label}.com")
            logfile = os.path.join(self.directory, f"{label}.log")
            cmd = f"g16 < {comfile}  > {logfile}"
            output = subprocess.getoutput(cmd)
            print(f"g16dihedral -- {g16.label}")
            print(cmd)
            print(output)

            (
                self.qm_scan[di]["angle_deg"],
                self.qm_scan[di]["energy_kjmol"],
                self.qm_scan[di]["atoms"],
            ) = load_g16scan(logfile)

    def _write_g16_input(self, di, label):
        """
        Write the Gaussian input of one dihedral scan and return the calculator.
        """
        # +1 because Gaussian numbers atoms from one
        d0, d1, d2, d3 = (i + 1 for i in self.scan_list[di])
        params = self.qmparams.copy()
        params["addsec"] = f"D {d0} {d1} {d2} {d3} {_G16_SCAN_STEPS}"

        g16 = Gaussian(label=label, charge=int(self.nc), **params)
        g16.directory = os.path.abspath(self.directory)
        g16.write_input(self.atoms, system_changes=0)

        comfile = os.path.join(self.directory, f"{label}.com")
        with open(comfile) as f:
            lines = f.readlines()
        del lines[-3]  # to cope with bug of ase
        with open(comfile, mode="w") as f:
            f.writelines(lines)
        return g16

    def do_ffscan(self, ffxml, dihed_idx=None, angles=None, ini_geom="QM"):
        """
        Perform dihedral angle calculations using force fields.

        Parameters
        ----------
        ffxml: str
            Path to the force field XML file.
        dihed_idx: list of int
            List of indices of the dihedral angles to be calculated.
        angles: QM or list of float
            If QM, use the angles from the quantum mechanical calculation.
            List of dihedral angles (in degrees) to scan.
        ini_geom: str
            Initial geometry for the dihedral scan. Can be "QM" or "FF".
            QM: Use the relaxed scan geometry from the quantum mechanical calculation.
            FF: Use the initial geometry from the force field calculation.
        """
        if dihed_idx is None:
            dihed_idx = [int(i) for i in range(len(self.scan_list))]

        # a bare index may be given instead of a list; a string is iterable, so
        # it has to be normalised here or it would be walked character by
        # character further down
        if isinstance(dihed_idx, (int, str)):
            dihed_idx = [int(dihed_idx)]

        for di in dihed_idx:
            if ini_geom == "QM":
                # walk the geometries the QM scan relaxed into
                (
                    self.ff_scan[di]["angle_deg"],
                    self.ff_scan[di]["energy_kjmol"],
                    self.ff_scan[di]["atoms"],
                ) = scan_ff_dihedral(
                    ffxml,
                    self.qm_scan[di]["angle_deg"],
                    self.scan_list[di],
                    atoms_list=self.qm_scan[di]["atoms"],
                )
            else:
                angles = self._resolve_angles(di, angles)
                (
                    self.ff_scan[di]["angle_deg"],
                    self.ff_scan[di]["energy_kjmol"],
                    self.ff_scan[di]["atoms"],
                ) = scan_ff_dihedral(
                    ffxml,
                    angles,
                    self.scan_list[di],
                    geoopt_atoms=self.atoms,
                )

    def _resolve_angles(self, di, angles):
        """
        Pick the angles to scan: the given ones, the QM ones, or a default grid.

        Raises
        ------
        ValueError
            When ``angles`` is a string other than "QM". Returning it unchanged
            would let a string reach the scan, where ``len()`` counts its
            characters instead of its angles.
        """
        if angles is None:
            if len(self.qm_scan[di]["angle_deg"]) == 0:
                return _DEFAULT_ANGLES_DEG
            return self.qm_scan[di]["angle_deg"]
        if not isinstance(angles, str):
            return np.array(angles)
        if angles == "QM":
            return self.qm_scan[di]["angle_deg"]
        raise ValueError(
            f'angles must be "QM" or a sequence of angles in degrees: {angles!r}'
        )


def _fix_g16_version_line(g16logfile):
    """
    Join the date of the Gaussian version banner to the revision.

    cclib expects five whitespace separated fields there, so
    "Gaussian 16: Apple_M1-G16RevC.02 7-Dec-2021" has to become
    "Gaussian 16: Apple_M1-G16RevC.02_7-Dec-2021". The file is rewritten.
    """
    with open(g16logfile) as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if (
            line.replace("\n", "") == " *********************************************"
            and len(lines[i + 1].split()) == 5
        ):
            parts = lines[i + 1].split()
            parts[2] = parts[2] + "_" + parts[3]
            del parts[3]
            lines[i + 1] = " ".join(parts) + "\n"
            break

    with open(g16logfile, mode="w") as f:
        f.writelines(lines)


def load_g16scan(g16logfile):
    """
    Load the results of a Gaussian 16 scan log file.

    Parameters
    ----------
    g16logfile: str
        Path to the Gaussian 16 log file.

    Returns
    -------
    angle: numpy.ndarray
        Array of dihedral angles (in degrees).
    energy: numpy.ndarray
        Array of energies (in kJ/mol) for each dihedral angle, relative to the
        lowest point of the scan.
    aseatoms: list of ase.Atoms
        List of ASE Atoms objects for each dihedral angle.
    """
    try:
        _fix_g16_version_line(g16logfile)

        dihed_cclib = cclib.io.ccread(g16logfile)
        energy = dihed_cclib.scanenergies
        angle = dihed_cclib.scanparm[0]

        ase_g16log = read(g16logfile)
        aseatoms = []
        for sc in dihed_cclib.scancoords:
            sc_tmp = ase_g16log.copy()
            for j in range(len(sc)):
                sc_tmp[j].position = sc[j]
            aseatoms.append(sc_tmp)

        # Sort in ascending order
        angle, energy, aseatoms = zip(*sorted(zip(angle, energy, aseatoms)))
        angle = np.array(angle)
        energy = np.array(energy)
        energy = (energy - energy.min()) / (units.kJ * units.mol**-1)
    except Exception:
        import traceback

        traceback.print_exc()
        energy = None
        angle = None
        aseatoms = None
        print(f"Warning: Failed reading results: {g16logfile}")

    return angle, energy, aseatoms


def _pick_outer_atom(atom, d1, d2):
    """
    Index of a neighbour of ``atom`` outside the rotatable bond ``d1``-``d2``,
    which gives the dihedral one of its two outer reference atoms.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        One end of the rotatable bond.
    d1, d2 : int
        Indices of the two atoms forming the rotatable bond.

    Returns
    -------
    int
        Index of the outer atom.

    Raises
    ------
    ValueError
        When the atom has no neighbour besides its bond partner. The rotatable
        bond SMARTS requires both ends to have a degree of at least two, so
        this cannot happen for a match of it; it only guards a caller that
        supplies its own bond list.
    """
    for bond in atom.GetBonds():
        for idx in (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()):
            if idx != d1 and idx != d2:
                return idx
    raise ValueError(
        f"Atom {atom.GetIdx()} of the rotatable bond {d1}-{d2} has no other "
        "neighbour, so no dihedral can be defined around that bond"
    )


def get_rotatable_dihedral(rdmol):
    """
    Get the list of rotatable dihedral angles in the molecule.

    Parameters
    ----------
    rdmol: rdkit.Chem.rdchem.Mol
        RDKit molecule object.

    Returns
    -------
    dihedral_list: list of list of int
        List of dihedral angles, each defined by a list of four atom indices.
        Every index is an int; see :func:`_pick_outer_atom`.
    dihedral_elem_list: list of list of str
        List of elements involved in the dihedral angles, each defined by a list of
        four element symbols.
    """
    id_mol = copy.deepcopy(rdmol)
    RotatableBond = Chem.MolFromSmarts(_ROTATABLE_BOND_SMARTS)

    dihedral_list = []
    dihedral_elem_list = []
    for d1, d2 in id_mol.GetSubstructMatches(RotatableBond):
        d0 = _pick_outer_atom(id_mol.GetAtoms()[d1], d1, d2)
        d3 = _pick_outer_atom(id_mol.GetAtoms()[d2], d1, d2)

        dihedral = [d0, d1, d2, d3]
        dihedral_list.append(dihedral)
        dihedral_elem_list.append(
            [id_mol.GetAtoms()[i].GetSymbol() for i in dihedral]
        )

    return dihedral_list, dihedral_elem_list


def _dihedral_restraint(topology, dihed_atidx, angle_deg):
    """
    Torsion restraint holding one dihedral at ``angle_deg``.

    Virtual sites shift the atom numbering of the topology, so the indices are
    mapped through the atoms that carry an element.
    """
    map_idx_wovsite2vsite = [
        i for i, atom in enumerate(topology.atoms()) if atom.element is not None
    ]
    restraint = PeriodicTorsionForce()
    restraint.addTorsion(
        *[map_idx_wovsite2vsite[i] for i in dihed_atidx],
        1,
        (angle_deg + 180) * degree,
        _RESTRAINT_K,
    )
    return restraint


def scan_ff_dihedral(
    ffxml, angles, dihed_atidx, atoms_list=None, geoopt_atoms=None, bonds=None
):
    """
    Relaxed dihedral scan using OpenMM

    Parameters
    ----------
    ffxml: str
        Path to the force field XML file.
    angles: list of float
        List of dihedral angles (in degrees) to scan.
    dihed_atidx: list of int
        List of atom indices defining the dihedral angle.
    atoms_list: list of ase.Atoms
        List of ASE Atoms objects for each scan step.
    geoopt_atoms: ase.Atoms
        ASE Atoms of the geometry optimizatied structure.
    bonds: list of tuple
        List of tuples defining the bonds in the system.

    Returns
    -------
    angles: tuple of float
        Scanned angles, sorted ascending.
    ff_pot_kjmol: tuple of float
        Potential energies (in kJ/mol) for each dihedral angle, relative to the
        lowest point of the scan.
    ff_dihedatoms: tuple of ase.Atoms
        ASE Atoms objects after energy minimization for each angle.
    """
    from ..crafter.asemol import aseatoms2pdb, asemol_wrapper
    from ..crafter.ffxml import check_vsite, delvsite_pdb

    dihedral_ffenergy = []
    ff_dihedatoms = []
    d1, d2, d3, d4 = dihed_atidx

    if atoms_list is None and geoopt_atoms is None:
        raise ValueError(
            "Both atoms_list and geoopt_atoms are None. Please provide one of them."
        )
    elif geoopt_atoms is not None:
        # start the scan at the angle the optimized geometry already has, so
        # that each step only has to rotate a little from the previous one
        angle_geoopt = _wrap_deg(geoopt_atoms.get_dihedral(d1, d2, d3, d4))
        min_idx = np.argmin(np.abs(angle_geoopt - angles))
        angles = np.concatenate((angles[min_idx:], angles[:min_idx]))
        aw = asemol_wrapper(geoopt_atoms)
        [atoms], _, [G] = aw.get_ase_molecules(out_nX=True)
        if bonds is None:
            bonds = aw.get_bonds()
        pos_prev = geoopt_atoms.get_positions()
    elif atoms_list is not None:
        if len(angles) != len(atoms_list):
            raise ValueError(
                "The length of angles and atoms_list must be the same: "
                f"{len(angles)} != {len(atoms_list)}"
            )
        aw = asemol_wrapper(atoms_list[0])
        [atoms], _, [G] = aw.get_ase_molecules(out_nX=True)
        if bonds is None:
            bonds = aw.get_bonds()

    num_vsites = check_vsite(ffxml)
    forcefield = ForceField(ffxml)

    with tempfile.TemporaryDirectory() as td:
        for i in range(len(angles)):
            temppdb = os.path.join(td, f"temp_dihed_{i}.pdb")
            if atoms_list is not None:
                pdb_ase = atoms_list[i].copy()
                pdb_ase.arrays["atomtypes"] = [i for i in range(len(pdb_ase))]
            else:
                # Set dihedral angle
                atoms.cell = None
                atoms.pbc = False
                atoms.positions = pos_prev
                pdb_ase = rotate_dihedral(atoms, dihed_atidx, angles[i])

            aseatoms2pdb(temppdb, pdb_ase)
            pdb_omm = PDBFile(temppdb)
            _add_bonds(pdb_omm, bonds)
            _write_pdb(pdb_omm.topology, pdb_omm.positions, temppdb)

            # relaxed scan
            # the extra particles have to be part of the topology the system,
            # the restraint and the simulation are all built from, otherwise
            # their particle counts do not line up
            if num_vsites > 0:
                modeller = Modeller(pdb_omm.topology, pdb_omm.positions)
                modeller.addExtraParticles(ForceField(ffxml))
                topology = modeller.topology
                pos = modeller.getPositions()
            else:
                topology = pdb_omm.topology
                pos = pdb_omm.positions

            system = forcefield.createSystem(topology, nonbondedMethod=NoCutoff)
            tempsysxml = os.path.join(td, "system.xml")
            with open(tempsysxml, "w") as output:
                output.write(XmlSerializer.serialize(system))

            system.addForce(_dihedral_restraint(topology, dihed_atidx, angles[i]))
            simulation = Simulation(topology, system, _make_integrator())
            simulation.context.setPositions(pos)
            simulation.minimizeEnergy()
            state = simulation.context.getState(getPositions=True, getEnergy=True)
            _write_pdb(topology, state.getPositions(), temppdb)

            pos_prev = _real_atom_positions_in_angstrom(topology, state)

            # energy of the relaxed geometry without the restraint
            pdb_omm = PDBFile(temppdb)
            dihedral_ffenergy.append(
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
            ff_dihedatoms.append(atoms)

    ff_pot = np.array(dihedral_ffenergy)
    ff_pot_kjmol = ff_pot - ff_pot.min()
    # Sort by ascending angle
    zip_sort = sorted(zip(angles, ff_pot_kjmol, ff_dihedatoms))
    angles, ff_pot_kjmol, ff_dihedatoms = zip(*zip_sort)

    return angles, ff_pot_kjmol, ff_dihedatoms


def rotate_dihedral(atoms, dihed_list, desired_angle, chemical_bonds=None):
    """
    Rotate the dihedral angle of a molecule.

    The bond in the middle of the dihedral is cut in the connectivity graph and
    the fragment carrying the third atom is turned around that bond, so that
    the rest of the molecule keeps its internal geometry.

    Parameters
    ----------
    atoms: ase.Atoms
        The molecule to be rotated.
    dihed_list: list of int
        The indices of the atoms defining the dihedral angle.
    desired_angle: float
        The desired dihedral angle in degrees.
    chemical_bonds: pandas.DataFrame
        The definition of chemical bonds: element1, element2, cutoff.

    Returns
    -------
    atoms_rotated: ase.Atoms
        The rotated molecule.
    """
    from ..crafter.asemol import asemol_wrapper

    d1, d2, d3, d4 = dihed_list
    r2 = atoms.positions[d2]
    r3 = atoms.positions[d3]
    dangle_in = atoms.get_dihedral(d1, d2, d3, d4)

    aw = asemol_wrapper(atoms, chemical_bonds=chemical_bonds)
    try:
        [atoms], _, [G] = aw.get_ase_molecules(out_nX=True)
    except Exception as exc:
        # the input is dumped so that the offending geometry can be inspected
        print(atoms)
        write("error.pdb", atoms)
        raise ValueError(
            "Failed to get ASE molecules. Please check the input. "
            "The geometry was written to error.pdb."
        ) from exc

    rot_axis = r3 - r2
    rot_axis /= np.linalg.norm(rot_axis)
    theta = np.radians(_wrap_deg(desired_angle) - _wrap_deg(dangle_in))

    # Rodrigues' rotation matrix about rot_axis
    K = np.array(
        [
            [0, -rot_axis[2], rot_axis[1]],
            [rot_axis[2], 0, -rot_axis[0]],
            [-rot_axis[1], rot_axis[0], 0],
        ]
    )
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)

    if G.has_edge(d2, d3):
        G.remove_edge(d2, d3)
    rotated = next(c for c in nx.connected_components(G) if d3 in c)

    pos = atoms.positions
    for idx in rotated:
        pos[idx] = r2 + np.dot(R, pos[idx] - r2)

    atoms_rotated = atoms.copy()
    atoms_rotated.positions = pos

    return atoms_rotated
