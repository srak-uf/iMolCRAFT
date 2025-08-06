from rdkit import Chem
from openmm import PeriodicTorsionForce, LangevinMiddleIntegrator
from openmm.unit import kelvin, picosecond, picoseconds, degree, kilojoules_per_mole, angstrom
from openmm.app import NoCutoff, Simulation, PDBFile, ForceField
from openmm.openmm import XmlSerializer
import numpy as np
from ase.io import read, write
from ase.calculators.gaussian import Gaussian
from ase import units
import copy, os, tempfile, subprocess
import cclib
import networkx as nx

class DihedCalculator:
    """
    Class for dihedral angle calculations using quantum mechanical methods and force fields.
    This class allows for the calculation of dihedral angles and their corresponding energies
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
    dihedral_list: list
        List of dihedral angles in the molecule.
    dihedral_elem_list: list
        List of elements involved in the dihedral angles.
    qm_calculators: list
        List of quantum mechanical calculators for each dihedral angle.
    ff_calculators: list
        List of force field calculators for each dihedral angle.
    qm_dihedscan: list
        List of dictionaries containing the results of the quantum mechanical dihedral scans.
    ff_dihedscan: list
        List of dictionaries containing the results of the force field dihedral scans.
    """
    def __init__(self, atoms, rdkitmol, label, directory=None, qmparams=None):
        """
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
        """
        self.atoms = atoms.copy()
        self.atoms.pbc = False
        self.atoms.cell = None
        self.rdmol = rdkitmol
        self.nc = sum([atom.GetFormalCharge() for atom in self.rdmol.GetAtoms()])
        self.label = label
        if qmparams is None:
            self.qmparams = {
                            "method": "wb97xd",
                            "basis": "6-311+g(2d,p)",
                            "opt": "modredundant",
                            # "mem": "92GB",
                            # "nprocshared": 40,
                            }
        else:
            self.qmparams = qmparams

        if directory is None:
            self.directory = os.getcwd()
        else:
            self.directory = directory
        
        # dihedral_list: [[d1_1, d1_2, d1_3, d1_4], [d2_1, d2_2, d2_3, d2_4],...]
        # dihedral_elem_list: [[H, C, C, H], [C, C, C, C],...]
        self.dihedral_list, self.dihedral_elem_list = get_rotatable_dihedral(self.rdmol)
        self.qm_calculators = [None for _ in range(len(self.dihedral_list))]
        self.ff_calculators = [None for _ in range(len(self.dihedral_list))]
        self.qm_dihedscan = [{"angles_deg": [], "energy_kjmol": [], "atoms": []} for _ in range(len(self.dihedral_list))]
        self.ff_dihedscan = [{"angles_deg": [], "energy_kjmol": [], "atoms": []} for _ in range(len(self.dihedral_list))]

    def get_dihedral_qm(self, dihed_idx=None, do_calc=True):
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
            dihed_idx = [i for i in range(len(self.dihedral_list))]
        
        for di in dihed_idx:
            label = f"{self.label}_dihed_{di}" # f"{key}_dihed_{di}"
            d0 = self.dihedral_list[di][0] + 1
            d1 = self.dihedral_list[di][1] + 1
            d2 = self.dihedral_list[di][2] + 1
            d3 = self.dihedral_list[di][3] + 1
            g16_addsec = f"D {d0} {d1} {d2} {d3} S 35 10.0"
            params = self.qmparams.copy()
            params["addsec"] = g16_addsec
            g16 = Gaussian(
                    label= label,
                    charge= int(self.nc),
                    **params,
                )
            g16.directory = os.path.abspath(self.directory)
            g16.write_input(self.atoms, system_changes=0)
            comfile = os.path.join(self.directory, f"{label}.com")
            logfile = os.path.join(self.directory, f"{label}.log")
            with open(comfile) as f:
                lines = f.readlines()
            del lines[-3]     # to cope with bug of ase
            with open(comfile, mode="w") as f:
                f.writelines(lines)

            self.qm_calculators[di] = g16

            if do_calc:
                cmd = f"g16 < {comfile}  > {logfile}"
                output = subprocess.getoutput(cmd)
                print(f"g16dihedral -- {g16.label}")
                print(cmd)
                print(output)

                self.qm_dihedscan[di]["angles_deg"],  \
                self.qm_dihedscan[di]["energies_kjmol"], \
                self.qm_dihedscan[di]["atoms"] = load_g16scan(logfile)
    
    def get_dihedral_ff(self, ffxml, dihed_idx=None, angles=None, ini_geom="QM"):
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
            dihed_idx = [int(i) for i in range(len(self.dihedral_list))]
        
        if isinstance(dihed_idx, int or str):
            dihed_idx = [int(dihed_idx)]
        
        for di in dihed_idx:
            label = f"{self.label}_dihed_{di}"
            d0 = self.dihedral_list[di][0]
            d1 = self.dihedral_list[di][1]
            d2 = self.dihedral_list[di][2]
            d3 = self.dihedral_list[di][3]
            if ini_geom == "QM":
                self.ff_dihedscan[di]["angles_deg"] ,\
                self.ff_dihedscan[di]["energies_kjmol"],\
                self.ff_dihedscan[di]["atoms"] = \
                            scan_ff_dihedral(ffxml,
                                             self.qm_dihedscan[di]["angles_deg"],
                                             self.dihedral_list[di],
                                             atoms_list=self.qm_dihedscan[di]["atoms"],
                                             )
            else:
                if angles is None:
                    if len(self.qm_dihedscan[di]["angles_deg"]) == 0:
                        angles = np.arange(-180, 181, 10)
                    else:
                        angles = self.qm_dihedscan[di]["angles_deg"]
                elif not isinstance(angles, str):
                    angles = np.array(angles)
                elif angles == "QM":
                    angles = self.qm_dihedscan[di]["angles_deg"]

                self.ff_dihedscan[di]["angles_deg"]  ,\
                self.ff_dihedscan[di]["energies_kjmol"] ,\
                self.ff_dihedscan[di]["atoms"] = \
                            scan_ff_dihedral(ffxml,
                                             angles,
                                             self.dihedral_list[di],
                                             geoopt_atoms=self.atoms,
                                             )

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
        Array of energies (in kJ/mol) for each dihedral angle.
    aseatoms: list of ase.Atoms
        List of ASE Atoms objects for each dihedral angle.
    """
    try:
        dihed_cclib = cclib.io.ccread(g16logfile)
        energy = dihed_cclib.scanenergies
        angle = dihed_cclib.scanparm[0]
        aseatoms = []
        ase_g16log = read(g16logfile)
        for i_sc, sc in enumerate(dihed_cclib.scancoords):
            sc_tmp = ase_g16log.copy()
            for j, atom in enumerate(sc):
                sc_tmp[j].position = sc[j]
            aseatoms.append(sc_tmp)
        
        zip_lists = zip(angle, energy, aseatoms)
        # 昇順でソート
        zip_sort = sorted(zip_lists)
        # zipを解除
        angle, energy, aseatoms = zip(*zip_sort)
        angle = np.array(angle)
        energy = np.array(energy)
        energy = (energy - energy.min()) / (units.kJ * units.mol**-1)
    except:
        import traceback
        traceback.print_exc()
        energy = None
        angle = None
        aseatoms = None
        print(f"Warning: Failed reading results: {g16logfile}")
    
    return angle, energy, aseatoms

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
    dihedral_elem_list: list of list of str
        List of elements involved in the dihedral angles, each defined by a list of four element symbols.
    """
    id_mol = copy.deepcopy(rdmol)
    # https://sourceforge.net/p/rdkit/mailman/message/34360982/
    RotatableBond = Chem.MolFromSmarts('[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]')
    rotatable_list = id_mol.GetSubstructMatches(RotatableBond)
    dihedral_list = []
    dihedral_elem_list = []
    for i in range(len(rotatable_list)):
        rot_i = rotatable_list[i]
        d1 = rot_i[0]
        d2 = rot_i[1]

        d1_bonds = id_mol.GetAtoms()[d1].GetBonds()
        d2_bonds = id_mol.GetAtoms()[d2].GetBonds()

        for bond0 in d1_bonds:
            b0 = bond0.GetBeginAtomIdx()
            b1 = bond0.GetEndAtomIdx()
            if b0 != d1 and b0 != d2:
                d0 = b0
                break
            if b1 != d1 and b1 != d2:
                d0 = b1
                break

        for bond1 in d2_bonds:
            b0 = bond1.GetBeginAtomIdx()
            b1 = bond1.GetEndAtomIdx()
            if b0 != d1 and b0 != d2:
                d3 = b0
                break
            if b1 != d1 and b1 != d2:
                d3 = b1
                break

        dihedral = [d0, d1, d2, d3]

        d0_elem = id_mol.GetAtoms()[d0].GetSymbol()
        d1_elem = id_mol.GetAtoms()[d1].GetSymbol()
        d2_elem = id_mol.GetAtoms()[d2].GetSymbol()
        d3_elem = id_mol.GetAtoms()[d3].GetSymbol()
        dihedral_elem = [d0_elem, d1_elem, d2_elem, d3_elem]

        dihedral_list.append(dihedral)
        dihedral_elem_list.append(dihedral_elem)
        
    return dihedral_list, dihedral_elem_list

def scan_ff_dihedral(ffxml, angles, dihed_atidx, atoms_list=None, geoopt_atoms=None, bonds=None):
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
    ff_pot_kjmol: numpy.ndarray
        Array of potential energies (in kJ/mol) for each dihedral angle.
    ff_dihedatoms: list of ase.Atoms
        List of ASE Atoms objects after energy minimization for each angle.
    """
    from ..crafter.asemol import aseatoms2pdb, asemol_wrapper

    dihedral_ffenergy = []
    ff_dihedatoms = []
    d1 = dihed_atidx[0]
    d2 = dihed_atidx[1]
    d3 = dihed_atidx[2]
    d4 = dihed_atidx[3]

    if atoms_list is None and geoopt_atoms is None:
        assert False, "Both atoms_list and geoopt_atoms are None. Please provide one of them."
    elif geoopt_atoms is not None:
        # Set dihedral angle
        ## change the dihedral angle from -180 to 180
        angle_geoopt = geoopt_atoms.get_dihedral(d1,d2,d3,d4)
        angle_geoopt = ((angle_geoopt + 180) % 360) - 180
        min_idx = np.argmin(np.abs(angle_geoopt - angles))
        angles = np.concatenate((angles[min_idx:], angles[:min_idx]))
        aw = asemol_wrapper(geoopt_atoms)
        [atoms], _, [G] = aw.get_ase_molecules(out_nX=True)
        if bonds is None:
            bonds = aw.get_bonds()
        pos_prev = geoopt_atoms.get_positions()
    elif atoms_list is not None:
        if len(angles) != len(atoms_list):
            assert False, "The length of angles and atoms_list must be the same."
        aw = asemol_wrapper(atoms_list[0])
        [atoms], _, [G] = aw.get_ase_molecules(out_nX=True)
        if bonds is None:
            bonds = aw.get_bonds()

    with tempfile.TemporaryDirectory() as td:
        for i in range(len(angles)):
            if atoms_list is not None:
                pdb_ase = atoms_list[i].copy()
                pdb_ase.arrays["atomtypes"] = [i for i in range(len(pdb_ase))]
                temppdb = os.path.join(td, f"temp_dihed_{i}.pdb")
                aseatoms2pdb(temppdb, pdb_ase)
                pdb_omm = PDBFile(temppdb)
                atomlist_openmm = [a for a in pdb_omm.topology.atoms()]
                for b in bonds:
                    a1 = atomlist_openmm[b[0]]
                    a2 = atomlist_openmm[b[1]]
                    pdb_omm.topology.addBond(a1, a2)
                PDBFile.writeFile(pdb_omm.topology, pdb_omm.positions, open(temppdb, 'w'))
            
            elif geoopt_atoms is not None:
                # Set dihedral angle
                temppdb = os.path.join(td, f"temp_dihed_{i}.pdb")
                atoms.cell = None
                atoms.pbc = False
                atoms.positions = pos_prev
                atoms_rotated = rotate_dihedral(atoms, dihed_atidx, angles[i])
                aseatoms2pdb(temppdb, atoms_rotated)
                pdb_omm = PDBFile(temppdb)
                atomlist_openmm = [a for a in pdb_omm.topology.atoms()]
                for b in bonds:
                    a1 = atomlist_openmm[b[0]]
                    a2 = atomlist_openmm[b[1]]
                    pdb_omm.topology.addBond(a1, a2)
                PDBFile.writeFile(pdb_omm.topology, pdb_omm.positions, open(temppdb, 'w'))
                
            # relaxed scan
            forcefield = ForceField(ffxml)
            system = forcefield.createSystem(pdb_omm.topology, nonbondedMethod=NoCutoff)
            tempsysxml = os.path.join(td, "system.xml")
            with open(tempsysxml, 'w') as output:
                output.write(XmlSerializer.serialize(system))

            restraint = PeriodicTorsionForce()
            restraint.addTorsion(d1,d2,d3,d4, 1, (angles[i]+180)*degree, 10000*kilojoules_per_mole)
            system.addForce(restraint)
            integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.004*picoseconds)
            simulation = Simulation(pdb_omm.topology, system,integrator)
            simulation.context.setPositions(pdb_omm.positions)
            simulation.minimizeEnergy()
            state = simulation.context.getState(getPositions=True, getEnergy=True)
            crd = simulation.context.getState(getPositions=True).getPositions()
            PDBFile.writeFile(pdb_omm.topology, crd, open(temppdb, 'w'))
            
            pos_prev = []
            for p in state.getPositions():
                p = np.array(p)
                xx = p[0].value_in_unit(angstrom)
                yy = p[1].value_in_unit(angstrom)
                zz = p[2].value_in_unit(angstrom)
                pos_prev.append([xx, yy, zz])

            pdb_omm = PDBFile(temppdb)
            system = forcefield.createSystem(pdb_omm.topology,nonbondedMethod=NoCutoff)
            for j, f in enumerate(system.getForces()):
                f.setForceGroup(j)
                integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.004*picoseconds)
                simulation = Simulation(pdb_omm.topology, system, integrator)
                simulation.context.setPositions(pdb_omm.positions)
            
            potential_energies = []
            for gi, f in enumerate(system.getForces()):
                state = simulation.context.getState(getEnergy=True, groups={gi})
                potential_energies.append(state.getPotentialEnergy().real)
            
            dihedral_ffenergy.append(sum(potential_energies))
            atoms = read(temppdb)
            atoms.cell = None
            atoms.pbc = False
            ff_dihedatoms.append(atoms)

    ff_pot = np.array(dihedral_ffenergy)
    ff_pot_kjmol = (ff_pot - ff_pot.min())
    # anglesを小さい順にソート
    zip_lists = zip(angles, ff_pot_kjmol, ff_dihedatoms)
    # 昇順でソート
    zip_sort = sorted(zip_lists)
    # zipを解除
    angles, ff_pot_kjmol, ff_dihedatoms = zip(*zip_sort)

    return angles, ff_pot_kjmol, ff_dihedatoms

def rotate_dihedral(atoms, dihed_list, desired_angle, chemical_bonds=None):
    """
    Rotate the dihedral angle of a molecule.

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
    from ..crafter.asemol import aseatoms2pdb, asemol_wrapper

    d1 = dihed_list[0]
    d2 = dihed_list[1]
    d3 = dihed_list[2]
    d4 = dihed_list[3]
    r1 = atoms.positions[d1]
    r2 = atoms.positions[d2]
    r3 = atoms.positions[d3]
    r4 = atoms.positions[d4]
    dangle_in = atoms.get_dihedral(d1, d2, d3, d4)

    aw = asemol_wrapper(atoms, chemical_bonds=chemical_bonds)
    try:
        [atoms], _, [G] = aw.get_ase_molecules(out_nX=True)
    except:
        print(atoms)
        write("error.pdb", atoms)
        assert False, "Failed to get ASE molecules. Please check the input."

    rot_axis = r3 - r2
    rot_axis /= np.linalg.norm(rot_axis)

    desired_angle = ((desired_angle + 180) % 360) - 180
    dangle_in = ((dangle_in + 180) % 360) - 180
    theta = np.radians(desired_angle-dangle_in)
    K = np.array([[0, -rot_axis[2], rot_axis[1]],
              [rot_axis[2], 0, -rot_axis[0]],
              [-rot_axis[1], rot_axis[0], 0]])
    R= np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)

    for e in G.edges():
        if (e[0] == d2 and e[1] == d3) or (e[0] == d3 and e[1] == d2):
            G.remove_edge(e[0], e[1])
            break
    S_list = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    for i, s in enumerate(S_list):
        # d3が属するかどうか
        if d3 in s.nodes():
            # d3が属するグラフのidを取得
            id = i
            break

    d3Graph = S_list[id]
    rot_idx = list(d3Graph.nodes.keys())
    pos = atoms.positions
    for idx in rot_idx:
        vec = pos[idx] - r2
        vec_rotated = np.dot(R, vec)
        pos[idx] = r2 + vec_rotated

    atoms_rotated = atoms.copy()
    atoms_rotated.positions = pos

    return atoms_rotated


def parse_g16scan(file):
    """
    Parse the output of a Gaussian 16 scan log file.

    Parameters
    ----------
    file: str
        Path to the Gaussian 16 log file.

    Returns
    -------
    scanned_energy: list of float
        List of energies for each scanned dihedral angle.
    """
    with open(file) as f:
        lines = f.readlines()
        parserd_lines = []
        scanned_energy = []
        for line in lines:
            if "Optimization completed." in line:
                scanned_energy.append(parserd_lines[-1])
                parserd_lines.append(line)
            elif "SCF Done:" in line:
                parserd_lines.append(float(line.split()[4]))
    return scanned_energy

# def draw_dihedral_plots(mol_info, molkey):
#     import matplotlib.pyplot as plt
#     plt.figure(figsize=(4, 3))
#     n_dihedrals = len(mol_info[molkey]["dihedral_angle"])
#     dihedral_pots_gt = []
#     dihedral_pots_ff = []
#     for i in range(n_dihedrals):
#         dihedral_pot_tmp = []
#         if mol_info[molkey]["dihedral_energy"][i] is not None:
#             x_angle = mol_info[molkey]["dihedral_angle"][i]
#             dihedral_qm = (mol_info[molkey]["dihedral_energy"][i] - mol_info[molkey]["dihedral_energy"][i].min()) / (units.kJ * (units.mol**-1))
#             dihedral_ff = (mol_info[molkey]["dihedral_ffenergy"][i] - mol_info[molkey]["dihedral_ffenergy"][i].min())

#             plt.scatter(x_angle, dihedral_qm)
#             plt.plot(x_angle, dihedral_ff,label=f"dihedral_{i}")
#             dihedral_pots_gt.append(dihedral_qm)
#             dihedral_pots_ff.append(dihedral_ff)
#         else:
#             dihedral_pots_gt.append(None)
#             dihedral_pots_ff.append(None)
            
#     plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncol=2)
#     plt.xlabel("Dihedral angle (deg)")
#     plt.ylabel("Potential energy (kJ/mol)")
#     plt.xticks(range(-180, 181, 60))
#     plt.grid()
#     return dihedral_pots_gt, dihedral_pots_ff

# def draw_dihedral_structures(mol_info, molkey):
#     # key_name = "MOL_0"
#     from IPython.display import SVG
#     from rdkit.Chem.Draw import rdMolDraw2D
#     tm_list = []
#     highlighAtoms_list = []
#     legends_list = []
#     n_dihedrals = len(mol_info[molkey]["rotatable_dihedral"])
#     for i_dihed in range(n_dihedrals):
#         highlighAtoms_list.append(mol_info[molkey]["rotatable_dihedral"][i_dihed])
#         tm = rdMolDraw2D.PrepareMolForDrawing(mol_info[molkey]["rdkitmol2d"])
#         tm_list.append(mol_info[molkey]["rdkitmol2d"])
#         legends_list.append(f"dihedral_{i_dihed}")

#     dec = [800 // n_dihedrals + (1 if i < 800 % n_dihedrals else 0) for i in range(n_dihedrals)]
#     # decの先頭に800を追加
#     dec.insert(0, 800)
#     l = [dec]
#     view = rdMolDraw2D.MolDraw2DSVG(*l[0])

#     view.DrawMolecules(tm_list, highlightAtoms=highlighAtoms_list, legends=legends_list)
#     view.FinishDrawing()

#     svg = view.GetDrawingText()
#     return SVG(svg)
