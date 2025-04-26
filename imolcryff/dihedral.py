from rdkit import Chem
from openmm import PeriodicTorsionForce, LangevinMiddleIntegrator
from openmm.unit import kelvin, picosecond, picoseconds, degree, kilojoules_per_mole
from openmm.app import NoCutoff, Simulation, PDBFile, ForceField
from openmm.openmm import XmlSerializer
import numpy as np
from ase.io import read, write
from ase.calculators.gaussian import Gaussian
import copy, os, tempfile, subprocess
import cclib

class DihedCalculator:
    def __init__(self, atoms, rdkitmol, label, directory=None, qmparams=None):
        self.atoms = atoms.copy()
        self.atoms.pbc = False
        self.atoms.cell = None
        self.rdmol = rdkitmol
        self.nc = sum([atom.GetFormalCharge() for atom in self.rdmol.GetAtoms()])
        self.label = label
        if qmparams is None:
            self.qmparams = {
                            "method": "wb97xd",
                            "basis": "6-311++g(d,p)",
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
        self.qm_dihedscan = [{"angle": [], "energy": [], "atoms": []} for _ in range(len(self.dihedral_list))]
        self.ff_dihedscan = [{"angle": [], "energy": [], "atoms": []} for _ in range(len(self.dihedral_list))]

    def get_dihedral_qm(self, dihed_list=None, do_calc=True):
        if dihed_list is None:
            dihed_list = self.dihedral_list
        
        for di in range(len(dihed_list)):
            label = f"{self.label}_dihed_{di}" # f"{key}_dihed_{di}"
            d0 = self.dihedral_elem_list[di][0] + 1
            d1 = self.dihedral_elem_list[di][1] + 1
            d2 = self.dihedral_elem_list[di][2] + 1
            d3 = self.dihedral_elem_list[di][3] + 1
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

                self.qm_dihedscan[di]["angle"],  \
                self.qm_dihedscan[di]["energy"], \
                self.qm_dihedscan[di]["atoms"] = load_g16scan(logfile)
    
    def get_dihedral_ff(self, ffxml=None, top=None, dihed_list=None):
        if dihed_list is None:
            dihed_list = self.dihedral_list
        
        for di in range(len(dihed_list)):
            label = f"{self.label}_dihed_{di}"
            d0 = self.dihedral_elem_list[di][0] + 1
            d1 = self.dihedral_elem_list[di][1] + 1
            d2 = self.dihedral_elem_list[di][2] + 1
            d3 = self.dihedral_elem_list[di][3] + 1

            self.ff_dihedscan[di]["energy"] ,\
            self.ff_dihedscan[di]["atoms"] = scan_ff_dihedral(ffxml, \
                                                               top, \
                                                               self.qm_dihedscan[di]["angles"], \
                                                               self.qm_dihedscan[di]["atoms"], \
                                                               self.dihedral_list[di])
            self.ff_dihedscan[di]["angle"] = self.qm_dihedscan[di]["angle"]


def load_g16scan(g16logfile):
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
    except:
        import traceback
        traceback.print_exc()
        energy = None
        angle = None
        aseatoms = None
        print(f"Warning: Failed reading results: {g16logfile}")
    
    return angle, energy, aseatoms

def get_rotatable_dihedral(rdmol):
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

def scan_ff_dihedral(ffxml, top, angles, atoms_list, dihed_atidx):
    dihedral_ffenergy = []
    ff_dihedatoms = []
    with tempfile.TemporaryDirectory() as td:
        for i in range(len(atoms_list)):
            pdb_ase = atoms_list[i].copy()
            pdb_ase.arrays["atomtypes"] = [i for i in range(len(pdb_ase))]
            temppdb = os.path.join(td, f"temp_dihed_{i}.pdb")
            write(temppdb, pdb_ase)
            pdb_omm = PDBFile(temppdb)
            atomlist_openmm = [a for a in pdb_omm.topology.atoms()]
            for b in top.bonds():
                a1 = atomlist_openmm[b.atom1.index]
                a2 = atomlist_openmm[b.atom2.index]
                pdb_omm.topology.addBond(a1, a2)

            # relaxed scan
            forcefield = ForceField(ffxml)
            system = forcefield.createSystem(pdb_omm.topology, nonbondedMethod=NoCutoff)
            tempsysxml = os.path.join(td, "system.xml")
            with open(tempsysxml, 'w') as output:
                output.write(XmlSerializer.serialize(system))
                
            restraint = PeriodicTorsionForce()
            d1 = dihed_atidx[0]
            d2 = dihed_atidx[1]
            d3 = dihed_atidx[2]
            d4 = dihed_atidx[3]
            restraint.addTorsion(d1,d2,d3,d4, 1, (angles[i]+180)*degree, 10000*kilojoules_per_mole)
            system.addForce(restraint)
            integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.004*picoseconds)
            simulation = Simulation(pdb_omm.topology, system,integrator)
            simulation.context.setPositions(pdb_omm.positions)
            simulation.minimizeEnergy()
            state = simulation.context.getState(getPositions=True, getEnergy=True)
            with open(temppdb, 'w') as output:
                PDBFile.writeFile(simulation.topology, state.getPositions(), output)

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
            ff_dihedatoms.append(read(temppdb))

    ff_pot = np.array(dihedral_ffenergy)
    ff_pot_kjmol = (ff_pot - ff_pot.min())
    return ff_pot_kjmol, ff_dihedatoms

def parse_g16scan(file):
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
