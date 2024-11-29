from rdkit import Chem
from rdkit.Chem import rdDetermineBonds, rdDepictor
from ase.io import read, write
import os
import shutil
import copy
import pickle
import subprocess
import numpy as np
from openff.toolkit.topology import Molecule
from openff.toolkit import Quantity
from openff import toolkit
from openmm.app import *
from openmm import *
from openmm.unit import kelvin, picosecond, picoseconds
import cclib

class Mol_Info(object):
    def __init__(self):
        self.mol_info = {}
    
    def append_fromdir(self, directory, Nmols: int = 1):
        """
        Append directory to mol_info dictionary
        directory: directory path
        Nmols: Number of molecules in the directory
        """
        key = os.path.basename(directory)
        if key in self.mol_info.keys():
            assert False, f"Key: {key} already exists"
        
        if os.path.abspath(directory) != os.path.abspath(key): 
            directory = shutil.copytree(directory, f"{key}", dirs_exist_ok=True)
            directory = os.path.basename(directory)
        else:
            directory = key

        self.mol_info[key] = {}
        self.mol_info[key]["metadata"] = {}
        self.mol_info[key]["directory"] = directory
        self.mol_info[key]["Nmols"] = Nmols

        # *pklファイルがあれば読み込む
        if os.path.exists(f"{directory}/mol_info.pkl"):
            print("load pickle file")
            with open(f"{directory}/mol_info.pkl", mode='rb') as f:
                mol_info = pickle.load(f)
                for infokey, value in mol_info.items():
                    self.mol_info[key][infokey] = value
        
    def append_fromAtomsList(self, atomslist: list, key: str, Nmols: int = None):
        """
        Append ase.Atoms to mol_info dictionary
        atomslist: List of ase.Atoms
        key: key of mol_info dictionary
        """
        self.mol_info[key] = {}
        self.mol_info[key]["metadata"] = {}
        self.mol_info[key]["aseatoms_list"] = atomslist
        self.mol_info[key]["Natoms"] = len(atomslist[0])
        if Nmols is None:
            self.mol_info[key]["Nmols"] = len(atomslist)
        else:
            self.mol_info[key]["Nmols"] = Nmols
        
        os.makedirs(key, exist_ok=True)
        self.mol_info[key]["directory"] = key
        write(f"{key}/{key}.xyz", atomslist[0], format="xyz")
    
    def auto_charge_assign(self):
        num_charge_none = 0
        total_charge = 0
        for key in self.mol_info.keys():
            totalnum_elec = self.mol_info[key]["aseatoms_list"][0].get_atomic_numbers().sum()
            openshell_flag = totalnum_elec % 2 == 1
            self.mol_info[key]["symbol"] = str(self.mol_info[key]["aseatoms_list"][0].symbols)
            if self.mol_info[key]["symbol"] in ["Li", "Na", "K", "Rb", "Cs", "Mg"]:
                self.mol_info[key]["charge"] = 1
            elif self.mol_info[key]["symbol"] in ["F", "Cl", "Br", "I"]:
                self.mol_info[key]["charge"] = -1
            elif self.mol_info[key]["symbol"] in ["Mg", "Ca", "Sr", "Ba"]:
                self.mol_info[key]["charge"] = 2
            elif openshell_flag:
                self.mol_info[key]["charge"] = None
            else:
                self.mol_info[key]["charge"] = 0

        for key in self.mol_info.keys():
            if self.mol_info[key]["charge"] is None:
                num_charge_none += 1 * self.mol_info[key]["Nmols"]
            else:
                total_charge += self.mol_info[key]["charge"] * self.mol_info[key]["Nmols"]

        if num_charge_none > 0 and total_charge % num_charge_none == 0:
            charge_per_none = int(-total_charge / num_charge_none)
            for key in self.mol_info.keys():
                if self.mol_info[key]["charge"] is None:
                    self.mol_info[key]["charge"] = charge_per_none

    def get_rdkitmol(self):
        for key in self.mol_info.keys():
            natoms = self.mol_info[key]["Natoms"]
            directory = self.mol_info[key]["directory"]

            if natoms > 1:
                mol = Chem.MolFromXYZFile(f"{directory}/{key}.xyz")
                mol = Chem.Mol(mol)
                charge = self.mol_info[key]["charge"]
                rdDetermineBonds.DetermineBonds(mol,charge=charge)
                mol2d = copy.deepcopy(mol)
                rdDepictor.Compute2DCoords(mol2d)
                self.mol_info[key]["rdkitmol"] = mol
                self.mol_info[key]["rdkitmol2d"] = mol2d
            else:
                symbol = self.mol_info[key]["symbol"]
                charge = self.mol_info[key]["charge"]
                if charge > 0:
                    mol = Chem.MolFromSmiles(f"[{symbol}+{abs(charge)}]")
                elif charge < 0:
                    mol = Chem.MolFromSmiles(f"[{symbol}-{abs(charge)}]")
                else:
                    mol = Chem.MolFromSmiles(f"[{symbol}]")
                self.mol_info[key]["rdkitmol"] = mol
                self.mol_info[key]["rdkitmol2d"] = mol


    def asemol2molinfo(self, atomslist_mols, molecule_list):
        """
        Convert List of ase.Atoms to mol_info dictionary
        atomslist_mols: List of ase.Atoms, index of list is molecule index
        molecule_list: List of molecule index
        """
        mol_info = {}
        for i, mol in enumerate(molecule_list):
            totalnum_elec = atomslist_mols[mol[0]].get_atomic_numbers().sum()
            openshell_flag = totalnum_elec % 2 == 1
            mol_info[f"MOL_{i}"] = {}
            mol_info[f"MOL_{i}"]["metadata"] = {}
            mol_info[f"MOL_{i}"]["molecule_list"] = mol
            mol_info[f"MOL_{i}"]["symbol"] = str(atomslist_mols[mol[0]].symbols)
            mol_info[f"MOL_{i}"]["aseatoms_list"] = [atomslist_mols[i] for i in mol]
            if mol_info[f"MOL_{i}"]["symbol"] in ["Li", "Na", "K", "Rb", "Cs", "Mg"]:
                mol_info[f"MOL_{i}"]["charge"] = 1
            elif mol_info[f"MOL_{i}"]["symbol"] in ["F", "Cl", "Br", "I"]:
                mol_info[f"MOL_{i}"]["charge"] = -1
            elif mol_info[f"MOL_{i}"]["symbol"] in ["Mg", "Ca", "Sr", "Ba"]:
                mol_info[f"MOL_{i}"]["charge"] = 2
            elif openshell_flag:
                mol_info[f"MOL_{i}"]["charge"] = None
            else:
                mol_info[f"MOL_{i}"]["charge"] = 0

        ####### auto charge assign on ion species #######
        num_charge_none = 0
        total_charge = 0
        for key, value in mol_info.items():
            if value["charge"] is None:
                num_charge_none += 1 * len(value["molecule_list"])
            else:
                total_charge += value["charge"] * len(value["molecule_list"])

        if num_charge_none > 0 and total_charge % num_charge_none == 0:
            charge_per_none = int(-total_charge / num_charge_none)
            for key in mol_info.keys():
                if mol_info[key]["charge"] is None:
                    mol_info[key]["charge"] = charge_per_none
        ####### auto charge assign on ion species #######

        for key in mol_info.keys():
            os.makedirs(key, exist_ok=True)
            mol_info[key]["directory"] = key
            write(f"{key}/{key}.xyz", atomslist_mols[mol_info[key]["molecule_list"][0]], format="xyz")
            natoms = len(atomslist_mols[mol_info[key]["molecule_list"][0]])

            if natoms > 1:
                mol = Chem.MolFromXYZFile(f"{key}/{key}.xyz")
                mol = Chem.Mol(mol)
                charge = mol_info[key]["charge"]
                rdDetermineBonds.DetermineBonds(mol,charge=charge)    

                mol2d = copy.deepcopy(mol)
                rdDepictor.Compute2DCoords(mol2d)

                mol_info[key]["rdkitmol"] = mol
                mol_info[key]["rdkitmol2d"] = mol2d
            
            else:
                symbol = mol_info[key]["symbol"]
                charge = mol_info[key]["charge"]
                if charge > 0:
                    mol = Chem.MolFromSmiles(f"[{symbol}+{abs(charge)}]")
                elif charge < 0:
                    mol = Chem.MolFromSmiles(f"[{symbol}-{abs(charge)}]")
                else:
                    mol = Chem.MolFromSmiles(f"[{symbol}]")
                mol_info[key]["rdkitmol"] = mol
                mol_info[key]["rdkitmol2d"] = mol

        return mol_info

    def get_smiles_from_molinfo(self):
        for key in self.mol_info.keys():
            mol = self.mol_info[key]["rdkitmol"]
            self.mol_info[key]["smiles"] = Chem.MolToSmiles(mol)

    def get_sdf_from_molinfo(self):
        for key in self.mol_info.keys():
            mol = self.mol_info[key]["rdkitmol"]
            output_dir = f"{key}"
            self.mol_info[key]["directory"] = output_dir
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_file = os.path.join(output_dir, f"{key}.sdf")
            writer = Chem.SDWriter(output_file)
            writer.write(mol)
            writer.close()

    def load_geoopt_results(self, g16logfiles: list = None, key: str = None):
        if g16logfiles is None and key is None:
            for key in self.mol_info.keys():
                self.mol_info[key]["geoopt_energy"] = []
                self.mol_info[key]["geoopt_done"] = []
                for g16 in self.mol_info[key]["g16opt"]:
                    try:
                        results = g16.read_results()
                        energy = g16.get_potential_energy()
                        self.mol_info[key]["geoopt_done"].append(True)
                        self.mol_info[key]["g16optlog"].append(g16.label+".log")
                    except:
                        energy = None
                        self.mol_info[key]["geoopt_done"].append(False)
                        self.mol_info[key]["g16optlog"].append(None)
                        print(f"Warning: No results found in {g16.label}")

                    self.mol_info[key]["geoopt_energy"].append(energy)

        elif g16logfiles is not None and key is not None:
            self.mol_info[key]["geoopt_energy"] = []
            self.mol_info[key]["geoopt_done"] = []
            self.mol_info[key]["g16optlog"] = []
            if type(g16logfiles) == list:
                for i, g16log in enumerate(g16logfiles):
                    try:
                        g16 = read(g16log)
                        energy = g16.get_potential_energy()
                        self.mol_info[key]["geoopt_done"].append(True)
                        self.mol_info[key]["g16optlog"].append(g16log)
                    except:
                        energy = None
                        self.mol_info[key]["geoopt_done"].append(False)
                        print(f"Warning: No results found in {g16log}")
                    self.mol_info[key]["geoopt_energy"].append(energy)
            else:
                print("Error: g16logfiles should be list")

    
    def load_dihedral_results(self):
        import os
        for key in self.mol_info.keys():
            self.mol_info[key]["dihedral_energy"] = []
            self.mol_info[key]["dihedral_angle"] = []
            self.mol_info[key]["dihedral_done"] = []
            self.mol_info[key]["dihedral_aseatoms"] = []
            if "g16dihedral" not in  self.mol_info[key].keys():
                continue
            else:
                self.mol_info[key]["dihedral_done"]     = [False for _ in self.mol_info[key]["g16dihedral"]]
                self.mol_info[key]["dihedral_energy"]   = [None  for _ in self.mol_info[key]["g16dihedral"]]
                self.mol_info[key]["dihedral_angle"]    = [None  for _ in self.mol_info[key]["g16dihedral"]]
                self.mol_info[key]["dihedral_aseatoms"] = [None  for _ in self.mol_info[key]["g16dihedral"]]
                for i, g16 in enumerate(self.mol_info[key]["g16dihedral"]):
                    try:
                        dihed_logfile =  g16.label + ".log"
                        dihed_logfile = os.path.abspath(dihed_logfile)
                        print(dihed_logfile)
                        dihed_cclib = cclib.io.ccread(dihed_logfile)
                        energy = dihed_cclib.scanenergies
                        angle = dihed_cclib.scanparm[0]
                        aseatoms = []
                        ase_g16log = read(dihed_logfile)
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

                        self.mol_info[key]["dihedral_done"][i] = True
                    except:
                        import traceback
                        traceback.print_exc()
                        energy = None
                        angle = None
                        aseatoms = None
                        print(f"Warning: Failed reading results: {g16.label}")
                    
                    self.mol_info[key]["dihedral_angle"][i] = angle
                    self.mol_info[key]["dihedral_energy"][i] = energy
                    self.mol_info[key]["dihedral_aseatoms"][i] = aseatoms
                    if aseatoms is not None:
                        write(f"scan_{i}.xyz", aseatoms)
                
    def get_dihedral_ff(self, forcefield=None):
        mmm = self.get_molecule_omm()
        for mol in self.mol_info.keys():
            for m in mmm:
                if m.name == mol:
                    bonds_info = m.bonds
            
            self.mol_info[mol]["dihedral_ffenergy"] = [ None for _ in self.mol_info[mol]["dihedral_done"]]

            for i_dihed in range(len(self.mol_info[mol]["dihedral_aseatoms"])):
                if self.mol_info[mol]["dihedral_done"][i_dihed] == True:
                    self.mol_info[mol]["dihedral_ffenergy"][i_dihed] = np.array([])
                    for i in range(len(self.mol_info[mol]["dihedral_aseatoms"][i_dihed])):
                        pdb_ase = self.mol_info[mol]["dihedral_aseatoms"][i_dihed][i].copy()
                        pdb_ase.arrays["atomtypes"] = [i for i in range(len(pdb_ase))]
                        write(f"temp_dihed_{i}.pdb", pdb_ase)
                        pdb_ase_cell = pdb_ase.copy()
                        pdb_ase_cell.cell = [1000,1000,1000] ; pdb_ase_cell.pbc = True
                        write(f"temp_dihed_{i}_cell.pdb", pdb_ase_cell)
                        pdb_omm = PDBFile(f"temp_dihed_{i}.pdb")
                        atomlist_openmm = [a for a in pdb_omm.topology.atoms()]
                        for b in bonds_info:
                            a1 = atomlist_openmm[b.atom1_index]
                            a2 = atomlist_openmm[b.atom2_index]
                            pdb_omm.topology.addBond(a1, a2)
                        
                        pdb_omm_cell = PDBFile(f"temp_dihed_{i}_cell.pdb")
                        atomlist_openmm_cell = [a for a in pdb_omm_cell.topology.atoms()]
                        for b in bonds_info:
                            a1 = atomlist_openmm_cell[b.atom1_index]
                            a2 = atomlist_openmm_cell[b.atom2_index]
                            pdb_omm_cell.topology.addBond(a1, a2)
                    
                        # gaff = template_generator(molecules=mmm, forcefield=params["ff_params"]["fftype"], il_assign=params["fsa_assign"])
                        # forcefield = ForceField(params["ff_params"]["iontype"])
                        # forcefield.registerTemplateGenerator(gaff.generator)
                        system = forcefield.createSystem(pdb_omm.topology, nonbondedMethod=NoCutoff)
                        for j, f in enumerate(system.getForces()):
                            f.setForceGroup(j)
                            integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.004*picoseconds)
                            simulation = Simulation(pdb_omm.topology, system, integrator)
                            simulation.context.setPositions(pdb_omm.positions)

                        potential_energies = []
                        for gi, f in enumerate(system.getForces()):
                            state = simulation.context.getState(getEnergy=True, groups={gi})
                            # print(f.getName(), state.getPotentialEnergy())
                            potential_energies.append(state.getPotentialEnergy().real)
                        
                        os.remove(f"temp_dihed_{i}.pdb")
                        os.remove(f"temp_dihed_{i}_cell.pdb")
                        self.mol_info[mol]["dihedral_ffenergy"][i_dihed] = np.append(self.mol_info[mol]["dihedral_ffenergy"][i_dihed], sum(potential_energies))


    def get_charges_from_molinfo(self):
        print(self.mol_info.keys())
        for key in self.mol_info.keys():
            print(key)
            natoms = len(self.mol_info[key]["aseatoms_list"][0])
            if natoms == 1:
                self.mol_info[key]["charges"] = [self.mol_info[key]["charge"]]
                output_mol2 = os.path.join(key, "single_charge.mol2")
                print(output_mol2)
                mol2_dict = {"@<TRIPOS>MOLECULE":[['MOL'], ['1', '0', '1', '0', '0'], ['SMALL'], ['single'],[],[]],\
                             '@<TRIPOS>ATOM': [['1', 'Li1', '0.0000', '0.0000', '0.0000', 'Li', '1', 'MOL', self.mol_info[key]["charge"]]],\
                             '@<TRIPOS>BOND': [],\
                             '@<TRIPOS>SUBSTRUCTURE': [['1', 'MOL', '1', 'TEMP', '0', '****', '****', '0', 'ROOT']]}
                write_mol2(output_mol2, mol2_dict)

            elif "g16charge" in self.mol_info[key] and \
                '6/33=2' in self.mol_info[key]["g16charge"].parameters["ioplist"] and \
                '6/42=6' in self.mol_info[key]["g16charge"].parameters["ioplist"]:
                chgmethod = "resp"
                g16chglog = self.mol_info[key]["g16charge"].label + ".log"
                dirname = os.path.dirname(g16chglog)
                output_mol2 =  os.path.join(dirname, "resp_charge.mol2")
                nc = self.mol_info[key]["charge"]
                cmd_antech = (
                    f"antechamber -i {g16chglog} -fi gout "
                    f"-o {output_mol2} -fo mol2 -at sybyl -c {chgmethod} -nc {nc} -pf y -dr no")
                print(cmd_antech)
                output = subprocess.getoutput(cmd_antech)
                print(output)
                mol2_dict = read_mol2(output_mol2)
            else:
                chgmethod = "bcc"
                if "getopt_done" in self.mol_info[key] and True in self.mol_info[key]["geoopt_done"]:
                    energy_list = self.mol_info[key]["geoopt_energy"]
                    energy_list = [1e10 if x is None else x for x in energy_list]
                    min_idx = self.mol_info[key]["geoopt_energy"].index(min(energy_list))
                    g16chglog = self.mol_info[key]["g16opt"][min_idx].label + ".log"
                    dirname = os.path.dirname(g16chglog)
                    output_mol2 =  os.path.join(dirname, "bcc_charge.mol2")
                    nc = self.mol_info[key]["charge"]
                    print(output_mol2)
                    cmd_antech = (
                        f"antechamber -i {g16chglog} -fi gout "
                        f"-o {output_mol2} -fo mol2 -at sybyl -c {chgmethod} -nc {nc} -pf y -dr no")
                    output = subprocess.getoutput(cmd_antech)
                    mol2_dict = read_mol2(output_mol2)
                else:
                    write(f"{key}/{key}_bcc.pdb", self.mol_info[key]["aseatoms_list"][0])
                    g16chglog = f"{key}/{key}_bcc.pdb"
                    output_mol2 =  f"{key}/bcc_charge.mol2"
                    nc = self.mol_info[key]["charge"]
                    cmd_antech = (
                        f"antechamber -i {g16chglog} -fi pdb "
                        f"-o {output_mol2} -fo mol2 -at sybyl -c {chgmethod} -nc {nc} -pf y -dr no")
                    output = subprocess.getoutput(cmd_antech)
                    mol2_dict = read_mol2(output_mol2)

            charges = [d[-1] for d in  mol2_dict["@<TRIPOS>ATOM"]]
            self.mol_info[key]["charges"] = charges
            self.mol_info[key]["mol2file"] = os.path.basename(output_mol2)

    # def get_molecules_omm(self):
    #     for mol_i in self.mol_info.keys():
    #         for key in self.mol_info[mol_i]["metadata"].keys():
    #             index_list = self.mol_info[mol_i]["metadata"][key]
    #             for i in index_list:
    #                 self.mol_info[mol_i]["molecules_omm"].append(self.mol_info[mol_i]["molecules"][i])
    #     return self.mol_info

    def adjust_charges(self, ff_params=None):
        for key in self.mol_info.keys():
            net_charge = self.mol_info[key]["charge"]
            charges = np.array([ np.float64(ee) for ee in self.mol_info[key]["charges"]])

            if ff_params is not None:
                if "charge_scale_ion" in ff_params.keys():
                    charge_scale_ion = ff_params["charge_scale_ion"]
                    if not np.isclose(net_charge, 0.0):
                        charges *= charge_scale_ion
                        net_charge *= charge_scale_ion
                elif "charge_scale_neutral" in ff_params.keys():
                    charge_scale_neutral = ff_params["charge_scale_neutral"]
                    if np.isclose(net_charge, 0.0):
                        charges *= charge_scale_neutral
                        net_charge *= charge_scale_neutral

            total_charge = np.sum(charges)

            charge_deficit = total_charge - net_charge
            print(charge_deficit)

            if not np.isclose(charge_deficit, 0.0):
                print(f"Net charge of {key} is {net_charge} and total charge is {total_charge}")
                charges = charges - charge_deficit / len(charges)
                total_charge = np.sum(charges)
                charge_deficit = total_charge - net_charge
                charges[0] -= charge_deficit
            
            self.mol_info[key]["charges"] = charges


    def get_molecule_omm(self):
        molecules_omm = []
        for mol in self.mol_info.keys():
            dirname = self.mol_info[mol]["directory"]
            sdffile = os.path.join(dirname, f"{mol}.sdf")

            if not os.path.exists(sdffile):
                self.mol_info = self.get_sdf_from_molinfo()
            
            molecule_mm = Molecule.from_file(sdffile)
            molecule_mm.partial_charges = Quantity(self.mol_info[mol]["charges"], toolkit.unit.elementary_charge)
            molecule_mm.name = mol
            # molecule_mm.total_charge = Quantity(self.mol_info[mol]["charge"], unit.elementary_charge)  ## why ??
            for meta_key in self.mol_info[mol]["metadata"].keys():
                for meta_ind in self.mol_info[mol]["metadata"][meta_key]:
                    molecule_mm.atoms[meta_ind].metadata[meta_key] = True
            if "mol2file" in self.mol_info[mol]:
                molecule_mm.mol2file = os.path.join(self.mol_info[mol]["directory"], self.mol_info[mol]["mol2file"])
            molecules_omm.append(molecule_mm)
        return molecules_omm
    
    def get_rotatable_dihedral(self):
        for m in self.mol_info.keys():
            if "rdkitmol" in self.mol_info[m].keys():
                mol = self.mol_info[m]["rdkitmol"]
                self.mol_info[m]["rotatable_dihedral"], self.mol_info[m]["rotatable_dihedral_elem"] = _get_rotatable_dihedral(mol)
            else:
                print(f"Warning: No rdkitmol found in {m}")
        return self.mol_info
        
    def do_geoopt(self, force=False):
        import shutil
        if shutil.which("g16") is None:
            print("Error: g16 is not found in PATH")
            return
        for m in self.mol_info.keys():
            if "g16opt" in self.mol_info[m].keys():
                for i in range(len(self.mol_info[m]['g16opt'])):
                    g16_infile = self.mol_info[m]['g16opt'][i].label + ".com"
                    g16_logfile = self.mol_info[m]['g16opt'][i].label + ".log"
                    if os.path.exists(g16_logfile) and not force:
                        print(f"Skip {g16_logfile}")
                        continue
                    elif not os.path.exists(g16_logfile) or force:
                        cmd = f"g16 < {g16_infile}  > {g16_logfile}"
                        output = subprocess.getoutput(cmd)
                        print(f"g16opt -- {m}_{i}")
                        print(cmd)
                        print(output)
    
    def do_charge(self, force=False):
        import shutil
        if shutil.which("g16") is None:
            print("Error: g16 is not found in PATH")
            return
        for m in self.mol_info.keys():
            if "g16charge" in self.mol_info[m].keys():
                g16_infile = self.mol_info[m]['g16charge'].label + ".com"
                g16_logfile = self.mol_info[m]['g16charge'].label + ".log"
                if os.path.exists(g16_logfile) and not force:
                    print(f"Skip {g16_logfile}")
                    continue
                elif not os.path.exists(g16_logfile) or force:
                    cmd = f"g16 < {g16_infile}  > {g16_logfile}"
                    output = subprocess.getoutput(cmd)
                    print(f"g16charge -- {m}")
                    print(cmd)
                    print(output)
    
    def do_dihedral(self, force=False):
        import shutil
        if shutil.which("g16") is None:
            print("Error: g16 is not found in PATH")
            return
        for m in self.mol_info.keys():
            if "g16dihedral" in self.mol_info[m].keys():
                for i in range(len(self.mol_info[m]['g16dihedral'])):
                    g16_infile = self.mol_info[m]['g16dihedral'][i].label + ".com"
                    g16_logfile = self.mol_info[m]['g16dihedral'][i].label + ".log"
                    if os.path.exists(g16_logfile):
                        print(f"Skip {g16_logfile}")
                        continue
                    elif not os.path.exists(g16_logfile) or force:
                        cmd = f"g16 < {g16_infile}  > {g16_logfile}"
                        output = subprocess.getoutput(cmd)
                        print(f"g16dihedral -- {m}")
                        print(cmd)
                        print(output)

    def save_molinfo(self, filename="mol_info.pkl"):
        with open(filename, mode='wb') as f:
            pickle.dump(self,f)
        
        for key in self.mol_info.keys():
            directory = self.mol_info[key]["directory"]
            with open(f"{directory}/{filename}", mode='wb') as f:
                mol_info = copy.deepcopy(self.mol_info[key])
                mol_info.pop("directory")
                pickle.dump(mol_info, f)

    def il_assign(self):
        for key in self.mol_info.keys():
            mol = self.mol_info[key]["rdkitmol"]
            mol2d = self.mol_info[key]["rdkitmol2d"]
            charge = self.mol_info[key]["charge"]
            fsalike_Nindex = []
            fsalike_Sindex = []
            if charge < 0:
                atoms = mol.GetAtoms()
                atoms2d = mol2d.GetAtoms()

                # Check FSA-like N and S atoms
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

                self.mol_info[key]["metadata"]["FSA_N"] = fsalike_Nindex
                self.mol_info[key]["metadata"]["FSA_S"] = fsalike_Sindex


def read_mol2(filename):
    from collections import OrderedDict
    with open(filename) as f:
        l = f.readlines()
        mol2_dict = OrderedDict()
        for i in range(len(l)):
            if l[i].startswith("@<TRIPOS>"):
                key_name = l[i].strip()
                mol2_dict[key_name] = []
            elif l[i].strip() != "":
                mol2_dict[key_name].append(l[i].strip().split())
    return mol2_dict


def load_molinfo(filename):
    with open(filename, mode='rb') as f:
        mol_info = pickle.load(f)
        return mol_info

def write_mol2(filename, mol2_dict):
    with open(filename, mode="w") as f:
        for key in mol2_dict:
            f.write(key + "\n")
            for i in range(len(mol2_dict[key])):
                outline = [ str(x) for x in mol2_dict[key][i] ]
                f.write(" ".join(outline) + "\n")

def get_idmol(rdmol):
    import copy
    id_mol = copy.deepcopy(rdmol)

    for atom in id_mol.GetAtoms():
        atom.SetProp("atomLabel", str(atom.GetIdx()))
    return id_mol

def _get_rotatable_dihedral(rdmol):
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

def draw_dihedral_plots(mol_info, molkey):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(4, 3))
    n_dihedrals = len(mol_info.mol_info[molkey]["dihedral_angle"])
    for i in range(n_dihedrals):
        if mol_info.mol_info[molkey]["dihedral_energy"][i] is not None:
            plt.scatter(mol_info.mol_info[molkey]["dihedral_angle"][i], (mol_info.mol_info[molkey]["dihedral_energy"][i] - mol_info.mol_info[molkey]["dihedral_energy"][i].min()) / (units.kJ * (units.mol**-1)))
            plt.plot(mol_info.mol_info[molkey]["dihedral_angle"][i], (mol_info.mol_info[molkey]["dihedral_ffenergy"][i] - mol_info.mol_info[molkey]["dihedral_ffenergy"][i].min()),label=f"dihedral_{i}")
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncol=2)
    plt.xlabel("Dihedral angle (deg)")
    plt.ylabel("Potential energy (kJ/mol)")
    plt.xticks(range(-180, 181, 60))
    plt.grid()

def draw_dihedral_structures(mol_info, molkey):
    # key_name = "MOL_0"
    from IPython.display import SVG
    from rdkit.Chem.Draw import rdMolDraw2D
    tm_list = []
    highlighAtoms_list = []
    legends_list = []
    n_dihedrals = len(mol_info.mol_info[molkey]["rotatable_dihedral"])
    for i_dihed in range(n_dihedrals):
        highlighAtoms_list.append(mol_info.mol_info[molkey]["rotatable_dihedral"][i_dihed])
        tm = rdMolDraw2D.PrepareMolForDrawing(mol_info.mol_info[molkey]["rdkitmol2d"])
        tm_list.append(mol_info.mol_info[molkey]["rdkitmol2d"])
        legends_list.append(f"dihedral_{i_dihed}")

    dec = [800 // n_dihedrals + (1 if i < 800 % n_dihedrals else 0) for i in range(n_dihedrals)]
    # decの先頭に800を追加
    dec.insert(0, 800)
    l = [dec]
    view = rdMolDraw2D.MolDraw2DSVG(*l[0])

    view.DrawMolecules(tm_list, highlightAtoms=highlighAtoms_list, legends=legends_list)
    view.FinishDrawing()

    svg = view.GetDrawingText()
    return SVG(svg)

