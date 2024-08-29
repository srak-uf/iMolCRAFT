from rdkit import Chem
from rdkit.Chem import rdDetermineBonds, rdDepictor
from ase.io import read, write
import os
import copy
import pickle
import subprocess
from openff.toolkit.topology import Molecule
from openff.toolkit import Quantity, unit

class Mol_Info(object):
    def __init__(self, atomslist_mols=None, molecule_list=None):
        self.asemols = atomslist_mols
        self.molecule_list = molecule_list
        if atomslist_mols is not None or molecule_list is not None:
            self.mol_info = self.asemol2molinfo(atomslist_mols, molecule_list)

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

        if total_charge % num_charge_none == 0:
            charge_per_none = int(-total_charge / num_charge_none)
            for key in mol_info.keys():
                if mol_info[key]["charge"] is None:
                    mol_info[key]["charge"] = charge_per_none
        ####### auto charge assign on ion species #######

        for key in mol_info.keys():
            os.makedirs(key, exist_ok=True)
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

    def load_geoopt_results(self):
        for key in self.mol_info.keys():
            self.mol_info[key]["geoopt_energy"] = []
            self.mol_info[key]["geoopt_done"] = []
            for g16 in self.mol_info[key]["g16opt"]:
                try:
                    results = g16.read_results()
                    energy = g16.get_potential_energy()
                    self.mol_info[key]["geoopt_done"].append(True)
                except:
                    energy = None
                    self.mol_info[key]["geoopt_done"].append(False)
                    print(f"Warning: No results found in {g16.label}")
                self.mol_info[key]["geoopt_energy"].append(energy)

    def get_charges_from_molinfo(self):
        for key in self.mol_info.keys():
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

            elif  "g16charge" in self.mol_info[key] and \
                '6/33=2' in self.mol_info[key]["g16charge"].parameters["ioplist"] and \
                '6/42=6' in self.mol_info[key]["g16charge"].parameters["ioplist"]:
                chgmethod = "resp"
                g16chglog = self.mol_info[key]["g16charge"].label + ".log"
                dirname = os.path.dirname(g16chglog)
                output_mol2 =  os.path.join(dirname, "resp_charge.mol2")
                nc = self.mol_info[key]["charge"]
                cmd_antech = (
                    f"antechamber -i {g16chglog} -fi gout "
                    f"-o {output_mol2} -fo mol2 -at sybyl -c {chgmethod} -nc {nc} -pf y")
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
                        f"-o {output_mol2} -fo mol2 -at sybyl -c {chgmethod} -nc {nc} -pf y")
                    output = subprocess.getoutput(cmd_antech)
                    mol2_dict = read_mol2(output_mol2)
                else:
                    write(f"{key}/{key}_bcc.pdb", self.mol_info[key]["aseatoms_list"][0])
                    g16chglog = f"{key}/{key}_bcc.pdb"
                    output_mol2 =  f"{key}/bcc_charge.mol2"
                    nc = self.mol_info[key]["charge"]
                    cmd_antech = (
                        f"antechamber -i {g16chglog} -fi pdb "
                        f"-o {output_mol2} -fo mol2 -at sybyl -c {chgmethod} -nc {nc} -pf y")
                    output = subprocess.getoutput(cmd_antech)
                    mol2_dict = read_mol2(output_mol2)

            charges = [d[-1] for d in  mol2_dict["@<TRIPOS>ATOM"]]
            self.mol_info[key]["charges"] = charges
            self.mol_info[key]["mol2file"] = output_mol2

    # def get_molecules_omm(self):
    #     for mol_i in self.mol_info.keys():
    #         for key in self.mol_info[mol_i]["metadata"].keys():
    #             index_list = self.mol_info[mol_i]["metadata"][key]
    #             for i in index_list:
    #                 self.mol_info[mol_i]["molecules_omm"].append(self.mol_info[mol_i]["molecules"][i])
    #     return self.mol_info
    
    def get_molecule_omm(self):
        molecules_omm = []
        for mol in self.mol_info.keys():
            dirname = self.mol_info[mol]["directory"]
            sdffile = os.path.join(dirname, f"{mol}.sdf")

            if not os.path.exists(sdffile):
                self.mol_info = self.get_sdf_from_molinfo()
            
            molecule_mm = Molecule.from_file(sdffile)
            molecule_mm.partial_charges = Quantity(self.mol_info[mol]["charges"], unit.elementary_charge)
            for meta_key in self.mol_info[mol]["metadata"].keys():
                for meta_ind in self.mol_info[mol]["metadata"][meta_key]:
                    molecule_mm.atoms[meta_ind].metadata[meta_key] = True
            if "mol2file" in self.mol_info[mol]:
                molecule_mm.mol2file = self.mol_info[mol]["mol2file"]
            molecules_omm.append(molecule_mm)
        return molecules_omm
    
    def get_rotatable_dihedral(self):
        for m in self.mol_info.keys():
            if "rdkitmol" in self.mol_info[m].keys():
                mol = self.mol_info[m]["rdkitmol"]
                self.mol_info[m]["rotatable_dihedral"] = _get_rotatable_dihedral(mol)
            else:
                print(f"Warning: No rdkitmol found in {m}")
        return self.mol_info
        
    def do_geoopt(self):
        import shutil
        if shutil.which("g16") is None:
            print("Error: g16 is not found in PATH")
            return
        for m in self.mol_info.keys():
            if "g16opt" in self.mol_info[m].keys():
                for i in range(len(self.mol_info[m]['g16opt'])):
                    g16_infile = self.mol_info[m]['g16opt'][i].label + ".com"
                    g16_logfile = self.mol_info[m]['g16opt'][i].label + ".log"
                    cmd = f"g16 < {g16_infile}  > {g16_logfile}"
                    output = subprocess.getoutput(cmd)
                    print(f"g16opt -- {m}_{i}")
                    print(cmd)
                    print(output)
    
    def do_charge(self):
        import shutil
        if shutil.which("g16") is None:
            print("Error: g16 is not found in PATH")
            return
        for m in self.mol_info.keys():
            if "g16charge" in self.mol_info[m].keys():
                g16_infile = self.mol_info[m]['g16charge'][0].label + ".com"
                g16_logfile = self.mol_info[m]['g16charge'][0].label + ".log"
                cmd = f"g16 < {g16_infile}  > {g16_logfile}"
                output = subprocess.getoutput(cmd)
                print(f"g16charge -- {m}")
                print(cmd)
                print(output)
    
    def do_dihedral(self):
        import shutil
        if shutil.which("g16") is None:
            print("Error: g16 is not found in PATH")
            return
        for m in self.mol_info.keys():
            if "g16dihedral" in self.mol_info[m].keys():
                for i in range(len(self.mol_info[m]['g16dihedral'])):
                    g16_infile = self.mol_info[m]['g16dihedral'][0].label + ".com"
                    g16_logfile = self.mol_info[m]['g16dihedral'][0].label + ".log"
                    cmd = f"g16 < {g16_infile}  > {g16_logfile}"
                    output = subprocess.getoutput(cmd)
                    print(f"g16dihedral -- {m}")
                    print(cmd)
                    print(output)

    def save_molinfo(self, filename="mol_info.pkl"):
        with open(filename, mode='wb') as f:
            pickle.dump(self,f)


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
        dihedral_list.append(dihedral)
        
    return dihedral_list

