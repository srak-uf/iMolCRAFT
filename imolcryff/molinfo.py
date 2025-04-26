from rdkit import Chem
from rdkit.Chem import rdDetermineBonds, rdDepictor
from ase.io import read, write
from ase import units
from ase import Atoms
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
from .asemol import asemol_wrapper, aseatoms2pdb, merge_asemols
from .dihedral import DihedCalculator
from .charge import ChargeCalculator

MOLINFO_KEYS = {
    "aseatoms_list": list,
    "aseatoms_geoopt": list,
    "aseatoms_stable": type(None),
    "ChargeCalc": type(None),
    "DihedCalc": type(None),
    "directory": str,
    "metadata": dict,
    "Molecule_OFF": type(None),
    "Natoms": type(None),
    "Nmols": type(None),
    "netcharge": type(None),
    "partial_charges": list,
    "rdkit": dict, 
    "SMILES": str, 
}

class Crafter:
    def __init__(self):
        self.mol_info = {}

    def _initialize_molinfo(self, key: str):
        """
        Initialize mol_info[key] with predefined keys.
        """
        self.mol_info[key] = {k: v() if callable(v) else v for k, v in MOLINFO_KEYS.items()}

    def from_crystal(self, filename: str, assign_totalcharge=True):
        """
        Read cif, xyz, POSCAR, ... file and append to mol_info dictionary
        filename: Structure file name (It should be ASE compatible)
        """
        atoms = read(filename)
        asemol_wrap = asemol_wrapper(atoms)
        self.atoms_unwrap = asemol_wrap.unwrap_molecules()
        self.molatoms, self.molecule_list = asemol_wrap.get_ase_molecules(ordered=True)

        for i, mol_idx in enumerate(self.molecule_list):
            al = [self.molatoms[i] for i in mol_idx]
            self.append_fromAtomsList(al, key=f"MOL_{i}", Nmols = len(mol_idx))
        
            if assign_totalcharge:
                self._assign_totalcharge()
            else:
                print(f"Warning: MOL_{i} has no charge information")
                print(f"Please define the total charge manually")

    def append_fromAtomsList(self, atomslist: list, key: str, Nmols: int = None):
        """
        Append ase.Atoms to mol_info dictionary
        atomslist: List of ase.Atoms
        key: key of mol_info dictionary
        """
        self._initialize_molinfo(key)
        self.mol_info[key]["aseatoms_list"] = atomslist
        self.mol_info[key]["Natoms"] = len(atomslist[0])
        self.mol_info[key]["Nmols"] = len(atomslist) if Nmols is None else Nmols     
        os.makedirs(key, exist_ok=True)
        self.mol_info[key]["directory"] = key
        write(f"{key}/{key}.xyz", atomslist[0], format="xyz")

    def get_rdkitmol(self, keys=None, il_assing=True):
        if keys is None:
            keys = self.mol_info.keys()

        for key in keys:
            natoms = self.mol_info[key]["Natoms"]
            directory = self.mol_info[key]["directory"]

            if natoms > 1:
                mol = Chem.MolFromXYZFile(f"{directory}/{key}.xyz")
                mol = Chem.Mol(mol)
                nc = self.mol_info[key]["netcharge"]
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
                self.mol_info[key]["rdkit"]["mol"] = mol
                self.mol_info[key]["rdkit"]["mol2d"] = mol2d
            else:
                symbol = self.mol_info[key]["symbol"]
                nc = self.mol_info[key]["netcharge"]
                if nc > 0:
                    mol = Chem.MolFromSmiles(f"[{symbol}+{abs(nc)}]")
                elif nc < 0:
                    mol = Chem.MolFromSmiles(f"[{symbol}-{abs(nc)}]")
                else:
                    mol = Chem.MolFromSmiles(f"[{symbol}]")
                self.mol_info[key]["rdkit"]["mol"] = mol
                self.mol_info[key]["rdkit"]["mol2d"] = mol

            nc = self.mol_info[key]["netcharge"]

            self._il_assign(mol, mol2d, int(nc))

    def get_smiles(self, keys=None):
        if keys is None:
            keys = self.mol_info.keys()
        for key in keys:
            if self.mol_info[key]["rdkit"] is None:
                self.get_rdkitmol(key)
            mol = self.mol_info[key]["rdkit"]["mol"]
            self.mol_info[key]["SMILES"] = Chem.MolToSmiles(mol)

    def get_sdf(self, keys=None):
        if keys is None:
            keys = self.mol_info.keys()

        for key in keys:
            if self.mol_info[key]["rdkit"] is None:
                self.get_rdkitmol(key)

            mol = self.mol_info[key]["rdkit"]["mol"]
            output_dir = f"{key}"
            self.mol_info[key]["directory"] = output_dir
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_file = os.path.join(output_dir, f"{key}.sdf")
            writer = Chem.SDWriter(output_file)
            writer.write(mol)
            writer.close()

    # def do_geoopt(self, force=False):
    #     import shutil
    #     if shutil.which("g16") is None:
    #         print("Error: g16 is not found in PATH")
    #         return
    #     for m in self.mol_info.keys():
    #         if "g16opt" in self.mol_info[m].keys():
    #             for i in range(len(self.mol_info[m]['g16opt'])):
    #                 g16_infile = self.mol_info[m]['g16opt'][i].label + ".com"
    #                 g16_logfile = self.mol_info[m]['g16opt'][i].label + ".log"
    #                 if os.path.exists(g16_logfile) and not force:
    #                     print(f"Skip {g16_logfile}")
    #                     continue
    #                 elif not os.path.exists(g16_logfile) or force:
    #                     cmd = f"g16 < {g16_infile}  > {g16_logfile}"
    #                     output = subprocess.getoutput(cmd)
    #                     print(f"g16opt -- {m}_{i}")
    #                     print(cmd)
    #                     print(output)
    # def molinfo_setg16opt(mol_info, params_opt=None):
    #     if params_opt is None:
    #         params_opt = default_params_opt
    #     for key in mol_info.keys():
    #         output_dir = key
    #         mol_info[key]["g16opt"] = []
    #         mol_info[key]["g16optlog"] = []
    #         for i in range(len(mol_info[key]["aseatoms_list"])):
    #             atoms = mol_info[key]["aseatoms_list"][i]
    #             charge = mol_info[key]["charge"]
    #             label = f"{key}_{i}"
    #             g16 = input_g16(atoms, params_opt, charge, output_dir, label)
    #             mol_info[key]["g16opt"].append(g16)

    def get_optstructure(self, keys=None, do_calc=True):
        from ase.calculators.gaussian import Gaussian, GaussianOptimizer
        geoopt_params = {
                "method": "wb97xd",
                "basis": "6-311++g(d,p)",
        }
        if keys is None:
            keys = self.mol_info.keys()

        for key in keys:
            for i, atoms in enumerate(self.mol_info[key]["aseatoms_list"]):
                atoms_tmp = atoms.copy()
                atoms_tmp.pbc = False
                atoms_tmp.cell = None
                if self.mol_info[key]["netcharge"] is None:
                    self._assign_totalcharge()
                nc = self.mol_info[key]["netcharge"]
                calc_geoopt = Gaussian(label=f'{key}_{i}', charge=nc, **geoopt_params)
                calc_geoopt.directory = self.mol_info[key]["directory"]
                opt = GaussianOptimizer(atoms_tmp, calc_geoopt)
                if do_calc:
                    opt.run(steps=256)

    def get_partial_charges(self, charge_type="resp", keys=None):
        """
        Get partial charges by RESP or AM1-BCC
        charge_type: "resp" or "am1bcc"
        keys: list of keys to get partial charges
        """
        if keys is None:
            keys = self.mol_info.keys()
        
        if charge_type not in ["resp", "am1bcc"]:
            assert False, "charge_type should be resp or am1bcc"

        for key in keys:
            if self.mol_info[key]["aseatoms_stable"] != None:
                atoms = self.mol_info[key]["aseatoms_stable"]
            else:
                atoms = self.mol_info[key]["aseatoms_list"][0]
            
            netcharge = 0
            label = key
            self.mol_info[key]["charge"] = ChargeCalculator(
                atoms,
                charge_type,
                netcharge,
                label,
                directory=self.mol_info[key]["directory"]
            )

            self.mol_info[key]["partial_charges"] = \
                self.mol_info[key]["charge"].get_partialcharges()
    
    def get_dihedral_qm(self, keys=None, do_calc=True):
        if keys is None:
            keys = self.mol_info.keys()

        for key in keys:
            if isinstance(self.mol_info[key]["aseatoms_stable"], Atoms):
                atoms = self.mol_info[key]["aseatoms_stable"]
            else:
                atoms = self.mol_info[key]["aseatoms_list"][0]

            if self.mol_info[key]["DihedCalc"] is None:
                self.mol_info[key]["DihedCalc"] = DihedCalculator(
                    atoms=atoms,
                    rdkitmol=self.mol_info[key]["rdkit"]["mol"],
                    label=key,
                    directory = self.mol_info[key]["directory"],
                )
            
            self.mol_info[key]["DihedCalc"].get_dihedral_qm(do_calc=do_calc)
    
    def get_dihedral_ff(self, keys=None, do_calc=True):
        if keys is None:
            keys = self.mol_info.keys()
        for key in keys:
            self.mol_info[key]["DihedCalc"].get_dihedral_ff(do_calc=do_calc)

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


    def get_molecule_off(self):
        molecules_off = []
        for mol in self.mol_info.keys():
            dirname = self.mol_info[mol]["directory"]
            sdffile = os.path.join(dirname, f"{mol}.sdf")

            if not os.path.exists(sdffile):
                self.get_sdf_from_molinfo()
            
            molecule_off = Molecule.from_file(sdffile)
            molecule_off.partial_charges = Quantity(self.mol_info[mol]["charges"], toolkit.unit.elementary_charge)
            molecule_off.name = mol
            # molecule_mm.total_charge = Quantity(self.mol_info[mol]["charge"], unit.elementary_charge)  ## why ??
            for meta_key in self.mol_info[mol]["metadata"].keys():
                for meta_ind in self.mol_info[mol]["metadata"][meta_key]:
                    molecule_off.atoms[meta_ind].metadata[meta_key] = True
            if "mol2file" in self.mol_info[mol]:
                molecule_off.mol2file = os.path.join(self.mol_info[mol]["directory"], self.mol_info[mol]["mol2file"])
            molecules_off.append(molecule_off)
        return molecules_off
        
    def save_crafter(self, filename="crafter.pkl"):
        with open(filename, mode='wb') as f:
            pickle.dump(self,f)
        
        for key in self.mol_info.keys():
            directory = self.mol_info[key]["directory"]
            with open(f"{directory}/mol_info.pkl", mode='wb') as f:
                mol_info = copy.deepcopy(self.mol_info[key])
                mol_info.pop("directory")
                pickle.dump(mol_info, f)
    
    def load_crafter(self, filename):
        with open(filename, mode='rb') as f:
            crafter = pickle.load(f)
        self.mol_info = crafter.mol_info

    def _assign_totalcharge(self):
        num_charge_none = 0
        total_charge = 0
        for key in self.mol_info.keys():
            totalnum_elec = self.mol_info[key]["aseatoms_list"][0].get_atomic_numbers().sum()
            openshell_flag = totalnum_elec % 2 == 1
            self.mol_info[key]["symbol"] = str(self.mol_info[key]["aseatoms_list"][0].symbols)
            if self.mol_info[key]["symbol"] in ["Li", "Na", "K", "Rb", "Cs", "Mg"]:
                self.mol_info[key]["netcharge"] = 1
            elif self.mol_info[key]["symbol"] in ["F", "Cl", "Br", "I"]:
                self.mol_info[key]["netcharge"] = -1
            elif self.mol_info[key]["symbol"] in ["Mg", "Ca", "Sr", "Ba"]:
                self.mol_info[key]["netcharge"] = 2
            elif openshell_flag:
                self.mol_info[key]["netcharge"] = None
            else:
                self.mol_info[key]["netcharge"] = 0
        for key in self.mol_info.keys():
            if self.mol_info[key]["netcharge"] is None:
                num_charge_none += 1 * self.mol_info[key]["Nmols"]
            else:
                total_charge += self.mol_info[key]["netcharge"] * self.mol_info[key]["Nmols"]
        if num_charge_none > 0 and total_charge % num_charge_none == 0:
            charge_per_none = int(-total_charge / num_charge_none)
            for key in self.mol_info.keys():
                if self.mol_info[key]["netcharge"] is None:
                    self.mol_info[key]["netcharge"] = charge_per_none

    def _il_assign(self, mol, mol2d, nc):
        fsalike_Nindex = []
        fsalike_Sindex = []
        fsalike_Oindex = []
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

            # self.mol_info[key]["metadata"]["FSA_N"] = fsalike_Nindex
            # self.mol_info[key]["metadata"]["FSA_S"] = fsalike_Sindex
            # self.mol_info[key]["metadata"]["FSA_O"] = fsalike_Oindex

def draw_dihedral_plots(mol_info, molkey):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(4, 3))
    n_dihedrals = len(mol_info[molkey]["dihedral_angle"])
    dihedral_pots_gt = []
    dihedral_pots_ff = []
    for i in range(n_dihedrals):
        dihedral_pot_tmp = []
        if mol_info[molkey]["dihedral_energy"][i] is not None:
            x_angle = mol_info[molkey]["dihedral_angle"][i]
            dihedral_qm = (mol_info[molkey]["dihedral_energy"][i] - mol_info[molkey]["dihedral_energy"][i].min()) / (units.kJ * (units.mol**-1))
            dihedral_ff = (mol_info[molkey]["dihedral_ffenergy"][i] - mol_info[molkey]["dihedral_ffenergy"][i].min())

            plt.scatter(x_angle, dihedral_qm)
            plt.plot(x_angle, dihedral_ff,label=f"dihedral_{i}")
            dihedral_pots_gt.append(dihedral_qm)
            dihedral_pots_ff.append(dihedral_ff)
        else:
            dihedral_pots_gt.append(None)
            dihedral_pots_ff.append(None)
            
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncol=2)
    plt.xlabel("Dihedral angle (deg)")
    plt.ylabel("Potential energy (kJ/mol)")
    plt.xticks(range(-180, 181, 60))
    plt.grid()
    return dihedral_pots_gt, dihedral_pots_ff

def draw_dihedral_structures(mol_info, molkey):
    # key_name = "MOL_0"
    from IPython.display import SVG
    from rdkit.Chem.Draw import rdMolDraw2D
    tm_list = []
    highlighAtoms_list = []
    legends_list = []
    n_dihedrals = len(mol_info[molkey]["rotatable_dihedral"])
    for i_dihed in range(n_dihedrals):
        highlighAtoms_list.append(mol_info[molkey]["rotatable_dihedral"][i_dihed])
        tm = rdMolDraw2D.PrepareMolForDrawing(mol_info[molkey]["rdkitmol2d"])
        tm_list.append(mol_info[molkey]["rdkitmol2d"])
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

