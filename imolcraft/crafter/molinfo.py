from rdkit import Chem
from ase.io import read, write
from ase import Atoms
from ase.calculators.gaussian import Gaussian
import os
import copy
import pickle
import yaml
import tempfile
import shutil
import numpy as np
from openff.toolkit.topology import Molecule
from openff.toolkit import Quantity
from openff import toolkit
from openmm.app import PDBFile, ForceField, PME
from openmm import XmlSerializer
from .asemol import (
    asemol_wrapper,
    cast_molecules,
    pdb2packmol,
    merge_asemols,
    aseatoms2pdb,
)
from .ffxml import gafftemplate2xml
from .gaffil_generators import GAFFilTemplateGenerator
from ..calculator import DihedralCalculator, Psi4GeoOptimizer
from ..calculator import ChargeCalculator, Psi4ChargeCalculator

from ..io.rdkit import atoms2rdkit

MOLINFO_KEYS = {
    "aseatoms_list": list,
    "aseatoms_geoopt": list,
    "aseatoms_stable": type(None),
    "ChargeCalc": type(None),
    "DihedCalc": type(None),
    "directory": str,
    "metadata": dict,
    "molecule_OFF": type(None),
    "natoms": type(None),
    "nmols": type(None),
    "netcharge": type(None),
    "networkX": list,
    "partial_charges": list,
    "rdkit": dict,
    "SMILES": str,
}


class Crafter:
    """
    Crafter is a class designed to handle molecular information and perform various
    operations such as geometry optimization, charge calculation, dihedral angle
    analysis, and force field generation. It integrates multiple tools and libraries
    like ASE, OpenFF, Psi4, and RDKit to streamline molecular modeling workflows.

    Parameters
    ----------
    yml : str, optional
        Path to a YAML file containing parameters for geometry optimization,
        charge calculation, force field generation, and molecular structure.
        If provided, the parameters and structure will be loaded automatically.

    Attributes
    ----------
    mol_info : dict
        A dictionary to store molecular information for each molecule.
    params_geoopt : dict or None
        Parameters for geometry optimization.
    params_charge : dict or None
        Parameters for charge calculation.
    params_ff : dict or None
        Parameters for force field generation.
    structure : dict or None
        Information about the molecular structure (e.g., crystal or liquid).
    """

    def __init__(self, yml=None):
        """
        Initialize the Crafter object.

        Parameters:
        ----------
        yml : str, optional
            Path to a YAML file containing parameters for geometry optimization,
            charge calculation, force field generation, and molecular structure.
            If provided, the parameters and structure will be loaded automatically.
        """
        self.mol_info = {}
        self.molatoms = None
        self.params_geoopt = None
        self.params_charge = None
        self.params_ff = None
        self.structure = None
        if yml is not None:
            self.from_yaml(yml)

    def from_yaml(self, filename):
        """
        Read a YAML file and extract parameters for geometry optimization,
        charge calculation, force field generation, and molecular structure.
        The YAML file should contain the following keys:
        - geoopt: Parameters for geometry optimization.
        - charge: Parameters for charge calculation.
        - forcefield: Parameters for force field generation.
        - structure: Information about the molecular structure (crystal or liquid).

        Parameters
        ----------
        filename : str
            Path to the YAML file.
        """
        data = self._parser_yaml(filename)
        self.params_geoopt = data["geoopt"]
        self.params_charge = data["charge"]
        self.params_ff = data["forcefield"]
        self.structure = data["structure"]

        if "cif" in self.structure.keys():
            if not os.path.isfile(self.structure["cif"]):
                new_cif = os.path.join(
                    os.path.dirname(filename), self.structure["cif"]
                )
                if os.path.isfile(new_cif):
                    self.structure["cif"] = new_cif
                else:
                    raise FileNotFoundError(f"CIF file not found: {self.structure['cif']}")
            else:
                self.structure["cif"] = self.structure["cif"]
        elif "molecules" in self.structure.keys():
            for i, molfile in enumerate(self.structure["molecules"]):
                if not os.path.isfile(molfile):
                    new_molfile = os.path.join(
                        os.path.dirname(filename), self.structure["molecules"][i]
                    )
                    if os.path.isfile(new_molfile):
                        self.structure["molecules"][i] = new_molfile
                    else:
                        raise FileNotFoundError(
                            f"Molecule file not found: {self.structure['molecules'][i]}"
                        )
                else:
                    self.structure["molecules"][i] = molfile

    def prep(self, do_opt=True, do_charge=True):
        """
        Prepare the system by reading the structure and generating the necessary
        molecular information. This includes reading the structure file,
        generating the RDKit molecule, and performing geometry optimization
        and charge calculation if specified.

        Parameters
        ----------
        do_opt : bool, optional
            If True, perform geometry optimization. Default is True.
        do_charge : bool, optional
            If True, perform charge calculation. Default is True.
        """
        # initial structure generation
        if self.structure["type"] == "crystal":
            self.from_crystal(self.structure["cif"])
        elif self.structure["type"] == "liquid":
            for i, atoms in enumerate(self.structure["molecules"]):
                atoms = read(atoms)
                asemol_wrap = asemol_wrapper(atoms)
                self.atoms_unwrap = asemol_wrap.unwrap_molecules()
                _, _, networkX = asemol_wrap.get_ase_molecules(out_nX=True)
                self.append_fromAtomsList(
                    [atoms],
                    f"MOL_{i}",
                    Nmols=int(self.structure["nmols"][i]),
                    networkX=networkX,
                )
            self._assign_totalcharge()
        self.get_rdkitmol()
        self.get_molecule_off()
        if do_opt is True:
            self.get_optstructure(**self.params_geoopt)
        if do_charge is True:
            self.get_partial_charges(params_ff=self.params_ff, **self.params_charge)
            self.get_molecule_off()
        
        # Save PDB files with bond information for each molecule
        self.save_molecule_pdbs_with_bonds()

    def build(self):
        """
        Build the system using the prepared molecules and force field parameters.
        This method generates the system XML files and creates the OpenMM system.
        It handles both crystal and liquid structures, creating the necessary
        supercell and adding bonds between atoms.
        The generated system is saved in an XML format for later use.
        """
        self.get_molecule_off()
        molecules = []

        for key in self.mol_info.keys():
            molecule = self.mol_info[key]["molecule_OFF"]
            molecule.mol2file = self.mol_info[key]["ChargeCalc"].mol2file
            molecules.append(molecule)

        if self.params_ff["fftype"].split("-")[0] == "gaff":
            fftemplate_gen = GAFFilTemplateGenerator(
                molecules=molecules,
                forcefield=self.params_ff["fftype"],
                il_assign=self.params_ff.get("fsa_assign", None),
            )
        else:
            assert False, "Unknown forcefield type. Please check the forcefield type."

        # output xml files
        ffxmlfiles = gafftemplate2xml(
            molecules,
            fftemplate_gen,
            ion_ffxml=self.params_ff.get("iontype", "amber/ions/ionsff99_tip3p.xml"),
        )

        # create structure
        # # crystal structure
        if self.structure["type"] == "crystal":
            unit_cell_atoms = merge_asemols(self.molatoms)
            repeat_factors = self.structure.get("repeat", [1, 1, 1])
            system_atoms = unit_cell_atoms.repeat(repeat_factors)
            with tempfile.NamedTemporaryFile() as temp_pdb:
                temp_pdb_name = temp_pdb.name
                aseatoms2pdb(temp_pdb_name, system_atoms)
                pdb = PDBFile(temp_pdb_name)
                
                # Use optimized bond calculation for supercells
                if repeat_factors != [1, 1, 1]:
                    # Supercell case - use optimized calculation
                    pdb_b = asemol_wrapper(system_atoms)
                    _ = pdb_b.get_bonds(unit_cell_atoms=unit_cell_atoms, repeat_factors=repeat_factors)
                else:
                    # Unit cell case - use standard calculation
                    pdb_b = asemol_wrapper(system_atoms)
                    _ = pdb_b.get_bonds()

                bonds_list = [[bond[0], bond[1]] for bond in pdb_b.bonds]
                atomlist_openmm = [a for a in pdb.topology.atoms()]
                for bond in bonds_list:
                    a1 = atomlist_openmm[bond[0]]
                    a2 = atomlist_openmm[bond[1]]
                    pdb.topology.addBond(a1, a2)
        # liquid structure
        elif self.structure["type"] == "liquid":
            molstructures = self.structure["molecules"]
            b, a, m = pdb2packmol(
                molstructures,
                fixed_property=self.structure.get("fixed_property", "cell"),
                priority_property=self.structure.get("priority_property", "density"),
                num_mols=self.structure.get("nmols", None),
                density=self.structure.get("density_kgm3", None),
                cell=self.structure.get("cell_A", None),
                outfile="supercell.pdb",
            )
            pdb = PDBFile("supercell.pdb")
            atomlist_openmm = [a for a in pdb.topology.atoms()]
            for bond in b:
                a1 = atomlist_openmm[bond[0]]
                a2 = atomlist_openmm[bond[1]]
                pdb.topology.addBond(a1, a2)

        # Save the supecell with bonds
        pdb.writeFile(pdb.topology, pdb.positions, open("supercell_bonds.pdb", "w"))
        pdb = PDBFile("supercell_bonds.pdb")

        # create system
        forcefield = ForceField(*ffxmlfiles)
        system = forcefield.createSystem(pdb.topology, nonbondedMethod=PME)

        with open("system.xml", "w") as output:
            output.write(XmlSerializer.serialize(system))

    def save_molecule_pdbs_with_bonds(self, keys=None, output_dir="."):
        """
        Save PDB files with bond information for each molecule.
        This enables using the PDB files with the Dihedral optimizer.

        Parameters
        ----------
        keys : list, optional
            List of molecule keys to save. If None, saves all molecules.
        output_dir : str, optional
            Directory to save the PDB files. Default is current directory.
        """
        import os
        if keys is None:
            keys = self.mol_info.keys()
        
        for key in keys:
            # Get the atoms from aseatoms_stable (geometry optimized) or aseatoms_list
            if self.mol_info[key]["aseatoms_stable"] is not None:
                if isinstance(self.mol_info[key]["aseatoms_stable"], list):
                    atoms = self.mol_info[key]["aseatoms_stable"][0]
                else:
                    atoms = self.mol_info[key]["aseatoms_stable"]
            elif len(self.mol_info[key]["aseatoms_list"]) > 0:
                atoms = self.mol_info[key]["aseatoms_list"][0]
            else:
                continue
            
            # Create a temporary PDB file
            with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as temp_pdb:
                temp_pdb_name = temp_pdb.name
                aseatoms2pdb(temp_pdb_name, atoms)
                pdb = PDBFile(temp_pdb_name)
                
                # Calculate bonds
                pdb_b = asemol_wrapper(atoms)
                _ = pdb_b.get_bonds()
                bonds_list = [[bond[0], bond[1]] for bond in pdb_b.bonds]
                
                # Add bonds to topology
                atomlist_openmm = [a for a in pdb.topology.atoms()]
                for bond in bonds_list:
                    a1 = atomlist_openmm[bond[0]]
                    a2 = atomlist_openmm[bond[1]]
                    pdb.topology.addBond(a1, a2)
                
                # Save PDB with bonds
                output_filename = os.path.join(output_dir, f"{key}_bonds.pdb")
                with open(output_filename, "w") as f:
                    PDBFile.writeFile(pdb.topology, pdb.positions, f)
                
                # Clean up temporary file
                os.unlink(temp_pdb_name)

    def get_ffxml(self):
        self.get_molecule_off()
        molecules = []
        for key in self.mol_info.keys():
            molecule = self.mol_info[key]["molecule_OFF"]
            molecule.mol2file = self.mol_info[key]["ChargeCalc"].mol2file
            molecules.append(molecule)
        if self.params_ff["fftype"].split("-")[0] == "gaff":
            fftemplate_gen = GAFFilTemplateGenerator(
                molecules=molecules,
                forcefield=self.params_ff["fftype"],
                il_assign=self.params_ff.get("fsa_assign", None),
            )
        else:
            assert False, "Unknown forcefield type. Please check the forcefield type."
        # output xml files
        _ = gafftemplate2xml(
            molecules,
            fftemplate_gen,
            ion_ffxml=self.params_ff.get("iontype", "amber/ions/ionsff99_tip3p.xml"),
        )

    def _initialize_molinfo(self, key: str):
        """
        Initialize mol_info[key] with predefined keys.
        """
        self.mol_info[key] = {
            k: v() if callable(v) else v for k, v in MOLINFO_KEYS.items()
        }

    def _parser_yaml(self, filename):
        with open(filename, "r") as file:
            data = yaml.safe_load(file)

        if set(data.keys()) <= set(["geoopt", "charge", "forcefield", "structure"]):
            pass
        else:
            assert False, "Unknown keys in the yaml file. Please check the file."

        return data

    def from_crystal(self, filename: str, assign_totalcharge=True):
        """
        Read cif, xyz, POSCAR, ... file and append to mol_info dictionary
        filename: Structure file name (It should be ASE compatible)
        """
        atoms = read(filename)
        asemol_wrap = asemol_wrapper(atoms)
        self.atoms_unwrap = asemol_wrap.unwrap_molecules()
        self.molatoms, self.molecule_list, self.networkX = (
            asemol_wrap.get_ase_molecules(out_nX=True)
        )

        for i, mol_idx in enumerate(self.molecule_list):
            al = [self.molatoms[i] for i in mol_idx]
            nX = [self.networkX[i] for i in mol_idx]
            self.append_fromAtomsList(
                al, key=f"MOL_{i}", Nmols=len(mol_idx), networkX=nX
            )

        if assign_totalcharge:
            self._assign_totalcharge()
        else:
            print(f"Warning: MOL_{i} has no charge information")
            print("Please define the total charge manually")

    def append_fromAtomsList(
        self, atomslist: list, key: str, Nmols: int = None, networkX: list = []
    ):
        """
        Append List of ase.Atoms to mol_info dictionary

        Parameters
        ----------
        atomslist: List of ase.Atoms
        key: key of mol_info dictionary
        Nmols: Number of molecules
        networkX: List of NetworkX graph
        """
        self._initialize_molinfo(key)
        self.mol_info[key]["aseatoms_list"] = atomslist
        self.mol_info[key]["aseatoms_geoopt"] = [None for _ in range(len(atomslist))]
        self.mol_info[key]["Natoms"] = len(atomslist[0])
        self.mol_info[key]["Nmols"] = len(atomslist) if Nmols is None else Nmols
        if networkX != []:
            assert len(atomslist) == len(
                networkX
            ), "The length of atomslist and networkX should be the same"
            self.mol_info[key]["networkX"] = networkX
        os.makedirs(key, exist_ok=True)
        self.mol_info[key]["directory"] = key
        write(os.path.join(key, f"{key}.xyz"), atomslist[0], format="xyz")

    def get_rdkitmol(self, keys=None, il_assign=True):
        """
        Get RDKit molecule from ASE atoms
        Parameters
        ----------
        keys: list of keys to get RDKit molecule
        il_assign: bool
            If True, assign special ionic liquids to the molecule
        """
        if keys is None:
            keys = self.mol_info.keys()

        for key in keys:
            if self.mol_info[key]["netcharge"] is None:
                self._assign_totalcharge()
                break

        for key in keys:
            nc = self.mol_info[key]["netcharge"]
            if il_assign:
                mol, mol2d, il_dict = atoms2rdkit(
                    self.mol_info[key]["aseatoms_list"][0], nc=nc, il_assign=True
                )
                if il_dict is not None:
                    self.mol_info[key]["metadata"].update(il_dict)
            else:
                mol, mol2d = atoms2rdkit(
                    self.mol_info[key]["aseatoms_list"][0], nc=nc, il_assign=False
                )
            self.mol_info[key]["rdkit"] = {"mol": mol, "mol2d": mol2d}

    def get_smiles(self, keys=None):
        """
        Get SMILES from RDKit molecule
        Parameters
        ----------
        keys: list of keys to get SMILES
        """
        if keys is None:
            keys = self.mol_info.keys()
        for key in keys:
            if self.mol_info[key]["rdkit"] is None:
                self.get_rdkitmol([key])
            mol = self.mol_info[key]["rdkit"]["mol"]
            self.mol_info[key]["SMILES"] = Chem.MolToSmiles(mol)

    def get_sdf(self, keys=None):
        """
        Get SDF from RDKit molecule

        Parameters
        ----------
        keys: list of keys to get SDF
        """
        if keys is None:
            keys = self.mol_info.keys()

        for key in keys:
            if "mol" not in self.mol_info[key]["rdkit"]:
                self.get_rdkitmol([key])

            mol = self.mol_info[key]["rdkit"]["mol"]
            output_dir = f"{key}"
            self.mol_info[key]["directory"] = output_dir
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_file = os.path.join(output_dir, f"{key}.sdf")
            writer = Chem.SDWriter(output_file)
            writer.write(mol)
            writer.close()

    # GeoOptimizerとして別ファイルに移す案もあり
    def get_optstructure(self, keys=None, do_calc=True, rmsd=0.2, **kwargs):
        """
        Get optimized structure using Gaussian or Psi4

        Parameters
        ----------
        keys: list of keys to get optimized structure
        do_calc: bool
            If True, perform geometry optimization. Default is True.
        rmsd: float
            If the RMSD between the current and previous molecules is less than this
            value, skip the geometry optimization. Default is 0.2 A^2.
        kwargs: dict
            Additional parameters for geometry optimization.
        """
        geoopt_params_g16 = {
            "method": "wb97xd",
            "basis": "6-311+g(2d,p)",
            "opt": "maxcycle=256",
        }
        geoopt_params_psi4 = {
            "method": "wb97x-d",
            "basis": "6-311+g(2d,p)",
        }

        if kwargs.get("software") == "psi4":
            geoopt_params_psi4.update(kwargs)
            method = geoopt_params_psi4.get("method")
            if method == "wb97xd":
                geoopt_params_psi4["method"] = "wb97x-d"
            geoopt_params_psi4.pop("software", None)
            psi4_flag = True
            g16_flag = False
        elif kwargs.get("software") == "g16":
            geoopt_params_g16.update(kwargs)
            geoopt_params_g16.pop("software", None)
            psi4_flag = False
            g16_flag = True
        else:
            software = kwargs.get("software")
            assert (
                False
            ), f"Unknown software: {software}. Please check the software name."

        if keys is None:
            keys = self.mol_info.keys()

        for key in keys:
            trajectory = []
            for i, atoms in enumerate(self.mol_info[key]["aseatoms_list"]):
                atoms_tmp = atoms.copy()
                atoms_tmp.pbc = False
                atoms_tmp.cell = None
                if self.mol_info[key]["netcharge"] is None:
                    self._assign_totalcharge()
                nc = self.mol_info[key]["netcharge"]

                if g16_flag is True:
                    calc_geoopt = Gaussian(
                        label=f"{self.mol_info[key]['directory']}/{key}_{i}",
                        charge=nc,
                        **geoopt_params_g16
                    )
                    # calc_geoopt.directory = self.mol_info[key]["directory"]
                    atoms_tmp.calc = calc_geoopt
                    trajectory.append(atoms_tmp.calc.label + ".log")
                elif psi4_flag is True:
                    calc_geoopt = Psi4GeoOptimizer(
                        atoms=atoms_tmp,
                        method=geoopt_params_psi4["method"],
                        basis_set=geoopt_params_psi4["basis"],
                        charge=nc,
                        multiplicity=1,
                        label=f"{self.mol_info[key]['directory']}/{key}_{i}_psi4",
                    )
                    # calc_geoopt.directory = self.mol_info[key]["directory"]
                    atoms_tmp.calc = calc_geoopt
                    print(calc_geoopt.label, calc_geoopt.directory, "label and directory")
                    trajectory.append(calc_geoopt.label + ".xyz")

                if do_calc:
                    rmsd_skip = False
                    for j in range(0, i):
                        gj = self.mol_info[key]["networkX"][j]
                        gi = self.mol_info[key]["networkX"][i]
                        rmsd_ji, _ = cast_molecules(gj, gi)
                        print(rmsd_ji, i, j)
                        if rmsd_ji < rmsd:
                            rmsd_skip = True
                            break
                    if rmsd_skip:
                        self.mol_info[key]["aseatoms_geoopt"][i] = read(
                            trajectory[j], index=-1
                        )
                        shutil.copy(trajectory[j], trajectory[i])
                        print(
                            f"Skip geometry optimization of {key}_{i}: "
                            f"RMSD = {rmsd_ji} A < {rmsd}"
                        )
                        rmsd_skip = False
                    else:
                        self.mol_info[key]["aseatoms_geoopt"][i] = atoms_tmp
                        print(f"Calculating geometry optimization of {key}_{i}")
                        _ = self.mol_info[key]["aseatoms_geoopt"][
                            i
                        ].get_potential_energy()
                        if psi4_flag is True:
                            self.mol_info[key]["aseatoms_geoopt"][i].positions = (
                                self.mol_info[key]["aseatoms_geoopt"][
                                    i
                                ].calc.atoms.positions
                            )
                        print(f"Finished geometry optimization of {key}_{i}")

            minidx = np.array(
                [
                    a.get_potential_energy()
                    for a in self.mol_info[key]["aseatoms_geoopt"]
                ]
            ).argmin()
            self.mol_info[key]["aseatoms_stable"] = self.mol_info[key][
                "aseatoms_geoopt"
            ][minidx]

    def get_molecule_off(self, keys=None, **kwargs):
        """
        Get OpenFF molecule from ASE atoms

        Parameters
        ----------
        keys: list of keys to get OpenFF molecule
        kwargs: dict
            Additional attributes for OpenFF molecule.
        """
        if keys is None:
            keys = self.mol_info.keys()

        for key in keys:
            dirname = self.mol_info[key]["directory"]
            sdffile = os.path.join(dirname, f"{key}.sdf")
            # if not os.path.exists(sdffile):
            #     self.get_sdf()
            self.get_sdf()

            molecule_off = Molecule.from_file(sdffile, allow_undefined_stereo=True)
            if list(self.mol_info[key]["partial_charges"]) != []:
                molecule_off.partial_charges = Quantity(
                    self.mol_info[key]["partial_charges"],
                    toolkit.unit.elementary_charge,
                )

            molecule_off.name = key
            for meta_key in self.mol_info[key]["metadata"].keys():
                for meta_ind in self.mol_info[key]["metadata"][meta_key]:
                    molecule_off.atoms[meta_ind].metadata[meta_key] = True

            for attr_key, attr_value in kwargs.items():
                setattr(molecule_off, attr_key, attr_value)

            self.mol_info[key]["molecule_OFF"] = molecule_off

    def get_partial_charges(self, keys=None, params_ff=None, **kwargs):
        """
        Get partial charges by RESP or AM1-BCC
        charge_type: "resp" or "am1bcc"

        Parameters
        ----------
        keys: list of keys to get partial charges
        params_ff: dict
            Force field parameters for charge calculation.
        kwargs: dict
            Additional parameters for charge calculation.
        """
        if keys is None:
            keys = self.mol_info.keys()

        charge_type = kwargs.get("type", "resp")
        software = kwargs.get("software", False)
        params = kwargs.get("params", None)
        if software == "psi4":
            psi4_flag = True
        else:
            psi4_flag = False
        kwargs.pop("type", None)
        kwargs.pop("software", None)

        if charge_type not in ["resp", "am1bcc"]:
            assert False, "charge_type should be resp or am1bcc"
        
        # assign total charge if not assigned
        if any(self.mol_info[key]["netcharge"] is None for key in keys):
            self._assign_totalcharge()

        for key in keys:
            if self.mol_info[key]["aseatoms_stable"] is not None:
                atoms = self.mol_info[key]["aseatoms_stable"]
            else:
                atoms = self.mol_info[key]["aseatoms_list"][0]

            netcharge = self.mol_info[key]["netcharge"]
            label = key

            if psi4_flag is False:
                self.mol_info[key]["ChargeCalc"] = ChargeCalculator(
                    atoms,
                    charge_type,
                    netcharge,
                    label,
                    directory=self.mol_info[key]["directory"],
                    params=params,
                )
            elif psi4_flag is True:
                if self.mol_info[key]["molecule_OFF"] is None:
                    self.get_molecule_off(keys=[key])
                molecule = self.mol_info[key]["molecule_OFF"]
                molecule.conformers[0].magnitude[:] = atoms.positions
                self.mol_info[key]["ChargeCalc"] = Psi4ChargeCalculator(
                    atoms,
                    charge_type,
                    netcharge,
                    label,
                    directory=self.mol_info[key]["directory"],
                    params=params,
                )

            self.mol_info[key]["ChargeCalc"].get_partialcharges()
            self.mol_info[key]["partial_charges"] = self.mol_info[key][
                "ChargeCalc"
            ].partial_charges
            self.get_molecule_off(
                keys=[key], **{"mol2file": self.mol_info[key]["ChargeCalc"].mol2file}
            )

        if params_ff is not None:
            self._adjust_charges(params_ff)

    def get_dihedral_qm(self, keys=None, do_calc=True):
        """
        Get dihedral angles using quantum mechanical calculations

        Parameters
        ----------
        keys: list of keys to get dihedral angles
        do_calc: bool
            If True, perform dihedral angle scan calculation. Default is True.
        """
        if keys is None:
            keys = self.mol_info.keys()

        for key in keys:
            if isinstance(self.mol_info[key]["aseatoms_stable"], Atoms):
                atoms = self.mol_info[key]["aseatoms_stable"]
            else:
                atoms = self.mol_info[key]["aseatoms_list"][0]

            if self.mol_info[key]["DihedCalc"] is None:
                self.mol_info[key]["DihedCalc"] = DihedralCalculator(
                    atoms=atoms,
                    label=key,
                    directory=self.mol_info[key]["directory"],
                )

            self.mol_info[key]["DihedCalc"].do_qmscan(do_calc=do_calc)

    def get_dihedral_ff(self, keys=None, do_calc=True):
        """
        Get dihedral angles using force field parameters

        Parameters
        ----------
        keys: list of keys to get dihedral angles
        do_calc: bool
            If True, perform dihedral angle scan calculation. Default is True.
        """
        if keys is None:
            keys = self.mol_info.keys()
        for key in keys:
            self.mol_info[key]["DihedCalc"].do_ffscan(do_calc=do_calc)

    def _adjust_charges(self, ff_params=None):
        for key in self.mol_info.keys():
            net_charge = self.mol_info[key]["netcharge"]
            charges = np.array(
                [np.float64(ee) for ee in self.mol_info[key]["partial_charges"]]
            )

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

            if not np.isclose(charge_deficit, 0.0, atol=1e-8):
                print(
                    (
                        f"Net charge of {key} is {net_charge} and "
                        f"total charge is {total_charge}"
                    )
                )
                charges = charges - charge_deficit / len(charges)
                total_charge = np.sum(charges)
                charge_deficit = total_charge - net_charge
                charges[0] -= charge_deficit

            self.mol_info[key]["partial_charges"] = charges

    def save_crafter(self, filename="crafter.pkl"):
        """
        Save the Crafter object to a file using pickle.

        Parameters
        ----------
        filename : str
            The name of the file to save the Crafter object to.
        """
        with open(filename, mode="wb") as f:
            pickle.dump(self, f)

        for key in self.mol_info.keys():
            directory = self.mol_info[key]["directory"]
            with open(f"{directory}/mol_info.pkl", mode="wb") as f:
                mol_info = copy.deepcopy(self.mol_info[key])
                mol_info.pop("directory")
                pickle.dump(mol_info, f)

    def load_crafter(self, filename):
        """
        Load the Crafter object from a file using pickle.

        Parameters
        ----------
        filename : str
            The name of the file to load the Crafter object from.
        """
        with open(filename, mode="rb") as f:
            crafter = pickle.load(f)
        self.mol_info = crafter.mol_info
        self.__dict__.update(crafter.__dict__)

    def _assign_totalcharge(self):
        import networkx as nx

        def _extract_ring(nxmol):
            ring = nx.cycle_basis(nxmol)
            ring_list = []
            element_list = []
            for r in ring:
                ring_list.append(r)
                element_list.append([nxmol.nodes[i]["element"] for i in r])
            return ring_list, element_list

        def _Ncation(nxmol):
            n_Ncation = 0
            for i in range(len(nxmol.nodes)):
                node = nxmol.nodes[i]
                edges_containing_node = list(nxmol.edges([i]))
                if node["element"] == "N" and len(edges_containing_node) == 4:
                    n_Ncation += 1
            return n_Ncation

        num_charge_none = 0
        total_charge = 0
        for key in self.mol_info.keys():
            totalnum_elec = (
                self.mol_info[key]["aseatoms_list"][0].get_atomic_numbers().sum()
            )
            openshell_flag = totalnum_elec % 2 == 1
            mim_flag = False
            if openshell_flag:
                rings, elements = _extract_ring(self.mol_info[key]["networkX"][0])
                for i in range(len(rings)):
                    r = rings[i]
                    e = elements[i]
                    if e.count("C") == 3 and e.count("N") == 2 and len(r) == 5:
                        mim_flag = True

                chg_cation = _Ncation(self.mol_info[key]["networkX"][0])
                if chg_cation > 0:
                    self.mol_info[key]["netcharge"] = chg_cation
                    openshell_flag = False

            self.mol_info[key]["symbol"] = str(
                self.mol_info[key]["aseatoms_list"][0].symbols
            )
            if self.mol_info[key]["symbol"] in ["Li", "Na", "K", "Rb", "Cs"]:
                self.mol_info[key]["netcharge"] = 1
            elif self.mol_info[key]["symbol"] in ["F", "Cl", "Br", "I"]:
                self.mol_info[key]["netcharge"] = -1
            elif self.mol_info[key]["symbol"] in ["Mg", "Ca", "Sr", "Ba"]:
                self.mol_info[key]["netcharge"] = 2
            elif mim_flag:
                self.mol_info[key]["netcharge"] = 1
            elif openshell_flag:
                self.mol_info[key]["netcharge"] = None
            elif self.mol_info[key]["netcharge"] is None:
                self.mol_info[key]["netcharge"] = 0
        for key in self.mol_info.keys():
            if self.mol_info[key]["netcharge"] is None:
                num_charge_none += 1 * self.mol_info[key]["Nmols"]
            else:
                total_charge += (
                    self.mol_info[key]["netcharge"] * self.mol_info[key]["Nmols"]
                )
        if num_charge_none > 0 and total_charge % num_charge_none == 0:
            charge_per_none = int(-total_charge / num_charge_none)
            for key in self.mol_info.keys():
                if self.mol_info[key]["netcharge"] is None:
                    self.mol_info[key]["netcharge"] = charge_per_none
