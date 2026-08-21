import copy
import os
import pickle
import shutil
import tempfile

import networkx as nx
import numpy as np
import yaml
from ase import Atoms
from ase.calculators.gaussian import Gaussian
from ase.io import read, write
from openff import toolkit
from openff.toolkit import Quantity
from openff.toolkit.topology import Molecule
from openmm import XmlSerializer
from openmm.app import PME, ForceField, PDBFile
from rdkit import Chem

from ..calculator import (
    ChargeCalculator,
    DihedralCalculator,
    Psi4ChargeCalculator,
    Psi4GeoOptimizer,
)
from ..io.rdkit import atoms2rdkit
from .asemol import (
    aseatoms2pdb,
    asemol_wrapper,
    cast_molecules,
    merge_asemols,
    pdb2packmol,
)
from .ffxml import gafftemplate2xml
from .gaffil_generators import GAFFilTemplateGenerator

#: Schema of a single molecule entry of ``Crafter.mol_info``: the key and the
#: zero-argument callable producing its initial value (``type(None)`` yields
#: ``None``). The key names must match the ones the writers use verbatim,
#: otherwise a typo silently adds a second key instead of raising.
MOLINFO_KEYS = {
    "aseatoms_list": list,
    "aseatoms_geoopt": list,
    "aseatoms_stable": type(None),
    "ChargeCalc": type(None),
    "DihedCalc": type(None),
    "directory": str,
    "metadata": dict,
    "molecule_OFF": type(None),
    "Natoms": type(None),
    "Nmols": type(None),
    "netcharge": type(None),
    "networkX": list,
    "partial_charges": list,
    "rdkit": dict,
    "SMILES": str,
    "symbol": type(None),
}

#: Top-level sections accepted in the input YAML file.
YAML_SECTIONS = ["geoopt", "charge", "forcefield", "structure"]

#: Default ion parameter file shipped with openmmforcefields.
DEFAULT_ION_FFXML = "amber/ions/ionsff99_tip3p.xml"

#: Partial charge schemes accepted by ``get_partial_charges``.
VALID_CHARGE_TYPES = ["resp", "am1bcc"]

#: Default geometry optimization settings per QM backend.
GEOOPT_DEFAULTS = {
    "g16": {
        "method": "wb97xd",
        "basis": "6-311+g(2d,p)",
        "opt": "maxcycle=256",
    },
    "psi4": {
        "method": "wb97x-d",
        "basis": "6-311+g(2d,p)",
    },
}

#: Net charge of monatomic ions identified by their chemical symbol.
MONATOMIC_ION_CHARGES = {
    **{symbol: 1 for symbol in ("Li", "Na", "K", "Rb", "Cs")},
    **{symbol: -1 for symbol in ("F", "Cl", "Br", "I")},
    **{symbol: 2 for symbol in ("Mg", "Ca", "Sr", "Ba")},
}


def _resolve_path(path: str, basedir: str, description: str) -> str:
    """
    Return ``path`` if it exists, otherwise the same name resolved relative to
    ``basedir`` (i.e. next to the YAML file that referenced it).
    """
    if os.path.isfile(path):
        return path
    candidate = os.path.join(basedir, path)
    if os.path.isfile(candidate):
        return candidate
    raise FileNotFoundError(f"{description} not found: {path}")


def _extract_ring(nxmol):
    """Return the rings of a molecule graph and the elements they contain."""
    rings = nx.cycle_basis(nxmol)
    elements = [[nxmol.nodes[i]["element"] for i in ring] for ring in rings]
    return rings, elements


def _has_imidazolium_ring(nxmol) -> bool:
    """True if the molecule contains a five-membered C3N2 (imidazolium) ring."""
    rings, elements = _extract_ring(nxmol)
    return any(
        len(ring) == 5 and elem.count("C") == 3 and elem.count("N") == 2
        for ring, elem in zip(rings, elements)
    )


def _count_quaternary_nitrogen(nxmol) -> int:
    """Number of four-coordinated nitrogen atoms, i.e. cationic centres."""
    return sum(
        1
        for i in range(len(nxmol.nodes))
        if nxmol.nodes[i]["element"] == "N" and len(list(nxmol.edges([i]))) == 4
    )


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

    # ------------------------------------------------------------------
    # input / setup
    # ------------------------------------------------------------------
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

        # structure files may be given relative to the YAML file
        basedir = os.path.dirname(filename)
        if "cif" in self.structure.keys():
            self.structure["cif"] = _resolve_path(
                self.structure["cif"], basedir, "CIF file"
            )
        elif "molecules" in self.structure.keys():
            self.structure["molecules"] = [
                _resolve_path(molfile, basedir, "Molecule file")
                for molfile in self.structure["molecules"]
            ]

    def _parser_yaml(self, filename):
        with open(filename, "r") as file:
            data = yaml.safe_load(file)

        unknown = sorted(set(data.keys()) - set(YAML_SECTIONS))
        if unknown:
            raise ValueError(
                f"Unknown keys in the yaml file: {unknown}. "
                f"Please check the file. Valid keys are {YAML_SECTIONS}."
            )

        return data

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
            for i, molfile in enumerate(self.structure["molecules"]):
                atoms = read(molfile)
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
            self.append_fromAtomsList(
                [self.molatoms[j] for j in mol_idx],
                key=f"MOL_{i}",
                Nmols=len(mol_idx),
                networkX=[self.networkX[j] for j in mol_idx],
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
        info = self.mol_info[key]
        info["aseatoms_list"] = atomslist
        info["aseatoms_geoopt"] = [None for _ in range(len(atomslist))]
        info["Natoms"] = len(atomslist[0])
        info["Nmols"] = len(atomslist) if Nmols is None else Nmols
        if networkX != []:
            if len(atomslist) != len(networkX):
                raise ValueError(
                    "The length of atomslist and networkX should be the same: "
                    f"{len(atomslist)} != {len(networkX)}"
                )
            info["networkX"] = networkX

        os.makedirs(key, exist_ok=True)
        info["directory"] = key
        write(os.path.join(key, f"{key}.xyz"), atomslist[0], format="xyz")

    def _initialize_molinfo(self, key: str):
        """
        Initialize mol_info[key] with predefined keys.
        """
        self.mol_info[key] = {k: factory() for k, factory in MOLINFO_KEYS.items()}

    def _keys_or_all(self, keys):
        """Return the requested mol_info keys, defaulting to all of them."""
        return self.mol_info.keys() if keys is None else keys

    # ------------------------------------------------------------------
    # force field / system building
    # ------------------------------------------------------------------
    def _molecules_with_mol2(self):
        """
        Refresh the OpenFF molecules and attach the mol2 file produced by the
        charge calculation, which the GAFF template generator needs.
        """
        self.get_molecule_off()
        molecules = []
        for key in self.mol_info.keys():
            molecule = self.mol_info[key]["molecule_OFF"]
            molecule.mol2file = self.mol_info[key]["ChargeCalc"].mol2file
            molecules.append(molecule)
        return molecules

    def _make_fftemplate_generator(self, molecules):
        if self.params_ff["fftype"].split("-")[0] == "gaff":
            return GAFFilTemplateGenerator(
                molecules=molecules,
                forcefield=self.params_ff["fftype"],
                il_assign=self.params_ff.get("fsa_assign", None),
            )
        raise ValueError(
            f"Unknown forcefield type: {self.params_ff['fftype']}. "
            "Please check the forcefield type."
        )

    def _generate_ffxml(self):
        """Write one force field XML per molecule and return the file names."""
        molecules = self._molecules_with_mol2()
        fftemplate_gen = self._make_fftemplate_generator(molecules)
        return gafftemplate2xml(
            molecules,
            fftemplate_gen,
            ion_ffxml=self.params_ff.get("iontype", DEFAULT_ION_FFXML),
        )

    def get_ffxml(self):
        _ = self._generate_ffxml()

    def build(self):
        """
        Build the system using the prepared molecules and force field parameters.
        This method generates the system XML files and creates the OpenMM system.
        It handles both crystal and liquid structures, creating the necessary
        supercell and adding bonds between atoms.
        The generated system is saved in an XML format for later use.
        """
        # output xml files
        ffxmlfiles = self._generate_ffxml()

        # create structure
        if self.structure["type"] == "crystal":
            pdb = self._build_crystal_pdb()
        elif self.structure["type"] == "liquid":
            pdb = self._build_liquid_pdb()

        # Save the supecell with bonds
        with open("supercell_bonds.pdb", "w") as f:
            pdb.writeFile(pdb.topology, pdb.positions, f)
        pdb = PDBFile("supercell_bonds.pdb")

        # create system
        forcefield = ForceField(*ffxmlfiles)
        system = forcefield.createSystem(pdb.topology, nonbondedMethod=PME)

        with open("system.xml", "w") as output:
            output.write(XmlSerializer.serialize(system))

    @staticmethod
    def _add_bonds_to_topology(pdb, bonds):
        """Add the given (index1, index2) bonds to an OpenMM PDB topology."""
        atomlist_openmm = [a for a in pdb.topology.atoms()]
        for bond in bonds:
            pdb.topology.addBond(atomlist_openmm[bond[0]], atomlist_openmm[bond[1]])

    def _build_crystal_pdb(self):
        """Build the supercell PDB (with bonds) of a crystalline system."""
        unit_cell_atoms = merge_asemols(self.molatoms)
        repeat_factors = self.structure.get("repeat", [1, 1, 1])
        system_atoms = unit_cell_atoms.repeat(repeat_factors)

        with tempfile.NamedTemporaryFile() as temp_pdb:
            aseatoms2pdb(temp_pdb.name, system_atoms)
            pdb = PDBFile(temp_pdb.name)

            # Use optimized bond calculation for supercells
            pdb_b = asemol_wrapper(system_atoms)
            if repeat_factors != [1, 1, 1]:
                # Supercell case - use optimized calculation
                _ = pdb_b.get_bonds(
                    unit_cell_atoms=unit_cell_atoms, repeat_factors=repeat_factors
                )
            else:
                # Unit cell case - use standard calculation
                _ = pdb_b.get_bonds()

            self._add_bonds_to_topology(
                pdb, [[bond[0], bond[1]] for bond in pdb_b.bonds]
            )
        return pdb

    def _build_liquid_pdb(self):
        """Pack the molecules with packmol and return the bonded PDB."""
        bonds, _, _ = pdb2packmol(
            self.structure["molecules"],
            fixed_property=self.structure.get("fixed_property", "cell"),
            priority_property=self.structure.get("priority_property", "density"),
            nmols=self.structure.get("nmols", None),
            density=self.structure.get("density_kgm3", None),
            cell=self.structure.get("cell_A", None),
            outfile="supercell.pdb",
        )
        pdb = PDBFile("supercell.pdb")
        self._add_bonds_to_topology(pdb, bonds)
        return pdb

    # ------------------------------------------------------------------
    # molecule representations
    # ------------------------------------------------------------------
    def get_rdkitmol(self, keys=None, il_assign=True):
        """
        Get RDKit molecule from ASE atoms

        Parameters
        ----------
        keys: list of keys to get RDKit molecule
        il_assign: bool
            If True, assign special ionic liquids to the molecule
        """
        keys = self._keys_or_all(keys)

        for key in keys:
            if self.mol_info[key]["netcharge"] is None:
                self._assign_totalcharge()
                break

        for key in keys:
            nc = self.mol_info[key]["netcharge"]
            atoms = self.mol_info[key]["aseatoms_list"][0]
            if il_assign:
                mol, mol2d, il_dict = atoms2rdkit(atoms, nc=nc, il_assign=True)
                if il_dict is not None:
                    self.mol_info[key]["metadata"].update(il_dict)
            else:
                mol, mol2d = atoms2rdkit(atoms, nc=nc, il_assign=False)
            self.mol_info[key]["rdkit"] = {"mol": mol, "mol2d": mol2d}

    def get_smiles(self, keys=None):
        """
        Get SMILES from RDKit molecule

        Parameters
        ----------
        keys: list of keys to get SMILES
        """
        for key in self._keys_or_all(keys):
            if self.mol_info[key]["rdkit"] is None:
                self.get_rdkitmol([key])
            self.mol_info[key]["SMILES"] = Chem.MolToSmiles(
                self.mol_info[key]["rdkit"]["mol"]
            )

    def get_sdf(self, keys=None):
        """
        Get SDF from RDKit molecule

        Parameters
        ----------
        keys: list of keys to get SDF
        """
        for key in self._keys_or_all(keys):
            if "mol" not in self.mol_info[key]["rdkit"]:
                self.get_rdkitmol([key])

            output_dir = f"{key}"
            self.mol_info[key]["directory"] = output_dir
            os.makedirs(output_dir, exist_ok=True)

            writer = Chem.SDWriter(os.path.join(output_dir, f"{key}.sdf"))
            writer.write(self.mol_info[key]["rdkit"]["mol"])
            writer.close()

    def get_molecule_off(self, keys=None, **kwargs):
        """
        Get OpenFF molecule from ASE atoms and save PDB file with bond information.

        Parameters
        ----------
        keys: list of keys to get OpenFF molecule
        kwargs: dict
            Additional attributes for OpenFF molecule.
        """
        for key in self._keys_or_all(keys):
            dirname = self.mol_info[key]["directory"]
            sdffile = os.path.join(dirname, f"{key}.sdf")
            self.get_sdf()

            molecule_off = Molecule.from_file(sdffile, allow_undefined_stereo=True)
            if list(self.mol_info[key]["partial_charges"]) != []:
                molecule_off.partial_charges = Quantity(
                    self.mol_info[key]["partial_charges"],
                    toolkit.unit.elementary_charge,
                )

            molecule_off.name = key
            # flag the atoms that were identified as part of a known ion
            for meta_key, meta_indices in self.mol_info[key]["metadata"].items():
                for meta_ind in meta_indices:
                    molecule_off.atoms[meta_ind].metadata[meta_key] = True

            for attr_key, attr_value in kwargs.items():
                setattr(molecule_off, attr_key, attr_value)

            self.mol_info[key]["molecule_OFF"] = molecule_off

            # Save PDB file with bond information
            molecule_off.to_file(
                os.path.join(dirname, f"{key}_bonds.pdb"), file_format="PDB"
            )

    # ------------------------------------------------------------------
    # geometry optimization
    # ------------------------------------------------------------------
    # GeoOptimizerとして別ファイルに移す案もあり
    @staticmethod
    def _resolve_geoopt_params(kwargs):
        """
        Merge the user settings into the defaults of the selected QM backend.

        Returns
        -------
        (software, params) : (str, dict)
        """
        software = kwargs.get("software")
        if software == "psi4":
            params = dict(GEOOPT_DEFAULTS["psi4"])
            params.update(kwargs)
            if params.get("method") == "wb97xd":
                params["method"] = "wb97x-d"
        elif software == "g16":
            params = dict(GEOOPT_DEFAULTS["g16"])
            params.update(kwargs)
        else:
            raise ValueError(
                f"Unknown software: {software}. Please check the software name."
            )
        params.pop("software", None)
        return software, params

    def _make_geoopt_calculator(self, key, index, atoms, software, params):
        """
        Attach a QM optimizer to ``atoms`` and return the trajectory file it
        will produce.
        """
        netcharge = self.mol_info[key]["netcharge"]
        label = f"{self.mol_info[key]['directory']}/{key}_{index}"

        if software == "g16":
            atoms.calc = Gaussian(label=label, charge=netcharge, **params)
            return atoms.calc.label + ".log"

        calc_geoopt = Psi4GeoOptimizer(
            atoms=atoms,
            method=params["method"],
            basis_set=params["basis"],
            charge=netcharge,
            multiplicity=1,
            label=f"{label}_psi4",
        )
        atoms.calc = calc_geoopt
        print(calc_geoopt.label, calc_geoopt.directory, "label and directory")
        return calc_geoopt.label + ".xyz"

    def _find_similar_conformer(self, key, index, rmsd):
        """
        Look for an already optimized conformer of the same molecule that is
        closer than ``rmsd``; its index is returned so the expensive
        optimization can be skipped.

        Returns
        -------
        (j, rmsd_ji) : (int, float) or (None, None)
        """
        networkX = self.mol_info[key]["networkX"]
        for j in range(0, index):
            rmsd_ji, _ = cast_molecules(networkX[j], networkX[index])
            print(rmsd_ji, index, j)
            if rmsd_ji < rmsd:
                return j, rmsd_ji
        return None, None

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
        software, geoopt_params = self._resolve_geoopt_params(kwargs)

        for key in self._keys_or_all(keys):
            trajectory = []
            for i, atoms in enumerate(self.mol_info[key]["aseatoms_list"]):
                atoms_tmp = atoms.copy()
                atoms_tmp.pbc = False
                atoms_tmp.cell = None
                if self.mol_info[key]["netcharge"] is None:
                    self._assign_totalcharge()

                trajectory.append(
                    self._make_geoopt_calculator(
                        key, i, atoms_tmp, software, geoopt_params
                    )
                )

                if not do_calc:
                    continue

                j, rmsd_ji = self._find_similar_conformer(key, i, rmsd)
                if j is not None:
                    # reuse the optimized geometry of the similar conformer
                    self.mol_info[key]["aseatoms_geoopt"][i] = read(
                        trajectory[j], index=-1
                    )
                    shutil.copy(trajectory[j], trajectory[i])
                    print(
                        f"Skip geometry optimization of {key}_{i}: "
                        f"RMSD = {rmsd_ji} A < {rmsd}"
                    )
                else:
                    self.mol_info[key]["aseatoms_geoopt"][i] = atoms_tmp
                    print(f"Calculating geometry optimization of {key}_{i}")
                    _ = atoms_tmp.get_potential_energy()
                    if software == "psi4":
                        atoms_tmp.positions = atoms_tmp.calc.atoms.positions
                    print(f"Finished geometry optimization of {key}_{i}")

            energies = [
                a.get_potential_energy()
                for a in self.mol_info[key]["aseatoms_geoopt"]
            ]
            self.mol_info[key]["aseatoms_stable"] = self.mol_info[key][
                "aseatoms_geoopt"
            ][np.array(energies).argmin()]

    # ------------------------------------------------------------------
    # partial charges
    # ------------------------------------------------------------------
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
        keys = self._keys_or_all(keys)

        charge_type = kwargs.get("type", "resp")
        psi4_flag = kwargs.get("software", False) == "psi4"
        params = kwargs.get("params", None)

        if charge_type not in VALID_CHARGE_TYPES:
            raise ValueError(
                f"charge_type should be one of {VALID_CHARGE_TYPES}: {charge_type}"
            )

        # assign total charge if not assigned
        if any(self.mol_info[key]["netcharge"] is None for key in keys):
            self._assign_totalcharge()

        for key in keys:
            info = self.mol_info[key]
            if info["aseatoms_stable"] is not None:
                atoms = info["aseatoms_stable"]
            else:
                atoms = info["aseatoms_list"][0]

            calculator_cls = ChargeCalculator
            if psi4_flag:
                calculator_cls = Psi4ChargeCalculator
                # Psi4 needs the OpenFF conformer to match the ASE geometry
                if info["molecule_OFF"] is None:
                    self.get_molecule_off(keys=[key])
                info["molecule_OFF"].conformers[0].magnitude[:] = atoms.positions

            info["ChargeCalc"] = calculator_cls(
                atoms,
                charge_type,
                info["netcharge"],
                key,
                directory=info["directory"],
                params=params,
            )
            info["ChargeCalc"].get_partialcharges()
            info["partial_charges"] = info["ChargeCalc"].partial_charges
            self.get_molecule_off(keys=[key], mol2file=info["ChargeCalc"].mol2file)

        if params_ff is not None:
            self._adjust_charges(params_ff)

    def _adjust_charges(self, ff_params=None):
        """
        Apply the charge scaling requested in the force field parameters and
        redistribute the rounding error so that the charges sum to the net
        charge exactly.
        """
        for key in self.mol_info.keys():
            net_charge = self.mol_info[key]["netcharge"]
            charges = np.array(
                [np.float64(ee) for ee in self.mol_info[key]["partial_charges"]]
            )

            # ions and neutral molecules are scaled by separate factors;
            # only the first key present in ff_params is honoured
            scale = None
            if ff_params is not None:
                if "charge_scale_ion" in ff_params.keys():
                    if not np.isclose(net_charge, 0.0):
                        scale = ff_params["charge_scale_ion"]
                elif "charge_scale_neutral" in ff_params.keys():
                    if np.isclose(net_charge, 0.0):
                        scale = ff_params["charge_scale_neutral"]
            if scale is not None:
                charges *= scale
                net_charge *= scale

            total_charge = np.sum(charges)
            charge_deficit = total_charge - net_charge

            if not np.isclose(charge_deficit, 0.0, atol=1e-8):
                print(
                    (
                        f"Net charge of {key} is {net_charge} and "
                        f"total charge is {total_charge}"
                    )
                )
                # spread the deficit over all atoms, then put the remainder
                # on the first atom
                charges = charges - charge_deficit / len(charges)
                charges[0] -= np.sum(charges) - net_charge

            self.mol_info[key]["partial_charges"] = charges

    # ------------------------------------------------------------------
    # dihedral scans
    # ------------------------------------------------------------------
    def get_dihedral_qm(self, keys=None, do_calc=True):
        """
        Get dihedral angles using quantum mechanical calculations

        Parameters
        ----------
        keys: list of keys to get dihedral angles
        do_calc: bool
            If True, perform dihedral angle scan calculation. Default is True.
        """
        for key in self._keys_or_all(keys):
            info = self.mol_info[key]
            if isinstance(info["aseatoms_stable"], Atoms):
                atoms = info["aseatoms_stable"]
            else:
                atoms = info["aseatoms_list"][0]

            if info["DihedCalc"] is None:
                info["DihedCalc"] = DihedralCalculator(
                    atoms=atoms, label=key, directory=info["directory"]
                )

            info["DihedCalc"].do_qmscan(do_calc=do_calc)

    def get_dihedral_ff(self, keys=None, do_calc=True):
        """
        Get dihedral angles using force field parameters

        Parameters
        ----------
        keys: list of keys to get dihedral angles
        do_calc: bool
            If True, perform dihedral angle scan calculation. Default is True.
        """
        for key in self._keys_or_all(keys):
            self.mol_info[key]["DihedCalc"].do_ffscan(do_calc=do_calc)

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
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

        # also drop a per-molecule copy next to that molecule's outputs
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

    # ------------------------------------------------------------------
    # net charge assignment
    # ------------------------------------------------------------------
    def _assign_totalcharge(self):
        """
        Guess the net charge of every molecule, then share out whatever charge
        is left over among the molecules that could not be assigned.
        """
        for key in self.mol_info.keys():
            self._assign_molecule_charge(key)
        self._distribute_unassigned_charge()

    def _assign_molecule_charge(self, key):
        """Guess the net charge of a single molecule."""
        info = self.mol_info[key]
        atoms = info["aseatoms_list"][0]

        # an odd electron count means the neutral molecule is a radical, so it
        # is more likely to be an ion whose charge we still have to identify
        openshell_flag = atoms.get_atomic_numbers().sum() % 2 == 1
        mim_flag = False
        if openshell_flag:
            mim_flag = _has_imidazolium_ring(info["networkX"][0])
            chg_cation = _count_quaternary_nitrogen(info["networkX"][0])
            if chg_cation > 0:
                info["netcharge"] = chg_cation
                openshell_flag = False

        info["symbol"] = str(atoms.symbols)
        if info["symbol"] in MONATOMIC_ION_CHARGES:
            info["netcharge"] = MONATOMIC_ION_CHARGES[info["symbol"]]
        elif mim_flag:
            info["netcharge"] = 1
        elif openshell_flag:
            info["netcharge"] = None
        elif info["netcharge"] is None:
            info["netcharge"] = 0

    def _distribute_unassigned_charge(self):
        """
        Give every molecule with an unknown net charge the share that makes the
        whole system neutral, when that share is an integer.
        """
        num_charge_none = 0
        total_charge = 0
        for info in self.mol_info.values():
            if info["netcharge"] is None:
                num_charge_none += 1 * info["Nmols"]
            else:
                total_charge += info["netcharge"] * info["Nmols"]

        if num_charge_none > 0 and total_charge % num_charge_none == 0:
            charge_per_none = int(-total_charge / num_charge_none)
            for info in self.mol_info.values():
                if info["netcharge"] is None:
                    info["netcharge"] = charge_per_none
