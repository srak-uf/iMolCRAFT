import copy
import os
import subprocess
import tempfile

from ase.calculators.gaussian import Gaussian
from ase.io import write
from openff import toolkit
from openff.recharge.charges.library import (
    LibraryChargeCollection,
    LibraryChargeGenerator,
)
from openff.recharge.charges.resp import generate_resp_charge_parameter
from openff.recharge.charges.resp.solvers import IterativeSolver
from openff.recharge.esp import ESPSettings
from openff.recharge.esp.psi4 import Psi4ESPGenerator
from openff.recharge.esp.storage import MoleculeESPRecord
from openff.recharge.grids import MSKGridSettings
from openff.toolkit import Quantity
from openff.toolkit.topology import Molecule
from rdkit import Chem

from imolcraft.io.mol2 import read_mol2, write_mol2
from imolcraft.io.rdkit import atoms2rdkit

resp_params = {
    "method": "hf",
    "basis": "6-31g(d)",
    # "mem": "92GB",
    # "nprocshared": 40,
    "ioplist": ["6/33=2", "6/42=6"],
    "pop": "mk",
}

psi4_resp_params = ESPSettings(
    method="hf", basis="6-31G*", grid_settings=MSKGridSettings(density=6.0)
)

#: Files antechamber leaves behind next to a RESP fit.
_RESP_SCRATCH_FILES = ("esout", "punch", "qout")

#: How many times antechamber is retried before giving up.
_ANTECHAMBER_ATTEMPTS = 10


def _merged_resp_params(params=None):
    """
    The default RESP settings with the caller's overrides applied.

    A deep copy is returned, so neither the module level defaults nor the
    settings of another calculator can be reached through the result. Merging
    with ``resp_params.update(params)`` instead would rewrite the defaults for
    every calculator built afterwards.

    Parameters
    ----------
    params : dict, optional
        Settings overriding the defaults.

    Returns
    -------
    dict
        A fresh dictionary of Gaussian keywords.
    """
    merged = copy.deepcopy(resp_params)
    if params is not None:
        merged.update(copy.deepcopy(params))
    return merged


def _charges_from_mol2(mol2file):
    """Read the charge column of the ATOM section of a mol2 file."""
    return [float(atom[-1]) for atom in read_mol2(mol2file)["@<TRIPOS>ATOM"]]


def _resolve_directory(directory):
    """The output directory, defaulting to the current working directory."""
    return os.getcwd() if directory is None else directory


class ChargeCalculator:
    """
    A class to calculate the partial charges of a molecule using different methods.
    Supported methods are: RESP, AM1-BCC.

    Parameters
    ----------
    atoms : ase.Atoms
        The atoms object to calculate the charge for.
    charge_type : str
        The type of charge calculation to perform. Supported types are: resp,
        am1-bcc
    netcharge : int
        The net charge of the molecule.
    label : str
        The label for the output files.
    directory : str, optional
        The directory to save the output files. If None, the current working
        directory is used.
    params : dict, optional
        The parameters for the charge calculation. If None, default parameters
        are used.
    """

    def __init__(
        self, atoms, charge_type, netcharge, label, directory=None, params=None
    ):
        self.atoms = atoms.copy()
        self.atoms.pbc = False
        self.atoms.cell = None
        self.charge_type = charge_type
        self.params = params
        self.charge = netcharge
        self.label = label
        self.directory = _resolve_directory(directory)

        if self.params is None:
            if self.charge_type == "resp":
                self.params = _merged_resp_params()
            elif self.charge_type == "am1bcc":
                pass
            else:
                raise ValueError(
                    "Unknown charge type. Supported types are: resp, am1-bcc"
                )
            self.g16 = None
        else:
            self.params = _merged_resp_params(self.params)

    def get_partialcharges(self):
        """
        Calculate the partial charges of the molecule using the specified method.
        """
        if self.charge_type == "resp":
            self.partial_charges = self._get_resp()
        elif self.charge_type == "am1bcc":
            self.partial_charges = self._get_am1bcc()
        else:
            raise ValueError("Unknown charge type. Supported types are: resp, am1-bcc")

    def _get_am1bcc(self):
        with tempfile.NamedTemporaryFile() as temp_pdb:
            # ASE calls the PDB writer "proteindatabank"; "pdb" is only an
            # extension alias and is rejected as a format name
            write(temp_pdb.name, self.atoms, format="proteindatabank")
            output_mol2 = os.path.join(self.directory, self.label + "_am1bcc.mol2")
            # antechamber calculation
            # -fi pdb because the input written just above is a PDB; antechamber
            # runs sqm on it to obtain the AM1 charges the BCC correction needs
            cmd_antech = (
                f"antechamber -i {temp_pdb.name} -fi pdb "
                f"-o {output_mol2} -fo mol2 -at sybyl "
                f"-c bcc -nc {self.charge} -pf y -dr no"
            )
            print(cmd_antech)
            print(subprocess.getoutput(cmd_antech))

        self.mol2file = output_mol2
        return _charges_from_mol2(self.mol2file)

    def _run_g16(self):
        """Run the Gaussian ESP calculation and return the path of its log."""
        g16calculator = Gaussian(
            label=self.label + "_resp",
            charge=self.charge,
            **self.params,
        )
        g16calculator.directory = os.path.abspath(self.directory)
        g16calculator.write_input(self.atoms, system_changes=0)

        comfile = os.path.join(self.directory, f"{self.label}_resp.com")
        logfile = os.path.join(self.directory, f"{self.label}_resp.log")
        cmd = f"g16 < {comfile}  > {logfile}"
        print(cmd)
        _ = subprocess.getoutput(cmd)
        return g16calculator.label

    def _get_resp(self):
        label = self._run_g16()

        # antechamber calculation
        g16logfile = os.path.abspath(label + ".log")
        output_mol2 = os.path.abspath(label + ".mol2")
        cmd_antech = (
            f"antechamber -i {g16logfile} -fi gout "
            f"-o {output_mol2} -fo mol2 -at sybyl -c resp "
            f"-nc {self.charge} -pf y -dr no"
        )
        print(cmd_antech)

        # antechamber occasionally fails to produce the mol2, so retry
        for _ in range(_ANTECHAMBER_ATTEMPTS):
            output = subprocess.getoutput(cmd_antech)
            if os.path.exists(output_mol2):
                print(output)
                self.mol2file = output_mol2
                partialcharges = _charges_from_mol2(self.mol2file)
                break

        if os.path.exists(output_mol2) is False:
            print(output)
            raise ValueError(
                "Antechamber failed to generate the mol2 file. Check the log."
            )
        else:
            # esout, punch, qout file is removed
            for f in _RESP_SCRATCH_FILES:
                if os.path.exists(f):
                    os.remove(f)

        return partialcharges


class Psi4ChargeCalculator(ChargeCalculator):
    """
    A class to calculate the partial charges (RESP) of a molecule using Psi4.

    Parameters
    ----------
    atoms : ase.Atoms
        The atoms object to calculate the charge for.
    charge_type : str
        The type of charge calculation to perform. Supported types are: resp
    netcharge : int
        The net charge of the molecule.
    label : str
        The label for the output files.
    directory : str, optional
        The directory to save the output files. If None, the current working
        directory is used.
    params : dict, optional
        The parameters for the charge calculation. If None, default parameters
        are used.
    """

    def __init__(
        self, atoms, charge_type, netcharge, label, directory=None, params=None
    ):
        # resolve the directory first: the SDF below is written into it, so it
        # must not still be None by then
        self.directory = _resolve_directory(directory)
        # openff-recharge works on an OpenFF molecule, so go through RDKit and
        # an SDF file to carry the geometry and the perceived bonds across
        self.molecule = self._molecule_from_atoms(
            atoms, netcharge, label, self.directory
        )
        self.charge_type = charge_type
        self.params = params
        self.charge = netcharge
        self.label = label

        if self.params is None:
            if self.charge_type == "resp":
                self.params = psi4_resp_params
            else:
                raise ValueError("Unknown charge type. Supported types are: resp")
            self.g16 = None
        else:
            self.params = ESPSettings(
                method=params["method"],
                basis=params["basis"],
                grid_settings=MSKGridSettings(density=6.0),
            )

    @staticmethod
    def _molecule_from_atoms(atoms, netcharge, label, directory):
        mol, _, _ = atoms2rdkit(atoms, nc=netcharge, il_assign=True)
        os.makedirs(directory, exist_ok=True)

        output_file = os.path.join(directory, f"{label}.sdf")
        writer = Chem.SDWriter(output_file)
        writer.write(mol)
        writer.close()
        return Molecule.from_file(output_file, allow_undefined_stereo=True)

    def get_partialcharges(self):
        """
        Calculate the partial charges of the molecule using the specified method.
        """
        if self.charge_type == "resp":
            self.partial_charges = self._get_resp()
        else:
            raise ValueError("Unknown charge type. Supported types are: resp")

    def _get_resp(self):
        conformer, grid, esp, electric_field = Psi4ESPGenerator.generate(
            self.molecule, self.molecule.conformers[0], self.params, minimize=False
        )
        qc_data_record = MoleculeESPRecord.from_molecule(
            self.molecule, conformer, grid, esp, None, self.params
        )
        resp_charge_parameter = generate_resp_charge_parameter(
            [qc_data_record], IterativeSolver()
        )
        resp_charges = LibraryChargeGenerator.generate(
            self.molecule, LibraryChargeCollection(parameters=[resp_charge_parameter])
        ).flatten()

        self.molecule.partial_charges = Quantity(
            resp_charges, toolkit.unit.elementary_charge
        )
        self.mol2file = os.path.join(self.directory, self.label + "_resp.mol2")
        write_mol2(self.mol2file, self.molecule)
        return resp_charges
