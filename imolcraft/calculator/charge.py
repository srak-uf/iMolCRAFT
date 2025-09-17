from imolcraft.io.mol2 import read_mol2, write_mol2
import os
import subprocess
from ase.io import write
from ase.calculators.gaussian import Gaussian
from imolcraft.io.rdkit import atoms2rdkit
from rdkit import Chem
import tempfile
from openff import toolkit
from openff.toolkit import Quantity
from openff.toolkit.topology import Molecule
from openff.recharge.esp import ESPSettings
from openff.recharge.grids import MSKGridSettings
from openff.recharge.esp.psi4 import Psi4ESPGenerator
from openff.recharge.esp.storage import MoleculeESPRecord
from openff.recharge.charges.resp.solvers import IterativeSolver
from openff.recharge.charges.resp import generate_resp_charge_parameter
from openff.recharge.charges.library import (
    LibraryChargeCollection,
    LibraryChargeGenerator,
)


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
        """
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
        self.atoms = atoms.copy()
        self.atoms.pbc = False
        self.atoms.cell = None
        self.charge_type = charge_type
        self.params = params
        self.charge = netcharge
        self.label = label
        if directory is None:
            self.directory = os.getcwd()
        else:
            self.directory = directory

        if self.params is None:
            if self.charge_type == "resp":
                self.params = resp_params
            elif self.charge_type == "am1bcc":
                pass
            else:
                raise ValueError(
                    "Unknown charge type. Supported types are: resp, am1-bcc"
                )
            self.g16 = None
        else:
            resp_params.update(self.params)
            self.params = resp_params

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
            temp_pdb_name = temp_pdb.name
            write(temp_pdb_name, self.atoms, format="pdb")
            output_mol2 = os.path.join(self.directory, self.label + "_am1bcc.mol2")
            # antechamber calculation
            cmd_antech = (
                f"antechamber -i {temp_pdb_name} -fi gout "
                f"-o {output_mol2} -fo mol2 -at sybyl "
                f"-c bcc -nc {self.charge} -pf y -dr no"
            )
            print(cmd_antech)
            output = subprocess.getoutput(cmd_antech)
            print(output)

        self.mol2file = output_mol2
        tripos_atom = read_mol2(self.mol2file)["@<TRIPOS>ATOM"]
        partialcharges = [float(atom[-1]) for atom in tripos_atom]

        return partialcharges

    def _get_resp(self):
        g16calculator = Gaussian(
            label=self.label + "_resp",
            charge=self.charge,
            **self.params,
        )
        g16calculator.directory = os.path.abspath(self.directory)
        comfile = os.path.join(self.directory, f"{self.label}_resp.com")
        logfile = os.path.join(self.directory, f"{self.label}_resp.log")
        g16calculator.write_input(self.atoms, system_changes=0)
        # g16 calculation
        cmd = f"g16 < {comfile}  > {logfile}"
        print(cmd)
        _ = subprocess.getoutput(cmd)

        # antechamber calculation
        g16logfile = os.path.abspath(g16calculator.label + ".log")
        output_mol2 = os.path.abspath(g16calculator.label + ".mol2")
        cmd_antech = (
            f"antechamber -i {g16logfile} -fi gout "
            f"-o {output_mol2} -fo mol2 -at sybyl -c resp "
            f"-nc {self.charge} -pf y -dr no"
        )
        print(cmd_antech)

        for _ in range(10):
            output = subprocess.getoutput(cmd_antech)
            if os.path.exists(output_mol2):
                print(output)
                self.mol2file = output_mol2
                tripos_atom = read_mol2(self.mol2file)["@<TRIPOS>ATOM"]
                partialcharges = [float(atom[-1]) for atom in tripos_atom]
                break
        if os.path.exists(output_mol2) is False:
            print(output)
            raise ValueError(
                "Antechamber failed to generate the mol2 file. Check the log."
            )
        else:
            # esout, punch, qout file is removed
            for f in ["esout", "punch", "qout"]:
                if os.path.exists(f):
                    os.remove(f)

        return partialcharges


class Psi4ChargeCalculator(ChargeCalculator):
    """
    A class to calculate the partial charges (RESP) of a molecule using Psi4.

    Parameters
    ----------
    molecule : openff.toolkit.topology.Molecule
        The molecule object to calculate the charge for.
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
        """
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
        mol, _, _ = atoms2rdkit(atoms, nc=netcharge, il_assign=True)
        if not os.path.exists(directory):
            os.makedirs(directory)
        output_file = os.path.join(directory, f"{label}.sdf")
        writer = Chem.SDWriter(output_file)
        writer.write(mol)
        writer.close()
        self.molecule = Molecule.from_file(output_file, allow_undefined_stereo=True)
        self.charge_type = charge_type
        self.params = params
        self.charge = netcharge
        self.label = label
        if directory is None:
            self.directory = os.getcwd()
        else:
            self.directory = directory

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
        resp_solver = IterativeSolver()
        resp_charge_parameter = generate_resp_charge_parameter(
            [qc_data_record], resp_solver
        )
        resp_charges = LibraryChargeGenerator.generate(
            self.molecule, LibraryChargeCollection(parameters=[resp_charge_parameter])
        )
        self.mol2file = os.path.join(self.directory, self.label + "_resp.mol2")
        resp_charges = resp_charges.flatten()
        self.molecule.partial_charges = Quantity(
            resp_charges, toolkit.unit.elementary_charge
        )
        write_mol2(self.mol2file, self.molecule)
        return resp_charges
