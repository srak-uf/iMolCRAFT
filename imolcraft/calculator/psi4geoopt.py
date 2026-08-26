import multiprocessing
import os

import cclib
import psutil
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

#: Psi4 input template. ``task`` is the driver call, which differs between a
#: geometry optimization and a single point.
_PSI4_INPUT_TEMPLATE = """
molecule {{
{geom}
}}

memory {memory} GB

set_num_threads({threads})

set {{
   basis {basis_set}
   maxiter 128
   g_convergence gau
   dft_spherical_points 590
   dft_radial_points 99
   geom_maxiter 128
}}

{task}
"""


class Psi4GeoOptimizer(Calculator):
    implemented_properties = ["energy"]

    def __init__(
        self,
        atoms,
        method,
        basis_set,
        charge=0,
        multiplicity=1,
        label="psi4_geoopt",
    ):
        Calculator.__init__(self, atoms=atoms)
        self.results = {}
        self.atoms = atoms
        self.method = method
        self.basis_set = basis_set
        self.nc = charge
        self.multiplicity = multiplicity
        # give psi4 the available memory rounded down to an even number of GB
        self.memory = int(psutil.virtual_memory().available / (1024**3))
        if self.memory % 2 != 0:
            self.memory -= 1
        # the largest power of two not exceeding the core count
        self.threads = 2 ** (multiprocessing.cpu_count().bit_length() - 1)  # 2^n
        self.label = label

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        inputfile = f"{self.label}.psi4in"
        outputfile = f"{self.label}.psi4out"
        self.generate_input(inputfile)

        print(f"psi4 {inputfile}  {outputfile}")
        os_value = os.system(f"psi4 {inputfile}  {outputfile}")
        if os_value != 0:
            raise RuntimeError(f"Psi4 optimization failed with exit code {os_value}")

        _clean_psi4output(outputfile)
        data = cclib.ccopen(outputfile).parse()
        energy = data.scfenergies[-1]

        atoms_geoopt = cclib.bridge.makease(
            data.converged_geometries[-1], data.atomnos
        )
        atoms_geoopt.calc = SinglePointCalculator(energy=energy, atoms=atoms_geoopt)
        self.atoms = atoms_geoopt
        self.results["energy"] = energy
        write(f"{self.label}.xyz", atoms_geoopt)

    def _molecule_block(self):
        """The ``molecule`` block: charge / multiplicity, geometry and settings."""
        geomline = "{}\t{:.15f}\t{:.15f}\t{:.15f}"
        lines = [f"{self.nc} {self.multiplicity}"]
        lines += [geomline.format(atom.symbol, *atom.position) for atom in self.atoms]
        lines.append("symmetry {}".format("c1"))
        lines.append("units angstrom")
        return "\n".join(lines)

    def generate_input(self, filename):
        # a lone atom has no geometry to optimize, so ask for a single point
        if len(self.atoms) > 1:
            task = f"optimize('{self.method}', engine='geometric')"
        else:
            task = f"energy('{self.method}')"

        with open(filename, "w") as f:
            f.write(
                _PSI4_INPUT_TEMPLATE.format(
                    geom=self._molecule_block(),
                    memory=self.memory,
                    threads=self.threads,
                    basis_set=self.basis_set,
                    task=task,
                )
            )


def _clean_psi4output(filename):
    """Drop the ``Git:`` banner line, which cclib cannot parse."""
    with open(filename, "r") as f:
        lines = f.readlines()
    with open(filename, "w") as f:
        f.writelines(
            line for line in lines if not line.lstrip().startswith("Git:")
        )
