from ase.io import read, write
from ase.calculators.calculator import Calculator, all_changes
import cclib
from ase.units import Hartree
from ase.calculators.singlepoint import SinglePointCalculator
import os
import psutil
import multiprocessing

class Psi4GeoOptimizer(Calculator):
    implemented_properties = ['energy']
    def __init__(self, atoms, method, basis_set, charge=0, multiplicity=1, label="psi4_geoopt"):
        Calculator.__init__(self, atoms=atoms)
        self.results = {}
        self.atoms = atoms
        self.method = method
        self.basis_set = basis_set
        self.nc = charge
        self.multiplicity = multiplicity
        self.memory = int(psutil.virtual_memory().available/(1024 ** 3))
        if self.memory % 2 != 0:
            self.memory -= 1
        self.threads = 2 ** (multiprocessing.cpu_count().bit_length() - 1) #2^n
        self.label = label
    
    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        self.generate_input(f'{self.label}.psi4in')
        print(f'psi4 {self.label}.psi4in  {self.label}.psi4out')
        os_value = os.system(f'psi4 {self.label}.psi4in  {self.label}.psi4out')
        if os_value != 0:
            raise RuntimeError(f'Psi4 optimization failed with exit code {os_value}')
        
        _clean_psi4output(f'{self.label}.psi4out')
        parser = cclib.ccopen(f'{self.label}.psi4out')
        data = parser.parse()
        atomcoords = data.converged_geometries[-1]
        numbers = data.atomnos
        energy = data.scfenergies[-1]
        atoms_geoopt = cclib.bridge.makease(atomcoords, numbers)
        self.atoms = atoms_geoopt
        atoms_geoopt.calc = SinglePointCalculator(energy=energy, atoms=atoms_geoopt)
        self.results['energy'] = energy
        write(f'{self.label}.xyz', atoms_geoopt)

    def generate_input(self, filename):
        geomline = '{}\t{:.15f}\t{:.15f}\t{:.15f}'
        geom = [geomline.format(atom.symbol, *atom.position) for atom in self.atoms]
        geom.append('symmetry {}'.format("c1"))
        geom.append('units angstrom')
        geom.insert(0, f'{self.nc} {self.multiplicity}')
        geom = '\n'.join(geom)
        if len(self.atoms) > 1:
            template = f"""
molecule {{
{geom}
}}

memory {self.memory} GB

set_num_threads({self.threads})

set {{
   basis {self.basis_set}
   maxiter 128
   g_convergence gau
   dft_spherical_points 590
   dft_radial_points 99
   geom_maxiter 128
}}

optimize('{self.method}', engine='geometric')
"""
        else:
            template = f"""
molecule {{
{geom}
}}

memory {self.memory} GB

set_num_threads({self.threads})

set {{
basis {self.basis_set}
maxiter 128
g_convergence gau
dft_spherical_points 590
dft_radial_points 99
geom_maxiter 128
}}

energy('{self.method}')
"""
        with open(filename, 'w') as f:
            f.write(template)
            
def _clean_psi4output(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    with open(filename, 'w') as f:
        for line in lines:
            if not line.lstrip().startswith("Git:"):
                f.write(line)
                