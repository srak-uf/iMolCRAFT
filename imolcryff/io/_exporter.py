from .exporter_gmx import exporter_gmx
from .exporter_lmp import exporter_lmp

def exporter(pdb, system, filename, format):
    """
    Export a system to the specified format.
    
    Parameters
    ----------
    pdb : str
        The path to the PDB file. 
        The pdb file should contain the topology information.
    system : str
        The path to the system xml file of openmm.
    filename : str
        The filename of the output file (***.top, ***.gro or ***.data).
    format : str
        The format to export to ('gmx' for GROMACS, 'lmp' for LAMMPS).
    """
    if format == 'gmx':
        exporter_gmx(pdb, system, filename)
    elif format == 'lmp':
        exporter_lmp(pdb, system, filename)
    else:
        raise ValueError("Unsupported format. Use 'gmx' for GROMACS or 'lmp' for LAMMPS.")
