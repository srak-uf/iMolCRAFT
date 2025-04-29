from openmm import XmlSerializer
from openmm.app import PDBFile
from parmed.openmm import load_topology

def exporter_gmx(pdb,
                 system,
                 filename):
    """
    Export a system to GROMACS format.
    Parameters
    ----------
    pdb : str
        The path to the PDB file. 
        The pdb file should contain the topology information.
    system : str
        The path to the system xml file of openmm.
    filename : str
        The filename of the output file (***.top, ***.gro).
    """
    pdb_omm = PDBFile(pdb)
    system_omm = XmlSerializer.deserialize(open(system).read())
    parm_top = load_topology(topology=pdb_omm.topology, system=system_omm, xyz=pdb_omm.positions)
    parm_top.save(f'{filename}.top', overwrite=True)
    parm_top.save(f'{filename}.gro', overwrite=True)
