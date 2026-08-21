from .exporter_gmx import exporter_gmx
from .exporter_lmp import exporter_lmp

#: Supported output formats and the function writing each one.
_EXPORTERS = {
    "gmx": exporter_gmx,
    "lmp": exporter_lmp,
}


def exporter(pdb, system, filename, fmt):
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
    fmt : str
        The format to export to ('gmx' for GROMACS, 'lmp' for LAMMPS).
        Named ``fmt`` rather than ``format`` so that the builtin of that name
        stays reachable inside this function.
    """
    if fmt not in _EXPORTERS:
        raise ValueError(
            "Unsupported format. Use 'gmx' for GROMACS or 'lmp' for LAMMPS."
        )
    _EXPORTERS[fmt](pdb, system, filename)
