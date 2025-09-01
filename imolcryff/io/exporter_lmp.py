from pathlib import Path
from typing import IO
from ase.geometry import cell_to_cellpar
import numpy
import os
from openmm import XmlSerializer
from openmm.app import PDBFile
from openff.toolkit.topology.molecule import Atom, unit
from openff.toolkit import Topology, Molecule
from openff.interchange import Interchange
from openff.interchange.interop.lammps.export import to_lammps
from openff.interchange.interop import openmm
from openff.interchange.interop.lammps.export.export import (
    _write_pair_coeffs,
    _write_bond_coeffs,
    _write_angle_coeffs,
    _write_proper_coeffs,
    _write_improper_coeffs,
    _write_atoms,
    _write_bonds,
    _write_angles,
    _write_propers,
    _write_impropers,
)


def exporter_lmp(pdb, system, filename):
    """
    Export a system to LAMMPS format.

    Parameters
    ----------
    pdb : str
        The path to the PDB file.
        The pdb file should contain the topology information.
    system : str
        The path to the system xml file of openmm.
    filename : str
        The filename of the output file (filename.data).
    """
    pdb_omm = PDBFile(pdb)
    system_omm = XmlSerializer.deserialize(open(system).read())
    res_picked = []
    molecules_off = []
    for r in pdb_omm.topology.residues():
        new_res = False
        if r.name not in res_picked:
            new_res = True
            res_picked.append(r.name)
        if new_res:
            molecules_off.append(Molecule())
            atom_index = []
            for atom in r.atoms():
                molecules_off[-1].add_atom(atom.element.atomic_number, 0, False)
                atom_index.append(atom.index)
            for b in r.bonds():
                b0_idx = atom_index.index(b[0].index)
                b1_idx = atom_index.index(b[1].index)
                molecules_off[-1].add_bond(
                    b0_idx, b1_idx, bond_order=1, is_aromatic=False
                )
    topology_off = Topology.from_openmm(
        pdb_omm.topology,
        unique_molecules=molecules_off,
        positions=pdb_omm.getPositions(),
    )
    os.environ["INTERCHANGE_EXPERIMENTAL"] = "1"
    a = openmm.from_openmm(
        system=system_omm, topology=topology_off, positions=pdb_omm.getPositions()
    )
    # to_lammps(a, f"{filename}.data")
    to_lammps_non_rectangular(a, f"{filename}.data")


def to_lammps_non_rectangular(interchange: Interchange, file_path: Path | str):
    """Write an Interchange object to a LAMMPS data file."""
    if isinstance(file_path, str):
        path = Path(file_path)
    if isinstance(file_path, Path):
        path = file_path

    n_atoms = interchange.topology.n_atoms
    if "Bonds" in interchange.collections:
        n_bonds = len(interchange["Bonds"].key_map.keys())
    else:
        n_bonds = 0
    if "Angles" in interchange.collections:
        n_angles = len(interchange["Angles"].key_map.keys())
    else:
        n_angles = 0
    if "ProperTorsions" in interchange.collections:
        n_propers = len(interchange["ProperTorsions"].key_map.keys())
    else:
        n_propers = 0
    if "ImproperTorsions" in interchange.collections:
        n_impropers = len(interchange["ImproperTorsions"].key_map.keys())
    else:
        n_impropers = 0

    with open(path, "w") as lmp_file:
        lmp_file.write("Title\n\n")

        lmp_file.write(f"{n_atoms} atoms\n")
        lmp_file.write(f"{n_bonds} bonds\n")
        lmp_file.write(f"{n_angles} angles\n")
        lmp_file.write(f"{n_propers} dihedrals\n")
        lmp_file.write(f"{n_impropers} impropers\n")

        lmp_file.write(f"\n{len(interchange['vdW'].potentials)} atom types")
        if n_bonds > 0:
            lmp_file.write(f"\n{len(interchange['Bonds'].potentials)} bond types")
        if n_angles > 0:
            lmp_file.write(f"\n{len(interchange['Angles'].potentials)} angle types")
        if n_propers > 0:
            lmp_file.write(
                f"\n{len(interchange['ProperTorsions'].potentials)} dihedral types",
            )
        if n_impropers > 0:
            lmp_file.write(
                f"\n{len(interchange['ImproperTorsions'].potentials)} improper types",
            )

        lmp_file.write("\n")

        # write types section

        non_rectangular_flag = False
        x_min, y_min, z_min = numpy.min(
            interchange.positions.to(unit.angstrom),
            axis=0,
        ).magnitude
        if interchange.box is None:
            L_x, L_y, L_z = 100, 100, 100
        elif (interchange.box.m == numpy.diag(numpy.diagonal(interchange.box.m))).all():
            L_x, L_y, L_z = numpy.diag(interchange.box.to(unit.angstrom).magnitude)
        else:
            abc_alphabetagamma = cell_to_cellpar(interchange.box.to(unit.angstrom).m)
            a = abc_alphabetagamma[0]
            b = abc_alphabetagamma[1]
            c = abc_alphabetagamma[2]
            alpha_deg = abc_alphabetagamma[3]
            beta_deg = abc_alphabetagamma[4]
            gamma_deg = abc_alphabetagamma[5]
            lx = a
            xy = b * numpy.cos(numpy.deg2rad(gamma_deg))
            xz = c * numpy.cos(numpy.deg2rad(beta_deg))
            ly = numpy.sqrt(b**2 - xy**2)
            yz = (b * c * numpy.cos(numpy.deg2rad(alpha_deg)) - xy * xz) / ly
            lz = numpy.sqrt(c**2 - xz**2 - yz**2)
            non_rectangular_flag = True

        if non_rectangular_flag == False:
            lmp_file.write(
                "{:.10g} {:.10g} xlo xhi\n"
                "{:.10g} {:.10g} ylo yhi\n"
                "{:.10g} {:.10g} zlo zhi\n".format(
                    x_min,
                    x_min + L_x,
                    y_min,
                    y_min + L_y,
                    z_min,
                    z_min + L_z,
                ),
            )
            lmp_file.write("0.0 0.0 0.0 xy xz yz\n")
        else:
            lmp_file.write(
                "{:.10g} {:.10g} xlo xhi\n"
                "{:.10g} {:.10g} ylo yhi\n"
                "{:.10g} {:.10g} zlo zhi\n".format(
                    x_min,
                    x_min + lx,
                    y_min,
                    y_min + ly,
                    z_min,
                    z_min + lz,
                ),
            )
            lmp_file.write("{:.10g} {:.10g} {:.10g} xy xz yz\n".format(xy, xz, yz))

        lmp_file.write("\nMasses\n\n")

        vdw_handler = interchange["vdW"]
        atom_type_map = dict(enumerate(vdw_handler.potentials))
        key_map_inv = dict({v: k for k, v in vdw_handler.key_map.items()})

        for atom_type_idx, smirks in atom_type_map.items():
            # Find just one topology atom matching this SMIRKS by vdW
            matched_atom_idx = key_map_inv[smirks].atom_indices[0]
            matched_atom = interchange.topology.atom(matched_atom_idx)
            mass = matched_atom.mass.m

            lmp_file.write(f"{atom_type_idx + 1:d}\t{mass:.8g}\n")

        lmp_file.write("\n\n")

        _write_pair_coeffs(
            lmp_file=lmp_file,
            interchange=interchange,
            atom_type_map=atom_type_map,
        )

        if n_bonds > 0:
            _write_bond_coeffs(lmp_file=lmp_file, interchange=interchange)
        if n_angles > 0:
            _write_angle_coeffs(lmp_file=lmp_file, interchange=interchange)
        if n_propers > 0:
            _write_proper_coeffs(lmp_file=lmp_file, interchange=interchange)
        if n_impropers > 0:
            _write_improper_coeffs(lmp_file=lmp_file, interchange=interchange)

        _write_atoms(
            lmp_file=lmp_file,
            interchange=interchange,
            atom_type_map=atom_type_map,
        )
        if n_bonds > 0:
            _write_bonds(lmp_file=lmp_file, interchange=interchange)
        if n_angles > 0:
            _write_angles(lmp_file=lmp_file, interchange=interchange)
        if n_propers > 0:
            _write_propers(lmp_file=lmp_file, interchange=interchange)
        if n_impropers > 0:
            _write_impropers(lmp_file=lmp_file, interchange=interchange)
