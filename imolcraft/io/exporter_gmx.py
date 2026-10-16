import copy
import re

import openmm
from openmm import XmlSerializer, app
from openmm.app import PDBFile
from parmed.openmm import load_topology

#: Tolerance on the weights of an averaging site summing to one, which the
#: GROMACS constructions (virtual_sites2/3, funct 1) assume.
_WEIGHT_SUM_TOL = 1e-6

#: Columns holding atom indices in each section of a [ moleculetype ] that
#: parmed writes (None: every column of the line is an index).
_INDEX_COLUMNS = {
    "atoms": 1,
    "bonds": 2,
    "pairs": 2,
    "angles": 3,
    "dihedrals": 4,
    "cmap": 5,
    "settles": 1,
    "exclusions": None,
}


def exporter_gmx(pdb, system, filename):
    """
    Export a system to GROMACS format.

    Virtual sites of the OpenMM System are written as massless particles
    placed by ``[ virtual_sites2 ]`` (``TwoParticleAverageSite``) or
    ``[ virtual_sites3 ]`` (``ThreeParticleAverageSite``: funct 1,
    ``OutOfPlaneSite``: funct 4), with the exclusions of the System listed in
    ``[ exclusions ]``, so that GROMACS evaluates the same model as OpenMM.
    ``LocalCoordinatesSite`` has no GROMACS counterpart here and is refused.

    Parameters
    ----------
    pdb : str
        The path to the PDB file.
        The pdb file should contain the topology information, with the
        virtual sites (extra particles) when the System has any.
    system : str
        The path to the system xml file of openmm.
    filename : str
        The filename of the output file (filename.top, filename.gro).
    """
    pdb_omm = PDBFile(pdb)
    with open(system) as f:
        system_omm = XmlSerializer.deserialize(f.read())

    vsites = _virtual_sites(system_omm)
    if not vsites:
        parm_top = load_topology(
            topology=pdb_omm.topology, system=system_omm, xyz=pdb_omm.positions
        )
        parm_top.save(f"{filename}.top", overwrite=True)
        parm_top.save(f"{filename}.gro", overwrite=True)
        return

    if pdb_omm.topology.getNumAtoms() != system_omm.getNumParticles():
        raise ValueError(
            f"{pdb} has {pdb_omm.topology.getNumAtoms()} atoms but the System "
            f"has {system_omm.getNumParticles()} particles; the PDB must "
            "contain the virtual sites (Modeller.addExtraParticles)."
        )

    # parmed turns an element-less atom into an ExtraPoint, whose frame it
    # guesses from the geometry and whose exclusions it writes from the bond
    # graph; neither matches OpenMM's virtual sites. The sites are handed to
    # parmed as ordinary massless atoms tied to their first constructing
    # particle, which keeps each site in its molecule; the tie and the
    # virtual-site sections are then rewritten below from the System itself.
    topology = _topology_with_tied_vsites(pdb_omm.topology, vsites)
    parm_top = load_topology(
        topology=topology, system=system_omm, xyz=pdb_omm.positions
    )
    # The sites' atom types were named after the placeholder element (H...);
    # rename them VS1, VS2, ... and make them massless.
    real = [atom for i, atom in enumerate(parm_top.atoms) if i not in vsites]
    taken = {atom.type for atom in real}
    real_types = {id(atom.atom_type) for atom in real}
    renamed = {}
    copies = {}
    for index in vsites:
        atom = parm_top.atoms[index]
        atom.mass = 0.0
        atom.atomic_number = 0
        atype = atom.atom_type
        if atype is None or isinstance(atype, str):
            continue
        if id(atype) in real_types:
            # Shared with a real atom of the same element and LJ: give the
            # sites their own copy rather than renaming the real atom's type.
            if id(atype) not in copies:
                copies[id(atype)] = copy.copy(atype)
            atype = atom.atom_type = copies[id(atype)]
        if id(atype) not in renamed:
            n = len(renamed) + 1
            while f"VS{n}" in taken:
                n += 1
            renamed[id(atype)] = f"VS{n}"
            taken.add(f"VS{n}")
            atype.name = renamed[id(atype)]
            atype.mass = 0.0
            atype.atomic_number = 0
        atom.type = renamed[id(atype)]
    parm_top.save(f"{filename}.top", overwrite=True)
    parm_top.save(f"{filename}.gro", overwrite=True)

    with open(f"{filename}.top") as handle:
        text = handle.read()
    text = _add_vsite_sections(text, system_omm, vsites)
    with open(f"{filename}.top", "w") as handle:
        handle.write(text)


def _virtual_sites(system):
    """``{particle index: VirtualSite}`` of every virtual site of ``system``."""
    vsites = {}
    for i in range(system.getNumParticles()):
        if system.isVirtualSite(i):
            site = system.getVirtualSite(i)
            if isinstance(site, openmm.LocalCoordinatesSite):
                raise ValueError(
                    f"Particle {i} is a LocalCoordinatesSite, which the GROMACS "
                    "exporter does not translate; only average2, average3 and "
                    "outOfPlane virtual sites are supported."
                )
            if not isinstance(
                site,
                (
                    openmm.TwoParticleAverageSite,
                    openmm.ThreeParticleAverageSite,
                    openmm.OutOfPlaneSite,
                ),
            ):
                raise ValueError(
                    f"Particle {i} is a {type(site).__name__}, which the "
                    "GROMACS exporter does not translate."
                )
            vsites[i] = site
    return vsites


def _site_particles(site):
    return [site.getParticle(k) for k in range(site.getNumParticles())]


def _topology_with_tied_vsites(topology, vsites):
    """
    Copy of ``topology`` where each virtual site has an element (so parmed
    keeps it an ordinary atom) and a bond to its first constructing particle.
    """
    new = app.Topology()
    new.setPeriodicBoxVectors(topology.getPeriodicBoxVectors())
    atoms = []
    for chain in topology.chains():
        new_chain = new.addChain(chain.id)
        for residue in chain.residues():
            new_res = new.addResidue(
                residue.name, new_chain, residue.id, residue.insertionCode
            )
            for atom in residue.atoms():
                element = atom.element
                if atom.index in vsites:
                    # Any element will do: the mass and atomic number are reset
                    # to zero once parmed has built the structure.
                    element = app.element.hydrogen
                atoms.append(new.addAtom(atom.name, element, new_res, atom.id))
    for bond in topology.bonds():
        new.addBond(atoms[bond[0].index], atoms[bond[1].index])
    for index, site in vsites.items():
        new.addBond(atoms[index], atoms[site.getParticle(0)])
    return new


def _site_line(local, site):
    """``[ virtual_sites2/3 ]`` section name and line for one site."""
    parts = [local[p] for p in _site_particles(site)]
    if isinstance(site, openmm.TwoParticleAverageSite):
        w0, w1 = site.getWeight(0), site.getWeight(1)
        _check_weights(w0 + w1, site)
        # GROMACS 2 (funct 1): x = (1 - a) x_i + a x_j
        return "virtual_sites2", (local["site"], *parts, 1, f"{w1:.10f}")
    if isinstance(site, openmm.ThreeParticleAverageSite):
        w0, w1, w2 = site.getWeight(0), site.getWeight(1), site.getWeight(2)
        _check_weights(w0 + w1 + w2, site)
        # GROMACS 3 (funct 1): x = x_i + a (x_j - x_i) + b (x_k - x_i)
        return "virtual_sites3", (local["site"], *parts, 1, f"{w1:.10f}", f"{w2:.10f}")
    # OutOfPlaneSite: x = x_1 + w12 r_12 + w13 r_13 + wc (r_12 x r_13), which is
    # GROMACS 3out (funct 4) with a = w12, b = w13, c = wc (nm^-1).
    return "virtual_sites3", (
        local["site"], *parts, 4,
        f"{site.getWeight12():.10f}", f"{site.getWeight13():.10f}",
        f"{site.getWeightCross():.10f}",
    )


def _check_weights(total, site):
    if abs(total - 1.0) > _WEIGHT_SUM_TOL:
        raise ValueError(
            f"The weights of a {type(site).__name__} sum to {total}, not 1; "
            "GROMACS places averaging sites with weights summing to one."
        )


def _exclusion_partners(system):
    """``{particle: set of particles}`` excluded from the normal nonbonded."""
    partners = {}
    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            for k in range(force.getNumExceptions()):
                i, j = force.getExceptionParameters(k)[:2]
                partners.setdefault(i, set()).add(j)
                partners.setdefault(j, set()).add(i)
    return partners


def _parse_molecules(lines):
    """
    The ``(moleculetype name, count)`` entries of ``[ molecules ]`` and the
    atom count of each ``[ moleculetype ]``.
    """
    natoms = {}
    molecules = []
    section = None
    current = None
    for line in lines:
        stripped = line.split(";", 1)[0].strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = re.match(r"\[\s*(\w+)\s*\]", stripped)
        if match:
            section = match.group(1)
            continue
        if section == "moleculetype":
            current = stripped.split()[0]
            natoms[current] = 0
        elif section == "atoms" and current is not None:
            natoms[current] += 1
        elif section == "molecules":
            name, count = stripped.split()[:2]
            molecules.append((name, int(count)))
    return natoms, molecules


def _add_vsite_sections(text, system, vsites):
    """
    Add the virtual sites to the ``.top`` parmed wrote from the tied topology.

    In each ``[ moleculetype ]`` holding virtual sites: the tie of a site to
    its constructing particle becomes a ``funct 5`` connection (no
    interaction), and ``[ virtual_sites2 ]`` / ``[ virtual_sites3 ]`` and an
    ``[ exclusions ]`` listing the System's exceptions of each site are
    appended. Each copy of a molecule type must place its sites the same way.
    """
    lines = text.splitlines()
    natoms, molecules = _parse_molecules(lines)
    exclusions = _exclusion_partners(system)

    # First particle of every molecule, and per molecule type the local
    # description of its sites (identical in every copy).
    per_type = {}
    offset = 0
    for name, count in molecules:
        n = natoms[name]
        for _ in range(count):
            first, last = offset, offset + n
            sites = {}
            for index in range(first, last):
                if index not in vsites:
                    continue
                site = vsites[index]
                particles = _site_particles(site)
                partners = exclusions.get(index, set())
                outside = [p for p in (*particles, *partners) if not first <= p < last]
                if outside:
                    raise ValueError(
                        f"Virtual site {index} is tied to particles {outside} "
                        "outside its molecule, which GROMACS cannot express."
                    )
                local = {p: p - first + 1 for p in particles}
                local["site"] = index - first + 1
                sites[index - first + 1] = (
                    _site_line(local, site),
                    particles[0] - first + 1,
                    sorted(p - first + 1 for p in partners),
                )
            if name in per_type and per_type[name] != sites:
                raise ValueError(
                    f"The copies of molecule type {name} place their virtual "
                    "sites differently."
                )
            per_type[name] = sites
            offset = last

    out = []
    current = None
    section = None

    def flush(name):
        sites = per_type.get(name)
        if not sites:
            return
        for kind in ("virtual_sites2", "virtual_sites3"):
            entries = [s[0][1] for s in sites.values() if s[0][0] == kind]
            if entries:
                out.append(f"[ {kind} ]")
                out.extend(" ".join(str(v) for v in e) for e in entries)
                out.append("")
        out.append("[ exclusions ]")
        for site_local, (_, _, partners) in sorted(sites.items()):
            out.append(" ".join(str(v) for v in (site_local, *partners)))
        out.append("")

    for line in lines:
        stripped = line.split(";", 1)[0].strip()
        match = re.match(r"\[\s*(\w+)\s*\]", stripped)
        if match:
            new_section = match.group(1)
            if new_section in ("moleculetype", "system") and current is not None:
                flush(current)
                current = None
            section = new_section
            out.append(line)
            continue
        if section == "moleculetype" and stripped:
            current = stripped.split()[0]
        elif section == "bonds" and stripped and per_type.get(current):
            fields = stripped.split()
            ai, aj = int(fields[0]), int(fields[1])
            sites = per_type[current]
            tie = (ai in sites and sites[ai][1] == aj) or (
                aj in sites and sites[aj][1] == ai
            )
            if tie:
                # The tie parmed wrote without parameters: a connection, no force.
                out.append(f"{ai:7d} {aj:6d}     5")
                continue
        out.append(line)
    if current is not None:
        flush(current)
    return "\n".join(out) + "\n"
