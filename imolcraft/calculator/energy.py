from openmm import Platform
from openmm import app
from openmm import unit
from openmm import LangevinMiddleIntegrator
from dmff import Hamiltonian
import mdtraj
import jax.numpy as jnp
from dmff.common.nblist import NeighborListFreud
from imolcraft.trainer.dmff_utils import (
    get_chgparams_from_rescharges,
    get_rescharges_from_residues,
)


def get_omm_potential_energy(pdbfile,
                             xmlfile,
                             cutoff_condition,
                             dispersion_correction=False,
                             platform_name='CPU',
                             nonbonded_cutoff=1.2,
                             ethresh=0.0005):
    """
    Calculate potential energy of a molecular system using OpenMM.

    This function loads a PDB structure and OpenMM-compatible force field XML,
    creates a System object, and computes the total potential energy including
    all force contributions (bonded, nonbonded, etc.).

    Parameters
    ----------
    pdbfile : str
        Path to the input PDB file containing the molecular structure.
    xmlfile : str
        Path to the OpenMM force field XML file.
    cutoff_condition : openmm.app.NonbondedMethod
        Nonbonded method specification (e.g., app.PME, app.CutoffPeriodic, 
        app.NoCutoff).
    dispersion_correction : bool, optional
        Whether to apply long-range dispersion correction, by default False.
    platform_name : str, optional
        Name of the OpenMM platform to use ('CPU', 'CUDA', 'OpenCL'), by default 'CPU'.
    nonbonded_cutoff : float, optional
        Nonbonded cutoff distance in nanometers, by default 1.2.
    ethresh : float, optional
        Ewald error tolerance for PME calculations. Controls the precision of 
        electrostatic interactions. Internally, this is passed to OpenMM as 
        'ewaldErrorTolerance'. Default 0.0005 matches OpenMM default behavior.

    Returns
    -------
    float
        Total potential energy in kJ/mol.

    Notes
    -----
    - Each force is assigned to a separate force group for individual energy tracking
    - Supported platforms depend on OpenMM installation and available hardware
    """
    pdb = app.PDBFile(pdbfile)
    ff = app.ForceField(xmlfile)
    system = ff.createSystem(pdb.topology,
                             nonbondedMethod=cutoff_condition,
                             nonbondedCutoff=nonbonded_cutoff * unit.nanometer,
                             useDispersionCorrection=dispersion_correction,
                             ewaldErrorTolerance=ethresh)
    for i, f in enumerate(system.getForces()):
        f.setForceGroup(i)
    integrator = LangevinMiddleIntegrator(300*unit.kelvin,
                                          1/unit.picosecond,
                                          0.004*unit.picoseconds)
    platform = Platform.getPlatformByName(platform_name)
    simulation = app.Simulation(pdb.topology, system, integrator, platform)
    simulation.context.setPositions(pdb.positions)
    print(pdb.positions[0])
    state = simulation.context.getState(getEnergy=True)

    # Calculate total energy by summing all force group energies
    total_energy = 0.0
    for i, f in enumerate(system.getForces()):
        state = simulation.context.getState(getEnergy=True, groups={i})
        energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        print(f.getName(), energy, "kJ/mol")
        total_energy += energy

    return float(total_energy)


def get_dmff_potential_energy(pdbfile,
                              xmlfile,
                              cutoff_condition=app.PME,
                              dispersion_correction=False,
                              nonbonded_cutoff=1.2,
                              ethresh=0.0005):
    """
    Calculate potential energy of a molecular system using DMFF (Differentiable Molecular Force Field).

    This function loads a PDB structure and OpenMM-compatible force field XML,
    creates a differentiable potential object using DMFF backend, and computes
    the total potential energy. The calculation preserves JAX differentiability
    for automatic differentiation.

    Parameters
    ----------
    pdbfile : str
        Path to the input PDB file containing the molecular structure.
    xmlfile : str
        Path to the OpenMM-compatible force field XML file.
    cutoff_condition : openmm.app.NonbondedMethod, optional
        Nonbonded method specification (e.g., app.PME, app.CutoffPeriodic, 
        app.NoCutoff), by default app.PME.
    dispersion_correction : bool, optional
        Whether to apply long-range dispersion correction, by default False.
    nonbonded_cutoff : float, optional
        Nonbonded cutoff distance in nanometers, by default 1.2.
    ethresh : float, optional
        Ewald error tolerance for PME calculations. In DMFF, this parameter 
        controls the precision of electrostatic interactions. For parameter 
        consistency with OpenMM (which calls it 'ewaldErrorTolerance'), use 
        the same value for both functions. Default 0.0005 matches OpenMM 
        default behavior.

    Returns
    -------
    float
        Total potential energy in kJ/mol.

    Notes
    -----
    - Uses DMFF Hamiltonian as differentiable backend
    - Sums energy contributions from all potential components (bonded, nonbonded, etc.)
    """
    ff = Hamiltonian(xmlfile)
    pdb = app.PDBFile(pdbfile)
    pots = ff.createPotential(
                    pdb.topology,
                    nonbondedMethod=cutoff_condition,
                    nonbondedCutoff=nonbonded_cutoff * unit.nanometer,
                    useDispersionCorrection=dispersion_correction,
                    ethresh=ethresh
                    )
    cov_map = pots.meta["cov_map"]
    traj = mdtraj.load(pdbfile)
    box = jnp.array(traj[0].openmm_boxes(0).value_in_unit(unit.nanometer))
    print(jnp.result_type(box), box.shape)
    positions = jnp.array(traj[0].xyz[0, :, :])
    print(jnp.result_type(positions), positions.shape)
    nbobj = NeighborListFreud(box, nonbonded_cutoff, cov_map)
    nbobj.capacity_multiplier = 1
    pairs = nbobj.allocate(positions)
    pairs = jnp.array(pairs)
    ffparams = ff.getParameters().parameters
    rescharges = get_rescharges_from_residues(ff)
    ffparams = get_chgparams_from_rescharges(ffparams, rescharges)

    # Calculate total energy by summing all potential energy components
    total_energy = 0.0
    for key in pots.dmff_potentials:
        energy = pots.dmff_potentials[key](positions, box, pairs, ffparams)
        print(key, energy)
        total_energy += energy

    return float(total_energy)
