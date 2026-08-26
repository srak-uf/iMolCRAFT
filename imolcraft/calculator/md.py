#!/usr/bin/env python
"""
Molecular dynamics sampling with OpenMM.

One :class:`MDCalculator` describes one thermodynamic state - its temperature,
ensemble, cutoff, annealing schedule and how long to run - and produces a
trajectory for whatever force field it is handed. Everything the state needs is
held on the calculator, so :meth:`MDCalculator.to_dict` can hand a checkpoint
the complete recipe, defaults included, and :meth:`MDCalculator.from_dict` can
rebuild it later without consulting anything else.
"""
import contextlib
import os
import sys

import openmm
from openmm import app
import openmm.unit as unit


#: Ensembles an MDCalculator knows how to set up.
VALID_ENSEMBLES = ("nve", "nvt", "isonpt", "anisonpt", "trinpt")

#: Where the progress messages and per-step state data may be sent.
MD_LOG_MODES = ("stdout", "file", "none")

#: Nonbonded methods accepted by name.
NONBONDED_METHODS = {"PME": app.PME, "LJPME": app.LJPME}


def resolve_nonbondedmethod(name):
    """Translate a nonbonded method name into its OpenMM constant."""
    if name in NONBONDED_METHODS.values():
        return name
    if name not in NONBONDED_METHODS:
        raise ValueError(
            f"Invalid nonbonded method: {name}. Must be one of "
            f"{list(NONBONDED_METHODS)}."
        )
    return NONBONDED_METHODS[name]


@contextlib.contextmanager
def _open_md_log(md_log, md_logfile, trajectory):
    """
    Resolve the requested MD log destination into a writable stream.

    Yields ``sys.stdout`` for ``"stdout"``, ``None`` for ``"none"`` (callers
    then emit nothing at all), or a freshly opened file for ``"file"``. Only a
    file opened here is closed on exit; ``sys.stdout`` is left alone.
    """
    mode = "none" if md_log is None else str(md_log).lower()
    if mode not in MD_LOG_MODES:
        raise ValueError(f"md_log must be one of {MD_LOG_MODES}, got {md_log!r}")
    if mode == "none":
        yield None
        return
    if mode == "stdout":
        yield sys.stdout
        return

    path = md_logfile
    if path is None:
        # One log per trajectory, so replicas never overwrite each other.
        stem = os.path.splitext(os.path.basename(trajectory))[0]
        path = os.path.join("mdlogs", f"{stem}.log")
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    handle = open(path, "w")
    try:
        yield handle
    finally:
        handle.close()


def _anneal_schedule(anneal_T, anneal_steps):
    """
    Validate an annealing schedule and expand it into ``(T_from, T_to, nsteps)``
    legs.

    ``anneal_T`` lists the temperature corners the run travels through, in
    kelvin, and ``anneal_steps`` the MD steps spent on each leg between them,
    so it holds exactly one entry fewer. ``anneal_T=[100, 1000, 300]`` with
    ``anneal_steps=[50000, 30000]`` heats 100 K -> 1000 K over 50000 steps and
    then cools 1000 K -> 300 K over 30000 steps.

    Both arguments are sequences, a list or a tuple. A ramp needs a temperature
    to start from and one to end at, so a bare number is a mistake rather than
    a one-leg schedule. Either argument being None or empty means no annealing,
    and an empty list of legs comes back.
    """
    for name, value in (("anneal_T", anneal_T), ("anneal_steps", anneal_steps)):
        if value is not None and not isinstance(value, (list, tuple)):
            raise TypeError(
                f"{name} must be a list or a tuple, got {type(value).__name__}: "
                "a schedule reads anneal_T: [300, 400, 300] with "
                "anneal_steps: [5000, 5000]"
            )
    if not anneal_T or not anneal_steps:
        return []
    T_points = [float(T) for T in anneal_T]
    leg_steps = [int(n) for n in anneal_steps]
    if len(T_points) < 2:
        raise ValueError(
            "anneal_T needs at least two temperatures to ramp between, got "
            f"{T_points}"
        )
    if len(leg_steps) != len(T_points) - 1:
        raise ValueError(
            "anneal_steps holds the MD steps of each leg, so it must have "
            f"exactly one entry fewer than anneal_T: got {len(leg_steps)} legs "
            f"for {len(T_points)} temperatures"
        )
    if any(n < 0 for n in leg_steps):
        raise ValueError(f"anneal_steps must not be negative, got {leg_steps}")
    return list(zip(T_points[:-1], T_points[1:], leg_steps))


def _ramp_temperature(simulation, integrator, T_from, T_to, nsteps, interval):
    """
    Run ``nsteps`` of MD while the thermostat set point moves linearly from
    ``T_from`` to ``T_to``.

    The set point is refreshed every ``interval`` steps, so the leg is a fine
    staircase rather than a single jump. Updating it on every step would make
    the ramp exactly continuous, but the Python round trip per step costs
    nothing on the CPU platform and up to a few times the step itself on the
    GPU ones, hence the interval. The last chunk is short when ``interval``
    does not divide ``nsteps``, so the leg always lands exactly on ``T_to``.
    A leg of zero (or negative) length runs no MD and leaves the set point
    untouched.
    """
    if nsteps <= 0:
        return
    interval = max(1, min(int(interval), nsteps))
    done = 0
    while done < nsteps:
        chunk = min(interval, nsteps - done)
        done += chunk
        integrator.setTemperature(
            (T_from + (T_to - T_from) * done / nsteps) * unit.kelvin
        )
        simulation.step(chunk)


def _make_barostat(ensemble, T, log=None):
    """
    Barostat matching the requested NPT flavour, or None for a fixed volume.

    The three NPT ensembles differ in how much of the box shape they let move:
    isotropic scaling, independent axes, or a fully flexible triclinic cell.
    ``log`` is an optional one-argument callable used to announce the choice.
    """
    if log is None:
        def log(_message):
            return None

    if ensemble == "isonpt":
        log("Isotropic pressure control")
        return openmm.MonteCarloBarostat(1.0 * unit.bar, T * unit.kelvin)
    if ensemble == "anisonpt":
        log("Anisotropic pressure control")
        return openmm.MonteCarloAnisotropicBarostat(
            [1.0 * unit.bar] * 3, T * unit.kelvin
        )
    if ensemble == "trinpt":
        return openmm.MonteCarloFlexibleBarostat(1.0 * unit.bar, T * unit.kelvin)
    return None


class MDCalculator:
    """
    MD sampling of one thermodynamic state with OpenMM.

    The state is fixed at construction, while the force field and the output
    trajectory change from run to run and are therefore arguments of
    :meth:`run`. That split is what makes the calculator recordable: what it
    holds is exactly the recipe of the run, so :meth:`to_dict` writes a
    complete one into a checkpoint and :meth:`from_dict` reads it back.

    The settings are named exactly as the sampling section of the YAML names
    them, so a sampling block needs no translation to become a calculator.

    Parameters
    ----------
    init_structure : str
        Structure the run starts from.
    rcut_nm : float, optional
        Nonbonded cutoff in nanometre. Default 1.2.
    temperature_K : float, optional
        Desired temperature in kelvin, held through the relaxation and the
        production run. Default 300.
    anneal_T : list of float, optional
        Temperature corners of the annealing schedule, in kelvin. The run ramps
        linearly from each corner to the next before the relaxation at ``T``
        begins, so ``[100, 1000, 300]`` heats to 1000 K and cools back to
        300 K. None (the default) or an empty list skips the annealing.
    anneal_steps : list of int, optional
        MD steps spent on each leg, so one entry fewer than ``anneal_T``.
    anneal_interval : int, optional
        How often, in MD steps, the thermostat set point is refreshed along a
        leg. Default 100, smooth enough to read as a continuous ramp while
        staying free on both the CPU and the GPU platforms.
    dt_fs : float, optional
        Timestep in femtosecond. Default 1.0.
    nstxout : int, optional
        Interval, in MD steps, between trajectory frames and log lines.
    relax_steps, prod_steps : int, optional
        Length of the relaxation and of the production run, in MD steps.
    ensemble : str, optional
        One of :data:`VALID_ENSEMBLES`. Default ``"nvt"``.
    nonbondedmethod : str, optional
        ``"PME"`` or ``"LJPME"``. Default ``"PME"``.
    dispcorr : bool, optional
        Whether the nonbonded force applies a long-range dispersion correction.
    useHbondConstraint : bool, optional
        Whether bonds to hydrogen are constrained. Default True.
    rigidWater : bool, optional
        Whether water molecules are kept rigid. Default False.
    device : str, optional
        OpenMM platform name. Default ``"CPU"``.
    md_log : {'stdout', 'file', 'none'}, optional
        Where the progress messages and the per-step state data go.
        ``'stdout'`` (default) writes to the terminal, ``'file'`` redirects
        everything to ``md_logfile``, and ``'none'`` suppresses the output
        entirely, attaching no StateDataReporter at all.
    md_logfile : str, optional
        Log file used when ``md_log='file'``. Default is
        ``mdlogs/<trajectory stem>.log``, so replicas do not overwrite each
        other. Ignored for the other modes.
    """

    #: Settings of a run and the value used when one is left out. Their
    #: defaults live here alone, which is what lets to_dict hand over a
    #: complete recipe even for settings the caller never mentioned.
    SETTINGS = {
        "rcut_nm": 1.2,
        "temperature_K": 300.0,
        "anneal_T": None,
        "anneal_steps": None,
        "anneal_interval": 100,
        "dt_fs": 1.0,
        "nstxout": 1000,
        "relax_steps": 100000,
        "prod_steps": 2000000,
        "ensemble": "nvt",
        "nonbondedmethod": "PME",
        "dispcorr": False,
        "useHbondConstraint": True,
        "rigidWater": False,
        "device": "CPU",
        "md_log": "stdout",
        "md_logfile": None,
    }

    def __init__(self, init_structure, **settings):
        unknown = sorted(set(settings) - set(self.SETTINGS))
        if unknown:
            raise TypeError(
                f"unknown MD settings: {', '.join(unknown)}. Known ones are "
                f"{', '.join(sorted(self.SETTINGS))}"
            )
        self.init_structure = init_structure
        for name, default in self.SETTINGS.items():
            setattr(self, name, settings.get(name, default))

        if self.ensemble not in VALID_ENSEMBLES:
            raise ValueError(
                f"Invalid ensemble {self.ensemble}. Must be one of "
                f"{list(VALID_ENSEMBLES)}."
            )
        # a broken schedule should show up now, not after the minimization
        self.anneal_legs = _anneal_schedule(self.anneal_T, self.anneal_steps)

    def to_dict(self):
        """
        The complete recipe of this run, ready to be written to a checkpoint.

        Every setting is listed, including the ones that were left to their
        default, so a run rebuilt from the record stays the same even if a
        default changes later.
        """
        record = {"init_structure": self.init_structure}
        record.update({name: getattr(self, name) for name in self.SETTINGS})
        return record

    @classmethod
    def from_dict(cls, record):
        """Rebuild a calculator from what :meth:`to_dict` wrote."""
        settings = dict(record)
        return cls(settings.pop("init_structure"), **settings)

    def _build_simulation(self, ffxml, log):
        """
        Assemble the OpenMM simulation of this state under one force field.

        Returns the simulation, its integrator and the starting positions,
        which the caller needs to seed the context.
        """
        pdb = app.PDBFile(self.init_structure)
        forcefield = app.ForceField(ffxml)

        modeller = app.Modeller(pdb.topology, pdb.getPositions())
        modeller.addExtraParticles(forcefield)
        pos = modeller.getPositions()
        topology = modeller.topology

        constraints = {"constraints": app.HBonds} if self.useHbondConstraint else {}
        system = forcefield.createSystem(
            topology,
            nonbondedMethod=resolve_nonbondedmethod(self.nonbondedmethod),
            nonbondedCutoff=self.rcut_nm * unit.nanometer,
            rigidWater=self.rigidWater,
            **constraints,
        )

        for force in system.getForces():
            if isinstance(force, openmm.NonbondedForce):
                force.setUseDispersionCorrection(self.dispcorr)

        log(f"Using {self.ensemble} ensemble")
        barostat = _make_barostat(self.ensemble, self.temperature_K, log=log)
        if barostat is not None:
            system.addForce(barostat)

        integrator = openmm.LangevinIntegrator(
            self.temperature_K * unit.kelvin,
            1 / unit.picosecond,
            self.dt_fs * unit.femtosecond,
        )
        simulation = app.Simulation(
            topology,
            system,
            integrator,
            openmm.Platform.getPlatformByName(self.device),
        )
        return simulation, integrator, pos

    def run(self, ffxml, trajectory):
        """
        Sample this state with the given force field.

        The run minimizes, walks the annealing schedule if there is one, relaxes
        at ``T`` and finally produces the trajectory.

        Parameters
        ----------
        ffxml : str
            Force field to sample with.
        trajectory : str
            Name of the trajectory file, written under ``xtcfiles/``.

        Returns
        -------
        xtcfile : str
            Path to the trajectory written by the production run.
        """
        anneal_steps_total = sum(nsteps for _, _, nsteps in self.anneal_legs)

        with _open_md_log(self.md_log, self.md_logfile, trajectory) as logstream:

            def log(message):
                if logstream is not None:
                    print(message, file=logstream, flush=True)

            simulation, integrator, pos = self._build_simulation(ffxml, log)

            xtcfile = os.path.join("xtcfiles", trajectory)
            try:
                os.remove(xtcfile)
            except Exception:
                pass
            simulation.context.setPositions(pos)
            log("== Energy minimization ==")
            simulation.minimizeEnergy()
            simulation.context.setVelocitiesToTemperature(
                self.temperature_K * unit.kelvin
            )

            if logstream is not None:
                simulation.reporters.append(
                    app.StateDataReporter(
                        logstream,
                        self.nstxout,
                        potentialEnergy=True,
                        temperature=True,
                        density=True,
                        step=True,
                        remainingTime=True,
                        speed=True,
                        totalSteps=(
                            anneal_steps_total + self.relax_steps + self.prod_steps
                        ),
                    )
                )

            # SA: walk the temperature through the corners of anneal_T
            if self.anneal_legs:
                log(
                    "== Start Simulated Annealing: "
                    + " ".join(
                        f"{T_from:g} K -[{nsteps} steps]->"
                        for T_from, _, nsteps in self.anneal_legs
                    )
                    + f" {self.anneal_legs[-1][1]:g} K =="
                )
                for T_from, T_to, nsteps in self.anneal_legs:
                    _ramp_temperature(
                        simulation, integrator, T_from, T_to, nsteps,
                        self.anneal_interval,
                    )

            # relax at desired temperature
            log("== Start Relaxation ==")
            integrator.setTemperature(self.temperature_K * unit.kelvin)
            simulation.step(self.relax_steps)

            # production run
            log("== Start Production ==")
            os.makedirs("xtcfiles", exist_ok=True)
            simulation.reporters.append(app.XTCReporter(xtcfile, self.nstxout))
            simulation.step(self.prod_steps)
            return xtcfile


def md_sample(init_structure, ffxml, trajectory, **settings):
    """
    Run one MD sampling with OpenMM.

    Thin wrapper over :class:`MDCalculator` for callers that only want the
    trajectory and have nothing to record. ``settings`` takes the same keywords
    as the calculator.

    Returns
    -------
    xtcfile : str
        Path to the trajectory written by the production run.
    """
    return MDCalculator(init_structure, **settings).run(ffxml, trajectory)
