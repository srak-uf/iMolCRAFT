#!/usr/bin/env python
"""
Molecular dynamics sampling with GROMACS.

:class:`GMXCalculator` is the GROMACS counterpart of
:class:`imolcraft.calculator.md.MDCalculator`. It takes the same ``.pdb``, is
configured with the same setting names and units, keeps the same
:meth:`~GMXCalculator.to_dict` / :meth:`~GMXCalculator.from_dict` contract and
the same ``run(ffxml, trajectory)`` signature, and returns the same
``xtcfiles/<trajectory>`` path, so a trainer can use one in place of the
other. GROMACS needs a ``.top`` / ``.gro`` pair rather than a ``.ffxml``:
``run`` builds the OpenMM *System* from the ``.ffxml`` exactly as
``MDCalculator.run`` does, serializes it, and hands it with the structure to
:func:`imolcraft.io.exporter` (``fmt="gmx"``) in a temporary directory.

The run is a sequence of ``gmx grompp`` / ``gmx mdrun`` calls, one per stage
(energy minimization, optional annealing, relaxation, production). Every
``.mdp`` is derived from a template shipped under ``imolcraft/data/mdp/``
(the files of srak-uf/gromacs_tutorial) with the settings of the calculator
written over it, and is kept in ``workdir`` next to the GROMACS ``.tpr`` /
``.log`` / ``.edr`` files for the record.

GROMACS is driven through :mod:`subprocess`; :func:`_run_command` is the only
place a process is started, so another backend (gmxapi) can be plugged in
there later without touching the public API.
"""
import importlib.resources
import math
import os
import pathlib
import shlex
import shutil
import subprocess
import sys
import tempfile

import openmm
from openmm import app
import openmm.unit as unit

from imolcraft.io import exporter

from .md import (
    MD_LOG_MODES,
    NONBONDED_METHODS,
    VALID_ENSEMBLES,
    _anneal_schedule,
    _open_md_log,
    resolve_nonbondedmethod,
)

__all__ = ["GMXCalculator", "gmx_sample", "GMX_STAGES", "GMX_PCOUPL", "GMX_ENV"]

#: Environment variables that supply the execution settings a calculator
#: leaves at None (``gmx_bin``, ``mpi_command``, ``ntmpi``, ``ntomp``). An
#: argument wins over the variable, the variable over the built-in default
#: (``gmx`` for the binary, nothing for the rest), see
#: :meth:`GMXCalculator._resolve_execution`. They belong to the job, not to
#: the state, which is why ``to_dict`` never records them.
GMX_ENV = {
    "gmx_bin": "IMOLCRAFT_GMX_BIN",
    "mpi_command": "IMOLCRAFT_GMX_MPI_COMMAND",
    "ntmpi": "IMOLCRAFT_GMX_NTMPI",
    "ntomp": "IMOLCRAFT_GMX_NTOMP",
}

#: Stages of one run, in order. Minimization is a preprocessing step, not an
#: ensemble; the annealing stage is skipped when there is no schedule.
GMX_STAGES = ("min", "anneal", "relax", "prod")

#: Pressure coupling of each ensemble of :data:`~imolcraft.calculator.md.VALID_ENSEMBLES`:
#: ``pcoupltype`` and the masks that turn the scalar ``pressure_bar`` /
#: ``compressibility_bar`` into the ``ref-p`` / ``compressibility`` vectors
#: (xx yy zz xy xz yz). None means no pressure coupling.
GMX_PCOUPL = {
    "nve": None,
    "nvt": None,
    "isonpt": ("isotropic", (1,), (1,)),
    "anisonpt": ("anisotropic", (1, 1, 1, 0, 0, 0), (1, 1, 1, 0, 0, 0)),
    "trinpt": ("anisotropic", (1, 1, 1, 0, 0, 0), (1, 1, 1, 1, 1, 1)),
}

#: Template of each stage that does not depend on the ensemble.
_STAGE_TEMPLATES = {"min": "min.mdp", "anneal": "nvt.mdp"}

#: Template of the relaxation and production stages of each ensemble.
_ENSEMBLE_TEMPLATES = {
    "nve": "nvt.mdp",
    "nvt": "nvt.mdp",
    "isonpt": "isonpt.mdp",
    "anisonpt": "anisonpt_xyz.mdp",
    "trinpt": "trinpt_xyz_xy_yz_zx.mdp",
}

#: Ewald error tolerance OpenMM uses when createSystem is not told otherwise
#: (NonbondedForce default, 5e-4, shared by the Coulomb and the LJ mesh).
_EWALD_TOLERANCE = openmm.NonbondedForce().getEwaldErrorTolerance()

#: mdp keys of the pressure coupling that must not leak out of a template
#: into a run without a barostat (nvt.mdp carries ``ref-p = 5.0``).
_PCOUPL_KEYS = ("pcoupltype", "tau-p", "compressibility", "ref-p", "nstpcouple")

#: Template keys removed from every stage. The templates fix the pair list
#: with ``verlet-buffer-tolerance = -1`` and ``rlist = 1.4``; the calculator
#: writes neither and leaves the pair-list radius to GROMACS (default
#: tolerance 0.005 kJ/mol/ps, from which grompp and mdrun set ``rlist`` and
#: ``nstlist``). ``mdp_extra`` can put both keys back for a fixed list.
_TEMPLATE_KEYS_DROPPED = ("verlet-buffer-tolerance", "rlist")


def _mdp_key(key):
    """
    Canonical spelling of an mdp key.

    GROMACS treats ``-`` and ``_`` in a key as the same character; the
    templates use ``-``, so every key is folded onto that spelling so that an
    override replaces the template entry instead of sitting next to it.
    """
    return str(key).strip().replace("_", "-")


def _read_mdp(path):
    """
    Parse an mdp file into an ordered ``{key: value}`` dict.

    ``path`` is a filesystem path or any object with ``read_text()`` (such as
    an :mod:`importlib.resources` traversable). Comments (``;`` to the end of
    the line) and blank lines are dropped; values are kept as strings, an
    empty value as ``""``.
    """
    if not hasattr(path, "read_text"):
        path = pathlib.Path(path)
    options = {}
    for line in path.read_text().splitlines():
        line = line.split(";", 1)[0].strip()
        if not line:
            continue
        key, sep, value = line.partition("=")
        if not sep:
            raise ValueError(f"Malformed mdp line (no '='): {line!r}")
        options[_mdp_key(key)] = value.strip()
    return options


def _format_mdp(options):
    """Render an ordered ``{key: value}`` dict as ``key = value`` lines."""
    return "".join(f"{key:<24s}= {value}\n" for key, value in options.items())


def _num(value):
    """
    mdp spelling of a real number.

    Rounded to 12 decimals so that ``1.2 + 0.2`` prints as ``1.4`` rather than
    ``1.4000000000000001``; ``1.0`` stays ``1.0``.
    """
    return repr(round(float(value), 12))


def _vector(scalar, mask):
    """Expand a scalar onto a mask: ``(1, 1, 1, 0, 0, 0)`` -> ``"P P P 0.0 0.0 0.0"``."""
    return " ".join(_num(scalar * m) for m in mask)


def _pressure_options(ensemble, pressure_bar, compressibility_bar,
                      tcoupl, pcoupl, tau_t_ps, tau_p_ps):
    """
    Thermostat and barostat mdp entries of an ensemble.

    Returns ``(set, drop)``: the entries to write and the keys to remove from
    the template. ``nve`` switches both couplings off, ``nvt`` keeps the
    thermostat only, and the three NPT flavours add the barostat with the
    ``ref-p`` / ``compressibility`` vectors of :data:`GMX_PCOUPL`. Pressure in
    bar, compressibility in 1/bar, coupling times in ps.
    """
    if ensemble not in GMX_PCOUPL:
        raise ValueError(
            f"Invalid ensemble {ensemble}. Must be one of {list(VALID_ENSEMBLES)}."
        )
    options = {
        "tcoupl": "no" if ensemble == "nve" else str(tcoupl),
        "tau-t": _num(tau_t_ps),
    }
    barostat = GMX_PCOUPL[ensemble]
    if barostat is None:
        options["pcoupl"] = "no"
        return options, list(_PCOUPL_KEYS)
    pcoupltype, p_mask, c_mask = barostat
    options.update({
        "pcoupl": str(pcoupl),
        "pcoupltype": pcoupltype,
        "tau-p": _num(tau_p_ps),
        "compressibility": _vector(compressibility_bar, c_mask),
        "ref-p": _vector(pressure_bar, p_mask),
    })
    return options, []


def _annealing_options(legs, dt_fs):
    """
    GROMACS native annealing entries for the legs of
    :func:`~imolcraft.calculator.md._anneal_schedule`.

    ``legs`` is a list of ``(T_from, T_to, nsteps)``; the corner times are the
    cumulative step counts converted to ps with ``dt_fs`` (this and ``dt``
    itself are the only unit conversions of the module). An empty list gives
    an empty dict.
    """
    if not legs:
        return {}
    temperatures = [legs[0][0]] + [T_to for _, T_to, _ in legs]
    times, elapsed = [0.0], 0.0
    for _, _, nsteps in legs:
        elapsed += nsteps * float(dt_fs) / 1000.0
        times.append(elapsed)
    return {
        "annealing": "single",
        "annealing-npoints": str(len(temperatures)),
        "annealing-time": " ".join(_num(t) for t in times),
        "annealing-temp": " ".join(_num(T) for T in temperatures),
    }


def _ewald_options(rcut_nm, ljpme, tol=_EWALD_TOLERANCE):
    """
    Ewald mdp entries reproducing the PME parameters OpenMM derives from
    its error tolerance ``tol`` (``NonbondedForce.getEwaldErrorTolerance``).

    OpenMM sets the Ewald splitting parameter to
    ``alpha = sqrt(-ln(2 tol)) / rc`` and the Coulomb mesh size along a box
    vector of length ``L`` to ``ceil(2 alpha L / (3 tol**0.2))``; its LJPME
    dispersion mesh is half as dense, ``ceil(alpha L / (3 tol**0.2))`` (14^3
    against 27^3 in the test system). GROMACS instead derives ``beta`` from
    the strength of the shifted direct-space potential at the cutoff:
    ``erfc(beta rc) = ewald-rtol`` for Coulomb and
    ``exp(-x^2) (1 + x^2 + x^4/2) = ewald-rtol-lj`` with ``x = beta rc`` for
    dispersion (GROMACS manual, "Ewald summation" / "Lennard-Jones PME").
    Evaluating those two definitions at OpenMM's ``x = alpha rc`` gives
    tolerances that make ``beta == alpha`` exactly, whatever ``rc``. The
    grid is matched to OpenMM's Coulomb mesh through
    ``fourierspacing = L / n = 3 tol**0.2 / (2 alpha)``, which is independent
    of the box; both codes then round the size up to an FFT-friendly number,
    so the Coulomb grids agree up to that rounding. GROMACS uses the same
    grid for the dispersion mesh, so under LJPME its dispersion mesh is finer
    than OpenMM's; that coarser OpenMM grid is the main source of the LJPME
    energy residual (4.4e-3 kJ/mol in the test system). ``pme-order`` is
    left to the template (6; OpenMM uses 5).
    """
    x = math.sqrt(-math.log(2.0 * tol))  # alpha * rc in OpenMM
    options = {
        "ewald-rtol": _num(math.erfc(x)),
        "fourierspacing": _num(3.0 * tol ** 0.2 * float(rcut_nm) / (2.0 * x)),
    }
    if ljpme:
        options["ewald-rtol-lj"] = _num(math.exp(-x * x) * (1.0 + x * x + x ** 4 / 2.0))
    return options


def _positive_int(value, name, source):
    """
    ``value`` as a positive int, or None when it is None.

    ``source`` is where the value came from (``"argument"`` or
    ``"environment"``) so that the error names the setting or the variable
    that holds the offending text.
    """
    if value is None:
        return None
    where = f"${GMX_ENV[name]}" if source == "environment" else name
    try:
        number = int(str(value).strip())
    except ValueError:
        raise ValueError(f"{where}={value!r} is not an integer") from None
    if number <= 0:
        raise ValueError(f"{where}={value!r} must be a positive integer")
    return number


def _run_command(argv, logstream, env=None):
    """
    Run one GROMACS command to completion.

    The only place a process is started. ``argv`` is a list (never a shell
    string). The child inherits the terminal when ``logstream`` is
    ``sys.stdout``, is silenced when it is None and otherwise writes into the
    stream, stderr merged into stdout. Raises
    :class:`subprocess.CalledProcessError` on a non-zero exit.
    """
    if logstream is None:
        stdout = subprocess.DEVNULL
    elif logstream is sys.stdout:
        stdout = None
    else:
        logstream.flush()
        stdout = logstream
    subprocess.run(
        argv, check=True, stdout=stdout, stderr=subprocess.STDOUT, env=env
    )


class GMXCalculator:
    """
    MD sampling of one thermodynamic state with GROMACS.

    Mirrors :class:`~imolcraft.calculator.md.MDCalculator`: the state is fixed
    at construction, the force field and the trajectory name are arguments of
    :meth:`run`, and :meth:`to_dict` / :meth:`from_dict` carry the complete
    recipe. All 17 settings of ``MDCalculator`` are accepted under the same
    names and units; the ones below them are GROMACS-specific additions.

    Units follow GROMACS: nm, ps, K, bar, kJ/mol. ``dt_fs`` (fs) and the
    annealing step counts are the only values converted (to ps).

    Parameters
    ----------
    init_structure : str
        Starting structure, a ``.pdb`` with bonds (CONECT records), exactly
        as ``MDCalculator`` takes it. Both the ``.gro`` and the ``.top`` are
        derived from it and from the ``.ffxml`` given to :meth:`run`.
    rcut_nm : float, optional
        Nonbonded cutoff in nm, written to ``rvdw`` and ``rcoulomb``. The
        pair-list radius is left to GROMACS (``verlet-buffer-tolerance`` at
        its default, from which grompp and mdrun set ``rlist`` and
        ``nstlist``); to fix it manually pass both
        ``verlet-buffer-tolerance = -1`` and ``rlist`` via ``mdp_extra``.
        Default 1.2.
    temperature_K : float, optional
        ``ref-t`` and ``gen-temp`` in kelvin for the relaxation and the
        production. Default 300.
    anneal_T : list of float, optional
        Temperature corners of the annealing schedule in kelvin, as in
        ``MDCalculator``. Mapped onto GROMACS native ``annealing = single``
        in a dedicated NVT stage before the relaxation. None or an empty
        list skips the stage.
    anneal_steps : list of int, optional
        MD steps of each leg (one entry fewer than ``anneal_T``); converted
        to ps for ``annealing-time``.
    anneal_interval : int, optional
        Accepted for compatibility with ``MDCalculator`` and **not used**:
        GROMACS interpolates the set point linearly on every step.
    dt_fs : float, optional
        Timestep in fs; written as ``dt`` in ps. Default 1.0.
    nstxout : int, optional
        Interval in MD steps of the ``.xtc`` frames (``nstxout-compressed``)
        and of the energy / log output. The ``.trr`` output (GROMACS's own
        ``nstxout`` / ``nstvout``) is switched off. Default 1000.
    relax_steps, prod_steps : int, optional
        ``nsteps`` of the relaxation and of the production stage. Note that
        the defaults (100000 and 2000000) mean hours of wall time; there is
        no timeout.
    ensemble : str, optional
        One of :data:`~imolcraft.calculator.md.VALID_ENSEMBLES`. ``nve``
        switches the thermostat and the barostat off, ``nvt`` the barostat
        only, ``isonpt`` / ``anisonpt`` / ``trinpt`` couple the pressure
        isotropically, per axis, or with the off-diagonal components as
        well. Default ``"nvt"``.
    nonbondedmethod : str, optional
        ``"PME"`` (``vdwtype = Cut-off``) or ``"LJPME"`` (``vdwtype = PME``);
        ``coulombtype`` is PME in both cases. Default ``"PME"``.
    dispcorr : bool, optional
        ``DispCorr = EnerPres`` when True, ``no`` when False. Default False.
    useHbondConstraint : bool, optional
        ``constraints = h-bonds`` when True, ``none`` when False. Default
        True. The constraint is applied by GROMACS from the mdp, not baked
        into the exported topology (see :meth:`_build_system`).
    rigidWater : bool, optional
        Accepted for compatibility with ``MDCalculator`` and **not used**:
        the System is exported with flexible water, and GROMACS keeps water
        rigid only through ``[ settles ]`` in the topology, which the
        exporter does not write.
    device : str, optional
        ``"CPU"`` runs the nonbonded kernels on the CPU (``mdrun -nb cpu``);
        any other value (``"CUDA"``, ``"OpenCL"``, ``"HIP"``) asks for the
        GPU (``-nb gpu``). Default ``"CPU"``.
    md_log : {'stdout', 'file', 'none'}, optional
        Where the progress messages and the output of the GROMACS commands
        go. ``'stdout'`` (default) inherits the terminal, ``'file'`` redirects
        everything to ``md_logfile`` (overwritten at the start of every run),
        ``'none'`` discards it. GROMACS's own
        ``.log`` / ``.edr`` files are written to ``workdir`` regardless.
    md_logfile : str, optional
        Log file used when ``md_log='file'``. Default
        ``mdlogs/<trajectory stem>.log``.
    pressure_bar : float, optional
        ``ref-p`` in bar of the NPT ensembles. Default 1.0. Unlike
        ``MDCalculator``, which fixes 1 bar and leaves the pressure to the
        trainer, the pressure is a setting of this calculator.
    compressibility_bar : float, optional
        Isotropic ``compressibility`` in 1/bar. Default 4.5e-5.
    tau_t_ps, tau_p_ps : float, optional
        ``tau-t`` and ``tau-p`` in ps. Defaults 1.0 and 5.0.
    tcoupl, pcoupl : str, optional
        Thermostat and barostat algorithms (``tcoupl`` / ``pcoupl`` of the
        mdp). Defaults ``"nose-hoover"`` and ``"Parrinello-Rahman"``.
    min_steps : int, optional
        ``nsteps`` of the steepest-descent minimization. Default 4096.
    emtol : float, optional
        Convergence criterion of the minimization in kJ/mol/nm. Default 100.
    gmx_bin : str, optional
        GROMACS executable: ``"gmx"``, ``"gmx_d"``, ``"gmx_mpi"`` or
        ``"gmx_mpi_d"``, a name on PATH or an absolute path. None (default)
        takes ``$IMOLCRAFT_GMX_BIN``, or ``"gmx"`` when that is unset.
    mpi_command : list of str or str, optional
        Launcher put in front of ``mdrun`` only (never grompp), e.g.
        ``["mpirun", "-np", "32"]``; a string (``"mpirun -np 32"``) is split
        with :func:`shlex.split` when the command is built, while
        :meth:`to_dict` keeps it as given. None (default) takes
        ``$IMOLCRAFT_GMX_MPI_COMMAND`` split with :func:`shlex.split`
        (``"srun --mpi=pmix -n 8"``), or no launcher when that is unset or
        empty. A launcher implies a real-MPI build, which does not accept
        ``-ntmpi``, so combining it with ``ntmpi`` is an error.
    ntmpi, ntomp : int, optional
        ``mdrun -ntmpi`` / ``-ntomp``. None (default) takes
        ``$IMOLCRAFT_GMX_NTMPI`` / ``$IMOLCRAFT_GMX_NTOMP``, or leaves the
        choice to mdrun when they are unset (without ``-ntomp`` GROMACS
        honours ``OMP_NUM_THREADS``).

        These four execution settings are resolved at run time, argument >
        environment > default (:data:`GMX_ENV`), and :meth:`to_dict`
        records only the arguments, so a checkpoint carries no job-specific
        launcher. A batch script therefore sets them once, e.g.::

            export IMOLCRAFT_GMX_BIN=gmx_mpi
            export IMOLCRAFT_GMX_MPI_COMMAND="srun --mpi=pmix -n $SLURM_NTASKS"
            export IMOLCRAFT_GMX_NTOMP=$SLURM_CPUS_PER_TASK
    maxwarn : int, optional
        ``grompp -maxwarn``. Default 0.
    mdp_templates : dict, optional
        ``{stage: path}`` of ``.mdp`` files to use instead of the shipped
        ones for the stages listed (see :data:`GMX_STAGES`). Default None.
    mdp_extra : dict, optional
        Raw ``{mdp key: value}`` entries applied last to every stage, on top
        of everything the settings write. Default None. The pair-list radius
        is left to GROMACS (``verlet-buffer-tolerance``); to fix it manually
        pass both ``verlet-buffer-tolerance = -1`` and ``rlist`` here.
    workdir : str, optional
        Directory receiving the ``.mdp``, ``.tpr``, ``.log``, ``.edr``,
        ``.cpt`` and intermediate ``.gro`` files, one set per stage named
        ``<trajectory stem>_<stage>``. Default ``"gmxfiles"``. The
        ``.top`` / ``.gro`` inputs live in a temporary directory and are
        removed when the run ends.
    """

    #: Settings of a run and the value used when one is left out. The first
    #: block is MDCalculator.SETTINGS verbatim; the rest is GROMACS-specific.
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
        "pressure_bar": 1.0,
        "compressibility_bar": 4.5e-5,
        "tau_t_ps": 1.0,
        "tau_p_ps": 5.0,
        "tcoupl": "nose-hoover",
        "pcoupl": "Parrinello-Rahman",
        "min_steps": 4096,
        "emtol": 100.0,
        "gmx_bin": None,
        "mpi_command": None,
        "ntmpi": None,
        "ntomp": None,
        "maxwarn": 0,
        "mdp_templates": None,
        "mdp_extra": None,
        "workdir": "gmxfiles",
    }

    def __init__(self, init_structure, **settings):
        unknown = sorted(set(settings) - set(self.SETTINGS))
        if unknown:
            raise TypeError(
                f"unknown GROMACS MD settings: {', '.join(unknown)}. Known ones "
                f"are {', '.join(sorted(self.SETTINGS))}"
            )
        self.init_structure = init_structure
        for name, default in self.SETTINGS.items():
            setattr(self, name, settings.get(name, default))

        if self.ensemble not in VALID_ENSEMBLES:
            raise ValueError(
                f"Invalid ensemble {self.ensemble}. Must be one of "
                f"{list(VALID_ENSEMBLES)}."
            )
        if self.nonbondedmethod not in NONBONDED_METHODS:
            raise ValueError(
                f"Invalid nonbonded method: {self.nonbondedmethod}. Must be one "
                f"of {list(NONBONDED_METHODS)}."
            )
        mode = "none" if self.md_log is None else str(self.md_log).lower()
        if mode not in MD_LOG_MODES:
            raise ValueError(
                f"md_log must be one of {MD_LOG_MODES}, got {self.md_log!r}"
            )
        unknown_stages = sorted(set(self.mdp_templates or {}) - set(GMX_STAGES))
        if unknown_stages:
            raise ValueError(
                f"mdp_templates has unknown stages {unknown_stages}; stages are "
                f"{list(GMX_STAGES)}"
            )
        # a broken schedule should show up now, not after the minimization
        self.anneal_legs = _anneal_schedule(self.anneal_T, self.anneal_steps)

    def to_dict(self):
        """
        The complete recipe of this run, ready to be written to a checkpoint.

        Every setting is listed, defaults included, as
        :meth:`MDCalculator.to_dict` does.
        """
        record = {"init_structure": self.init_structure}
        record.update({name: getattr(self, name) for name in self.SETTINGS})
        return record

    @classmethod
    def from_dict(cls, record):
        """Rebuild a calculator from what :meth:`to_dict` wrote."""
        settings = dict(record)
        return cls(settings.pop("init_structure"), **settings)

    # -- mdp -----------------------------------------------------------------

    def _template(self, stage):
        """Path (or package traversable) of the template of a stage."""
        if self.mdp_templates and stage in self.mdp_templates:
            return self.mdp_templates[stage]
        name = _STAGE_TEMPLATES.get(stage) or _ENSEMBLE_TEMPLATES[self.ensemble]
        return importlib.resources.files("imolcraft.data") / "mdp" / name

    def _common_options(self):
        """
        mdp entries shared by every stage: cutoffs, PME, constraints.

        The nonbonded setup mirrors OpenMM's. ``"PME"`` is Coulomb PME with a
        plain Lennard-Jones cutoff (no shift, like OpenMM). ``"LJPME"`` adds
        the dispersion mesh with ``lj-pme-comb-rule = Geometric``: as in
        OpenMM's LJPME, the mesh uses geometric C6 mixing while the pairs
        inside the cutoff use the Lorentz-Berthelot parameters of the
        topology (``comb-rule 2`` in the ``.top``), and GROMACS corrects the
        difference in real space; grompp's NOTE that the C6 parameters "do
        not follow" the geometric rule is inherent to this OpenMM scheme and
        expected. ``vdw-modifier = Potential-Shift`` follows
        the LJ-PME configuration of the GROMACS reference manual
        (https://manual.gromacs.org/current/reference-manual/functions/long-range-vdw.html#using-lj-pme)
        and is also the GROMACS default; OpenMM applies no shift, so the
        energies differ by a small constant (~1e-4 kJ/mol in the test
        system) while the forces are unaffected. The Ewald parameters come
        from :func:`_ewald_options`.
        """
        ljpme = self.nonbondedmethod == "LJPME"
        options = {
            "pbc": "xyz",
            "cutoff-scheme": "Verlet",
            "coulombtype": "PME",
            "coulomb-modifier": "None",
            "rcoulomb": _num(self.rcut_nm),
            "vdwtype": "PME" if ljpme else "Cut-off",
            # OpenMM shifts neither potential. Potential-Shift for LJ-PME is
            # the manual's configuration and the GROMACS default: a small
            # constant offset of the energy, not of the forces.
            "vdw-modifier": "Potential-Shift" if ljpme else "None",
            "rvdw": _num(self.rcut_nm),
            # No rlist / verlet-buffer-tolerance: the template's values are
            # dropped (_TEMPLATE_KEYS_DROPPED) and GROMACS sets the pair list.
            "DispCorr": "EnerPres" if self.dispcorr else "no",
            "constraints": "h-bonds" if self.useHbondConstraint else "none",
            # Trajectories are xtc only; trr output off in every stage.
            "nstxout": "0",
            "nstvout": "0",
            "nstfout": "0",
        }
        if ljpme:
            # Geometric mesh, Lorentz-Berthelot pairs: the OpenMM LJPME scheme.
            # grompp notes that the C6 parameters do not follow the geometric
            # rule; that NOTE is inherent to this scheme and expected.
            options["lj-pme-comb-rule"] = "Geometric"
        options.update(_ewald_options(self.rcut_nm, ljpme))
        return options

    def _md_options(self, stage):
        """mdp entries of the dynamical stages (anneal, relax, prod)."""
        anneal_total = sum(nsteps for _, _, nsteps in self.anneal_legs)
        if stage == "anneal":
            nsteps = anneal_total
            T = self.anneal_legs[0][0] if self.anneal_legs else self.temperature_K
            fresh_start = True
        elif stage == "relax":
            nsteps = self.relax_steps
            T = self.temperature_K
            fresh_start = not self.anneal_legs
        else:
            nsteps = self.prod_steps
            T = self.temperature_K
            fresh_start = False

        options = {
            "integrator": "md",
            "dt": _num(float(self.dt_fs) / 1000.0),
            "nsteps": str(int(nsteps)),
            "tc-grps": "System",
            "ref-t": _num(T),
            "gen-vel": "yes" if fresh_start else "no",
            "gen-temp": _num(T),
            "continuation": "no" if fresh_start else "yes",
            "nstxout-compressed": str(int(self.nstxout)),
            "nstenergy": str(int(self.nstxout)),
            "nstlog": str(int(self.nstxout)),
        }
        # The annealing stage needs a thermostat whatever the ensemble is,
        # and never a barostat; the other stages follow the ensemble.
        coupling, drop = _pressure_options(
            "nvt" if stage == "anneal" else self.ensemble,
            self.pressure_bar, self.compressibility_bar,
            self.tcoupl, self.pcoupl, self.tau_t_ps, self.tau_p_ps,
        )
        options.update(coupling)
        if stage == "anneal":
            options.update(_annealing_options(self.anneal_legs, self.dt_fs))
        return options, drop

    def mdp_options(self, stage):
        """
        The complete mdp of one stage as an ordered ``{key: str}`` dict.

        The template of the stage is read, the keys the calculator does not
        carry over (:data:`_TEMPLATE_KEYS_DROPPED`) are removed, the entries
        derived from the settings are written over it, the keys that must not
        survive (the barostat of a template in a run without one) are
        removed, and ``mdp_extra`` is applied last. Pure: nothing is written.

        Parameters
        ----------
        stage : str
            One of :data:`GMX_STAGES`.
        """
        if stage not in GMX_STAGES:
            raise ValueError(f"stage must be one of {GMX_STAGES}, got {stage!r}")
        options = _read_mdp(self._template(stage))
        for key in _TEMPLATE_KEYS_DROPPED:
            options.pop(key, None)
        options.update(self._common_options())
        if stage == "min":
            options.update({
                "integrator": "steep",
                "nsteps": str(int(self.min_steps)),
                "emtol": _num(self.emtol),
            })
            drop = []
        else:
            stage_options, drop = self._md_options(stage)
            options.update(stage_options)
        for key in drop:
            options.pop(key, None)
        for key, value in (self.mdp_extra or {}).items():
            options[_mdp_key(key)] = str(value)
        return options

    def write_mdp(self, stage, path):
        """
        Write the mdp of ``stage`` to ``path`` and return ``path``.

        Parent directories are created as needed.
        """
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w") as handle:
            handle.write(_format_mdp(self.mdp_options(stage)))
        return path

    # -- commands ------------------------------------------------------------

    def _resolve_execution(self):
        """
        The execution settings actually used, and where each came from.

        Returns ``(values, sources)``: ``values`` holds ``gmx_bin`` (str),
        ``mpi_command`` (list of str or None), ``ntmpi`` and ``ntomp`` (int or
        None); ``sources`` maps the same names to ``"argument"``,
        ``"environment"`` or ``"default"``. An argument given to the
        calculator wins, then the variable of :data:`GMX_ENV`, then the
        default (``"gmx"`` for the binary, None for the rest). An empty
        variable counts as unset. A launcher together with ``ntmpi`` is
        refused: ``mpi_command`` means a real-MPI ``gmx_mpi``, whose mdrun
        takes the rank count from the launcher and rejects ``-ntmpi``.
        """
        defaults = {"gmx_bin": "gmx", "mpi_command": None, "ntmpi": None, "ntomp": None}
        values, sources = {}, {}
        for name, default in defaults.items():
            given = getattr(self, name)
            env = os.environ.get(GMX_ENV[name], "").strip()
            if given is not None:
                values[name], sources[name] = given, "argument"
            elif env:
                values[name], sources[name] = env, "environment"
            else:
                values[name], sources[name] = default, "default"

        # A launcher given as one string (argument or variable) is split like
        # a shell would, so quoted arguments survive; a list is taken as is.
        launcher = values["mpi_command"]
        if isinstance(launcher, str):
            launcher = shlex.split(launcher) or None
        elif launcher is not None:
            launcher = [str(x) for x in launcher]
        values["mpi_command"] = launcher

        for name in ("ntmpi", "ntomp"):
            values[name] = _positive_int(values[name], name, sources[name])
        values["gmx_bin"] = str(values["gmx_bin"])
        if values["mpi_command"] and values["ntmpi"] is not None:
            raise ValueError(
                f"mpi_command ({sources['mpi_command']}) and ntmpi "
                f"({sources['ntmpi']}) cannot be combined: a launcher means a "
                "real-MPI gmx_mpi, whose mdrun takes the rank count from the "
                "launcher and does not accept -ntmpi. Drop ntmpi (or "
                f"${GMX_ENV['ntmpi']}) and use ntomp for the threads per rank."
            )
        return values, sources

    def grompp_command(self, mdp, structure, topology, tpr, *, checkpoint=None):
        """
        ``gmx grompp`` argument list building ``tpr`` from ``mdp``,
        ``structure`` (``.gro``) and ``topology`` (``.top``).

        ``checkpoint`` (a ``.cpt``) hands the velocities of the previous stage
        over with ``-t``. The processed mdp (``-po``) goes next to the
        ``.tpr`` instead of the current directory. The binary comes from
        :meth:`_resolve_execution`; ``mpi_command`` never applies to grompp.
        """
        execution, _ = self._resolve_execution()
        argv = [
            execution["gmx_bin"], "grompp",
            "-f", mdp, "-c", structure, "-p", topology, "-o", tpr,
            "-po", os.path.splitext(tpr)[0] + "_mdout.mdp",
            "-maxwarn", str(int(self.maxwarn)),
        ]
        if checkpoint is not None:
            argv += ["-t", checkpoint]
        return argv

    def mdrun_command(self, deffnm):
        """
        ``gmx mdrun -deffnm <deffnm>`` argument list, prefixed with the
        launcher when one is resolved.

        Binary, launcher and thread counts come from
        :meth:`_resolve_execution`; ``device`` picks ``-nb cpu`` or
        ``-nb gpu``; ``-ntmpi`` / ``-ntomp`` appear only when resolved.
        """
        execution, _ = self._resolve_execution()
        argv = list(execution["mpi_command"] or [])
        argv += [execution["gmx_bin"], "mdrun", "-deffnm", deffnm]
        argv += ["-nb", "cpu" if str(self.device).upper() == "CPU" else "gpu"]
        if execution["ntmpi"] is not None:
            argv += ["-ntmpi", str(int(execution["ntmpi"]))]
        if execution["ntomp"] is not None:
            argv += ["-ntomp", str(int(execution["ntomp"]))]
        return argv

    def _check_binary(self, gmx_bin):
        """Raise FileNotFoundError with a hint when ``gmx_bin`` is not on PATH."""
        if shutil.which(gmx_bin) is None:
            raise FileNotFoundError(
                f"GROMACS executable {gmx_bin!r} not found on PATH. "
                "Activate the conda environment that provides it (imc_cpu) or "
                "source GMXRC.bash of your GROMACS installation, or point "
                f"gmx_bin (or ${GMX_ENV['gmx_bin']}) at the executable."
            )

    def _build_system(self, ffxml):
        """
        Topology, positions and OpenMM System of this structure under ``ffxml``.

        Kept step for step in line with ``MDCalculator._build_simulation``
        (md.py, which is deliberately left untouched): same ``PDBFile`` ->
        ``ForceField`` -> ``Modeller.addExtraParticles`` -> ``createSystem``
        with the same ``nonbondedMethod`` / ``nonbondedCutoff`` and the same
        ``setUseDispersionCorrection``, so GROMACS samples the model OpenMM
        would. Two arguments differ on purpose: ``constraints`` and
        ``rigidWater`` are not passed, because OpenMM drops the bond
        parameters of a constrained bond and parmed would then write a
        ``[ bonds ]`` entry without parameters, which grompp rejects. The
        hydrogen constraints come from the mdp (``constraints = h-bonds``)
        instead; ``rigidWater`` has no GROMACS counterpart here.
        """
        pdb = app.PDBFile(self.init_structure)
        forcefield = app.ForceField(ffxml)

        modeller = app.Modeller(pdb.topology, pdb.getPositions())
        modeller.addExtraParticles(forcefield)
        topology = modeller.topology

        system = forcefield.createSystem(
            topology,
            nonbondedMethod=resolve_nonbondedmethod(self.nonbondedmethod),
            nonbondedCutoff=self.rcut_nm * unit.nanometer,
            rigidWater=False,
        )
        for force in system.getForces():
            if isinstance(force, openmm.NonbondedForce):
                force.setUseDispersionCorrection(self.dispcorr)
        return topology, modeller.getPositions(), system

    def _export_inputs(self, ffxml, tmpdir):
        """
        Write the ``.top`` / ``.gro`` pair of this structure under ``ffxml``
        into ``tmpdir`` and return their paths.

        The System of :meth:`_build_system` is serialized to ``system.xml``,
        the (extra-particle complete) topology and positions to a ``.pdb``,
        and both go through :func:`imolcraft.io.exporter` with ``fmt="gmx"``.
        A force field with virtual sites is refused: the GROMACS exporter
        does not translate OpenMM virtual sites to ``[ virtual_sites2 ]``
        (parmed's ``load_topology`` keeps them as extra points and fails when
        writing the ``.gro``), so such force fields are not supported yet.
        """
        topology, positions, system = self._build_system(ffxml)
        n_vsites = sum(
            system.isVirtualSite(i) for i in range(system.getNumParticles())
        )
        if n_vsites:
            raise ValueError(
                f"{ffxml} places {n_vsites} virtual sites; the GROMACS exporter "
                "does not translate OpenMM virtual sites to [ virtual_sites2 ], "
                "so force fields with virtual sites are not supported yet. Use "
                "MDCalculator for this force field."
            )
        pdbfile = os.path.join(tmpdir, "system.pdb")
        with open(pdbfile, "w") as handle:
            app.PDBFile.writeFile(topology, positions, handle)
        system_xml = os.path.join(tmpdir, "system.xml")
        with open(system_xml, "w") as handle:
            handle.write(openmm.XmlSerializer.serialize(system))

        stem = os.path.join(tmpdir, "system")
        exporter(pdbfile, system_xml, stem, "gmx")
        return f"{stem}.top", f"{stem}.gro"

    def _run_stage(self, stage, deffnm, structure, topology, checkpoint, logstream):
        """
        grompp and mdrun one stage; every output is named ``deffnm.*``.

        Returns the ``.gro`` the next stage starts from. A failing command
        is reported with the stage name and GROMACS's own log.
        """
        mdp = self.write_mdp(stage, f"{deffnm}.mdp")
        tpr = f"{deffnm}.tpr"
        env = dict(os.environ, GMX_MAXBACKUP="-1")
        for argv in (
            self.grompp_command(mdp, structure, topology, tpr, checkpoint=checkpoint),
            self.mdrun_command(deffnm),
        ):
            try:
                _run_command(argv, logstream, env=env)
            except subprocess.CalledProcessError as error:
                raise RuntimeError(
                    f"GROMACS failed in the {stage} stage (exit code "
                    f"{error.returncode}): {' '.join(argv)}. See {deffnm}.log "
                    "and the MD log for details."
                ) from error
        return f"{deffnm}.gro"

    def run(self, ffxml, trajectory):
        """
        Sample this state with the given force field.

        The run minimizes, walks the annealing schedule if there is one,
        relaxes at ``T`` and finally produces the trajectory, each as a
        ``grompp`` / ``mdrun`` pair whose files are kept in ``workdir``.
        Same arguments and return value as :meth:`MDCalculator.run`.

        Parameters
        ----------
        ffxml : str
            OpenMM force field (``.ffxml``) to sample with, as
            :meth:`MDCalculator.run` takes it. The ``.top`` / ``.gro``
            GROMACS needs are generated from it and ``init_structure`` in a
            temporary directory that is removed when the run ends. A force
            field with virtual sites is refused (see :meth:`_export_inputs`).
        trajectory : str
            Name of the trajectory file, written under ``xtcfiles/``.

        Returns
        -------
        xtcfile : str
            Path to the ``.xtc`` written by the production run.
        """
        execution, sources = self._resolve_execution()
        self._check_binary(execution["gmx_bin"])
        stem = os.path.splitext(os.path.basename(trajectory))[0]
        stages = [s for s in GMX_STAGES if s != "anneal" or self.anneal_legs]
        os.makedirs(self.workdir, exist_ok=True)

        with _open_md_log(self.md_log, self.md_logfile, trajectory) as logstream, \
                tempfile.TemporaryDirectory(prefix="imolcraft_gmx_") as tmpdir:

            def log(message):
                if logstream is not None:
                    print(message, file=logstream, flush=True)

            log(f"Using {self.ensemble} ensemble")
            log(
                "mdrun command: " + shlex.join(self.mdrun_command("<deffnm>"))
                + " (" + ", ".join(f"{k}: {v}" for k, v in sources.items()) + ")"
            )
            topology, structure = self._export_inputs(ffxml, tmpdir)
            checkpoint = None
            for stage in stages:
                log(f"== Start {stage} ==")
                deffnm = os.path.join(self.workdir, f"{stem}_{stage}")
                structure = self._run_stage(
                    stage, deffnm, structure, topology, checkpoint, logstream
                )
                # Minimization writes no checkpoint; velocities start at anneal/relax.
                checkpoint = None if stage == "min" else f"{deffnm}.cpt"

            os.makedirs("xtcfiles", exist_ok=True)
            xtcfile = os.path.join("xtcfiles", trajectory)
            shutil.copyfile(os.path.join(self.workdir, f"{stem}_prod.xtc"), xtcfile)
            return xtcfile


def gmx_sample(init_structure, ffxml, trajectory, **settings):
    """
    Run one MD sampling with GROMACS.

    Thin wrapper over :class:`GMXCalculator` for callers that only want the
    trajectory and have nothing to record, with the arguments of
    :func:`~imolcraft.calculator.md.md_sample`. ``settings`` takes the same
    keywords as the calculator.

    Returns
    -------
    xtcfile : str
        Path to the ``.xtc`` written by the production run.
    """
    return GMXCalculator(init_structure, **settings).run(ffxml, trajectory)
