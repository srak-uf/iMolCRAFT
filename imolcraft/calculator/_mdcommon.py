#!/usr/bin/env python
"""
Parts of the MD sampling that do not depend on the MD software.

The ensemble and log-mode names, the nonbonded methods, the annealing
schedule and the MD log destination are the same whether OpenMM
(:mod:`imolcraft.calculator.omm`) or GROMACS (:mod:`imolcraft.calculator.gmx`)
does the sampling, so both take them from here and neither imports the
other. The public names are re-exported by :mod:`imolcraft.calculator.md`.
"""
import contextlib
import os
import sys

from openmm import app

__all__ = [
    "VALID_ENSEMBLES",
    "MD_LOG_MODES",
    "NONBONDED_METHODS",
    "resolve_nonbondedmethod",
]


#: Ensembles an MD calculator knows how to set up.
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
