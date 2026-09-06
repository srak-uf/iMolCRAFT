#!/usr/bin/env python
"""
Molecular dynamics sampling with the MD software of one's choice.

:class:`MDCalculator` describes one thermodynamic state and runs it with
OpenMM (:class:`~imolcraft.calculator.omm.OpenMMCalculator`) or with GROMACS
(:class:`~imolcraft.calculator.gmx.GMXCalculator`), chosen by its ``software``
setting. It holds the calculator of that software and delegates the run to
it, so the trainer builds, records and restores one kind of object whichever
software does the sampling. The public names of
:mod:`imolcraft.calculator._mdcommon` are re-exported here, so
``from imolcraft.calculator.md import VALID_ENSEMBLES`` keeps working.
"""
from ._mdcommon import (
    MD_LOG_MODES,
    NONBONDED_METHODS,
    VALID_ENSEMBLES,
    resolve_nonbondedmethod,
)
from .gmx import GMXCalculator
from .omm import OpenMMCalculator

__all__ = [
    "MDCalculator",
    "SOFTWARE_SETTINGS",
    "md_sample",
    # re-exported from _mdcommon so imolcraft.calculator.md keeps its names
    "VALID_ENSEMBLES",
    "MD_LOG_MODES",
    "NONBONDED_METHODS",
    "resolve_nonbondedmethod",
]

#: Settings that only make sense for one software and are refused (with a
#: non-default value) when the other one is selected. GMXCalculator accepts
#: and ignores them, which is why they cannot be derived from its SETTINGS.
_OPENMM_ONLY = ("anneal_interval", "rigidWater")

#: Settings each software reads, derived from the SETTINGS of its calculator:
#: ``{software: {setting: default}}``. A setting appearing under both names
#: has the same default under both.
SOFTWARE_SETTINGS = {
    "openmm": dict(OpenMMCalculator.SETTINGS),
    "gromacs": {
        name: default
        for name, default in GMXCalculator.SETTINGS.items()
        if name not in _OPENMM_ONLY
    },
}

_CALCULATORS = {"openmm": OpenMMCalculator, "gromacs": GMXCalculator}


class MDCalculator:
    """
    MD sampling of one thermodynamic state with OpenMM or GROMACS.

    The state is fixed at construction, while the force field and the output
    trajectory change from run to run and are therefore arguments of
    :meth:`run`. The ``software`` setting picks which program samples it and
    the calculator of that software (:attr:`backend`) is built from the other
    settings; :meth:`run`, :meth:`to_dict` and the ``device`` / ``md_log`` /
    ``md_logfile`` attributes are forwarded to it explicitly, nothing else.

    The settings are named exactly as the sampling section of the YAML names
    them, so a sampling block needs no translation to become a calculator.
    :attr:`SETTINGS` is the union of both softwares' settings plus
    ``software``; :data:`SOFTWARE_SETTINGS` says which apply to which. A
    setting of the software that was *not* chosen is dropped when it holds
    its default value and refused with a ``ValueError`` otherwise, so an
    OpenMM sampling block that spells out ``anneal_interval: 100`` still runs
    under GROMACS, while ``ntomp: 4`` under OpenMM is not silently ignored.

    Parameters
    ----------
    init_structure : str
        Structure the run starts from.
    software : {'openmm', 'gromacs'}, optional
        MD software, matched exactly as the QM ``software`` of the crafter
        is (lower case, no normalisation). Default ``"openmm"``.
    **settings
        The settings of the chosen software, see
        :class:`~imolcraft.calculator.omm.OpenMMCalculator` and
        :class:`~imolcraft.calculator.gmx.GMXCalculator`. Units are the same
        for both: nm, K, fs, bar.
    """

    #: Every setting either software takes, with its default, plus the
    #: software itself. What to_dict records depends on the software chosen.
    SETTINGS = {
        "software": "openmm",
        **SOFTWARE_SETTINGS["openmm"],
        **SOFTWARE_SETTINGS["gromacs"],
    }

    def __init__(self, init_structure, software="openmm", **settings):
        if software not in SOFTWARE_SETTINGS:
            raise ValueError(
                f"Unknown software: {software!r}. Must be one of "
                f"{list(SOFTWARE_SETTINGS)}."
            )
        unknown = sorted(set(settings) - set(self.SETTINGS))
        if unknown:
            raise TypeError(
                f"unknown MD settings: {', '.join(unknown)}. Known ones are "
                f"{', '.join(sorted(self.SETTINGS))}"
            )
        relevant = SOFTWARE_SETTINGS[software]
        foreign = sorted(
            name for name, value in settings.items()
            if name not in relevant and value != self.SETTINGS[name]
        )
        if foreign:
            other = next(name for name in SOFTWARE_SETTINGS if name != software)
            noun = f"{other} settings" if len(foreign) > 1 else f"{other} setting"
            raise ValueError(
                f"{', '.join(foreign)}: {noun} that "
                f"{'do' if len(foreign) > 1 else 'does'} not apply to "
                f"software={software!r}; the {software} settings are "
                f"{', '.join(sorted(relevant))}"
            )
        self.init_structure = init_structure
        self.software = software
        self._backend = _CALCULATORS[software](
            init_structure,
            **{name: value for name, value in settings.items() if name in relevant},
        )

    @property
    def backend(self):
        """The :class:`OpenMMCalculator` or :class:`GMXCalculator` doing the MD."""
        return self._backend

    # the three run-time settings the trainer overrides after a restart; they
    # are properties so that an assignment reaches the backend that reads them
    @property
    def device(self):
        """Platform name for OpenMM; anything but ``"CPU"`` means the GPU for GROMACS."""
        return self._backend.device

    @device.setter
    def device(self, value):
        self._backend.device = value

    @property
    def md_log(self):
        """Where the MD progress goes: ``'stdout'``, ``'file'`` or ``'none'``."""
        return self._backend.md_log

    @md_log.setter
    def md_log(self, value):
        self._backend.md_log = value

    @property
    def md_logfile(self):
        """Log file used when ``md_log='file'``."""
        return self._backend.md_logfile

    @md_logfile.setter
    def md_logfile(self, value):
        self._backend.md_logfile = value

    def to_dict(self):
        """
        The complete recipe of this run, ready to be written to a checkpoint.

        ``init_structure``, ``software`` and every setting of that software,
        defaults included, so a run rebuilt from the record stays the same
        even if a default changes later.
        """
        record = {"init_structure": self.init_structure, "software": self.software}
        record.update(
            {name: getattr(self._backend, name) for name in SOFTWARE_SETTINGS[self.software]}
        )
        return record

    @classmethod
    def from_dict(cls, record):
        """
        Rebuild a calculator from what :meth:`to_dict` wrote.

        A record written before the software could be chosen carries no
        ``software`` and is read as an OpenMM run, which is what it was.
        """
        settings = dict(record)
        init_structure = settings.pop("init_structure")
        settings.setdefault("software", "openmm")
        return cls(init_structure, **settings)

    def run(self, ffxml, trajectory):
        """
        Sample this state with the given force field, using the chosen software.

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
        return self._backend.run(ffxml, trajectory)


def md_sample(init_structure, ffxml, trajectory, **settings):
    """
    Run one MD sampling with OpenMM or GROMACS.

    Thin wrapper over :class:`MDCalculator` for callers that only want the
    trajectory and have nothing to record. ``settings`` takes the same
    keywords as the calculator, ``software`` included.

    Returns
    -------
    xtcfile : str
        Path to the trajectory written by the production run.
    """
    return MDCalculator(init_structure, **settings).run(ffxml, trajectory)
