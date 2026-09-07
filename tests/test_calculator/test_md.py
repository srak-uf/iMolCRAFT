"""Unit tests for MDCalculator, the facade that picks OpenMM or GROMACS"""
import pytest

from imolcraft.calculator import (
    SOFTWARE_SETTINGS,
    GMXCalculator,
    MDCalculator,
    OpenMMCalculator,
    md,
    omm,
)
from imolcraft.calculator.md import md_sample


# -- software selection ------------------------------------------------------


def test_default_software_is_openmm():
    calc = MDCalculator("x.pdb")
    assert calc.software == "openmm"
    assert type(calc.backend) is OpenMMCalculator
    assert calc.init_structure == "x.pdb"


def test_gromacs_builds_a_gmx_calculator():
    calc = MDCalculator("x.pdb", software="gromacs")
    assert calc.software == "gromacs"
    assert type(calc.backend) is GMXCalculator


def test_the_facade_is_not_the_openmm_calculator():
    assert MDCalculator is not OpenMMCalculator
    assert md.MDCalculator is MDCalculator and omm.OpenMMCalculator is OpenMMCalculator


@pytest.mark.parametrize("software", ["GROMACS", "OpenMM", "lammps", None, ""])
def test_unknown_software_is_refused_as_the_qm_software_is(software):
    """Lower-case exact match, as crafter's `software: psi4 | g16`; no normalisation"""
    with pytest.raises(ValueError, match=r"^Unknown software:") as excinfo:
        MDCalculator("x.pdb", software=software)
    assert "openmm" in str(excinfo.value) and "gromacs" in str(excinfo.value)


# -- settings of the other software -----------------------------------------


def test_a_non_default_setting_of_the_other_software_is_refused():
    with pytest.raises(ValueError, match="anneal_interval") as excinfo:
        MDCalculator("x.pdb", software="gromacs", anneal_interval=25)
    assert "openmm setting" in str(excinfo.value)
    assert "software='gromacs'" in str(excinfo.value)

    with pytest.raises(ValueError, match="ntomp") as excinfo:
        MDCalculator("x.pdb", ntomp=4)
    assert "gromacs setting" in str(excinfo.value)
    assert "software='openmm'" in str(excinfo.value)


def test_a_default_valued_setting_of_the_other_software_is_dropped():
    """An OpenMM YAML that spells out anneal_interval: 100 still runs under GROMACS"""
    calc = MDCalculator("x.pdb", software="gromacs", anneal_interval=100, rigidWater=False)
    assert type(calc.backend) is GMXCalculator
    assert "anneal_interval" not in calc.to_dict()

    calc = MDCalculator("x.pdb", ntomp=None, workdir="gmxfiles")
    assert type(calc.backend) is OpenMMCalculator
    assert "ntomp" not in calc.to_dict()


def test_an_unknown_setting_is_a_type_error_for_both():
    with pytest.raises(TypeError, match="unknown MD settings: temperature"):
        MDCalculator("x.pdb", temperature=350.0)
    with pytest.raises(TypeError, match="unknown MD settings: temperature"):
        MDCalculator("x.pdb", software="gromacs", temperature=350.0)


def test_the_backend_still_validates_its_own_settings():
    with pytest.raises(ValueError, match="Invalid ensemble"):
        MDCalculator("x.pdb", ensemble="npt")
    with pytest.raises(ValueError, match="Invalid ensemble"):
        MDCalculator("x.pdb", software="gromacs", ensemble="npt")
    with pytest.raises(ValueError, match="one entry fewer"):
        MDCalculator("x.pdb", anneal_T=[300, 400, 300], anneal_steps=[50])


# -- SETTINGS / SOFTWARE_SETTINGS -------------------------------------------


def test_settings_is_the_union_of_both_softwares_plus_software():
    """T4: SOFTWARE_SETTINGS partitions MDCalculator.SETTINGS - {software}"""
    union = set(SOFTWARE_SETTINGS["openmm"]) | set(SOFTWARE_SETTINGS["gromacs"])
    assert union == set(MDCalculator.SETTINGS) - {"software"}
    assert MDCalculator.SETTINGS["software"] == "openmm"
    assert set(SOFTWARE_SETTINGS) == {"openmm", "gromacs"}


def test_shared_settings_have_the_same_default_under_both_softwares():
    """T4: a shared key means the same thing, so its default must agree"""
    shared = set(SOFTWARE_SETTINGS["openmm"]) & set(SOFTWARE_SETTINGS["gromacs"])
    assert "pressure_bar" in shared and "temperature_K" in shared
    for name in shared:
        assert SOFTWARE_SETTINGS["openmm"][name] == SOFTWARE_SETTINGS["gromacs"][name], name
        assert MDCalculator.SETTINGS[name] == SOFTWARE_SETTINGS["openmm"][name], name


def test_software_settings_are_derived_from_the_calculators():
    assert SOFTWARE_SETTINGS["openmm"] == OpenMMCalculator.SETTINGS
    assert set(SOFTWARE_SETTINGS["gromacs"]) == set(GMXCalculator.SETTINGS) - {
        "anneal_interval", "rigidWater",
    }
    for name, default in SOFTWARE_SETTINGS["gromacs"].items():
        assert GMXCalculator.SETTINGS[name] == default


# -- delegation ---------------------------------------------------------------


@pytest.mark.parametrize("software", ["openmm", "gromacs"])
def test_run_time_settings_are_forwarded_to_the_backend(software):
    """F5: the trainer reassigns these after a restart, so they must reach the backend"""
    calc = MDCalculator("x.pdb", software=software)
    calc.device = "CUDA"
    calc.md_log = "none"
    calc.md_logfile = "a.log"
    assert calc.backend.device == "CUDA"
    assert calc.backend.md_log == "none"
    assert calc.backend.md_logfile == "a.log"
    assert (calc.device, calc.md_log, calc.md_logfile) == ("CUDA", "none", "a.log")


def test_the_facade_does_not_forward_everything():
    """No __getattr__ magic: internals live on the backend"""
    calc = MDCalculator("x.pdb", anneal_T=[300, 400], anneal_steps=[10])
    assert not hasattr(calc, "anneal_legs")
    assert calc.backend.anneal_legs == [(300.0, 400.0, 10)]
    assert "__getattr__" not in vars(MDCalculator)


@pytest.mark.parametrize("software", ["openmm", "gromacs"])
def test_run_is_delegated_with_the_arguments_and_the_return_value(software, monkeypatch):
    seen = {}

    def fake_run(self, ffxml, trajectory):
        seen["self"], seen["args"] = self, (ffxml, trajectory)
        return "xtcfiles/" + trajectory

    calc = MDCalculator("x.pdb", software=software)
    monkeypatch.setattr(type(calc.backend), "run", fake_run)
    assert calc.run("ff.xml", "s_0.xtc") == "xtcfiles/s_0.xtc"
    assert seen["self"] is calc.backend
    assert seen["args"] == ("ff.xml", "s_0.xtc")


def test_md_sample_reaches_the_chosen_software(monkeypatch):
    """F11: md_sample(..., software='gromacs') goes through GMXCalculator.run"""
    calls = []
    monkeypatch.setattr(
        GMXCalculator, "run", lambda self, ffxml, trajectory: calls.append(("gmx", self.ntomp)) or "g"
    )
    monkeypatch.setattr(
        OpenMMCalculator, "run", lambda self, ffxml, trajectory: calls.append(("omm", None)) or "o"
    )
    assert md_sample("x.pdb", "ff.xml", "s.xtc", software="gromacs", ntomp=2) == "g"
    assert md_sample("x.pdb", "ff.xml", "s.xtc", temperature_K=350.0) == "o"
    assert calls == [("gmx", 2), ("omm", None)]


# -- to_dict / from_dict -------------------------------------------------------


def test_openmm_record_carries_the_openmm_settings_only():
    record = MDCalculator("x.pdb", temperature_K=350.0, pressure_bar=2.0).to_dict()
    assert record["init_structure"] == "x.pdb"
    assert record["software"] == "openmm"
    assert record["temperature_K"] == 350.0 and record["pressure_bar"] == 2.0
    assert set(record) == {"init_structure", "software"} | set(OpenMMCalculator.SETTINGS)
    assert "ntomp" not in record and "workdir" not in record


def test_gromacs_record_carries_the_gromacs_settings_only():
    record = MDCalculator("x.pdb", software="gromacs", ntomp=4, ensemble="isonpt").to_dict()
    assert record["software"] == "gromacs" and record["ntomp"] == 4
    assert set(record) == {"init_structure", "software"} | set(SOFTWARE_SETTINGS["gromacs"])
    assert "anneal_interval" not in record and "rigidWater" not in record


@pytest.mark.parametrize("software", ["openmm", "gromacs"])
def test_record_round_trips(software):
    calc = MDCalculator(
        "x.pdb", software=software, temperature_K=350.0, ensemble="isonpt",
        pressure_bar=5.0, anneal_T=[300, 400, 300], anneal_steps=[10, 20], md_log="none",
    )
    restored = MDCalculator.from_dict(calc.to_dict())
    assert restored.software == software
    assert type(restored.backend) is type(calc.backend)
    assert restored.to_dict() == calc.to_dict()
    assert restored.backend.pressure_bar == 5.0


def test_a_record_written_before_the_software_could_be_chosen_is_openmm():
    """F7: 17 settings, neither software nor pressure_bar, restores as before"""
    old = {
        "init_structure": "x.pdb",
        "rcut_nm": 1.0, "temperature_K": 350.0, "anneal_T": [300, 400],
        "anneal_steps": [10], "anneal_interval": 50, "dt_fs": 2.0, "nstxout": 10,
        "relax_steps": 20, "prod_steps": 30, "ensemble": "isonpt",
        "nonbondedmethod": "LJPME", "dispcorr": True, "useHbondConstraint": False,
        "rigidWater": True, "device": "CPU", "md_log": "none", "md_logfile": None,
    }
    assert len(old) == 18  # init_structure + the 17 settings of the old record
    calc = MDCalculator.from_dict(old)
    assert calc.software == "openmm"
    assert type(calc.backend) is OpenMMCalculator
    assert calc.backend.pressure_bar == 1.0
    for name, value in old.items():
        if name != "init_structure":
            assert getattr(calc.backend, name) == value, name
    assert calc.to_dict() == dict(old, software="openmm", pressure_bar=1.0)


# -- re-exports ---------------------------------------------------------------


def test_md_re_exports_the_shared_names_of_omm():
    from imolcraft.calculator.md import (
        MD_LOG_MODES,
        NONBONDED_METHODS,
        VALID_ENSEMBLES,
        resolve_nonbondedmethod,
    )

    assert VALID_ENSEMBLES is omm.VALID_ENSEMBLES
    assert MD_LOG_MODES is omm.MD_LOG_MODES
    assert NONBONDED_METHODS is omm.NONBONDED_METHODS
    assert resolve_nonbondedmethod is omm.resolve_nonbondedmethod
    assert set(md.__all__) == {
        "MDCalculator", "SOFTWARE_SETTINGS", "md_sample", "VALID_ENSEMBLES",
        "MD_LOG_MODES", "NONBONDED_METHODS", "resolve_nonbondedmethod",
    }
