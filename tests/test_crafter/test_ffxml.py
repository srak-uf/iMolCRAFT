from imolcraft.crafter.ffxml import (
    check_vsite,
    delvsite_pdb,
    resolve_ion_ffxml,
    DEFAULT_ION_FFXML,
    _write_ion_xml,
)
import os
import re
import pytest


@pytest.mark.parametrize(
    "xmlfile, n_vsite",
    [
        (os.path.join(
            os.path.dirname(__file__),
            "..",
            "data",
            "vsite_average2.xml"), 2),
        (os.path.join(
            os.path.dirname(__file__),
            "..",
            "data",
            "vsite_average3.xml"), 1)
    ]
)
def test_check_vsite(xmlfile, n_vsite):
    assert check_vsite(xmlfile) == n_vsite


def _write_pdb(path, lines):
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def test_delvsite_pdb_removes_only_vsites(tmp_path):
    """Only the EP records are removed; the other lines stay as they are"""
    pdbfile = tmp_path / "vsite.pdb"
    _write_pdb(pdbfile, [
        "REMARK   1 CREATED WITH OPENMM",
        "MODEL        1",
        "ATOM      1  C   MOL A   1       0.000   0.000   0.000  1.00  0.00           C  ",
        "ATOM      2 EP1  MOL A   1       0.500   0.000   0.000  1.00  0.00          EP  ",
        "ATOM      3  H   MOL A   1       1.000   0.000   0.000  1.00  0.00           H  ",
        "TER       4      MOL A   1",
        "ENDMDL",
        "END",
    ])

    delvsite_pdb(str(pdbfile))

    lines = pdbfile.read_text().splitlines()
    assert len(lines) == 7
    assert not any(line.split()[-1] == "EP" for line in lines)
    assert [line.split()[1] for line in lines if line.startswith("ATOM")] == ["1", "3"]


def test_delvsite_pdb_keeps_blank_lines(tmp_path):
    """Blank lines raise no IndexError and are preserved"""
    pdbfile = tmp_path / "blank.pdb"
    _write_pdb(pdbfile, [
        "ATOM      1  C   MOL A   1       0.000   0.000   0.000  1.00  0.00           C  ",
        "",
        "ATOM      2 EP1  MOL A   1       0.500   0.000   0.000  1.00  0.00          EP  ",
        "   ",
        "END",
    ])

    delvsite_pdb(str(pdbfile))

    assert pdbfile.read_text().splitlines() == [
        "ATOM      1  C   MOL A   1       0.000   0.000   0.000  1.00  0.00           C  ",
        "",
        "   ",
        "END",
    ]


@pytest.mark.parametrize(
    "iontype, xmlname",
    [
        ("Gmanr", "Gmanr_Li.xml"),
        ("Madrid", "Madrid_Li.xml"),
        ("Wu-Wick", "Wu-Wick_Li.xml"),
        ("SMM", "SMM_Li.xml"),
        ("madrid", "Madrid_Li.xml"),
    ]
)
def test_resolve_ion_ffxml_bundled(iontype, xmlname):
    """A name resolves to the bundled Li ion force field XML (case-insensitive)"""
    path = resolve_ion_ffxml(iontype, "Li")
    assert os.path.basename(path) == xmlname
    assert os.path.exists(path)


@pytest.mark.parametrize("symbol", ["Na", "K", "Cl"])
def test_resolve_ion_ffxml_falls_back_per_element(symbol):
    """An element the named force field lacks falls back to the amber library"""
    path = resolve_ion_ffxml("Madrid", symbol)
    assert path == resolve_ion_ffxml(None, symbol)
    assert path.endswith(os.path.join("ffxml", DEFAULT_ION_FFXML))
    assert os.path.exists(path)


@pytest.mark.parametrize("iontype", [None, DEFAULT_ION_FFXML])
def test_resolve_ion_ffxml_default(iontype):
    """Both no spec and the old relative path point at the default amber library"""
    path = resolve_ion_ffxml(iontype, "Li")
    assert path.endswith(os.path.join("ffxml", DEFAULT_ION_FFXML))
    assert os.path.exists(path)


def test_resolve_ion_ffxml_is_idempotent():
    """Passing an already resolved path back returns the same path"""
    path = resolve_ion_ffxml("Madrid", "Li")
    assert resolve_ion_ffxml(path, "Li") == path


def _ion_molecule(smiles, charge):
    """Build a monatomic ion with a charge (assumed charge_scale_ion applied)"""
    from openff.toolkit import Molecule, Quantity
    from openff.units import unit

    molecule = Molecule.from_smiles(smiles)
    molecule.partial_charges = Quantity([charge], unit.elementary_charge)
    return molecule


@pytest.mark.parametrize(
    "iontype, symbol, smiles, epsilon, sigma",
    [
        # Madrid replaces only Li; the other ions come from the default amber library
        ("Madrid", "Li", "[Li+]", "0.435136", "0.144"),
        ("Madrid", "Na", "[Na+]", "0.01158968", "0.3328397610972308"),
        ("Madrid", "Cl", "[Cl-]", "1.1087600000000002", "0.3470941405874762"),
        (None, "Li", "[Li+]", "0.0765672", "0.20259036850511314"),
        (None, "Na", "[Na+]", "0.01158968", "0.3328397610972308"),
    ]
)
def test_write_ion_xml_uses_the_parameters_of_the_element(
    tmp_path, iontype, symbol, smiles, epsilon, sigma
):
    """The LJ parameters in the written XML are the ones for that element"""
    outxml = str(tmp_path / "ion.xml")
    molecule = _ion_molecule(smiles, 0.8 if "+" in smiles else -0.8)

    _write_ion_xml(molecule, resolve_ion_ffxml(iontype, symbol), outxml)

    text = open(outxml).read()
    atoms = re.findall(r'<Atom epsilon="([^"]*)" sigma="([^"]*)" type="([^"]*)"', text)
    assert atoms == [(epsilon, sigma, f"ionsff99_tip3p-{symbol}{'+' if '+' in smiles else '-'}")]
    assert text.count("<Type ") == 1


@pytest.mark.parametrize("scale", [1.0, 0.8, 0.7])
def test_write_ion_xml_keeps_the_scaled_charge(tmp_path, scale):
    """The residue charge becomes the passed (charge_scale_ion applied) charge"""
    outxml = str(tmp_path / "ion.xml")
    molecule = _ion_molecule("[Li+]", 1.0 * scale)

    _write_ion_xml(molecule, resolve_ion_ffxml("Madrid", "Li"), outxml)

    charges = re.findall(r'<Atom charge="([^"]*)"', open(outxml).read())
    assert [float(c) for c in charges] == [scale]


def test_write_ion_xml_reports_a_missing_element(tmp_path):
    """Passing an element the bundled force fields lack says which file lacks it"""
    madrid = resolve_ion_ffxml("Madrid", "Li")
    with pytest.raises(ValueError, match="Na"):
        _write_ion_xml(_ion_molecule("[Na+]", 0.8), madrid, str(tmp_path / "na.xml"))
