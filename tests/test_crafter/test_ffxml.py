from imolcraft.crafter.ffxml import check_vsite, delvsite_pdb
import os
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
    """EP レコードだけが削除され、他の行はそのまま残る"""
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
    """空行があっても IndexError にならず、空行は保持される"""
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
