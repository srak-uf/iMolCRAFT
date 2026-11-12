"""
AMOEBA force fields from Poltype2 / Tinker parameter files.

Poltype2 writes the parameters of a molecule as a Tinker ``.key`` file whose
first line, ``parameters <file>``, points at a header ``.prm`` holding the
global settings of the force field (the anharmonic constants, the vdW
combining rules, the multipole scale factors, ...). Together with the Tinker
``.xyz`` file of the molecule that is everything OpenMM needs, and
:func:`tinker2xml` rewrites it as an OpenMM ForceField XML file, so an AMOEBA
molecule goes through ``app.ForceField(ffxml)`` exactly like a GAFF one.
The files of the molecules of a system are joined with
:func:`merge_amoeba_xml`, since OpenMM cannot load several of them at once.

The unit conversions are the ones ``openmm.app.TinkerFiles`` applies before
handing the values to the AMOEBA force builders, which the XML parsers of
``openmm.app.ForceField`` share, so a System built from the XML reproduces the
one ``TinkerFiles`` builds from the Tinker files themselves.

Only the terms Poltype2 produces for small molecules are translated. A
keyword for an energy term that is not handled here raises
``NotImplementedError`` rather than being dropped, while lines that carry no
parameters (the literature references Poltype2 writes into the key file, the
settings of the Tinker programs) are skipped the way Tinker itself skips them.
"""
import math
import os
import shlex
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Sequence

from openmm.app.element import Element

__all__ = [
    "merge_amoeba_xml",
    "read_tinker_parameters",
    "read_tinker_xyz",
    "tinker2xml",
]

#: kcal/mol -> kJ/mol
KCAL2KJ = 4.184
#: Bohr radius in nanometre, for the dipoles and quadrupoles
BOHR_NM = 0.052917720859
DEG2RAD = math.pi / 180.0

#: Global settings of the force field and the value Tinker assumes when a
#: parameter file leaves one out. Later files override earlier ones, so a key
#: file can change a value of its header.
DEFAULT_SCALARS = {
    "bond-cubic": "0.0",
    "bond-quartic": "0.0",
    "angle-cubic": "0.0",
    "angle-quartic": "0.0",
    "angle-pentic": "0.0",
    "angle-sextic": "0.0",
    "opbendtype": "W-D-C",
    "opbend-cubic": "0.0",
    "opbend-quartic": "0.0",
    "opbend-pentic": "0.0",
    "opbend-sextic": "0.0",
    "torsionunit": "1.0",
    "pitorsunit": "1.0",
    "strbndunit": "1.0",
    "vdwtype": "BUFFERED-14-7",
    "radiusrule": "CUBIC-MEAN",
    "radiustype": "R-MIN",
    "radiussize": "DIAMETER",
    "epsilonrule": "HHG",
    "vdw-12-scale": "0.0",
    "vdw-13-scale": "0.0",
    "vdw-14-scale": "1.0",
    "vdw-15-scale": "1.0",
    "mpole-12-scale": "0.0",
    "mpole-13-scale": "0.0",
    "mpole-14-scale": "1.0",
    "mpole-15-scale": "1.0",
    "polar-12-scale": "0.0",
    "polar-13-scale": "0.0",
    "polar-14-scale": "1.0",
    "polar-15-scale": "1.0",
    "polar-12-intra": "0.0",
    "polar-13-intra": "0.0",
    "polar-14-intra": "0.5",
    "polar-15-intra": "1.0",
    "direct-11-scale": "0.0",
    "direct-12-scale": "1.0",
    "direct-13-scale": "1.0",
    "direct-14-scale": "1.0",
    "mutual-11-scale": "1.0",
    "mutual-12-scale": "1.0",
    "mutual-13-scale": "1.0",
    "mutual-14-scale": "1.0",
}

#: Scale factors OpenMM's AmoebaMultipoleForce hard-codes. The XML cannot
#: carry other values, so a parameter set that asks for them is refused.
OPENMM_MULTIPOLE_SCALES = {
    "mpole-12-scale": 0.0,
    "mpole-13-scale": 0.0,
    "mpole-14-scale": 0.4,
    "mpole-15-scale": 0.8,
    "polar-12-scale": 0.0,
    "polar-13-scale": 0.0,
    "polar-14-scale": 1.0,
    "polar-15-scale": 1.0,
    "polar-14-intra": 0.5,
    "direct-11-scale": 0.0,
    "direct-12-scale": 1.0,
    "direct-13-scale": 1.0,
    "direct-14-scale": 1.0,
    "mutual-11-scale": 1.0,
    "mutual-12-scale": 1.0,
    "mutual-13-scale": 1.0,
    "mutual-14-scale": 1.0,
}

#: Parameter keywords read into lists of fields, and the list each one goes
#: to. ``anglep`` joins ``angle`` because OpenMM tells in-plane angles apart
#: by the out-of-plane bend terms, not by the keyword.
_TERM_KEYWORDS = {
    "bond": "bond",
    "angle": "angle",
    "anglep": "angle",
    "strbnd": "strbnd",
    "opbend": "opbend",
    "torsion": "torsion",
    "pitors": "pitors",
    "vdw": "vdw",
    "vdwpr": "vdwpr",
    "vdwpair": "vdwpr",
    "polarize": "polarize",
    "chgpen": "chgpen",
}

#: Energy terms Tinker knows but this converter does not translate yet.
UNSUPPORTED_KEYWORDS = {
    "tortors", "strtors", "angtors", "ureybrad", "improper", "imptors",
    "charge", "dipole", "quadpole", "chgtrn", "chgflx", "exchpol",
    "repulsion", "dispersion", "bond3", "bond4", "bond5",
    "angle3", "angle4", "angle5",
}


def _tokenize(path: str) -> List[List[str]]:
    """Split a Tinker parameter file into the fields of its non-empty lines."""
    lines = []
    with open(path, "r") as f:
        for raw in f:
            text = raw.strip()
            # the description of an ``atom`` line is quoted and may hold a '#'
            if not text.lower().startswith("atom"):
                text = text.split("#", 1)[0].strip()
            if text:
                lines.append(shlex.split(text))
    return lines


def _parameters_files(keyfile: str, header: Optional[str]) -> List[str]:
    """
    The header files a key file builds on, in the order Tinker reads them.

    ``header`` replaces whatever the key file names. Otherwise each
    ``parameters`` line is followed: the path as written, with ``.prm``
    appended if that is how the file exists, and failing both the file of the
    same name next to the key file, since Poltype2 writes the absolute path of
    the machine it ran on.
    """
    if header is not None:
        return [header]

    found = []
    keydir = os.path.dirname(os.path.abspath(keyfile))
    for fields in _tokenize(keyfile):
        if fields[0].lower() != "parameters" or len(fields) < 2:
            continue
        written = fields[1]
        candidates = [written, written + ".prm"]
        candidates += [os.path.join(keydir, os.path.basename(c)) for c in candidates]
        for candidate in candidates:
            if os.path.isfile(candidate):
                found.append(candidate)
                break
        else:
            raise FileNotFoundError(
                f"{keyfile} reads its header from {written}, which was not "
                f"found, nor {os.path.basename(written)} next to the key file. "
                "Copy the header from the Poltype2 installation (it lives in "
                "ParameterFiles/) or pass it as header=."
            )
    return found


def read_tinker_parameters(paths: Sequence[str]) -> Dict:
    """
    Read Tinker parameter files (``.prm`` / ``.key``) in order.

    Parameters
    ----------
    paths : sequence of str
        Files to read, the header first. Scalar settings of a later file
        override those of an earlier one; parameter lines accumulate.

    Returns
    -------
    dict
        ``scalars`` (setting -> value as written), ``atom`` (type ->
        ``{"class", "symbol", "atomic_number", "mass"}``), ``multipole`` (one
        ``[head, dipole, q11, q21 q22, q31 q32 q33]`` block per entry) and one
        list of field lists per keyword of ``_TERM_KEYWORDS``.

    Raises
    ------
    NotImplementedError
        For an energy term the converter does not translate, including
        charge penetration (AMOEBA+), which OpenMM does not implement.
    ValueError
        For multipole scale factors other than the ones OpenMM hard-codes.
    """
    params = {"scalars": dict(DEFAULT_SCALARS), "atom": {}, "multipole": []}
    for name in set(_TERM_KEYWORDS.values()):
        params[name] = []

    for path in paths:
        lines = _tokenize(path)
        i = 0
        while i < len(lines):
            fields = lines[i]
            keyword = fields[0].lower()
            if keyword in params["scalars"] and len(fields) > 1:
                params["scalars"][keyword] = fields[1]
            elif keyword == "atom":
                # atom <type> <class> <symbol> "<description>" <Z> <mass> <valence>
                # the symbol is a label ("Li+", "CT"); the element is the Z
                params["atom"][fields[1]] = {
                    "class": fields[2],
                    "symbol": fields[3],
                    "atomic_number": int(fields[5]),
                    "mass": fields[6],
                }
            elif keyword in _TERM_KEYWORDS:
                params[_TERM_KEYWORDS[keyword]].append(fields[1:])
            elif keyword == "multipole":
                # the charge line is followed by the dipole and three
                # lines of the lower triangle of the quadrupole
                params["multipole"].append([fields[1:]] + lines[i + 1:i + 5])
                i += 4
            elif keyword in UNSUPPORTED_KEYWORDS:
                raise NotImplementedError(
                    f"{path}: the Tinker keyword '{fields[0]}' is not "
                    "translated to OpenMM XML yet."
                )
            i += 1

    for fields in params["chgpen"]:
        if any(float(x) != 0.0 for x in fields[1:]):
            raise NotImplementedError(
                f"Charge penetration (chgpen {' '.join(fields)}) is an "
                "AMOEBA+ term that OpenMM does not implement."
            )
    for name, value in OPENMM_MULTIPOLE_SCALES.items():
        if abs(float(params["scalars"][name]) - value) > 1e-12:
            raise ValueError(
                f"{name} is {params['scalars'][name]}, but OpenMM fixes it "
                f"at {value} for AMOEBA."
            )
    return params


def read_tinker_xyz(path: str) -> List[Dict]:
    """
    Atoms of a Tinker ``.xyz`` file.

    Returns
    -------
    list of dict
        One entry per atom with its 1-based ``index``, ``symbol``, atom
        ``type`` (a string) and the indices of the atoms it is ``bonded`` to.
    """
    with open(path, "r") as f:
        natoms = int(f.readline().split()[0])
        rows = [line.split() for line in f if line.strip()]
    # an optional second line holds the box: six numbers and nothing else,
    # where an atom line has its element symbol in the second field
    if rows and len(rows[0]) == 6 and _is_number(rows[0][1]):
        rows = rows[1:]
    return [
        {
            "index": int(row[0]),
            "symbol": row[1],
            "type": row[5],
            "bonded": [int(b) for b in row[6:]],
        }
        for row in rows[:natoms]
    ]


def _element_symbol(atomic_number: int) -> str:
    return Element.getByAtomicNumber(atomic_number).symbol


def _is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def _fmt(value: float) -> str:
    return repr(float(value))


def _add_terms(root, params):
    """Write the bonded force elements."""
    s = params["scalars"]

    bonds = ET.SubElement(root, "AmoebaBondForce", {
        "bond-cubic": _fmt(float(s["bond-cubic"]) * 10),
        "bond-quartic": _fmt(float(s["bond-quartic"]) * 100),
    })
    for c1, c2, k, r0 in params["bond"]:
        ET.SubElement(bonds, "Bond", {
            "class1": c1, "class2": c2,
            "length": _fmt(float(r0) * 0.1),
            "k": _fmt(float(k) * KCAL2KJ * 100),
        })

    angles = ET.SubElement(root, "AmoebaAngleForce", {
        name: s[name]
        for name in ("angle-cubic", "angle-quartic", "angle-pentic", "angle-sextic")
    })
    for c1, c2, c3, k, *theta in params["angle"]:
        attrib = {
            "class1": c1, "class2": c2, "class3": c3,
            "k": _fmt(float(k) * KCAL2KJ * DEG2RAD**2),
        }
        # several ideal angles, chosen by the number of attached hydrogens
        for n, theta0 in enumerate(theta, 1):
            attrib[f"angle{n}"] = theta0
        ET.SubElement(angles, "Angle", attrib)

    # OpenMM refuses angle and stretch-bend terms without this element, even
    # when it holds no out-of-plane bend at all
    opbends = ET.SubElement(root, "AmoebaOutOfPlaneBendForce", {
        "type": s["opbendtype"],
        **{name: s[name] for name in
           ("opbend-cubic", "opbend-quartic", "opbend-pentic", "opbend-sextic")},
    })
    for c1, c2, c3, c4, k in params["opbend"]:
        ET.SubElement(opbends, "Angle", {
            "class1": c1, "class2": c2,
            "class3": "" if c3 == "0" else c3,
            "class4": "" if c4 == "0" else c4,
            "k": _fmt(float(k) * KCAL2KJ * DEG2RAD**2),
        })

    stretchbends = ET.SubElement(root, "AmoebaStretchBendForce", stretchBendUnit="1.0")
    strbnd_scale = 10 * KCAL2KJ * DEG2RAD * float(s["strbndunit"])
    for c1, c2, c3, k1, k2 in params["strbnd"]:
        ET.SubElement(stretchbends, "StretchBend", {
            "class1": c1, "class2": c2, "class3": c3,
            "k1": _fmt(float(k1) * strbnd_scale),
            "k2": _fmt(float(k2) * strbnd_scale),
        })

    # Poltype2 lists every torsion in both directions; one is enough, since
    # OpenMM matches a proper torsion either way round
    torsions = ET.SubElement(root, "PeriodicTorsionForce")
    torsion_scale = float(s["torsionunit"]) * KCAL2KJ
    seen = set()
    for fields in params["torsion"]:
        classes = tuple(fields[:4])
        if classes in seen or classes[::-1] in seen:
            continue
        seen.add(classes)
        attrib = {f"class{n}": c for n, c in enumerate(classes, 1)}
        terms = fields[4:]
        for n in range(len(terms) // 3):
            amplitude, phase, periodicity = terms[3 * n:3 * n + 3]
            attrib[f"k{n + 1}"] = _fmt(float(amplitude) * torsion_scale)
            attrib[f"phase{n + 1}"] = _fmt(float(phase) * DEG2RAD)
            attrib[f"periodicity{n + 1}"] = periodicity
        ET.SubElement(torsions, "Proper", attrib)

    if params["pitors"]:
        pitorsions = ET.SubElement(root, "AmoebaPiTorsionForce", piTorsionUnit="1.0")
        for c1, c2, k in params["pitors"]:
            ET.SubElement(pitorsions, "PiTorsion", {
                "class1": c1, "class2": c2,
                "k": _fmt(float(k) * float(s["pitorsunit"]) * KCAL2KJ),
            })


def _add_nonbonded(root, params):
    """Write the vdW and the multipole / polarization elements."""
    s = params["scalars"]

    vdw = ET.SubElement(root, "AmoebaVdwForce", {
        name: s[name]
        for name in ("radiusrule", "radiustype", "radiussize", "epsilonrule",
                     "vdw-13-scale", "vdw-14-scale", "vdw-15-scale")
    })
    vdw.set("type", s["vdwtype"])
    for fields in params["vdw"]:
        cls, radius, epsilon = fields[:3]
        ET.SubElement(vdw, "Vdw", {
            "class": cls,
            "sigma": _fmt(float(radius) * 0.1),
            "epsilon": _fmt(float(epsilon) * KCAL2KJ),
            "reduction": fields[3] if len(fields) > 3 else "0.0",
        })
    for c1, c2, radius, epsilon in params["vdwpr"]:
        ET.SubElement(vdw, "Pair", {
            "class1": c1, "class2": c2,
            "sigma": _fmt(float(radius) * 0.1),
            "epsilon": _fmt(float(epsilon) * KCAL2KJ),
        })

    multipoles = ET.SubElement(root, "AmoebaMultipoleForce")
    quadrupole_scale = BOHR_NM**2 / 3.0
    for head, dipole, q1, q2, q3 in params["multipole"]:
        atom_type, axes, charge = head[0], head[1:-1], head[-1]
        attrib = {"type": atom_type}
        # a negative axis type marks a bisector frame
        for name, axis in zip(("kz", "kx", "ky"), axes):
            attrib[name] = axis
        attrib["c0"] = charge
        for n, d in enumerate(dipole[:3], 1):
            attrib[f"d{n}"] = _fmt(float(d) * BOHR_NM)
        lower = (q1[0], q2[0], q2[1], q3[0], q3[1], q3[2])
        for name, q in zip(("q11", "q21", "q22", "q31", "q32", "q33"), lower):
            attrib[name] = _fmt(float(q) * quadrupole_scale)
        ET.SubElement(multipoles, "Multipole", attrib)

    for atom_type, alpha, thole, *group in params["polarize"]:
        if len(group) > 6:
            raise ValueError(
                f"Atom type {atom_type} has {len(group)} polarization-group "
                "partners, but the OpenMM XML holds at most 6."
            )
        attrib = {
            "type": atom_type,
            "polarizability": _fmt(float(alpha) * 1e-3),
            "thole": thole,
        }
        for n, partner in enumerate(group, 1):
            attrib[f"pgrp{n}"] = partner
        ET.SubElement(multipoles, "Polarize", attrib)


def tinker2xml(
        keyfile: str,
        xyzfile: str,
        outxml: str,
        resname: str = "MOL",
        header: Optional[str] = None,
        ) -> str:
    """
    Write an OpenMM ForceField XML file from a Poltype2 result.

    Parameters
    ----------
    keyfile : str
        Tinker key file written by Poltype2 (``final.key``).
    xyzfile : str
        Tinker xyz file of the same molecule (``final.xyz``). It supplies the
        residue template: the atoms, their types and the bonds.
    outxml : str
        Name of the XML file to write.
    resname : str, optional
        Name of the residue template. Default ``"MOL"``. OpenMM matches
        templates by elements and bonds, so the name is only a label, but
        the XML files loaded into one force field must not share it.
    header : str, optional
        Header parameter file to use instead of the one named on the
        ``parameters`` line of ``keyfile``.

    Returns
    -------
    str
        ``outxml``.
    """
    params = read_tinker_parameters(_parameters_files(keyfile, header) + [keyfile])
    atoms = read_tinker_xyz(xyzfile)

    root = ET.Element("ForceField")

    atom_types = ET.SubElement(root, "AtomTypes")
    for atom_type in sorted({atom["type"] for atom in atoms}, key=int):
        if atom_type not in params["atom"]:
            raise ValueError(f"Atom type {atom_type} of {xyzfile} is not defined in {keyfile}.")
        info = params["atom"][atom_type]
        ET.SubElement(atom_types, "Type", {
            "name": atom_type,
            "class": info["class"],
            "element": _element_symbol(info["atomic_number"]),
            "mass": info["mass"],
        })

    residue = ET.SubElement(ET.SubElement(root, "Residues"), "Residue", name=resname)
    names = {
        atom["index"]:
            _element_symbol(params["atom"][atom["type"]]["atomic_number"]) + str(atom["index"])
        for atom in atoms
    }
    for atom in atoms:
        ET.SubElement(residue, "Atom", name=names[atom["index"]], type=atom["type"])
    for atom in atoms:
        for other in atom["bonded"]:
            if other > atom["index"]:
                ET.SubElement(residue, "Bond", atomName1=names[atom["index"]],
                              atomName2=names[other])

    _add_terms(root, params)
    _add_nonbonded(root, params)

    ET.indent(root)
    ET.ElementTree(root).write(outxml)
    return outxml


#: Attributes of a :func:`tinker2xml` file that hold an atom type number,
#: per element. Class numbers are held by the ``class*`` attributes of any
#: element (``_CLASS_ATTRIBUTES``).
_TYPE_ATTRIBUTES = {
    "Type": ("name",),
    "Atom": ("type",),
    "Multipole": ("type", "kz", "kx", "ky"),
    "Polarize": ("type",) + tuple(f"pgrp{n}" for n in range(1, 7)),
}
_CLASS_ATTRIBUTES = ("class", "class1", "class2", "class3", "class4")


def _shift(number: str, offset: int) -> str:
    """Shift a type/class number away from zero, keeping 0, "" and the sign."""
    if number in ("", "0"):
        return number
    value = int(number)
    return str(value + offset if value > 0 else value - offset)


def _renumber(root, offset: int) -> None:
    """Shift every atom type and class number of a tinker2xml file."""
    for element in root.iter():
        for name in _TYPE_ATTRIBUTES.get(element.tag, ()) + _CLASS_ATTRIBUTES:
            if name in element.attrib:
                element.set(name, _shift(element.get(name), offset))


def merge_amoeba_xml(ffxml_list: Sequence[str], outxml: str) -> str:
    """
    Merge AMOEBA force field XML files into one.

    OpenMM 8.5 cannot load several AMOEBA XML files into one ``ForceField``:
    the second ``<AmoebaAngleForce>`` (or ``<AmoebaOutOfPlaneBendForce>``)
    makes it compare scale factors it never stored and raise an
    ``AttributeError``. The molecules of a system are therefore gathered into
    a single file, one element per force.

    Poltype2 numbers the atom types of every molecule from 401 (unless told
    otherwise by ``atmidx``), and OpenMM requires AMOEBA types to be
    integers, so no prefix can tell the molecules apart. A file whose type
    or class numbers clash with those of the files before it is therefore
    renumbered: all its numbers move up by the same multiple of 100, past
    the largest number taken so far, so 401 becomes 501, 601, ... The first
    file and the files that do not clash keep their numbers, and every shift
    is printed.

    Parameters
    ----------
    ffxml_list : sequence of str
        XML files written by :func:`tinker2xml`, each with its own residue
        name.
    outxml : str
        Name of the merged XML file.

    Returns
    -------
    str
        ``outxml``.

    Raises
    ------
    ValueError
        If two files share a residue name or give the same force different
        global settings.
    """
    merged = ET.Element("ForceField")
    sections = {}
    used_types, used_classes, residues = set(), set(), set()

    for ffxml in ffxml_list:
        root = ET.parse(ffxml).getroot()
        types = {int(t.get("name")) for t in root.iter("Type")}
        classes = {int(t.get("class")) for t in root.iter("Type")}
        if types & used_types or classes & used_classes:
            top = max(used_types | used_classes)
            offset = (top // 100 + 1) * 100 - min(types | classes) // 100 * 100
            _renumber(root, offset)
            print(
                f"merge_amoeba_xml: the atom types of {ffxml} clash with "
                f"earlier files and move up by {offset} "
                f"({min(types)} -> {min(types) + offset})"
            )
            types = {t + offset for t in types}
            classes = {c + offset for c in classes}
        used_types |= types
        used_classes |= classes

        for section in root:
            target = sections.get(section.tag)
            if target is None:
                target = ET.SubElement(merged, section.tag, dict(section.attrib))
                sections[section.tag] = target
            elif target.attrib != section.attrib:
                raise ValueError(
                    f"<{section.tag}> of {ffxml} has the settings "
                    f"{dict(section.attrib)}, the one merged so far "
                    f"{dict(target.attrib)}."
                )
            for child in section:
                if section.tag == "Residues":
                    if child.get("name") in residues:
                        raise ValueError(
                            f"The residue {child.get('name')} of {ffxml} is "
                            "already defined. Give each molecule its own resname."
                        )
                    residues.add(child.get("name"))
                target.append(child)

    ET.indent(merged)
    ET.ElementTree(merged).write(outxml)
    return outxml
