#!/usr/bin/env python
"""
The properties a force field is fitted to or judged by, declared once.

The loss and the validation both need to know which properties exist, what
shape each of them has and what it takes to compute one, so the tables live
here rather than in either of them: this module imports nothing of the package
and is imported by both.
"""

#: Every property that can be compared with a reference, and how it is
#: compared: a "scalar" is one number per frame, a "distribution" a curve of
#: them. The comparison is a weighted difference for the former and a
#: distribution metric for the latter, wherever the comparison happens.
PROPERTY_KINDS = {
    "density_gcm3": "scalar",
    "La_A": "scalar",
    "Lb_A": "scalar",
    "Lc_A": "scalar",
    "rdf": "distribution",
    "adf": "distribution",
    "dself_cm2s": "scalar",
}

#: Keys a property needs to be computed from a trajectory, besides the
#: reference it is compared with.
PROPERTY_KEYS = {
    "density_gcm3": (),
    "La_A": (),
    "Lb_A": (),
    "Lc_A": (),
    "rdf": ("elem1", "elem2", "rcut12_A"),
    "adf": ("elem1", "elem2", "elem3", "rcut12_A", "rcut23_A"),
    "dself_cm2s": ("select",),
}

#: Properties the thermodynamic perturbation cannot fit, and which are
#: therefore only available to validation: it reweights the configurations a
#: trajectory stored, which says nothing about how fast they interconvert, so
#: a diffusion coefficient has to be measured on the trajectory as it was run.
VALIDATION_ONLY_PROPERTIES = ("dself_cm2s",)

#: Targets given as a single reference number.
SCALAR_TARGETS = tuple(
    name
    for name, kind in PROPERTY_KINDS.items()
    if kind == "scalar" and name not in VALIDATION_ONLY_PROPERTIES
)

#: Targets given as a reference distribution function.
DISTRIBUTION_TARGETS = tuple(
    name
    for name, kind in PROPERTY_KINDS.items()
    if kind == "distribution" and name not in VALIDATION_ONLY_PROPERTIES
)

#: Keys every target must provide. Scalar targets carry the reference and its
#: weight directly, while rdf / adf carry one such block per named pair or
#: triplet.
REQUIRED_TARGET_KEYS = {
    name: ("gt", "weight") if kind == "scalar" else PROPERTY_KEYS[name]
    for name, kind in PROPERTY_KINDS.items()
    if name not in VALIDATION_ONLY_PROPERTIES
}

#: Keys every validation block must provide besides ``property``. A validated
#: distribution always needs its reference, since what it records is the
#: distance to it, while a scalar can be monitored without one.
REQUIRED_VALIDATION_KEYS = {
    name: PROPERTY_KEYS[name] + (("gt",) if kind == "distribution" else ())
    for name, kind in PROPERTY_KINDS.items()
}
