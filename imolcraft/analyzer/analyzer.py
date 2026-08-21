#!/usr/bin/env python
from collections import defaultdict

import MDAnalysis
import MDAnalysis.analysis.rdf as mda
import mdtraj as md
import numpy as np

#: Mass of one atomic mass unit in gram.
_AMU_TO_G = 1.66053886e-24

#: Volume of one cubic angstrom in cubic centimetre.
_ANG3_TO_CM3 = 1e-24

#: Bin edges in degrees shared by every angle distribution function.
_ADF_BINS = np.arange(0, 180 + 0.001, 1)

#: Cell parameters accepted by :func:`calc_cellpar_frame`, mapped to their
#: position in ``ts.dimensions``.
CELLPAR_INDICES = {
    "La_A": 0,
    "Lb_A": 1,
    "Lc_A": 2,
    "alpha_deg": 3,
    "beta_deg": 4,
    "gamma_deg": 5,
}


def _build_interrdf(u, elem1, elem2, rmax, dr, only_intermolecular):
    """
    Build the InterRDF analysis of the elem1-elem2 pairs.

    Parameters
    ----------
    u : MDAnalysis.Universe
        MDAnalysis universe object.
    elem1, elem2 : str
        Element symbols of the two atom types.
    rmax : float
        Maximum distance for RDF calculation.
    dr : float
        Bin width for RDF calculation.
    only_intermolecular : bool
        If True, pairs of atoms in the same residue are excluded.

    Returns
    -------
    rdf : MDAnalysis.analysis.rdf.InterRDF
        The analysis object, not run yet.
    """
    kwargs = {"range": (0, rmax), "nbins": int(rmax / dr)}
    if only_intermolecular:
        kwargs["exclude_same"] = "residue"
    return mda.InterRDF(
        u.select_atoms(f"element {elem1}"),
        u.select_atoms(f"element {elem2}"),
        **kwargs,
    )


def _drop_spurious_first_bin(g):
    """
    Zero the first RDF bin when it is unphysically large.

    The innermost bin covers r ~ 0, where the shell volume vanishes and the
    normalisation blows up, so a non-zero count there produces a spike rather
    than a meaningful value. The array is modified in place and returned.
    """
    if g[0] > 1:
        g[0] = 0.0
    return g


def calc_rdf(u: MDAnalysis.Universe, elem1, elem2,
             rmax=8.0, dr=0.01, only_intermolecular=False,
             start=None, stop=None, step=None):
    """
    Calculate averaged radial distribution function (RDF) between two elements.

    Parameters
    ----------
    u : MDAnalysis.Universe
        MDAnalysis universe object.
    elem1 : str
        Element symbol of the first atom type.
    elem2 : str
        Element symbol of the second atom type.
    rmax : float, optional
        Maximum distance for RDF calculation (default is 8.0).
    dr : float, optional
        Bin width for RDF calculation (default is 0.01).
    only_intermolecular : bool, optional
        If True, calculate RDF only for intermolecular pairs (default is False).
        (exclude pairs of atoms in the same residue)
    start : int, optional
        Starting frame index for RDF calculation
        (default is None, which means the first frame).
    stop : int, optional
        Ending frame index for RDF calculation
        (default is None, which means the last frame).
    step : int, optional
        Step size for frame selection in RDF calculation
        (default is None, which means every frame).

    Returns
    -------
    r : numpy.ndarray
        Array of distances.
    g : numpy.ndarray
        Array of RDF values.
    """
    rdf = _build_interrdf(u, elem1, elem2, rmax, dr, only_intermolecular)
    rdf.run(start=start, stop=stop, step=step)
    return rdf.results.bins, _drop_spurious_first_bin(rdf.results.rdf)


def calc_rdf_frame(u: MDAnalysis.Universe, elem1, elem2, rmax=8.0, dr=0.01,
                   only_intermolecular=False, start=None, stop=None, step=None):
    """
    Calculate radial distribution function (RDF) between two elements for each frame.

    Parameters
    ----------
    u : MDAnalysis.Universe
        MDAnalysis universe object.
    elem1 : str
        Element symbol of the first atom type.
    elem2 : str
        Element symbol of the second atom type.
    rmax : float, optional
        Maximum distance for RDF calculation (default is 8.0).
    dr : float, optional
        Bin width for RDF calculation (default is 0.01).
    only_intermolecular : bool, optional
        If True, calculate RDF only for intermolecular pairs (default is False).
        (exclude pairs of atoms in the same residue)
    start : int, optional
        Starting frame index for RDF calculation
        (default is None, which means the first frame).
    stop : int, optional
        Ending frame index for RDF calculation
        (default is None, which means the last frame).
    step : int, optional
        Step size for frame selection in RDF calculation
        (default is None, which means every frame).


    Returns
    -------
    rdf_list : numpy.ndarray
        Array of RDF values for each frame.
    """
    rdf = _build_interrdf(u, elem1, elem2, rmax, dr, only_intermolecular)

    if start is None:
        start = 0
    if stop is None:
        stop = len(u.trajectory)
    if step is None:
        step = 1

    rdf_list = []
    for i_frame in range(start, stop, step):
        rdf.run(frames=[i_frame])
        rdf_list.append(_drop_spurious_first_bin(rdf.results.rdf))
    return np.array(rdf_list)


def _element_pairs(idx_a, idx_b):
    """Every (a, b) atom index pair between two element selections."""
    return [(a, b) for a in idx_a for b in idx_b]


def _join_on_center(pairs_12, pairs_23):
    """
    Join (a, b) and (b, c) pairs that share the central atom b into (a, b, c).

    Indexing the second list by its central atom keeps this linear in the
    number of matches instead of quadratic in the number of pairs.

    Parameters
    ----------
    pairs_12 : numpy.ndarray
        Array of (a, b) atom index pairs.
    pairs_23 : numpy.ndarray
        Array of (b, c) atom index pairs.

    Returns
    -------
    triplets : list
        List of [a, b, c] atom index triplets.
    """
    by_center = defaultdict(list)
    for b, c in pairs_23:
        by_center[b].append(c)
    return [[a, b, c] for a, b in pairs_12 for c in by_center[b]]


def calc_adf_frame(xtcfile, pdbfile, elem1, elem2, elem3, rcut12=3.0, rcut23=3.0,
                   start=None, stop=None, step=None):
    """
    Calculate angle distribution function (ADF) between three elements.
    The function calculates the angle formed by three atoms of different types
    (elem1-elem2-elem3) in a molecular dynamics simulation. It computes the
    distances between pairs of atoms, filters them based on cutoff distances,
    and then calculates the angles formed by the selected pairs. The resulting
    angles are binned into a histogram to create the angle distribution function.

    Parameters
    ----------
    xtcfile : str
        Path to the XTC file.
    pdbfile : str
        Path to the PDB file.
    elem1 : str
        Element symbol of the first atom type.
    elem2 : str
        Element symbol of the second atom type.
    elem3 : str
        Element symbol of the third atom type.
    rcut12 : float, optional
        Cutoff distance for the first pair of elements (default is 3.0).
    rcut23 : float, optional
        Cutoff distance for the second pair of elements (default is 3.0).
    start : int, optional
        Starting frame index for ADF calculation
        (default is None, which means the first frame).
    stop : int, optional
        Ending frame index for ADF calculation
        (default is None, which means the last frame).
    step : int, optional
        Step size for frame selection in ADF calculation
        (default is None, which means every frame).

    Returns
    -------
    prob_123 : numpy.ndarray
        Array of angle distribution function values for each frame.
    """
    # angle elem1-elem2-elem3
    # mdtraj works in nm, the cutoffs are given in angstrom
    rcut12 /= 10.0
    rcut23 /= 10.0

    t = md.load(xtcfile, top=pdbfile)
    t = t[start:stop:step]  # select frames

    elem1_idx = t.topology.select(f"element {elem1}")
    elem2_idx = t.topology.select(f"element {elem2}")
    elem3_idx = t.topology.select(f"element {elem3}")

    pairs_1_2 = _element_pairs(elem1_idx, elem2_idx)
    pairs_2_3 = _element_pairs(elem2_idx, elem3_idx)
    dists_1_2 = md.compute_distances(t, pairs_1_2, periodic=True)
    dists_2_3 = md.compute_distances(t, pairs_2_3, periodic=True)

    # keep only the pairs closer than the cutoff, frame by frame, then join
    # them on their shared central atom
    arr_1_2 = np.array(pairs_1_2)
    arr_2_3 = np.array(pairs_2_3)
    triplets = [
        _join_on_center(arr_1_2[d_1_2 < rcut12], arr_2_3[d_2_3 < rcut23])
        for d_1_2, d_2_3 in zip(dists_1_2, dists_2_3)
    ]

    angles_list = [
        np.rad2deg(md.compute_angles(t[i], triplets[i], periodic=True, opt=True))
        for i in range(t.n_frames)
    ]
    return np.array(
        [np.histogram(ang, bins=_ADF_BINS, density=True)[0] for ang in angles_list]
    )


def calc_adf(xtcfile, pdbfile, elem1, elem2, elem3, rcut12=3.0, rcut23=3.0,
             start=None, stop=None, step=None):
    """
    Calculate angle distribution function (ADF) between three elements.
    The function calculates the angle formed by three atoms of different types
    (elem1-elem2-elem3) in a molecular dynamics simulation. It computes the
    distances between pairs of atoms, filters them based on cutoff distances,
    and then calculates the angles formed by the selected pairs. The resulting
    angles are binned into a histogram to create the angle distribution function.

    Parameters
    ----------
    xtcfile : str
        Path to the XTC file.
    pdbfile : str
        Path to the PDB file.
    elem1 : str
        Element symbol of the first atom type.
    elem2 : str
        Element symbol of the second atom type.
    elem3 : str
        Element symbol of the third atom type.
    rcut12 : float, optional
        Cutoff distance for the first pair of elements (default is 3.0).
    rcut23 : float, optional
        Cutoff distance for the second pair of elements (default is 3.0).
    start : int, optional
        Starting frame index for ADF calculation
        (default is None, which means the first frame).
    stop : int, optional
        Ending frame index for ADF calculation
        (default is None, which means the last frame).
    step : int, optional
        Step size for frame selection in ADF calculation
        (default is None, which means every frame).

    Returns
    -------
    deg_123 : numpy.ndarray
        Array of angles in degrees.
    prob_123 : numpy.ndarray
        Array of angle distribution function values for each frame.
    """
    prob_123 = calc_adf_frame(
        xtcfile, pdbfile, elem1, elem2, elem3, rcut12=rcut12, rcut23=rcut23,
        start=start, stop=stop, step=step
    )
    return _ADF_BINS[:-1], np.mean(prob_123, axis=0)


def calc_density_frame(u: MDAnalysis.Universe):
    """
    Calculate density for each frame in the trajectory.
    The density is calculated using the formula:
    density = (total mass of atoms) / (volume of the system)
    The volume is obtained from the dimensions of the simulation box.
    The density is returned in g/cm^3.

    Parameters
    ----------
    u : MDAnalysis.Universe
        MDAnalysis universe object.

    Returns
    -------
    density : numpy.ndarray
        Array of density values for each frame.
    """
    total_mass = sum(u.atoms.masses)
    return np.array(
        [
            (total_mass * _AMU_TO_G) / (ts.volume * _ANG3_TO_CM3)
            for ts in u.trajectory
        ]
    )


def calc_density(u: MDAnalysis.Universe):
    """
    Calculate average density of the system.
    The density is calculated using the formula:
    density = (total mass of atoms) / (volume of the system)
    The volume is obtained from the dimensions of the simulation box.
    The density is returned in g/cm^3.

    Parameters
    ----------
    u : MDAnalysis.Universe
        MDAnalysis universe object.
    Returns
    -------
    density : float
        Average density of the system in g/cm^3.
    """
    return np.mean(calc_density_frame(u))  # density (g cm-3)


def calc_cellpar_frame(u: MDAnalysis.Universe, target="all"):
    """
    Calculate cell parameters for each frame in the trajectory.
    The cell parameters are extracted from the dimensions of the simulation box.

    Parameters
    ----------
    u : MDAnalysis.Universe
        MDAnalysis universe object.
    target : str, optional
        Target cell parameter to extract. Options are:
        'all' (default), 'La_A', 'Lb_A', 'Lc_A', 'alpha_deg', 'beta_deg', 'gamma_deg'.
        'all' returns all cell parameters.
        'La_A', 'Lb_A', 'Lc_A' return the lengths of the cell vectors.
        'alpha_deg', 'beta_deg', 'gamma_deg' return the angles between the cell vectors
        in degrees.

    Returns
    -------
    cellpar : numpy.ndarray
        Array of cell parameters for each frame.
        If target is 'all', shape is (n_frames, 6).
        Otherwise shape is (n_frames,).
    """
    if target == "all":
        idx = range(0, 6)
    elif target in CELLPAR_INDICES:
        idx = CELLPAR_INDICES[target]
    else:
        names = ["all", *CELLPAR_INDICES]
        listed = ", ".join(repr(n) for n in names[:-1]) + f" or {names[-1]!r}"
        raise ValueError(f"target must be {listed}")

    return np.array([np.array(ts.dimensions)[idx] for ts in u.trajectory])
