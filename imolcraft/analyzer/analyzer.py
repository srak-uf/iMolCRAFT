#!/usr/bin/env python
import MDAnalysis
import MDAnalysis.analysis.rdf as mda
import numpy as np
import mdtraj as md


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
    u_select1 = u.select_atoms(f"element {elem1}")
    u_select2 = u.select_atoms(f"element {elem2}")
    if only_intermolecular:
        rdf = mda.InterRDF(u_select1, u_select2, 
                        range=(0, rmax), nbins=int(rmax / dr),
                        exclude_same="residue"
                        )
    else:
        rdf = mda.InterRDF(u_select1, u_select2, range=(0, rmax), nbins=int(rmax / dr))
    rdf.run(start=start, stop=stop, step=step)
    r = rdf.results.bins
    g = rdf.results.rdf
    if g[0] > 1:
        g[0] = 0.0
    return r, g


def calc_rdf_frame(u: MDAnalysis.Universe, elem1, elem2, rmax=8.0, dr=0.01,
                   start=None, stop=None, step=None):
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
    u_select1 = u.select_atoms(f"element {elem1}")
    u_select2 = u.select_atoms(f"element {elem2}")
    rdf = mda.InterRDF(u_select1, u_select2, range=(0, rmax), nbins=int(rmax / dr))
    rdf_list = []
    if start is None:
        start = 0
    if stop is None:
        stop = len(u.trajectory)
    if step is None:
        step = 1
    for i_frame in range(start, stop, step):
        rdf.run(frames=[i_frame])
        g = rdf.results.rdf
        if g[0] > 1:
            g[0] = 0.0
        rdf_list.append(g)
    return np.array(rdf_list)


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
    rcut12 /= 10.0
    rcut23 /= 10.0
    t = md.load(xtcfile, top=pdbfile)
    t = t[start:stop:step]  # select frames
    elem1_idx = t.topology.select(f"element {elem1}")
    elem2_idx = t.topology.select(f"element {elem2}")
    elem3_idx = t.topology.select(f"element {elem3}")
    pairs_1_2 = [
        (elem1_idx[i], elem2_idx[j])
        for i in range(len(elem1_idx))
        for j in range(len(elem2_idx))
    ]
    pairs_2_3 = [
        (elem2_idx[i], elem3_idx[j])
        for i in range(len(elem2_idx))
        for j in range(len(elem3_idx))
    ]
    dists_1_2 = md.compute_distances(t, pairs_1_2, periodic=True)
    dists_2_3 = md.compute_distances(t, pairs_2_3, periodic=True)
    # 泥臭いコード
    # pairs_1_2_cut = [
    #     [p for i, p in enumerate(pairs_1_2) if dists_1_2[itrj][i] < rcut12]
    #     for itrj in range(len(t))
    # ]
    pairs_1_2_cut = []
    for itrj in range(len(t)):
        pairs_1_2_cut.append(np.array(pairs_1_2)[dists_1_2[itrj] < rcut12])
    pairs_2_3_cut = []
    for itrj in range(len(t)):
        pairs_2_3_cut.append(np.array(pairs_2_3)[dists_2_3[itrj] < rcut23])
    # pairs_1_2_3_cut[i_step] = [anglespair_1, anglespair_2, anglespair_3]
    pairs_1_2_3_cut = []
    for itrj in range(len(t)):
        pairs_1_2_3_cut.append([])
        for i_pair in range(len(pairs_1_2_cut[itrj])):
            for j_pair in range(len(pairs_2_3_cut[itrj])):
                if pairs_1_2_cut[itrj][i_pair][1] == pairs_2_3_cut[itrj][j_pair][0]:
                    pairs_1_2_3_cut[itrj].append(
                        [
                            pairs_1_2_cut[itrj][i_pair][0],
                            pairs_1_2_cut[itrj][i_pair][1],
                            pairs_2_3_cut[itrj][j_pair][1],
                        ]
                    )
    angles_list = []
    for i in range(t.n_frames):
        angles = md.compute_angles(t[i], pairs_1_2_3_cut[i], periodic=True, opt=True)
        angles_list.append(angles)
    angles_list = [np.rad2deg(ang) for ang in angles_list]
    bins = np.arange(0, 180 + 0.001, 1)
    prob_123 = np.array(
        [np.histogram(ang, bins=bins, density=True)[0] for ang in angles_list]
    )

    return prob_123


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
    bins = np.arange(0, 180 + 0.001, 1)
    deg_123 = bins[:-1]
    prob_123 = np.mean(prob_123, axis=0)
    return deg_123, prob_123


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
    densities = []
    for ts in u.trajectory:
        volume_ang3 = ts.volume
        volume_cm3 = volume_ang3 * 1e-24
        density_g_cm3 = (total_mass * 1.66053886e-24) / volume_cm3
        densities.append(density_g_cm3)
    density = np.array(densities)
    return density


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
    density_frame = calc_density_frame(u)
    density = np.mean(density_frame)  # density (g cm-3)
    return density


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
        If target is 'La_A', 'Lb_A', 'Lc_A', shape is (n_frames, 1).
        If target is 'alpha_deg', 'beta_deg', 'gamma_deg', shape is (n_frames, 1).
    """
    if target == "all":
        idx = range(0, 6)
    elif target == "La_A":
        idx = 0
    elif target == "Lb_A":
        idx = 1
    elif target == "Lc_A":
        idx = 2
    elif target == "alpha_deg":
        idx = 3
    elif target == "beta_deg":
        idx = 4
    elif target == "gamma_deg":
        idx = 5
    else:
        raise ValueError(
            ("target must be 'all', 'La_A', 'Lb_A', 'Lc_A', 'alpha_deg', "
             "'beta_deg' or 'gamma_deg'")
        )
    cellpar = []
    for ts in u.trajectory:
        cellpar_tmp = np.array(ts.dimensions)[idx]
        cellpar.append(cellpar_tmp)
    return np.array(cellpar)
