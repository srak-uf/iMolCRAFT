#!/usr/bin/env python
from collections import defaultdict

import MDAnalysis
import MDAnalysis.analysis.msd as mda_msd
import MDAnalysis.analysis.rdf as mda
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import mdtraj as md
import numpy as np
from MDAnalysis.transformations.nojump import NoJump

#: Public API re-exported by ``imolcraft.analyzer``. Without it the wildcard
#: import in ``__init__.py`` would also leak the imported modules.
__all__ = [
    "CELLPAR_INDICES",
    "calc_adf",
    "calc_adf_frame",
    "calc_cellpar_frame",
    "calc_density",
    "calc_density_frame",
    "calc_dself",
    "calc_msd",
    "calc_rdf",
    "calc_rdf_frame",
]

#: Mass of one atomic mass unit in gram.
_AMU_TO_G = 1.66053886e-24

#: Volume of one cubic angstrom in cubic centimetre.
_ANG3_TO_CM3 = 1e-24

#: Conversion of a diffusion coefficient from angstrom^2/ps to cm^2/s.
#: 1 angstrom^2 = 1e-16 cm^2 and 1 ps = 1e-12 s, so the ratio is 1e-4.
_ANG2_PS_TO_CM2_S = 1e-4

#: Number of degrees of freedom (2 * dimensionality) entering the Einstein
#: relation MSD = 2 * d * D * t, for every ``msd_type`` of ``EinsteinMSD``.
_MSD_DOF = {
    "xyz": 6,
    "xy": 4,
    "xz": 4,
    "yz": 4,
    "x": 2,
    "y": 2,
    "z": 2,
}

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


def _zero_first_bin(g):
    """
    Zero the first RDF bin, which stands for r = 0 and must vanish there.

    The innermost bin starts at r = 0, where the shell volume goes to zero and
    the normalisation blows up, so whatever lands in it is a spike rather than
    a meaningful value.

    Note that this assumes the bin is narrow enough to really represent r ~ 0.
    With a coarse ``dr`` the first bin reaches out to genuine coordination
    shells and zeroing it discards a real peak.

    The array is modified in place and returned.
    """
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
    return rdf.results.bins, _zero_first_bin(rdf.results.rdf)


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
        # copy: the analysis object is reused across frames, so keeping a
        # reference to results.rdf would tie every entry to whatever the last
        # run left there
        rdf_list.append(_zero_first_bin(np.array(rdf.results.rdf)))
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
    rcut12_nm = rcut12 / 10.0
    rcut23_nm = rcut23 / 10.0

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
        _join_on_center(arr_1_2[d_1_2 < rcut12_nm], arr_2_3[d_2_3 < rcut23_nm])
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
    # calc_adf_frame normalises every frame on its own, so each frame carries
    # the same weight here regardless of how many triplets it contained. This
    # is deliberate: the result is the mean of the per-frame distributions,
    # not the distribution of all triplets pooled together.
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
    total_mass = u.atoms.masses.sum()
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


def _ensure_nojump(u: MDAnalysis.Universe):
    """
    Attach the :class:`NoJump` transformation unless the trajectory has one.

    An MSD only makes sense on unwrapped coordinates: in a periodic box an atom
    that leaves one face and re-enters the opposite one looks like a jump of a
    whole box length, which swamps the real displacement. ``NoJump`` undoes
    those jumps on the fly.

    MDAnalysis refuses to add transformations twice, so a trajectory that
    already carries some is left untouched: the caller is assumed to have set
    up the unwrapping (or to have supplied an already unwrapped trajectory).
    """
    if not u.trajectory.transformations:
        u.trajectory.add_transformations(NoJump())


def calc_msd(u: MDAnalysis.Universe, select="all", msd_type="xyz", fft=True,
             nojump=True, start=None, stop=None, step=None):
    """
    Calculate the mean squared displacement (MSD) of a selection.

    The MSD is averaged over the selected atoms and over every time origin
    (Einstein MSD), so ``msd[i]`` is the displacement squared after a lag of
    ``i`` analysed frames.

    Parameters
    ----------
    u : MDAnalysis.Universe
        MDAnalysis universe object.
    select : str, optional
        MDAnalysis selection string for the atoms to follow
        (default is 'all'). A single atom per molecule, e.g. 'element S' for
        sulfolane, tracks the molecular diffusion without the intramolecular
        vibrations of the lighter atoms.
    msd_type : str, optional
        Directions to include, one of 'xyz' (default), 'xy', 'xz', 'yz',
        'x', 'y' or 'z'.
    fft : bool, optional
        If True (default), use the FFT based algorithm, which is much faster
        than the direct double loop over time origins.
    nojump : bool, optional
        If True (default), attach the :class:`NoJump` transformation to the
        trajectory so that the coordinates are unwrapped. This modifies ``u``
        in place, and is skipped when the trajectory already carries
        transformations.
    start : int, optional
        Starting frame index for MSD calculation
        (default is None, which means the first frame).
    stop : int, optional
        Ending frame index for MSD calculation
        (default is None, which means the last frame).
    step : int, optional
        Step size for frame selection in MSD calculation
        (default is None, which means every frame). Beware that ``NoJump``
        can only undo a jump it can recognise as one, i.e. a displacement of
        more than half a box length between two consecutive analysed frames;
        a large ``step`` makes real displacements that big and the unwrapping
        unreliable.

    Returns
    -------
    lagtime_ps : numpy.ndarray
        Array of lag times in picoseconds, starting at 0.
    msd_A2 : numpy.ndarray
        Array of MSD values in angstrom^2.
    """
    if msd_type not in _MSD_DOF:
        listed = ", ".join(repr(t) for t in _MSD_DOF)
        raise ValueError(f"msd_type must be one of {listed}")

    if nojump:
        _ensure_nojump(u)

    # verbose=False asks for a silent run; note that the FFT path of
    # MDAnalysis 2.10 calls tqdm unconditionally, so a progress bar may still
    # reach stderr regardless
    msd = mda_msd.EinsteinMSD(
        u, select=select, msd_type=msd_type, fft=fft, verbose=False
    )
    msd.run(start=start, stop=stop, step=step, verbose=False)

    msd_A2 = msd.results.timeseries
    # the lag is counted in analysed frames, so skipping frames stretches the
    # time between two consecutive points by the same factor
    dt_ps = u.trajectory.dt * (1 if step is None else step)
    lagtime_ps = np.arange(len(msd_A2)) * dt_ps
    return lagtime_ps, msd_A2


def calc_dself(u: MDAnalysis.Universe, select="all", msd_type="xyz", fft=True,
               nojump=True, fit_range=(0.0, 0.5),
               start=None, stop=None, step=None,
               save=False, basename="dself"):
    """
    Calculate the self-diffusion coefficient from the slope of the MSD.

    The Einstein relation ``MSD = 2 * d * D * t`` is used, with ``d`` the
    number of directions included in ``msd_type``, so ``D`` is the slope of a
    straight line fitted to the MSD divided by ``2 * d``.

    The fit is restricted to ``fit_range`` because neither end of the MSD is
    diffusive: the short lags are still ballistic/cage-rattling, and the long
    lags average over so few time origins that they are dominated by noise.

    Parameters
    ----------
    u : MDAnalysis.Universe
        MDAnalysis universe object.
    select : str, optional
        MDAnalysis selection string for the atoms to follow
        (default is 'all'). See :func:`calc_msd`.
    msd_type : str, optional
        Directions to include, one of 'xyz' (default), 'xy', 'xz', 'yz',
        'x', 'y' or 'z'.
    fft : bool, optional
        If True (default), use the FFT based algorithm.
    nojump : bool, optional
        If True (default), unwrap the trajectory with :class:`NoJump`.
        This modifies ``u`` in place. See :func:`calc_msd`.
    fit_range : tuple of float, optional
        Fraction of the lag time axis used for the linear fit, as
        ``(begin, end)`` with values in [0, 1] (default is (0.0, 0.5),
        i.e. the first half of the MSD).
    start : int, optional
        Starting frame index for MSD calculation
        (default is None, which means the first frame).
    stop : int, optional
        Ending frame index for MSD calculation
        (default is None, which means the last frame).
    step : int, optional
        Step size for frame selection in MSD calculation
        (default is None, which means every frame).
    save : bool, optional
        If True, write the MSD curve to ``{basename}_msd.csv`` and plot it
        together with the fitted straight line in ``{basename}_msd.pdf``
        (default is False).
    basename : str, optional
        Stem of the files written when ``save`` is True (default is 'dself').

    Returns
    -------
    dself_cm2s : float
        Self-diffusion coefficient in cm^2/s.
    """
    lagtime_ps, msd_A2 = calc_msd(
        u, select=select, msd_type=msd_type, fft=fft, nojump=nojump,
        start=start, stop=stop, step=step
    )
    slope, intercept = _fit_msd_line(lagtime_ps, msd_A2, fit_range)
    if save:
        _write_msd_csv(lagtime_ps, msd_A2, basename)
        _plot_msd(lagtime_ps, msd_A2, slope, intercept, basename)
    return slope / _MSD_DOF[msd_type] * _ANG2_PS_TO_CM2_S


def _write_msd_csv(lagtime_ps, msd_A2, basename):
    """
    Write the MSD curve to ``{basename}_msd.csv``.

    The figure only shows the curve; the csv keeps the numbers so that the fit
    can be redone or the curve replotted without rerunning the analysis.

    Parameters
    ----------
    lagtime_ps : numpy.ndarray
        Array of lag times in picoseconds.
    msd_A2 : numpy.ndarray
        Array of MSD values in angstrom^2.
    basename : str
        Stem of the file to write.
    """
    np.savetxt(
        f"{basename}_msd.csv",
        np.column_stack((lagtime_ps, msd_A2)),
        delimiter=",",
        header="time_ps,msd_A2",
        comments="",  # keep the header a plain csv line instead of a comment
    )


def _plot_msd(lagtime_ps, msd_A2, slope, intercept, basename):
    """
    Save the MSD and the straight line fitted to it as ``{basename}_msd.pdf``.

    Parameters
    ----------
    lagtime_ps : numpy.ndarray
        Array of lag times in picoseconds.
    msd_A2 : numpy.ndarray
        Array of MSD values in angstrom^2.
    slope : float
        Slope of the fitted line in angstrom^2/ps.
    intercept : float
        Intercept of the fitted line in angstrom^2.
    basename : str
        Stem of the file to write.
    """
    fig, ax = plt.subplots()
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

    ax.plot(lagtime_ps, msd_A2)
    # the fit is dashed so that it stays distinguishable from the MSD itself
    ax.plot(lagtime_ps, lagtime_ps * slope + intercept, linestyle="--")
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("MSD ($\\mathrm{\\AA}^2$)")
    fig.savefig(f"{basename}_msd.pdf", bbox_inches="tight")
    plt.close(fig)


def _fit_msd_line(lagtime_ps, msd_A2, fit_range):
    """
    Straight line fitted to a window of the MSD.

    Parameters
    ----------
    lagtime_ps : numpy.ndarray
        Array of lag times in picoseconds.
    msd_A2 : numpy.ndarray
        Array of MSD values in angstrom^2.
    fit_range : tuple of float
        Fraction ``(begin, end)`` of the axis to fit, with values in [0, 1].

    Returns
    -------
    slope : float
        Slope of the fitted line in angstrom^2/ps.
    intercept : float
        Intercept of the fitted line in angstrom^2.
    """
    begin, end = fit_range
    if not 0.0 <= begin < end <= 1.0:
        raise ValueError(
            f"fit_range must satisfy 0 <= begin < end <= 1, got {fit_range}"
        )

    n_points = len(msd_A2)
    i_begin = int(n_points * begin)
    i_end = int(n_points * end)
    if i_end - i_begin < 2:
        raise ValueError(
            f"fit_range {fit_range} selects {i_end - i_begin} of {n_points} "
            "MSD points, at least 2 are needed for a linear fit"
        )

    slope, intercept = np.polyfit(
        lagtime_ps[i_begin:i_end], msd_A2[i_begin:i_end], 1
    )
    return slope, intercept
