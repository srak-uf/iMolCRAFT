#!/usr/bin/env python
import MDAnalysis
import MDAnalysis.analysis.rdf as mda
import numpy as np
import mdtraj as md


def calc_rdf(u: MDAnalysis.Universe, elem1, elem2, rmax=8.0, dr=0.01):
    u_select1 = u.select_atoms(f"element {elem1}")
    u_select2 = u.select_atoms(f"element {elem2}")
    rdf = mda.InterRDF(u_select1, u_select2, range=(0,rmax), nbins=int(rmax/dr))
    rdf.run()
    r = rdf.results.bins
    g = rdf.results.rdf
    if g[0] > 1:
        g[0] = 0.0
    return r, g

def calc_rdf_frame(u: MDAnalysis.Universe, elem1, elem2, rmax=8.0, dr=0.01):
    u_select1 = u.select_atoms(f"element {elem1}")
    u_select2 = u.select_atoms(f"element {elem2}")
    rdf = mda.InterRDF(u_select1, u_select2, range=(0,rmax), nbins=int(rmax/dr))
    rdf_list = []    
    for i_frame in range(len(u.trajectory)):
        rdf.run(frames=[i_frame])
        g = rdf.results.rdf
        if g[0] > 1:
            g[0] = 0.0
        rdf_list.append(g)
    return np.array(rdf_list)

def calc_adf_frame(xtcfile, pdbfile, elem1, elem2, elem3,rcut12=3.0, rcut23=3.0):
    # angle elem1-elem2-elem3
    rcut12 /= 10.0
    rcut23 /= 10.0
    t = md.load(xtcfile, top=pdbfile)
    elem1_idx = t.topology.select(f"element {elem1}")
    elem2_idx = t.topology.select(f"element {elem2}")
    elem3_idx = t.topology.select(f"element {elem3}")
    pairs_1_2 = [(elem1_idx[i], elem2_idx[j]) for i in range(len(elem1_idx)) for j in range(len(elem2_idx))]
    pairs_2_3 = [(elem2_idx[i], elem3_idx[j]) for i in range(len(elem2_idx)) for j in range(len(elem3_idx))]
    dists_1_2 = md.compute_distances(t, pairs_1_2, periodic=True)
    dists_2_3 = md.compute_distances(t, pairs_2_3, periodic=True)
    # 泥臭いコード
    # pairs_1_2_cut = [[p for i, p in enumerate(pairs_1_2) if dists_1_2[itrj][i] < rcut12] for itrj in range(len(t))]
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
                    pairs_1_2_3_cut[itrj].append([pairs_1_2_cut[itrj][i_pair][0], \
                                                pairs_1_2_cut[itrj][i_pair][1], \
                                                pairs_2_3_cut[itrj][j_pair][1]])
    angles_list = []
    for i in range(t.n_frames):
        angles = md.compute_angles(t[i], pairs_1_2_3_cut[i], periodic=True, opt=True)
        angles_list.append(angles)
    angles_list = [np.rad2deg(ang) for ang in angles_list] 
    bins = np.arange(0, 180+0.001, 1)
    prob_123 = np.array([np.histogram(ang, bins=bins, density=True)[0] for ang in angles_list])
    deg_123 = bins[:-1]

    return prob_123


def calc_adf(xtcfile, pdbfile, elem1, elem2, elem3,rcut12=3.0, rcut23=3.0):
    prob_123 = calc_adf_frame(xtcfile, pdbfile, elem1, elem2, elem3, rcut12=rcut12, rcut23=rcut23)
    bins = np.arange(0, 180+0.001, 1)
    deg_123 = bins[:-1]
    prob_123 = np.mean(prob_123, axis=0)
    return deg_123, prob_123

def calc_density_frame(u: MDAnalysis.Universe):
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
    density_frame = calc_density_frame(u)
    density = np.mean(density_frame) # density (g cm-3)
    return density

def calc_cellpar_frame(u: MDAnalysis.Universe, target="all"):
    if target == "all":
        idx = range(0,6)
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
        raise ValueError("target must be 'all', 'La_A', 'Lb_A', 'Lc_A', 'alpha_deg', 'bet_deg' or 'gamma_deg'")
    cellpar = []
    for ts in u.trajectory:
        cellpar_tmp = np.array(ts.dimensions)[idx]
        cellpar.append(cellpar_tmp)
    return np.array(cellpar)

