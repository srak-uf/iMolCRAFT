#!/usr/bin/env python
import MDAnalysis
import MDAnalysis.analysis.rdf as mda
import numpy as np


def calc_rdf(xtcfile, pdbfile, elem1, elem2, rmax=8.0, dr=0.01):
    u = MDAnalysis.Universe(pdbfile, xtcfile)
    u_select1 = u.select_atoms(f"element {elem1}")
    u_select2 = u.select_atoms(f"element {elem2}")
    rdf = mda.InterRDF(u_select1, u_select2, range=(0,rmax), nbins=int(rmax/dr))
    rdf.run()
    r = rdf.results.bins
    g = rdf.results.rdf
    return r, g



def calc_rdf_frame(xtcfile, pdbfile, elem1, elem2, rmax=8.0, dr=0.01):
    u = MDAnalysis.Universe(pdbfile, xtcfile)
    u_select1 = u.select_atoms(f"element {elem1}")
    u_select2 = u.select_atoms(f"element {elem2}")
    rdf = mda.InterRDF(u_select1, u_select2, range=(0,rmax), nbins=int(rmax/dr))
    rdf_list = []    
    for i_frame in range(len(u.trajectory)):
        rdf.run(frames=[i_frame])
        g = rdf.results.rdf
        rdf_list.append(g)
    return np.array(rdf_list)

