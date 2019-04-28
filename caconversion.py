#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example code to that uses pyEMMA to convert 
a structural ensemble into a set of pairwise 
CA distances for use in the forest scoring method.
"""


import pyemma as pe
from glob import glob
import numpy as np

def get_CA_Distances():
    
    #folders containing the trajectories
    trajs = sorted(glob("- - - -"))

    #topology file
    top = "abeta/topol.gro"

    feat_guide = pe.coordinates.featurizer(top)
    feat_guide.add_distances_ca(excluded_neighbors=0)
   
    inp = pe.coordinates.source(trajs, feat_guide)
    con = inp.get_output()

    combined_data = []
    for x in con:
        for y in x:
             combined_data.append(y)
    return np.array(combined_data)
