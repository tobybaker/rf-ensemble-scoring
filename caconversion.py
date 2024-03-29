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

def get_CA_Distances(trajectory_folders,topology_file):
    
    #folders containing the trajectories
    trajs = sorted(glob(trajectory_folders))


    feat_guide = pe.coordinates.featurizer(topology_file)
    feat_guide.add_distances_ca(excluded_neighbors=0)
   
    inp = pe.coordinates.source(trajs, feat_guide)
    con = inp.get_output()

    combined_data = []
    for x in con:
        for y in x:
             combined_data.append(y)
    return np.array(combined_data)
