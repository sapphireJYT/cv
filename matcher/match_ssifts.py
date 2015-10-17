#!usr/bin/env python

# EN.600.661 HW #2
#
# Usage: python [files]
#
# Match Simple-SIFT features using Ratio Test
#
# Author: Yating Jing <yating@jhu.edu>
#         2015-10-15

from __future__ import division
import cv2
import numpy as np
import matplotlib.pyplot as plt

def match_ssifts(feature_coords1, feature_coords2, descriptor1, descriptor2):
    """
    Computer Vision 600.461/661 Assignment 2
    Args:
        feature_coords1 (list of tuples): list of (row,col) tuple feature coordinates from image1
        feature_coords2 (list of tuples): list of (row,col) tuple feature coordinates from image2
        descriptor1 (dictionary): ssift descriptors (dictionary{(row,col): 128 dimensional list}) for feature_coords1
        descriptor2 (dictionary): ssift descriptors (dictionary{(row,col): 128 dimensional list}) for feature_coords2
    Returns:
        matches (list of tuples): list of index pairs of possible matches. For example, if the 4-th feature in feature_coords1 and the 0-th feature
                                  in feature_coords2 are determined to be matches, the list should contain (4,0).
    """
    
    matches = list()

    thres = 0.6
    couple = tuple() # best match 
    
    for (f1, (i1, j1)) in enumerate(feature_coords1):
        if (i1, j1) not in descriptor1:
            continue
        desc1 = np.array(descriptor1[(i1, j1)])
        min_dist = float('inf')  # L2-distance of the top match 
        min_dist2 = float('inf') # L2-distance of the second best match
        
        for (f2, (i2, j2)) in enumerate(feature_coords2):
            if (i2, j2) not in descriptor2:
                continue
            desc2 = np.array(descriptor2[(i2, j2)])

            # Calculate L2-distances of the Simple-SIFT descriptors
            diff = desc1 - desc2
            dist = np.sqrt(np.dot(diff, diff))

            if dist < min_dist:
                min_dist = dist
                couple = (f1, f2)
            elif dist < min_dist2:
                min_dist2 = dist

        ratio = min_dist / min_dist2
        if ratio <= thres:
            matches.append(couple)
    
    return matches