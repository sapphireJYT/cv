#!usr/bin/env python

# EN.600.661 HW #2
#
# Usage: python [files]
#
# NCC matching
# Match features based on maximized normalized cross-correlation criterion
#
# Author: yatbear <sapphirejyt@gmail.com>
#         2015-10-12

import numpy as np

def match_features(feature_coords1, feature_coords2, image1, image2):
    """
    Computer Vision 600.461/661 Assignment 2
    Args:
        feature_coords1 (list of tuples): list of (row,col) tuple feature coordinates from image1
        feature_coords2 (list of tuples): list of (row,col) tuple feature coordinates from image2
        image1 (numpy.ndarray): The input image corresponding to features_coords1
        image2 (numpy.ndarray): The input image corresponding to features_coords2
    Returns:
        matches (list of tuples): list of index pairs of possible matches. For example, if the 4-th feature in feature_coords1 and the 0-th feature
                                  in feature_coords2 are determined to be matches, the list should contain (4,0).
    """
    
    matches = list()

    '''
    # Minimize Sum-of-Squared-Distances 
    matches1 = list()

    for (f1, (i1, j1)) in enumerate(feature_coords1):
        min_ssd = float('inf')
        couple = tuple()
        
        for (f2, (i2, j2)) in enumerate(feature_coords2):
            win2 = image2[i2-2:i2+3, j2-2:j2+3]
            d = win2 - image1[i1][j1]
            ssd = np.sum(d * d)     
            
            if (ssd < min_ssd):
                min_ssd = ssd
                couple = (f1, f2)

        matches1.append(couple)

    matches2 = list()
    
    for (f2, (i2, j2)) in enumerate(feature_coords2):
        min_ssd = float('inf')
        couple = tuple()

        for (f1, (i1, j1)) in enumerate(feature_coords1):
            win1 = image1[i1-2:i1+3, j1-2:j1+3]
            d = win1 - image2[i2][j2]
            ssd = np.sum(d * d)
            
            if (ssd < min_ssd):
                min_ssd = ssd
                couple = (f1, f2)
                
        matches2.append(couple) 
    '''

    # Maximize Normalized Cross-Correlation
    matches1 = list()   
    (m, n) = image1.shape
    r = 8 # window radius
    N = 4.0 * (r+1) * (r+1)

    for (f1, (i1, j1)) in enumerate(feature_coords1):
        # Ignore the features that are too close to the boundary
        if i1 < r or i1 > m-r-1 or j1 < r or j1 > n-r-1:
            continue
        max_ncc = -1
        couple = tuple()
        win1 = image1[i1-r:i1+r+1, j1-r:j1+r+1] 
        diff1 = (win1 - np.mean(win1)) / np.std(win1)
        
        for (f2, (i2, j2)) in enumerate(feature_coords2):
            if i2 < r or i2 > m-r-1 or j2 < r or j2 > n-r-1:
                continue
            win2 = image2[i2-r:i2+r+1, j2-r:j2+r+1]
            diff2 = (win2 - np.mean(win2)) / np.std(win2)
            ncc = np.sum(diff1 * diff2) / N

            if (ncc > max_ncc):
                max_ncc = ncc
                couple = (f1, f2)

        matches1.append(couple)

    matches2 = list()

    for (f2, (i2, j2)) in enumerate(feature_coords2):
        if i2 < r or i2 > m-r-1 or j2 < r or j2 > n-r-1:
            continue
        max_ncc = -1
        couple = tuple()
        win2 = image2[i2-r:i2+r+1, j2-r:j2+r+1]
        diff2 = (win2 - np.mean(win2)) / np.std(win2)
    
        for (f1, (i1, j1)) in enumerate(feature_coords1):
            if i1 < r or i1 > m-r-1 or j1 < r or j1 > n-r-1:
                continue
            win1 = image1[i1-r:i1+r+1, j1-r:j1+r+1]
            diff1 = (win1 - np.mean(win1)) / np.std(win1)
            ncc = np.sum(diff1 * diff2) / N

            if (ncc > max_ncc):
                max_ncc = ncc
                couple = (f1, f2)
                
        matches2.append(couple) 
    
    # Mutual Marriages
    matches = list(set(matches1) & set(matches2))
    
    return matches