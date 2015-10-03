#!usr/bin/env python

# EN.600.661 HW #1  
#
# Usage: python [files]
#
# Hough Transform for Line Detection
#
# Author: Yating Jing <yating@jhu.edu>
#         2015-09-16

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

# Normalize the pixel values to [0, 255]
def norm(img):
    emax = np.max(img)
    emin = np.min(img)
    norm_img = (img - emin) * 255.0 / (emax - emin)
    return norm_img

def p6(edge_image_in, edge_thresh): 
    (m, n) = edge_image_in.shape
    edge_image_thresh_out = np.zeros((m, n))
    
    # Thresholding
    high_ind = np.where(edge_image_in >= edge_thresh)
    edge_image_thresh_out[high_ind] = 1

    # Find all the edge points
    edge_pts = zip(high_ind[0], high_ind[1])    

    # Hough Transform for line detection

    # Step 1: Quantize parameter space
    theta_range = np.deg2rad(np.arange(-90.0, 90.0, 1.0))
    diag = np.ceil(np.sqrt((m-1)**2 + (n-1)**2)) # image diagonal
    rho_range = np.linspace(-diag, diag, diag*2.0)

    # Step 2: Create accumulator array
    # Step 3: Set A = 0 for all parameters
    A = np.zeros((len(theta_range), len(rho_range)))

    # Step 4: Voting approach
    for (x, y) in edge_pts:
        for (t, theta) in enumerate(theta_range):
            sin = np.sin(theta)
            cos = np.cos(theta)
            r = round(y * cos - x * sin) + diag 
            A[t][r] +=1

    # Scale the image
    hough_out = norm(A)

    return [edge_image_thresh_out, hough_out]