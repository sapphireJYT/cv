#!usr/bin/env python

# EN.600.661 HW #2
#
# Usage: python [files]
#
# Identify points of interest using Harris Corner Detection
#
# Author: yatbear <sapphirejyt@gmail.com>
#         2015-10-12

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import filters
from nonmaxsuppts import *

def detect_features(image):
    """
    Computer Vision 600.461/661 Assignment 2
    Args:
    	image (numpy.ndarray): The input image to detect features on. Note: this is NOT the image name or image path.
    Returns:
    	pixel_coords (list of tuples): A list of (row,col) tuples of detected feature locations in the image
    """  
    
    pixel_coords = list()
  
    # Compute Gaussian derivatives at each pixel
    (m, n) = image.shape
    Ix = np.zeros((m, n))
    Iy = np.zeros((m, n))
    sigma = 3
    filters.gaussian_filter(image, (sigma, sigma), (0, 1), Ix)
    filters.gaussian_filter(image, (sigma, sigma), (1, 0), Iy)
    
    A = Ix * Ix
    B = Ix * Iy
    C = Iy * Iy
    
    R = np.zeros((m, n))
    r = 3 # Gaussian window radius
    k = 0.05

    # Compute corner response function R
    for j in xrange(r, n-r-1):
        for i in xrange(r, m-r-1):
            # Make a Gaussian window around each pixel
            a = np.sum(A[i-r:i+r+1, j-r:j+r+1])
            b = np.sum(B[i-r:i+r+1, j-r:j+r+1])
            c = np.sum(C[i-r:i+r+1, j-r:j+r+1])

            # Compute second moment matrix H in a 
            # H = [[a, b],[b, c]]

            # Compute eigenvalues of H and calculate response
            # [l1, l2] = np.linalg.eigvals(H)
            # R[i][j] = l1 * l2 - k * (l1 + l2)**2
    
            # Alternately, R = det(H) - k * trace(H)^2 (faster)
            R[i][j] = a * c - b * b  - k * (a + c)**2

    # Normalize R map 
    min_R = np.min(R)
    R = (R - min_R) * 10.0 / (np.max(R) - min_R)

    # Print R map
    # plt.axis('off')
    # plt.imshow(R)
    # plt.show()
    
    # Find local maxima of response function
    radius = 5
    thres = np.mean(R) * 1.2
    pixel_coords = nonmaxsuppts(R, radius, thres)

    return pixel_coords