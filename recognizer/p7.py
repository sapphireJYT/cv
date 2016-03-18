#!usr/bin/env python

# EN.600.661 HW #1  
#
# Usage: python [files]
#
# Plot high voted lines  
#
# Author: yatbear <sapphirejyt@gmail.com>
#         2015-09-19

from __future__ import division
import numpy as np
import cv2
import matplotlib.pyplot as plt

def p7(image_in, hough_image_in, hough_thresh): 
    # Darken the original edges 
    edge_inds = np.where(image_in > 0)
    image_in[edge_inds] = 20

    (m, n) = image_in.shape
    line_image_out = image_in.copy()    

    # Thresholding
    ind = np.where(hough_image_in >= hough_thresh)
    param_inds = zip(ind[0], ind[1])

    # Parameter space
    theta_range = np.deg2rad(np.arange(-90.0, 90.0, 1.0))
    diag = np.ceil(np.sqrt((m-1)**2 + (n-1)**2))
    rho_range = np.linspace(-diag, diag, diag*2.0)
    
    for (t, r) in param_inds:
        theta = theta_range[t]
        rho = rho_range[r]
        sin = np.sin(theta)
        cos = np.cos(theta) 

        # Find interceptions with image boundaries
        (x1, y1) = (rho / cos, 0) \
                        if cos != 0 else (np.nan, 0)
        (x2, y2) = (0, -rho / sin) \
                        if sin != 0 else (0, np.nan)
        (x3, y3) = (((m-1)*sin + rho) / cos, m - 1) \
                        if cos != 0 else (np.nan, m - 1)
        (x4, y4) = (n - 1, ((n-1)*cos - rho) / sin) \
                        if sin != 0 else (n - 1, np.nan)

        # Find end points for each line
        endpts = []
        pts = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        for (x, y) in pts:
            if x >= 0 and x <= n - 1:
                if y >= 0 and y <= m - 1:
                    endpts.append((int(x), int(y)))

        if len(endpts) != 2:
            # print len(endpts)
            continue

        color = 60  
        cv2.line(line_image_out, endpts[0], endpts[1], color)
    
    plt.axis("off")
    plt.imshow(line_image_out)
    plt.show()

    return line_image_out

# from p6 import *
# from p5 import *

# path = "pgm/hough_simple_1.pgm"
# image_in = cv2.imread(path, 0)
# edge_image_thresh_out = p5(image_in)
# [edge_image_thresh_out, hough_out] = p6(edge_image_thresh_out, 40)
# p7(edge_image_thresh_out, hough_out, 164)