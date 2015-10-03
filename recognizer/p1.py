#!usr/bin/env python

# EN.600.661 HW #1  
#
# Usage: python [files]
#
# Convert a gray-level image to a binary one using a threshold value
#
# Author: Yating Jing <yating@jhu.edu>
#         2015-09-03

import cv2
import numpy as np

def p1(gray_in, thresh_val):
    binary_out = np.zeros(gray_in.shape)
    
    # Thresholding
    low_ind = np.where(gray_in < thresh_val)
    high_ind = np.where(gray_in >= thresh_val)
    binary_out[low_ind] = 0
    binary_out[high_ind] = 1  

    return binary_out