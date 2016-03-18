#!usr/bin/env python

# EN.600.661 HW #1  
#
# Usage: python [files]
#
# Locate edges using squared-gradient operator and Sobel mask
#
# Author: yatbear <sapphirejyt@gmail.com>
#         2015-09-16

from __future__ import division
from scipy import signal as sg
import cv2
import numpy as np

def p5(image_in): 
    (m, n) = image_in.shape
    edge_image_out = np.zeros((m, n))

    # Roberts
    # mask_x = [[0, 1], [-1, 0]]
    # mask_y = [[1, 0], [0, -1]]

    # Prewitt
    # mask_x = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
    # mask_y = [[1, 1, 1], [0, 0, 0], [-1, -1, 1]]

    # Sobel 3x3
    mask_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    mask_y = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
    
    # Sobel 5x5
    # mask_x = [[-1, -2, 0, 2, 1], [-2, -3, 0, 3, 2], [-3, -5, 0, 5, 3],
    #             [-2, -3, 0, 3, 2], [-1, -2, 0, 2, 1]]
    # mask_y = [[1, 2, 3, 2, 1], [2, 3, 5, 3, 2], [0, 0, 0, 0, 0], 
    #             [-2, -3, -5, -3, -2], [-1, -2, -3, -2, -1]]
   
    # Finite difference approximation of gradient
    # Implement by convolution
    grad_x = sg.convolve2d(image_in, mask_x, 'valid') / 2
    grad_y = sg.convolve2d(image_in, mask_y, 'valid') / 2
    
    # Gradient magnitue
    edge_image_out = np.sqrt(grad_x**2 + grad_y**2) 

    # Normalize the pixel values to [0, 255]
    emax = np.max(edge_image_out)
    emin = np.min(edge_image_out)
    edge_image_out = (edge_image_out - emin) * 255.0 / (emax - emin)

    # cv2.imshow('img', edge_image_out)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return edge_image_out 