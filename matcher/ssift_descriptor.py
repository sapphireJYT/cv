#!usr/bin/env python

# EN.600.661 HW #2
#
# Usage: python [files]
#
# Construct Simple-SIFT feature for each point of interest 
#
# Author: Yating Jing <yating@jhu.edu>
#         2015-10-15

from __future__ import division
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import filters

def ssift_descriptor(feature_coords, image):
	"""
	Computer Vision 600.461/661 Assignment 2
	Args:
		feature_coords (list of tuples): list of (row,col) tuple feature coordinates from image
		image (numpy.ndarray): The input image to compute ssift descriptors on. Note: this is NOT the image name or image path.
	Returns:
		descriptors (dictionary{(row,col): 128 dimensional list}): the keys are the feature coordinates (row,col) tuple and
										   the values are the 128 dimensional ssift feature descriptors.
	"""
	
	descriptors = dict()

	# Compute Gaussian derivatives at each pixel
	(m, n) = image.shape
	Ix = np.zeros((m, n))
	Iy = np.zeros((m, n))
	filters.gaussian_filter(image, (2, 2), (0, 1), Ix)
	filters.gaussian_filter(image, (2, 2), (1, 0), Iy)

	# Compute image gradient orientation and magnitude
	orientations = np.arctan2(Iy, Ix)
	magnitudes = np.sqrt(Ix**2 + Iy**2)
	hog_range = [i * 2 * np.pi / 8.0 for i in xrange(8)] # range for gradient orientation

	# Construct feature descriptors
	for (i, j) in feature_coords:
		# Ignore the features that are too close to the boundary
		if i < 20 or i > m-21 or j < 20 or j > n-21:
			continue
		
		# Fit a 41x41 window around each feature
		orient_win = orientations[i-20:i+21, j-20:j+21]
		mag_win = magnitudes[i-20:i+21, j-20:j+21]

		grange = [k*10 for k in xrange(4)]
		ssift = list()
		
		for g1 in grange:
			for g2 in grange:
				orients = orient_win[g1:g1+10, g2:g2+10]
				mags = mag_win[g1:g1+10, g2:g2+10]
				hogs = [0] * 8 # 8-element gradient orientation histogram
				
				# Voting each histogram with gradient magnitude
				for y in xrange(10):
					for x in xrange(10):
						orient = orients[x][y]
						mag = mags[x][y] 
					ind = np.argmin(abs(hog_range - orient)) 
					hogs[ind] += mag 

				if len(ssift) == 0:
					ssift = hogs
				else:
					ssift += hogs

		descriptors[(i, j)] = ssift

	# Normalize the descriptor to (somewhat) account for lighting differences
	for coord, ssift in descriptors.items():
		# Normalize the feature descriptor to unit length
		unit_ssift = np.array(ssift) / sum(ssift)
		
		# Threshold each "bin" at the value 0.2
		thres_ssift = [(s if s < 0.2 else 0.2) for s in unit_ssift]
		
		# Re-normalize the feature to unit length
		new_ssift = np.array(thres_ssift) / sum(thres_ssift)
		
		descriptors[coord] = new_ssift

	return descriptors