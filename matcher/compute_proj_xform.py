#!usr/bin/env python

# EN.600.661 HW #2
#
# Usage: python [files]
#
# Fully-Projective Alignment Model (8 degrees of freedom)
#
# Author: Yating Jing <yating@jhu.edu>
#         2015-10-16

import random
import numpy as np

def compute_proj_xform(matches, features1, features2, image1, image2):
	"""
	Computer Vision 600.461/661 Assignment 2
	Args:
		matches (list of tuples): list of index pairs of possible matches. For example, if the 4-th feature in feature_coords1 and the 0-th feature
								  in feature_coords2 are determined to be matches, the list should contain (4,0).
        features1 (list of tuples) : list of feature coordinates corresponding to image1
        features2 (list of tuples) : list of feature coordinates corresponding to image2
		image1 (numpy.ndarray): The input image corresponding to features_coords1
		image2 (numpy.ndarray): The input image corresponding to features_coords2
	Returns:
		affine_xform (numpy.ndarray): a 3x3 Affine transformation matrix between the two images, computed using the matches.
	"""
	
	proj_xform = np.zeros((3,3))

	# Set s to be the minimum sample size to fit the model
	# 8 unknowns, 2 equations per match, need at least 4 matches 
	s = 5

	# Indices of matching pairs
	match_inds = [i for i in xrange(len(matches))] 
	matched = [] # remember used combos

	N = 30000 # repeat N times
	max_in = -1 # maximum number of inliers

	# RAndom SAmple Consensus Procedure
	for n in xrange(N):
		# Randomly choose s samples
		random.shuffle(match_inds) # in-place shuffle
		inds = match_inds[:s] # take s matches
		# Skip used matching combos
		if inds in matched:
			n -=1 
			continue
		matched.append(inds)

		# Compute affine transformation using the chosen samples
		A = list()
		b = list()
		for i in inds:
			# Extract matching feature coordinates
			(f1, f2) = matches[i]
			(y1, x1) = features1[f1]
			(y2, x2) = features2[f2]
			A.append([x1, y1, 1, 0, 0, 0, 0, 0, 0])
			A.append([0, 0, 0, x1, y1, 1, 0, 0, 0])
			A.append([0, 0, 0, 0, 0, 0, x1, y1, 1])
			b.append(x2)
			b.append(y2)
			b.append(1)

		# Solve for transformation matrix 
		h = np.linalg.lstsq(A, b)[0] 
		h = np.resize(h, (3, 3)) 

		# Count x, y errors for the remaining matches
		ex = list()
		ey = list()
		for j in match_inds:
			if j in inds:
				continue
			(f1, f2) = matches[j]
			(y1, x1) = features1[f1]
			(y2, x2) = features2[f2]		
			[y, x, one] = np.dot(h, [y1, x1, 1]) 

			ex.append((x - x2)**2)
			ey.append((y - y2)**2)

		# Fit a line: ey = a * ex + b
		[a, b] = np.polyfit(ex, ey, 1)

		# Count inliers 
		num_in = 0 # number of inliers
		err_thres = 30 # error threshold to filter inliers
		for x, y in zip(ex, ey):
			# Calculate the distance between each point (ex, ey) to the line
			d = abs(a * x - y + b) / np.sqrt(a * a + 1)
			if d <= err_thres:
				num_in +=1

		# Choose the transformation that yields maximum number of inliers
		if num_in > max_in:
			max_in = num_in
			proj_xform = h

	return proj_xform