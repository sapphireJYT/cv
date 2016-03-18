#!usr/bin/env python

# EN.600.661 HW #3
#
# Usage: python [files]
#
# Stereo Matching and Reconstruction
#
# Author: yatbear <sapphirejyt@gmail.com>
#         2015-11-19

import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

img_l = cv2.imread('HW3_images/scene_l.bmp', 0) # left camera image
img_r = cv2.imread('HW3_images/scene_r.bmp', 0) # right camera image

(n, m) = img_l.shape
min_ncc = np.empty((n, m)) # minimized normalized cross-correlation map
min_ncc[:] = float(sys.maxint)
depth = np.zeros((n, m)) # depth image

b = 100.0  # baseline (mm)
f = 400    # focal length (pixels)  
r = 7      # 15x15 window
max_d = 50 # maximum disparity

num_map = np.zeros((n, m))
denom1_map = np.zeros((n, m)) 
denom2_map = np.zeros((n, m))

for y in xrange(n):
    for x in xrange(m):
        for d in xrange(max_d):
            if x >= d + r and x < m - r and y >= r and y < n - r:
                # Determine disparity using Template Matching
                win_l = img_l[y-r:y+r+1, x-r:x+r+1]
                win_r = img_r[y-r:y+r+1, x-d-r:x-d+r+1] # x_r = x_l - d
                num = np.sum((win_l - win_r)**2)
                denom1 = np.sum(win_l**2)
                denom2 = np.sum(win_r**2)
                
                # Compute normalized cross-correlation 
                ncc = float(sys.maxint)
                if denom1 > 0 and denom2 > 0:
                    ncc = num / np.sqrt(denom1 * denom2)

                # Update minimized normalized cross-correlation map
                if ncc < min_ncc[y, x]:
                    min_ncc[y, x] = ncc
                    depth[y, x] = d * 255.0 / max_d

# Create 3D point cloud
cloud = list()
for j in xrange(n):
    for i in xrange(m):
        d = depth[j, i]
        if d == 0:
            x, y, z = float(sys.maxint), float(sys.maxint), float(sys.maxint)
        else:
            x = b * (2 * i - d) / (2 * d) # x_l + x_r = 2 * x_l - d
            y = b * (j + j) / (2 * d)
            z = b * f / d 
        point = str(x) + ' ' + str(y) + ' ' + str(z)
        cloud.append(point)

# Save the 3D point cloud to file
with open('3D_Point_Cloud.txt', 'w') as f:
    for coord in cloud:
        f.write(coord + '\n')
    f.close()

# Display the depth image
plt.axis('off')
plt.imshow(depth, vmin=0, vmax=255)
plt.show()