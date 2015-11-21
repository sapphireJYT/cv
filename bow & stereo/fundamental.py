#!usr/bin/env python

# EN.600.661 HW #3
#
# Usage: python [files]
#
# Fundamental Matrix and Epipolar Lines 
#
# Author: Yating Jing <yating@jhu.edu>
#         2015-11-20

import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from draw_matches import draw_matches

path1 = 'HW3_images/hopkins1.JPG'
path2 = 'HW3_images/hopkins2.JPG'
img1 = cv2.imread(path1, 0)
img2 = cv2.imread(path2, 0)

sift = cv2.SIFT()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# Create matcher
bf = cv2.BFMatcher()
matches = bf.match(des1, des2)

matches = sorted(matches, key=lambda x: x.distance)
good_matches = matches[:60]
img3 = draw_matches(img1, img2, kp1, kp2, good_matches)

# Display the matches
plt.axis('off')
plt.imshow(img3)
plt.show()

A = list()
for mat in good_matches:
    u_l, v_l = int(kp1[mat.queryIdx].pt[0]), int(kp1[mat.queryIdx].pt[1])
    u_r, v_r = int(kp2[mat.trainIdx].pt[0]), int(kp2[mat.trainIdx].pt[1])
    A.append([u_l*u_r, u_l*v_r, u_l, v_l*u_r, v_l*v_r, v_l, u_r, v_r, 1])    

# Find the eigenvector f with smallest eigenvalue of matrix (A^T)A
A = np.array(A)
w, v = np.linalg.eig(np.dot(A.T, A))
f = v[np.argmin(w)]
F = np.reshape(f, (3,3)) # fundamental matrix

all_inds = [i for i in xrange(60)]
random.shuffle(all_inds) # in-place shuffle
inds = all_inds[:8] # take 8 features

# Concatenate two RGB images side by side
rgb_img1 = plt.imread(path1)
rgb_img1 = plt.imread(path2)
SbS = np.concatenate((rgb_img1, rgb_img1), axis=1)
(m, n) = img1.shape

for i in inds:
    mat = good_matches[i]
    u_l, v_l = int(kp1[mat.queryIdx].pt[0]), int(kp1[mat.queryIdx].pt[1])

    r = lambda: random.randint(0, 255)
    color = (r(),r(),r())
    cv2.circle(SbS, (v_l, u_l), 10, color, 5)

    # Compute coefficients (a, b, c) of the epipolar line
    x = np.array([u_l, v_l, 1])
    F = F.T
    a = np.dot(F[0], x.T)
    b = np.dot(F[1], x.T)
    c = np.dot(F[2], x.T) 

plt.axis('off')
plt.imshow(SbS)
plt.show()