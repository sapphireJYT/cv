#!usr/bin/env python

# EN.600.661 HW #2
#
# Usage: python [files]
#
# Driver program 
#
# Author: Yating Jing <yating@jhu.edu>
#         2015-10-12

import cv2
import matplotlib.pyplot as plt
from detect_features import *
from match_features import *
from ssift_descriptor import *
from match_ssifts import *
from compute_affine_xform import *
from compute_proj_xform import *

def main():
    # Read input images
    path1 = 'bikes1.png'
    path2 = 'bikes2.png'
    image1 = cv2.imread(path1, 0)
    image2 = cv2.imread(path2, 0)

    # Make sure the images are of same size
    (m, n) = image1.shape
    if image2.shape != image1.shape:
        image2 = cv2.resize(image2, (n, m))
    
    # Harris corner detection, find points of interest
    features1 = detect_features(image1)
    features2 = detect_features(image2)

    # Describe features as Simple-SIFT
    ssift1 = ssift_descriptor(features1, image1)
    ssift2 = ssift_descriptor(features2, image2)
    
    # Feature Matching 
    # matches = match_features(features1, features2, image1, image2)
    matches = match_ssifts(features1, features2, ssift1, ssift2)

    # Concatenate two RGB images side by side
    img1 = plt.imread(path1)
    img2 = plt.imread(path2)
    if img2.shape != img1.shape:
        img2 = cv2.resize(img2, (n, m))
    SbS = np.concatenate((img1, img2), axis=1)

    # Display the matches
    for (f1, f2) in matches:
        (y1, x1) = features1[f1]
        (y2, x2) = features2[f2]
        x2 +=n # offset by the width of the image on the left
        plt.plot(x1, y1, 'ob')
        plt.plot(x2, y2, 'ob')
        cv2.line(SbS, (x1, y1), (x2, y2), (0, 255, 0))

    plt.axis('off')
    plt.imshow(SbS)
    plt.show()

    # Compute affine transformation matrix that transforms image1 to image2
    affine_xform = compute_affine_xform(matches, features1, features2, image1, image2)

    # Compute projective transformation matrix that transforms image1 to image2
    # proj_xform = compute_proj_xform(matches, features1, features2, image1, image2)

    # Pad zeros to image borders
    pad_width = 60
    npad = ((pad_width, pad_width), (pad_width, pad_width))
    img1 = np.pad(image1, pad_width=npad, mode='constant', constant_values=0)
    img2 = np.pad(image2, pad_width=npad, mode='constant', constant_values=0)
    (m, n) = img1.shape

    # Warp image1 using the affine transformation matrix
    warp_img1 = cv2.warpAffine(img1, affine_xform[:2], (n, m))

    # Warp image1 using the fully projective transformation matrix
    # warp_img1 = cv2.warpPerspective(img1, proj_xform, (n, m))

    # Stitch the warped image and image2 together 
    img = cv2.addWeighted(warp_img1, 0.5, img2, 0.5, 0)

    # Display the stitched image
    plt.axis('off')
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.show()

if __name__ == '__main__':
    main()