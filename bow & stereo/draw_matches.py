#!usr/bin/env python
import cv2
import numpy as np

def draw_matches(img1, img2, kp1, kp2, matches):
    (n, m) = img1.shape
    img3 = np.zeros((max(n, n), 2*m, 3), np.uint8)
    img3[:n, :m, 0] = img1
    img3[:n, m:, 0] = img2
    img3[:, :, 1] = img3[:, :, 0]
    img3[:, :, 2] = img3[:, :, 0]

    for mat in matches:
        color = tuple([np.random.randint(0, 255) for _ in xrange(3)])
        cv2.line(img3, (int(kp1[mat.queryIdx].pt[0]), int(kp1[mat.queryIdx].pt[1])), 
                    (int(kp2[mat.trainIdx].pt[0] + m), int(kp2[mat.trainIdx].pt[1])), color)
    return img3