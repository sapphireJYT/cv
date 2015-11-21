#!usr/bin/env python

# EN.600.661 HW #3
#
# Usage: python [files]
#
# Scene Recognition with Bag of Words
#
# Author: Yating Jing <yating@jhu.edu>
#         2015-11-14

import cv2
import numpy as np
from sklearn import neighbors

sift = cv2.SIFT()
names = ['buildings', 'food', 'people', 'faces', 'cars', 'trees']
features = list()
descriptors = list()
first = True

for name in names:
    for i in xrange(1, 10):
        # Read training images
        img = cv2.imread('%s/%d.png' % (name, i), 0)

        # Compute SIFT features for each image
        kp, des = sift.detectAndCompute(img, None)
        descriptors.append(des)

        # Collect SIFT features for all training images into one large vector
        if first:
            features = des
            first = False
        else:
            features = np.vstack((features, des))

features = np.float32(features)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# Run K-Means with 1000 cluster centers on the SIFT vectors  
# and store the resulting centers(vocabulary)
ret, label, center = cv2.kmeans(features, 1000, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

bow = list() # bag-of-words encoding vectors
scene_class = list() 
des_ind = 0

# For each training image
for name in names:
    for i in xrange(1, 10):
        # Read training images
        img = cv2.imread('%s/%d.png' % (name, i), 0)

        # Extract SIFT features 
        des = descriptors[des_ind] 
        des_ind += 1

        # Compute a Bag-of-Words encoding vector
        hg = np.zeros(1000) # histogram
        for feat in des:
            min_diff = float('inf')
            min_ind = 0
            # Match feature descriptor with the vocabulary 
            for (j, c) in enumerate(center):
                diff = sum((feat - c)**2)
                if diff < min_diff:
                    min_diff = diff
                    min_ind = j
            # Build the histogram
            hg[min_ind] += 1.0

        # Store the encoding vector for the image
        bow.append(hg)  
        # Store the scene class associated with each image
        scene_class.append(name)

acc = 0.0 # classification accuracy on test set

# For each image in the test set 
for name in names:
    for i in xrange(10, 12):
        # Read test images
        img = cv2.imread('%s/%d.png' % (name, i), 0)
 
        # compute SIFT features, use codebook from training
        kp, des = sift.detectAndCompute(img, None)

        # Compute a Bag-of-Words encoding vector
        hg = np.zeros(1000) # histogram
        for feat in des:
            min_diff = float('inf')
            min_ind = 0
            # Match feature descriptor with the vocabulary 
            for (j, c) in enumerate(center):
                diff = sum((feat - c)**2)
                if diff < min_diff:
                    min_diff = diff
                    min_ind = j
            # Build the histogram
            hg[min_ind] += 1.0

        # Use a nearest-neighbor search to your training features to label the best scene class
        clf = neighbors.KNeighborsClassifier(3, 'uniform')
        clf.fit(bow, scene_class)
        scene = clf.predict(hg)[0].strip()

        if scene == name:
            acc += 1

print 'Test accuracy = ', acc / 12.0