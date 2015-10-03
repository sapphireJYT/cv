#!usr/bin/env python

# EN.600.661 HW #1  
#
# Usage: python [files]
#
# Compute object attributes and generate the objects database
#
# Author: Yating Jing <yating@jhu.edu>
#         2015-09-10

from __future__ import division
import numpy as np
import math

def p3(labels_in): 
    overlays_out = labels_in

    # Get input image information 
    obj_num = int(np.max(labels_in)) # number of objects
    (m, n) = labels_in.shape # image size
    database_out = []

    for i in xrange(obj_num):
        ind = zip(*np.where(labels_in == i + 1)) # object pixels 
        area = len(ind) # area: zeroth moment

        (x_sum, y_sum) = (0, 0)
        for (y, x) in ind:
            x_sum += x 
            y_sum += y

        # Compute center position
        x_position = int(x_sum / area)
        y_position = int(y_sum / area)
        
        (a, b, c) = (0, 0, 0)
        for (y, x) in ind:
            x_prime = x - x_position
            y_prime = y - y_position
            a += x_prime * x_prime
            b += x_prime * y_prime
            c += y_prime * y_prime
        b = 2 * b

        theta = math.atan2(b, a-c) / 2 # orientation

        # Calculate the minimum moment of inertia
        min_moment = a * math.sin(theta) * math.sin(theta)  \
                    - b * math.sin(theta) * math.cos(theta) \
                    + c * math.cos(theta) * math.cos(theta)

        # Calculate the maximum moment 
        theta2 = math.atan2(-b, c-a) / 2 
        max_moment = a * math.sin(theta2) * math.sin(theta2)  \
                    - b * math.sin(theta2) * math.cos(theta2) \
                    + c * math.cos(theta2) * math.cos(theta2)

        roundness = min_moment / max_moment 

        ind = np.transpose(ind)
        radius = math.sqrt(area / math.pi)
        # radius = max(max(ind[0]) - min(ind[0]), max(ind[1]) - min(ind[1])) / 2

        obj = {}
        obj['object_label'] = i + 1 # index label of the object
        obj['x_position'] = x_position
        obj['y_position'] = y_position
        obj['orientation'] = float('%.3f' % theta)
        obj['min_moment'] = float('%.3f' % min_moment)
        obj['roundness'] = float('%.3f' % roundness)
        obj['area'] = float(area)
        obj['radius'] = int(radius)

        database_out.append(obj)
    
    return [database_out, overlays_out]