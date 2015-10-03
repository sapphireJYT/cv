#!usr/bin/env python

# EN.600.661 HW #1  
#
# Usage: python [files]
#
# Recognize objects from the database
#
# Author: Yating Jing <yating@jhu.edu>
#         2015-09-11

from __future__ import division
from p3 import *
import cv2
import math

def p4(labels_in, database_in): 
    [database_all, overlays_out] = p3(labels_in)

    for obj in database_all:
        for target in database_in:
            # Compare the object with to recognition target
            if abs(target['min_moment']-obj['min_moment']) / obj['min_moment'] < 0.2 and \
                abs(target['roundness']-obj['roundness']) / obj['roundness'] < 0.2:

                # Draw a circle
                center = (obj['x_position'], obj['y_position'])
                radius = obj['radius']
                color = 200
                cv2.circle(overlays_out, center, radius, color)

                # Draw lines to show orientation
                theta = obj['orientation']
                (x, y) = center
                endpt = (int(x + radius * math.cos(theta)), 
                            int(y + radius * math.sin(theta)))
                cv2.line(overlays_out, center, endpt, color)
                # cv2.line(overlays_out, center, (x+radius, y), color)

    return overlays_out