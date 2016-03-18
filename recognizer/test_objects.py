#!usr/bin/env python

# EN.600.661 HW #1  
#
# Usage: python [files]
#
# Driver program
#
# Author: yatbear <sapphirejyt@gmail.com>
#         2015-09-14

from p1 import * 
from p2 import *
from p3 import *
from p4 import *
import matplotlib.pyplot as plt

def main():
    two_path = 'pgm/two_objects.pgm'
    many_path = 'pgm/many_objects_1.pgm'
    # many_path = 'pgm/many_objects_2.pgm'

    # Load the image in grayscale
    gray_in = cv2.imread(two_path, 0)
    many_in = cv2.imread(many_path, 0)
 
    # Convert a gray-level image to a binary one
    binary_in = p1(gray_in, 120) 
    many_in = p1(many_in, 120)

    # Segment the binary image into several connected regions
    two_labels = p2(binary_in)
    many_labels = p2(many_in)

    # Display the colored the image
    # plt.imshow(two_labels)
    # plt.show()

    # Computes object attributes, and generate the objects database
    [database_two, overlays] = p3(two_labels)

    # Recognize objects from the database
    overlays_out = p4(many_labels, database_two)

    # Display the results
    plt.axis("off")
    plt.imshow(overlays_out)
    plt.show()

if __name__ == "__main__":
    main()