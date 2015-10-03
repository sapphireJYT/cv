#!usr/bin/env python

# EN.600.661 HW #1  
#
# Usage: python [files]
#
# Implement the sequential labeling algorithm that segments  
# a binary image into several connected regions
#
# Author: Yating Jing <yating@jhu.edu>
#         2015-09-06

import numpy as np

def p2(binary_in): 
    # Get the image size
    (m, n) = binary_in.shape

    # Initialization
    new_label = 0 
    labels_out = np.zeros((m, n)) # labeling output, initialized to zeros
    eq_table = [] # equivalence table of indices of equivalent points 

    # Pass 1: Raster scanning starting from index (1, 1)
    # down to the right
    for y in xrange(1, n):
        for x in xrange(1, m):
            # Seeds
            a = binary_in[x][y]
            b = binary_in[x][y-1]
            c = binary_in[x-1][y]
            d = binary_in[x-1][y-1]

            # Labels
            la = -1
            lb = labels_out[x][y-1]
            lc = labels_out[x-1][y]
            ld = labels_out[x-1][y-1]

            if a == 0: 
                la = 0 # background
            else:
                if d == 1:
                    la = ld  
                else:
                    if b == 0 and c == 0: 
                        new_label +=1                     
                        la = new_label     
                    elif b == 0 and c == 1:
                        la = lc
                    elif b == 1 and c == 0:
                        la = lb
                    else:
                        if lb == lc:
                            la = lb
                        else:
                            # Set the label
                            la = lb
                            # Update equivalence table
                            hasEq = False
                            for (i, eq_list) in enumerate(eq_table):
                                if lb in eq_list and lc not in eq_list:
                                    eq_table[i].append(lc)
                                    hasEq = True
                                    break
                                elif lb not in eq_list and lc in eq_list:
                                    eq_table[i].append(lb)
                                    hasEq = True
                                    break
                                elif lb in eq_list and lc in eq_list:
                                    hasEq = True
                                    break
                                    
                            if not hasEq:
                                eq_table.append([lb, lc])
            
            labels_out[x][y] = la
    
    # Remove redundant entries in equivalence table by union operation
    new_eq_table = []
    for i in xrange(len(eq_table)):
        l = eq_table[i]
        for j in xrange(i + 1, len(eq_table)):
            l2 = eq_table[j]
            if list(set(l) & set(l2)) != []:
                l = list(set(l) | set(l2)) 
        new_eq_table.append(l)

    # Pass 2: Resolve equivalence
    for y in xrange(1, n):
        for x in xrange(1, m):
            for (i, eq_list) in enumerate(new_eq_table):
                if labels_out[x][y] in eq_list:
                    labels_out[x][y] = eq_list[0]
                    break

    # Set labels to be consecutive natural numbers
    unique_labels = np.unique(labels_out)
    for (i, label) in enumerate(unique_labels):
        if i != label:
            ind = np.where(labels_out == label)
            labels_out[ind] = i

    return labels_out