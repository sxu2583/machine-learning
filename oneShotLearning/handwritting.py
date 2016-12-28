# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 15:11:47 2016
Based off of the one shot learning algo demonstrated and explained online
https://www.youtube.com/watch?v=FIjy3lV_KJU
@Teacher: Siraj Raval
@Editor: shihab uddin
"""

import numpy as np
import copy
from scipy.ndimage import imread
from scipy.spatial.distance import cdist

#The Parameters
nrun = 20 #This is the number of classifications being done
fname_label = 'class_labels.txt'

if __name__ == "__main__":
    print ('One shot classification demo with Modified Hausdorff Distance')
    perror = np.zeros(nrun)
    #Now the classificiation step
    for r in range(1, nrun+1):
        rs = str(r)
        if len(rs) == 1:
            rs = '0' + rs
            perror[r-1] = classification_run('run'=rs, LoadImgAsPoints, ModHausdorffDistance, 'cost')
            print (" run " + str(r) + " (error " + str( perror[r-1] ) + "%)")
    total = np.mean(perror)
    print(" Average error " + str(total) + "%")

def LoadImgAsPoints



    