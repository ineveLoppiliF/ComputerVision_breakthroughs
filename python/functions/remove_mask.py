## Import libraries
import numpy as np
from shapely.geometry import Point

## Create a mask over the left good matches of the ones that are in the polygon
def remove_mask(test_keypoints, good_matches, polygon):
    
    ## Create a mask over the left good matches of the ones that are in the polygon
    in_square_mask = np.zeros(len(good_matches))
    for i in range(len(good_matches)):
        point = Point((test_keypoints[good_matches[i].queryIdx].pt)[0], (test_keypoints[good_matches[i].queryIdx].pt)[1])
        if polygon.contains(point):
            in_square_mask[i] = 1
            
    return in_square_mask