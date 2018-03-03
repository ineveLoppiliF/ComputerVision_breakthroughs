## Import libraries
import numpy as np

## Function that computes, given a cloud of points in 2D, the one farthest from the centroid, 
## and then returns its index wrt the list of points
def indexToEliminate( dst_inliers, index_inliers ):
    
    ## Compute the centroid, averaging coordinates over each axis
    tot_x = 0 
    tot_y = 0
    for point in dst_inliers:
        tot_x = tot_x + point[0][0]
        tot_y = tot_y + point[0][1]
    centroid = np.array([tot_x/len(dst_inliers), tot_y/len(dst_inliers)])
    
    ## Compute the element farhest from the centroid
    max_dist = 0
    index_max_dist = 0
    for i,point in enumerate(dst_inliers):
        dist = np.linalg.norm(centroid-point[0])
        if max_dist < dist:
            max_dist = dist
            index_max_dist = i
            
    ## Return the farthest point index
    return index_inliers[index_max_dist]