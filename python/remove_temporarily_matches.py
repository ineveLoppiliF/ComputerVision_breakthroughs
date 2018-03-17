from support_functions import indexToEliminate
import numpy as np

def remove_temporarily_matches(good_matches,temporary_removed_matches,dst_inliers,index_inliers):
    
    ## Remove temporary the match whose coordinates in the test image are farthest from the centroid
    ## computed considering inliers of the computed homograpy
    # This is done in order to allow RANSAC algorithm to find other homograpies
    remove_mask = np.ones(len(good_matches))
    remove_mask[indexToEliminate(dst_inliers, index_inliers)] = 0
    
    ## Add the point to a buffer of temporary removed ones
    temporary_removed_matches.extend([good_matches[i] for i in range(len(good_matches)) if not remove_mask[i]])
    
    ## Remove the point from the left good matches
    good_matches = [good_matches[i] for i in range(len(good_matches)) if remove_mask[i]] 
    
    return good_matches, temporary_removed_matches    