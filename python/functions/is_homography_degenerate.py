## Import libraries
import numpy as np

SINGULAR_VALUES_THRESHOLD = 0.00000001 # is an (arbitrary) threshold

## Checks if an homography H is degenerate
def is_homography_degenerate(H_inv, corner_scene, discarded_file, discarded_homographies):
    ## Where corner_template = H_inv * corner_scene, and corner_template are
    ## the coordinates of the template image corners
    
    ## Reshape corner_scene list to a more suitable one
    new_corner_scene = corner_scene.reshape(-1, 2)
    
    ## Transform points to projective coordinates padding data
    ## with homogeneous scale factor of 1    
    h, w = new_corner_scene.shape
    if w < 3:
        new_corner_scene = np.insert(new_corner_scene, w, 1, axis=1)
    
    ## Compute the singular values for the homography matrix,
    ## sorted in descending order
    S = np.linalg.svd(H_inv, compute_uv=False)
    
    ## By default the homography is considered degenerate
    is_degenerate = False
    
    ## Instead of checking the rank, more robustly check the ratio between
    ## the largest and the smallest singular value
    singular_values_ratio = False
    twisted_homography = False
    if not S[2]/S[0] > SINGULAR_VALUES_THRESHOLD:
        is_degenerate = True
        singular_values_ratio = True
    else:
        if not all(np.dot(H_inv[2,:],corner) > 0 for corner in new_corner_scene):
            is_degenerate = True
            twisted_homography = True
    
    if is_degenerate:    
        ## Write on debug file if the homography is degenerate
        discarded_homographies[1]+=1
        discarded_file.write("HOMOGRAPHY DISCARDED #"+
                             str(discarded_homographies[0]+
                                 discarded_homographies[1]+
                                 discarded_homographies[2]+
                                 discarded_homographies[3])+
                            " (degenerate homography)\n")
        if singular_values_ratio:
            discarded_file.write("Max ratio: "+str(SINGULAR_VALUES_THRESHOLD)+"\nRatio: "+str(S[2]/S[0])+"\n\n")
        if twisted_homography:
            discarded_file.write("Twisted homography\n\n")
    
    return is_degenerate