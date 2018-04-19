## Import libraries
import cv2
import numpy as np

from matplotlib import pyplot as plt

## Show the rectified self-similar inliers and the ones eliminated
def print_self_similar_inliers_and_eliminated(rect_test_image, 
                                              object_test_keypoints,
                                              template_image,
                                              template_keypoints,
                                              inliers_matches,
                                              good_matches,
                                              remove_mask,
                                              matches_mask,
                                              good_rescued_self_similar_mask,
                                              index_inliers_matches):
    
    ## Create a mask of the rescued self-similar elements eliminated by the
    ## last homography
    last_removed_self_similar_mask = np.zeros(len(remove_mask))
    for i in range(len(last_removed_self_similar_mask)):
        if remove_mask[i] == good_rescued_self_similar_mask[i] == 1:
            last_removed_self_similar_mask[i]=1
    
    ## Create a mask of the rescued self-similar elements considered as 
    ## inliers by the last homography
    self_similar_inliers = np.zeros(len(matches_mask))
    for i in range(len(matches_mask)):
        if matches_mask[i] == good_rescued_self_similar_mask[index_inliers_matches[i]] == 1:
            self_similar_inliers[i]=1
    
    ## Specify parameters for the function that shows eliminated self-similar
    ## matches
    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green
                       singlePointColor=None,
                       matchesMask=last_removed_self_similar_mask,#.tolist(), # draw only 
                                                                   # eliminated
                                                                   # ones
                       flags=2)
    
    ## Draw clustered rectified matches
    matches_image = cv2.drawMatches(rect_test_image,
                                    object_test_keypoints,
                                    template_image,
                                    template_keypoints,
                                    good_matches,
                                    None,
                                    **draw_params)
    
    ## Show the rectified matches
    plt.imshow(cv2.cvtColor(matches_image, cv2.COLOR_BGR2RGB)), 
    plt.title('Eliminated self-similar matches'), plt.show()
    
    
    ## Specify parameters for the function that shows inliers self-similar
    ## matches
    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green
                       singlePointColor=None,
                       matchesMask=self_similar_inliers,#.tolist(), # draw only 
                                                                   # inliers
                       flags=2)
    
    ## Draw clustered rectified matches
    matches_image = cv2.drawMatches(rect_test_image,
                                    object_test_keypoints,
                                    template_image,
                                    template_keypoints,
                                    inliers_matches,
                                    None,
                                    **draw_params)
    
    ## Show the rectified matches
    plt.imshow(cv2.cvtColor(matches_image, cv2.COLOR_BGR2RGB)), 
    plt.title('Inliers self-similar matches'), plt.show()