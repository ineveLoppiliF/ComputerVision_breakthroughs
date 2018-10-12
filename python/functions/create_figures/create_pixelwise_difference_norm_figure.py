# Import libraries
import cv2
import itertools
import numpy as np

# Create the pixelwise difference norm figure
def create_pixelwise_difference_norm_figure(diff_norm_image, object_test_keypoints,
                                            inliers_matches, axis):
    # Create the normalized RGB gray figure
    pixelwise_difference_norm_figure = np.stack(((diff_norm_image * (1/np.amax(diff_norm_image)) * 255).astype(np.uint8),)*3,
                                                -1)

    ## Create the mask of the keypoints to draw
    keypoints_mask = np.zeros((len(object_test_keypoints),), dtype=int)
    for match in inliers_matches:
        keypoints_mask[match.queryIdx] = 1
    keypoints_mask = keypoints_mask.tolist()
    ## Draw keypoints
    rectified_test_keypoints_to_draw = [object_test_keypoint for (object_test_keypoint,
                                                                     keypoints_mask_element) in itertools.zip_longest(object_test_keypoints,
                                                                                                                      keypoints_mask) if keypoints_mask_element==1]
    pixelwise_difference_norm_figure = cv2.drawKeypoints(pixelwise_difference_norm_figure,
                                                         rectified_test_keypoints_to_draw,
                                                         None)
                                                         #flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # Create final figure
    axis.imshow(pixelwise_difference_norm_figure)
    axis.set_title('Pixelwise difference\nnorm image', size=5)
    axis.tick_params(labelsize=2, width=0.1, length=1)
    #axis.set_aspect(1./axis.get_data_ratio())