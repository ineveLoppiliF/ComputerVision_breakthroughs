## Import libraries
import cv2
import numpy as np

## Apply the homography to all test_keypoints in order to plot them
def project_keypoints(test_keypoints, H_inv):
    object_test_keypoints_array = [0, 0]
    for keypoint in test_keypoints:
        object_test_keypoints_array = np.vstack((object_test_keypoints_array, [keypoint.pt[0], keypoint.pt[1]]))
    object_test_keypoints_array = np.delete(object_test_keypoints_array, (0), axis=0)
    object_test_keypoints_array = object_test_keypoints_array.reshape(-1, 1, 2)
    object_test_keypoints_array = cv2.perspectiveTransform(object_test_keypoints_array, H_inv)
    
    object_test_keypoints = list()
    for i,keypoint  in enumerate(object_test_keypoints_array):
        object_test_keypoints.append(cv2.KeyPoint(keypoint[0][0], keypoint[0][1],
                                                  test_keypoints[i].size, test_keypoints[i].angle,
                                                  test_keypoints[i].response, test_keypoints[i].octave,
                                                  test_keypoints[i].class_id))
        
    return object_test_keypoints