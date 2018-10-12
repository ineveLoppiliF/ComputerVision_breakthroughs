# Import libraries
import cv2
import numpy as np

# Compute intersection over union measure
def intersection_over_union_computation(test_image, test_keypoints, dst_vertices,
                                        true_mask_image_path):
    # Load the mask image
    true_mask_image = cv2.imread(true_mask_image_path, cv2.IMREAD_GRAYSCALE)
    ## Rescale the mask image in order to keep proportions and have the
    ## bigger dimension with 1000 pixels
    height, width = true_mask_image.shape[:2]
    if max([height, width])>1000:
        if height>width:
            true_mask_image = cv2.resize(true_mask_image,
                                         (int(1000*(width/height)), 1000),
                                         interpolation = cv2.INTER_CUBIC)
        elif height<width:
            true_mask_image = cv2.resize(true_mask_image,
                                         (1000, int(1000*(height/width))),
                                         interpolation = cv2.INTER_CUBIC)
        else:
            true_mask_image = cv2.resize(true_mask_image,
                                         (1000, 1000),
                                         interpolation = cv2.INTER_CUBIC)
    
    # Create the cluster mask
    cluster_mask_image = mask_creation(test_keypoints, dst_vertices, test_image)
    
    # Convert masks to boolean
    true_mask_image = np.asarray(true_mask_image, dtype=np.bool)
    cluster_mask_image = np.asarray(cluster_mask_image, dtype=np.bool)
    
    # Extract intersection and union
    intersection = true_mask_image*cluster_mask_image
    union = true_mask_image+cluster_mask_image
    
    # Compute intersection over union
    iou_measure = intersection.sum()/float(union.sum())
    
    return iou_measure


def mask_creation(test_keypoints, dst_vertices, image):
    # Reshape and change type of destination vertices
    vertices = dst_vertices.reshape(-1,2).astype(np.int32)
    
    # Make mask
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.drawContours(mask, [vertices], -1, (255, 255, 255), -1, cv2.LINE_AA)
    return mask