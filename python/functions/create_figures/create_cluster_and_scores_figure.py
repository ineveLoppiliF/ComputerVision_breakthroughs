# Import libraries
import cv2
import numpy as np

# Create a scene figure in which the cluster boundary is drawn, together with
# its scores
def create_cluster_and_scores_figure(test_image, dst_vrtx, iou_measure, median,
                                     axis):
    # Create the image
    image = test_image.copy()
    
    # Retrieve the coordinates of the scene cluster boundary
    scene_cluster_boundary_points = dst_vrtx.reshape(-1,2).astype(np.int32)
    
    # Extract centroid of the scene cluster boundary
    scene_cluster_boundary_centroid = centroid(scene_cluster_boundary_points)
    
    # Plot on the test image the boundary
    cv2.polylines(image, np.asarray([scene_cluster_boundary_points],dtype=np.int32),
                  True, (255, 0, 0), 2)
    
    # Plot on the test image the scores near the centroid position
    font = cv2.FONT_HERSHEY_SIMPLEX
    padding = 5
    iou_measure_text = 'IoU: '+str(iou_measure)
    median_text = 'median: '+str(median)
    iou_measure_text_x = np.int32(scene_cluster_boundary_centroid[0]-cv2.getTextSize(iou_measure_text, font, 1, 2)[0][0]/2)
    iou_measure_text_y = np.int32(scene_cluster_boundary_centroid[1])
    median_text_x = np.int32(scene_cluster_boundary_centroid[0]-cv2.getTextSize(median_text, font, 1, 2)[0][0]/2)
    median_text_y = np.int32(scene_cluster_boundary_centroid[1]+cv2.getTextSize(median_text, font, 1, 2)[0][1]+padding)
    cv2.putText(image, iou_measure_text, (iou_measure_text_x, iou_measure_text_y),
                font, 1, (255, 0, 0), 2)
    cv2.putText(image, median_text, (median_text_x, median_text_y),
                font, 1, (255, 0, 0), 2)
    
    # Create final figure
    axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axis.set_title('Object found scores', size=5)
    axis.tick_params(labelsize=2, width=0.1, length=1)
    #axis.set_aspect(1./axis.get_data_ratio())    
    
# Compute centroid
def centroid(arr):
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return sum_x/length, sum_y/length