# Import libraries
from .create_figures import (create_cluster_and_scores_figure,
                             create_cluster_figure,
                             create_pixelwise_difference_norm_figure,
                             create_pixelwise_difference_norm_histogram_figure,
                             create_rectified_cluster_figure,
                             create_rectified_matched_cluster_figure,
                             create_rgb_difference_figure,
                             create_rgb_difference_histogram_figure)
import matplotlib as mpl
from matplotlib import pyplot as plt

# Create a unique global image containing all information useful for the
# algorithm output
def global_image_creation_and_saving(test_image, matches_image, rectified_matches_image,
                                     equalized_rect_stacked_image, abs_diff_image,
                                     diff_norm_image, object_test_keypoints,
                                     inliers_matches, dst_vrtx, cluster_number,
                                     iou_measure, saving_folder):
    # Initial initializations
    mpl.rcParams['axes.linewidth'] = 0.1
    
    # Set the number of subplots to create
    number_of_subplots = 11
    actual_subplot_index = 0
    
    # Create subplots
    fig, axes = plt.subplots(number_of_subplots,1)
    
    create_cluster_figure(matches_image,
                          axes[actual_subplot_index])
    actual_subplot_index+=1
    
    create_rectified_cluster_figure(rectified_matches_image,
                                    axes[actual_subplot_index])
    actual_subplot_index+=1
    
    create_rectified_matched_cluster_figure(equalized_rect_stacked_image,
                                            axes[actual_subplot_index])
    actual_subplot_index+=1
    
    create_rgb_difference_figure(abs_diff_image,
                                 axes[actual_subplot_index:actual_subplot_index+4])
    actual_subplot_index+=4
    
    create_rgb_difference_histogram_figure(abs_diff_image,
                                           axes[actual_subplot_index])
    actual_subplot_index+=1
    
    create_pixelwise_difference_norm_figure(diff_norm_image, object_test_keypoints,
                                            inliers_matches, axes[actual_subplot_index])
    actual_subplot_index+=1
    
    median = create_pixelwise_difference_norm_histogram_figure(diff_norm_image,
                                                               axes[actual_subplot_index])
    actual_subplot_index+=1
    
    create_cluster_and_scores_figure(test_image, dst_vrtx, iou_measure, median,
                                     axes[actual_subplot_index])
    actual_subplot_index+=1
    
    fig.suptitle('Global result image of the cluster #'+str(cluster_number),
                 fontsize=10, y = 0.925)
    plt.subplots_adjust(hspace = 0.6)
    plt.savefig(saving_folder + '/Global result image of the cluster #'+str(cluster_number) + '.png',
                dpi=1200, format='png', bbox_inches='tight')
    plt.show()
    
    return median