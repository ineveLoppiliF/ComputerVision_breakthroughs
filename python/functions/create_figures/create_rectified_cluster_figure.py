# Import libraries
import cv2

# Create the rectified cluster figure, with all cluster matches
def create_rectified_cluster_figure(rectified_matches_image,
                                    axis):
    # Create final figure
    axis.imshow(cv2.cvtColor(rectified_matches_image, cv2.COLOR_BGR2RGB))
    axis.set_title('Rectified object\nmatches', size=5)
    axis.tick_params(labelsize=2, width=0.1, length=1)
    #axis.set_aspect(1./axis.get_data_ratio())  