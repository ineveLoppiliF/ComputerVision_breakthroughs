# Import libraries
import cv2

# Create a figure containing rectified histogram matched cluster figure and
# the template one
def create_rectified_matched_cluster_figure(equalized_rect_stacked_image,
                                            axis):
    # Create final figure
    axis.imshow(cv2.cvtColor(equalized_rect_stacked_image, cv2.COLOR_BGR2RGB))
    axis.set_title('Template and histogram\nmatched object image', size=5)
    axis.tick_params(labelsize=2, width=0.1, length=1)
    #axis.set_aspect(1./axis.get_data_ratio())