# Import libraires
import cv2

# Create figure of the cluster and its matches
def create_cluster_figure(matches_image, axis):
    # Create final figure
    axis.imshow(cv2.cvtColor(matches_image, cv2.COLOR_BGR2RGB))
    axis.set_title('Clustered matches', size=5)
    axis.tick_params(labelsize=2, width=0.1, length=1)
    #axis.set_aspect(1./axis.get_data_ratio())