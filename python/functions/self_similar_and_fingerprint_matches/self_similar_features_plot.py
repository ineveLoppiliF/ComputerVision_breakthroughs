#%% Import libraries
import cv2
from matplotlib import pyplot as plt

## Plot self-similar features
def self_similar_features_plot(template_image, template_keypoints, self_similar_list):
    ## Extract self_similar features keypoints
    self_similar_features_keypoints = []
    for i,self_similar_matches in enumerate(self_similar_list):
        if self_similar_matches:
            self_similar_features_keypoints.append(template_keypoints[self_similar_matches[0].queryIdx])
            
    ## Self-similar features represented on another image
    self_similar_features_image = cv2.drawKeypoints(cv2.cvtColor(template_image,cv2.COLOR_BGR2GRAY),
                                                    self_similar_features_keypoints,
                                                    None,
                                                    None,
                                                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    ## Plot the self-similar features
    plt.imshow(cv2.cvtColor(self_similar_features_image, cv2.COLOR_BGR2RGB)), plt.title('Self-similar features'), plt.show()