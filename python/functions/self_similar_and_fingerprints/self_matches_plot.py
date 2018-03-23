#%% Import libraries
import cv2
from matplotlib import pyplot as plt

## Plot self-similar matches and fingerprint matches
def self_matches_plot(template_image,template_keypoints,match_list,title):
    
    ## Specify parameters for the function that shows matches graphically
    draw_params = dict(matchColor = (0,255,0), # draw matches in green
                       singlePointColor = (255,0,0), # draw lone points in blue
                       flags = 0)
    
    ## Good matches represented on another image
    matches_image = cv2.drawMatchesKnn(template_image, template_keypoints, template_image, template_keypoints, match_list, None, **draw_params)
    
    ## Plot the good matches
    plt.imshow(cv2.cvtColor(matches_image, cv2.COLOR_BGR2RGB)), plt.title(title), plt.show()