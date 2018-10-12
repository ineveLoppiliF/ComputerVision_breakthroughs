# Import libraries
import cv2
import numpy as np

# Create rgb difference figure
def create_rgb_difference_figure(abs_diff_image, axes):    
    axes[0].imshow(cv2.cvtColor((abs_diff_image * (1/np.amax(abs_diff_image)) * 255).astype('uint8'), cv2.COLOR_BGR2RGB))
    axes[0].set_title('BGR absolute\ndifference image', size=5)
    axes[0].tick_params(labelsize=2, width=0.1, length=1)
    #axes[0].set_aspect(1./axes[0].get_data_ratio())
    
    color = ('B','G','R')
    for i in range(np.size(abs_diff_image,2)):
        axes[i+1].imshow(np.stack(((abs_diff_image[:,:,i] * (1/np.amax(abs_diff_image[:,:,i])) * 255).astype('uint8'),)*3, -1),
                       'gray')
        axes[i+1].set_title(str(color[i])+' difference', size=5)
        axes[i+1].tick_params(labelsize=2, width=0.1, length=1)
        #axes[i+1].set_aspect(1./axes[i+1].get_data_ratio())