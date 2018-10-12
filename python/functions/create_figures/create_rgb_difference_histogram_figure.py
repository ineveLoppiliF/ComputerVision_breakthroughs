# Import libraries
import cv2
import numpy as np

# Create rgb difference histogram figure
def create_rgb_difference_histogram_figure(abs_diff_image,
                                           axis):
    # Create differences histograms
    abs_diff_hist = list()
    for i in range(np.size(abs_diff_image,2)):
        abs_diff_hist.append(cv2.calcHist([abs_diff_image],[i],None,[256],[0,256]))
    
    # Create differences histogram figure
    color = ('B','G','R')
    for i,col in enumerate(color):
        axis.plot(abs_diff_hist[i],color = col,linewidth=0.1)
        axis.set_xlim([0,256])
        axis.set_ylim(ymin=0)
    axis.set_title('BGR difference\nhistogram', size=5)
    axis.set_xticks(np.arange(0, 256, 20))
    axis.tick_params(labelsize=1, width=0.1, length=1)
    axis.set_aspect(1./axis.get_data_ratio())