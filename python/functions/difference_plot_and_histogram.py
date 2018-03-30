## Import libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt

## Plot the difference between equalized template and equalized rectified image and its histogram
def difference_plot_and_histogram(abs_diff_image):
        plt.imshow(cv2.cvtColor(abs_diff_image, cv2.COLOR_BGR2RGB)), plt.title('Absolute difference image'),plt.show()
        color = ('B','G','R')
        for i in range(np.size(abs_diff_image,2)):
            plt.subplot(1,3,i+1)
            plt.imshow(abs_diff_image[:,:,i],'gray', vmin=0, vmax=255)
            plt.title(str(color[i])+' difference')
        plt.show()
        
        ## Create differences histograms
        abs_diff_hist = list()
        for i in range(np.size(abs_diff_image,2)):
            abs_diff_hist.append(cv2.calcHist([abs_diff_image],[i],None,[256],[0,256]))
 
        ## Print differences histograms
        for i,col in enumerate(color):
            plt.plot(abs_diff_hist[i],color = col) 
            plt.xlim([0,256])
        plt.show()