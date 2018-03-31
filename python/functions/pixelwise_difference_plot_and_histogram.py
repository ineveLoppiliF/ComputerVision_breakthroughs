## Import libraries
import numpy as np
from matplotlib import pyplot as plt

HISTOGRAM_BINS_MULTIPLIER = 3

## Plot the images of the pixelwise difference norm, and the histrogram
## of the one representing the central part, highlighting the median
def pixelwise_difference_plot_and_histogram(diff_norm_image, diff_norm_image_central, MEDIAN_THRESHOLD):
    
    ## Plot the images of the pixelwise difference norm
    plt.imshow(diff_norm_image, 'gray'), plt.title('Pixelwise difference norm image'),plt.show()
    plt.imshow(diff_norm_image_central, 'gray'), plt.title('Pixelwise difference norm image cropped'),plt.show()
        
    ## Compute the median over the norm difference image
    median = np.median(diff_norm_image_central)
    
    ## Create the histogram on the plot
    max_value = int(np.ceil(np.max(diff_norm_image_central)))
    unrolled_image = diff_norm_image_central.ravel().tolist()
    plt.hist(unrolled_image, bins=max_value*HISTOGRAM_BINS_MULTIPLIER, range=[0,max_value], histtype='stepfilled',
             color='gray')
 
    ## Print the histogram together with the median
    plt.xlim([0,256])
    plt.axvline(x=MEDIAN_THRESHOLD, color='red', linewidth=1)
    plt.axvline(x=median, color='green', linewidth=1)
    plt.show()