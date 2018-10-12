# Import libraries
import cv2
import numpy as np

# Create pixelwise difference norm histogram figure
def create_pixelwise_difference_norm_histogram_figure(diff_norm_image,
                                                      axis):    
    # Compute the median over the norm difference image
    median = np.median(diff_norm_image)
    
    # Create difference histogram
    diff_norm_hist = cv2.calcHist([diff_norm_image],[0],None,[256],[0,256])
    
    # Create difference histogram figure
    axis.fill_between(np.arange(diff_norm_hist.shape[0]),
                      diff_norm_hist.flatten(),
                      color = 'gray', linewidth=0.1)
    #plt.plot(diff_norm_hist,color = 'gray')
    axis.set_xlim([0,256])
    axis.set_ylim(ymin=0)
 
    # Create also the median on the histogram
    #plt.axvline(x=MEDIAN_THRESHOLD, color='red', linewidth=1)
    axis.axvline(x=median, color='green', linewidth=0.1)
    axis.set_title('Pixelwise difference\nnorm histogram', size=5)
    axis.set_xticks(np.arange(0, 256, 20))
    axis.tick_params(labelsize=1, width=0.1, length=1)
    axis.set_aspect(1./axis.get_data_ratio())
    
    return median