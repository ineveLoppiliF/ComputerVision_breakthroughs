## Import libraries
import numpy as np

## Check that the pixelwise difference norm median of a central region of the
## difference image is under a certain threshold
def pixelwise_difference_norm_check(diff_norm_image_central, MEDIAN_THRESHOLD, discarded_file, discarded_homographies, discarded_cont_count):
    ## Compute the median over the norm difference image
    median = np.median(diff_norm_image_central)
    
    ## Check if the difference image has a median lower than this
    if median < MEDIAN_THRESHOLD:
        return True
    
    ## Otherwise return false, and write on debug file
    discarded_cont_count[0] += 1
    discarded_homographies[3]+=1
    discarded_file.write("HOMOGRAPHY DISCARDED #"+str(sum(discarded_homographies))+" (norm difference histogram median too big)\n")
    discarded_file.write("Max treshold: "+str(MEDIAN_THRESHOLD)+"\nMedian: "+str(median)+"\n\n")
    
    return False