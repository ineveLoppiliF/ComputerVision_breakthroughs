## Import libraries
import numpy as np

## Compute the image representing the pixelwise difference norm, and the
## version of it in which only the central part is keeped
def difference_norm_image_computation(abs_diff_image, IMAGE_RATIO_TO_CROP):

    ## Compute the image representing the pixelwise difference norm
    abs_diff_norm_image = np.linalg.norm(abs_diff_image, axis=2)
    
    ## Crop the central part of the image representing
    ## the pixelwise difference norm
    h, w = abs_diff_norm_image.shape
    startH = int(np.rint(h//2-(h*IMAGE_RATIO_TO_CROP//2)))
    endH = int(np.rint(h//2+(h*IMAGE_RATIO_TO_CROP//2)))
    startW = int(np.rint(w//2-(w*IMAGE_RATIO_TO_CROP//2)))
    endW = int(np.rint(w//2+(w*IMAGE_RATIO_TO_CROP//2)))
    abs_diff_norm_image_central = abs_diff_norm_image[startH:endH,startW:endW]
    
    return abs_diff_norm_image, abs_diff_norm_image_central