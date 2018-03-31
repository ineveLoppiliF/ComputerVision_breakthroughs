## Import libraries
from numpy import linalg

## Compute the image representing the pixelwise difference norm, and the
## version of it in which only the central part is keeped
def difference_norm_image_computation(diff_image, IMAGE_RATIO_TO_CROP):
    print(str(diff_image.shape))
    ## Compute the image representing the pixelwise difference norm
    diff_norm_image = linalg.norm(diff_image, axis=2)
    print(str(diff_norm_image.shape))
    
    ## Crop the central part of the image representing
    ## the pixelwise difference norm
    h, w = diff_norm_image.shape
    startH = h//2-(w*IMAGE_RATIO_TO_CROP//2)
    endH = h//2+(w*IMAGE_RATIO_TO_CROP//2)
    startW = w//2-(h*IMAGE_RATIO_TO_CROP//2)
    endW = w//2+(h*IMAGE_RATIO_TO_CROP//2)
    diff_norm_image_central = diff_norm_image[startH:endH,startW:endW]
    
    return diff_norm_image, diff_norm_image_central