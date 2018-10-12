# Import libraries
import cv2
import numpy as np

# Perform histogram matching over each images channel
def rgb_histogram_matching(source, template):
    # Extract channels from images
    sb, sg, sr = cv2.split(source)
    tb, tg, tr = cv2.split(template)
    
    # Perform histogram matching over each channel
    matched_blue = hist_match(sb, tb)
    matched_green = hist_match(sg, tg)
    matched_red = hist_match(sr, tr)
    
    return cv2.merge((matched_blue, matched_green, matched_red))
    

# Perform histogram matching between two grayscale images
def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape).astype(np.uint8)

## Equalize both template and rectified image, and plot them
#def equalize_template_and_rectified_scene(template_image, rect_test_image):
#    
#    ## Equalize both template and rectified image
#    equalized_template_image = histogram_equalize(template_image)
#    equalized_rect_test_image = histogram_equalize(rect_test_image)
#    
#    return equalized_template_image, equalized_rect_test_image
#
## Equalize RGB image
#def histogram_equalize(img):
#    b, g, r = cv2.split(img)
#    red = cv2.equalizeHist(r)
#    green = cv2.equalizeHist(g)
#    blue = cv2.equalizeHist(b)
#    return cv2.merge((blue, green, red))