## Function that computes the ration between the area of the computed square out from test image and the area of the test image itself
def out_area_ratio(pol_test_image, pol_square):
    
    pol_in = pol_test_image.intersection(pol_square)
    area_out = pol_square.area - pol_in.area
    return area_out/pol_square.area