## Function that computes the ration between the area of the computed square out from test image and the area of the test image itself
def out_area_ratio(pol_test_image, pol_square, discarded_file, OUT_OF_IMAGE_THRESHOLD, discarded_homographies, discarded_cont_count):
    
    pol_in = pol_test_image.intersection(pol_square)
    area_out = pol_square.area - pol_in.area
    if(area_out/pol_square.area <= OUT_OF_IMAGE_THRESHOLD): return True
    else:
        discarded_cont_count[0] += 1
        discarded_homographies[2]+=1
        discarded_file.write("HOMOGRAPHY DISCARDED #"+str(sum(discarded_homographies))+" (polygon out)\n")
        discarded_file.write("Area out: "+str(area_out)+"\nArea polygon: "+str(pol_square.area)+"\nRatio: "+str(area_out/pol_square.area)+"\n\n")
        return False