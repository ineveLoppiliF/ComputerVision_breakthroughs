## Function that print statistics about discarde homography
def print_discarded(discarded_homographies):
    
    print("----------")
    print("DISCARDED HOMOGRAPHIES UNTIL NOW (tot: "+str(sum(discarded_homographies))+")")
    print("Degenerate homography (rank not full): "+str(discarded_homographies[1]))
    print("Degenerate homography (points out): "+str(discarded_homographies[4]))
    print("Degenerate homography (non valid polygon)): "+str(discarded_homographies[5]))
    print("Polygon mostly out from test image: "+str(discarded_homographies[2]))
    print("Differences distribution median too big: "+str(discarded_homographies[3]))
    print("----------")
    ## 0 is reserved to not used VALIDATE AREA