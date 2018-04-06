## Function that ..
def print_discarded(discarded_homographies):
    
    print("----------")
    print("DISCARDED HOMOGRAPHIES UNTIL NOW (tot: "+str(discarded_homographies[0]+discarded_homographies[1]+discarded_homographies[2]+discarded_homographies[3])+")")
    print("Area is too big: "+str(discarded_homographies[0]))
    print("Degenerate homography: "+str(discarded_homographies[1]))
    print("Polygon mostly out from test image: "+str(discarded_homographies[2]))
    print("Differences distribution median too big: "+str(discarded_homographies[3]))
    print("----------")