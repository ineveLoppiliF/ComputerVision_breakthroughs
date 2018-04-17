import numpy as np

## Function that verify if matrix rank is full
def is_rank_full(H, discarded_cont_count, discarded_homographies, discarded_file):
    
    if np.linalg.matrix_rank(H) == 3:
        return True
    discarded_homographies[1]+=1
    discarded_cont_count[0]+=1
    discarded_file.write("HOMOGRAPHY DISCARDED #"+str(sum(discarded_homographies))+" (not full rank)\n")
    discarded_file.write("Rank: "+str(np.linalg.matrix_rank(H)))
    return False