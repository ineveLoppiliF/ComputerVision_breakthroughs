import random
import numpy as np

##Shuffle an array of matches and a related mask. Matches and mask must be of the same size
def shuffle_matches(matches, mask):
    
    ##Create array of indices
    indices = np.arange(len(matches))
    
    ##Shuffle array of indices
    random.shuffle(indices)
    
    temp_matches=[]
    temp_mask=[]
    ##Shuffle of matches and mask based on shuffled indices
    for i,match in enumerate(matches):
        temp_matches.append(matches[indices[i]])
        temp_mask.append(mask[indices[i]])
    matches = temp_matches
    mask = temp_mask