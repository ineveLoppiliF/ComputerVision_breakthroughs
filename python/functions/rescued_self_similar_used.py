## Import libraries
import numpy as np
 
## Show the number of rescued self-similar matches effectively used to find
## a good homography until now
def rescued_self_similar_used(flat_rescued_self_similar_mask, good_rescued_self_similar_mask, 
                              new_good_rescued_self_similar_mask, remove_mask, self_similar_per_image):        
    
    ## Create a mask of the rescued self-similar elements eliminated by the
    ## last homography
    last_removed_self_similar_mask = np.zeros(len(remove_mask))
    for i in range(len(last_removed_self_similar_mask)):
        if remove_mask[i] == good_rescued_self_similar_mask[i] == 1:
            last_removed_self_similar_mask[i]=1
    
    ## Compute the number of rescued self similar feature matches
    ## eliminated until now
    rescued_self_similar_used_until_now = (np.count_nonzero(flat_rescued_self_similar_mask) - 
                                           np.count_nonzero(new_good_rescued_self_similar_mask))
    
    self_similar_per_image.append(np.count_nonzero(good_rescued_self_similar_mask)-np.count_nonzero(new_good_rescued_self_similar_mask))
    
    ## Show informations
    print('----------')
    print('Total rescued self-similar features: ' + str(np.count_nonzero(flat_rescued_self_similar_mask)))
    print('Rescued self-similar features used until now: ' + str(rescued_self_similar_used_until_now))
    
    print('Total features eliminated from the last homography: ' + str(np.count_nonzero(remove_mask)))
    print('Rescued self-similar features used in the last homography: ' + str(np.count_nonzero(good_rescued_self_similar_mask)-
                                                                              np.count_nonzero(new_good_rescued_self_similar_mask)))
    print('----------')