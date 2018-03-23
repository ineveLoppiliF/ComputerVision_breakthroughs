## Import libraries
import numpy as np
from scipy.stats import norm

ALPHA=0.95 # this constant allow us to determine the quantiles to be used to discriminate self-similar matches

## This function discriminates between normal matches and fingerprint matches,
## i.e. matches between a feature and its nearest neighbour in which they are not similar
def fingerprint_matches_extraction(matches,template_descriptors,norm_parameters):
    
    ## Creation of an array of all second matches
    second_matches = [item[0] for item in matches]
    
    ## Define the quantiles used to discriminate self-similar matches
    fingerprint_quantiles = norm.interval(ALPHA,norm_parameters[0],norm_parameters[1])
    
    ## Compute the list that contains, for each template feature, its second
    ## match only if it not pass the quantile test.
    ## This means that this match could be considered fingerprint
    fingerprint_list=[[] for i in range(len(matches))]
    
    for i,match in enumerate(second_matches):
        ## Extract the SIFT descriptor for the current feature and its considered match
        template_descriptor1 = np.float32(template_descriptors[match.trainIdx])
        template_descriptor2 = np.float32(template_descriptors[match.queryIdx])
        
        ## Compute the distance between the descriptors using the Euclidean norm
        distance = np.linalg.norm(template_descriptor1-template_descriptor2)
        
        ## Quantile test executed only on the right tail, since in a fingerprint
        ## match the features are not similar
        if distance>fingerprint_quantiles[1]:
            fingerprint_list[i].append(match)    
    
    return fingerprint_list, fingerprint_quantiles