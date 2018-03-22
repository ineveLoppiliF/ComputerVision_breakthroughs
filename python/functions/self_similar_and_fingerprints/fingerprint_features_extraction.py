## Import libraries
import numpy as np
from scipy.stats import t

ALPHA=0.95 # this constant allow us to determine the quantiles to be used to discriminate self-similar features

## This function discriminates between normal features and fingerprint features,
## i.e. features that have no similar matches
def fingerprint_features_extraction(matches,template_descriptors,t_parameters):
    
    ## Creation of an array of all second matches
    second_matches = [item[0] for item in matches]
    
    ## Define the quantiles used to discriminate self-similar features
    fingerprint_quantiles = t.interval(ALPHA,t_parameters[0],t_parameters[1],t_parameters[2])
    
    ## Compute the list that contains, for each template feature, its second
    ## match only if it not pass the quantile test.
    ## This means that this feature could be considered fingerprint
    fingerprint_list=[[] for i in range(len(matches))]
    
    for i,match in enumerate(second_matches):
        ## Extract the SIFT descriptor for the current feature and its considered match
        template_descriptor1 = np.float32(template_descriptors[match.trainIdx])
        template_descriptor2 = np.float32(template_descriptors[match.queryIdx])
        
        ## Compute the distance between the descriptors using the Euclidean norm
        distance = np.linalg.norm(template_descriptor1-template_descriptor2)
        
        ## Quantile test executed only on the left tail, since a self-similar
        ## feature has more similar matches
        if distance>fingerprint_quantiles[1]:
            fingerprint_list[i].append(match)    
    
    return fingerprint_list, fingerprint_quantiles