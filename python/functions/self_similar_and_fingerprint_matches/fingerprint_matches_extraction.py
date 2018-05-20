## Import libraries
import numpy as np

ALPHA=0.05 # this constant allow us to determine the quantiles to be used to discriminate self-similar matches

## This function discriminates between normal matches and fingerprint matches,
## i.e. matches between a feature and its nearest neighbour in which they are not similar
def fingerprint_matches_extraction(matches,second_matches,template_descriptors):
    
    ## Creation of an array of all the distances between each feature and its second match
    distances = np.zeros(len(second_matches))
    for i,match in enumerate(second_matches):
        template_descriptor1 = np.float32(template_descriptors[match.trainIdx])
        template_descriptor2 = np.float32(template_descriptors[match.queryIdx])
        distances[i]=np.linalg.norm(template_descriptor1-template_descriptor2)
    ## Define the quantiles used to discriminate self-similar matches
    fingerprint_quantile = np.percentile(distances, (1-ALPHA)*100)
    
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
        if distance>fingerprint_quantile:
            fingerprint_list[i].append(match)    
    
    return fingerprint_list, fingerprint_quantile