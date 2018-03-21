## Import libraries
import numpy as np
from scipy.stats import t

ALPHA=0.95 # this constant allow us to determine the quantiles to be used to discriminate self-similar features

## This function discriminates between normal features and self-similar features, i.e. features that could have legitimate
## ambiguous matches (this features could not pass the ratio test).
## This happens because the template itself has many similar parts
def self_similar_features_extraction(matches,template_descriptors,t_parameters):
    
    ## Define the quantiles used to discriminate self-similar features
    quantiles = t.interval(ALPHA,t_parameters[0],t_parameters[1],t_parameters[2])
    
    ## Compute the list that contains, for each template feature, its matches that not pass the quantile test.
    ## This means that this feature could be considered self-similar with all these matches
    self_similar_list=[[] for i in range(len(matches))]
    
    search_for_more_neighbors = False # if at least one of the features has k self-similar features, more neighbors has to be computed to find other possible self-simlar features
    for i,kmatches in enumerate(matches):
        j=0
        no_more_selfs = False # if true, no more self-simlar features possible for the current feature, and so the search stops
        while j<len(kmatches) and no_more_selfs==False:
            ## Extract the SIFT descriptor for the current feature and its actual considered match
            template_descriptor1 = np.float32(template_descriptors[kmatches[j].trainIdx])
            template_descriptor2 = np.float32(template_descriptors[kmatches[j].queryIdx])
            
            ## Compute the distance between the descriptors using the Euclidean norm
            distance = np.linalg.norm(template_descriptor1-template_descriptor2)
            
            ## Quantile test executed only on the left tail, since a self-similar
            ## feature has more similar matches
            if distance<min(quantiles):
                self_similar_list[i].append(kmatches[j])
                if j==len(kmatches)-1:
                    search_for_more_neighbors=True
            else:
                no_more_selfs= True
            j+=1
            
    return self_similar_list, search_for_more_neighbors