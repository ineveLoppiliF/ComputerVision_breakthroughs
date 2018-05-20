## Import libraries
import numpy as np
from matplotlib import pyplot as plt
from math import floor,ceil

def fingerprint_distances_plot(fingerprint_list, template_descriptors, interval, quantile):
    
    ## Unroll the list of fingerprint matches
    unrolled_fingerprint_list = [match for fingerprint_matches in fingerprint_list for match in fingerprint_matches]
    
    ## Creation of an array of all the distances of each fingerprint match
    distances = np.zeros(len(unrolled_fingerprint_list))
    for i,match in enumerate(unrolled_fingerprint_list):
        template_descriptor1 = np.float32(template_descriptors[match.trainIdx])
        template_descriptor2 = np.float32(template_descriptors[match.queryIdx])
        distances[i]=np.linalg.norm(template_descriptor1-template_descriptor2)
    
    ## Plot the histogram of distances, together with quantiles    
    plt.hist(distances,bins=range(floor(min(interval)),ceil(max(interval))),
             density=False,color="blue",alpha=.2) #alpha, from 0 (transparent) to 1 (opaque)
    plt.xlabel('Distances')
    plt.ylabel('Number of matches')       
    
    plt.axvline(x=quantile, color='red', linewidth=0.5)
    
    plt.title("Fingerprint matches distances")
    plt.show()