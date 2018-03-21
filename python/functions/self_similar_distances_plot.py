## Import libraries
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import t
from math import ceil, floor

def self_similar_distances_plot(self_similar_list, template_descriptors, t_parameters, interval):
    
    ## Unroll the list of self-similar matches
    unrolled_self_similar_list = [match for self_similar_matches in self_similar_list for match in self_similar_matches]
    
    ## Creation of an array of all the distances of each self-similar match
    distances = np.zeros(len(unrolled_self_similar_list))
    for i,match in enumerate(unrolled_self_similar_list):
        template_descriptor1 = np.float32(template_descriptors[match.trainIdx])
        template_descriptor2 = np.float32(template_descriptors[match.queryIdx])
        distances[i]=np.linalg.norm(template_descriptor1-template_descriptor2)
    
    ## Generate the fitted distribution
    fitted_pdf = t.pdf(interval,df=t_parameters[0],loc = t_parameters[1],scale = t_parameters[2])
    
    ## Plot the distribution over the histogram of distances
    plt.plot(interval,fitted_pdf,"green", linewidth=4)
    plt.hist(distances,bins=range(floor(min(interval)),ceil(max(interval))),density=True,color="blue",alpha=.2) #alpha, from 0 (transparent) to 1 (opaque)
    plt.title("Self-similar features on T-student dist")
    plt.show()