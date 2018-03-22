## Import libraries
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import t
from math import ceil, floor

def fingerprint_distances_plot(fingerprint_list, template_descriptors, t_parameters, interval, quantiles):
    
    ## Unroll the list of fingerprint matches
    unrolled_fingerprint_list = [match for fingerprint_matches in fingerprint_list for match in fingerprint_matches]
    
    ## Creation of an array of all the distances of each fingerprint match
    distances = np.zeros(len(unrolled_fingerprint_list))
    for i,match in enumerate(unrolled_fingerprint_list):
        template_descriptor1 = np.float32(template_descriptors[match.trainIdx])
        template_descriptor2 = np.float32(template_descriptors[match.queryIdx])
        distances[i]=np.linalg.norm(template_descriptor1-template_descriptor2)
    
    ## Generate the fitted distribution
    fitted_pdf = t.pdf(interval,df=t_parameters[0],loc = t_parameters[1],scale = t_parameters[2])
    
    ## Plot the distribution and the histogram of distances, together with quantiles
    fig=plt.figure()
    distribution_subpolt=fig.add_subplot(111, label="1")
    distances_subplot=fig.add_subplot(111, label="2", frame_on=False)
    
    distribution_subpolt.plot(interval, fitted_pdf, "green", linewidth=4)
    distribution_subpolt.set_xlabel("Distances")
    distribution_subpolt.set_ylabel("Probabilities")
    
    distances_subplot.hist(distances,bins=range(floor(min(interval)),ceil(max(interval))),density=False,color="blue",alpha=.2) #alpha, from 0 (transparent) to 1 (opaque)
    distances_subplot.xaxis.set_visible(False)
    distances_subplot.yaxis.tick_right()
    distances_subplot.set_ylabel('Number of matches')       
    distances_subplot.yaxis.set_label_position('right')
    
    plt.axvline(x=quantiles[1], color='red', linewidth=0.5)
    
    plt.title("Fingerprint features on T-student dist")
    plt.show()