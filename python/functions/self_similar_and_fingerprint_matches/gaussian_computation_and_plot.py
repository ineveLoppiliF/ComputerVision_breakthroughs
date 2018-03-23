## Import libraries
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
from math import ceil, floor

## Compute distances between each template feature and its second nearest neighbour (the first is the feature itself).
## Then a Gaussian distribution is estimated over these distances in order to represent the mean and the sparsity of them.
## At the end the distribution is plotted together with an histogram of the distances.
## The Gaussian's list of two parameters is then returned: (mean=parameters[0] and std=parameters[1])
def gaussian_computation_and_plot(matches, template_descriptors):
    
    ## Creation of an array of all the distances between each feature and its second match
    distances = np.zeros(len(matches))
    for i,match in enumerate(matches):
        template_descriptor1 = np.float32(template_descriptors[match.trainIdx])
        template_descriptor2 = np.float32(template_descriptors[match.queryIdx])
        distances[i]=np.linalg.norm(template_descriptor1-template_descriptor2)
    
    ## Creation of the Gaussian distribution on the previous computed distances
    norm_parameters = norm.fit(distances) # returned a list of two parameters (mean=parameters[0] and std=parameters[1])
    
    ## Define the interval and the number of points through which plot the distribution
    interval = np.linspace(floor(min(distances)),ceil(max(distances)),1000)
    
    ## Generate the fitted distribution
    fitted_pdf = norm.pdf(interval,loc=norm_parameters[0],scale = norm_parameters[1])
    
    ## Plot the distribution and the histogram of distances 
    fig=plt.figure()
    distribution_subpolt=fig.add_subplot(111, label="1")
    distances_subplot=fig.add_subplot(111, label="2", frame_on=False)
    
    distribution_subpolt.plot(interval, fitted_pdf, "green", linewidth=4)
    distribution_subpolt.set_xlabel("Distances")
    distribution_subpolt.set_ylabel("Probabilities")
    
    distances_subplot.hist(distances,bins=range(floor(min(distances)),ceil(max(distances))),density=False,color="green",alpha=.2) #alpha, from 0 (transparent) to 1 (opaque)
    distances_subplot.xaxis.set_visible(False)
    distances_subplot.yaxis.tick_right()
    distances_subplot.set_ylabel('Number of matches')       
    distances_subplot.yaxis.set_label_position('right')
    
    plt.title("Gaussian distribution fitting")
    plt.show()
    
    return norm_parameters, interval