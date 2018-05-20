## Import libraries
import numpy as np
from matplotlib import pyplot as plt
from math import ceil, floor

## Compute distances between each template feature and its second nearest neighbour (the first is the feature itself).
## Then distances histogram is seen as a discrete distribution, and its range of
## distances values is extracted.
## At the end the histogram is plotted.
def histogram_computation_and_plot(matches, template_descriptors):
    
    ## Creation of an array of all the distances between each feature and its second match
    distances = np.zeros(len(matches))
    for i,match in enumerate(matches):
        template_descriptor1 = np.float32(template_descriptors[match.trainIdx])
        template_descriptor2 = np.float32(template_descriptors[match.queryIdx])
        distances[i]=np.linalg.norm(template_descriptor1-template_descriptor2)
    
    ## Define the interval and the number of points through which plot the distribution
    interval = np.linspace(floor(min(distances)),ceil(max(distances)),1000)
    
    ## Plot the distribution and the histogram of distances     
    plt.hist(distances,bins=range(floor(min(distances)),ceil(max(distances))),
             density=False,color="green",alpha=.2) #alpha, from 0 (transparent) to 1 (opaque)
    plt.xlabel('Distances')
    plt.ylabel('Number of matches')       
    
    plt.title("Descriptor vectors histogram distribution")
    plt.show()
    
    return interval