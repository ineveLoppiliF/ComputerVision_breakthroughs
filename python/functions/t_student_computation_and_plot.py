## Import libraries
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import t
from math import ceil, floor

## Compute distances between each template feature and its second nearest neighbour (the first is the feature itself).
## Then a t-student distribution is estimated over these distances in order to represent the mean and the sparsity of them.
## At the end the distribution is plotted together with an histogram of the distances.
## The t-student list of three parameters is then returned: (shape=parameters[0], mean=parameters[1] and std=parameters[2])
def t_student_computation_and_plot(matches, template_descriptors):
    
    ## Creation of an array of all the distances between each feature and its second match
    distances = np.zeros(len(matches))
    for i,match in enumerate(matches):
        template_descriptor1 = np.float32(template_descriptors[match.trainIdx])
        template_descriptor2 = np.float32(template_descriptors[match.queryIdx])
        distances[i]=np.linalg.norm(template_descriptor1-template_descriptor2)
    
    ## Creation of the t-student distribution on the previous computed distances
    t_parameters = t.fit(distances) # returned a list of three parameters (shape=parameters[0], mean=parameters[1] and std=parameters[2])
    
    ## Define the interval and the number of points through which plot the distribution
    x = np.linspace(min(distances),max(distances),1000)
    
    ## Generate the fitted distribution
    fitted_pdf = t.pdf(x,df=t_parameters[0],loc = t_parameters[1],scale = t_parameters[2])
    
    ## Plot the distribution over the histogram of distances
    plt.plot(x,fitted_pdf,"green",label="Fitted T-student dist", linewidth=4)
    plt.hist(distances,bins=range(floor(min(distances)),ceil(max(distances))),normed=1,color="green",alpha=.2) #alpha, from 0 (transparent) to 1 (opaque)
    plt.title("T-student distribution fitting")
    plt.legend()
    plt.show()
    
    return t_parameters