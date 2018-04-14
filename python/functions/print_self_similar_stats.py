import numpy as np
import matplotlib.pyplot as plt

## Print statistics about self-similar usage in the identified images
def print_self_similar_stats(self_similar_per_image, number_rescued_self_similar):
    
    labels = []
    ss = []
    for i in range(0,len(self_similar_per_image)):
        labels.append(str(i+1))
        ss.append(self_similar_per_image[i])
    
    print('----------')
    print('Used '+str(sum(self_similar_per_image))+' of '+str(number_rescued_self_similar)+' rescued self similar as inliers')
    
    pos = np.arange(len(labels))    
    ax = plt.axes()
    ax.set_xticks(pos)
    ax.set_xticklabels(labels)
    plt.title("Self-similar for each image")
    plt.bar(pos, ss, 0.8, color='b')
    plt.show()
    
    print('----------')
