import numpy as np
import matplotlib.pyplot as plt
 
## Print statistics about self-similar usage in the identified images
def print_self_similar_stats(inliers_per_image, self_similar_per_image, number_rescued_self_similar, good_rescued_self_similar_mask):
    
    labels = []
    ss = []
    inl = []
    for i in range(0,len(self_similar_per_image)):
        labels.append(str(i+1))
        ss.append(self_similar_per_image[i])
        inl.append(inliers_per_image[i])
    
    print('----------')
    print('Used '+str(int(sum(self_similar_per_image)))+' of '+str(number_rescued_self_similar)+' rescued self similar as inliers')
    print(str(int(number_rescued_self_similar-sum(good_rescued_self_similar_mask)))+" rescued self similar found inside extracted images")
    print('Remain '+str(int(sum(good_rescued_self_similar_mask)))+" out of images")
    
    pos = np.arange(len(labels))    
    ax = plt.axes()
    ax.set_xticks(pos)
    ax.set_xticklabels(labels)
    plt.title("Inliers(red) and self-similar-inliers(blue) for each image")
    plt.bar(pos+0.2, ss, 0.4, color='b')
    plt.bar(pos-0.2, inl, 0.4, color='r')
    plt.show()