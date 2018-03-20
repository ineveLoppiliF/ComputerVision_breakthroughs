#%% Import libraries
import numpy as np
from matplotlib import pyplot as plt
import cv2
from scipy.stats import t
from math import ceil,floor 

## Load the template
template_image = cv2.imread('../data/images/template/template_twinings.jpg', cv2.IMREAD_COLOR) # template image

## Show the loaded template
plt.imshow(cv2.cvtColor(template_image, cv2.COLOR_BGR2RGB)), plt.title('template'),plt.show()

## Create sift_detector detector
sift_detector = cv2.xfeatures2d.SIFT_create()

## Find the keypoints and descriptors with sift_detector from the template and test image
template_keypoints,template_descriptors = sift_detector.detectAndCompute(template_image, None)

## Show the number of keypoints found in the template and test image
print('found ' + str(len(template_keypoints)) + ' keypoints in the image')

#%% Initialize a flann_matcher object to match keypoint witn nearest neighborhood. 

# = From flann_matcher documentation ==================================================
# 	flann_matcher_INDEX_LINEAR = 0,
# 	flann_matcher_INDEX_KDTREE = 1,
# 	flann_matcher_INDEX_KMEANS = 2,
# 	flann_matcher_INDEX_COMPOSITE = 3,
# 	flann_matcher_INDEX_KDTREE_SINGLE = 4,
# 	flann_matcher_INDEX_HIERARCHICAL = 5,
# 	flann_matcher_INDEX_LSH = 6,
# 	flann_matcher_INDEX_KDTREE_CUDA = 7, // available if compiled with CUDA
# 	flann_matcher_INDEX_SAVED = 254,
# 	flann_matcher_INDEX_AUTOTUNED = 255,
# =============================================================================

## Specify a constant representing the type of algorithm used by flann_matcher
flann_matcher_INDEX_KDTREE = 1 # algorithm used is KDTREE

## Specify flann_matcher matcher creator parameters
index_params = dict(algorithm=flann_matcher_INDEX_KDTREE, trees=5) # 5 trees used in the KDTREE search
search_params = dict(checks=50) # number of times the trees in the index should be recursively traversed

## Create FLANN matcher
flann_matcher = cv2.FlannBasedMatcher(index_params, search_params)


## Invoke flann_matcher methods to obtain k outputs: for each feature in the test_descriptors image returns the k closest features in the template_descriptors image
matches =  flann_matcher.knnMatch(template_descriptors,template_descriptors,k=5) # there is no trehsold, the closest point is returned

## Show the number features in test_descriptors image that have at least one match in template_descriptors image
print('found ' + str(len(matches)) + ' putative matches')

## Deletion of the first match that is the feature itself
for i,kmatches in enumerate(matches):
    kmatches.pop(0)
    
## Creation of a gaussian distribution of all the distances of the second matches from the first one

#Creation of an array of all second matches and than of an array of all the matches distances
second_matches = [item[0] for item in matches]
distances = np.zeros(len(second_matches))
for i,match in enumerate(second_matches):
    template_descriptor1 = np.float32(template_descriptors[match.trainIdx])
    template_descriptor2 = np.float32(template_descriptors[match.queryIdx])
    distances[i]=np.linalg.norm(template_descriptor1-template_descriptor2)

#Creation of the t-student distribution of the average of the distances
#returns a list of two parameters (shape, parameters[0], mean, parameters[1] and std, parameters[2])
t_parameters = t.fit(distances)
#define the interval 
x = np.linspace(min(distances),max(distances),1000)
#generate the fitted distribution
fitted_pdf = t.pdf(x,df=t_parameters[0],loc = t_parameters[1],scale = t_parameters[2])
plt.plot(x,fitted_pdf,"green",label="Fitted T-student dist", linewidth=4)
plt.hist(distances,bins=range(floor(min(distances)),ceil(max(distances))),normed=1,color="green",alpha=.2) #alpha, from 0 (transparent) to 1 (opaque)
plt.title("T-student distribution fitting")
# insert a legend in the plot (using label)
plt.legend()
# we finally show our work
plt.show()



