#%% Import libraries
from matplotlib import pyplot as plt
import cv2
from functions import t_student_computation_and_plot, self_similar_features_extraction

#%% Load and show template image

## Load the template
template_image = cv2.imread('../data/images/template/template_twinings.jpg', cv2.IMREAD_COLOR) # template image

## Show the loaded template
plt.imshow(cv2.cvtColor(template_image, cv2.COLOR_BGR2RGB)), plt.title('template'),plt.show()

#%%  Initiate sift_detector detector

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

#%% Find correspondences by matching the template features with itself

## Invoke flann_matcher methods to obtain k outputs: for each feature in the template_descriptors image returns the k closest features in the template_descriptors image
matches =  flann_matcher.knnMatch(template_descriptors,template_descriptors,k=5) # there is no trehsold, the closest point is returned

## Show the number of features in template_descriptors image that have at least one match in template_descriptors image
print('found ' + str(len(matches)) + ' putative matches')

#%% Find self-similar features and fingerprints

## Compute distances between each template feature and its second nearest neighbour (the first is the feature itself).
## Then a t-student distribution is estimated over these distances in order to represent the mean and the sparsity of them.
## At the end the distribution is plotted together with an histogram of the distances.
## The t-student list of three parameters is then returned: (shape=parameters[0], mean=parameters[1] and std=parameters[2])
t_parameters = t_student_computation_and_plot(matches, template_descriptors)

search_for_more_neighbors = False
#while not search_for_more_neighbors:
self_similar_list, search_for_more_neighbors = self_similar_features_extraction(matches,template_descriptors,t_parameters)