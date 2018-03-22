#%% Import libraries
from matplotlib import pyplot as plt
import cv2
from functions import self_similar_features_and_fingerprints_extraction

#%% Initialize constants

INITIAL_KNN = 10 # at first iteration of FLANN KDTREE we search for this number of neighborhood for each feature
KNN_STEP = 5 # when k self-similar features are been found for at least one feature, more neighbours are searched, incrementing the actual number by this

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

#%% Find self-similar features, fingerprint features
## Furthermore compute the description vector distances distribution, plot quantiles,
## self-similar features, fingerprint features together with the distances distribution

self_similar_list, fingerprint_list = self_similar_features_and_fingerprints_extraction(template_descriptors)