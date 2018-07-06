#%% Import libraries
import matplotlib
from matplotlib import pyplot as plt
import cv2
from functions import (self_similar_and_fingerprint_matches_extraction,
                       self_similar_features_plot,
                       self_matches_plot)

#%% Initialize constants

## Set the size of the figure to show
matplotlib.rcParams["figure.figsize"]=(15,12)

#%% Load and show template image

## Load the template
template_image = cv2.imread('../data/images/template/momath_template2.jpg', cv2.IMREAD_COLOR) # template image

## Show the loaded template
plt.imshow(cv2.cvtColor(template_image, cv2.COLOR_BGR2RGB)), plt.title('template'),plt.show()

#%%  Initiate sift_detector detector

## Create sift_detector detector
sift_detector = cv2.xfeatures2d.SIFT_create()

## Find the keypoints and descriptors with sift_detector from the template and test image
template_keypoints,template_descriptors = sift_detector.detectAndCompute(template_image, None)

## Show the number of keypoints found in the template and test image
print('found ' + str(len(template_keypoints)) + ' keypoints in the image')

#%% Find self-similar matches, fingerprint matches
## Furthermore compute the description vector distances distribution, plot quantiles,
## self-similar matches, fingerprint matches together with the distances distribution

self_similar_list, fingerprint_list = self_similar_and_fingerprint_matches_extraction(template_descriptors)

#%% Plot self-similar matches and fingerprint matches

self_similar_features_plot(template_image, template_keypoints, self_similar_list)
self_matches_plot(template_image, template_keypoints, self_similar_list, 'Self-similar matches')
self_matches_plot(template_image, template_keypoints, fingerprint_list, 'Fingerprint matches')