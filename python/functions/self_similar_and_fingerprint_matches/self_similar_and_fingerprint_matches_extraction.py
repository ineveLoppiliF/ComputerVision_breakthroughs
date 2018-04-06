## Import libraries
import cv2
from . import gaussian_computation_and_plot, self_similar_matches_extraction, fingerprint_matches_extraction, self_similar_distances_plot, fingerprint_distances_plot

## Initialize constants

INITIAL_KNN = 10 # at first iteration of FLANN KDTREE we search for this number of neighborhood for each feature
KNN_STEP = 5 # when k self-similar matches are been found for at least one feature, more neighbours are searched, incrementing the actual number by this

## Find self-similar matches, fingerprint matches.
## Furthermore compute the description vector distances distribution, plot quantiles,
## self-similar matches, fingerprint matches together with the distances distribution
def self_similar_and_fingerprint_matches_extraction(template_descriptors):
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
    index_params = dict(algorithm=flann_matcher_INDEX_KDTREE, trees=20) # 5 trees used in the KDTREE search
    search_params = dict(checks=200) # number of times the trees in the index should be recursively traversed
    
    ## Create FLANN matcher
    flann_matcher = cv2.FlannBasedMatcher(index_params, search_params)
    
    #%% Find correspondences by matching the template features with itself
    
    ## Invoke flann_matcher methods to obtain k outputs: for each feature in the template_descriptors image returns the k closest features in the template_descriptors image
    actual_knn = INITIAL_KNN
    matches =  flann_matcher.knnMatch(template_descriptors,template_descriptors,k=actual_knn) # there is no threshold, the k closest points are returned
    
    ## Show the number of features in template_descriptors image that have at least one match in template_descriptors image
    print('found ' + str(len(matches)) + ' putative matches')
    
    #%% Compute the description vector distances distribution
    
    ## Deletion of the first match that is the feature itself
    for i,kmatches in enumerate(matches):
        kmatches.pop(0)
    
    ## Creation of an array of all second matches
    second_matches = [item[0] for item in matches]
    
    ## Compute distances between each template feature and its second nearest neighbour (the first is the feature itself).
    ## Then a Gaussian distribution is estimated over these distances in order to represent the mean and the sparsity of them.
    ## At the end the distribution is plotted together with an histogram of the distances.
    ## The Gaussian's list of three parameters is then returned: (shape=parameters[0], mean=parameters[1] and std=parameters[2])
    norm_parameters, interval = gaussian_computation_and_plot(second_matches, template_descriptors)
    
    #%% Find self-similar matches
    
    ## Compute a list with the same shape of "matches", but conserving only
    ## self-similar matches
    self_similar_list, search_for_more_neighbors, self_similar_quantiles = self_similar_matches_extraction(matches,template_descriptors,norm_parameters)
    
    ## If at least one of the features has k self-similar matches, 
    ## more neighbors has to be computed to find other possible self-simlar matches
    while search_for_more_neighbors:
        actual_knn = actual_knn+KNN_STEP
        matches =  flann_matcher.knnMatch(template_descriptors,template_descriptors,k=actual_knn)
        for i,kmatches in enumerate(matches):
            kmatches.pop(0)
        self_similar_list, search_for_more_neighbors, self_similar_quantiles = self_similar_matches_extraction(matches,template_descriptors,norm_parameters)
    print('Maximum number of self-similar matches: ' + str(len(max(self_similar_list,key=len))))
    print('Number of features that have at least one self-similar match: ' + str(len([self_similar_matches for self_similar_matches in self_similar_list if self_similar_matches])))
        
    #%% Find fingerprint matches
        
    ## Compute a list with the same shape of "matches", but conserving only
    ## fingerprint matches
    fingerprint_list, fingerprint_quantiles = fingerprint_matches_extraction(matches,template_descriptors,norm_parameters)
    print('Number of features that have a fingerprint match: ' + str(len([fingerprint_match for fingerprint_match in fingerprint_list if fingerprint_match])))
    
    #%% Plot quantiles, self-similar matches, fingerprint matches together with the distances distribution
    
    self_similar_distances_plot(self_similar_list, template_descriptors, norm_parameters, interval, self_similar_quantiles)
    fingerprint_distances_plot(fingerprint_list, template_descriptors, norm_parameters, interval, fingerprint_quantiles)
    
    return self_similar_list, fingerprint_list