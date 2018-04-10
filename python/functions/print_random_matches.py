## Import libraries
from math import floor
from matplotlib import pyplot as plt

import cv2
import logging
import numpy as np
import random

## Taken two images print on them random self-similar matches
def print_random_matches(test_image, test_keypoints, template_image, 
                         template_keypoints, matches, self_similar_mask, 
                         random_match_ratio, iterations):
    
    ## Set graphical parameters
    ## Specify parameters for the function that shows random matches
    draw_params = dict(matchColor = (0,255,0), # draw matches in green
                   singlePointColor = (0,0,255), # draw lone points in red
                   matchesMask = [], # draw only random matches
                   flags = 0)
    
    try:
        ## Normalize the number of iterations, if is greater than the maximum
        ## possible
        max_iterations =  floor(1/random_match_ratio)
        if iterations > max_iterations:
            iterations = max_iterations
            
        ## Extract and shuffle self similar indices in the match list
        index_self_similar_matches = np.nonzero(self_similar_mask)[0]
        random.shuffle(index_self_similar_matches)
        index_self_similar_matches = index_self_similar_matches.tolist()
        
        ## Compute the number of elements to pick up every cycle
        number_picked_up = floor(len(index_self_similar_matches)*random_match_ratio)
        
        ## For each iteration print a random_match_ratio fraction 
        ## of the total matches
        for i in range(iterations):
            ## Mask used to select random matches to show
            random_mask = [[0,0] for i in iter(range(len(matches)))]
            
            ## Pick up indices of matches to print and delete them from
            ## the old list
            max_elements_to_pick_up = min(len(index_self_similar_matches),number_picked_up)
            indices = index_self_similar_matches[:max_elements_to_pick_up]
            del index_self_similar_matches[:max_elements_to_pick_up]
            
            for index in indices:
                random_mask[index]=[1,0] # mask modified to consider the i-th 
                                         # match as to print        
            
            ## Modify the mask in the parameters dictionary
            draw_params['matchesMask'] = random_mask
            
            ## Random matches represented on another image
            matches_image = cv2.drawMatchesKnn(test_image, test_keypoints,
                                               template_image, template_keypoints,
                                               matches, None, **draw_params)
            
            ## Plot the random matches
            plt.imshow(cv2.cvtColor(matches_image, cv2.COLOR_BGR2RGB))
            plt.title('Random self-similar feature matches'), plt.show()
        
    except ZeroDivisionError as err:
        logging.exception('Random match ratio cannot be equal to 0')
        