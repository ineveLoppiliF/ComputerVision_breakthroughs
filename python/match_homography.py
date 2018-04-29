#%% Import libraries
from functions import (difference_norm_image_computation,
                       difference_plot_and_histogram, 
                       equalize_template_and_rectified_scene, 
                       is_homography_degenerate,
                       out_area_ratio,
                       pixelwise_difference_norm_check,
                       pixelwise_difference_plot_and_histogram,
                       print_discarded,
                       print_random_matches,
                       rescued_self_similar_used,
                       project_keypoints,
                       remove_temporarily_matches,
                       remove_mask,
                       self_matches_plot,
                       self_similar_and_fingerprint_matches_extraction,
                       print_self_similar_inliers_and_eliminated,
                       print_self_similar_stats,
                       shuffle_matches,
                       out_points_ratio,
                       is_rank_full
                       )
                       #, validate_area)
from matplotlib import pyplot as plt
from numpy.linalg import inv
from shapely.geometry.polygon import Polygon

import cv2
import matplotlib
import numpy as np
                       
#%% Initial initializations

## Constant parameters to be tuned
MIN_MATCH_COUNT = 30 # search for the template whether there are at least
                     # MIN_MATCH_CURENT good matches in the scene
MIN_MATCH_CURRENT = 10 # stop when your matched homography has less than that features
LOWE_THRESHOLD = 0.8 # a match is kept only if the distance with the closest
                     # match is lower than LOWE_THRESHOLD * the distance with
                     # the second best match
IN_POLYGON_THRESHOLD = 0.95 # homography kept only if at least this fraction
                            # of inliers are in the polygon
OUT_OF_IMAGE_THRESHOLD = 0.1 # Homography kept only if the square is not 
                             # too much out from test image
#ALPHA=0.9999999999999 # this constant allow us to determine the quantiles
                      # to be used to discriminate areas
IMAGE_RATIO_TO_CROP = 0.8 # after the computation of the image representing
                          # the pixelwise difference norm, a cropped version
                          # of it is computed, in which only the central part is keeped
MEDIAN_THRESHOLD = np.multiply(441.672956,0.35) # threshold on the median, used to
                                             # discard wrong matches if the
                                             # cropped pixelwise difference norm 
                                             # have it greater than this.
                                             # 441.672956 is the maximum
                                             # possible cropped pixelwise difference
RANDOM_MATCH_RATIO = 0.1 # number of randomly plotted matches at each
                         # print_random_matches iteration, in order to see
                         # better the matches connections
ITERATIONS = 5 # number of iterations of print_random_matches
MAX_DISCARDED_CONTINUOUSLY = 100 # max number of homography discarded in a row before stopping 
                                # the algorithm

## Set the size of the figure to show
matplotlib.rcParams["figure.figsize"]=(15,12)

## Load images 
template_image = cv2.imread('../data/images/template/template_twinings.jpg', cv2.IMREAD_COLOR) # template image
test_image = cv2.imread('../data/images/test/twinings4.JPG', cv2.IMREAD_COLOR)  # test image

## Show the loaded images
plt.imshow(cv2.cvtColor(template_image, cv2.COLOR_BGR2RGB)), plt.title('template'),plt.show()
plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)), plt.title('image'),plt.show()

#%%  Initiate sift_detector detector

## Create sift_detector detector
sift_detector = cv2.xfeatures2d.SIFT_create()

## Find the keypoints and descriptors with sift_detector from the template and test image
template_keypoints,template_descriptors = sift_detector.detectAndCompute(template_image, None)
test_keypoints,test_descriptors  = sift_detector.detectAndCompute(test_image, None)

## Show the number of keypoints found in the template and test image
print('found ' + str(len(template_keypoints)) + ' keypoints in the template')
print('found ' + str(len(test_keypoints)) + ' keypoints in the test image')

# kp list of keypoints such that:
#   kp[0].pt = location
#   kp[0].angle = orientation
#   kp[0].octave = scale information

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

#%% Find correspondences by matching the image features with the template features(this is not the same as matching template_descriptors with test_descriptors)

## Invoke flann_matcher methods to obtain k outputs: for each feature in the test_descriptors image returns the k closest features in the template_descriptors image
matches =  flann_matcher.knnMatch(test_descriptors,template_descriptors,k=2) # there is no trehsold, the k closest points are returned

## Show the number features in test_descriptors image that have at least one match in template_descriptors image
print('Found ' + str(len(matches)) + ' putative matches')

## Extract self similar and fingerprint list
self_similar_list, fingerprint_list = self_similar_and_fingerprint_matches_extraction(template_descriptors)

## Plot self-similar feature matches
self_matches_plot(template_image, template_keypoints, self_similar_list, 'Self-similar matches')

## Create a reorganized self similar template matches list
## and create an always true mask
self_similar_template_matches = [[item] for items in self_similar_list for item in items]
self_similar_template_mask = [[1] for i in iter(range(len(self_similar_template_matches)))]

## Plot randomly a subset of self-similar feature matches in the template
print_random_matches(template_image, template_keypoints, template_image, 
                         template_keypoints, self_similar_template_matches, 
                         self_similar_template_mask, 
                         RANDOM_MATCH_RATIO, ITERATIONS)

#%% Store all the good matches as per Lowe's ratio test
# Lowe's ratio test removes the ambiguous and false matches:
#   It keeps only matches where the distance with the closest match is lower 
#   than LOWE_THRESHOLD * the distance with the second best match
good_matches = []
good_rescued_self_similar_mask = [] # mask of rescued self similar matches
                                         # with same length of good_matches

## Need to keep only good matches, so create a mask, each row corresponds to a match
matches_mask = [[0,0] for i in iter(range(len(matches)))]

## Create also a mask to keep rescued self-similar feature matches
rescued_self_similar_mask = [[0,0] for i in iter(range(len(matches)))]

number_rescued_self_similar=0
## Apply Lowe's test for each match, modifying the mask accordingly
for i,(m,n) in enumerate(matches):
    if m.distance < LOWE_THRESHOLD*n.distance:
        good_matches.append(m) # match appended to the list of good matches 
        good_rescued_self_similar_mask.append(0)
        matches_mask[i]=[1,0] # mask modified to consider the i-th match as good
    else:
        if len(self_similar_list[m.trainIdx])!=0:
                second_nearest_keypoint = template_keypoints[n.trainIdx]
                self_similar_keypoints = [template_keypoints[match.trainIdx] for match in self_similar_list[m.trainIdx]]
                if second_nearest_keypoint in self_similar_keypoints:
                    good_matches.append(m) # match appended to the list of good matches
                    good_rescued_self_similar_mask.append(1)
                    rescued_self_similar_mask[i]=[1,0]  # mask modified to consider
                                                        # the i-th match as self-similar
                    matches_mask[i]=[1,0] # mask modified to consider the i-th match as good
                    number_rescued_self_similar+=1
good_rescued_self_similar_mask = np.asarray(good_rescued_self_similar_mask)

## Compute a flat array of rescued self similar matches
flat_rescued_self_similar_mask = np.zeros(len(rescued_self_similar_mask))
for i,items in enumerate(rescued_self_similar_mask):
    if items[0]==1:
        flat_rescued_self_similar_mask[i]=1

## Show the number of good matches found
print('Found ' + str(len(good_matches)) + 
      ' matches validated by the distance ratio test, ' + 
      str(number_rescued_self_similar) + ' self similar')

## Specify parameters for the function that shows good matches graphically
draw_params = dict(matchColor = (0,255,0), # draw matches in green
                   singlePointColor = (0,0,255), # draw lone points in red
                   matchesMask = matches_mask, # draw only good matches
                   flags = 0)

## Good matches represented on another image
matches_image = cv2.drawMatchesKnn(test_image, test_keypoints, template_image, 
                                   template_keypoints, matches, None, **draw_params)

## Plot the good matches
plt.imshow(cv2.cvtColor(matches_image, cv2.COLOR_BGR2RGB))
plt.title('All matches after ratio test'), plt.show()

## Plot rescued self-similar feature matches on the images
## Specify parameters
draw_params = dict(matchColor = (0,255,0), # draw matches in green
                   singlePointColor = (0,0,255), # draw lone points in red
                   matchesMask = rescued_self_similar_mask, # draw only rescued
                                                            # self-similar matches
                   flags = 0)

## Rescued self-similar feature matches represented on another image
self_similar_matches_image = cv2.drawMatchesKnn(test_image, test_keypoints, 
                                                template_image, template_keypoints,
                                                matches, None, **draw_params)

## Plot the self-similar feature matches
plt.imshow(cv2.cvtColor(self_similar_matches_image, cv2.COLOR_BGR2RGB))
plt.title('Rescued self-similar feature matches'), plt.show()

## Plot randomly a subset of self-similar feature matches between template
## and test image
print_random_matches(test_image, test_keypoints, template_image, 
                         template_keypoints, matches, rescued_self_similar_mask, 
                         RANDOM_MATCH_RATIO, ITERATIONS)

#%% Cluster good matches by fitting homographies through RANSAC

input("Press Enter to start finding homographies...")

## Initilalize discarded homograpies counters (see print_discarded for more info)
discarded_homographies = [0,0,0,0,0,0]

## Initialize areas of founded homography
areas = []

## Initialize self similar per image and inliers per image counters
self_similar_per_image = []
inliers_per_image = []

## Initialize the buffer of temporary removed matches
#temporary_removed_matches = list()

## Initialize the test image used to draw projected squares
test_image_squares = test_image.copy()

## Create a polygon using test image vertices
img_polygon = Polygon([(0,0), (0,test_image.shape[0]), (test_image.shape[1],test_image.shape[0]), (test_image.shape[1],0)])

## Create debug file
discarded_file = open("debug.txt","w")

## Counter of continuously discarded homography
discarded_cont_count = []
discarded_cont_count.append(0)

## Continue to look for other homographies
end = False
while not end:
    
    #Shuffle matches and related masks in order to randomize ransac
    good_matches, good_rescued_self_similar_mask = shuffle_matches(good_matches, good_rescued_self_similar_mask)
    
    ## If have been discarded a large number of homograpies in a row, is likely that there aren't
    ## other good homograpies, and the algorithm ends 
    if discarded_cont_count[0] < MAX_DISCARDED_CONTINUOUSLY:
        ## If the number of remaining matches is low, is likely that there aren't
        ## other good homograpies, and the algorithm ends
        if len(good_matches) >= MIN_MATCH_COUNT:
            ## Retrieve coordinates of features keypoints in its image
            ## (the feature m.queryIdx inside test_image has been matched
            ##  with feature m.trainIdx inside template_image)
            src_pts = np.float32([template_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([test_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            ## Apply RANSAC algorithm to fit homograpy: M is the final homography,
            ## mask represents the inliers
            H, inliers_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            ## If no available homograpies exist, the algorithm ends
            if H is not None:      
                ## Create a list representing all the inliers of the
                ## retrieved homography
                matches_mask = inliers_mask.ravel().tolist()
                
                ## Retrieve coordinates of the inliers in the test image, 
                ## and their index wrt the actual good matches list
                dst_inliers = [dst_pts[i] for i in range(len(dst_pts)) if matches_mask[i]]
                index_inliers = np.nonzero(matches_mask)[0]
                
                ## Project the vertices of the abstract rectangle around the
                ## template image in the test one, using the found homography,
                ## in order to localize the template in the scene
                h, w = template_image.shape[0:2]
                src_vrtx = np.float32([[0, 0], 
                                       [0, h - 1], 
                                       [w - 1, h - 1], 
                                       [w - 1, 0]]).reshape(-1, 1, 2)
                dst_vrtx = cv2.perspectiveTransform(src_vrtx, H)  
                
                ## If the homography is degenerate, it is discarded
                if is_rank_full(H, discarded_cont_count, discarded_homographies, discarded_file):
                    ## If the retrieved homography has been fitted using few matches, 
                    ## is likely that has poor performances and that there aren't other good homograpies, so the algorithm ends
                    if np.count_nonzero(matches_mask) >= MIN_MATCH_CURRENT:
                        ## Create a polygon using the projected vertices
                        polygon = Polygon([(dst_vrtx[0][0][0], dst_vrtx[0][0][1]), 
                                           (dst_vrtx[1][0][0], dst_vrtx[1][0][1]), 
                                           (dst_vrtx[2][0][0], dst_vrtx[2][0][1]), 
                                           (dst_vrtx[3][0][0], dst_vrtx[3][0][1])])
                                
                        ## Homography kept only if the projected polygon
                        ## is mostly inside the image
                        if (out_points_ratio(dst_inliers, 
                                             polygon, 
                                             discarded_file, 
                                             IN_POLYGON_THRESHOLD, 
                                             discarded_homographies,
                                             discarded_cont_count) and out_area_ratio(img_polygon,
                                                                                      polygon,
                                                                                      discarded_file,
                                                                                      OUT_OF_IMAGE_THRESHOLD,
                                                                                      discarded_homographies,
                                                                                      discarded_cont_count)):
                            
                            ## Create a mask over the left good matches of the 
                            ## ones that are inliers
                            inliers_mask = np.zeros(len(good_matches))
                            for i in range(len(good_matches)):
                                if i in index_inliers:
                                    inliers_mask[i] = 1
                            
                            ## Retrieve matches that are inliers, and their index
                            ## wrt the actual good matches list
                            inliers_matches = [good_matches[i] for i in range(len(good_matches)) if inliers_mask[i]]
                            index_inliers_matches = [i for i in range(len(good_matches)) if inliers_mask[i]]
                            
                            ## Retrieve coordinates of features keypoints in its
                            ## image, for ones that are inliers
                            new_src_pts = np.float32([template_keypoints[m.trainIdx].pt for m in inliers_matches]).reshape(-1, 1, 2)
                            new_dst_pts = np.float32([test_keypoints[m.queryIdx].pt for m in inliers_matches]).reshape(-1, 1, 2)
                            
                            ## Apply LMEDS algorithm to fit a new homograpy,
                            ## taking into account all previous inliers
                            H, inliers_mask = cv2.findHomography(new_src_pts,new_dst_pts,cv2.LMEDS, 10.0)
                            
                            ## If no available homograpies exist, the algorithm ends
                            if H is not None:
                                ## Create a list representing all the inliers of the retrieved hompgrapy
                                matches_mask = inliers_mask.ravel().tolist()
                                
                                ## Retrieve coordinates of the inliers in the test image, and their index wrt the actual good matches list
                                dst_inliers = [new_dst_pts[i] for i in range(len(new_dst_pts)) if inliers_mask[i]]
                                index_inliers = [index for i,index in enumerate(index_inliers_matches) if inliers_mask[i]]
                                
                                ## Project the vertices of the abstract 
                                ## rectangle around the template image
                                ## in the test one, using the found homography,
                                ## in order to localize the template in the scene
                                dst_vrtx = cv2.perspectiveTransform(src_vrtx, H)
                                
                                ## If the homography is degenerate, it is discarded
                                if is_rank_full(H, discarded_cont_count, discarded_homographies, discarded_file):
                                    
                                    ## Create a polygon using the projected vertices
                                    polygon = Polygon([(dst_vrtx[0][0][0], dst_vrtx[0][0][1]),
                                                       (dst_vrtx[1][0][0], dst_vrtx[1][0][1]),
                                                       (dst_vrtx[2][0][0], dst_vrtx[2][0][1]),
                                                       (dst_vrtx[3][0][0], dst_vrtx[3][0][1])])
    
                                    ## Homography kept only if the projected polygon
                                    ## is mostly inside the image
                                    if (out_points_ratio(dst_inliers, 
                                                         polygon, 
                                                         discarded_file, 
                                                         IN_POLYGON_THRESHOLD, 
                                                         discarded_homographies,
                                                         discarded_cont_count) and out_area_ratio(img_polygon,
                                                                                                  polygon,
                                                                                                  discarded_file,
                                                                                                  OUT_OF_IMAGE_THRESHOLD,
                                                                                                  discarded_homographies,
                                                                                                  discarded_cont_count)):
                                   
                                        ## Apply the inverse of the found homography to the scene image
                                        ## in order to rectify the object in the polygon and extract the 
                                        ## bounded image region from the rectified one containing the template instance
                                        H_inv = inv(H)
                                        rect_test_image = cv2.warpPerspective(test_image,H_inv,(w,h))
                                        
                                        ## Equalize both template and rectified image
                                        (equalized_template_image,
                                         equalized_rect_test_image) = equalize_template_and_rectified_scene(template_image,
                                                                                                            rect_test_image)
                                        
                                        ## Compute the difference between equalized
                                        ## template and equalized rectified image
                                        abs_diff_image = cv2.absdiff(equalized_template_image,
                                                                     equalized_rect_test_image)
                                        
                                        ## Compute the image representing the
                                        ## pixelwise difference norm, and the
                                        ## version of it in which only the central
                                        ## part is keeped
                                        (diff_norm_image, 
                                         diff_norm_image_central) = difference_norm_image_computation(abs_diff_image, 
                                                                                                      IMAGE_RATIO_TO_CROP)
                                        
                                        ## Check that the pixelwise difference norm
                                        ## median of a central region of the
                                        ## difference image is under a certain
                                        ## threshold
                                        if pixelwise_difference_norm_check(diff_norm_image_central, 
                                                                           MEDIAN_THRESHOLD, 
                                                                           discarded_file,
                                                                           discarded_homographies,
                                                                           discarded_cont_count):
                                        
                                        ## Area confidence test
                                        #if validate_area(ALPHA, areas, polygon.area, discarded_file, discarded_homographies): 
                                            
                                            print('NEW HOMOGRAPHY FOUND!')
                                            
                                            ##print('Number of inliers out of the homography:' +  str(len(dst_inliers) - (out_points_ratio(dst_inliers, polygon)*len(dst_inliers))))
                                            ##print('Fraction of inliers out of the homography:' +  str((len(dst_inliers) - (out_points_ratio(dst_inliers, polygon)*len(dst_inliers)))/len(dst_inliers)))
                                        
                                            areas.append(polygon.area) 
                                            
                                            ## Draw the projected polygon in the test image, in order to visualize the found template in the test image
                                            polygons_image = cv2.polylines(test_image_squares, [np.int32(dst_vrtx)], True, [255,255,255], 3, cv2.LINE_AA)
                                            
                                            ## Specify parameters for the function that shows clustered matches, i.e. all the inliers for the selceted homography
                                            draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green
                                                               singlePointColor=None,
                                                               matchesMask=matches_mask,  # draw only inliers
                                                               flags=2)
                                            
                                            ## Draw clustered matches
                                            matches_image = cv2.drawMatches(polygons_image, test_keypoints, template_image, template_keypoints, inliers_matches, None, **draw_params)
                                            
                                            ## Plot the clustered matches
                                            plt.imshow(cv2.cvtColor(matches_image, cv2.COLOR_BGR2RGB)), plt.title('Clustered matches'), plt.show()
                                            
                                            ## Put back, inside the good matches list, points temporary removed
                                            #good_matches.extend(temporary_removed_matches)
                                            #temporary_removed_matches.clear()
                                            
                                            ## Remove all matches in the polygon
                                            keep_mask = 1 - remove_mask(test_keypoints, good_matches, polygon)
                                            new_good_matches = [good_matches[i] for i in range(len(keep_mask)) if keep_mask[i]]
                                            
                                            ## Keep the length of this mask compatible
                                            ## with good_matches length
                                            new_good_rescued_self_similar_mask = [good_rescued_self_similar_mask[i] for i in
                                                                              range(len(keep_mask)) if keep_mask[i]]
                                            new_good_rescued_self_similar_mask = np.asarray(new_good_rescued_self_similar_mask)
                                            
                                            ## Apply the homography to all test_keypoints in order to plot them
                                            object_test_keypoints = project_keypoints(test_keypoints, H_inv)
                                            
                                            ## Specify parameters for the function that shows clustered matches, i.e. all the inliers for the selceted homography
                                            draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green
                                                               singlePointColor=None,
                                                               matchesMask=matches_mask,  # draw only inliers
                                                               flags=2)
                                            
                                            ## Draw clustered rectified matches
                                            matches_image = cv2.drawMatches(rect_test_image,
                                                                            object_test_keypoints,
                                                                            template_image,
                                                                            template_keypoints,
                                                                            inliers_matches,
                                                                            None,
                                                                            **draw_params)
                                            
                                            ## Show the rectified matches
                                            plt.imshow(cv2.cvtColor(matches_image, cv2.COLOR_BGR2RGB)), 
                                            plt.title('Rectified object matches'), plt.show()
                                            
                                            ## Show the rectified self-similar
                                            ## inliers and the ones eliminated
                                            print_self_similar_inliers_and_eliminated(rect_test_image, 
                                                                                      object_test_keypoints,
                                                                                      template_image,
                                                                                      template_keypoints,
                                                                                      inliers_matches,
                                                                                      good_matches,
                                                                                      1-keep_mask,
                                                                                      matches_mask,
                                                                                      good_rescued_self_similar_mask,
                                                                                      index_inliers_matches)
                                            
                                            ## Show the rectified image
                                            rect_stacked_image = np.hstack((rect_test_image, template_image))
                                            plt.imshow(cv2.cvtColor(rect_stacked_image, cv2.COLOR_BGR2RGB)),
                                            plt.title('Rectified object image'), plt.show()
                                            
                                            ## Plot the equalized template and
                                            ## rectified image
                                            equalized_rect_stacked_image = np.hstack((equalized_rect_test_image,
                                                                                      equalized_template_image))
                                            plt.imshow(cv2.cvtColor(equalized_rect_stacked_image, cv2.COLOR_BGR2RGB)), 
                                            plt.title('Equalized template and object image'), plt.show()
                                            
                                            ## Plot the difference between equalized
                                            ## template and equalized rectified image and its histogram
                                            difference_plot_and_histogram(abs_diff_image)
                                            
                                            ## Plot the images of the pixelwise
                                            ## difference norm, and the histrogram
                                            ## of the one representing the
                                            ## central part, highlighting the median
                                            pixelwise_difference_plot_and_histogram(diff_norm_image,
                                                                                    diff_norm_image_central,
                                                                                    MEDIAN_THRESHOLD)
            
                                            ## Show the number of discarded homographies until now
                                            print_discarded(discarded_homographies)
                                            
                                            good_matches = new_good_matches
                                            ## Show the number of good matches left
                                            print('There remains: ' + str(len(good_matches)) + ' features')
                                            
                                            ## Show the number of good homograpies until now
                                            print("Found " + str(len(areas)) + " homographies until now")
                                            discarded_file.write("HOMOGRAPHY FOUNDED #"+str(len(areas))+"\n\n")
                                            
                                            ## Show the number of rescued self-similar
                                            ## matches effectively used to find a good
                                            ## homography until now
                                            rescued_self_similar_used(flat_rescued_self_similar_mask,
                                                                      good_rescued_self_similar_mask,
                                                                      new_good_rescued_self_similar_mask,
                                                                      1-keep_mask,
                                                                      self_similar_per_image,
                                                                      matches_mask,
                                                                      index_inliers_matches,
                                                                      inliers_per_image)
                                            good_rescued_self_similar_mask = new_good_rescued_self_similar_mask
                                            
                                            discarded_cont_count[0] = 0
                                            
                                            ## Search for the next template in the test image after a user command
                                        
                                            input("Press Enter to find new homography...")
                            else:
                                print("Not possible to find another homography")
                                end = True
                    else:
                        print("Not enough matches are found in the last homography - {}/{}".format(np.count_nonzero(matches_mask), MIN_MATCH_CURRENT))
                        end = True
            else:
                print("Not possible to find another homography")
                end = True
        else:
            print("Not enough matches remain - {}/{}".format(len(good_matches), MIN_MATCH_COUNT))
            end = True
    else:
       print("Discarded "+str(discarded_cont_count[0])+" homography in a row. Not able to find other homographies")
       end = True 
    
## Show the final image, in which all templates found are drawn
if len(areas)!=0: plt.imshow(cv2.cvtColor(polygons_image, cv2.COLOR_BGR2RGB)), plt.title('final image'),plt.show()

## Show the number of discarded homographies until now
print_discarded(discarded_homographies)

## Show the final number of good homographies found
print("Found " + str(len(areas)) + " homographies")

## Show self similar statistics
print_self_similar_stats(inliers_per_image, self_similar_per_image, number_rescued_self_similar, good_rescued_self_similar_mask)

## Close debug file
discarded_file.close()