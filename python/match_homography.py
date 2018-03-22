#%% Import libraries
import numpy as np
import cv2
import matplotlib
from numpy.linalg import inv
from matplotlib import pyplot as plt
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from functions import out_area_ratio, out_points_ratio, remove_temporarily_matches, validate_area

#%% Initial initializations

## Constant parameters to be tuned
MIN_MATCH_COUNT = 30 # search for the template whether there are at least MIN_MATCH_CURENT good matches in the scene
MIN_MATCH_CURRENT = 10 # stop when your matched homography has less than that features
LOWE_THRESHOLD = 0.8 # a match is kept only if the distance with the closest match is lower than LOWE_THRESHOLD * the distance with the second best match
IN_POLYGON_THRESHOLD = 0.95 # homography kept only if at least this fraction of inliers are in the polygon
OUT_OF_IMAGE_THRESHOLD = 0.1 # Homography kept only if the square is not too much out from test image
ALPHA=0.9999999999999 # this constant allow us to determine the quantiles to be used to discriminate areas

## Load images 
template_image = cv2.imread('../data/images/template/lipton_front.jpg', cv2.IMREAD_COLOR) # template image
test_image = cv2.imread('../data/images/test/lipton_front_shuffle.jpg', cv2.IMREAD_COLOR)  # test image

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
print('found ' + str(len(matches)) + ' putative matches')

#%% Store all the good matches as per Lowe's ratio test
# Lowe's ratio test removes the ambiguous and false matches:
#   It keeps only matches where the distance with the closest match is lower 
#   than LOWE_THRESHOLD * the distance with the second best match
good_matches = []

## Need to keep only good matches, so create a mask, each row corresponds to a match
matches_mask = [[0,0] for i in iter(range(len(matches)))]

## Apply Lowe's test for each match, modifying the mask accordingly
for i,(m,n) in enumerate(matches):
    if m.distance < LOWE_THRESHOLD*n.distance:
        good_matches.append(m) # match appended to the list of good matches 
        matches_mask[i]=[1,0] # mask modified to consider the i-th match as good

## Show the number of good matches found
print('found ' + str(len(good_matches)) + ' matches validated by the distance ratio test')

## Specify parameters for the function that shows good matches graphically
draw_params = dict(matchColor = (0,255,0), # draw matches in green
                   singlePointColor = (255,0,0), # draw lone points in red
                   matchesMask = matches_mask, # draw only good matches
                   flags = 0)

## Good matches represented on another image
matches_image = cv2.drawMatchesKnn(test_image, test_keypoints, template_image, template_keypoints, matches, None, **draw_params)

## Set the size of the figure to show
matplotlib.rcParams["figure.figsize"]=(15,12)

## Plot the good matches
plt.imshow(cv2.cvtColor(matches_image, cv2.COLOR_BGR2RGB)), plt.title('All matches after ratio test'), plt.show()

#%% Cluster good matches by fitting homographies through RANSAC

input("Press Enter to continue...")

## Initilalize discarded homograpies counters
discarded_homographies = 0

## Initialize areas of founded homography
areas = []

## Initialize rectified objects' images list
#rectified_images = list()

## Initialize the buffer of temporary removed matches
temporary_removed_matches = list()

## Initialize the test image used to draw projected squares
test_image_squares = test_image.copy()

## Create a polygon using image dimension
## Create a polygon using the projected vertices
img_polygon = Polygon([(0,0), (0,test_image.shape[0]), (test_image.shape[1],test_image.shape[0]), (test_image.shape[1],0)])

## Continue to look for other homographies
end = False
while not end:
    ## If the number of remaining matches is low, is likely that there aren't other good homograpies, and the algorithm ends
    if len(good_matches) >= MIN_MATCH_COUNT:
        ## Retrieve coordinates of features keypoints in its image(the feature m.queryIdx inside test_image has been matched with feature m.trainIdx inside template_image)
        src_pts = np.float32([template_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([test_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        ## Apply RANSAC algorithm to fit homograpy: M is the final homography, mask represents the inliers
        H, inliers_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        ## If no available homograpies exist, the algorithm ends
        if H is not None:      
            ## Create a list representing all the inliers of the retrieved hompgrapy
            matches_mask = inliers_mask.ravel().tolist()
            
            ## Retrieve coordinates of the inliers in the test image, and their index wrt the actual good matches list
            dst_inliers = [dst_pts[i] for i in range(len(dst_pts)) if matches_mask[i]]
            index_inliers = [i for i in range(len(dst_pts)) if matches_mask[i]] 
            
            ## If the homography is degenerate, it is discarded
            if np.linalg.matrix_rank(H) == 3:
                ## If the retrieved homography has been fitted using few matches, 
                ## is likely that has poor performances and that there aren't other good homograpies, so the algorithm ends
                if np.count_nonzero(matches_mask) >= MIN_MATCH_CURRENT:
                    ## Project the vertices of the abstract rectangle around the template image
                    ## in the test one, using the found homography, in order to localize the template in the scene
                    h, w = template_image.shape[0:2]
                    src_vrtx = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                    dst_vrtx = cv2.perspectiveTransform(src_vrtx, H)  
                    
                    ## Create a polygon using the projected vertices
                    polygon = Polygon([(dst_vrtx[0][0][0], dst_vrtx[0][0][1]), (dst_vrtx[1][0][0], dst_vrtx[1][0][1]), (dst_vrtx[2][0][0], dst_vrtx[2][0][1]), (dst_vrtx[3][0][0], dst_vrtx[3][0][1])])
                            
                    ## Homography kept only if at least INSQUARE_TRESHOLD fraction of inliers are in the polygon, if the polygon is valid (no loop) and if is mostly inside the image
                    if polygon.is_valid and out_points_ratio(dst_inliers, polygon) >= IN_POLYGON_THRESHOLD and out_area_ratio(img_polygon, polygon) <= OUT_OF_IMAGE_THRESHOLD:
                        
                        ## Create a mask over the left good matches of the ones that are inliers
                        inliers_mask = np.zeros(len(good_matches))
                        for i in range(len(good_matches)):
                            if i in index_inliers:
                                inliers_mask[i] = 1
                        
                        ## Retrieve matches that are in the polygon, and their index wrt the actual good matches list
                        in_polygon_pts = [good_matches[i] for i in range(len(good_matches)) if inliers_mask[i]]
                        index_in_polygon_pts = [i for i in range(len(good_matches)) if inliers_mask[i]]
                        
                        ## Retrieve coordinates of features keypoints in its image, for ones that are in the polygon
                        new_src_pts = np.float32([template_keypoints[m.trainIdx].pt for m in in_polygon_pts]).reshape(-1, 1, 2)
                        new_dst_pts = np.float32([test_keypoints[m.queryIdx].pt for m in in_polygon_pts]).reshape(-1, 1, 2)
                        
                        ## Apply LMEDS algorithm to fit a new homograpy, taking into account all previous inliers
                        H, inliers_mask = cv2.findHomography(new_src_pts,new_dst_pts,cv2.LMEDS, 10.0)
                        
                        ## If no available homograpies exist, the algorithm ends
                        if H is not None:
                            ## Create a list representing all the inliers of the retrieved hompgrapy
                            matches_mask = inliers_mask.ravel().tolist()
                            
                            ## Retrieve coordinates of the inliers in the test image, and their index wrt the actual good matches list
                            dst_inliers = [new_dst_pts[i] for i in range(len(new_dst_pts)) if inliers_mask[i]]
                            index_inliers = [index for i,index in enumerate(index_in_polygon_pts) if inliers_mask[i]]
                            
                            ## If the homography is degenerate, it is discarded
                            if np.linalg.matrix_rank(H) == 3:
                            
                                ## Project the vertices of the abstract rectangle around the template image
                                ## in the test one, using the found homography, in order to localize the template in the scene
                                dst_vrtx = cv2.perspectiveTransform(src_vrtx, H)
                                
                                ## Create a polygon using the projected vertices
                                polygon = Polygon([(dst_vrtx[0][0][0], dst_vrtx[0][0][1]), (dst_vrtx[1][0][0], dst_vrtx[1][0][1]), (dst_vrtx[2][0][0], dst_vrtx[2][0][1]), (dst_vrtx[3][0][0], dst_vrtx[3][0][1])])
                               
                                ## Homography kept only if at least INSQUARE_TRESHOLD fraction of inliers are in the polygon and the polygon area is not too different from previous
                                if out_points_ratio(dst_inliers, polygon) >= IN_POLYGON_THRESHOLD:
                                    
                                    print('Number of inliers out of the homography:' +  str(len(dst_inliers) - (out_points_ratio(dst_inliers, polygon)*len(dst_inliers))))
                                    print('Fraction of inliers out of the homography:' +  str((len(dst_inliers) - (out_points_ratio(dst_inliers, polygon)*len(dst_inliers)))/len(dst_inliers)))
                                    
                                    ## Area confidence test
                                    if validate_area(ALPHA, areas, polygon.area): 
                                    
                                        areas.append(polygon.area) 
    
                                        ## Show the number of discarded homographies until now
                                        print('Discarded ' + str(discarded_homographies) + ' homographies until now')
                                        
                                        ## Draw the projected polygon in the test image, in order to visualize the found template in the test image
                                        polygons_image = cv2.polylines(test_image_squares, [np.int32(dst_vrtx)], True, [255,255,255], 3, cv2.LINE_AA)
                                        
                                        ## Specify parameters for the function that shows clustered matches, i.e. all the inliers for the selceted homography
                                        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green
                                                           singlePointColor=None,
                                                           matchesMask=matches_mask,  # draw only inliers
                                                           flags=2)
                                        
                                        ## Draw clustered matches
                                        matches_image = cv2.drawMatches(polygons_image, test_keypoints, template_image, template_keypoints, in_polygon_pts, None, **draw_params)
                                        
                                        ## Set the size of the figure to show
                                        matplotlib.rcParams["figure.figsize"]=(15,12)
                                        
                                        ## Plot the clustered matches
                                        plt.imshow(cv2.cvtColor(matches_image, cv2.COLOR_BGR2RGB)), plt.title('Clustered matches'), plt.show()
                                        
                                        ## Put back, inside the good matches list, points temporary removed
                                        good_matches.extend(temporary_removed_matches)
                                        temporary_removed_matches.clear()
                                        
                                        ## Create a mask over the left good matches of the ones that are in the polygon
                                        in_square_mask = np.zeros(len(good_matches))
                                        for i in range(len(good_matches)):
                                            point = Point((test_keypoints[good_matches[i].queryIdx].pt)[0], (test_keypoints[good_matches[i].queryIdx].pt)[1])
                                            if polygon.contains(point):
                                                in_square_mask[i] = 1
                                        
                                        ## Remove all matches in the polygon
                                        remove_mask = 1 - in_square_mask
                                        good_matches = [good_matches[i] for i in range(len(good_matches)) if remove_mask[i]]
                                
                                        ## Apply the inverse of the found homography to the scene image
                                        ## in order to rectify the object in the polygon and extract the 
                                        ## bounded image region from the rectified one containing the template instance
                                        H_inv = inv(H)
                                        rect_test_image = cv2.warpPerspective(test_image,H_inv,(w,h))
                                        
                                        ## Append the rectified image to the proper list
                                        #rectified_images.append(rect_test_image)
                                        
                                        ## Apply the homography to all test_keypoints in order to plot them
                                        object_test_keypoints_array = [0, 0]
                                        for keypoint in test_keypoints:
                                            object_test_keypoints_array = np.vstack((object_test_keypoints_array, [keypoint.pt[0], keypoint.pt[1]]))
                                        object_test_keypoints_array = np.delete(object_test_keypoints_array, (0), axis=0)
                                        object_test_keypoints_array = object_test_keypoints_array.reshape(-1, 1, 2)
                                        object_test_keypoints_array = cv2.perspectiveTransform(object_test_keypoints_array, H_inv)
                                        
                                        object_test_keypoints = list()
                                        for i,keypoint  in enumerate(object_test_keypoints_array):
                                            object_test_keypoints.append(cv2.KeyPoint(keypoint[0][0], keypoint[0][1], test_keypoints[i].size))
                                        
                                        ## Specify parameters for the function that shows clustered matches, i.e. all the inliers for the selceted homography
                                        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green
                                                           singlePointColor=None,
                                                           matchesMask=matches_mask,  # draw only inliers
                                                           flags=2)
                                        
                                        ## Draw clustered rectified matches
                                        matches_image = cv2.drawMatches(rect_test_image, object_test_keypoints, template_image, template_keypoints, in_polygon_pts, None, **draw_params)
                                        
                                        ## Show the rectified matches and image
                                        plt.imshow(cv2.cvtColor(matches_image, cv2.COLOR_BGR2RGB)), plt.title('Rectified object matches'), plt.show()
                                        rect_stacked_image = np.hstack((rect_test_image, template_image))
                                        plt.imshow(cv2.cvtColor(rect_stacked_image, cv2.COLOR_BGR2RGB)), plt.title('Rectified object image'), plt.show()
                                        
                                        ## Compute the difference between template and rectified image and plot it
                                        abs_diff_image = np.abs(template_image -  rect_test_image)
                                        plt.imshow(cv2.cvtColor(abs_diff_image, cv2.COLOR_BGR2RGB)), plt.title('Absolute difference image'),plt.show()
                                        color = ('B','G','R')
                                        for i in range(np.size(abs_diff_image,2)):
                                            plt.subplot(1,3,i+1)
                                            plt.imshow(abs_diff_image[:,:,i],'gray')
                                            plt.title(str(color[i])+' difference')
                                        plt.show()
                                        
                                        ## Create differences histograms
                                        abs_diff_hist = list()
                                        for i in range(np.size(abs_diff_image,2)):
                                            abs_diff_hist.append(cv2.calcHist([abs_diff_image],[i],None,[256],[0,256]))
         
                                        ## Print differences histograms
                                        for i,col in enumerate(color):
                                            plt.plot(abs_diff_hist[i],color = col) 
                                            plt.xlim([0,256])
                                        plt.show()                                        
                                        
                                        ## Show the number of good matches left
                                        print('There remains: ' + str(len(good_matches)) + ' features')
                                        
                                        ## Show the number of good homograpies until now
                                        print("Found " + str(len(areas)) + " homographies until now")
                                        
                                        ## Search for the next template in the test image after a user command
                                        input("Press Enter to continue...")
                                    else:
                                        print("Homography too big respect to previous founded")
                                        discarded_homographies+=1
                                        good_matches, temporary_removed_matches = remove_temporarily_matches(good_matches,temporary_removed_matches,dst_inliers,index_inliers)
                                else:
                                    discarded_homographies+=1
                                    good_matches, temporary_removed_matches = remove_temporarily_matches(good_matches,temporary_removed_matches,dst_inliers,index_inliers)
                            else:
                                print("Degenerate homography")
                                discarded_homographies+=1
                                good_matches, temporary_removed_matches = remove_temporarily_matches(good_matches,temporary_removed_matches,dst_inliers,index_inliers)
                        else:
                            print("Not possible to find another homography")
                            end = True
                    else:
                        discarded_homographies+=1
                        good_matches, temporary_removed_matches = remove_temporarily_matches(good_matches,temporary_removed_matches,dst_inliers,index_inliers)
                else:
                    print("Not enough matches are found in the last homography - {}/{}".format(np.count_nonzero(matches_mask), MIN_MATCH_CURRENT))
                    end = True
            else:
                print("Degenerate homography")
                discarded_homographies+=1
                good_matches, temporary_removed_matches = remove_temporarily_matches(good_matches,temporary_removed_matches,dst_inliers,index_inliers)
        else:
            print("Not possible to find another homography")
            end = True
    else:
        print("Not enough matches are found - {}/{}".format(len(good_matches), MIN_MATCH_COUNT))
        end = True
        

    
## Show the final image, in which all templates found are drawn
if len(areas)!=0: plt.imshow(cv2.cvtColor(polygons_image, cv2.COLOR_BGR2RGB)), plt.title('final image'),plt.show()

## Show the final number of good homographies found
print("Found " + str(len(areas)) + " homographies")

## Show all the rectified image regions
#answer = input("Show rectified images? [Y/n]")
#if answer == "" or answer.lower() == "y":
#    for i,rectified_image in enumerate(rectified_images):
#        plt.imshow(np.hstack((rectified_image, template_image)), 'gray'), plt.title('Rectified obect n° ' + str(i)), plt.show()