import numpy as np
import cv2
import matplotlib
from matplotlib import pyplot as plt
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from support_functions import indexToEliminate, plot_inliers

MIN_MATCH_COUNT = 30 # search for the template whether there are at least MIN_MATCH_CURENT good matches in the scene
MIN_MATCH_CURRENT =10  #stop when your matched homography has less than that features
INSQUARE_THRESHOLD = 0.95

matplotlib.rcParams["figure.figsize"]=(15,12)

img1 = cv2.imread('./../../data/images/template/lipton_front.jpg', 0) # template
img2 = cv2.imread('./../../data/images/test/lipton_front_shuffle.jpg', 0)  # testImage

plt.imshow(img1, 'gray'), plt.title('template'),plt.show()
plt.imshow(img2, 'gray'), plt.title('image'),plt.show()

#%%  Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT from the template
kp1,des1 = sift.detectAndCompute(img1, None)
# kp1 list of keypoints kp1[0].pt = location, kp1[0].angle = orientation kp[0].octave provide scale information
print('found ' + str(len(kp1)) + ' keypoints in image 1')

kp2,des2  = sift.detectAndCompute(img2, None)
print('found ' + str(len(kp2)) + ' keypoints in image 2')

#%% Initialize a FLANN object to match keypoint witn nearest neighborhood. 

# = This comes from FLANN documentation ============================================================================
# 	FLANN_INDEX_LINEAR = 0,
# 	FLANN_INDEX_KDTREE = 1,
# 	FLANN_INDEX_KMEANS = 2,
# 	FLANN_INDEX_COMPOSITE = 3,
# 	FLANN_INDEX_KDTREE_SINGLE = 4,
# 	FLANN_INDEX_HIERARCHICAL = 5,
# 	FLANN_INDEX_LSH = 6,
# 	FLANN_INDEX_KDTREE_CUDA = 7, // available if compiled with CUDA
# 	FLANN_INDEX_SAVED = 254,
# 	FLANN_INDEX_AUTOTUNED = 255,
# =============================================================================

FLANN_INDEX_KDTREE = 1

index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

#%% find correspondences by matching the image features with the template features, 
# note it is not the same as matching des1 with des2!
# invoke flann methods to obtain 2 outputs: both the first and the second match for each feature in img2

matches =  flann.knnMatch(des2,des1,k=2)#there is no trehsold, the closest point is returned.
# For each feature in the des2 image returns the two closest features in the first
print('found ' + str(len(matches)) + ' putative matches')

#%% 

# store all the good matches as per Lowe's ratio test.
good = []
# Need to draw only good matches, so create a mask, each row corresponds to a match
matchesMask = [[0,0] for i in iter(range(len(matches)))]

# the ratio test removes the ambiguous and false matches 
# keep only matches where the distance with the cloesest match is lower 
# than 0.8 * the distance with the second best match
for i,(m,n) in enumerate(matches):
    # implement ratio test check and in case it passes, use the following instructions
    if m.distance < 0.8*n.distance:
        good.append(m)
        matchesMask[i]=[1,0]
       
print('found ' + str(len(good)) + ' matches validated by the distance ratio test')

# plot all the good matches
draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)
img3 = cv2.drawMatchesKnn(img2, kp2, img1, kp1, matches,None,**draw_params)

plt.imshow(img3, 'gray'), plt.title('all matches after ratio test'), plt.show()

#%% cluster good matches by fitting homographies through RANSAC
input("Press Enter to continue...")

found_homographies = 0
discarded_homographies = 0
removed = []
while matchesMask != None:
    if len(good) >= MIN_MATCH_COUNT:
        # the feature m.queryIdx inside img2 has been matched with feature m.trainIdx inside img1
        src_pts = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        
        # RANSAC to fit homograpy: M is the final homography, mask represent the inliers
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if M is not None:
            # this is a list containing all the inliers
            matchesMask = mask.ravel().tolist()
            dst_inliers = [dst_pts[i] for i in range(len(dst_pts)) if matchesMask[i]]
            index_inliers = np.nonzero(matchesMask)[0]
            
            # Control whether the homography is non-degenerate
            if np.linalg.matrix_rank(M) == 3:
                if np.count_nonzero(matchesMask) >= MIN_MATCH_CURRENT:
                    # draw the rectangle in the scene corresponding to the estimated homography
                    h, w = img1.shape
                    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                    dst = cv2.perspectiveTransform(pts, M)    
                    
                    polygon = Polygon([(dst[0][0][0], dst[0][0][1]), (dst[1][0][0], dst[1][0][1]), (dst[2][0][0], dst[2][0][1]), (dst[3][0][0], dst[3][0][1])])
                    
                    # Plot the projected polygon in blue, in_polygon inliers in green and out_polygon inliers in red
                    plot_inliers(img2, dst, dst_inliers, 'First homography inliers')
                    
                    count = 0
                    for i in range(len(dst_inliers)):
                        point = Point(dst_inliers[i][0][0], dst_inliers[i][0][1])                    
                        if polygon.contains(point):
                            count+=1
                            
                    if count/len(dst_inliers) >= INSQUARE_THRESHOLD:                    
                        # Create a mask over the left good matches of the ones that are inliers
                        inliers_mask = np.zeros(len(good))
                        for i in range(len(good)):
                            if i in index_inliers:
                                inliers_mask[i] = 1
                        in_square_points = [good[i] for i in range(len(good)) if inliers_mask[i]]
                        index_in_square_points = [i for i in range(len(good)) if inliers_mask[i]]
                        
                        new_src_pts = np.float32([kp1[m.trainIdx].pt for m in in_square_points]).reshape(-1, 1, 2)
                        new_dst_pts = np.float32([kp2[m.queryIdx].pt for m in in_square_points]).reshape(-1, 1, 2)
                        M, mask = cv2.findHomography(new_src_pts, new_dst_pts, cv2.LMEDS, 10.0)
                        
                        if M is not None:
                            matchesMask = mask.ravel().tolist()
                            dst_inliers = [new_dst_pts[i] for i in range(len(new_dst_pts)) if mask[i]]
                            index_inliers = [index for i,index in enumerate(index_in_square_points) if mask[i]]
                            
                            if np.linalg.matrix_rank(M) == 3:
                            
                                dst = cv2.perspectiveTransform(pts, M)
                                
                                polygon = Polygon([(dst[0][0][0], dst[0][0][1]), (dst[1][0][0], dst[1][0][1]), (dst[2][0][0], dst[2][0][1]), (dst[3][0][0], dst[3][0][1])])
                                
                                # Plot the projected polygon in blue, in_polygon inliers in green and out_polygon inliers in red
                                plot_inliers(img2, dst, dst_inliers, 'Second homography inliers')
                                
                                count = 0
                                for i in range(len(dst_inliers)):
                                    point = Point(dst_inliers[i][0][0], dst_inliers[i][0][1])                    
                                    if polygon.contains(point):
                                        count+=1
                                
                                if count/len(dst_inliers) >= INSQUARE_THRESHOLD:
                                    print('Discarded ' + str(discarded_homographies) + ' homographies until now')
                                                        
                                    img3 = cv2.polylines(img2, [np.int32(dst)], True, 255, 10, cv2.LINE_AA)
                                    
                                    # draw clustered matches, i.e. all the inliers for the selceted homography
                                    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                                                       singlePointColor=None,
                                                       matchesMask=matchesMask,  # draw only inliers
                                                       flags=2)
                                    img4 = cv2.drawMatches(img3, kp2, img1, kp1, in_square_points, None, **draw_params)                
                                    
                                    plt.imshow(img4, 'gray'), plt.show()
                                    
                                    polygon = Polygon([(dst[0][0][0], dst[0][0][1]), (dst[1][0][0], dst[1][0][1]), (dst[2][0][0], dst[2][0][1]), (dst[3][0][0], dst[3][0][1])])
                                    inSquareMask = np.zeros(len(good))
                                    for i in range(len(dst_pts)):
                                        point = Point(dst_pts[i][0][0], dst_pts[i][0][1])
                                        if polygon.contains(point):
                                            inSquareMask[i] = 1
                                    
                                    #Remove all matches in the square
                                    removeMask = 1 - inSquareMask
                                    good = [good[i] for i in range(len(good)) if removeMask[i]]
                                    
                                    good.extend(removed)
                                    removed.clear()
                                    
                                    print('There remains: ' + str(len(good)) + ' features')
                                    
                                    found_homographies+=1
                                    print("Found " + str(found_homographies) + " homographies until now")
                                    
                                    input("Press Enter to continue...")
                                else:
                                    discarded_homographies+=1
                                
                                    index_elem_to_elim = indexToEliminate(dst_inliers, index_inliers)
                                    removeMask = np.ones(len(good))
                                    removeMask[index_elem_to_elim] = 0
                                    removed.extend([good[i] for i in range(len(good)) if not removeMask[i]])
                                    good = [good[i] for i in range(len(good)) if removeMask[i]]                    
                            else:
                                print("Degenerate homography")
                                discarded_homographies+=1
                                
                                index_elem_to_elim = indexToEliminate(dst_inliers, index_inliers)
                                removeMask = np.ones(len(good))
                                removeMask[index_elem_to_elim] = 0
                                removed.extend([good[i] for i in range(len(good)) if not removeMask[i]])
                                good = [good[i] for i in range(len(good)) if removeMask[i]] 
                        else:
                            print("Not possible to find another homography")
                            matchesMask = None
                    else:
                        discarded_homographies+=1
                        
                        index_elem_to_elim = indexToEliminate(dst_inliers, index_inliers)
                        removeMask = np.ones(len(good))
                        removeMask[index_elem_to_elim] = 0
                        removed.extend([good[i] for i in range(len(good)) if not removeMask[i]])
                        good = [good[i] for i in range(len(good)) if removeMask[i]]
                else:
                    print("Not enough matches are found in the last homography - {}/{}".format(np.count_nonzero(matchesMask), MIN_MATCH_CURRENT))
                    matchesMask = None
            else:
                print("Degenerate homography")
                discarded_homographies+=1
                            
                index_elem_to_elim = indexToEliminate(dst_inliers, index_inliers)
                removeMask = np.ones(len(good))
                removeMask[index_elem_to_elim] = 0
                removed.extend([good[i] for i in range(len(good)) if not removeMask[i]])
                good = [good[i] for i in range(len(good)) if removeMask[i]] 
        else:
            print("Not possible to find another homography")
            matchesMask = None
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None
plt.imshow(img3, 'gray'), plt.title('final image'),plt.show()
print("Found " + str(found_homographies) + " homographies")
