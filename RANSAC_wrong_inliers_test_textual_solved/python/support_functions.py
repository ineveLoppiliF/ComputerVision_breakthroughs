import numpy as np
from matplotlib import pyplot as plt
import cv2
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


SINGULAR_VALUES_THRESHOLD = 0.00000001 # 0.1 is an (arbitrary) threshold


def indexToEliminate( dst_inliers, index_inliers ):
    totX = 0 
    totY = 0
    for point in dst_inliers:
        totX = totX + point[0][0]
        totY = totY + point[0][1]
    centroid = np.array([totX/len(dst_inliers), totY/len(dst_inliers)])
    
    maxDist = 0
    indexMaxDist = 0
    for i,point in enumerate(dst_inliers):
        dist = np.linalg.norm(centroid-point[0])
        if maxDist < dist:
            maxDist = dist
            indexMaxDist = i
    index_elem_to_elim = index_inliers[indexMaxDist]
    return index_elem_to_elim

def plot_inliers( img2, dst, dst_inliers, text ):
    polygon = Polygon([(dst[0][0][0], dst[0][0][1]), (dst[1][0][0], dst[1][0][1]),
                       (dst[2][0][0], dst[2][0][1]), (dst[3][0][0], dst[3][0][1])])
    in_inliers, out_inliers = [],[]
    for i in range(len(dst_inliers)):
        point = Point(dst_inliers[i][0][0], dst_inliers[i][0][1])                    
        if polygon.contains(point):
            in_inliers.append(dst_inliers[i][0])
        else:
            out_inliers.append(dst_inliers[i][0])
    print('Number of inliers IN the polygon: ' + str(len(in_inliers)))
    print('Number of inliers OUT the polygon: ' + str(len(out_inliers)))
    try:
        print('IN/OUT ratio: ' + str(len(in_inliers)/len(out_inliers)))
    except ZeroDivisionError:
        print('Not possible to show IN/OUT ratio since OUT == 0')
    
    img3 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    img3 = cv2.polylines(img3, [np.int32(dst)], True, (0,0,255), 10, cv2.LINE_AA)
    for i,point in enumerate(in_inliers):
        img3 = cv2.circle(img3, (point[0],point[1]), 10, (0,255,0), -1)
    for i,point in enumerate(out_inliers):
        img3 = cv2.circle(img3, (point[0],point[1]), 10, (255,0,0), -1)
    plt.imshow(img3), plt.title(str(text)), plt.show()
    
def project_and_plot_all_image_points(img1, img2, M, text):
    h, w = img1.shape
    h1, w1 = img2.shape
    
    temp = np.zeros((h1,w1,2), np.uint8)
    img3 = cv2.warpPerspective(img1,M, (w1, h1))
    img3 = np.dstack((temp,img3))
    
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    plt.imshow(img2+img3), plt.title(str(text)), plt.show()
    

## Checks if an homography H is degenerate
def is_homography_degenerate(H_inv, corner_scene):
    ## Where corner_template = H_inv * corner_scene, and corner_template are
    ## the coordinates of the template image corners
    
    ## Reshape corner_scene list to a more suitable one
    new_corner_scene = corner_scene.reshape(-1, 2)
    
    ## Transform points to projective coordinates padding data
    ## with homogeneous scale factor of 1    
    h, w = new_corner_scene.shape
    if w < 3:
        new_corner_scene = np.insert(new_corner_scene, w, 1, axis=1)
        
    ## By default the homography is considered degenerate
    is_degenerate = True
    
    ## Compute the singular values for the homography matrix,
    ## sorted in descending order
    S = np.linalg.svd(H_inv, compute_uv=False)
    
    ## Instead of checking the rank, more robustly check the ratio between
    ## the largest and the smallest singular value
    if S[2]/S[0] > SINGULAR_VALUES_THRESHOLD:
        if all(np.dot(H_inv[2,:],corner) > 0 for corner in new_corner_scene):            
            is_degenerate = False
        else:
            print("Twisted homography")
    else:
        print("Singular value threshold not passed")
    
    return is_degenerate