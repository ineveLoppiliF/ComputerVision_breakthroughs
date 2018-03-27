import numpy as np
from matplotlib import pyplot as plt
import cv2
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

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