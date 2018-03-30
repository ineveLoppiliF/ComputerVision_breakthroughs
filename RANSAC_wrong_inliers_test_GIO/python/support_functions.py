import numpy as np
import math
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
    
def plot_inliersXY( img2, dst, polygon, x, y, text ):
    in_inliers, out_inliers = [],[]
    for i in range(len(x)):
        point = Point(x[i], y[i])                    
        if polygon.contains(point):
            in_inliers.append(np.array([x[i],y[i]]))
        else:
            out_inliers.append(np.array([x[i],y[i]]))
    print('Number IN the polygon: ' + str(len(in_inliers)))
    print('Number OUT the polygon: ' + str(len(out_inliers)))
    try:
        print('IN/OUT ratio: ' + str(len(in_inliers)/len(out_inliers)))
    except ZeroDivisionError:
        print('Not possible to show IN/OUT ratio since OUT == 0')
    
    img3 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    img3 = cv2.polylines(img3, [np.int32(dst)], True, (0,0,255), 10, cv2.LINE_AA)
    print(in_inliers[1][0])
    for i in range(len(in_inliers)):
        point = np.array([in_inliers[i][0],in_inliers[i][1]])
        point.dtype=np.float32
        img3 = cv2.circle(img3, point, 10, (0,255,0), -1)
    for i in range(len(out_inliers)):
        point = np.array([out_inliers[i][0],out_inliers[i][1]])
        point.dtype=np.float32
        img3 = cv2.circle(img3, point, 10, (255,0,0), -1)
    plt.imshow(img3), plt.title(str(text)), plt.show()
    
def distance(M, src, dst):
    src = (src[0][0], src[0][1])
    dst = (dst[0][0], dst[0][1])
    dst2 = M.dot(np.array([src[0], src[1], 1]))
    dst2 = (dst2[0]/dst2[2], dst2[1]/dst2[2])
    return math.sqrt((dst[0]-dst2[0])**2 + (dst[1]-dst2[1])**2)

def transform(M, pts):
    pts = (pts[0][0], pts[0][1])
    pts2 = M.dot(np.array([pts[0], pts[1], 1]))
    pts2 = (pts2[0]/pts2[2], pts2[1]/pts2[2])
    return pts2

def transformXY(M, X, Y):
    pts2 = M.dot(np.array([X, Y, 1]))
    pts2 = (pts2[0]/pts2[2], pts2[1]/pts2[2])
    return Point(pts2[0],pts2[1])
    