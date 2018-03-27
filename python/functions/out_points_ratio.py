## Import libraries
from shapely.geometry import Point

## Function that count the number of points lying in a polygon and return the ratio respect tothe number of points itself
def out_points_ratio(points, pol, discarded_file, IN_POLYGON_THRESHOLD, discarded_homographies):
    
     count = 0
     for i in range(len(points)):
         point = Point(points[i][0][0], points[i][0][1])                    
         if pol.contains(point):
             count+=1
     if((count/len(points))>=IN_POLYGON_THRESHOLD): return True
     else: 
         discarded_homographies[2]+=1
         discarded_file.write("HOMOGRAPHY DISCARDED #"+str(discarded_homographies[0]+discarded_homographies[1]+discarded_homographies[2]+discarded_homographies[3]+discarded_homographies[4])+" (inliers outside polygon)\n")
         discarded_file.write("Tot inliers: "+str(len(points))+"\nIn polygon inliers: "+str(count)+"\nRatio: "+str(count/len(points))+"\n\n")
         return False