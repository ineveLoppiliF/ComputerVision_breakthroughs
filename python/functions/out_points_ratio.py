## Import libraries
from shapely.geometry import Point

## Function that count the number of points lying in a polygon and return the ratio respect tothe number of points itself
def out_points_ratio(points, pol, discarded_file, IN_POLYGON_THRESHOLD, discarded_homographies, discarded_cont_count):
    
     if pol.is_valid:
         count = 0
         for i in range(len(points)):
             point = Point(points[i][0][0], points[i][0][1])                    
             if pol.contains(point):
                 count+=1
         if((count/len(points))>=IN_POLYGON_THRESHOLD): return True
         else:
             discarded_cont_count[0]+=1
             discarded_homographies[4]+=1
             discarded_file.write("HOMOGRAPHY DISCARDED #"+str(sum(discarded_homographies))+" (inliers outside polygon)\n")
             discarded_file.write("Tot inliers: "+str(len(points))+"\nIn polygon inliers: "+str(count)+"\nRatio: "+str(count/len(points))+"\n\n")
             return False
     else:
         discarded_cont_count[0]+=1
         discarded_homographies[5]+=1
         discarded_file.write("HOMOGRAPHY DISCARDED #"+str(sum(discarded_homographies))+" (non valid polygon)\n")
         return False