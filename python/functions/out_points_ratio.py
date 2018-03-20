## Import libraries
from shapely.geometry import Point

## Function that count the number of points lying in a polygon and return the ratio respect tothe number of points itself
def out_points_ratio(points, pol):
    
     count = 0
     for i in range(len(points)):
         point = Point(points[i][0][0], points[i][0][1])                    
         if pol.contains(point):
             count+=1
     return count/len(points)