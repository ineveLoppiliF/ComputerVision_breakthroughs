from scipy.stats import norm

def validate_area(ALPHA, areas, area, discarded_file, discarded_homographies):
    
    if len(areas) < 2: return True
    norm_areas_parameters = norm.fit(areas) # returned a list of two parameters (mean=parameters[0] and std=parameters[1])
    areas_quantiles = norm.interval(ALPHA,norm_areas_parameters[0],norm_areas_parameters[1])
    
    ##print('-----')
    ##print('Area: '+str(areas))
    ##print('Mean: '+str(norm_areas_parameters[0]))
    ##print('Std: '+str(norm_areas_parameters[1]))
    ##print(str(areas_quantiles[0])+' < '+str(area)+' < '+str(areas_quantiles[1]))
    ##print('-----')  
    
    if area >= areas_quantiles[0] and area <= areas_quantiles[1]: return True
    else: 
        discarded_homographies[0]+=1
        discarded_file.write("HOMOGRAPHY DISCARDED #"+str(discarded_homographies[0]+discarded_homographies[1]+discarded_homographies[2]+discarded_homographies[3]+discarded_homographies[4])+" (area too big)\n")
        discarded_file.write("Min bound: "+str(areas_quantiles[0])+"\nMax bound: "+str(areas_quantiles[1])+"\nArea: "+str(area)+"\n\n")
        return False