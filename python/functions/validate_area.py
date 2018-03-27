from scipy.stats import norm

def validate_area(ALPHA, areas, area):
    
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
    else: return False 
    