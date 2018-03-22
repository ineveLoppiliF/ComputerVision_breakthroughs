from scipy.stats import norm

def validate_area(ALPHA, areas, area):
    
    if len(areas) < 2: return True
    t_areas_parameters = norm.fit(areas) # returned a list of three parameters (shape=parameters[0], mean=parameters[1] and std=parameters[2])
    areas_quantiles = norm.interval(ALPHA,t_areas_parameters[0],t_areas_parameters[1])
    print('-----')
    print('Area: '+str(areas))
    print('Mean: '+str(t_areas_parameters[0]))
    print('Std: '+str(t_areas_parameters[1]))
    print(str(areas_quantiles[0])+' < '+str(area)+' < '+str(areas_quantiles[1]))
    print('-----')
    if area >= areas_quantiles[0] and area <= areas_quantiles[1]: return True
    else: return False 
    