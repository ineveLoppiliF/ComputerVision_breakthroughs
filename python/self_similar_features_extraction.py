import numpy as np
from scipy.stats import t
ALPHA=0.95

def self_similar_features_extraction(matches,template_descriptors,t_parameters):
    
    quantiles=t.interval(ALPHA,t_parameters[0],t_parameters[1],t_parameters[2])
    #list of list that contains the self_similar features and their good matches 
    self_similar_list=[[] for i in range(len(matches))]
    execution_flag= False
    for i,kmatches in matches:
        j=0
        no_more_selfs= False
        while(j<len(kmatches) and no_more_selfs==False):
            template_descriptor1 = np.float32(template_descriptors[kmatches[j].trainIdx])
            template_descriptor2 = np.float32(template_descriptors[kmatches[j].queryIdx])
            distance = np.linalg.norm(template_descriptor1-template_descriptor2)
            if distance< min(quantiles):
                self_similar_list[i].extend(kmatches[j])
                if j==len(kmatches)-1:
                    execution_flag= True
            else:
                no_more_selfs= True
            
            j+=1
            
    return self_similar_list, execution_flag
            