import matplotlib.pyplot as plt
from match_homography_excel_version import match_homography_excel_version
from self_similar_and_fingerprint_main_excel_version import self_similar_and_fingerprint_main_excel_version

templates=['../data/images/template/mickey.jpeg','../data/images/template/lipton_front.jpg','../data/images/template/template_twinings.jpg','../data/images/template/adidas-logoRB.jpg']
tests=['../data/images/test/find_mouse_hard.jpeg','../data/images/test/lipton_back_ordered.jpg','../data/images/test/twinings1.jpg','../data/images/test/pressAds.png' ]

self_count=[] ##list of the numbers of ss in the templates
self_similar_used=[] ##list of quantities of ss features used 
self_similar_not_used=[] #list of quantities of ss features fished out but not used
used_of_total_ss=[] #for every image used of total ss features

good_matches=[]#list of good matches for the used images
ss_of_good=[]#for every image quantity of ss used of good matches
for i in range(len(tests)):
    (ssu,ssnu,good)=match_homography_excel_version(templates[i],tests[i])
    self_count.append(self_similar_and_fingerprint_main_excel_version(tests[i])) #self similar template
    self_similar_used.append(ssu) 
    self_similar_not_used.append(ssnu)
    good_matches.append(good)
    if (ssu+ssnu)!=0:
        used_of_total_ss.append(ssu/(ssu+ssnu))
    else:
        used_of_total_ss.append(ssu)
    if (good-ssnu)!=0:    
        ss_of_good.append(ssu/(good-ssnu))
    else:
        ss_of_good.append(ssu)
##plot the quantity of ssused of all the ss features respect to the number of the ssfeatures in the template 
plt.plot(self_count, used_of_total_ss,'ro')
plt.show()
##plot the quantity of ssused of all the good matches with the growth of good matches in the image (more definition)
plt.plot(good_matches,ss_of_good,'ro')
plt.show() 
