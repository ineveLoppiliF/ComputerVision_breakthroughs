import matplotlib.pyplot as plt
from match_homography_excel_version import match_homography_excel_version
from self_similar_and_fingerprint_main_excel_version import self_similar_and_fingerprint_main_excel_version

templates=['../data/images/template/mickey.jpeg','../data/images/template/lipton_front.jpg','../data/images/template/template_twinings.jpg']
tests=['../data/images/test/find_mouse_hard.jpeg','../data/images/test/lipton_back_ordered.jpg','../data/images/test/twinings1.jpg']

self_count=[]
self_similar_used=[]
self_similar_not_used=[]
used_of_total_ss=[]
for i in range(len(tests)):
    (ssu,ssnu)=match_homography_excel_version(templates[i],tests[i])
    self_count.append(self_similar_and_fingerprint_main_excel_version(tests[i])) #self similar template
    self_similar_used.append(ssu) 
    self_similar_not_used.append(ssnu)
    if (ssu+ssnu)!=0:
        used_of_total_ss.append(ssu/(ssu+ssnu))
    else:
        used_of_total_ss.append(ssu)
plt.plot(self_count, used_of_total_ss,'ro')
##plt.axis([0, 6, 0, 20])
plt.show()