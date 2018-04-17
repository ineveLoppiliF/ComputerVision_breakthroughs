import matplotlib.pyplot as plt
from match_homography_excel_version import match_homography_excel_version

templates=['../data/images/template/mickey.jpeg','../data/images/template/lipton_front.jpg','../data/images/template/template_twinings.jpg']
tests=['../data/images/test/find_mouse_hard.jpeg','../data/images/test/lipton_back_ordered.jpg','../data/images/test/twinings1.jpg']

self_similar_perc=[]
tot_matches=[]
for i in range(len(tests)):
    (ss,tm)=match_homography_excel_version(templates[i],tests[i])
    self_similar_perc.append(ss)
    tot_matches.append(tm)

plt.plot(tot_matches, self_similar_perc,'ro')
##plt.axis([0, 6, 0, 20])
plt.show()