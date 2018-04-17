import matplotlib.pyplot as plt
from match_homography_excel_version import match_homography_excel_version

templates=['../data/images/template/template_twinings.jpg','../data/images/template/template_twinings.jpg']
tests=['../data/images/test/twinings5.jpg','../data/images/test/twinings1.jpg']

for i in range(len(tests)):
    match_homography_excel_version(templates[i],tests[i])
    

plt.plot([1,2,3,4], [1,4,9,16],'ro')
plt.axis([0, 6, 0, 20])
plt.show()