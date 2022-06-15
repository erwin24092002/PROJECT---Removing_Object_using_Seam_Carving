import cv2
import numpy as np
from seam_carving import SeamCarving

# DEMO:
# Read image
img = cv2.imread('img/khinhkhicau.jpeg', 1)

# Init Seam Carving Object
sc = SeamCarving(img)

# Get mask
re_mask = sc.get_mask(desc='Locating the Removal Object') # removal mask
cv2.imwrite('mask/re_khinhkhicau.jpeg', re_mask)
pro_mask = sc.get_mask(desc='Locating the Protective Object') # protective mask
cv2.imwrite('mask/pro_khinhkhicau.jpeg', pro_mask)


# Remove Object
# result = sc.remove_object(removal_mask=re_mask)
result = sc.remove_object(removal_mask=re_mask, protective_mask=pro_mask)
cv2.imwrite('result/khinhkhicau.jpeg', result)

# Visualizing process
sc.visual_process('visual_process/khinhkhicau.gif')


# emap = sc.gen_emap()
# smap = sc.calc_backward_emap(emap)

# cv2.imwrite('emap.jpg', emap)
# cv2.imwrite('bemap.jpg', smap.astype(np.uint8))


