import cv2
import numpy as np
from seam_carving import SeamCarving

img = cv2.imread('img/img2.png', 1)

sc = SeamCarving(img)
mask = sc.get_mask()
cv2.imwrite('mask/test_mask.jpg', mask)

emap = sc.gen_emap()
cv2.imwrite('result/emap2.jpg', emap.astype(np.uint8))

smap = sc.gen_smap(emap)
cv2.imwrite('result/smap2.jpg', smap.astype(np.uint8))

result = sc.remove_object(mask)
cv2.imwrite('result/result2.jpg', result)

sc.visual_process('visual_process/process.gif')

