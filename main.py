import cv2
from seam_carving import SeamCarving

img = cv2.imread('img/img4.jpg')

sc = SeamCarving(img)

emap = sc.gen_emap()
smap = sc.gen_smap(emap)
cv2.imwrite('result/emap4.jpg', emap)
cv2.imwrite('result/smap4.jpg', smap)
# (1080, 1620, 3)
result = sc.resize((1080, 1600))
sc.visual_process('visual_process/process.gif')

