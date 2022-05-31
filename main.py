import cv2
from seam_carving import SeamCarving

img = cv2.imread('img/img4.jpg')

sc = SeamCarving(img)

result = sc.gen_emap()
cv2.imwrite('result/result4.jpg', result)

