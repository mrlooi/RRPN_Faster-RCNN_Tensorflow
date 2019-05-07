import cv2
import numpy as np

RED = (0,0,255)
BLUE = (255,0,0)

x = np.array([
	[50,50],[0,50],[0,0],[50,0]
])
x2 = np.array([
	[60,60],[100,60],[100,100],[60,100]
])
x3 = np.array([
	[40,40],[60,70],[30,70]
])

polygons = [x,x2,x3]
cnt = np.concatenate(polygons)

img = np.zeros((200,200,3), dtype=np.uint8)
for poly in polygons:
	cv2.drawContours(img, [poly], 0, BLUE, 1)

rect = cv2.minAreaRect(cnt)
print(rect)
box = cv2.boxPoints(rect)
box = np.int0(box)
cv2.drawContours(img,[box],0,RED,2)

cv2.imshow("img", img)
cv2.waitKey(0)
