import cv2
import numpy as np
name='20200130_024710.jpg'
img=cv2.imread(name,cv2.IMREAD_GRAYSCALE)
img=cv2.resize(img,(96,96))
img = np.array(img, dtype=np.uint8)
cv2.imshow('embend image', img)
cv2.waitKey(0) 
img.tofile("embedded_image")