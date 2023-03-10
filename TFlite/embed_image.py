import cv2
import numpy as np
import os
from tqdm import tqdm
destination="put_my_content_on_sdcard"
if os.path.exists(destination):
    pass
else:
    os.makedirs(destination)

l=os.listdir('original_images')
count=0
for name in tqdm(l):
    name='original_images/'+name
    img=cv2.imread(name,cv2.IMREAD_GRAYSCALE)
    img=cv2.resize(img,(96,96))
    img = np.array(img, dtype=np.uint8)
    #cv2.imshow('embend image', img)
    #cv2.waitKey(0) 
    img.tofile(destination+"/"+"img"+str(count))
    count=count+1