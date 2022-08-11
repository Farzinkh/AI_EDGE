import numpy as np
import cv2
#name="static_images/sample_images/image8"
name='embedded_image'
file=np.fromfile('{}'.format(name),dtype=np.dtype('u1'))
file=np.reshape(file,(96,96))
file = np.array(file, dtype=np.uint8)
# plt.imshow(file,cmap='gray')
# plt.show()
# plt.imsave('test.jpg',file,format='jpg',cmap='gray')
cv2.imshow('loaded image', file)
cv2.waitKey(0) 
cv2.imwrite("loaded_image.jpg",file)