# Add your functions here 

def decrese_dimention(images,size,val):
    return images.reshape((images.shape[0],images.shape[1],images.shape[2]))

def expand_dimention(images,size,val):
    return images[:, :, :, None]

def normalise_image(images,size,val):
    return images / 255.0

def preprocess(images,size,process,val=False):
    for f in process:
        images=globals()[f](images,size,val)
    return images