# Add your functions here 

def decrese_dimention(images,size,val):
    return images.reshape((len(images),size))

def expand_dimention(images,size,val):
    if not val:
        return images[:, :, :, None]
    else:
        return images[:,:, :,None]

def normalise_image(images,size,val):
    return images / 255.0

def preprocess(images,size,process,val=False):
    for f in process:
        images=globals()[f](images,size,val)
    return images