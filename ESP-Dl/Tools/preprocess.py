# Add your functions here 

process=['expand_dimention']

def decrese_dimention(images,size):
    return images.reshape((len(images),size))

def expand_dimention(images,size):
    if (len(images.shape)==3):
        return images[:, :, :, None]
    else:
        return images[:, :, None]

def normalise_image(images,size):
    return images / 255.0

def preprocess(images,size):
    for f in process:
        images=globals()[f](images,size)
    return images