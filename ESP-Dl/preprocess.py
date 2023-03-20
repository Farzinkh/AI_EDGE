# Add your functions here 

process=['expand_dimention']

def expand_dimention(images):
    if (len(images.shape)==3):
        return images[:, :, :, None]
    else:
        return images[:, :, None]

def normalise_image(images):
    return images / 255.0

def preprocess(images):
    for f in process:
        images=globals()[f](images)
    return images