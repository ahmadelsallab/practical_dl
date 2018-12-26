import keras
import numpy as np
from PIL import Image
from keras.applications.vgg16 import preprocess_input

def unpreprocess(im,mode):
    
    if mode == 'tf':
        tmp = (im+1)*127.5
        return tmp.astype(int)
    else:
        mean = [103.939, 116.779, 123.68]
        
        im_cp = np.copy(im)          
        im_cp[:, :,0] +=  mean[0]
        im_cp[:, :,1] += mean[1]
        im_cp[:, :,2] += mean[2]
    
        im_cp = im_cp[..., ::-1]
    
        return np.clip(im_cp/225.,0,1)
