import keras
import numpy as np
import pandas as pd
from PIL import Image
import shutil
import os

def create_subdataset(PATH):
    os.mkdir(PATH/'subdataset')
    os.mkdir(PATH/'subdataset/train')
    os.mkdir(PATH/'subdataset/train/dogs')
    os.mkdir(PATH/'subdataset/train/cats')
    os.mkdir(PATH/'subdataset/valid')
    os.mkdir(PATH/'subdataset/valid/dogs')
    os.mkdir(PATH/'subdataset/valid/cats')
        
    input_path = PATH/'train/cats'
    output_path = PATH/'subdataset/train/cats'
    for f in os.listdir(input_path)[:2000]:
        shutil.copyfile(input_path/f,output_path/f)
    
    input_path = PATH/'train/dogs'
    output_path = PATH/'subdataset/train/dogs'
    for f in os.listdir(input_path)[:2000]:
        shutil.copyfile(input_path/f,output_path/f)
        
    input_path = PATH/'valid/cats'
    output_path = PATH/'subdataset/valid/cats'
    for f in os.listdir(input_path)[0:500]:
        shutil.copyfile(input_path/f,output_path/f)  
        
    input_path = PATH/'valid/dogs'
    output_path = PATH/'subdataset/valid/dogs'
    for f in os.listdir(input_path)[0:500]:
        shutil.copyfile(input_path/f,output_path/f)    


class Generator_SingleObject(keras.utils.Sequence):
    'Generates data from a Dataframe'
    def __init__(self, df, folder,preprocess_fct,batch_size=32, dim=(32,32), shuffle=True):
        'Initialization'
        self.preprocess_fct = preprocess_fct
        self.dim = dim
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.folder = folder
        self.class_name = ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair',
                           'cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep',
                           'sofa','train','tvmonitor']
        self.NbClasses = len(self.class_name)
        self.class_dict =dict((self.class_name[o],o) for o in range(self.NbClasses))

        self.df = df
        self.n = len(df)            
        self.nb_iteration = int(np.floor(self.n  / self.batch_size))
        
        self.on_epoch_end()
                    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.nb_iteration

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def next_item(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.df))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, index):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, 3))
        Y_bb = np.zeros((self.batch_size,4))
        Y_clas = np.zeros((self.batch_size,1))

        # Generate data
        for i, ID in enumerate(index):
            # Read the image
            img = Image.open(self.folder/self.df['filename'][ID])
            bb = self.df['bbox'][ID]
            bb = np.fromstring( bb, dtype=np.int, sep=' ' ) 
                              
            width, height = img.size
            RatioX = width/self.dim[0]
            RatioY = height/self.dim[1]
                                                        
            img = np.asarray(img.resize(self.dim))
            bb = [bb[0]/RatioY,bb[1]/RatioX,bb[2]/RatioY,bb[3]/RatioX]
                                                 
            X[i,] = self.preprocess_fct(np.asarray(img))
            Y_bb[i] = bb
            Y_clas[i] = self.class_dict[self.df['cat'][ID]]
            

        Y_clas = keras.utils.to_categorical(Y_clas, self.NbClasses)

        return X, [Y_bb,Y_clas]
    
class Generator_MultiObject(keras.utils.Sequence):
    'Generates data from a Dataframe'
    def __init__(self, df, folder,preprocess_fct,batch_size=32, dim=(32,32), shuffle=True):
        'Initialization'
        self.preprocess_fct = preprocess_fct
        self.dim = dim
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.folder = folder

        #self.NbClasses = len(self.class_name)
        #self.class_dict =dict((self.class_name[o],o) for o in range(self.NbClasses))

        self.df = df
        self.n = len(df)            
        self.nb_iteration = int(np.floor(self.n  / self.batch_size))
        
        self.on_epoch_end()
                    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.nb_iteration

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y
   
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.df))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, index):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        nb_label_max = 20
        X = np.empty((self.batch_size, *self.dim, 3))
        Y = []
        
        # Generate data
        for i, ID in enumerate(index):
            # Read the image
            img = Image.open(self.folder/self.df['filename'][ID])
            
            # extract the number of label
            c = self.df['class'][ID]
            nb_label = len(c)            
            
            # Class in a form of a one hot encoding
            y = np.zeros((nb_label_max,1+4));
            y[:nb_label,0] = c
            
            # reshape the bounding box and resize
            bbox = np.asmatrix(self.df['bbox'][ID])
            bbox = bbox.reshape(nb_label,4)
            
            bbox_rescaled = np.copy(bbox)    
            bbox_rescaled = bbox_rescaled.astype(float)
            width, height = img.size
            RatioX = width/self.dim[0]
            RatioY = height/self.dim[1]

            bbox_rescaled[:,0] = bbox_rescaled[:,0]/RatioY/self.dim[1]
            bbox_rescaled[:,1] = bbox_rescaled[:,1]/RatioX/self.dim[0]
            bbox_rescaled[:,2] = bbox_rescaled[:,2]/RatioY/self.dim[1]
            bbox_rescaled[:,3] = bbox_rescaled[:,3]/RatioX/self.dim[0]        
        
            # save the bb coordinates
            y[:nb_label,1:5] = bbox_rescaled
            
            # save the number of label
            #y[:nb_label,5] = nb_label
        
            # reshape to a vector
            y=np.reshape(y,nb_label_max*5)
                                                                    
            img = np.asarray(img.resize(self.dim))
            X[i,] = self.preprocess_fct(np.asarray(img))
            
            Y.append(np.asarray(y))

        Y = np.asarray(Y)
        
        return X, Y 
