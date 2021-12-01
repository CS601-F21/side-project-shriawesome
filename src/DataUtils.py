import numpy as np
import tensorflow as tf
import cv2

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers.experimental.preprocessing import StringLookup


from ModelConfig import ModelConfig
"""
Helper class for the driver class to perfom different data preprocessing task
"""
class DataUtils:

    def __init__(self,charList):
        self.charList = charList
        self.char_to_num = StringLookup(vocabulary=list(self.charList), mask_token=None)
        self.num_to_char = StringLookup(vocabulary=self.char_to_num.get_vocabulary(), mask_token=None, invert=True)
        tf.compat.v1.enable_eager_execution()
    
    """
    Splits the data into training and testing.

    Parameters
    -----------
    X : input data for the model
    Y : target variable for the dataset
    threshold : % split in the data
    shuffle : if the data needs to be shuffled

    Returns
    -------
    xTrain, xVal, yTrain, yTest
    """
    def splitData(self,X, Y, threshold = 0.8, shuffle=True):
        size = X.shape[0]
        indices = np.arange(size)

        # Randomize the order of data
        if shuffle:
            np.random.shuffle(indices)
        
        splitPoint = int(size*threshold)
        xTrain, xVal = X[indices[:splitPoint]], X[indices[splitPoint:]]
        yTrain, yVal = Y[indices[:splitPoint]],Y[indices[splitPoint:]]

        return xTrain, yTrain, xVal, yVal 


    """
    Maps a character to an integer value

    Parameters
    ----------
    txt : texutal data to be converted to list of integers

    Returns
    -------
    Numpy array

    """
    def charToNum(self,txt):
        labelStr = []
        txt=txt.strip()
        for alpha in txt:
            labelStr.append(self.charList.index(alpha))

        return pad_sequences([labelStr],maxlen=ModelConfig.MAX_TEXT_LEN,padding="post",value=ModelConfig.PAD_TOKEN)[0]

    def numToChar(self,indices):
        txt=[]
        for idx in indices:
            if idx<len(self.charList) and idx>=0:
                txt.append(self.charList[idx])
        return("".join(txt))


    def resizeImg(self,img):
        w, h = ModelConfig.IMG_WIDTH, ModelConfig.IMG_HEIGHT
        img = tf.image.resize(img,size=(h,w))
        img = tf.transpose(img, perm=[1,0,2])
        img = tf.image.flip_left_right(img)
    
        return img



    def processSingleSample(self,imgPath, Label):
        # Read the input file
        img = tf.io.read_file(imgPath)
        img = tf.io.decode_png(img,1)
    
        # Rescaling the image to 64,800 
        img = tf.image.convert_image_dtype(img, tf.float32)   
        img = self.resizeImg(img)

        return {"image":img,"label":Label}

        

    def createPipeline(self, xTrain, yTrain, xVal, yVal):
        # Creating pipeline for training data
        try:
            trainData = tf.data.Dataset.from_tensor_slices((xTrain,yTrain))
            trainData = trainData.map(self.processSingleSample).batch(ModelConfig.BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            validationData = tf.data.Dataset.from_tensor_slices((xVal,yVal))
            validationData = validationData.map(self.processSingleSample,num_parallel_calls=tf.data.AUTOTUNE).batch(ModelConfig.BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        except:
            print("Error creating a pipeline")
    
        # Creating pipleline for validation data
       
        return trainData, validationData