import numpy as np
import tensorflow as tf
import cv2
from tensorflow._api.v2 import image
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.ops.gen_array_ops import pad
from tensorflow.python.ops.gen_math_ops import Mod

from ModelConfig import ModelConfig
"""
Helper class for the driver class to perfom different data preprocessing task
"""
class DataUtils:

    def __init__(self,charList):
        self.charList = charList
    
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
        for alpha in txt:
            labelStr.append(self.charList.index(alpha))

        return pad_sequences([labelStr],maxlen=ModelConfig.MAX_TEXT_LEN,padding="post",value=len(self.charList))[0]

    def resizeImg(self,img):
        w, h = ModelConfig.IMG_WIDTH, ModelConfig.IMG_HEIGHT
        img = tf.image.resize(img,[h,w],preserve_aspect_ratio=True)

        # Padding to be done
        padHeight = h - img.shape[0]
        padWidth = w  - img.shape[1]
        print(padHeight)

        # Padding on both the ends, if even the same amount of padding on top and
        # bottom else more on top and less on bottom
        if padHeight%2 != 0 :
            height = padHeight // 2
            padHeightTop = height + 1
            padHeightBottom = height
        else:
            padHeightTop = padHeightBottom = padHeight // 2 

        if padWidth%2 != 0:
            width = padWidth // 2
            padWidthLeft = width + 1
            padWidthRight = width
        else:
            padWidthLeft = padWidthRight = padWidth // 2

        img = tf.pad(img,paddings=[
            [padHeightTop,padHeightBottom],
            [padWidthLeft,padWidthRight],
            [0,0]
        ],)

        img = tf.transpose(img, perm=[1,0,2])
        img = tf.image.flip_left_right(img)
        return img



    def processSingleSample(self,imgPath, Label):
        # Read the input file
        img = tf.io.read_file(imgPath)
        img = tf.io.decode_png(img,1)
        # scalling the image between [0,1]
        img = tf.image.convert_image_dtype(img, tf.float32]
    
        # Rescaling the image to 64,800    
        img = self.resizeImg(img)
        # print(img.shape)
        # cv2.imshow("",np.array(img))
        # cv2.waitKey(0)

        

    def createPipeline(self, xTrain, yTrain, xVal, yVal):
        trainData = tf.data.Dataset.from_tensor_slices((xTrain,yTrain))
        # encoding the image in a numpy array
        # trainData = trainData.map(self.encodeSingleSample).batch(ModelConfig.BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        
        
        validationData = tf.data.Dataset.from_tensor_slices((xVal,yVal))

        return trainData, validationData