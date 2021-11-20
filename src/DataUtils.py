import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

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


    def charToNum(self,txt):
        labelStr = []
        for alpha in txt:
            labelStr.append(self.charList.index(alpha))

        return pad_sequences([labelStr],maxlen=ModelConfig.MAX_TEXT_LEN,padding="post",value=len(self.charList))[0]


