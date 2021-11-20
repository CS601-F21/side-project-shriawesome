import numpy as np

"""
Helper class for the driver class to perfom different data preprocessing task
"""
class DataUtils:
    
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
    def splitData(X, Y, threshold = 0.8, shuffle=True):
        size = X.shape[0]
        indices = np.arange(size)

        # Randomize the order of data
        if shuffle:
            np.random.shuffle(indices)
        
        splitPoint = int(size*threshold)
        xTrain, xVal = X[indices[:splitPoint]], X[indices[splitPoint:]]
        yTrain, yVal = Y[indices[:splitPoint]],Y[indices[splitPoint:]]

        return xTrain, yTrain, xVal, yVal 

