from tensorflow._api.v2 import data
from tensorflow.python.ops.variables import trainable_variables
import DataLoader
from ModelConfig import ModelConfig
from DataUtils import DataUtils

import numpy as np

class Driver:

    def saveFile(fileName,data):
        with open("../"+fileName,"w") as file:
            file.write(data)

    if __name__=="__main__":
        # Generate dataset
        dataLoader = DataLoader.DataLoader(ModelConfig.DATA_CONFIG_FILE,ModelConfig.MAX_TEXT_LEN)
        dataSet = dataLoader.getData()

        # Save character list to the file
        saveFile("charList.txt",''.join(dataSet["Labels"]))

        # Extract the input and the target variable from the dataset
        X = np.array([imgPath.imgPath for imgPath in dataSet["dataset"]])
        Y = np.array([text.text for text in dataSet["dataset"]])
        
    
        # Splitting the dataset into training and validation 
        dataUtil = DataUtils(dataSet["Labels"])   
        xTrain, yTrain, xVal, yVal = dataUtil.splitData(X,Y,0.8)
        print("Shapes : X_TRAIN : {}, X_VAL : {}, Y_TRAIN : {}, Y_VAL : {}".format(xTrain.shape,xVal.shape,yTrain.shape,yVal.shape))
        
        # Mapping a string value to an Integer value
        yTrain = list(map(dataUtil.charToNum,yTrain))
        yVal = list(map(dataUtil.charToNum,yVal))
    
        # Creating efficient input pipelines for tensorflow
        # trainData, validationData = dataUtil.createPipeline(xTrain, yTrain, xVal, yVal)
        #print(X[0])
        dataUtil.processSingleSample("../dataset/lines/h02/h02-004/h02-004-09.png","this is it")