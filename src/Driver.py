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
        xTrain, yTrain, xTest, yTest = dataUtil.splitData(X,Y,0.8)
        print("Shapes : X_TRAIN : {}, X_VAL : {}, Y_TRAIN : {}, Y_VAL : {}".format(xTrain.shape,xTest.shape,yTrain.shape,yTest.shape))
        
        # Mapping a string value to an Integer value
        yTrain = list(map(dataUtil.charToNum,yTrain))
        yTest = list(map(dataUtil.charToNum,yTest))
        print(len(yTrain),len(yTest))
       # print(DataUtils.charToNum("hello",dataSet["Labels"],ModelConfig.MAX_TEXT_LEN))