import DataLoader
from Model import Model
from ModelConfig import ModelConfig
from DataUtils import DataUtils
import cv2

import numpy as np

class Driver:

    def saveFile(fileName,data):
        with open("../"+fileName,"w") as file:
            file.write(data)

    def validateData(datapipeline):
        for batch in datapipeline.take(1):
            images = batch["image"]
            labels = batch["label"]
            cv2.imshow("",np.array(images[1]))
            cv2.waitKey(0)

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

        print(np.shape(yTrain))
    
        # Creating efficient input pipelines for tensorflow
        trainData, validationData = dataUtil.createPipeline(xTrain, yTrain, xVal, yVal)
        
        # Validate batches
        # validateData(trainData)

        # Creating Model Architecture
        getModel = Model(dataSet["Labels"])
        model = getModel.buildModel()
        model.summary()

