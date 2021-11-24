import DataLoader
from Model import Model
from ModelConfig import ModelConfig
from DataUtils import DataUtils
import tensorflow as tf
from tensorflow import keras
import cv2

import numpy as np

class Driver:

    def saveFile(fileName,data):
        with open("../"+fileName,"w") as file:
            file.write(data)

    def validateData(datapipeline,dataUtils):
        for batch in datapipeline.take(1):
            images = batch["image"]
            labels = batch["label"]
            img = tf.image.flip_left_right(images[1])
            img = tf.transpose(img,perm=[1,0,2])
            img = (img*255.0).numpy().clip(0,255).astype(np.uint8)
            img = img[:,:,0]
            indices = labels[1].numpy()
            label = dataUtils.numToChar(indices)
            #print(indices)
            cv2.imshow(label,img)
            cv2.waitKey(0)

    def startTraining(model, trainData, valData):
        # Setting up early stopping checkpoint
        earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=ModelConfig.EARLY_STOP_PATIENCE,
                                                    restore_best_weights=True)

        checkpoints = keras.callbacks.ModelCheckpoint(filepath=ModelConfig.SAVE_MODEL,
                                                      monitor='val_loss',
                                                      verbose=1,
                                                      save_best_only=True,
                                                      mode='min')

        # Starting training the model
        history = model.fit(trainData,
                            validation_data=valData,
                            epochs=ModelConfig.EPOCHS,
                            callbacks=[checkpoints,earlyStopping])
        


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
        trainData, valData = dataUtil.createPipeline(xTrain, yTrain, xVal, yVal)
        
        # Validate batches
        # validateData(trainData,dataUtil)

        # # Creating Model Architecture
        # getModel = Model(dataSet["Labels"])
        # model = getModel.buildModel()
        # # model.summary()

        # # Training the model
        # startTraining(model, trainData, valData)

