from DataLoader import DataLoader
from DataUtils import DataUtils
from Model import Model
import tensorflow as tf
from CTCLayer import CTCLayer
import cv2
import numpy as np

def loadSaveModel():
    loadModel = tf.keras.models.load_model("../models/HWRModel_v4.h5", custom_objects={'CTCLayer':CTCLayer})
    predictModel = tf.keras.models.Model(
        loadModel.get_layer(name='image').input, loadModel.get_layer(name="dense2").output
    )
    # predictModel.summary()

    return predictModel


def getImg(path):
    return dataUtil.processSingleSample(path," ")["image"]

def predictText(model,img):
    # Since there's just single image
    img = tf.expand_dims(img,0)
    # print(img.shape)
    prediction = model.predict(img)
    return Model.decodePrediction(prediction)

if __name__=="__main__":
    model = loadSaveModel()
    imgPath = "/Users/shrikantkendre/Documents/USF/Sem1/PSD/side-project-shriawesome/dataset/words/a01/a01-000u/a01-000u-00-05.png"
    #imgPath="/Users/shrikantkendre/Documents/USF/Sem1/PSD/side-project-shriawesome/dataset/test/good.png"
    loader = DataLoader(" ",100)
    charSet = loader.loadCharSet()
    dataUtil = DataUtils(charSet)
    processedImg = getImg(imgPath)
    prediction = predictText(model,processedImg)
    print(prediction)
