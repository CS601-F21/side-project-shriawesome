from numpy import float32
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.backend import dtype
from CTCLayer import CTCLayer
from ModelConfig import ModelConfig

class Model:
    
    def __init__(self,charList):
        self.charList = charList 

    def buildModel(self):
        inputImgs = keras.layers.Input(shape=(ModelConfig.IMG_WIDTH,ModelConfig.IMG_HEIGHT,1),name="image")

        # First Layer : Conv(5X5) + Pool (2x2) - output: 400 x 32 x 64
        x = keras.layers.Conv2D(64,(5,5),(1,1),padding="same",kernel_initializer="truncated_normal")(inputImgs)
        x = keras.layers.LeakyReLU(alpha=0.01)(x)
        x = keras.layers.MaxPool2D((2,2),(2,2),padding='valid')(x)

        # Second Layer : Conv(5x5) + Pool(1x2) - output: 400x16x128
        x = keras.layers.Conv2D(128,(5,5),(1,1),padding="same",kernel_initializer="truncated_normal")(x)
        x = keras.layers.LeakyReLU(alpha=0.01)(x)
        x = keras.layers.MaxPool2D((1,2),(1,2),padding='valid')(x)

        # Third Layer : Conv(3x3) + Pool(2x2) - output: 200x8x128
        x = keras.layers.Conv2D(128,(3,3),(1,1),padding="same",kernel_initializer="truncated_normal")(x)
        x = keras.layers.BatchNormalization(axis=[1,2,3],epsilon=0.001)(x)
        x = keras.layers.LeakyReLU(alpha=0.01)(x)
        x = keras.layers.MaxPool2D((2,2),(2,2),padding='valid')(x)

        # Forth Layer : Conv(3x3) - output: 200x8x256
        x = keras.layers.Conv2D(256,(3,3),(1,1),padding="same",kernel_initializer="truncated_normal")(x)
        x = keras.layers.LeakyReLU(alpha=0.01)(x)

        # Fifth Layer : Conv(3x3) + Pool(2x2) - output: 100x4x256
        x = keras.layers.Conv2D(256,(3,3),(1,1),padding="same",kernel_initializer="truncated_normal")(x)
        x = keras.layers.LeakyReLU(alpha=0.01)(x)
        x = keras.layers.MaxPool2D((2,2),(2,2),padding='valid')(x)

        # Sixth Layer : Conv(3x3) + Pool(1x2) + Simple Batch Norm - output: 100x2x512
        x = keras.layers.Conv2D(512,(3,3),(1,1),padding="same",kernel_initializer="truncated_normal")(x)
        x = keras.layers.BatchNormalization(axis=[1,2,3],epsilon=0.001)(x)
        x = keras.layers.LeakyReLU(alpha=0.01)(x)
        x = keras.layers.MaxPool2D((1,2),(1,2),padding='valid')(x)
    
        # Seventh Layer : Conv(3x3) + Pool(1x2) : Output : 100 x 1 x 512
        x = keras.layers.Conv2D(512,(3,3),(1,1),padding="same",kernel_initializer="truncated_normal")(x)
        x = keras.layers.LeakyReLU(alpha=0.01)(x)
        x = keras.layers.MaxPool2D((1,2),(1,2),padding='valid')(x)

        # Removing the dimension 100x1x512
        x = keras.layers.Reshape((100,512))(x)

        # Creating 2 stacked LSTM cells used to build BRNN
        x = keras.layers.Bidirectional(keras.layers.LSTM(512,return_sequences=True,dropout=0.25))(x)
        x = keras.layers.Bidirectional(keras.layers.LSTM(512,return_sequences=True,dropout=0.25))(x)

        # Output Layer
        x = keras.layers.Dense(len(self.charList)+1,activation="softmax",name="dense2")(x)

        labels = keras.layers.Input(name="label",shape=(None,))

        # Adding CTC Layer for calculating CTC loss at each step
        output = CTCLayer(name="ctc_loss")(labels,x)

        # Define the model
        model = keras.models.Model(inputs=[inputImgs,labels],
                                    outputs=output,
                                    name="HWR_v1")

        # Adding optimizer and compiling the model
        model.compile(optimizer='adam')

        return model
