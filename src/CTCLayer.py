from tensorflow.keras import layers 
from tensorflow import keras
import tensorflow as tf
from tensorflow.python.keras.backend import dtype

class CTCLayer(layers.Layer):

    def __init__(self,name=None,**kwargs):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost
        # self.ignore_longer_outputs_than_inputs=True
        super(CTCLayer,self).__init__(**kwargs)

    def call(self,yTrue,yPred):
        batchLen = tf.cast(tf.shape(yTrue)[0],dtype='int64')
        inputLength = tf.cast(tf.shape(yPred)[1],dtype='int64')
        labelLength = tf.cast(tf.shape(yTrue)[1],dtype='int64')

        inputLength = inputLength * tf.ones(shape=(batchLen,1),dtype='int64')
        labelLength = labelLength * tf.ones(shape=(batchLen,1),dtype='int64')

        loss = self.loss_fn(yTrue,yPred,inputLength,labelLength)
        self.add_loss(loss)

        return yPred
