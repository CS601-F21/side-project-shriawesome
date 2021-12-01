from tensorflow.keras import layers 
from tensorflow import keras
import tensorflow as tf
from tensorflow.python.keras.backend import dtype

"""
Connectionist Temporal Classification loss for training the model
"""
class CTCLayer(layers.Layer):

    def __init__(self,name=None,**kwargs):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost
        # self.ignore_longer_outputs_than_inputs=True
        super(CTCLayer,self).__init__(**kwargs)

    """
    Ref : https://keras.io/examples/vision/handwriting_recognition/
    """
    def call(self, y_true, y_pred):
        batchLen = tf.cast(tf.shape(y_true)[0], dtype="int64")
        inputLength = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        labelLength = tf.cast(tf.shape(y_true)[1], dtype="int64")

        inputLength = inputLength * tf.ones(shape=(batchLen, 1), dtype="int64")
        labelLength = labelLength * tf.ones(shape=(batchLen, 1), dtype="int64")
        loss = self.loss_fn(y_true, y_pred, inputLength, labelLength)
        self.add_loss(loss)

        # At test time, just return the computed predictions.
        return y_pred
