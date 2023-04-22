import tensorflow as tf

from ConvBlock import ConvBlock

class InceptionBlock(tf.keras.layers.Layer):
    """
    Inception module.
    """
    def __init__(self,
                 filters_1x1: int,
                 filters_3x3: int,
                 filters_5x5: int,
                 activation: tf.keras.layers.Activation | str = tf.keras.layers.ReLU,
                 name: str = "", **kwargs):
        super(InceptionBlock, self).__init__(name=name, **kwargs)

        self.conv1x1 = ConvBlock(filters=filters_1x1, kernel_size=(1, 1, 1),
                                 use_batch_norm=True, activation=activation, name=f"{name}_conv1x1")

        self.conv3x3_1 = ConvBlock(filters=filters_1x1, kernel_size=(1, 1, 1),
                                   use_batch_norm=True, activation=activation, name=f"{name}_conv3x3_1")
        self.conv3x3_2 = ConvBlock(filters=filters_3x3, kernel_size=(3, 3, 3),
                                   use_batch_norm=True, activation=activation, name=f"{name}_conv3x3_2")

        self.conv5x5_1 = ConvBlock(filters=filters_1x1, kernel_size=(1, 1, 1),
                                   use_batch_norm=True, activation=activation, name=f"{name}_conv5x5_1")
        self.conv5x5_2 = ConvBlock(filters=filters_5x5, kernel_size=(5, 5, 5),
                                   use_batch_norm=True, activation=activation, name=f"{name}_conv5x5_2")

        self.pooling_1 = tf.keras.layers.MaxPool3D(pool_size=(3, 3, 3), padding="same", strides=(1, 1, 1), name=f"{name}_pooling_1")
        self.pooling_2 = ConvBlock(filters=filters_1x1, kernel_size=(1, 1, 1),
                                   use_batch_norm=True, activation=activation, name=f"{name}_pooling_2")

        self.concat = tf.keras.layers.Concatenate(axis=-1, name=f"{name}_concat")

class InceptionClassifier(tf.keras.models.Model):

    def __init__(self):
        super(InceptionClassifier, self).__init__()

    def call(self, inputs: tf.Tensor, **kwargs):
        pass
