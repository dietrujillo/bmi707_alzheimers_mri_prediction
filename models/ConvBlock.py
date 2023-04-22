import tensorflow as tf

class ConvBlock(tf.keras.layers.Layer):
    """
    Convolutional layer, with batch normalization and an activation function.
    """

    def __init__(self, filters: int, kernel_size: tuple[int, int, int],
                 use_batch_norm: bool = True,
                 pool_size: tuple[int, int, int] = None,
                 activation: tf.keras.layers.Activation | str = tf.keras.activations.relu,
                 name: str = "", **kwargs):
        super(ConvBlock, self).__init__(name=name, **kwargs)

        self.conv = tf.keras.layers.Conv3D(filters, kernel_size, padding="same", name=name + "_conv")
        self.use_batch_norm = use_batch_norm
        self.pool_size = pool_size

        if self.use_batch_norm:
            self.bn = tf.keras.layers.BatchNormalization(name=name + "_bn")

        try:
            self.activation = activation(name=name + "_act")
        except TypeError:
            self.activation = tf.keras.layers.Activation(activation, name=name + "_act")

        if self.pool_size is not None:
            self.pooling = tf.keras.layers.MaxPool3D(pool_size=pool_size, name=name + "_pool")

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        out = self.conv(inputs)
        if self.use_batch_norm:
            out = self.bn(out)

        out = self.activation(out)
        if self.pool_size is not None:
            out = self.pooling(out)

        return out