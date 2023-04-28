import tensorflow as tf

# TODO: under construction
class VisionTransformer(tf.keras.models.Model):
    """
    Image classifier based on the Vision Transformer (ViT) architecture.
    """
    def __init__(self, *args, **kwargs):
        super(VisionTransformer, self).__init__(*args, **kwargs)
        pass

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        pass