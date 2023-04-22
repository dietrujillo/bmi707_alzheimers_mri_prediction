import tensorflow as tf


class MRIEncoder(tf.keras.layers.Layer):
    def __init__(self):
        super(MRIEncoder, self).__init__()

    def call(self, inputs, **kwargs):
        pass


class PETEncoder(tf.keras.layers.Layer):
    def __init__(self):
        super(PETEncoder, self).__init__()

    def call(self, inputs, **kwargs):
        pass


class MLPClassifier(tf.keras.layers.Layer):
    def __init__(self):
        super(MLPClassifier, self).__init__()

    def call(self, inputs, **kwargs):
        pass

class AlzheimerClassifier(tf.keras.models.Model):

    def __init__(self):
        super(AlzheimerClassifier, self).__init__()

        self.pet_encoder = PETEncoder()
        self.mri_encoder = MRIEncoder()
        self.concatenate = tf.keras.layers.Concatenate()
        self.classifier = MLPClassifier()

    def call(self, inputs, **kwargs):
        pet_embedding = self.pet_encoder(inputs)
        mri_embedding = self.mri_encoder(inputs)

        out = self.concatenate(pet_embedding, mri_embedding)
        out = self.classifier(out)

        return out


