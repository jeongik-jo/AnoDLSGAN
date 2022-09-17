import tensorflow as tf
import tensorflow.keras as kr
import Layers
import os
import HyperParameters as hp
import numpy as np


class Decoder(object):
    def build_model(self):
        latent_vector = kr.Input([hp.latent_vector_dim])
        fake_image = Layers.Decoder()(latent_vector)
        return kr.Model(latent_vector, fake_image)

    def __init__(self):
        self.model = self.build_model()
        self.optimizer = kr.optimizers.Adam(learning_rate=hp.learning_rate, beta_1=0.0, beta_2=0.99)
        self.latent_var_trace = tf.Variable(tf.ones([hp.latent_vector_dim]))

    def save(self):
        if not os.path.exists('models'):
            os.makedirs('models')
        self.model.save_weights('models/decoder.h5')
        np.save('models/latent_var_trace.npy', self.latent_var_trace)

    def load(self):
        self.model.load_weights('models/decoder.h5')
        self.latent_var_trace = np.load('models/latent_var_trace.npy')


class Encoder(object):
    def build_model(self):
        input_image = kr.Input([hp.image_resolution, hp.image_resolution, 1])
        adv_value, feature_vector = Layers.Encoder()(input_image)
        return kr.Model(input_image, [adv_value, feature_vector])

    def __init__(self):
        self.model = self.build_model()
        self.optimizer = kr.optimizers.Adam(learning_rate=hp.learning_rate, beta_1=0.0, beta_2=0.99)

    def save(self):
        if not os.path.exists('models'):
            os.makedirs('models')
        self.model.save_weights('models/encoder.h5')

    def load(self):
        self.model.load_weights('models/encoder.h5')


class Svm(object):
    def build_model(self):
        input_image = kr.Input([hp.latent_vector_dim])
        predict_logit = Layers.Svm()(input_image)
        return kr.Model(input_image, predict_logit)

    def __init__(self):
        self.model = self.build_model()
        self.optimizer = kr.optimizers.Adam(learning_rate=hp.learning_rate, beta_1=0.0, beta_2=0.99)

    def save(self):
        if not os.path.exists('models'):
            os.makedirs('models')
        self.model.save_weights('models/svm.h5')

    def load(self):
        self.model.load_weights('models/svm.h5')
