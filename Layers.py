import tensorflow as tf
import tensorflow.keras as kr
import HyperParameters as hp


class EqDense(kr.layers.Layer):
    def __init__(self, units, activation=kr.activations.linear, use_bias=True, lr_scale=1.0):
        super(EqDense, self).__init__()
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.lr_scale = lr_scale

    def build(self, input_shape):
        self.w = tf.Variable(tf.random.normal([input_shape[-1], self.units]) / self.lr_scale, name=self.name + '_w')
        self.he_std = tf.sqrt(2.0 / tf.cast(input_shape[-1], 'float32')) * self.lr_scale

        if self.use_bias:
            self.b = tf.Variable(tf.zeros([1, self.units]), name=self.name + '_b')

    def call(self, inputs, *args, **kwargs):
        feature_vector = tf.matmul(inputs, self.w) * self.he_std
        if self.use_bias:
            feature_vector = feature_vector + self.b

        return self.activation(feature_vector)


class Fir(kr.layers.Layer):
    def __init__(self, kernel, upscale=False, downscale=False):
        super(Fir, self).__init__()
        self.kernel = kernel
        self.upscale = upscale
        self.downscale = downscale

        assert (upscale and downscale) != True
        assert self.kernel.shape[0] == self.kernel.shape[1]

    def build(self, input_shape):
        if self.downscale:
            padding_0 = tf.maximum((self.kernel.shape[0] - 2) // 2, 0)
            padding_1 = tf.maximum(self.kernel.shape[0] - 2 - padding_0, 0)
            self.padding = [[0, 0], [0, 0], [padding_1, padding_0], [padding_1, padding_0]]
        else:
            padding_0 = (self.kernel.shape[0] - 1) // 2
            padding_1 = self.kernel.shape[0] - 1 - padding_0
            self.padding = [[0, 0], [0, 0], [padding_0, padding_1], [padding_0, padding_1]]

        self.kernel = tf.tile(self.kernel[:, :, tf.newaxis, tf.newaxis], [1, 1, input_shape[1], 1])
        if self.upscale:
            self.reshape_layer = kr.layers.Reshape([input_shape[1], input_shape[2] * 2, input_shape[3] * 2])
            self.kernel *= 4

    def call(self, inputs, *args, **kwargs):
        if self.upscale:
            feature_maps = tf.stack([tf.zeros_like(inputs), inputs], axis=3)
            feature_maps = tf.stack([tf.zeros_like(feature_maps), feature_maps], axis=5)
            feature_maps = self.reshape_layer(feature_maps)
            return tf.nn.depthwise_conv2d(input=feature_maps, filter=self.kernel, strides=[1, 1, 1, 1],
                                          padding=self.padding, data_format='NCHW')

        elif self.downscale:
            return tf.nn.depthwise_conv2d(input=inputs, filter=self.kernel, strides=[1, 1, 2, 2],
                                          padding=self.padding, data_format='NCHW')

        else:
            return tf.nn.depthwise_conv2d(input=inputs, filter=self.kernel, strides=[1, 1, 1, 1],
                                          padding=self.padding, data_format='NCHW')


def get_blur_kernel():
    kernel = tf.cast([1, 3, 3, 1], 'float32')
    kernel = tf.tensordot(kernel, kernel, axes=0)
    kernel = kernel / tf.reduce_sum(kernel)

    return kernel


class EqConv2D(kr.layers.Layer):
    def __init__(self, filters, kernel_size, activation=kr.activations.linear, use_bias=True):
        super(EqConv2D, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.use_bias = use_bias

    def build(self, input_shape):
        input_filters = input_shape[1]

        self.w = tf.Variable(tf.random.normal([self.kernel_size, self.kernel_size, input_filters, self.filters]),
                             name=self.name + '_w')
        self.he_std = tf.sqrt(2.0 / tf.cast(self.kernel_size * self.kernel_size * input_filters, 'float32'))
        padding_0 = (self.kernel_size - 1) // 2
        padding_1 = self.kernel_size - 1 - padding_0
        self.padding = [[0, 0], [0, 0], [padding_0, padding_1], [padding_0, padding_1]]

        if self.use_bias:
            self.b = tf.Variable(tf.zeros([1, self.filters, 1, 1]), name=self.name + '_b')

    def call(self, inputs, *args, **kwargs):
        feature_maps = inputs
        feature_maps = tf.nn.conv2d(feature_maps, self.w, strides=1, padding=self.padding, data_format='NCHW') * self.he_std

        if self.use_bias:
            feature_maps = feature_maps + self.b

        return self.activation(feature_maps)


class Decoder(kr.layers.Layer):
    def __init__(self):
        super(Decoder, self).__init__()

    def build(self, input_shape):
        latent_vector = kr.Input([hp.latent_vector_dim])

        feature_vector = EqDense(units=1024 * 4 * 4, activation=tf.nn.leaky_relu)(latent_vector)
        feature_maps = kr.layers.Reshape([1024, 4, 4])(feature_vector)

        for filters in [512, 256, 128]:
            feature_maps = Fir(kernel=get_blur_kernel(), upscale=True)(feature_maps)
            feature_maps = EqConv2D(filters=filters, kernel_size=3, activation=tf.nn.leaky_relu)(feature_maps)
            feature_maps = EqConv2D(filters=filters, kernel_size=3, activation=tf.nn.leaky_relu)(feature_maps)
        fake_image = EqConv2D(filters=1, kernel_size=1)(feature_maps)
        fake_image = tf.transpose(fake_image, [0, 2, 3, 1])
        self.model = kr.Model(latent_vector, fake_image)

    def call(self, inputs, *args, **kwargs):
        return self.model(inputs)


class Encoder(kr.layers.Layer):
    def __init__(self):
        super(Encoder, self).__init__()

    def build(self, input_shape):
        input_image = kr.Input([hp.image_resolution, hp.image_resolution, 1])
        feature_maps = tf.transpose(input_image, [0, 3, 1, 2])

        feature_maps = EqConv2D(filters=128, kernel_size=1, activation=tf.nn.leaky_relu)(feature_maps)
        for filters in [128, 256, 512]:
            feature_maps = EqConv2D(filters=filters, kernel_size=3, activation=tf.nn.leaky_relu)(feature_maps)
            feature_maps = EqConv2D(filters=filters * 2, kernel_size=3, activation=tf.nn.leaky_relu)(feature_maps)
            feature_maps = Fir(kernel=get_blur_kernel(), downscale=True)(feature_maps)

        feature_vector = kr.layers.Flatten()(feature_maps)
        adv_value = tf.squeeze(EqDense(units=1)(feature_vector))
        latent_vector = EqDense(units=hp.latent_vector_dim)(feature_vector)

        self.model = kr.Model(input_image, [adv_value, latent_vector])

    def call(self, inputs, *args, **kwargs):
        return self.model(inputs)


class Svm(kr.layers.Layer):
    def __init__(self):
        super(Svm, self).__init__()

    def build(self, input_shape):
        latent_vector = kr.Input([hp.latent_vector_dim])
        predict_logit = EqDense(units=hp.class_size)(latent_vector)

        self.model = kr.Model(latent_vector, predict_logit)

    def call(self, inputs, *args, **kwargs):
        return self.model(inputs)