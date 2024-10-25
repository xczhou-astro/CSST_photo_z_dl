import tensorflow as tf
from tensorflow.keras import layers

def loss_func(y_true, y_pred):
    return tf.keras.losses.mean_absolute_error(y_pred=y_pred, 
                                               y_true=y_true)


class DenseLayer(tf.keras.layers.Layer):

    def __init__(self, n_neurons, norm=True):
        super().__init__()
        self.dense = layers.Dense(n_neurons)
        self.batchnorm = layers.BatchNormalization()
        self.activation = layers.ReLU()
        self.norm = norm

    def call(self, inputs):
        x = self.dense(inputs)
        if self.norm is True:
            x = self.batchnorm(x)
        x = self.activation(x)
        return x


    
def ConvBlock(layer_in, n_kernels, kernel_size, strides, padding='same'):
    x = layers.Conv2D(n_kernels, kernel_size, strides=strides,
                      padding=padding)(layer_in)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    return x


def MLP(train_size=None):
    inputs = tf.keras.Input(shape=(20,))
    dense = DenseLayer(40, norm=False)(inputs)
    for _ in range(5):
        dense = DenseLayer(40)(dense)
    outputs = layers.Dense(1)(dense)

    model = tf.keras.Model(inputs, outputs)
    return model

def InceptionModule(layer_in, f_1x1, f_3x3_r, f_3x3, f_5x5_r, f_5x5, f_pp):

    br1 = ConvBlock(layer_in, f_1x1, kernel_size=1, strides=2, padding='same')

    br2 = ConvBlock(layer_in, f_3x3_r, kernel_size=1,
                    strides=1, padding='same')
    br2 = ConvBlock(br2, f_3x3, kernel_size=3, strides=2, padding='same')

    br3 = ConvBlock(layer_in, f_5x5_r, kernel_size=1,
                    strides=1, padding='same')
    br3 = ConvBlock(br3, f_5x5, kernel_size=5, strides=2, padding='same')

    br4 = layers.MaxPooling2D(3, strides=1, padding='same')(layer_in)
    br4 = ConvBlock(br4, f_pp, kernel_size=1, strides=2, padding='same')

    layer_out = layers.concatenate([br1, br2, br3, br4], axis=-1)

    return layer_out


def inception(train_size=None):
    img_inputs = tf.keras.Input(shape=(32, 32, 7))
    conv = ConvBlock(img_inputs, 32, kernel_size=3,
                     strides=2, padding='same')  # 32

    inc1 = InceptionModule(conv, 16, 16, 32, 8, 16, 8)
    inc1 = InceptionModule(inc1, 16, 16, 32, 8, 16, 8)

    inc2 = InceptionModule(inc1, 16, 16, 32, 8, 16, 8)


    pooling = layers.GlobalAveragePooling2D()(inc2)


    dense = layers.Dense(40)(pooling)
    dense = layers.ReLU()(dense)
    outputs = layers.Dense(1)(dense)

    model = tf.keras.models.Model(img_inputs, outputs)

    return model

def hybrid_network(train_size=None):

    img_inputs = tf.keras.Input(shape=(32, 32, 7))
    conv = ConvBlock(32, kernel_size=3,
                     strides=2, padding='same')(img_inputs)  # 32

    inc1 = InceptionModule(conv, 16, 16, 32, 8, 16, 8)
    inc1 = InceptionModule(inc1, 16, 16, 32, 8, 16, 8)
    inc2 = InceptionModule(inc1, 16, 16, 32, 8, 16, 8)

    pooling = layers.GlobalAveragePooling2D()(inc2)

    dense = layers.Dense(40)(pooling)
    feature_cnn = layers.ReLU()(dense)

    mlp_inputs = tf.keras.Input(shape=(20, ), name='mlp_inputs')
    dense = DenseLayer(40, norm=False)(mlp_inputs)
    for _ in range(5):
        dense = DenseLayer(40)(dense)
    concat = tf.concat([dense, feature_cnn], axis=-1)

    dense = DenseLayer(80)(concat)
    for _ in range(5):
        dense = DenseLayer(80)(dense)

    outputs = layers.Dense(1)(dense)

    model = tf.keras.models.Model([img_inputs, mlp_inputs], outputs)

    return model

def hybrid_transfer_network(cnn_transfer, mlp_transfer, train_size=None):
    cnn_transfer.layers[0]._name = 'img_input'
    img_inputs = cnn_transfer.input
    img_output = cnn_transfer.layers[-2].output

    for layer in cnn_transfer.layers[:-3]:
        layer.trainable = False

    mlp_transfer.layers[0]._name = 'flux_input'
    mlp_inputs = mlp_transfer.input
    mlp_output = mlp_transfer.layers[-2].output

    for layer in mlp_transfer.layers[:-4]:  # last dense layer trainable
        layer.trainable = False

    concat = layers.concatenate([mlp_output, img_output], axis=-1)

    dense = DenseLayer(80)(concat)
    for _ in range(5):
        dense = DenseLayer(80)(dense)

    outputs = layers.Dense(1)(dense)

    model = tf.keras.Model([img_inputs, mlp_inputs], outputs)

    return model