import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers

def loss_func(y, rv_y): return -rv_y.log_prob(y)


def myacc(y_true, y_pred):
    # y_pred is a distribution
    delta = tf.math.abs(y_pred - y_true) / (1 + y_true)
    return tf.reduce_mean(tf.cast(delta <= 0.15, tf.float32), axis=-1)


def DenseFlipoutLayer(layer_in, n_neurons, norm=True, act=True, train_size=30000):

    kl_divergence_fn = (lambda q, p, _: tfp.distributions.kl_divergence(q, p) /
                        tf.cast(train_size, dtype=tf.float32))

    x = tfp.layers.DenseFlipout(
        n_neurons,
        kernel_divergence_fn=kl_divergence_fn)(layer_in)
    if norm is True:
        x = layers.BatchNormalization()(x)
    if act is True:
        x = layers.ReLU()(x)
    return x


def ConvFlipoutBlock(layer_in, n_kernels, kernel_size, strides=2, padding='same', norm=True, train_size=30000):

    kl_divergence_fn = (lambda q, p, _: tfp.distributions.kl_divergence(q, p) /
                        tf.cast(train_size, dtype=tf.float32))

    x = tfp.layers.Convolution2DFlipout(
        n_kernels, kernel_size, strides=strides, padding=padding,
        kernel_divergence_fn=kl_divergence_fn
    )(layer_in)

    if norm is True:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x


def Conv2d(layer_in, n_kernels, kernel_size, strides=2, padding='same', norm=True):
    x = layers.Conv2D(n_kernels, kernel_size, strides=strides,
                      padding=padding)(layer_in)

    if norm is True:
        x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def InceptionModule(layer_in, f_1x1, f_3x3_r, f_3x3, f_5x5_r, f_5x5, f_pp, train_size=30000):

    br1 = Conv2d(layer_in, f_1x1, kernel_size=1,
                 strides=2, padding='same', norm=False)

    br2 = Conv2d(layer_in, f_3x3_r, kernel_size=1,
                 strides=1, padding='same', norm=False)
    br2 = ConvFlipoutBlock(br2, f_3x3, kernel_size=3,
                           strides=2, padding='same', norm=True, train_size=train_size)

    br3 = Conv2d(layer_in, f_5x5_r, kernel_size=1,
                 strides=1, padding='same', norm=False)
    br3 = ConvFlipoutBlock(br3, f_5x5, kernel_size=5,
                           strides=2, padding='same', norm=True, train_size=train_size)

    br4 = layers.MaxPooling2D(3, strides=1, padding='same')(layer_in)
    br4 = Conv2d(br4, f_pp, kernel_size=1,
                 strides=2, padding='same', norm=False)
    
    layer_out = layers.concatenate([br1, br2, br3, br4], axis=-1)

    return layer_out


def inception(train_size=30000):

    img_inputs = tf.keras.Input(shape=(32, 32, 7))
    conv = ConvFlipoutBlock(img_inputs, 32, kernel_size=3,
                            strides=2, padding='same', norm=False, train_size=train_size)

    inc1 = InceptionModule(conv, 16, 16, 32, 8, 16, 8, train_size=train_size)
    inc1 = InceptionModule(inc1, 16, 16, 32, 8, 16, 8, train_size=train_size)
    inc2 = InceptionModule(inc1, 16, 16, 32, 8, 16, 8, train_size=train_size)

    pooling = layers.GlobalAveragePooling2D()(inc2)

    dense = DenseFlipoutLayer(pooling, 40, norm=True,
                              act=True, train_size=train_size)

    params = DenseFlipoutLayer(
        dense, 2, norm=False, act=False, train_size=train_size)

    dist = tfp.layers.DistributionLambda(
        lambda t: tfp.distributions.Normal(loc=t[..., :1],
                                           scale=1e-3 + tf.math.softplus(0.01 * t[..., 1:])))(params)

    model = tf.keras.Model(img_inputs, dist)

    # model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
    #               loss=negloglik, metrics=[myacc])

    return model


def MLP(train_size=30000):

    inputs = tf.keras.Input(shape=(20,))
    dense = DenseFlipoutLayer(
        inputs, 40, norm=False, train_size=train_size)
    for _ in range(5):
        dense = DenseFlipoutLayer(dense,40, norm=True, train_size=train_size)

    params = DenseFlipoutLayer(
        dense, 2, norm=False, act=False, train_size=train_size)

    dist = tfp.layers.DistributionLambda(
        lambda t: tfp.distributions.Normal(loc=t[..., :1],
                                           scale=1e-3 + tf.math.softplus(0.01 * t[..., 1:])))(params)

    model = tf.keras.Model(inputs, dist)

    # model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
    #               loss=negloglik, metrics=[myacc])
    return model


def hybrid_network(train_size=30000):

    img_inputs = tf.keras.Input(shape=(32, 32, 7))
    conv = ConvFlipoutBlock(img_inputs, 32, kernel_size=3,
                            strides=2, padding='same', norm=False, train_size=train_size)

    inc1 = InceptionModule(conv, 16, 16, 32, 8, 16, 8, train_size=train_size)
    inc1 = InceptionModule(inc1, 16, 16, 32, 8, 16, 8, train_size=train_size)
    inc2 = InceptionModule(inc1, 16, 16, 32, 8, 16, 8, train_size=train_size)

    pooling = layers.GlobalAveragePooling2D()(inc2)

    feature_cnn = DenseFlipoutLayer(
        pooling, 40, norm=True, act=True, train_size=train_size)

    mlp_inputs = tf.keras.Input(shape=(20,), name='mlp_inputs')
    dense = DenseFlipoutLayer(
        mlp_inputs, 40, norm=False, train_size=train_size)
    for _ in range(5):
        dense = DenseFlipoutLayer(dense, 40, train_size=train_size)
    concat = tf.concat([dense, feature_cnn], axis=-1)

    dense = DenseFlipoutLayer(concat, 80, train_size=train_size)
    for _ in range(5):
        dense = DenseFlipoutLayer(dense, 80, train_size=train_size)

    params = DenseFlipoutLayer(
        dense, 2, norm=False, act=False, train_size=train_size)

    dist = tfp.layers.DistributionLambda(
        lambda t: tfp.distributions.Normal(loc=t[..., :1],
                                           scale=1e-3 + tf.math.softplus(0.01 * t[..., 1:])))(params)

    model = tf.keras.Model([img_inputs, mlp_inputs], dist)

    # model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
    #               loss=negloglik, metrics=[myacc])
    return model


def hybrid_transfer_network(cnn_transfer, mlp_transfer, train_size=30000):

    cnn_transfer.layers[0]._name = 'img_input'
    img_inputs = cnn_transfer.input
    img_output = cnn_transfer.layers[-3].output

    for layer in cnn_transfer.layers[:-4]:
        layer.trainable = False

    mlp_transfer.layers[0]._name = 'flux_input'
    mlp_inputs = mlp_transfer.input
    mlp_output = mlp_transfer.layers[-3].output

    for layer in mlp_transfer.layers[:-5]:
        layer.trainable = False

    concat = layers.concatenate([mlp_output, img_output], axis=-1)

    dense = DenseFlipoutLayer(concat, 80, train_size=train_size)
    for _ in range(5):
        dense = DenseFlipoutLayer(dense, 80, train_size=train_size)

    params = DenseFlipoutLayer(
        dense, 2, norm=False, act=False, train_size=train_size)

    dist = tfp.layers.DistributionLambda(
        lambda t: tfp.distributions.Normal(loc=t[..., :1],
                                           scale=1e-3 + tf.math.softplus(0.01 * t[..., 1:])))(params)

    model = tf.keras.Model([img_inputs, mlp_inputs], dist)

    # model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
    #               loss=negloglik, metrics=[myacc])

    return model