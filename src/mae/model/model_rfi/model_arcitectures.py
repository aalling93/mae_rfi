import numpy as np
import tensorflow as tf

from .model_util import ssim_loss
from ..util import get_lr_metric


def modelPoolingDropout(img_size, latent_space_dim: int = 128):
    # Encoder
    latent_space_dim = latent_space_dim - latent_space_dim % 5
    x = tf.keras.layers.Input(shape=img_size, name="encoder_input")

    x = tf.keras.layers.Input(shape=img_size, name="encoder_input")

    encoder_conv_layer1 = tf.keras.layers.Conv2D(
        filters=img_size[-1],
        kernel_size=(3, 3),
        padding="same",
        kernel_regularizer=tf.keras.regularizers.L1L2(0.0001),
        bias_regularizer=tf.keras.regularizers.L1L2(0.0001),
        strides=1,
        name="encoder_conv_1",
    )(x)
    encoder_activ_layer1 = tf.keras.layers.LeakyReLU(name="encoder_leakyrelu_1")(
        encoder_conv_layer1
    )
    encoder_norm_layer1 = tf.keras.layers.BatchNormalization(name="encoder_norm_1")(
        encoder_activ_layer1
    )

    encoder_pool_layer1 = tf.keras.layers.MaxPooling2D(
        pool_size=(3, 3), padding="same", strides=(1, 1), name="encoder_maxpooling_1"
    )(encoder_norm_layer1)
    encoder_drop_layer1 = tf.keras.layers.Dropout(0.1, name="encoder_dropout_1")(
        encoder_pool_layer1
    )

    encoder_conv_layer2 = tf.keras.layers.Conv2D(
        filters=164,
        kernel_size=(3, 3),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.0001),
        bias_regularizer=tf.keras.regularizers.L1L2(0.0001),
        padding="same",
        strides=1,
        name="encoder_conv_2",
    )(encoder_drop_layer1)

    encoder_activ_layer2 = tf.keras.layers.LeakyReLU(name="encoder_activ_layer_2")(
        encoder_conv_layer2
    )

    encoder_norm_layer2 = tf.keras.layers.BatchNormalization(name="encoder_norm_2")(
        encoder_activ_layer2
    )

    encoder_pool_layer2 = tf.keras.layers.MaxPooling2D(
        pool_size=(3, 3), padding="same", strides=(2, 2), name="encoder_maxpooling_2"
    )(encoder_norm_layer2)

    encoder_drop_layer2 = tf.keras.layers.Dropout(0.1, name="encoder_dropout_2")(
        encoder_pool_layer2
    )

    encoder_conv_layer3 = tf.keras.layers.Conv2D(
        filters=164,
        kernel_size=(3, 3),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.0001),
        bias_regularizer=tf.keras.regularizers.L1L2(0.0001),
        padding="same",
        strides=2,
        name="encoder_conv_3",
    )(encoder_drop_layer2)

    encoder_activ_layer3 = tf.keras.layers.LeakyReLU(name="encoder_activ_layer_3")(
        encoder_conv_layer3
    )

    encoder_norm_layer3 = tf.keras.layers.BatchNormalization(name="encoder_norm_3")(
        encoder_activ_layer3
    )

    # encoder_pool_layer3 = tf.keras.layers.MaxPooling2D(
    #    pool_size=(3, 3), padding="same", strides=(2, 2), name="encoder_maxpooling_3"
    # )(encoder_norm_layer3)

    encoder_drop_layer3 = tf.keras.layers.Dropout(0.1, name="encoder_dropout_3")(
        encoder_norm_layer3
    )

    encoder_conv_layer4 = tf.keras.layers.Conv2D(
        filters=56,
        kernel_size=(3, 3),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.0001),
        bias_regularizer=tf.keras.regularizers.L1L2(0.0001),
        padding="same",
        strides=1,
        name="encoder_conv_4",
    )(encoder_drop_layer3)

    encoder_activ_layer4 = tf.keras.layers.LeakyReLU(name="encoder_activ_layer_4")(
        encoder_conv_layer4
    )

    encoder_norm_layer4 = tf.keras.layers.BatchNormalization(name="encoder_norm_4")(
        encoder_activ_layer4
    )

    encoder_drop_layer4 = tf.keras.layers.Dropout(0.1, name="encoder_dropout_4")(
        encoder_norm_layer4
    )

    encoder_conv_layer5 = tf.keras.layers.Conv2D(
        filters=28,
        kernel_size=(3, 3),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.0001),
        bias_regularizer=tf.keras.regularizers.L1L2(0.0001),
        padding="same",
        strides=1,
        name="encoder_conv_5",
    )(encoder_drop_layer4)

    encoder_activ_layer5 = tf.keras.layers.LeakyReLU(name="encoder_activ_layer_5")(
        encoder_conv_layer5
    )

    encoder_norm_layer5 = tf.keras.layers.BatchNormalization(name="encoder_norm_5")(
        encoder_activ_layer5
    )

    encoder_drop_layer5 = tf.keras.layers.Dropout(0.1, name="encoder_dropout_5")(
        encoder_norm_layer5
    )

    shape_before_flatten = tf.keras.backend.int_shape(encoder_drop_layer5)[1:]
    encoder_flatten = tf.keras.layers.Flatten()(encoder_drop_layer5)
    encoder_output = tf.keras.layers.Dense(
        units=latent_space_dim, name="encoder_output"
    )(encoder_flatten)

    encoder_output_drop = tf.keras.layers.Dropout(0.1, name="encoder_output_drop")(
        encoder_output
    )

    encoder_output_drop = tf.keras.layers.Reshape((5, -1))(encoder_output_drop)

    encoder = tf.keras.models.Model(x, encoder_output_drop, name="encoder_model")

    decoder_input = tf.keras.layers.Input(
        shape=(5, int(latent_space_dim / 5)), name="decoder_input"
    )
    decoder_flatten1 = tf.keras.layers.Flatten()(decoder_input)
    decoder_dense_layer1 = tf.keras.layers.Dense(
        units=np.prod(shape_before_flatten), name="decoder_dense_1"
    )(decoder_flatten1)
    decoder_reshape = tf.keras.layers.Reshape(target_shape=shape_before_flatten)(
        decoder_dense_layer1
    )

    decoder_conv_tran_layer1 = tf.keras.layers.Conv2DTranspose(
        filters=28,
        kernel_size=(3, 3),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.0001),
        bias_regularizer=tf.keras.regularizers.L1L2(0.0001),
        padding="same",
        strides=1,
        name="decoder_conv_tran_1",
    )(decoder_reshape)

    decoder_activ_layer1 = tf.keras.layers.LeakyReLU(name="decoder_leakyrelu_1")(
        decoder_conv_tran_layer1
    )

    decoder_norm_layer1 = tf.keras.layers.BatchNormalization(name="decoder_norm_1")(
        decoder_activ_layer1
    )

    decoder_conv_tran_layer2 = tf.keras.layers.Conv2DTranspose(
        filters=56,
        kernel_size=(3, 3),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.0001),
        bias_regularizer=tf.keras.regularizers.L1L2(0.0001),
        padding="same",
        strides=2,
        name="decoder_conv_tran_2",
    )(decoder_norm_layer1)

    decoder_activ_layer2 = tf.keras.layers.LeakyReLU(name="decoder_leakyrelu_2")(
        decoder_conv_tran_layer2
    )

    decoder_norm_layer2 = tf.keras.layers.BatchNormalization(name="decoder_norm_2")(
        decoder_activ_layer2
    )

    decoder_conv_tran_layer3 = tf.keras.layers.Conv2DTranspose(
        filters=164,
        kernel_size=(3, 3),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.0001),
        bias_regularizer=tf.keras.regularizers.L1L2(0.0001),
        padding="same",
        strides=2,
        name="decoder_conv_tran_3",
    )(decoder_norm_layer2)

    decoder_activ_layer3 = tf.keras.layers.LeakyReLU(name="decoder_leakyrelu_3")(
        decoder_conv_tran_layer3
    )

    decoder_norm_layer3 = tf.keras.layers.BatchNormalization(name="decoder_norm_3")(
        decoder_activ_layer3
    )

    decoder_conv_tran_layer4 = tf.keras.layers.Conv2DTranspose(
        filters=164,
        kernel_size=(3, 3),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.0001),
        bias_regularizer=tf.keras.regularizers.L1L2(0.0001),
        padding="same",
        strides=1,
        name="decoder_conv_tran_4",
    )(decoder_norm_layer3)

    decoder_activ_layer4 = tf.keras.layers.LeakyReLU(name="decoder_leakyrelu_4")(
        decoder_conv_tran_layer4
    )

    decoder_norm_layer4 = tf.keras.layers.BatchNormalization(name="decoder_norm_4")(
        decoder_activ_layer4
    )

    decoder_conv_tran_layer6 = tf.keras.layers.Conv2DTranspose(
        filters=img_size[-1],
        kernel_size=(3, 3),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.0001),
        bias_regularizer=tf.keras.regularizers.L1L2(0.0001),
        padding="same",
        strides=1,
        name="decoder_conv_tran_6",
    )(decoder_norm_layer4)

    decoder_output = tf.keras.layers.Softmax(name="decoder_leakyrelu_6")(
        decoder_conv_tran_layer6
    )

    decoder = tf.keras.models.Model(decoder_input, decoder_output, name="decoder_model")

    ae_input = tf.keras.layers.Input(shape=img_size, name="AE_input")
    ae_encoder_output = encoder(ae_input)
    ae_decoder_output = decoder(ae_encoder_output)

    ae = tf.keras.models.Model(ae_input, ae_decoder_output, name="AE_rfi")

    optimizer = optimizer = tf.keras.optimizers.Adam(
        clipnorm=0.8, clipvalue=0.8, learning_rate=0.001
    )

    lr_metric = get_lr_metric(optimizer)
    ae.compile(optimizer, loss=ssim_loss, metrics=["mse", lr_metric])

    return ae, encoder, decoder
