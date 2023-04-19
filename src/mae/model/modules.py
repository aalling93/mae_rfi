import tensorflow as tf
from ..CONSTANTS import *


def mlp(x, dropout_rate, hidden_units):
    for units in hidden_units:
        x = tf.keras.layers.Dense(units, activation=tf.nn.gelu)(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    return x

def create_encoder(
    num_heads=ENC_NUM_HEADS,
    num_layers=ENC_LAYERS,
    enc_transformer_units=ENC_TRANSFORMER_UNITS,
    epsilon=LAYER_NORM_EPS,
    enc_projection_dim=ENC_PROJECTION_DIM,
    dropout=DROPOUT_RATE,
):
    inputs = tf.keras.layers.Input((None, enc_projection_dim))
    x = inputs

    for _ in range(num_layers):
        # Layer normalization 1.
        x1 = tf.keras.layers.LayerNormalization(epsilon=epsilon)(x)
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=enc_projection_dim, dropout=dropout
        )(x1, x1)
        x2 = tf.keras.layers.Add()([attention_output, x])
        x3 = tf.keras.layers.LayerNormalization(epsilon=epsilon)(x2)
        x3 = mlp(x3, hidden_units=enc_transformer_units, dropout_rate=dropout)
        x = tf.keras.layers.Add()([x3, x2])

    outputs = tf.keras.layers.LayerNormalization(epsilon=epsilon)(x)
    
    return tf.keras.Model(inputs, outputs, name="mae_encoder")



def create_decoder(
    num_layers=DEC_LAYERS,
    num_heads=DEC_NUM_HEADS,
    image_size=IMAGE_SIZE,
    dropout=DROPOUT_RATE,
    num_patches = NUM_PATCHES,
    enc_projection_dim=ENC_PROJECTION_DIM,
    dec_projection_dim = DEC_PROJECTION_DIM,
    epsilon=LAYER_NORM_EPS,
    dec_transformer_units = DEC_TRANSFORMER_UNITS
):
    inputs = tf.keras.layers.Input((num_patches, enc_projection_dim))
    x = tf.keras.layers.Dense(dec_projection_dim)(inputs)

    for _ in range(num_layers):
        x1 = tf.keras.layers.LayerNormalization(epsilon=epsilon)(x)
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=dec_projection_dim, dropout=dropout
        )(x1, x1)
        x2 = tf.keras.layers.Add()([attention_output, x])
        x3 = tf.keras.layers.LayerNormalization(epsilon=epsilon)(x2)
        x3 = mlp(x3, hidden_units=dec_transformer_units, dropout_rate=dropout)
        x = tf.keras.layers.Add()([x3, x2])
    x = tf.keras.layers.LayerNormalization(epsilon=epsilon)(x)
    x = tf.keras.layers.Flatten()(x)
    
    if DECODER_ARCITECTURE_FULL == True:
        x = tf.keras.layers.Dense(
                100)(x)
        x = tf.keras.layers.Dense(
                units=image_size[0] * image_size[1] * image_size[2], activation="sigmoid"
                )(x)
        outputs = tf.keras.layers.Reshape((image_size[0], image_size[1],image_size[2]))(x)

    else:
        x = tf.keras.layers.Dense(
            units=PATCH_SIZE[0] * PATCH_SIZE[1] * image_size[2], activation="sigmoid"
        )(x)
        x = tf.keras.layers.Reshape((PATCH_SIZE[0], PATCH_SIZE[1], IMAGE_SIZE[2]))(x)
        outputs = tf.keras.layers.UpSampling2D(size=(10, 10))(x)
    return tf.keras.Model(inputs, outputs, name="mae_decoder")
