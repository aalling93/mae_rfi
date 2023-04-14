import tensorflow as tf


def get_train_augmentation_model():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
        ],
        name="train_data_augmentation",
    )
    return model


def get_test_augmentation_model(imsize: tuple = (340, 500, 2)):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Resizing(imsize[0], imsize[1]),
        ],
        name="test_data_augmentation",
    )
    return model
