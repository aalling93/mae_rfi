import tensorflow as tf
from ..CONSTANTS import IMAGE_SIZE, RANDOM_CROP


def get_train_augmentation_model(random_crop: bool = RANDOM_CROP):
    model = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal"),
            ],
            name="train_data_augmentation",
        )

    return model


def get_test_augmentation_model(imsize: tuple = IMAGE_SIZE):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Resizing(imsize[0], imsize[1]),
        ],
        name="test_data_augmentation",
    )
    return model
