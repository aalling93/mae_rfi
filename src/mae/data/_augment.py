"""
This script defines two functions that return TensorFlow Keras models for augmenting training and test data.

get_train_augmentation_model takes an optional boolean argument random_crop that determines whether to apply random cropping to the input data. It returns a sequential model that applies horizontal flipping and, if random_crop is True, random cropping to the input data.

get_test_augmentation_model takes an optional tuple argument imsize that specifies the desired size of the output image. It returns a sequential model that resizes the input image to the desired size.

Examples:

train_aug_model = get_train_augmentation_model(random_crop=True)
test_aug_model = get_test_augmentation_model(imsize=(256, 256))


Inputs and Outputs:
get_train_augmentation_model takes a boolean argument and returns a TensorFlow Keras sequential model.

get_test_augmentation_model takes a tuple argument and returns a TensorFlow Keras sequential model.

"""
import tensorflow as tf
from ..CONSTANTS import IMAGE_SIZE, RANDOM_CROP


def get_train_augmentation_model(random_crop: bool = RANDOM_CROP):

    """Returns a TensorFlow Keras model for augmenting training data.
    
    Args:
        random_crop (bool, optional): Whether to apply random cropping. Defaults to RANDOM_CROP constant.
    
    Returns:
        tf.keras.Sequential: A sequential model that applies data augmentation techniques for training data.
    """

    model = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal"),
            ],
            name="train_data_augmentation",
        )

    return model


def get_test_augmentation_model(imsize: tuple = IMAGE_SIZE):
    """Returns a TensorFlow Keras model for augmenting test data.
    
    Args:
        imsize (tuple, optional): The desired size of the image. Defaults to IMAGE_SIZE constant.
    
    Returns:
        tf.keras.Sequential: A sequential model that applies data augmentation techniques for test data.
    """
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Resizing(imsize[0], imsize[1]),
        ],
        name="test_data_augmentation",
    )
    return model
