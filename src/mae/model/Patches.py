"""This script defines a custom Keras layer for extracting patches from images.

The layer creates patches from the input images and returns the patches reshaped
into (batch, num_patches, patch_area) for use in neural network training.

Classes:
    Patches: A custom Keras layer for extracting patches from images.

Methods:
    call: Extracts patches from the input images and returns the patches reshaped.
    show_patched_image: A utility function that helps visualize one image and its
        patches side by side.
    reconstruct_from_patch: A utility function that takes patches from a single
        image and reconstructs it back into the image.

Example usage:
    # Create a Patches layer with a patch size of (16, 16).
    patches_layer = Patches(patch_size=(16, 16))

    # Pass a batch of images through the layer to extract patches.
    images = tf.random.normal((32, 256, 256, 3))
    patches = patches_layer(images)

    # Visualize a random image and its patches.
    patches_layer.show_patched_image(images, patches)

    # Reconstruct the original image from a single patch.
    patch = patches[0, 0]
    reconstructed = patches_layer.reconstruct_from_patch(patch)

Author: Original OpenAI (modified by [Kristian Soerensen])
Date: April 2023
"""


from ..CONSTANTS import *
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class Patches(tf.keras.layers.Layer):
    """A custom Keras layer for extracting patches from images."""

    def __init__(self, patch_size=PATCH_SIZE, **kwargs):
        """Initializes the Patches layer.

        Args:
            patch_size: A tuple representing the size of the patches to extract.
            **kwargs: Additional keyword arguments passed to the base class.
        """
        super().__init__(**kwargs)
        self.patch_size = patch_size
        # Assuming the image has three channels each patch would be
        # of size (patch_size, patch_size, pol).
        self.resize = tf.keras.layers.Reshape(
            (-1, self.patch_size[0] * self.patch_size[1] * IMAGE_SIZE[2])
        )

    def call(self, images):
        """Extracts patches from the input images and returns the patches reshaped.

        Args:
            images: A tensor representing a batch of input images.

        Returns:
            A tensor of shape (batch_size, num_patches, patch_area), where
            batch_size is the number of images in the batch, num_patches is
            the number of patches extracted from each image, and patch_area
            is the area of each patch.
        """
        # Create patches from the input images
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size[0], self.patch_size[1], 1],
            strides=[1, self.patch_size[0], self.patch_size[1], 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )

        # Reshape the patches to (batch, num_patches, patch_area) and return it.
        patches = self.resize(patches)
        return patches

    def show_patched_image(self, images, patches):
        """A utility function that helps visualize one image and its patches side by side.

        Args:
            images: A tensor representing a batch of input images.
            patches: A tensor representing the patches extracted from the input images.

        Returns:
            The index of the image chosen to validate it outside the method.
        """
        # This is a utility function which accepts a batch of images and its
        # corresponding patches and help visualize one image and its patches
        # side by side.
        idx = np.random.choice(patches.shape[0])
        print(f"Index selected: {idx}.")

        plt.figure(figsize=(4, 4))
        plt.imshow(images[idx][:, :, 1])
        plt.axis("off")
        plt.show()

        n = int(np.sqrt(patches.shape[1]))
        plt.figure(figsize=(4, 4))
        for i, patch in enumerate(patches[idx]):
            ax = plt.subplot(n, n, i + 1)
            patch_img = tf.reshape(
                patch, (self.patch_size[0], self.patch_size[1], IMAGE_SIZE[2])
            )
            plt.imshow(patch_img[:, :, 1])
            plt.axis("off")
        plt.show()

        # Return the index chosen to validate it outside the method.
        return idx

    # taken from https://stackoverflow.com/a/58082878/10319735
    def reconstruct_from_patch(self, patch):
        # This utility function takes patches from a *single* image and
        # reconstructs it back into the image. This is useful for the train
        # monitor callback.
        num_patches = patch.shape[0]
        n = int(np.sqrt(num_patches))
        patch = tf.reshape(
            patch, (num_patches, self.patch_size[0], self.patch_size[1], IMAGE_SIZE[2])
        )
        rows = tf.split(patch, n, axis=0)
        rows = [tf.concat(tf.unstack(x), axis=1) for x in rows]
        reconstructed = tf.concat(rows, axis=0)
        return reconstructed
