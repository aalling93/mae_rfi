"""
This module defines the MaskedAutoencoder class which is used for training and testing a Masked Autoencoder Model.

Classes:

MaskedAutoencoder: A class used for training and testing a Masked Autoencoder model.
Functions:

None
"""

import tensorflow as tf
from ..CONSTANTS import *
from .modules import *


class MaskedAutoencoder(tf.keras.Model):
    """
    A class used for training and testing a Masked Autoencoder model.
    Args:
    - train_augmentation_model (tf.keras.Model): The model used for data augmentation during training.
    - test_augmentation_model (tf.keras.Model): The model used for data augmentation during testing.
    - patch_layer (tf.keras.layers.Layer): The layer used for patching the images.
    - patch_encoder (tf.keras.Model): The model used for encoding the patches.
    - encoder (tf.keras.Model): The model used for encoding the input.
    - decoder (tf.keras.Model): The model used for decoding the encoded input.

    Attributes:
    - train_augmentation_model (tf.keras.Model): The model used for data augmentation during training.
    - test_augmentation_model (tf.keras.Model): The model used for data augmentation during testing.
    - patch_layer (tf.keras.layers.Layer): The layer used for patching the images.
    - patch_encoder (tf.keras.Model): The model used for encoding the patches.
    - encoder (tf.keras.Model): The model used for encoding the input.
    - decoder (tf.keras.Model): The model used for decoding the encoded input.

    Methods:
    - calculate_loss(images, test=False): Calculates the loss of the model.
    - train_step(images): Performs one training step.
    - test_step(images): Performs one testing step.
    """

    def __init__(
        self,
        train_augmentation_model,
        test_augmentation_model,
        patch_layer,
        patch_encoder,
        encoder,
        decoder,
        **kwargs,
    ):
        """
        Initializes the MaskedAutoencoder class.

        Args:
        - train_augmentation_model (tf.keras.Model): The model used for data augmentation during training.
        - test_augmentation_model (tf.keras.Model): The model used for data augmentation during testing.
        - patch_layer (tf.keras.layers.Layer): The layer used for patching the images.
        - patch_encoder (tf.keras.Model): The model used for encoding the patches.
        - encoder (tf.keras.Model): The model used for encoding the input.
        - decoder (tf.keras.Model): The model used for decoding the encoded input.
        - **kwargs: Additional keyword arguments to be passed to the super class.
        """

        super().__init__(**kwargs)
        self.train_augmentation_model = train_augmentation_model
        self.test_augmentation_model = test_augmentation_model
        self.patch_layer = patch_layer
        self.patch_encoder = patch_encoder
        self.encoder = encoder
        self.decoder = decoder

    def calculate_loss(self, images, test=False):
        """
        Calculates the loss for the masked autoencoder model.

        Args:
            images (tf.Tensor): Input tensor of shape (batch_size, height, width, channels).
            test (bool, optional): Flag indicating if the function is called during testing. Defaults to False.

        Returns:
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor]: A tuple of three tensors:
                - total_loss: Scalar tensor representing the total loss of the model.
                - loss_patch: Tensor of shape (batch_size, num_masked_patches, patch_height, patch_width, channels) representing the patches of the input images that were masked during training.
                - loss_output: Tensor of shape (batch_size, num_masked_patches, patch_height, patch_width, channels) representing the output patches of the decoder network corresponding to the masked patches in the input images.

        Raises:
            ValueError: If images is not a 4D tensor or if test is not a boolean value.

        """
        # Augment the input images.
        if test:
            augmented_images = self.test_augmentation_model(images)
        else:
            augmented_images = self.train_augmentation_model(images)

        # Patch the augmented images.
        patches = self.patch_layer(augmented_images)

        # Encode the patches.
        (
            unmasked_embeddings,
            masked_embeddings,
            unmasked_positions,
            mask_indices,
            unmask_indices,
        ) = self.patch_encoder(patches)

        # Pass the unmaksed patche to the encoder.
        encoder_outputs = self.encoder(unmasked_embeddings)

        # Create the decoder inputs.
        encoder_outputs = encoder_outputs + unmasked_positions
        decoder_inputs = tf.concat([encoder_outputs, masked_embeddings], axis=1)

        # Decode the inputs.
        decoder_outputs = self.decoder(decoder_inputs)
        decoder_patches = self.patch_layer(decoder_outputs)

        # Compute the patch loss and output loss.
        loss_patch = tf.gather(patches, mask_indices, axis=1, batch_dims=1)
        loss_output = tf.gather(decoder_patches, mask_indices, axis=1, batch_dims=1)

        # Compute the total loss.
        total_loss = self.compiled_loss(loss_patch, loss_output)

        return total_loss, loss_patch, loss_output

    def train_step(self, images):
        """
        Performs a single training step on a batch of input images.

        Args:
            images: A batch of input images in the shape of (batch_size, image_height, image_width, num_channels).

        Returns:
            A dictionary containing the metrics computed during the training step.
        """
        with tf.GradientTape() as tape:
            total_loss, loss_patch, loss_output = self.calculate_loss(images)

        # Apply gradients.
        train_vars = [
            self.train_augmentation_model.trainable_variables,
            self.patch_layer.trainable_variables,
            self.patch_encoder.trainable_variables,
            self.encoder.trainable_variables,
            self.decoder.trainable_variables,
        ]
        grads = tape.gradient(total_loss, train_vars)
        tv_list = []
        for (grad, var) in zip(grads, train_vars):
            for g, v in zip(grad, var):
                tv_list.append((g, v))
        self.optimizer.apply_gradients(tv_list)

        # Report progress.
        self.compiled_metrics.update_state(loss_patch, loss_output)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, images):
        total_loss, loss_patch, loss_output = self.calculate_loss(images, test=True)

        # Update the trackers.
        self.compiled_metrics.update_state(loss_patch, loss_output)
        return {m.name: m.result() for m in self.metrics}