"""
This script contains the PatchEncoder class that is used in image transformer models.
The PatchEncoder class encodes image patches into projection vectors with positional embeddings.
The class also allows for masking of some patches in the image, which can be useful in training the decoder.

Functions:
- init: Initializes the PatchEncoder object with the specified parameters.
- build: Builds the PatchEncoder object.
- call: Encodes the image patches into projection vectors with positional embeddings.
- get_random_indices: Gets random mask and unmask indices for the patches.
- generate_masked_image: Generates a masked image based on the unmask indices.



Examples:

# Example for get_random_indices function
patch_encoder = PatchEncoder()
mask_indices, unmask_indices = patch_encoder.get_random_indices(batch_size=2)
print(mask_indices)
# Output: tf.Tensor(
# [[ 6 12  2  1  8 11  0  9 13  4]
#  [ 0 10  1 11  6  5  8  9  7 13]], shape=(2, 10), dtype=int32)

print(unmask_indices)
# Output: tf.Tensor(
# [[ 3 14  5 15 10  7  4 16 17 18]
#  [ 2  3 12  4  9 14 15  7  5 18]], shape=(2, 10), dtype=int32)


# Example for generate_masked_image function
patch_encoder = PatchEncoder()
patches = np.random.rand(5, 16, 256)  # (batch_size, num_patches, patch_area)
(_, _, _, _, unmask_indices) = patch_encoder(patches)
new_patch, idx = patch_encoder.generate_masked_image(patches, unmask_indices)
print(new_patch.shape)
# Output: (16, 256)

print(idx)
# Output: 1


Author: OpenAI (modified by Kristian Soerensen)
        April 2023
"""


from ..CONSTANTS import *
import tensorflow as tf
import numpy as np


class PatchEncoder(tf.keras.layers.Layer): 
    """
    Encodes image patches into projection vectors with positional embeddings.
    """

    def __init__(
        self,
        patch_size=PATCH_SIZE,
        projection_dim=ENC_PROJECTION_DIM,
        mask_proportion=MASK_PROPORTION,
        downstream=False,
        **kwargs,
    ):
        """
        Initializes the PatchEncoder object.

        Args:
        - patch_size: Tuple representing the size of the patches.
        - projection_dim: The dimension of the projected vector for each patch.
        - mask_proportion: The proportion of patches to mask in the image.
        - downstream: Boolean indicating whether the layer is being used downstream.
        - kwargs: Keyword arguments.

        Returns:
        None.
        """
       
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.projection_dim = projection_dim
        self.mask_proportion = mask_proportion
        self.downstream = downstream

        # This is a trainable mask token initialized randomly from a normal
        # distribution.
        self.mask_token = tf.Variable(
            tf.random.normal(
                [1, self.patch_size[0] * self.patch_size[1] * IMAGE_SIZE[2]]
            ),
            trainable=True,
        )

    def build(self, input_shape):
        """
        Builds the PatchEncoder object.

        Args:
        - input_shape: Tuple representing the shape of the input.

        Returns:
        None.
        """
        (_, self.num_patches, self.patch_area) = input_shape
        # Create the projection layer for the patches.
        self.projection = tf.keras.layers.Dense(units=self.projection_dim)

        # Create the positional embedding layer.
        self.position_embedding = tf.keras.layers.Embedding(
            input_dim=self.num_patches, output_dim=self.projection_dim
        )

        # Number of patches that will be masked.
        self.num_mask = int(self.mask_proportion * self.num_patches)

    def call(self, patches):
        """
        Encodes the image patches into projection vectors with positional embeddings.

        Args:
        - patches: Tensor representing the image patches.

        Returns:
        If downstream is True:
        - patch_embeddings: Tensor representing the encoded patches.

        If downstream is False:
        - unmasked_embeddings: Tensor representing the unmasked patch embeddings. Input to the encoder.
        - masked_embeddings: Tensor representing the masked embeddings for the tokens. First part of input to the decoder.
        - unmasked_positions: Tensor representing the unmasked position embeddings. Added to the encoder outputs.
        - mask_indices: Tensor representing the indices that were masked.
        - unmask_indices: Tensor representing the indices that were unmasked.
        """
        # Get the positional embeddings.
        batch_size = tf.shape(patches)[0]
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        pos_embeddings = self.position_embedding(positions[tf.newaxis, ...])
        pos_embeddings = tf.tile(
            pos_embeddings, [batch_size, 1, 1]
        )  # (B, num_patches, projection_dim)
        # Embed the patches.
        patch_embeddings = (
            self.projection(patches) + pos_embeddings
        )  # (B, num_patches, projection_dim)

        if self.downstream:
            return patch_embeddings
        else:
            mask_indices, unmask_indices = self.get_random_indices(batch_size)
            # The encoder input is the unmasked patch embeddings. Here we gather
            # all the patches that should be unmasked.
            unmasked_embeddings = tf.gather(
                patch_embeddings, unmask_indices, axis=1, batch_dims=1
            )  # (B, unmask_numbers, projection_dim)

            # Get the unmasked and masked position embeddings. We will need them
            # for the decoder.
            unmasked_positions = tf.gather(
                pos_embeddings, unmask_indices, axis=1, batch_dims=1
            )  # (B, unmask_numbers, projection_dim)
            masked_positions = tf.gather(
                pos_embeddings, mask_indices, axis=1, batch_dims=1
            )  # (B, mask_numbers, projection_dim)

            # Repeat the mask token number of mask times.
            # Mask tokens replace the masks of the image.
            mask_tokens = tf.repeat(self.mask_token, repeats=self.num_mask, axis=0)
            mask_tokens = tf.repeat(
                mask_tokens[tf.newaxis, ...], repeats=batch_size, axis=0
            )

            # Get the masked embeddings for the tokens.
            masked_embeddings = self.projection(mask_tokens) + masked_positions
            return (
                unmasked_embeddings,  # Input to the encoder.
                masked_embeddings,  # First part of input to the decoder.
                unmasked_positions,  # Added to the encoder outputs.
                mask_indices,  # The indices that were masked.
                unmask_indices,  # The indices that were unmaksed.
            )

    def get_random_indices(self, batch_size):
        """
        Generates random indices for masking patches in the input image.

        Args:
            batch_size (int): The number of samples in the batch.

        Returns:
            A tuple containing the following two tensors:
            - mask_indices (tf.Tensor): A tensor of shape (batch_size, num_mask) containing
            random indices for masking patches.
            - unmask_indices (tf.Tensor): A tensor of shape (batch_size, num_patches - num_mask)
            containing random indices for unmasking patches.

        """
        # Create random indices from a uniform distribution and then split
        # it into mask and unmask indices.
        rand_indices = tf.argsort(
            tf.random.uniform(shape=(batch_size, self.num_patches), dtype=tf.float64),
            axis=-1,
        )
        mask_indices = rand_indices[:, : self.num_mask]
        # mask_indices = tf.where(tf.less_equal(mask_indices, 0),0, mask_indices)

        unmask_indices = rand_indices[:, self.num_mask :]
        # unmask_indices = tf.where(tf.less_equal(unmask_indices, 0),0, unmask_indices)
        return mask_indices, unmask_indices

    def generate_masked_image(self, patches, unmask_indices, random_patch: bool = True):
        """
        Generates a masked version of a patch by randomly selecting patches to mask.

        Args:
            patches (numpy.ndarray): A numpy array of shape (batch_size, num_patches, patch_area)
                containing image patches.
            unmask_indices (tf.Tensor): A tensor of shape (batch_size, num_patches - num_mask)
                containing random indices for unmasking patches.
            random_patch (bool): If True, a random patch is selected for masking. Otherwise, the first
                patch in the batch is used.

        Returns:
            A tuple containing the following two elements:
            - new_patch (numpy.ndarray): A numpy array of the same shape as a patch, containing the
                masked version of the selected patch.
            - idx (int): The index of the patch in the batch that was masked.

        """
        # Choose a random patch and it corresponding unmask index.
        if random_patch == True:
            idx = np.random.choice(patches.shape[0])
        else:
            idx = 0

        patch = patches[idx]
        unmask_index = unmask_indices[idx]
        # Build a numpy array of same shape as patch.
        new_patch = np.zeros_like(patch)

        # Iterate of the new_patch and plug the unmasked patches.
        count = 0

        for i in range(unmask_index.shape[0]):
            new_patch[unmask_index[i]] = patch[unmask_index[i]]
        return new_patch, idx
