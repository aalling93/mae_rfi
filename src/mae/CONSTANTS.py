import tensorflow as tf
from datetime import datetime

# MODEL
MODEL_NAME = "RFI_mae"
MODEL_FOLDER = f"models"
NAME_APPEND = datetime.now().strftime("%d_%m_%Y_%H_%M")
#
BUFFER_SIZE = 1024
BATCH_SIZE = 10
AUTO = tf.data.AUTOTUNE
INPUT_SHAPE = (340, 500, 2)
NUM_CLASSES = 10
SEED = 42
DATA_FOLDER = "data/processed/train_zm_jsd.npy"
DATA_SPLIT = 0.8
VERBOSE = 1
GPU_NUMBER = 1
GPU_MEMORY = 20000
# OPTIMIZER
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-4

# PRETRAINING
EPOCHS = 1000
WARMUP_EPOCH_PERCENTAGE = 0.1
# AUGMENTATION
IMAGE_SIZE = (340, 500)
PATCH_SIZE = (34, 50)  # Size of the patches to be extracted from the input images.


NUM_PATCHES = int(
    int(IMAGE_SIZE[0] / PATCH_SIZE[0]) * int(IMAGE_SIZE[1] / PATCH_SIZE[1])
)
MASK_PROPORTION = 0.60  # We have found 75% masking to give us the best results.

# ENCODER and DECODER
LAYER_NORM_EPS = 1e-6
ENC_PROJECTION_DIM = 128
DEC_PROJECTION_DIM = 64
ENC_NUM_HEADS = 4
ENC_LAYERS = 3
DEC_NUM_HEADS = 4
DROPOUT_RATE = 0.1
DEC_LAYERS = (
    2  # The decoder is lightweight but should be reasonably deep for reconstruction.
)
DEC_LAYERS = 1
ENC_TRANSFORMER_UNITS = [
    ENC_PROJECTION_DIM * 2,
    ENC_PROJECTION_DIM,
]  # Size of the transformer layers.
DEC_TRANSFORMER_UNITS = [
    DEC_PROJECTION_DIM * 2,
    DEC_PROJECTION_DIM,
]
