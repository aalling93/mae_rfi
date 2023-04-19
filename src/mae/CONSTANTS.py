import tensorflow as tf
from datetime import datetime

# MODEL METADATA
MODEL_NAME = "RFI_mae"
MODEL_FOLDER = f"models"
NAME_APPEND = datetime.now().strftime("%d_%m_%Y_%H_%M")

# DATA
INPUT_SHAPE = (340, 500, 2)
IMAGE_SIZE = (60, 100, 1)
DATA_FOLDER = "data/processed/train_zm_jsd.npy"
DATA_SPLIT = 0.8
RANDOM_CROP = True
NUM_CLASSES = 1

# TRAINING
BUFFER_SIZE = 1024
BATCH_SIZE = 1
VERBOSE = 1
GPU_NUMBER = 1
GPU_MEMORY = 40000
EPOCHS = 1000
WARMUP_EPOCH_PERCENTAGE = 0.05
MASK_PROPORTION = 0.60  # We have found 75% masking to give us the best results.

# OPTIMIZER
LEARNING_RATE = 5e-3
LEARNING_RATE_WARM_UP = LEARNING_RATE * 1e-8
WEIGHT_DECAY = 1e-4

# MODEL
DECODER_ARCITECTURE_FULL = True

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




######### OTHER

ENC_TRANSFORMER_UNITS = [
    ENC_PROJECTION_DIM * 2,
    ENC_PROJECTION_DIM,
]  # Size of the transformer layers.
DEC_TRANSFORMER_UNITS = [
    DEC_PROJECTION_DIM * 2,
    DEC_PROJECTION_DIM,
]

if IMAGE_SIZE[2] == 1:
    ONLY_VH = True
else:
    ONLY_VH = False

PATCH_SIZE = (
    int(IMAGE_SIZE[0] / 10),
    int(IMAGE_SIZE[1] / 10),
)  # Size of the patches to be extracted from the input images.


NUM_PATCHES = int(
    int(IMAGE_SIZE[0] / PATCH_SIZE[0]) * int(IMAGE_SIZE[1] / PATCH_SIZE[1])
)


SEED = 42


AUTO = tf.data.AUTOTUNE
