"""'
TRAINING SCRIPT.

"""

import argparse
from sys import platform

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from clearml import Task

from src.mae import *
from src.mae._mae_util import load_gpu
from src.mae.CONSTANTS import *
from src.mae.data import Data
from src.mae.data._augment import *
from src.mae.CL_Logger import *
from src.mae.model._callbacks import *
from src.mae.model._callbacks import get_callbacks
from src.mae.model.util import get_lr_metric
from src.mae.CL_Logger.Clearml import clearml_plot_model

# change this
parser = argparse.ArgumentParser(description="Training RFI mae")
parser.add_argument("-BUFFER_SIZE", "--BUFFER_SIZE", help="BUFFER_SIZE", default=BUFFER_SIZE, type=int)
parser.add_argument("-BATCH_SIZE","--BATCH_SIZE",help="Batch size integer input.",default=BATCH_SIZE,type=int,)
parser.add_argument("-EPOCHS", "--EPOCHS", help="EPOCHS", default=EPOCHS, type=int)
parser.add_argument("-seed", "--seed", help="seed value", default=SEED, type=int)
parser.add_argument("-data","--data", help="relative path to data",  default=DATA_FOLDER,  type=str,)
parser.add_argument(  "-data_split",  "--data_split",  help="split percentage (train)", default=DATA_SPLIT, type=float,)
parser.add_argument("-verbose", "--verbose", help="verbose", default=VERBOSE, type=int)
parser.add_argument("-GPU", "--GPU", help="which GPU", default=GPU_NUMBER, type=int)
parser.add_argument( "-GPU_memory", "--GPU_memory", help="GPU memory", default=GPU_MEMORY, type=int)
parser.add_argument( "-model_folder",  "--model_folder",  help="Folder for model",  default=MODEL_FOLDER,)
parser.add_argument(  "-MODEL_NAME",  "--MODEL_NAME",  help="your name to clear ml",  default=MODEL_NAME,  type=str,)
parser.add_argument( "-notes", "--notes", help="notes", default="This is a Masked autoencoder model. The model is using a masked encoding strategy with a transformer structure to reconstruct original iamges.", type=str,)
parser.add_argument( "-upload_data", "--upload_data", help="upload_data", default=False, type=bool)
parser.add_argument(  "-NAME_APPEND",  "--NAME_APPEND",  help="NAME_APPEND",  default=NAME_APPEND,  type=str,)
#### training
parser.add_argument(  "-LEARNING_RATE",  "--LEARNING_RATE",  help="LEARNING_RATE", default=LEARNING_RATE,  type=float,)
parser.add_argument(  "-WEIGHT_DECAY",  "--WEIGHT_DECAY",  help="WEIGHT_DECAY",  default=WEIGHT_DECAY,  type=float,)
parser.add_argument(  "-MASK_PROPORTION",  "--MASK_PROPORTION",  help="MASK_PROPORTION",  default=MASK_PROPORTION,  type=float,)
parser.add_argument(  "-WARMUP_EPOCH_PERCENTAGE",  "--WARMUP_EPOCH_PERCENTAGE",  help="WARMUP_EPOCH_PERCENTAGE",  default=WARMUP_EPOCH_PERCENTAGE,  type=float,)
### model
parser.add_argument(  "-LAYER_NORM_EPS", "--LAYER_NORM_EPS",  help="LAYER_NORM_EPS", default=LAYER_NORM_EPS,  type=float,)
parser.add_argument(  "-ENC_PROJECTION_DIM",  "--ENC_PROJECTION_DIM",  help="ENC_PROJECTION_DIM", default=ENC_PROJECTION_DIM, type=int,)
parser.add_argument(  "-DEC_PROJECTION_DIM",  "--DEC_PROJECTION_DIM",  help="DEC_PROJECTION_DIM",  default=DEC_PROJECTION_DIM, type=int,)
parser.add_argument(  "-ENC_NUM_HEADS",  "--ENC_NUM_HEADS",  help="ENC_NUM_HEADS", default=ENC_NUM_HEADS,  type=int,)
parser.add_argument(  "-ENC_LAYERS",  "--ENC_LAYERS",  help="ENC_LAYERS",  default=ENC_LAYERS,  type=int,)
parser.add_argument(  "-DEC_LAYERS",  "--DEC_LAYERS",  help="DEC_LAYERS",  default=DEC_LAYERS,  type=int,)

parser.add_argument(  "-ENC_TRANSFORMER_UNITS",  "--ENC_TRANSFORMER_UNITS",  help="ENC_TRANSFORMER_UNITS",  default=ENC_TRANSFORMER_UNITS,  type=int,)
parser.add_argument(  "-DROPOUT_RATE",  "--DROPOUT_RATE",  help="DROPOUT_RATE",  default=DROPOUT_RATE,  type=float,)
parser.add_argument(  "-DEC_NUM_HEADS",  "--DEC_NUM_HEADS",  help="DEC_NUM_HEADS",  default=DEC_NUM_HEADS,  type=int,)
parser.add_argument(  "-DEC_TRANSFORMER_UNITS",  "--DEC_TRANSFORMER_UNITS",  help="DEC_TRANSFORMER_UNITS",  default=DEC_TRANSFORMER_UNITS,  type=int,)
parser.add_argument(  "-downstream",  "--downstream",  help="downstream",  default=False,  type=bool,)

args = parser.parse_args()


if args.verbose > 0:
    print(
        f"\nTraining models with the following parms: \nEpochs: {args.EPOCHS} \nBatch size: {args.BATCH_SIZE} \n"
    )


# other parms
ENC_TRANSFORMER_UNITS = [
    ENC_PROJECTION_DIM * 2,
    ENC_PROJECTION_DIM,
]  # Size of the transformer layers.
DEC_TRANSFORMER_UNITS = [
    DEC_PROJECTION_DIM * 2,
    DEC_PROJECTION_DIM,
]


####################### DONE WITH ARGS ############################
if platform == "linux" or platform == "linux2":
    strategy = load_gpu(which=int(args.GPU), memory=int(args.GPU_memory))


# name_append = datetime.now().strftime("%d_%m_%Y_%H_%M")
model_name = f"{args.MODEL_NAME}" + "_" + args.NAME_APPEND
task = Task.create(project_name="RFI_mae", task_name=f"{model_name}")


#### Original Data
if args.verbose > 0:
    print("\nloading data")

data = Data()
data.load_data(
    train_data="data/processed/train_zm_d.npy",
    test_data="data/processed/train_zm_d.npy",
    imsize=IMAGE_SIZE,
    only_VH=ONLY_VH,
)
np.random.seed(args.seed)


task = Task.init(project_name="RFI_mae", task_name=f"{model_name}")
task.connect(args, "args")


if args.verbose > 0:
    print(
        f"\nNumber of training examples: {len(data.train)} \nNumber of test example: {len(data.test)}\nTime: {args.NAME_APPEND}"
    )


if args.verbose > 0:
    print("\nMaking encoder")
encoder = create_encoder(
    num_heads=args.ENC_NUM_HEADS,
    num_layers=args.ENC_LAYERS,
    enc_transformer_units=args.ENC_TRANSFORMER_UNITS,
    epsilon=args.LAYER_NORM_EPS,
    enc_projection_dim=args.ENC_PROJECTION_DIM,
    dropout=args.DROPOUT_RATE,
)
if args.verbose > 0:
    print("\nMaking decoder")
decoder = create_decoder(
    num_layers=args.DEC_LAYERS,
    num_heads=args.DEC_NUM_HEADS,
    image_size=IMAGE_SIZE,
    dropout=args.DROPOUT_RATE,
    num_patches=NUM_PATCHES,
    enc_projection_dim=args.ENC_PROJECTION_DIM,
    dec_projection_dim=args.DEC_PROJECTION_DIM,
    epsilon=args.LAYER_NORM_EPS,
    dec_transformer_units=args.DEC_TRANSFORMER_UNITS,
)

patch_encoder = PatchEncoder(
    patch_size=PATCH_SIZE,
    projection_dim=args.ENC_PROJECTION_DIM,
    mask_proportion=args.MASK_PROPORTION,
    downstream=args.downstream,
)


if args.verbose > 0:
    print("\nMaking mae")
mae_model = MaskedAutoencoder(
    train_augmentation_model=get_train_augmentation_model(),
    test_augmentation_model=get_test_augmentation_model(),
    patch_layer=Patches(),
    patch_encoder=patch_encoder,
    encoder=encoder,
    decoder=decoder,
)
# clearml_plot_model(encoder)
# clearml_plot_model(decoder)

mae_model._name = model_name





if args.verbose > 0:
    print("\npreparing scheduler")


#### Optimizer
total_steps = int((len(data.train) / BATCH_SIZE) * EPOCHS)
# Compute the number of warmup batches or steps.
warmup_steps = int(total_steps * WARMUP_EPOCH_PERCENTAGE)

if args.verbose > 0:
    print(f"\nwarmup steps: {warmup_steps}")

hold_base_rate_steps = 0
cosine_warm_up_lr = WarmUpCosineDecayScheduler(
    learning_rate_base=args.LEARNING_RATE,
    total_steps=total_steps,
    warmup_learning_rate=LEARNING_RATE_WARM_UP,
    warmup_steps=warmup_steps,
    hold_base_rate_steps=hold_base_rate_steps,
)
# Define the optimizer with the learning rate schedule
optimizer = tfa.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
learning_rate_metric = get_lr_metric(optimizer)

WarmUpCosineDecayScheduler = {
    "learning_rate_base": LEARNING_RATE,
    "total_steps": total_steps,
    "warmup_steps": warmup_steps,
    "LEARNING_RATE_WARM_UP": LEARNING_RATE_WARM_UP,
    "hold_base_rate_steps": hold_base_rate_steps,
}

hyperparameters = {
    "ENC_TRANSFORMER_UNITS": args.ENC_TRANSFORMER_UNITS,
    "DEC_TRANSFORMER_UNITS": args.DEC_TRANSFORMER_UNITS,
    "total_steps": total_steps,
    "warmup_steps": warmup_steps,
}





task.connect(hyperparameters, "hyperparameters")
task.connect(WarmUpCosineDecayScheduler, "WarmUpCosineDecayScheduler")
# Compile and pretrain the model.

if args.verbose > 0:
    print("\nCompiling model")
mae_model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=["mae", learning_rate_metric],
)


model_parms = {
        "model name": mae_model._name,
        "decoder name": mae_model.decoder.name,
        "decoder": get_model_summary(mae_model.decoder),
        "encoder name": mae_model.encoder.name,
        "encoder": get_model_summary(mae_model.encoder),

}
task.connect(model_parms, "model summary")

train_callbacks = get_callbacks(
    mae_model,
    save_path=args.model_folder,
    epoch_interval=5,
    test_images=data.train[:10],
)
train_callbacks.append(cosine_warm_up_lr)


if args.verbose > 0:
    print("\nTraining model")
history = mae_model.fit(
    data.train_ds,
    epochs=args.EPOCHS,
    validation_data=data.train_ds,
    workers=25,
    max_queue_size=50,
    callbacks=train_callbacks,
)

if args.verbose > 0:
    print("\n=====================================\nTraining done. Evaluating model")
# Measure its performance.


####
loss, mae = mae_model.evaluate(data.train_ds)
print(f"Loss: {loss:.2f}")
print(f"MAE: {mae:.2f}")
