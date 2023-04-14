

import argparse
from datetime import datetime

import clearml
from clearml import Dataset
from clearml import Task
from datetime import datetime

import tensorflow as tf

import numpy as np

from sys import platform





import tensorflow_addons as tfa
from src.mae.data import Data
from src.mae._mae_util import load_gpu

from sklearn.model_selection import KFold
from src.mae.CONSTANTS import *
#python3 train.py -data_features="23,25,27,29,31" -session_name='Qd_inputs' -notes="This is a training run for AE on Qd data." -GPU_memory=5000
parser = argparse.ArgumentParser(description="Training RFI mae")
parser.add_argument("-BUFFER_SIZE", "--BUFFER_SIZE", help="BUFFER_SIZE", default=1024, type=int)
parser.add_argument("-BATCH_SIZE", "--BATCH_SIZE", help="Batch size integer input.", default=10, type=int)
parser.add_argument("-EPOCHS", "--EPOCHS", help="EPOCHS", default=400, type=int)
parser.add_argument("-seed", "--seed", help="seed value", default=42, type=int)
parser.add_argument( "-data",  "--data",  help="relative path to data",  default="data/processed/train_zm_jsd.npy'",  type=str,)
parser.add_argument(  "-data_split",  "--data_split",  help="split percentage (train)",  default=0.8,  type=float,)
parser.add_argument("-verbose", "--verbose", help="verbose", default=1, type=int)
parser.add_argument("-GPU", "--GPU", help="which GPU", default=0, type=int)
parser.add_argument( "-GPU_memory", "--GPU_memory", help="GPU memory", default=20000, type=int)


parser.add_argument(  "-latent_space_dim",  "--latent_space_dim",  help="Latens space dimension for model.",  default=15,  type=int,)
parser.add_argument(  "-neurons", "--neurons", help="neurons", default=[210, 160, 360, 60, 310], type=list)
parser.add_argument(  "-model_folder",  "--model_folder",  help="Folder for model",  default="models/cv",)
parser.add_argument(  "-dropout_prob", "--dropout_prob", help="dropout_prob", default=0.1, type=float)
parser.add_argument( "-session_name",  "--session_name",  help="your name to clear ml",  default="all data",  type=str,)
parser.add_argument( "-notes",  "--notes",  help="notes",  default="NA",  type=str,)


####
parser.add_argument(  "-LEARNING_RATE",  "--LEARNING_RATE",  help="LEARNING_RATE",  default=5e-3,  type=float,)
parser.add_argument(  "-WEIGHT_DECAY",  "--WEIGHT_DECAY",  help="WEIGHT_DECAY",  default=1e-4,  type=float,)
parser.add_argument(  "-MASK_PROPORTION",  "--MASK_PROPORTION",  help="MASK_PROPORTION",  default=0.6,  type=float,)
### model

parser.add_argument(  "-LAYER_NORM_EPS",  "--LAYER_NORM_EPS",  help="LAYER_NORM_EPS",  default=1e-6,  type=float,)
parser.add_argument(  "-ENC_PROJECTION_DIM",  "--ENC_PROJECTION_DIM",  help="ENC_PROJECTION_DIM",  default=128,  type=float,)
parser.add_argument(  "-DEC_PROJECTION_DIM",  "--DEC_PROJECTION_DIM",  help="DEC_PROJECTION_DIM",  default=64,  type=float,)
parser.add_argument(  "-ENC_NUM_HEADS",  "--ENC_NUM_HEADS",  help="ENC_NUM_HEADS",  default=4,  type=float,)
parser.add_argument(  "-ENC_LAYERS",  "--ENC_LAYERS",  help="ENC_LAYERS",  default=3,  type=int,)
parser.add_argument(  "-DEC_LAYERS",  "--DEC_LAYERS",  help="DEC_LAYERS",  default=1,  type=int,)


ENC_TRANSFORMER_UNITS = [
    ENC_PROJECTION_DIM * 2,
    ENC_PROJECTION_DIM,
]  # Size of the transformer layers.
DEC_TRANSFORMER_UNITS = [
    DEC_PROJECTION_DIM * 2,
    DEC_PROJECTION_DIM,
]


parser.add_argument(  "-hyperturner_name",  "--hyperturner_name",  help="hyperturner_name",  default=0,  type=int,)
parser.add_argument(  "-kfolds",  "--kfolds",  help="Amount of K-folds for CV. (K models will be made)",  default=5,  type=int,)
args = parser.parse_args()


if args.verbose > 0:
    print(
        f"\nTraining models with the following parms: \nEpochs: {args.epochs} \nBatch size: {args.batchSize} \nSeed: {args.seed} \nKfolds: {args.kfolds} \nLatens space dim: {args.latent_space_dim}\n"
    )




####################### DONE WITH ARGS ############################
if platform == "linux" or platform == "linux2":
    strategy = load_gpu(which=int(args.GPU), memory=int(args.GPU_memory))


name_append = datetime.now().strftime("%d_%m_%Y_%H_%M")
model_name = f"mae" + "_" + name_append 
task = Task.create(project_name="RFI_mae", task_name=f"{model_name}")




#### Original Data
if args.verbose > 0:
    print("\nloading data")
data = Data()
data.load_data('../data/processed/train_zm_jsd.npy')
np.random.seed(args.seed)


cvscores = []
kfold = KFold(n_splits=args.kfolds, random_state=None, shuffle=False)
name_append = datetime.now().strftime("%d_%m_%Y_%H")
i = 1
if args.verbose > 0:
    print("\nPreparing K-folds")
    print(f"\nTime: {name_append}")

os.makedirs(f"{args.model_folder}", exist_ok=True)

for train_idx, val_idx in kfold.split(
    data.norm_data.norm_train, data.norm_data.norm_train
):
    tf.keras.backend.clear_session()
    if args.verbose > 0:
        print("training_model ", i)
    # making model for split i
    model_name = "AE_falter" + "_" + name_append + "_" + str(i)
    task = Task.init(project_name="Drone_em_dl", task_name=f"Drone_em_dl_{args.session_name}_{model_name}")

    with Fae() as ae:
        ae = ae.make_model(
            input_size=data.norm_data.norm_train[0].shape,
            latent_space_dim=args.latent_space_dim,
            dense_neurons=args.neurons,
            dropout_prob=args.dropout_prob,
            name=model_name,
        )

    callbacks = get_callbacks(ae)
    model_fit(
        ae,
        data.norm_data.norm_train[train_idx],
        data.norm_data.norm_train[train_idx],
        batch_size=args.batchSize,
        epochs=args.epochs,
        verbose=args.verbose,
        callbacks=callbacks,
    )

    i = i + 1
