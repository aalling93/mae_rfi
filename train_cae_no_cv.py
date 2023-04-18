
from datetime import datetime
from sys import platform

import numpy as np
import tensorflow_addons as tfa
from clearml import Task

from src.mae import CL_Logger as CL_Logger
from src.mae import *
from src.mae.CONSTANTS import *
from src.mae.model._callbacks import *
from src.mae.model.model_rfi.model_arcitectures import \
    modelPoolingDropout as cae

####################### DONE WITH ARGS ############################
if platform == "linux" or platform == "linux2":
    strategy = load_gpu(which=0, memory=60000)


train = np.load("data/processed/train_zm_jsd.npy", allow_pickle=True)
test = np.load("data/processed/train_zm_jsd.npy", allow_pickle=True)

train = np.array(
    [
        center_crop(im, [500, 340])
        for im in train
        if (im.shape[0] >= 340 and im.shape[1] >= 500)
    ]
)

test = np.array(
    [
        center_crop(im, [500, 340])
        for im in test
        if (im.shape[0] >= 340 and im.shape[1] >= 500)
    ]
)



BATCH_SIZE = 5
EPOCHS = 16
WARMUP_EPOCH_PERCENTAGE = 0.01
LEARNING_RATE = 0.01
LEARNING_RATE_WARM_UP = LEARNING_RATE * 1e-4


try:
    task.close()
except:
    pass


for latent in [25, 75, 125, 250, 350, 500]:
    name_append = datetime.now().strftime("%d_%m_%Y_%H_%M")

    task = Task.init(
        project_name="RFI_mae", task_name=f"RFI_reconstruction_{name_append}"
    )
    ae_model, __, __ = cae(img_size=train[0].shape, latent_space_dim=latent)
    ae_model._name = f"RFI_reconstruction_{name_append}"
    os.makedirs(f"../models/{ae_model.name}", exist_ok=True)
    os.makedirs(f"../models/{ae_model.name}/logs", exist_ok=True)
    os.makedirs(f"../models/{ae_model.name}/logs", exist_ok=True)

    # callbacks

    log_dir = f"logs/{ae_model.name}"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=True,
        update_freq="epoch",
        profile_batch=2,
        embeddings_freq=1,
    )
    # train_images,test_images
    ssim_callback = SSIMMonitor_ae(train, test)
    debug = np.vstack((train[-10:], test[-10:]))
    debug_callback = TrainMonitor_ae(epoch_interval=5, test_images=debug)

    best_model_file = f"../models/{ae_model.name}/best_model_{ae_model.name}"
    csv_logger = tf.keras.callbacks.CSVLogger(
        f"../models/{ae_model.name}/traininglog.csv"
    )

    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=100)
    best_model = tf.keras.callbacks.ModelCheckpoint(
        best_model_file,
        monitor="loss",
        mode="auto",
        save_weights_only=False,
        verbose=0,
        save_best_only=True,
    )
    # mae_model.save_weights("ckpt")
    X_model = tf.keras.callbacks.ModelCheckpoint(
        filepath=f"../models/{ae_model.name}/model_{name_append}",
        monitor="loss",
        save_weights_only=True,
        save_freq="epoch",
        mode="auto",
        verbose=0,
        period=50,
        save_format="tf",
    )

    log_dir = f"../models/{ae_model.name}/logs"
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="loss", factor=0.002, patience=20, min_lr=1e-25
    )

    #### Optimizer
    total_steps = int((len(train) / BATCH_SIZE) * EPOCHS)
    # Compute the number of warmup batches or steps.
    warmup_steps = int(total_steps * WARMUP_EPOCH_PERCENTAGE)

    hold_base_rate_steps = 0
    cosine_warm_up_lr = WarmUpCosineDecayScheduler(
        learning_rate_base=LEARNING_RATE,
        total_steps=total_steps,
        warmup_learning_rate=LEARNING_RATE_WARM_UP,
        warmup_steps=warmup_steps,
        hold_base_rate_steps=hold_base_rate_steps,
    )
    # Define the optimizer with the learning rate schedule
    optimizer = tfa.optimizers.AdamW(
        learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    learning_rate_metric = get_lr_metric(optimizer)

    callbacks = [
        tensorboard_callback,
        ssim_callback,
        debug_callback,
        csv_logger,
        best_model,
        X_model,
        reduce_lr,
        cosine_warm_up_lr,
    ]

    # saving parms

    WarmUpCosineDecayScheduler_parms = {
        "optimizer": cosine_warm_up_lr,
        "learning_rate_base": LEARNING_RATE,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "LEARNING_RATE_WARM_UP": LEARNING_RATE_WARM_UP,
        "hold_base_rate_steps": hold_base_rate_steps,
    }

    parms = {
        "data": "../data/processed/train_zm_jsd.npy",
        "crop": "center",
        "train size": train.shape,
        "test size": test.shape,
        "debug size": debug.shape,
        "debug comment": "10 train 10 test",
        "latent space": latent,
        "model name": ae_model.name,
        "model summary": get_model_summary(ae_model),
        "encoder summary": get_model_summary(ae_model.layers[1]),
        "decoder summary": get_model_summary(ae_model.layers[2]),
    }
    task.connect(parms, "parms")
    task.connect(WarmUpCosineDecayScheduler_parms, "optimizer")

    ae_model.fit(
        train,
        train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        shuffle=True,
        verbose=1,
        callbacks=callbacks,
    )

    loss = ae_model.evaluate(train)

    results = {"loss [loss, mae, lr]": loss, "latent space": latent}
    task.connect(results, "results")
    task.close()
