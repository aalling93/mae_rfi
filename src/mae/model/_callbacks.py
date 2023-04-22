"""
This script defines a set of callbacks for Keras/Tensorflow models. 
The get_callbacks function returns a list of callbacks including early stopping, 
best model checkpointing, learning rate reduction, and custom callbacks for logging and monitoring.

The SSIMMonitor callback calculates the Structural Similarity Index (SSIM) 
between the model's reconstructed image and the original image for a set of test images. 

The TrainMonitor callback visualizes the model's progress by generating images of 
masked patches and their reconstructions at regular intervals during training.

The script requires a set of dependencies including numpy, pandas, tensorflow, and keras. 
It also imports custom modules that are not included in the script 
(CL_Logger, CustomModelCheckpoint, PatchEncoder, and PatchLayer), 
so it's not possible to run this script as is.


Functions:

classes (for callbacks):
1) MemoryPrintingCallback: get memory usage for each epoch (def, print to clearml. no in terminal)
2) SSIMMonitor (mae)
3) TrainMonitor (mae)
4) saveModel
5) CustomModelCheckpoint
6) WarmUpCosineDecayScheduler
7) SSIMMonitor_ae (cae)
8) TrainMonitor_ae (cae)
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
import gc
from ..CL_Logger import *


def get_callbacks(
    model,
    save_path: str = "",
    epoch_interval: int = 5,
    test_images=None,
):
    """ """
    os.makedirs(f"{save_path}/{model.name}", exist_ok=True)
    name_append = datetime.now().strftime("%d_%m_%Y_%H_%M")
    best_model_file = f"{save_path}/{model.name}/best_model_{model.name}"

    csv_logger = tf.keras.callbacks.CSVLogger(
        f"{save_path}/{model.name}/traininglog.csv"
    )

    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=100)
    best_model = tf.keras.callbacks.ModelCheckpoint(
        best_model_file,
        monitor="val_loss",
        mode="auto",
        save_weights_only=True,
        verbose=0,
        save_best_only=True,
    )
    # mae_model.save_weights("ckpt")
    X_model = tf.keras.callbacks.ModelCheckpoint(
        filepath=f"{save_path}/{model.name}/model_{name_append}",
        monitor="val_loss",
        save_weights_only=True,
        save_freq="epoch",
        mode="auto",
        verbose=0,
        period=50,
        save_format="tf",
    )

    log_dir = f"logs/{model.name}"

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=True,
        update_freq="epoch",
        profile_batch=2,
        embeddings_freq=1,
    )

    cbk = CustomModelCheckpoint(model, save_path)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="loss", factor=0.002, patience=20, min_lr=1e-25
    )
    # save_encoder = saveModel(encoder)
    callbacks = [
        early_stop,
        reduce_lr,
        cbk,
        csv_logger,
        best_model,
        SSIMMonitor(test_images=test_images),
        tensorboard_callback,
        TrainMonitor(epoch_interval=epoch_interval, test_images=test_images),
    ]

    return callbacks


class MemoryPrintingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gpu_dict = tf.config.experimental.get_memory_info("GPU:0")
        clearml_log_scalar(
            (float(gpu_dict["peak"]) / (1024**3)),
            epoch,
            series="Peak",
            title="Model fit memory GB",
        )
        clearml_log_scalar(
            (float(gpu_dict["peak"]) / (1024**3)),
            epoch,
            series="Current",
            title="Model fit memory GB",
        )

        # tf.print('\n GPU memory details [current: {} gb, peak: {} gb]'.format(
        #    float(gpu_dict['current']) / (1024 ** 3),
        #    float(gpu_dict['peak']) / (1024 ** 3)))


class SSIMMonitor(tf.keras.callbacks.Callback):
    def __init__(self, test_images):
        self.test_images = test_images

    def on_epoch_end(self, epoch, logs=None):
        self.test_augmented_images = self.model.test_augmentation_model(
            self.test_images
        )
        self.test_patches = self.model.patch_layer(self.test_augmented_images)

        (
            test_unmasked_embeddings,
            test_masked_embeddings,
            test_unmasked_positions,
            test_mask_indices,
            test_unmask_indices,
        ) = self.model.patch_encoder(self.test_patches)

        test_encoder_outputs = self.model.encoder(test_unmasked_embeddings)
        test_encoder_outputs = test_encoder_outputs + test_unmasked_positions
        test_decoder_inputs = tf.concat(
            [test_encoder_outputs, test_masked_embeddings], axis=1
        )
        test_decoder_outputs = self.model.decoder(test_decoder_inputs)
        sim = tf.reduce_mean(
            tf.image.ssim(
                self.test_augmented_images,
                test_decoder_outputs,
                1.0,
            )
        )

        clearml_log_scalar(sim, epoch, "test_all", "SSIM")


class TrainMonitor(tf.keras.callbacks.Callback):
    def __init__(self, epoch_interval=None, test_images=None):
        self.epoch_interval = epoch_interval
        self.test_images = test_images

    def on_epoch_end(self, epoch, logs=None):
        if self.epoch_interval and epoch % self.epoch_interval == 0:
            # Show a maksed patch image.
            for img_ix in range(len(self.test_images)):
                test_augmented_images = self.model.test_augmentation_model(
                    self.test_images[img_ix : img_ix + 1]
                )
                test_patches = self.model.patch_layer(test_augmented_images)
                (
                    test_unmasked_embeddings,
                    test_masked_embeddings,
                    test_unmasked_positions,
                    test_mask_indices,
                    test_unmask_indices,
                ) = self.model.patch_encoder(test_patches)
                test_encoder_outputs = self.model.encoder(test_unmasked_embeddings)
                test_encoder_outputs = test_encoder_outputs + test_unmasked_positions
                test_decoder_inputs = tf.concat(
                    [test_encoder_outputs, test_masked_embeddings], axis=1
                )
                test_decoder_outputs = self.model.decoder(test_decoder_inputs)

                test_masked_patch, idx = self.model.patch_encoder.generate_masked_image(
                    test_patches, test_unmask_indices, random_patch=False
                )
                original_image = test_augmented_images[idx]
                masked_image = self.model.patch_layer.reconstruct_from_patch(
                    test_masked_patch
                )
                reconstructed_image = test_decoder_outputs[idx]

                sim_vv = tf.image.ssim(
                    original_image[:, :, 0:1], reconstructed_image[:, :, 0:1], 1.0
                )
                clearml_log_scalar(sim_vv, epoch, "pol", "SSIM")
                try:
                    sim_vh = tf.image.ssim(
                        original_image[:, :, 1:], reconstructed_image[:, :, 1:], 1.0
                    )
                    ssims = (sim_vv + sim_vh) / 2
                    clearml_log_scalar(sim_vv, epoch, "vh", "SSIM")
                    clearml_log_scalar(ssims, epoch, "mean", "SSIM")
                except:
                    pass

                if self.test_images.shape[-1] == 2:

                    clearml_plot_examples(
                        original_image, masked_image, reconstructed_image, epoch, img_ix
                    )
                else:

                    clearml_plot_one_polari_all(
                        original_image[:, :, 0],
                        masked_image[:, :, 0],
                        reconstructed_image[:, :, 0],
                        test_decoder_inputs[idx],
                        epoch,
                        img_ix,
                    )


class saveModel(tf.keras.callbacks.Callback):
    def __init__(self, model, model_folder):
        super().__init__()
        self.epoch_count = 0
        self.learning_rates = []
        self.model = model
        self.model_folder = model_folder

    def on_epoch_end(self, epoch, logs=None):
        if epoch > 0:
            self.model.save(
                f"{self.model.name}/encoder_newest_epoch.h5", overwrite=True
            )


class CustomModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, model, model_folder):
        super().__init__()
        self.epoch_count = 0
        self.learning_rates = []
        self.model = model
        self.model_folder = model_folder
        os.makedirs(f"{model_folder}/{model.name}/history", exist_ok=True)
        os.makedirs(f"{model_folder}/{model.name}/weights", exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if epoch > 0:
            if self.model.history.history["val_loss"][-1] == min(
                self.model.history.history["val_loss"]
            ):
                pd.DataFrame(self.model.history.history).to_pickle(
                    f"{self.model_folder}/{self.model.name}/history/history_lowest_loss_epoch_{self.model.name}.pkl"
                )
                lr = K.get_value(self.model.optimizer.lr)
                self.learning_rates.append(lr)

            pd.DataFrame(self.model.history.history).to_pickle(
                f"{self.model_folder}/{self.model.name}/history/history_newest_epoch_{self.model.name}.pkl"
            )


class WarmUpCosineDecayScheduler(tf.keras.callbacks.Callback):
    def __init__(
        self,
        learning_rate_base,
        total_steps,
        global_step_init=0,
        warmup_learning_rate=0.0,
        warmup_steps=0,
        hold_base_rate_steps=0,
        verbose=0,
    ):

        super(WarmUpCosineDecayScheduler, self).__init__()
        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.global_step = global_step_init
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.hold_base_rate_steps = hold_base_rate_steps
        self.verbose = verbose
        self.learning_rates = []

    def on_batch_end(self, batch, logs=None):
        self.global_step = self.global_step + 1
        lr = K.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)

    def on_batch_begin(self, batch, logs=None):
        lr = cosine_decay_with_warmup(
            global_step=self.global_step,
            learning_rate_base=self.learning_rate_base,
            total_steps=self.total_steps,
            warmup_learning_rate=self.warmup_learning_rate,
            warmup_steps=self.warmup_steps,
            hold_base_rate_steps=self.hold_base_rate_steps,
        )
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print(
                "\nBatch %05d: setting learning "
                "rate to %s." % (self.global_step + 1, lr.numpy())
            )


def cosine_decay_with_warmup(
    global_step,
    learning_rate_base,
    total_steps,
    warmup_learning_rate=0.0,
    warmup_steps=0,
    hold_base_rate_steps=0,
):

    if total_steps < warmup_steps:
        raise ValueError("total_steps must be larger or equal to " "warmup_steps.")
    learning_rate = (
        0.5
        * learning_rate_base
        * (
            1
            + tf.cos(
                np.pi
                * (
                    tf.cast(global_step, tf.float32)
                    - warmup_steps
                    - hold_base_rate_steps
                )
                / float(total_steps - warmup_steps - hold_base_rate_steps)
            )
        )
    )
    if hold_base_rate_steps > 0:
        learning_rate = tf.where(
            global_step > warmup_steps + hold_base_rate_steps,
            learning_rate,
            learning_rate_base,
        )
    if warmup_steps > 0:
        if learning_rate_base < warmup_learning_rate:
            raise ValueError(
                "learning_rate_base must be larger or equal to " "warmup_learning_rate."
            )
        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
        warmup_rate = slope * tf.cast(global_step, tf.float32) + warmup_learning_rate
        learning_rate = tf.where(global_step < warmup_steps, warmup_rate, learning_rate)
    return tf.where(global_step > total_steps, 0.0, learning_rate, name="learning_rate")


class SSIMMonitor_ae(tf.keras.callbacks.Callback):
    def __init__(self, train_images, test_images):
        self.test_images = tf.cast(test_images, tf.float32)
        self.train_images = tf.cast(train_images, tf.float32)

    def on_epoch_end(self, epoch, logs=None):

        test_decoder_outputs = self.model.predict(self.test_images)
        # test_decoder_outputs = tf.cast(test_decoder_outputs, tf.float32)
        # test_images = tf.cast(self.test_images, tf.float32)
        # print(test_decoder_outputs.shape)
        # print(test_images.shape)
        sim_test = tf.reduce_mean(
            tf.image.ssim(
                abs(self.test_images),
                abs(test_decoder_outputs),
                1.0,
            )
        )
        sim_vv_test = tf.reduce_mean(
            tf.image.ssim(
                abs(self.test_images[:, :, :, 0:1]),
                abs(test_decoder_outputs[:, :, :, 0:1]),
                1.0,
            )
        )
        sim_vh_test = tf.reduce_mean(
            tf.image.ssim(
                abs(self.test_images[:, :, :, 1:]),
                abs(test_decoder_outputs[:, :, :, 1:]),
                1.0,
            )
        )

        train_decoder_outputs = self.model.predict(self.train_images)
        # test_decoder_outputs = tf.cast(test_decoder_outputs, tf.float32)
        # test_images = tf.cast(self.train_images, tf.float32)
        # print(test_decoder_outputs.shape)
        # print(test_images.shape)
        sim_train = tf.reduce_mean(
            tf.image.ssim(
                abs(self.train_images),
                abs(train_decoder_outputs),
                1.0,
            )
        )
        sim_vv_train = tf.reduce_mean(
            tf.image.ssim(
                abs(self.train_images[:, :, :, 0:1]),
                abs(train_decoder_outputs[:, :, :, 0:1]),
                1.0,
            )
        )
        sim_vh_train = tf.reduce_mean(
            tf.image.ssim(
                abs(self.train_images[:, :, :, 1:]),
                abs(train_decoder_outputs[:, :, :, 1:]),
                1.0,
            )
        )

        clearml_log_scalar(sim_test, epoch, "test_all", "SSIM")
        clearml_log_scalar(sim_vv_test, epoch, "test_vv", "SSIM")
        clearml_log_scalar(sim_vh_test, epoch, "test_vh", "SSIM")

        clearml_log_scalar(sim_train, epoch, "train_all", "SSIM")
        clearml_log_scalar(sim_vv_train, epoch, "train_vv", "SSIM")
        clearml_log_scalar(sim_vh_train, epoch, "train_vh", "SSIM")
        tf.keras.backend.clear_session()
        gc.collect()


class TrainMonitor_ae(tf.keras.callbacks.Callback):
    def __init__(self, epoch_interval: int = 5, test_images=None):
        self.epoch_interval = epoch_interval
        self.test_images = test_images

    def on_epoch_end(self, epoch, logs=None):
        if self.epoch_interval and epoch % self.epoch_interval == 0:

            for img_ix in range(len(self.test_images)):

                encoded = self.model.layers[1].predict(
                    self.test_images[img_ix : img_ix + 1]
                )
                decoded = self.model.layers[2].predict(encoded)

                # clearml_plot_org_latent_recon(
                #    self.test_images[img_ix], encoded[0], decoded[0], epoch, img_ix
                # )

                # print(encoded.shape)
                # print(encoded.shape)
                if self.test_images.shape[-1] == 2:
                    clearml_plot_org_latent_recon(
                        self.test_images[img_ix], encoded[0], decoded[0], epoch, img_ix
                    )
                else:
                    clearml_plot_org_latent_recon_single_pil(
                        self.test_images[img_ix][:, :, 0],
                        encoded[0],
                        decoded[0][:, :, 0],
                        epoch,
                        img_ix,
                    )
