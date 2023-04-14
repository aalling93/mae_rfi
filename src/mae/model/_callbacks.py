import numpy as np
import tensorflow as tf
import os
import pandas as pd
from keras import backend as K
from datetime import datetime
from ..Logger import *


def get_callbacks(
    model, save_path: str = "", epoch_interval: int = 5, test_images=None
):
    """ """
    os.makedirs(f"{save_path}/{model.name}", exist_ok=True)
    name_append = datetime.now().strftime("%d_%m_%Y_%H_%M")
    best_model_file = f"{save_path}/{model.name}/best_model_{model.name}"

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
    callbacks = [
        early_stop,
        reduce_lr,
        cbk,
        best_model,
        SSIMMonitor(test_images=test_images),
        tensorboard_callback,
        TrainMonitor(epoch_interval=epoch_interval, test_images=test_images),
    ]

    return callbacks


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
        sim = tf.reduce_mean(tf.image.ssim(
            self.test_augmented_images,
            test_decoder_outputs,
            1.0,))
        
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
                sim_vh = tf.image.ssim(
                    original_image[:, :, 1:], reconstructed_image[:, :, 1:], 1.0
                )
                ssims = (sim_vv + sim_vh) / 2

                clearml_log_scalar(sim_vv, epoch, "vv", "SSIM")
                clearml_log_scalar(sim_vh, epoch, "vh", "SSIM")
                clearml_log_scalar(ssims, epoch, "mean", "SSIM")

                clearml_plot_examples(
                    original_image, masked_image, reconstructed_image, epoch, img_ix
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

            # self.model.save(f'{self.model.name}/model_{self.model.name}_newest_epoch', overwrite=True)

            pd.DataFrame(self.model.history.history).to_pickle(
                f"{self.model_folder}/{self.model.name}/history/history_newest_epoch_{self.model.name}.pkl"
            )
        if epoch % 50 == 0:
            self.model.save_weights(
                f"{self.model_folder}/{self.model.name}/weights/weights_epoch{epoch}"
            )
