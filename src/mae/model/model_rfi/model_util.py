import pandas as pd
#import tensorflow.compat.v2 as tf
import tensorflow.keras.backend as K
import tensorflow as tf
from collections import defaultdict
import tensorflow_probability as tfp
from . import model_arcitectures as arc
import keras_tuner as kt
import numpy as np
from datetime import datetime
tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions
from clearml import Task

def ssim_loss(y_true, y_pred):

    #https://medium.com/srm-mic/all-about-structural-similarity-index-ssim-theory-code-in-pytorch-6551b455541e
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))


def mssim_loss(y_true, y_pred):
    
    return 1 - tf.reduce_mean(tf.image.tf.image.ssim_multiscale(y_true, y_pred, 1.0))


def warmup(epoch):
    value = (epoch / 100000.0) * (epoch <= 100000.0) + 1.0 * (epoch > 100000.0)
    tf.keras.backend.set_value(beta_warmup, value)
    print(
        f"beta old: {tf.keras.backend.get_value(beta_warmup)}. Learning rate: {tf.keras.backend.eval(vae.optimizer.lr)}"
    )


def get_lr_metric(optimizer):
    @tf.autograph.experimental.do_not_convert
    def lr(y_true, y_pred):
        return optimizer.lr

    return lr


class CustomModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self,path):
        super().__init__()
        self.epoch_count = 0
        self.learning_rates = []
        self.path = path

    def on_epoch_end(self, epoch, logs=None):
        if epoch > 0:
            if self.model.history.history["val_loss"][-1] == min(
                self.model.history.history["val_loss"]
            ):
                self.model.save_weights(
                    f"{self.path}/{self.model.name}/weights_lowest_loss_{self.model.name}.h5",
                    overwrite=True,
                )
                pd.DataFrame(self.model.history.history).to_pickle(
                    f"{self.path}/{self.model.name}/history_lowest_loss_epoch_{self.model.name}.pkl"
                )

            lr = K.get_value(self.model.optimizer.lr)
            self.learning_rates.append(lr)
            self.model.save(
                f"{self.path}/{self.model.name}/model_{self.model.name}_newest_epoch.h5",
                overwrite=True,
            )
            self.model.save_weights(
                f"{self.path}/{self.model.name}/weights_{self.model.name}_newest_epoch.h5",
                overwrite=True,
            )
            pd.DataFrame(self.model.history.history).to_pickle(
                f"{self.path}/{self.model.name}/history_newest_epoch_{self.model.name}.pkl"
            )

        #if epoch in [0, 1, 5, 10, 20, 30, 40, 50, 60, 70, 200, 500, 1000]:
        #    self.model.save_weights(
        #        f"{self.path}/{self.model.name}/weights_{self.model.name}_epoch_{epoch}.h5",
        #        overwrite=True,
        #    )


def get_callbacks(model,all_models_folder):
    import os

    os.makedirs(f"{all_models_folder}/{model.name}", exist_ok=True)
    best_model_file = f"{all_models_folder}/{model.name}/best_model_{model.name}.h5"
    callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=150)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="loss", factor=0.2, patience=10, min_lr=0.00001
    )

    best_model = tf.keras.callbacks.ModelCheckpoint(
        best_model_file, monitor="val_loss", mode="auto", verbose=0, save_best_only=True
    )
    cbk = CustomModelCheckpoint(path =all_models_folder )

    callbacks = [callback, reduce_lr, cbk, best_model]

    return callbacks



def model_builder(hp,which:str='1'):
    """
    Build model for hyperparameters tuning
    
    hp: HyperParameters class instance
    """
    
    tf.keras.backend.clear_session()
    # defining a set of hyperparametrs for tuning and a range of values for each
    #filters = hp.Int(name = 'filters', min_value = 60, max_value = 230, step = 20)
    #filterSize = hp.Int(name = 'filterSize', min_value = 3, max_value = 7,step=2)
    latent_space_dim = hp.Int(name = 'latentSpaceDim', min_value = 25, max_value = 500,step=25)
    #dropout = hp.Boolean(name = 'dropout', default = False)
    #bn_after_act = hp.Boolean(name = 'bn_after_act', default = False)
    #activation = hp.Choice(name = 'activation', values = ['mish', 'elu', 'lrelu'], ordered = False)
    #l1l2 = hp.Float("l1l2",min_value=1e-9, max_value=1e-3, sampling="log")
    #input_size = (544,544,3)
    #target_labels = [str(i) for i in range(21)]
    clip = hp.Float("clipping",min_value=0.4, max_value=1, step=0.2)
    learning_rate = hp.Float("lr", min_value=1e-6, max_value=1e-2, sampling="log")
    
    #patience = hp.Int("lrPatience", min_value=5, max_value=30, step=5)
    if which=='1':
        model,  __, __  = arc.modelPoolingDropoutHyperparms(img_size = (340,500,2), 
                        filters = 228,
                        filterSize=3,
                        latent_space_dim = latent_space_dim,
                        L1L2= 1e-9,
                        dropOut =True)
        model._name = name_append

    else:
        # building a model
        filters1 = hp.Int(name = 'filtersFirstTwoLayers', min_value = 60, max_value = 230, step = 20)
        filters2 = hp.Int(name = 'filtersThirdLayer', min_value = 30, max_value = 100, step = 20)
        filters3 = hp.Int(name = 'filtersFourthLayer', min_value = 20, max_value = 50, step = 20)
        model,  __, __  = arc.modelPoolingDropoutHyperparms(img_size = (340,500,2), 
                            filters1 = filters1,
                            filters2 = filters2,
                            filters3 = filters3, 
                            filterSize=filterSize,
                            latent_space_dim = latent_space_dim,
                            L1L2= l1l2,
                            dropOut =dropout)
    


    optimizer = tf.keras.optimizers.Adam(clipnorm=1, clipvalue=1, learning_rate=0.001
    )

    lr_metric = get_lr_metric(optimizer)
    model.compile(optimizer, loss=ssim_loss, metrics=["mse", lr_metric])

    
    return model







class WRWDScheduler(tf.keras.callbacks.Callback):
    

    """ """

    @tf.autograph.experimental.do_not_convert
    def __init__(
        self,
        steps_per_epoch,
        lr,
        wd_norm=0.004,
        eta_min=0.000002,
        eta_max=2,
        eta_decay=0.04,
        cycle_length=30,
        cycle_mult_factor=2,
    ):
        """Constructor for warmup learning rate scheduler"""

        super(WRWDScheduler, self).__init__()
        self.lr = lr
        self.wd_norm = wd_norm

        self.steps_per_epoch = steps_per_epoch

        self.eta_min = eta_min
        self.eta_max = eta_max
        self.eta_decay = eta_decay

        self.steps_since_restart = 0
        self.next_restart = cycle_length

        self.cycle_length = cycle_length
        self.cycle_mult_factor = cycle_mult_factor

        self.wd = wd_norm / (steps_per_epoch * cycle_length) ** 0.5

        self.history = defaultdict(list)
        self.batch_count = 0
        self.learning_rates = []

        self.batch_count = 0
        self.learning_rates = []

    @tf.autograph.experimental.do_not_convert
    def cal_eta(self):
        """Calculate eta"""
        fraction_to_restart = self.steps_since_restart / (
            self.steps_per_epoch * self.cycle_length
        )
        eta = self.eta_min + 0.5 * (self.eta_max - self.eta_min) * (
            1.0 + np.cos(fraction_to_restart * np.pi)
        )
        return eta

    @tf.autograph.experimental.do_not_convert
    def on_train_batch_begin(self, batch, logs={}):
        """update learning rate and weight decay"""
        eta = self.cal_eta()
        # self.model.optimizer._learning_rate = eta * self.lr
        lr = eta * self.lr

        K.set_value(self.model.optimizer.lr, lr)
        # self.model.optimizer._weight_decay = eta * self.wd

    @tf.autograph.experimental.do_not_convert
    def on_train_batch_end(self, batch, logs={}):
        """Record previous batch statistics"""
        logs = logs or {}

        # self.history['wd'].append(self.model.optimizer.optimizer._weight_decay)
        for k, v in logs.items():
            self.history[k].append(v)

        self.steps_since_restart += 1

    @tf.autograph.experimental.do_not_convert
    def on_epoch_end(self, epoch, logs={}):

        """Check for end of current cycle, apply restarts when necessary"""

        def on_epoch_end(self, epoch, logs=None):
            print(K.eval(self.model.optimizer.lr))

        self.batch_count = self.batch_count + 1
        lr = K.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)
        # self.model.history.history.append(lr)

        self.history["lr"].append(lr)
        self.learning_rates.append(lr)

        if epoch + 1 == self.next_restart:
            self.steps_since_restart = 0
            self.cycle_length = np.ceil(self.cycle_length * self.cycle_mult_factor)
            self.next_restart += self.cycle_length
            self.eta_min *= self.eta_decay
            self.eta_max *= self.eta_decay
            self.wd = self.wd_norm / (self.steps_per_epoch * self.cycle_length) ** 0.5
