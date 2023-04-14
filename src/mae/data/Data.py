import numpy as np
from .data_util import *
import tensorflow as tf
from ..CONSTANTS import *


class Data:
    def __init__(self):
        super(Data, self).__init__()
        self.callbacks = []

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        pass

    def load_data(
        self,
        data: str = "../data/processed/train_zm_jsd.npy",
        imsize: tuple = (340, 500, 2),
    ):
        train = np.load(data, allow_pickle=True)
        self.train = np.array(
            [
                center_crop(im, [imsize[1], imsize[0]])
                for im in train
                if (im.shape[0] >= imsize[0] and im.shape[1] >= imsize[1])
            ]
        )

        train_ds = tf.data.Dataset.from_tensor_slices(self.train[0:100])
        self.train_ds = train_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(AUTO)

        return None
