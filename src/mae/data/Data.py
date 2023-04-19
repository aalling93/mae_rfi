import tensorflow as tf

from ..CONSTANTS import *
from .data_loader import _load_data
import numpy as np


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
        train_data,
        test_data,
        imsize: tuple = (340, 500, 2),
        only_VH: bool = False,
        test_samples: int = 100,
    ):

        self.train,self.val,self.test = _load_data(
            train=train_data,
            test=test_data,
            imsize=imsize,
            only_VH=only_VH,
            test_samples=test_samples,
        )

        if np.max(self.train[0][:,:,0])>1.1:
            self.train = self.train/255.0
            self.test = self.test/255.0
            self.val = self.val/255.0

        self.train = self.train[0:30]
        self.test = self.test[0:30]
        self.val = self.val[0:30]


        train_ds = tf.data.Dataset.from_tensor_slices(self.train)
        self.train_ds = train_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(AUTO)

        val_ds = tf.data.Dataset.from_tensor_slices(self.val)
        self.val_ds = val_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(AUTO)

        test_ds = tf.data.Dataset.from_tensor_slices(self.test)
        self.test_ds = test_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(AUTO)

        del test_ds, val_ds, train_ds

        return None
