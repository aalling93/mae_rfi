import numpy as np
import tensorflow as tf

from ..CONSTANTS import *
from .data_util import _load_data


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
        train_data: str = "../data/processed/train_zm_jsd.npy",
        test_data: str = "",
        imsize: tuple = (340, 500, 2),
    ):
        
        self.train = _load_data(train_data,crop=True,imsize=imsize)
        if len(test_data)>0:
            test = _load_data(test_data,crop=True,imsize=imsize)
            self.test  = test[:20] #just picking 20 images.. Just to use less memory.. Could do whatever.


        train_ds = tf.data.Dataset.from_tensor_slices(self.train)
        test_ds = tf.data.Dataset.from_tensor_slices(self.test)
        
        self.train_ds = train_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(AUTO)
        self.test_ds = test_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(AUTO)

        return None
    

    


