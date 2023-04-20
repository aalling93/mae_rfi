import tensorflow as tf

from ..CONSTANTS import *
from .data_loader import _load_data
import numpy as np


class Data:
    """
    This class loads the data, preprocesses it, and converts it to TensorFlow datasets.

    Attributes:
        callbacks (list): A list of callbacks to be executed during training.

    Methods:
        load_data(train_data, test_data, imsize, only_VH, test_samples): 
            Loads the train and test data, preprocesses it, and converts it to TensorFlow datasets.
    """
    def __init__(self):
        """
        Initializes the Data class with an empty callbacks list.
        """
        super(Data, self).__init__()
        self.callbacks = []

    def __enter__(self):
        """
        Returns the current instance of the Data class.
        """
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        """
        Passes through any exceptions that occur during execution.
        """
        pass

    def load_data(
        self,
        train_data,
        test_data,
        imsize: tuple = (340, 500, 2),
        only_VH: bool = False,
        test_samples: int = 100,
    ):
        """
        Loads the train and test data, preprocesses it, and converts it to TensorFlow datasets.

        Args:
            train_data (str/list): Path to the train data. Either string (npy) or list [.npy,.npy,...]
            test_data (str/list): Path to the test data. Either string (npy) or list [.npy,.npy,...]
            imsize (tuple): Image size in pixels (height, width, channels). Defaults to (340, 500, 2).
            only_VH (bool): If True, only use the VH polarization. Defaults to False.
            test_samples (int): Number of test samples. Defaults to 100.

        Returns:
            None
        """

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

        self.train = self.train[0:70]
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
