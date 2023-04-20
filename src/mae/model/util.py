"""
Author: Kristian Sørensen
        kaaso@space.dtu.dk
        Marts 2023

This script contains functions and a class for machine learning using TensorFlow.

Overview:
    - `WarmUpCosine` is a custom learning rate scheduler that anneals the learning rate using a cosine function, 
    and also applies warm-up during the initial steps.
    - `model_builder` defines a function to build a TensorFlow model with hyperparameters tuning using HyperOpt.
    - `load_gpu` sets up the GPU for TensorFlow to use.
    - `get_lr_metric` returns a function that retrieves the current learning rate from the optimizer.
    - `get_model_summary` returns a summary of the model's layers and parameters.

More:
    - `WarmUpCosine` can be used as a learning rate schedule in a TensorFlow optimizer. See TensorFlow documentation for details.
    - `model_builder` can be used to build and compile a TensorFlow model for hyperparameters tuning with HyperOpt. 
    Inputs: `hp` (HyperParameters class instance) - a set of hyperparameters for tuning.
    Outputs: a TensorFlow model.
    - `load_gpu` is used to set up the GPU. Inputs: `which` (int) - the ID of the GPU to use, `memory` (int) - the memory limit in MB.
    Outputs: a TensorFlow strategy object.
    - `get_lr_metric` is used to retrieve the current learning rate from the optimizer. Inputs: `optimizer` - a TensorFlow optimizer.
    Outputs: a function that retrieves the current learning rate.
    - `get_model_summary` is used to retrieve a summary of the model's layers and parameters. 
    Inputs: `model` - a TensorFlow model.
    Outputs: a summary string.

"""

import os
import numpy as np
import tensorflow as tf
import io

class WarmUpCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self, learning_rate_base, total_steps, warmup_learning_rate, warmup_steps
    ):
        """
        A custom learning rate scheduler that gradually increases the learning rate from a warmup value to a base
        value using a cosine annealing schedule. It takes the total number of training steps, the base learning rate,
        the warmup learning rate, and the number of warmup steps as input parameters.

        Args:
            learning_rate_base (float): Base learning rate.
            total_steps (int): Total number of training steps.
            warmup_learning_rate (float): Warmup learning rate.
            warmup_steps (int): Number of warmup steps

        Outputs:
            Learning rate (float): The learning rate for a given step, according to the cosine annealing schedule with warm-up.
        """

        super().__init__()

        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.pi = tf.constant(np.pi)

    def __call__(self, step):
        """
        Computes the learning rate for the given step using the WarmUpCosine schedule.

        Args:
            step (int): The current training step.

        Returns:
            learning_rate (float): The learning rate for the current step.
        """
        if self.total_steps < self.warmup_steps:
            raise ValueError("Total_steps must be larger or equal to warmup_steps.")
        # Compute cosine annealed learning rate
        cos_annealed_lr = tf.cos(
            self.pi
            * (tf.cast(step, tf.float32) - self.warmup_steps)
            / float(self.total_steps - self.warmup_steps)
        )
        learning_rate = 0.5 * self.learning_rate_base * (1 + cos_annealed_lr)
        # learning_rate = tf.Variable(learning_rate, trainable=False)
        ## learning_rate = tf.cast(learning_rate,dtype=tf.float64)

        # Apply warmup learning rate
        if self.warmup_steps > 0:
            if self.learning_rate_base < self.warmup_learning_rate:
                raise ValueError(
                    "Learning_rate_base must be larger or equal to "
                    "warmup_learning_rate."
                )
            slope = (
                self.learning_rate_base - self.warmup_learning_rate
            ) / self.warmup_steps
            warmup_rate = slope * tf.cast(step, tf.float32) + self.warmup_learning_rate
            learning_rate = tf.where(
                step < self.warmup_steps, warmup_rate, learning_rate
            )
        return tf.where(
            step > self.total_steps, 0.0, learning_rate, name="learning_rate"
        )


def model_builder(hp):
    """
    Builds a machine learning model for hyperparameter tuning.

    Args:
        hp (HyperParameters): A HyperParameters class instance containing the hyperparameters to tune.

    Returns:
        model (tf.keras.Model): The model built with the given hyperparameters.


    hp: HyperParameters class instance
    """
    tf.keras.backend.clear_session()
    # defining a set of hyperparametrs for tuning and a range of values for each
    # filters = hp.Int(name = 'filters', min_value = 60, max_value = 230, step = 20)

    # Define hyperparameters for tuning
    filterSize = hp.Int(name="filterSize", min_value=2, max_value=7, step=1)
    latent_space_dim = hp.Int(name="latentSpaceDim", min_value=1, max_value=20, step=1)

    clip = hp.Float("clipping", min_value=0.4, max_value=1, step=0.2)
    learning_rate = hp.Float("lr", min_value=1e-6, max_value=1e-2, sampling="log")

    filters = []
    for i in range(filterSize):
        filters.append(
            hp.Int(name=f"filters{i+1}", min_value=60, max_value=500, step=50)
        )

    drop_rate = hp.Float(name="dropout_prob", min_value=0, max_value=0.4, step=0.05)
    laten_space_regularisation_L1 = hp.Float(
        name="laten_space_regularisation_L1",
        min_value=0.00001,
        max_value=0.001,
        step=0.05,
    )

    return None


def load_gpu(which: int = 0, memory: int = 60000):
    """
    Sets up and returns a distributed strategy to train models on GPUs.

    Args:
        which (int): Index of the GPU to use (default is 0).
        memory (int): Amount of GPU memory to allocate in MB (default is 60000).

    Returns:
        A distributed strategy object that can be used to distribute training on multiple GPUs.

    Example:
        strategy = load_gpu(which=0, memory=10000)
    """
    # Set environment variables to select the desired GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(which)

    # Check available physical GPUs
    gpus = tf.config.experimental.list_physical_devices("GPU")
    # Restrict memory usage of the selected GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory)],
        )
    except RuntimeError as e:
        print("\n error '\n")
        print(e)

    # Create a distributed strategy object to use the selected GPU(s)
    strategy = tf.device(f"/GPU:0")

    return strategy


def get_lr_metric(optimizer):
    """
    Returns a function that can be used to get the current learning rate of an optimizer.

    Args:
        optimizer (tf.keras.optimizers.Optimizer): The optimizer object.

    Returns:
        A function that takes two arguments (y_true, y_pred) and returns the current learning rate of the optimizer.

    Example:
        # Create an optimizer with a learning rate of 0.001
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        # Get a function that returns the current learning rate of the optimizer
        lr_function = get_lr_metric(optimizer)
    """

    @tf.autograph.experimental.do_not_convert
    def lr(y_true, y_pred):
        return optimizer.lr

    return lr


def get_model_summary(model):
    """
    Returns a string containing the summary of a Keras model.

    Args:
        model (tf.keras.models.Model): The Keras model to summarize.

    Returns:
        A string containing the model summary.

    Example:
        summary = get_model_summary(model)
    """
    # Redirect the summary output to a StringIO object
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + "\n"))
    summary_string = stream.getvalue()
    stream.close()
    return summary_string
