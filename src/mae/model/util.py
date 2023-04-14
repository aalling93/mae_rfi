import os
import numpy as np
import tensorflow as tf

"""'
For Mac M1 there are many errors with TF.. One of them is the usage of these Learningrate schedulaers..

We are specifying a global step variable explicitly and using that variable instead of calling tf.compat.v1.train.get_global_step().


"""


class WarmUpCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self, learning_rate_base, total_steps, warmup_learning_rate, warmup_steps
    ):
        super().__init__()

        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.pi = tf.constant(np.pi)

    def __call__(self, step):
        if self.total_steps < self.warmup_steps:
            raise ValueError("Total_steps must be larger or equal to warmup_steps.")

        cos_annealed_lr = tf.cos(
            self.pi
            * (tf.cast(step, tf.float32) - self.warmup_steps)
            / float(self.total_steps - self.warmup_steps)
        )
        learning_rate = 0.5 * self.learning_rate_base * (1 + cos_annealed_lr)
        # learning_rate = tf.cast(learning_rate,dtype=tf.float64)
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
    Build model for hyperparameters tuning

    hp: HyperParameters class instance
    """
    tf.keras.backend.clear_session()
    # defining a set of hyperparametrs for tuning and a range of values for each
    # filters = hp.Int(name = 'filters', min_value = 60, max_value = 230, step = 20)
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
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(which)
    print("loading gpu")
    gpus = tf.config.experimental.list_physical_devices("GPU")
    print(gpus)
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory)],
        )
    except RuntimeError as e:
        print("\n error '\n")
        print(e)
    strategy = tf.device(f"/GPU:0")

    return strategy


def get_lr_metric(optimizer):
    @tf.autograph.experimental.do_not_convert
    def lr(y_true, y_pred):
        return optimizer.lr

    return lr
